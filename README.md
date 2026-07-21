# macsima-pipeline

Config-driven pipeline for MACSIMA multiplexed imaging data on a SLURM HPC.

You write one YAML config per experiment. A single CLI, `macsima-pipeline`, drives five stages: **stage → mcmicro → preprocess → phenotype → viz**. Each stage submits a SLURM job, using an array where that makes sense. The `all` command chains the stages together with `afterok` dependencies.

---

## Table of contents

1. [Concepts](#1-concepts)
2. [Layout](#2-repository-layout)
3. [Install](#3-install)
4. [First run: step by step](#4-first-run-step-by-step)
5. [Pipeline stages in detail](#5-pipeline-stages-in-detail)
6. [Config reference](#6-config-reference)
7. [SLURM behaviour](#7-slurm-behaviour)
8. [Outputs](#8-outputs)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Concepts

**Goal.** Turn raw MACSima cycle output into one per-experiment `results/<exp>/` tree. That tree holds four kinds of output: processed OME-TIFFs (one per ROI, with or without background subtraction); cell segmentations as GeoParquet; a per-cell `AnnData` (`.h5ad`) carrying expression values and ROI metadata; and QC/visualisation PDFs. No pixel data is duplicated on disk, and no SpatialData zarr store is written.

**Why this repo exists.** Replaces per-experiment copy-paste scripts in `metpredict-macsima` (`expr10_*.py`, `expr10_*.sh`). One config, one CLI, same code for every experiment.

**Data flow.**

```
raw MACSima cycles (per ROI)
        │  panel   (sanity-check marker panel, at plan time)
        │  stage   (native: tifffile + ome-types + BaSiCPy; SLURM array, 1 task per ROI)
        ▼
mcmicro_output/{exp}/rack-…-roi-…-exp-2/   ← mcmicro-ingestable per-sample dirs (scratch)
        │  mcmicro (Nextflow + Singularity, SLURM array, 1 task per sample)
        │          publish_dir_mode=link → OME-TIFFs are hardlinks (no copy)
        │  finalize (hardlink OME-TIFFs + markers into the results tree; auto, idempotent)
        ▼
results/{exp}/images/registration/{roi}.ome.tif   ← no-bg variant  (0 extra bytes)
results/{exp}/images/backsub/{roi}.ome.tif         ← bg-sub variant
        │  preprocess (SLURM GPU array, 1 task per ROI, then merge)
        ▼
results/{exp}/segmentation/{exp}_ROI{n}_segmentation{suffix}.parquet  ← GeoParquet polygons
results/{exp}/cells/{exp}_cells{suffix}.h5ad        ← AnnData (cells × markers, obs merged with roi_metadata.csv)
        │  phenotype (1 GPU job; normalize + scyan + Leiden + spatial QC)
        ▼
results/{exp}/cells/{exp}_cells_phenotyped{suffix}.h5ad  ← + obs['cell_type'], layers, obsm['spatial'], uns['phenotype']
        │  viz (1 CPU job; joblib-parallel)
        ▼
results/{exp}/qc/rois/*.pdf                  ← per-marker grids, per-ROI grids, RGB combinations
results/{exp}/qc/*cell_maps_summary*.pdf     ← cell XY maps (colored by cell type) + QC summary
results/{exp}/qc/phenotype/*phenotype_summary*.pdf ← composition, confidence, spatial coherence
```

**Config inheritance.** `configs/default.yaml` holds every key. An experiment config (`experiments/<exp>/config.yaml`) needs only `extends:` plus the few keys that differ, usually `experiment.name`, `experiment.raw_root`, and sometimes `viz.combinations`. The merge is a right-biased deep merge, and a child `None` never clobbers a non-null base value.

**Suffix.** With `mcmicro.background_subtraction: false`, the per-variant filenames (segmentation parquet, cells h5ad, QC) get a `_no_bs` suffix. With `true`, the pipeline uses background-subtracted images and adds no suffix. The shipped default is `"auto"`: it runs whichever variants mcmicro produced, i.e. both when both exist.

**No duplication.** mcmicro publishes with `publish_dir_mode=link`, and the `finalize` step hardlinks the per-ROI OME-TIFFs into `results/<exp>/images/`. Each image is therefore one inode with several names, never a second copy on disk. Nothing is deleted automatically. Reclaim the Nextflow `work/` directory and the staged `raw/` tiles whenever you want with `macsima-pipeline clean`.

---

## 2. Repository layout

```
configs/
  default.yaml            full schema + defaults; every key documented
  mcmicro_params.yaml     mcmicro params (passed via --params)
  cemm.nextflow.config    cluster-specific nextflow config
  *.sbatch.j2             Jinja2 sbatch templates (one per stage)
experiments/
  expr10/config.yaml      worked example (LN + ovarian + breast cancer)
  expr34/config.yaml      worked example (mouse models, 16 ROIs)
  <exp>/roi_metadata.csv  optional: columns ROI,<anything> joined into adata.obs
src/macsima_pipeline/
  cli.py                  cyclopts CLI entrypoints
  config.py               pydantic schema + extends/merge loader
  staging.py              stage 1 orchestration (SLURM array; runs panel at plan time)
  staging_core.py         native macsima2mc port: parse → stack → OME-TIFF + markers.csv (+BaSiCPy)
  panel.py                pre-staging marker panel sanity check
  mcmicro.py              stage 2
  finalize.py             consolidate mcmicro OME-TIFFs into results/<exp>/images (hardlinks)
  preprocess.py           stage 3 (run_inproc + SLURM worker array + merge)
  phenotype/              stage 4 (normalize + scyan + Leiden + spatial QC + report)
  viz/                    stage 5 (workers + plotting)
  clean.py                opt-in scratch reclamation (work/, raw/, orphaned zarr)
  slurm.py                sbatch render + sbatch submit
jobs/                     generated per-stage csv + sbatch (gitignored)
logs/                     SLURM stdout/stderr (gitignored)
results/{exp}/            all deliverables: images/ segmentation/ cells/ qc/ panel/ (gitignored)
```

---

## 3. Install

You need `uv` on the cluster. The mcmicro stage also needs Nextflow and `apptainer`/singularity, but nothing else does: staging is now native Python and needs no container.

```bash
# 1. Clone
git clone <repo> && cd macsima-pipeline

# 2. Python env (uv reads pyproject.toml + uv.lock)
uv sync

# 3. Verify (staging is native; no macsima2mc container to pull)
uv run macsima-pipeline --help
```

mcmicro pulls its own images via Nextflow on first run; nothing to do here besides having Nextflow on `PATH`.

---

## 4. First run: step by step

Walk through with the bundled `expr34` example. Substitute your own paths for a new experiment.

### 4.1 Create an experiment directory

```bash
mkdir -p experiments/myexp
cp experiments/expr34/config.yaml experiments/myexp/config.yaml
```

### 4.2 Edit the config

Open `experiments/myexp/config.yaml`. Only these keys typically change:

```yaml
extends: "../../configs/default.yaml"

experiment:
  name: "myexp"                                  # short slug used in filenames
  raw_root: "/path/to/RawData/R1/B1"             # contains ROI0/, ROI1/, ...
  roi_exclude: ["ROI0"]                          # ROI0 is usually a calibration scan
  roi_metadata_csv: "experiments/myexp/roi_metadata.csv"   # optional

mcmicro:
  background_subtraction: false

viz:
  combinations: []                               # or list of {name, markers: [..]}
```

> **Gotcha.** `combinations:` followed by ONLY commented-out items parses as `None` and fails pydantic validation. Always keep an explicit `[]` if the list is empty. (The loader was hardened to ignore `None` overrides, but writing `[]` is clearer.)

### 4.3 (Optional) ROI metadata

CSV with a `ROI` column plus any columns you want joined into `adata.obs`:

```csv
ROI,Sample,Patient_ID
ROI1,LN,P1
ROI2,LN,P1
ROI3,Ovarian Cancer,30108919
```

Path is resolved relative to `paths.work_dir` (default `.`).

Generate a template for the exact ROIs the pipeline will process. It reuses the
staging ROI discovery, so `roi_exclude` and friends are applied. Then fill in the
columns:

```bash
uv run macsima-pipeline gen-roi-metadata --config experiments/myexp/config.yaml \
    --columns Sample --columns Patient_ID
# → wrote roi_metadata_myexp.csv  (or to experiment.roi_metadata_csv if set)
```

It needs `raw_root` mounted (run on a node where RawData is reachable) and refuses
to overwrite an existing file unless you pass `--force`.

### 4.3b (Optional) Phenotype signature

The phenotype stage (scyan + Leiden) needs a **signature matrix**, a marker-to-cell-type
table (`phenotype.signature_matrix`). If it is unset, the phenotype stage skips (the chain
still runs). The pre-staging `panel` command scaffolds one shared `signature.yaml` next to
your config (union of markers across the config's experiments):

```bash
uv run macsima-pipeline panel --config experiments/myexp/config.yaml
# → wrote signature.yaml (panel markers listed + example cell types to edit)
```

Fill in `positive`/`negative` markers per cell type, then set
`phenotype.signature_matrix` to that path. `panel` won't overwrite your edits (pass
`--force` to regenerate). See `configs/signature_example.yaml`.

### 4.4 Dry-run each stage

Dry-run writes the jobs CSV + sbatch and prints the `sbatch` command it WOULD run. Nothing submitted.

```bash
uv run macsima-pipeline stage --config experiments/myexp/config.yaml
# → wrote jobs/staging_myexp.csv (N rows)
# → (dry-run) sbatch --array=1-N jobs/staging_myexp.sbatch
```

Inspect `jobs/staging_myexp.csv` and `jobs/staging_myexp.sbatch` before submitting.

### 4.5 Submit one stage at a time (recommended for first run)

```bash
uv run macsima-pipeline stage --config experiments/myexp/config.yaml --submit
# returns <job-id-1>

# Wait for stage 1 to finish (or chain with --dependency)
uv run macsima-pipeline mcmicro --config experiments/myexp/config.yaml --submit \
    --dependency <job-id-1>

uv run macsima-pipeline preprocess --config experiments/myexp/config.yaml --submit \
    --dependency <job-id-2>

uv run macsima-pipeline viz --config experiments/myexp/config.yaml --submit \
    --dependency <job-id-3>
```

### 4.6 Or submit the whole chain at once

```bash
uv run macsima-pipeline all --config experiments/myexp/config.yaml --submit
# stage=12345 planner=12346
```

Each downstream job uses `--dependency=afterok:<prev>` so it only runs on success.
For `all --submit`, the reported `planner` id is a short continuation job:
after staging has produced sample folders, it submits the real MCMICRO array,
then submits a short preprocess planner with a dependency on the real MCMICRO
job id. That planner runs after MCMICRO has produced concrete OME-TIFF paths,
submits the preprocess worker array, submits the preprocess merge job, and
finally submits viz with a dependency on the merge job. No node allocation has
to sit around waiting for the full MCMICRO array.

### 4.7 Monitor

```bash
squeue -u $USER
tail -F logs/staging_myexp_<jobid>_*.out
```

### 4.8 Re-run a single stage

Stage outputs are deterministic per (config, raw data). To redo just one stage, delete its outputs (e.g. `mcmicro_output/myexp/`) and submit it again. The `--dependency` flag lets you re-attach downstream stages without rerunning everything.

### 4.9 Batch: multiple experiments in one config

MACSima data usually arrives in batches over weeks. Instead of one config file per
experiment, a **batch config** lists several under a top-level `experiments:` key.
Every other section is shared (via `extends` + shared overrides); each entry
overrides only its per-experiment fields. See `configs/batch_example.yaml`.

```yaml
extends: "configs/default.yaml"
mcmicro:
  background_subtraction: "auto"        # shared by every experiment
experiments:
  - name: expr37
    raw_root: "/.../Expr37/.../RawData/R1/B1"
    roi_metadata_csv: "configs/batch_example/expr37_roi_metadata.csv"
  - name: expr35
    raw_root: "/.../Expr35/.../RawData/R1/A1"
    roi_exclude: ["ROI0", "ROI7"]
    roi_metadata_csv: "configs/batch_example/expr35_roi_metadata.csv"
```

Every command works on a batch config unchanged. It runs **one independent chain
per experiment**, with outputs kept separate by `experiment.name`:

```bash
uv run macsima-pipeline all --config configs/batch_example.yaml --submit
# batch: 2 experiments ['expr37', 'expr35']  → two independent SLURM chains
```

As new data arrives, append an entry and submit just the new one with `--only`:

```bash
uv run macsima-pipeline all --config configs/batch_example.yaml --only expr40 --submit
```

The scaffold utilities loop over the batch too, writing one file per experiment
(`--experiment NAME` targets one; ROI names collide across experiments, so metadata
stays per-experiment):

```bash
uv run macsima-pipeline gen-roi-metadata --config configs/batch_example.yaml --columns Sample
```

Notes: each entry inherits the shared `experiment:` defaults (e.g. `roi_exclude: ["ROI0"]`)
and overrides only what differs. Experiment names must be unique. Batch submission
materializes a flattened per-experiment config under `jobs/batch/<name>.yaml` (this is
what the SLURM continuation jobs re-load); single-experiment configs are used as-is and
write nothing extra. Marker panels are **not** assumed shared across experiments; each
reads its own `markers.csv` exactly as before.

**Phenotyping in a batch.** `all` phenotypes each experiment **independently**: one model per
experiment. The cell-type *names* still come from the shared signature, but scyan calibration and
Leiden cluster ids are not comparable across experiments, and there is no cross-experiment batch
correction. For **joint** phenotyping (one scyan/Leiden model over all experiments, giving comparable
labels and cross-experiment batch correction), run the dedicated command once the per-experiment
preprocess outputs exist:

```bash
uv run macsima-pipeline phenotype-joint --config experiments/<batch>/config.yaml --submit
```

It concatenates every experiment's cell-expression h5ad (inner-join on shared markers), tags
each cell with `experiment` + a unique `sample` (experiment|ROI), fits the engines once,
writes a combined phenotyped h5ad under `results/<batch-folder>/cells/`, and splits the joint
labels back into each experiment's phenotyped h5ad (so per-experiment viz picks them up).
`--batch-key` (default `sample`) sets the batch-correction unit; `--inproc` runs locally;
`--only NAME` restricts the set.

---

## 5. Pipeline stages in detail

### Stage 0: `panel` (runs automatically before `stage`)

From the raw filenames alone (fast, no pixel reads) writes `results/<exp>/panel/marker_panel.csv`
(a per-cycle panel summary, so you can sanity-check that the run acquired what you expect) and validates the
panel: reference marker present in every cycle, consistent markers across ROIs, background
acquisitions present. `stage` runs the marker-panel check at plan time; the standalone
`macsima-pipeline panel --config …` command also scaffolds a shared `signature.yaml`
(phenotyping cell-type template) next to the config for you to curate before submitting.

### Stage 1: `stage`

Native Python, no container. `staging_core.py` reimplements macsima2mc v1.3.1 with `tifffile` and
`ome-types`. It discovers ROIs from `experiment.raw_root` and `experiment.roi_glob`, minus
`roi_exclude`, and writes `jobs/staging_<exp>.csv` with one row per ROI. One row is one SLURM array
task, and each task stages every `*Cycle*` folder of its ROI.

For each cycle it parses the MACSima filenames and groups tiles by `(source, exposure_level)`. It
orders channels reference-marker-first, backfilling DAPI into exposure levels that did not reacquire
it. It optionally applies BaSiCPy flatfield correction (`staging.illumination_correction`, on by
default, which adds a `corr_` prefix). Finally it writes multi-series OME-TIFFs plus `markers.csv` to
`mcmicro_output/<exp>/rack-X-well-Y-roi-Z-exp-N/`. Everything here is tunable under `staging:` in the
config, and the output is drop-in compatible with the previous container.

### Stage 2: `mcmicro`

For each staged sample dir matching `mcmicro.sample_pattern` (default `rack-*-well-*-roi-*-exp-2`), this stage runs `nextflow run labsyspharm/mcmicro -profile singularity` with `mcmicro.params_yaml`. It produces Ashlar-registered pyramidal OME-TIFFs at `<sample>/registration/<…>exp-2.ome.tif`, or at `<sample>/background/<…>_backsub.ome.tif` when `background_subtraction: true`.

The run passes `--publish_dir_mode link` and a pinned `-work-dir`, so the published OME-TIFFs are **hardlinks** into the Nextflow `work/` scratch: one inode, no copy. This works only when `work/` and the output tree share a filesystem (both live under `/nobackup` here).

### Finalize (image consolidation)

Between mcmicro and preprocess, `finalize.consolidate_images` hardlinks each sample's `registration`/`background` OME-TIFF into a flat, readable layout: `results/<exp>/images/{registration,backsub}/{roi}.ome.tif`. It also copies the markers CSVs once per experiment. All of this still adds zero bytes. The step is idempotent and runs automatically at the start of `preprocess` and `viz`. To run it on its own, use `macsima-pipeline finalize --config <cfg>`.

### Stage 3: `preprocess`

On SLURM, this is a two-phase stage: a GPU worker array runs one task per concrete `(background variant, ROI image)`, then a CPU merge job assembles the final per-variant cell table. `--inproc` is still available for local/debug runs and iterates all ROIs in one process. Steps per ROI:

1. Load the consolidated OME-TIFF (`results/<exp>/images/<variant>/<roi>.ome.tif`) as a dask array, keeping the channels in `markers.csv` with `remove != True`.
2. Wrap it as an in-memory `Image2DModel` with the `scale_factors` pyramid (default `[2, 4]`). This is never persisted to disk.
3. `sopa.make_image_patches` → `sopa.segmentation.custom_staining_based` with Cellpose4 (`cpsam` model) on DAPI.
4. `sopa.aggregate` → per-cell expression table.

Each worker writes its ROI's segmentation polygons straight to the final file, `results/<exp>/segmentation/<exp>_ROI<n>_segmentation<suffix>.parquet` (GeoParquet, with columns `cell_id`, `ROI`, `centroid_x`, `centroid_y`, and `area`). It also writes a transient per-ROI cell-table part. The merge job then validates the parts, concatenates the per-ROI cell tables into one `AnnData`, records an explicit `obs["ROI"]`, left-joins `roi_metadata.csv` on `ROI`, and writes `results/<exp>/cells/<exp>_cells<suffix>.h5ad`. It deletes the transient parts afterward. No SpatialData zarr is written.

The public `macsima-pipeline preprocess --submit` command submits the worker array first and then a merge job dependent on that array. Before mcmicro outputs exist, dry-run reports that exact array planning is deferred until the OME-TIFFs are available.

### Stage 4: `phenotype`

Reads the per-variant cells `.h5ad` (all ROIs jointly) and assigns cell types, writing a
separate `{exp}_cells_phenotyped{suffix}.h5ad`. One GPU job per variant; `--inproc`
runs locally. Skips gracefully (exit 0, chain still proceeds) when disabled or when no
`phenotype.signature_matrix` is set. Steps:

1. **Normalize** (`phenotype.normalize`): stash raw → per-marker winsorize → arcsinh →
   z-score. Keeps `layers['counts']` (raw) and `layers['zscore']` (normalized).
2. **Batch** (`phenotype.batch`): per-ROI z-score by default (ComBat / quantile options),
   at the intensity stage so markers stay interpretable.
3. **Engines** (`phenotype.engines`), both driven by the same signature matrix:
   - **scyan**: a Bayesian normalizing-flow model (torch + lightning) that gives probabilistic
     per-cell assignments. It reads the arcsinh + z-scored layer, and a GPU speeds up training.
   - **Leiden**: a scanpy kNN graph plus Leiden clustering (`flavor="igraph"`). Clusters are
     auto-labeled against the signature, so labels are comparable to scyan's. Reads the z-scored layer.
4. **Cross-engine agreement** (Cohen's κ, ARI): a confidence signal. Disagreement flags
   batch effects or ambiguous markers.
5. **Spatial QC** (`phenotype.spatial_qc`): neighborhood enrichment + same-type homophily
   per ROI. This is the automatic version of "do the labels make sense on the map?".

Writes `obs['cell_type', 'cell_type_coarse', 'cell_type_confidence', 'scyan_celltype',
'leiden', 'leiden_celltype', 'pheno_agree']`, `obsm['spatial']`, `uns['phenotype']`, and a
QC PDF under `results/{exp}/qc/phenotype/`. The **signature matrix** is a small YAML that names the
expected positive and negative markers for each cell type, with an optional lineage `parent`. Point
`phenotype.signature_matrix` at it.

### Stage 5: `viz`

Loads the consolidated OME-TIFFs from `results/<exp>/images/<variant>/`. For cell-map QC it also loads
the variant cells `.h5ad` when present, preferring the phenotyped one; that colors cell maps by
`cell_type` and adds a spatial-coherence page. For each marker it picks the pyramid level whose largest
XY dim is ≤ `viz.target_max_dim` (default 2048), computes 1–99 percentile clips (cached to parquet for
resume), and renders:

- one PDF per marker showing that marker across all ROIs (grid),
- one PDF per ROI showing all markers (grid),
- one PDF per entry in `viz.combinations` (RGB composite of 3 markers).
- one multi-page cell-map QC summary PDF with an experiment overview page plus one XY cell-location page per ROI.

Parallelised via joblib (`viz.parallel.workers`, `backend`). Rasterised imshow at `viz.dpi` (300), PDF compression 9.

---

## 6. Config reference

See `configs/default.yaml` for every key with inline comments. Selected keys:

| Section | Key | Meaning |
|---|---|---|
| `experiment` | `name` | slug used in filenames (`{experiment_name}` placeholder) |
| `experiment` | `raw_root` | absolute path to `RawData/R*/[B\|C]*` containing `ROI*` subdirs |
| `experiment` | `roi_glob` | glob applied under `raw_root` (default `ROI*`) |
| `experiment` | `roi_exclude` | ROI names to drop (default `["ROI0"]`) |
| `experiment` | `roi_include` | if set, ONLY these ROIs (post-exclude) |
| `experiment` | `roi_metadata_csv` | optional CSV joined into `adata.obs` on `ROI` |
| `paths` | `*` | output paths; support `{experiment_name}` + `{suffix}` placeholders |
| `mcmicro` | `background_subtraction` | false → uses `registration_pattern`, suffix `_no_bs`; true → `background_pattern`, no suffix |
| `mcmicro` | `sample_pattern` | glob for stage 2 to find staged samples |
| `preprocess.segmentation` | `model`, `channels`, `min_area`, `gpu` | Cellpose4 params |
| `preprocess.patches` | `patch_width` | Sopa patch size for segmentation; lower this if CUDA memory is tight |
| `preprocess.parallel.max_workers` | worker array throttle | maximum concurrent preprocessing workers |
| `phenotype` | `signature_matrix` | path to the signature YAML; `null` → stage skips (chain still runs) |
| `phenotype` | `engines`, `primary_engine` | which engines to run (`scyan`, `leiden`); which populates `cell_type` |
| `phenotype.normalize` | `transform`, `cofactor`, `clip_percentile`, `zscore` | per-marker normalization (arcsinh; tune `cofactor` to your intensity scale) |
| `phenotype.batch` | `method`, `batch_key` | intensity-stage batch handling (`zscore_per_roi` default) |
| `phenotype.scyan` | `max_epochs`, `lr`, `prior_std`, `temperature`, `log_prob_th`, `min_confidence` | scyan engine (normalizing-flow; reads z-scored layer) |
| `phenotype.leiden` | `n_neighbors`, `resolution`, `n_iterations`, `tau` | Leiden engine (kNN graph + Leiden; reads z-scored layer) |
| `phenotype.spatial_qc` | `n_neighs`, `nhood_enrichment`, `homophily` | spatial-coherence QC |
| `viz` | `combinations` | list of `{name, markers: [m1, m2, m3]}` → RGB plots |
| `viz` | `cell_maps`, `cell_map_marker_top_n`, `cell_map_point_size` | default-on cell XY + expression QC summary PDF controls |
| `slurm.<stage>` | `partition`, `qos`, `cpus`, `mem`, `time`, `gres`, `comment` | sbatch header values |

**Placeholder expansion.** `{experiment_name}` and `{suffix}` are expanded in `paths.results_dir`, `paths.cells_out`, `paths.phenotype_cells_out`.

---

## 7. SLURM behaviour

- Templates live in `templates/*.sbatch.j2`. `slurm.py` renders the `#SBATCH` headers from `slurm.<stage>` and embeds the stage-specific body.
- Rendered sbatch + jobs CSV go to `jobs/<stage>_<exp>.{sbatch,csv}`.
- Logs go to `logs/<stage>_<exp>_%A_%a.out`.
- Array stages (`stage`, `mcmicro`, preprocess workers) read their work item from a jobs CSV by `$SLURM_ARRAY_TASK_ID`.
- Non-array stages (`preprocess_merge`, `viz`) submit a single job.
- `all --submit` submits staging first, then a short MCMICRO planner job with an `afterok` dependency on staging. The planner submits the real MCMICRO array, then a preprocess/viz planner with an `afterok` dependency on MCMICRO. That second planner submits the worker array, merge job, and viz with real job IDs.

Defaults (override under `slurm.<stage>` in your config):

| Stage | Partition | CPU / Mem / Time | GPU |
|---|---|---|---|
| stage | tinyq | 8 / 32G / 2h | — |
| mcmicro | shortq | 16 / 64G / 8h | — |
| preprocess | gpu | 16 / 100G / 6h | `gpu:h100pcie:1` |
| preprocess_merge | shortq | 8 / 100G / 4h | — |
| viz | shortq | 8 / 100G / 4h | — |

---

## 8. Outputs

Relative to `paths.work_dir` (default `.`):

Everything a user consumes lives under `results/<exp>/`; scratch/state stays outside it.

| Path | Stage | Contents |
|---|---|---|
| `results/<exp>/images/{registration,backsub}/<roi>.ome.tif` | finalize | processed OME-TIFF per ROI (hardlink; no-bg / bg-sub) |
| `results/<exp>/images/markers.csv`, `background/markers_bs.csv` | finalize | channel panels for each variant |
| `results/<exp>/segmentation/<exp>_ROI<n>_segmentation<suffix>.parquet` | stage 3 | GeoParquet cell polygons (+ centroid/area/cell_id/ROI) |
| `results/<exp>/cells/<exp>_cells<suffix>.h5ad` | stage 3 | AnnData (cells × markers, obs with ROI + metadata) |
| `results/<exp>/cells/<exp>_cells_phenotyped<suffix>.h5ad` | stage 4 | + `cell_type`, layers, `obsm['spatial']`, `uns['phenotype']` |
| `results/<exp>/qc/rois/*.pdf` | stage 5 | marker grids, ROI grids, RGB combinations |
| `results/<exp>/qc/*cell_maps_summary*.pdf`, `*channel_qc*` | stage 5 | cell-map + channel QC summaries |
| `results/<exp>/qc/phenotype/*phenotype_summary*.pdf` | stage 4 | composition, confidence, spatial coherence |
| `results/<exp>/panel/marker_panel.csv` | panel | pre-staging marker-panel summary |
| `mcmicro_output/`, `work/`, `jobs/`, `logs/` | all | scratch/state (reclaim `work/` + `raw/` with `macsima-pipeline clean`) |

---

## 9. Troubleshooting

**`ValidationError: viz.combinations Input should be a valid list`**
Your child config has `viz: combinations:` with only commented-out items, which parses as `None`. Use `combinations: []` instead. (The loader now tolerates `None`, but being explicit is clearer.)

**`raw_root not found`**
`experiment.raw_root` must be the directory that *directly contains* `ROI*` subdirs (typically `…/RawData/R1/B1` or `…/RawData/R1/C1`). Check with `ls "$raw_root"/ROI* | head`.

**`No ROIs found … matching ROI*`**
Either `raw_root` is wrong or every ROI is excluded. Check `roi_include` / `roi_exclude`.

**`Staged output dir not found: mcmicro_output/<exp>`**
Stage 2 ran before stage 1 produced output. Either run them sequentially with `--dependency`, or use `all --submit`.

**`No images found under … matching registration/*exp-2.ome.tif`**
Stage 2 failed or you set `background_subtraction: true` without producing `*_backsub.ome.tif`. Check `mcmicro_output/<exp>/<sample>/registration/`.

**`ROI metadata csv missing`**
Path is resolved relative to `paths.work_dir`. From the repo root, `experiments/<exp>/roi_metadata.csv` is correct; `examples/<exp>/…` is wrong (legacy path).

**Preprocess OOM / OOT.**
Bump `slurm.preprocess.mem` / `time` for CPU RAM or walltime issues. For CUDA OOM, lower `preprocess.patches.patch_width`, e.g. `1024` for large ROIs or smaller GPUs. The default is bounded at `2048` so Cellpose does not receive full ROIs at once. If the GPU is still too small, request a larger GPU in `slurm.preprocess.gres` or set `preprocess.segmentation.gpu: false` as a slower CPU fallback.

**Viz resume.**
Percentile cache is parquet under `results/<exp>/qc/_cache/`; deleting it forces re-computation. `viz.cache_percentiles: false` disables.

**Reclaiming disk.**
Nothing is auto-deleted. `macsima-pipeline clean --config <cfg>` is dry-run by default; pass `--yes` with `--work` (Nextflow `work/` + `.nextflow*`, disables `-resume`), `--raw` (staged `raw/` tiles for samples already in `results/`), `--orphaned-zarr` (pre-refactor `artifacts/<exp>/*.zarr`), or `--everything`.

---

## License / containers

The native staging code is a reimplementation of [macsima2mc](https://github.com/SchapiroLabor/macsima2mc) (BSD-3-Clause). The mcmicro images (pulled by Nextflow for stage 2) carry their own licenses. This repo bundles only orchestration code.
