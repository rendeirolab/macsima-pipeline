# macsima-pipeline

Config-driven pipeline for MACSIMA multiplexed imaging data on a SLURM HPC.

One YAML config per experiment. One CLI (`macsima-pipeline`) drives five stages: **stage → mcmicro → preprocess → phenotype → viz**. Each stage submits a SLURM job (array where it makes sense); `all` chains them with `afterok` dependencies.

---

## Table of contents

1. [Concepts](#1-concepts)
2. [Layout](#2-repository-layout)
3. [Install](#3-install)
4. [First run — step by step](#4-first-run--step-by-step)
5. [Pipeline stages in detail](#5-pipeline-stages-in-detail)
6. [Config reference](#6-config-reference)
7. [SLURM behaviour](#7-slurm-behaviour)
8. [Outputs](#8-outputs)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Concepts

**Goal.** Take raw MACSima cycle output and produce: registered OME-TIFFs (per ROI), a `SpatialData` zarr with cell segmentations, a per-cell `AnnData` (`.h5ad`) with expression + ROI metadata, PDF marker/ROI visualisation grids, and a cell-map QC summary PDF.

**Why this repo exists.** Replaces per-experiment copy-paste scripts in `metpredict-macsima` (`expr10_*.py`, `expr10_*.sh`). One config, one CLI, same code for every experiment.

**Data flow.**

```
raw MACSima cycles (per ROI)
        │  panel   (sanity-check marker panel — at plan time)
        │  stage   (native: tifffile + ome-types + BaSiCPy; SLURM array, 1 task per ROI)
        ▼
mcmicro_output/{exp}/rack-…-roi-…-exp-2/   ← mcmicro-ingestable per-sample dirs
        │  mcmicro (Nextflow + Singularity, SLURM array, 1 task per sample)
        ▼
        sample/registration/*.ome.tif       ← Ashlar-registered pyramidal OME-TIFFs
        │  preprocess (SLURM GPU array, 1 task per ROI, then merge)
        ▼
{exp}_mcmicro_no_bs.zarr                    ← SpatialData (images + segmentations + cell tables)
{exp}_cell_expression_mcmicro_no_bs.h5ad    ← AnnData (cells × markers, obs merged with roi_metadata.csv)
        │  phenotype (1 GPU job; normalize + Astir + FlowSOM + spatial QC)
        ▼
{exp}_phenotyped_mcmicro_no_bs.h5ad         ← AnnData + obs['cell_type'], layers, obsm['spatial'], uns['phenotype']
        │  viz (1 CPU job; joblib-parallel)
        ▼
figures/{exp}/ *.pdf                        ← per-marker grids, per-ROI grids, RGB combinations
figures/{exp}/qc/*cell_maps_summary*.pdf    ← cell XY maps (colored by cell type) + QC summary
figures/{exp}/phenotype/*phenotype_summary*.pdf ← composition, confidence, spatial coherence
```

**Config inheritance.** `configs/default.yaml` holds every key. Each experiment cfg (`experiments/<exp>/config.yaml`) only needs `extends:` + the few keys that differ (typically `experiment.name`, `experiment.raw_root`, maybe `viz.combinations`). Merge is right-biased deep merge; child `None` does not clobber a non-null base.

**Suffix.** `mcmicro.background_subtraction: false` (default) appends `_no_bs` to zarr/h5ad filenames. Set true to use background-subtracted images and drop suffix.

---

## 2. Repository layout

```
configs/
  default.yaml            full schema + defaults — every key documented
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
  preprocess.py           stage 3 (run_inproc + SLURM worker array + merge)
  phenotype/              stage 4 (normalize + engines + spatial QC + report)
  lib/astir/              clean-room Astir model (independent impl; see NOTICE)
  viz/                    stage 5 (workers + plotting)
  slurm.py                sbatch render + sbatch submit
jobs/                     generated per-stage csv + sbatch (gitignored)
logs/                     SLURM stdout/stderr (gitignored)
artifacts/{exp}/          marker_panel.csv, zarr/h5ad (gitignored)
```

---

## 3. Install

Prereqs on the cluster: `uv`; plus Nextflow + `apptainer`/singularity **for the mcmicro stage only** — staging is now native Python and needs no container.

```bash
# 1. Clone
git clone <repo> && cd macsima-pipeline

# 2. Python env (uv reads pyproject.toml + uv.lock)
uv sync

# 3. Verify (staging is native — no macsima2mc container to pull)
uv run macsima-pipeline --help
```

mcmicro pulls its own images via Nextflow on first run; nothing to do here besides having Nextflow on `PATH`.

---

## 4. First run — step by step

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

Generate a template for the exact ROIs the pipeline will process (reuses the
staging ROI discovery, so `roi_exclude` etc. are applied) — then just fill in the
columns:

```bash
uv run macsima-pipeline gen-roi-metadata --config experiments/myexp/config.yaml \
    --columns Sample --columns Patient_ID
# → wrote roi_metadata_myexp.csv  (or to experiment.roi_metadata_csv if set)
```

It needs `raw_root` mounted (run on a node where RawData is reachable) and refuses
to overwrite an existing file unless you pass `--force`.

You can also snapshot the marker panel `macsima2mc` produced (a review/curation
artifact with a normalized `remove` column) once staging has run:

```bash
uv run macsima-pipeline gen-markers --config experiments/myexp/config.yaml
```

### 4.3b (Optional) Phenotype signature

The phenotype stage (Astir + FlowSOM) needs a **signature matrix** — a marker→cell-type
table (`phenotype.signature_matrix`). If it is unset, the phenotype stage skips (the chain
still runs). Scaffold a template from your panel (post-staging), then curate it:

```bash
uv run macsima-pipeline gen-signature --config experiments/myexp/config.yaml
# → wrote signature_myexp.yaml (panel markers listed + example cell types to edit)
```

Fill in `positive`/`negative` markers per cell type, then set
`phenotype.signature_matrix` to that path. See `configs/signature_example.yaml`.

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

### 4.9 Batch — multiple experiments in one config

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

Every command works on a batch config unchanged — it runs **one independent chain
per experiment** (outputs stay isolated by `experiment.name`):

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
write nothing extra. Marker panels are **not** assumed shared across experiments — each
reads its own `markers.csv` exactly as before.

**Phenotyping in a batch.** `all` phenotypes each experiment **independently** (one model
per experiment; cell-type *names* come from the shared signature, but model calibration and
FlowSOM cluster ids are not comparable across experiments, and there is no cross-experiment
batch correction). For **joint** phenotyping — one Astir/FlowSOM model over all experiments,
comparable labels, cross-experiment batch correction — run the dedicated command after the
per-experiment preprocess outputs exist:

```bash
uv run macsima-pipeline phenotype-joint --config experiments/<batch>/config.yaml --submit
```

It concatenates every experiment's cell-expression h5ad (inner-join on shared markers), tags
each cell with `experiment` + a unique `sample` (experiment|ROI), fits the engines once,
writes a combined phenotyped h5ad under `artifacts/<batch-folder>/`, and splits the joint
labels back into each experiment's phenotyped h5ad (so per-experiment viz picks them up).
`--batch-key` (default `sample`) sets the batch-correction unit; `--inproc` runs locally;
`--only NAME` restricts the set.

---

## 5. Pipeline stages in detail

### Stage 0 — `panel` (runs automatically before `stage`)

From the raw filenames alone (fast, no pixel reads) writes `artifacts/<exp>/marker_panel.csv`
(per-cycle panel summary — sanity-check that the run acquired what you expect) and validates the
panel: reference marker present in every cycle, consistent markers across ROIs, background
acquisitions present. `stage` runs this at plan time; run it standalone with
`macsima-pipeline panel --config …`. (Cell-type signatures for phenotyping are produced
separately — see the phenotype stage / `gen-signature`.)

### Stage 1 — `stage`

Native Python — no container. `staging_core.py` reimplements macsima2mc v1.3.1 with `tifffile` +
`ome-types`. Discovers ROIs via `experiment.raw_root` + `experiment.roi_glob` minus `roi_exclude`,
writes `jobs/staging_<exp>.csv` (one row per ROI = one SLURM array task); each task stages every
`*Cycle*` folder of its ROI. Per cycle it parses the MACSima filenames, groups tiles by
`(source, exposure_level)`, orders channels reference-marker-first (backfilling DAPI into exposure
levels that didn't reacquire it), optionally applies BaSiCPy flatfield correction
(`staging.illumination_correction`, default on → `corr_` prefix), and writes multi-series
OME-TIFFs + `markers.csv` to `mcmicro_output/<exp>/rack-X-well-Y-roi-Z-exp-N/`. Behaviour is
tunable under `staging:` in the config; output is drop-in compatible with the previous container.

### Stage 2 — `mcmicro`

For each staged sample dir matching `mcmicro.sample_pattern` (default `rack-*-well-*-roi-*-exp-2`), runs `nextflow run labsyspharm/mcmicro -profile singularity` with `mcmicro.params_yaml`. Produces Ashlar-registered pyramidal OME-TIFFs at `<sample>/registration/<…>exp-2.ome.tif` (or `<sample>/background/<…>_backsub.ome.tif` if `background_subtraction: true`).

### Stage 3 — `preprocess`

On SLURM, this is a two-phase stage: a GPU worker array runs one task per concrete `(background variant, ROI image)`, then a CPU merge job assembles final experiment-level outputs. `--inproc` is still available for local/debug runs and iterates all ROIs in one process. Steps per ROI:

1. Load registered OME-TIFF as dask array, keep channels listed in the mcmicro `markers.csv` with `remove != True`.
2. Wrap as `Image2DModel` with `scale_factors` pyramid (default `[2, 4]`).
3. `sopa.make_image_patches` → `sopa.segmentation.custom_staining_based` with Cellpose4 (`cpsam` model) on DAPI.
4. `sopa.aggregate` → per-cell expression table.

Each worker writes intermediate per-ROI parts under `paths.preprocess_parts_dir`. The merge job validates all expected parts, concatenates per-ROI cell tables into one `AnnData`, records explicit `obs["ROI"]`, left-joins `roi_metadata.csv` on `ROI`, writes the final `.h5ad`, and combines per-ROI SpatialData elements into the final `.zarr`.

The public `macsima-pipeline preprocess --submit` command submits the worker array first and then a merge job dependent on that array. Before MCMICRO outputs exist, dry-run reports that exact array planning is deferred until the OME-TIFFs are available.

### Stage 4 — `phenotype`

Reads the per-variant `.h5ad` (all ROIs jointly) and assigns cell types, writing a
separate `{exp}_phenotyped_mcmicro{suffix}.h5ad`. One GPU job per variant; `--inproc`
runs locally. Skips gracefully (exit 0, chain still proceeds) when disabled or when no
`phenotype.signature_matrix` is set. Steps:

1. **Normalize** (`phenotype.normalize`): stash raw → per-marker winsorize → arcsinh →
   z-score. Keeps `layers['counts']` (raw) and `layers['zscore']` (normalized).
2. **Batch** (`phenotype.batch`): per-ROI z-score by default (ComBat / quantile options),
   at the intensity stage so markers stay interpretable.
3. **Engines** (`phenotype.engines`), both driven by the same signature matrix:
   - **Astir** — clean-room probabilistic model (`src/macsima_pipeline/lib/astir`, an
     independent implementation of Geuenich et al. 2021; **not** the GPL-2.0 package —
     see its `NOTICE`). Reads RAW counts; GPU-batched. → per-cell probabilities + labels.
   - **FlowSOM** — SOM + consensus metaclustering (reproducible), metaclusters labeled
     against the signature. Reads the z-scored layer.
4. **Cross-engine agreement** (Cohen's κ, ARI) — a confidence signal; disagreement
   flags batch/ambiguous markers.
5. **Spatial QC** (`phenotype.spatial_qc`): neighborhood enrichment + same-type homophily
   per ROI. This is the automatic version of "do the labels make sense on the map?".

Writes `obs['cell_type' | 'cell_type_coarse' | 'cell_type_confidence' | 'flowsom' |
'pheno_agree']`, `obsm['spatial']`, `uns['phenotype']`, and a QC PDF under
`figures/{exp}/phenotype/`. The **signature matrix** is a small YAML naming expected
positive/negative markers per cell type (optional lineage `parent`); point
`phenotype.signature_matrix` at it.

### Stage 5 — `viz`

Loads the mcmicro OME-TIFFs, plus the variant `.h5ad` for cell-map QC when available
(prefers the phenotyped h5ad, so cell maps are colored by `cell_type` and a spatial
coherence page is added). For each marker, picks the pyramid level whose largest XY dim ≤
`viz.target_max_dim` (default 2048), computes 1–99 percentile clips (cached to parquet
for resume), and renders:

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
| `phenotype` | `engines`, `primary_engine` | which engines to run (`astir`, `flowsom`); which populates `cell_type` |
| `phenotype.normalize` | `transform`, `cofactor`, `clip_percentile`, `zscore` | per-marker normalization (arcsinh; tune `cofactor` to your intensity scale) |
| `phenotype.batch` | `method`, `batch_key` | intensity-stage batch handling (`zscore_per_roi` default) |
| `phenotype.astir` | `cofactor`, `min_confidence`, `include_batch_covariate` | Astir engine (reads RAW counts; per-ROI baseline) |
| `phenotype.flowsom` | `grid_size`, `n_metaclusters`, `train_subsample` | FlowSOM engine (reads z-scored layer) |
| `phenotype.spatial_qc` | `n_neighs`, `nhood_enrichment`, `homophily` | spatial-coherence QC |
| `viz` | `combinations` | list of `{name, markers: [m1, m2, m3]}` → RGB plots |
| `viz` | `cell_maps`, `cell_map_marker_top_n`, `cell_map_point_size` | default-on cell XY + expression QC summary PDF controls |
| `slurm.<stage>` | `partition`, `qos`, `cpus`, `mem`, `time`, `gres`, `comment` | sbatch header values |

**Placeholder expansion.** `{experiment_name}` and `{suffix}` are expanded in `paths.zarr_out`, `paths.h5ad_out`, `paths.figures_dir`.

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

| Path | Stage | Contents |
|---|---|---|
| `mcmicro_output/<exp>/<sample>/` | stage 1 | mcmicro-ingestable cycle dirs |
| `mcmicro_output/<exp>/<sample>/registration/*.ome.tif` | stage 2 | registered pyramidal OME-TIFF per ROI |
| `<exp>_mcmicro_no_bs.zarr/` | stage 3 | SpatialData (images + segmentations + cell expression tables) |
| `<exp>_cell_expression_mcmicro_no_bs.h5ad` | stage 3 | AnnData (cells × markers, obs with ROI + metadata) |
| `figures/<exp>/*.pdf` | stage 4 | marker grids, ROI grids, RGB combinations |
| `figures/<exp>/qc/*cell_maps_summary*.pdf` | stage 4 | experiment and per-ROI cell-map QC summary |
| `jobs/`, `logs/` | all | sbatch + CSV + SLURM logs |

---

## 9. Troubleshooting

**`ValidationError: viz.combinations Input should be a valid list`**
Your child config has `viz: combinations:` with only commented items — that parses as `None`. Use `combinations: []` (loader now also tolerates `None`, but be explicit).

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
Percentile cache is parquet under `figures/<exp>/`; deleting it forces re-computation. `viz.cache_percentiles: false` disables.

---

## License / containers

The native staging code is a reimplementation of [macsima2mc](https://github.com/SchapiroLabor/macsima2mc) (BSD-3-Clause). The mcmicro images (pulled by Nextflow for stage 2) carry their own licenses. This repo bundles only orchestration code.
