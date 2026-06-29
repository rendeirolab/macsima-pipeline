# macsima-pipeline

Config-driven pipeline for MACSIMA multiplexed imaging data on a SLURM HPC.

One YAML config per experiment. One CLI (`macsima-pipeline`) drives four stages: **stage → mcmicro → preprocess → viz**. Each stage submits a SLURM job (array where it makes sense); `all` chains them with `afterok` dependencies.

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

**Goal.** Take raw MACSima cycle output and produce: registered OME-TIFFs (per ROI), a `SpatialData` zarr with cell segmentations, a per-cell `AnnData` (`.h5ad`) with expression + ROI metadata, and PDF marker/ROI visualisation grids.

**Why this repo exists.** Replaces per-experiment copy-paste scripts in `metpredict-macsima` (`expr10_*.py`, `expr10_*.sh`). One config, one CLI, same code for every experiment.

**Data flow.**

```
raw MACSima cycles (per ROI)
        │  stage   (apptainer macsima2mc.sif, SLURM array, 1 task per ROI)
        ▼
mcmicro_output/{exp}/rack-…-roi-…-exp-2/   ← mcmicro-ingestable per-sample dirs
        │  mcmicro (Nextflow + Singularity, SLURM array, 1 task per sample)
        ▼
        sample/registration/*.ome.tif       ← Ashlar-registered pyramidal OME-TIFFs
        │  preprocess (1 GPU job; iterates all ROIs in-process)
        ▼
{exp}_mcmicro_no_bs.zarr                    ← SpatialData (images + segmentations + cell tables)
{exp}_cell_expression_mcmicro_no_bs.h5ad    ← AnnData (cells × markers, obs merged with roi_metadata.csv)
        │  viz (1 CPU job; joblib-parallel)
        ▼
figures/{exp}/ *.pdf                        ← per-marker grids, per-ROI grids, RGB combinations
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
  staging.py              stage 1
  mcmicro.py              stage 2
  preprocess.py           stage 3 (run_inproc + sbatch wrapper)
  viz/                    stage 4 (workers + plotting)
  slurm.py                sbatch render + sbatch submit
jobs/                     generated per-stage csv + sbatch (gitignored)
logs/                     SLURM stdout/stderr (gitignored)
macsima2mc.sif            apptainer image (not in repo — see Install)
```

---

## 3. Install

Prereqs on the cluster: `apptainer` (a.k.a. singularity), Nextflow (for mcmicro), `uv`.

```bash
# 1. Clone
git clone <repo> && cd macsima-pipeline

# 2. Python env (uv reads pyproject.toml + uv.lock)
uv sync

# 3. Containers — not bundled. Pull macsima2mc:
module load apptainer
apptainer pull macsima2mc.sif docker://ghcr.io/schapirolabor/macsima2mc:v1.3.1

# 4. Verify
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
# stage=12345 mcmicro=12346 preprocess=12347 viz=12348
```

Each downstream job uses `--dependency=afterok:<prev>` so it only runs on success.
For `all --submit`, the reported `mcmicro` id is a small launcher/barrier job:
it starts the real MCMICRO array after staging has produced sample folders, waits
for that array to finish, and then releases preprocess.

### 4.7 Monitor

```bash
squeue -u $USER
tail -F logs/staging_myexp_<jobid>_*.out
```

### 4.8 Re-run a single stage

Stage outputs are deterministic per (config, raw data). To redo just one stage, delete its outputs (e.g. `mcmicro_output/myexp/`) and submit it again. The `--dependency` flag lets you re-attach downstream stages without rerunning everything.

---

## 5. Pipeline stages in detail

### Stage 1 — `stage`

Calls `macsima2mc.py` (inside `macsima2mc.sif`) once per **(ROI × cycle)**. Discovers ROIs via `experiment.raw_root` + `experiment.roi_glob` minus `roi_exclude`. Writes `jobs/staging_<exp>.csv` (one row per ROI = one SLURM array task). Output: `mcmicro_output/<exp>/rack-X-well-Y-roi-Z-exp-2/` per-sample dirs ready for mcmicro.

### Stage 2 — `mcmicro`

For each staged sample dir matching `mcmicro.sample_pattern` (default `rack-*-well-*-roi-*-exp-2`), runs `nextflow run labsyspharm/mcmicro -profile singularity` with `mcmicro.params_yaml`. Produces Ashlar-registered pyramidal OME-TIFFs at `<sample>/registration/<…>exp-2.ome.tif` (or `<sample>/background/<…>_backsub.ome.tif` if `background_subtraction: true`).

### Stage 3 — `preprocess`

Single GPU job (NOT a SLURM array — one process iterates all ROIs because sopa shares state). Steps per ROI:

1. Load registered OME-TIFF as dask array, keep channels listed in the mcmicro `markers.csv` with `remove != True`.
2. Wrap as `Image2DModel` with `scale_factors` pyramid (default `[2, 4]`).
3. `sopa.make_image_patches` → `sopa.segmentation.custom_staining_based` with Cellpose4 (`cpsam` model) on DAPI.
4. `sopa.aggregate` → per-cell expression table.

After all ROIs: concat per-ROI cell tables into one `AnnData`, derive `obs["ROI"]` from the slide id, left-join `roi_metadata.csv` on `ROI`, write `.h5ad` and the full `.zarr`.

The SLURM wrapper just re-invokes `macsima-pipeline preprocess --inproc` inside the allocated job.

### Stage 4 — `viz`

Loads the `.zarr`. For each marker, picks the pyramid level whose largest XY dim ≤ `viz.target_max_dim` (default 2048), computes 1–99 percentile clips (cached to parquet for resume), and renders:

- one PDF per marker showing that marker across all ROIs (grid),
- one PDF per ROI showing all markers (grid),
- one PDF per entry in `viz.combinations` (RGB composite of 3 markers).

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
| `viz` | `combinations` | list of `{name, markers: [m1, m2, m3]}` → RGB plots |
| `slurm.<stage>` | `partition`, `qos`, `cpus`, `mem`, `time`, `gres`, `comment` | sbatch header values |

**Placeholder expansion.** `{experiment_name}` and `{suffix}` are expanded in `paths.zarr_out`, `paths.h5ad_out`, `paths.figures_dir`.

---

## 7. SLURM behaviour

- Templates live in `templates/*.sbatch.j2`. `slurm.py` renders the `#SBATCH` headers from `slurm.<stage>` and embeds the stage-specific body.
- Rendered sbatch + jobs CSV go to `jobs/<stage>_<exp>.{sbatch,csv}`.
- Logs go to `logs/<stage>_<exp>_%A_%a.out`.
- Array stages (`stage`, `mcmicro`) read their work item from the CSV by `$SLURM_ARRAY_TASK_ID` (`awk` on column 1).
- Non-array stages (`preprocess`, `viz`) submit a single job.
- `all --submit` submits staging first, then a MCMICRO launcher job with an `afterok` dependency on staging. The launcher plans/submits the real MCMICRO array with `sbatch --wait`, so preprocess depends on MCMICRO completion instead of on early sample discovery.

Defaults (override under `slurm.<stage>` in your config):

| Stage | Partition | CPU / Mem / Time | GPU |
|---|---|---|---|
| stage | tinyq | 8 / 32G / 2h | — |
| mcmicro | shortq | 16 / 64G / 8h | — |
| preprocess | gpu | 16 / 100G / 6h | `gpu:h100pcie:1` |
| viz | shortq | 8 / 40G / 4h | — |

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
Bump `slurm.preprocess.mem` / `time`. Single GPU process iterates all ROIs; if you have many large ROIs, expect long runtime.

**Viz resume.**
Percentile cache is parquet under `figures/<exp>/`; deleting it forces re-computation. `viz.cache_percentiles: false` disables.

---

## License / containers

`macsima2mc.sif` (ghcr.io/schapirolabor/macsima2mc) and the mcmicro images carry their own licenses. This repo bundles only orchestration code.
