"""Stage 3: segment cells from the consolidated OME-TIFFs and export cell tables.

Each ROI is loaded into an in-memory SpatialData, segmented with Cellpose4 (via
sopa), then two artifacts are written directly under ``results/<exp>/``:
per-ROI segmentation polygons as GeoParquet (``segmentation/``) and a per-ROI
AnnData part that the merge step concatenates into ``cells/<exp>_cells.h5ad``.
No SpatialData Zarr store is persisted — the pixels already live once as
OME-TIFFs and nothing downstream reads a merged store.

The local/debug path still runs one in-process pass over all ROIs. The SLURM
path fans out one GPU array task per (background variant, ROI), then a CPU merge
job concatenates the AnnData parts and deletes them.
"""

from __future__ import annotations

import csv
import logging
import os
import shlex
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .finalize import consolidate_images
from .slurm import render_sbatch, submit, write_sbatch
from .utils import ensure_dir, roi_name_from_results_stem

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessJob:
    job_id: int
    bg: bool
    variant: str
    roi_name: str
    image_path: Path
    seg_path: Path  # final per-ROI segmentation parquet under results/<exp>/segmentation/
    part_h5ad: Path  # transient per-ROI cell-table part (removed after merge)


@dataclass(frozen=True)
class PreprocessPlan:
    jobs_csv: Path
    worker_sbatch: Path
    merge_sbatch: Path
    n_jobs: int
    array_limit: int | None


class PreprocessPlanningDeferred(RuntimeError):
    """Raised when exact ROI array planning must wait for mcmicro outputs."""


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


_XY_COLUMN_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("X", "Y"),
    ("centroid_x", "centroid_y"),
    ("x_centroid", "y_centroid"),
    ("center_x", "center_y"),
    ("CenterX", "CenterY"),
    ("X_centroid", "Y_centroid"),
)


def _variant_label(bg: bool) -> str:
    return "bg-sub" if bg else "no-bg-sub"


def _roi_label_from_mcmicro_name(roi_name: str) -> str:
    if roi_name.startswith("ROI"):
        return roi_name
    try:
        return f"ROI{int(roi_name)}"
    except ValueError:
        return f"ROI{roi_name}"


def _find_images(cfg: Config, bg: bool) -> list[Path]:
    """Consolidated per-ROI OME-TIFFs for one variant: results/<exp>/images/<variant>/*.ome.tif."""
    base = cfg.variant_images_dir(bg)
    images = sorted(base.glob("*.ome.tif"))
    if not images:
        raise FileNotFoundError(f"No consolidated images found under {base}")
    return images


def _load_channel_info(cfg: Config, images: list[Path], bg: bool) -> pd.DataFrame:
    """Resolve the markers CSV consolidated alongside the images (results/<exp>/images/)."""
    csv_path = cfg.images_dir() / (cfg.mcmicro.markers_bs_csv if bg else cfg.mcmicro.markers_csv)
    ci = pd.read_csv(csv_path)
    # Capture the TIFF channel position before any filtering. DataFrame row
    # numbers after filtering are not physical channel indices.
    ci["channel_index"] = range(len(ci))
    if bg:
        ci = ci.drop_duplicates(subset="marker_name").reset_index(drop=True)
    else:
        ci = ci[ci["remove"] != True]  # noqa: E712 — pandas mask, not bool identity
        ci = ci.drop(columns=["remove"]).drop_duplicates(subset="marker_name").reset_index(drop=True)
    log.info("channel_info: [count]%d[/] markers from [path]%s[/]", len(ci), csv_path)
    return ci


def _obs_has_xy(obs: pd.DataFrame) -> bool:
    for x_col, y_col in _XY_COLUMN_CANDIDATES:
        if x_col in obs.columns and y_col in obs.columns:
            x = pd.to_numeric(obs[x_col], errors="coerce")
            y = pd.to_numeric(obs[y_col], errors="coerce")
            if bool((x.notna() & y.notna()).any()):
                return True
    return False


def _centroids_from_shapes(shapes) -> pd.DataFrame | None:
    if shapes is None:
        return None
    if hasattr(shapes, "compute"):
        shapes = shapes.compute()
    if not hasattr(shapes, "geometry"):
        return None
    try:
        centroids = shapes.geometry.centroid
        df = pd.DataFrame(
            {
                "centroid_x": centroids.x.to_numpy(),
                "centroid_y": centroids.y.to_numpy(),
            },
            index=shapes.index.astype(str),
        )
    except Exception as e:  # pragma: no cover - depends on SpatialData/Sopa shape backend
        log.warning("could not compute segmentation centroids: %s", e)
        return None
    if "cell_id" in getattr(shapes, "columns", []):
        df.index = shapes["cell_id"].astype(str).to_numpy()
    return df


def _add_centroids_from_shapes(table, shapes, table_key: str) -> None:
    """Best-effort centroid preservation for downstream cell-map QC."""
    if _obs_has_xy(table.obs):
        return
    centroids = _centroids_from_shapes(shapes)
    if centroids is None or centroids.empty:
        log.warning("no centroid coordinates available for %s", table_key)
        return

    obs = table.obs
    for key_col in ("cell_id", "instance_id", "label", "index"):
        if key_col not in obs.columns:
            continue
        keys = obs[key_col].astype(str)
        if bool(keys.isin(centroids.index).any()):
            obs["centroid_x"] = keys.map(centroids["centroid_x"])
            obs["centroid_y"] = keys.map(centroids["centroid_y"])
            return

    obs_index = obs.index.astype(str)
    if bool(pd.Series(obs_index).isin(centroids.index).any()):
        obs["centroid_x"] = obs_index.map(centroids["centroid_x"])
        obs["centroid_y"] = obs_index.map(centroids["centroid_y"])
        return

    if len(obs) == len(centroids):
        obs["centroid_x"] = centroids["centroid_x"].to_numpy()
        obs["centroid_y"] = centroids["centroid_y"].to_numpy()
        return

    log.warning("could not align segmentation centroids to table %s", table_key)


def _segmentation_callable(cfg: Config):
    """Return a sopa-compatible callable that runs the configured segmentation model."""
    seg = cfg.preprocess.segmentation
    if seg.method != "cellpose4":
        raise NotImplementedError(f"Segmentation method {seg.method!r} not implemented yet")

    model = None
    torch = None

    def _cellpose4(image):
        nonlocal model, torch
        from cellpose import models

        if model is None:
            model = models.CellposeModel(pretrained_model=seg.model, gpu=seg.gpu)
        if torch is None:
            try:
                import torch as torch_module
            except ImportError:
                torch_module = False
            torch = torch_module

        if torch:
            with torch.inference_mode():
                masks, _, _ = model.eval(image)
            if seg.gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            masks, _, _ = model.eval(image)
        return masks

    return _cellpose4


def _build_spatialdata_for_images(cfg: Config, images: Iterable[Path], channel_info: pd.DataFrame):
    from dask_image.imread import imread
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel

    imgs_data: dict[str, Image2DModel] = {}
    for img_path in images:
        img = imread(str(img_path))[channel_info["channel_index"]]
        roi_name = roi_name_from_results_stem(img_path.stem)
        imgs_data[roi_name] = Image2DModel.parse(
            img,
            dims=("c", "y", "x"),
            c_coords=channel_info["marker_name"].values,
            scale_factors=cfg.preprocess.scale_factors,
        )
        log.info("loaded ROI [stage]%s[/] from [path]%s[/]", roi_name, img_path)
    return SpatialData(images=imgs_data)


def _segment_spatialdata(cfg: Config, sdata) -> None:
    import sopa

    seg_fn = _segmentation_callable(cfg)
    for key in list(sdata.images.keys()):
        log.info("[ok]segmenting[/] ROI [stage]%s[/]", key)
        sopa.make_image_patches(
            sdata,
            image_key=key,
            patch_width=cfg.preprocess.patches.patch_width,
        )
        sopa.segmentation.custom_staining_based(
            sdata,
            method=seg_fn,
            image_key=key,
            channels=cfg.preprocess.segmentation.channels,
            key_added=f"{key}_segmentation",
            min_area=cfg.preprocess.segmentation.min_area,
        )
        sopa.aggregate(
            sdata,
            image_key=key,
            shapes_key=f"{key}_segmentation",
            key_added=f"{key}_cell_expression",
        )


def _adata_from_sdata_tables(sdata, roi_labels: dict[str, str] | None = None):
    import anndata as ad

    cellexp_keys = [k for k in sdata.tables if "cell_expression" in k]
    tables = []
    for key in cellexp_keys:
        table = sdata[key]
        image_key = key.replace("_cell_expression", "")
        shapes_key = key.replace("_cell_expression", "_segmentation")
        shapes = sdata.shapes.get(shapes_key) if hasattr(sdata, "shapes") else None
        _add_centroids_from_shapes(table, shapes, key)
        if roi_labels is not None:
            table.obs["ROI"] = roi_labels[image_key]
        tables.append(table)

    adata = ad.concat(tables)
    if "ROI" not in adata.obs.columns:
        if "slide" not in adata.obs.columns:
            raise KeyError("Cannot derive ROI labels: AnnData obs lacks both ROI and slide columns")
        adata.obs["ROI"] = adata.obs["slide"].map(lambda x: f"ROI{int(str(x).split('_')[0])}")

    adata.obs.reset_index(inplace=True)
    adata.obs.index = adata.obs.index.astype(str)
    # Keep `cell_id` — it is the join key back to the segmentation parquet.
    drop_cols = [c for c in ("region", "index", "slide") if c in adata.obs.columns]
    adata.obs.drop(columns=drop_cols, inplace=True)
    return adata


def _join_roi_metadata(cfg: Config, adata) -> None:
    if cfg.experiment.roi_metadata_csv is None:
        return
    meta_path = cfg.paths.work_dir / cfg.experiment.roi_metadata_csv
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        adata.obs = adata.obs.merge(meta, on="ROI", how="left").set_index(adata.obs.index)
        log.info("joined ROI metadata from [path]%s[/]", meta_path)
    else:
        log.warning("[bad]ROI metadata csv missing[/]: [path]%s[/]", meta_path)


# --------------------------------------------------------------------------- #
#  Local in-process pipeline                                                  #
# --------------------------------------------------------------------------- #


def _run_variant(cfg: Config, bg: bool) -> tuple[list[Path], Path]:
    """Run the full pipeline for one background-subtraction variant.

    Returns (list of per-ROI segmentation parquet paths, merged cells h5ad path).
    """
    label = _variant_label(bg)
    log.info("[ok]variant[/]: [stage]%s[/] (suffix=[path]%s[/])", label, cfg.suffix_for(bg) or "(none)")

    images = _find_images(cfg, bg)
    log.info("found [count]%d[/] image(s)", len(images))
    channel_info = _load_channel_info(cfg, images, bg)
    roi_labels = {
        roi_name_from_results_stem(img_path.stem): _roi_label_from_mcmicro_name(
            roi_name_from_results_stem(img_path.stem)
        )
        for img_path in images
    }

    sdata = _build_spatialdata_for_images(cfg, images, channel_info)
    _segment_spatialdata(cfg, sdata)

    seg_paths: list[Path] = []
    for roi_key, roi_label in roi_labels.items():
        dest = _segmentation_path(cfg, bg, roi_key)
        _write_segmentation_parquet(sdata, roi_key, roi_label, dest)
        seg_paths.append(dest)

    adata = _adata_from_sdata_tables(sdata, roi_labels)
    _join_roi_metadata(cfg, adata)

    h5ad_path = cfg.h5ad_path(bg)
    ensure_dir(h5ad_path.parent)
    log.info("writing AnnData -> [path]%s[/]", h5ad_path)
    adata.write_h5ad(str(h5ad_path))
    return seg_paths, h5ad_path


def run_inproc(cfg: Config) -> list[tuple[list[Path], Path]]:
    """Execute the preprocessing pipeline for every configured bg-sub variant.

    Returns a list of (segmentation_parquet_paths, h5ad_path) tuples — one per
    variant. In "auto" mode this can be two entries (bg-sub + no-bg-sub) when
    both image sets exist; in explicit bool mode it's always a single entry.
    """
    consolidate_images(cfg)  # idempotent: hardlink mcmicro outputs into results/<exp>/images
    modes = cfg.bg_modes()
    log.info(
        "preprocess variants to run: [count]%d[/] (%s)",
        len(modes),
        ", ".join(_variant_label(m) for m in modes),
    )
    results: list[tuple[list[Path], Path]] = []
    for bg in modes:
        try:
            results.append(_run_variant(cfg, bg))
        except FileNotFoundError as e:
            # In auto mode, tolerate a missing variant (e.g. mcmicro hasn't
            # produced one of the two image sets).
            if isinstance(cfg.mcmicro.background_subtraction, bool):
                raise
            log.warning("[warn]skipping variant[/] bg=%s: %s", bg, e)
    if not results:
        raise FileNotFoundError("No preprocess variants produced output; check mcmicro outputs")
    return results


# --------------------------------------------------------------------------- #
#  Worker jobs and merge                                                       #
# --------------------------------------------------------------------------- #


def _segmentation_path(cfg: Config, bg: bool, roi_name: str) -> Path:
    """Final per-ROI segmentation parquet under results/<exp>/segmentation/."""
    label = _roi_label_from_mcmicro_name(roi_name)
    return cfg.segmentation_dir() / f"{cfg.experiment.name}_{label}_segmentation{cfg.suffix_for(bg)}.parquet"


def _part_paths(cfg: Config, bg: bool, roi_name: str) -> tuple[Path, Path]:
    """(final segmentation parquet, transient per-ROI cell-table h5ad part)."""
    seg_path = _segmentation_path(cfg, bg, roi_name)
    part_h5ad = cfg.preprocess_parts_path(bg) / f"{roi_name}.h5ad"
    return seg_path, part_h5ad


def discover_jobs(cfg: Config) -> list[PreprocessJob]:
    """Discover concrete preprocessing work items from the consolidated images."""
    consolidate_images(cfg)  # idempotent: hardlink mcmicro outputs into results/<exp>/images
    jobs: list[PreprocessJob] = []
    missing: list[str] = []
    for bg in cfg.bg_modes():
        try:
            images = _find_images(cfg, bg)
        except FileNotFoundError as e:
            missing.append(f"{_variant_label(bg)}: {e}")
            continue

        for img_path in images:
            roi_name = roi_name_from_results_stem(img_path.stem)
            seg_path, part_h5ad = _part_paths(cfg, bg, roi_name)
            jobs.append(
                PreprocessJob(
                    job_id=len(jobs) + 1,
                    bg=bg,
                    variant=_variant_label(bg),
                    roi_name=roi_name,
                    image_path=img_path,
                    seg_path=seg_path,
                    part_h5ad=part_h5ad,
                )
            )

    if not jobs:
        details = "; ".join(missing) if missing else "no matching variants found"
        raise PreprocessPlanningDeferred(
            "No preprocess images are available yet; exact SLURM array planning is "
            f"deferred until mcmicro has produced OME-TIFF outputs. Missing: {details}"
        )
    if missing and isinstance(cfg.mcmicro.background_subtraction, str):
        log.warning("[warn]skipping missing preprocess variant(s)[/]: %s", "; ".join(missing))
    return jobs


def write_jobs_csv(cfg: Config, jobs: list[PreprocessJob]) -> Path:
    path = cfg.jobs_csv("preprocess")
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["job_id", "bg", "variant", "roi_name", "image_path", "seg_path", "part_h5ad"],
            lineterminator="\n",
        )
        w.writeheader()
        for job in jobs:
            w.writerow(
                {
                    "job_id": job.job_id,
                    "bg": str(job.bg).lower(),
                    "variant": job.variant,
                    "roi_name": job.roi_name,
                    "image_path": str(job.image_path),
                    "seg_path": str(job.seg_path),
                    "part_h5ad": str(job.part_h5ad),
                }
            )
    log.info("wrote [path]%s[/] ([count]%d[/] rows)", path, len(jobs))
    return path


def read_jobs_csv(path: Path) -> list[PreprocessJob]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    jobs = []
    for row in rows:
        jobs.append(
            PreprocessJob(
                job_id=int(row["job_id"]),
                bg=row["bg"].strip().lower() in {"1", "true", "yes"},
                variant=row["variant"],
                roi_name=row["roi_name"],
                image_path=Path(row["image_path"]),
                seg_path=Path(row["seg_path"]),
                part_h5ad=Path(row["part_h5ad"]),
            )
        )
    return jobs


def read_job_for_task(jobs_csv: Path, task_id: int) -> PreprocessJob:
    for job in read_jobs_csv(jobs_csv):
        if job.job_id == task_id:
            return job
    raise RuntimeError(f"No row for task_id={task_id} in {jobs_csv}")


def _write_segmentation_parquet(sdata, roi_key: str, roi_label: str, dest: Path) -> None:
    """Persist one ROI's cell-segmentation polygons as GeoParquet.

    Read from ``sdata.shapes`` *after* ``sopa.aggregate`` so the rows are the
    surviving cells, index-aligned to the cell table by ``cell_id``. Adds
    ``cell_id``/``ROI``/``centroid_x``/``centroid_y``/``area`` so the file is
    self-contained for external tools (napari/QuPath/spatial stats). Written
    atomically (temp -> os.replace).
    """
    shapes = sdata.shapes.get(f"{roi_key}_segmentation") if hasattr(sdata, "shapes") else None
    if shapes is None:
        log.warning("no segmentation shapes for ROI %s; skipping parquet", roi_key)
        return
    if hasattr(shapes, "compute"):
        shapes = shapes.compute()
    gdf = shapes.copy()
    gdf["cell_id"] = gdf.index.astype(str)
    gdf["ROI"] = roi_label
    centroids = gdf.geometry.centroid
    gdf["centroid_x"] = centroids.x.to_numpy()
    gdf["centroid_y"] = centroids.y.to_numpy()
    gdf["area"] = gdf.geometry.area.to_numpy()
    gdf = gdf.reset_index(drop=True)

    ensure_dir(dest.parent)
    tmp = dest.with_name(dest.name + ".tmp")
    gdf.to_parquet(tmp)
    os.replace(tmp, dest)
    log.info("segmentation parquet -> [path]%s[/] ([count]%d[/] cells)", dest, len(gdf))


def process_job(cfg: Config, job: PreprocessJob) -> tuple[Path, Path]:
    """Run one ROI/variant worker: write the segmentation parquet + a cell-table part."""
    log.info(
        "preprocess worker: job=[count]%d[/] variant=[stage]%s[/] roi=[stage]%s[/] image=[path]%s[/]",
        job.job_id,
        job.variant,
        job.roi_name,
        job.image_path,
    )
    channel_info = _load_channel_info(cfg, [job.image_path], job.bg)
    sdata = _build_spatialdata_for_images(cfg, [job.image_path], channel_info)
    _segment_spatialdata(cfg, sdata)

    roi_label = _roi_label_from_mcmicro_name(job.roi_name)
    _write_segmentation_parquet(sdata, job.roi_name, roi_label, job.seg_path)

    adata = _adata_from_sdata_tables(sdata, {job.roi_name: roi_label})
    ensure_dir(job.part_h5ad.parent)
    log.info("writing ROI cell-table part -> [path]%s[/]", job.part_h5ad)
    adata.write_h5ad(str(job.part_h5ad))
    return job.seg_path, job.part_h5ad


def run_worker(cfg: Config, jobs_csv: Path, task_id: int) -> tuple[Path, Path]:
    return process_job(cfg, read_job_for_task(jobs_csv, task_id))


def _validate_parts(jobs: list[PreprocessJob]) -> None:
    missing = []
    for job in jobs:
        if not job.seg_path.is_file():
            missing.append(str(job.seg_path))
        if not job.part_h5ad.is_file():
            missing.append(str(job.part_h5ad))
    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing preprocess part output(s):\n{preview}{extra}")


def merge_preprocess_parts(cfg: Config, jobs_csv: Path) -> list[Path]:
    """Concatenate per-ROI cell-table parts into the final per-variant cells h5ad.

    The segmentation parquets are already written to their final location by the
    workers, so the merge only assembles ``cells/<exp>_cells{suffix}.h5ad`` and
    then deletes the transient per-ROI h5ad parts. Returns the h5ad paths.
    """
    import anndata as ad

    jobs = read_jobs_csv(jobs_csv)
    if not jobs:
        raise RuntimeError(f"No preprocess jobs found in {jobs_csv}")

    outputs: list[Path] = []
    grouped: dict[bool, list[PreprocessJob]] = {}
    for job in jobs:
        grouped.setdefault(job.bg, []).append(job)

    for bg, bg_jobs in grouped.items():
        label = _variant_label(bg)
        log.info("merging cell-table parts for [stage]%s[/] ([count]%d[/] ROI jobs)", label, len(bg_jobs))
        _validate_parts(bg_jobs)

        adatas = [ad.read_h5ad(job.part_h5ad) for job in bg_jobs]
        adata = ad.concat(adatas)
        adata.obs.index = pd.Index(range(len(adata.obs))).astype(str)
        _join_roi_metadata(cfg, adata)

        h5ad_path = cfg.h5ad_path(bg)
        ensure_dir(h5ad_path.parent)
        log.info("writing merged AnnData -> [path]%s[/]", h5ad_path)
        adata.write_h5ad(str(h5ad_path))

        # Segmentation parquets are already final; drop the transient h5ad parts.
        parts_dir = cfg.preprocess_parts_path(bg)
        shutil.rmtree(parts_dir, ignore_errors=True)
        log.info("removed transient cell-table parts -> [path]%s[/]", parts_dir)

        outputs.append(h5ad_path)

    return outputs


# --------------------------------------------------------------------------- #
#  SLURM plan/submit                                                          #
# --------------------------------------------------------------------------- #


def _worker_body_cmd(config_path: Path, jobs_csv: Path) -> str:
    python = Path(sys.executable)
    return (
        "task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}\n"
        f"uv run --frozen --no-sync {shlex.quote(str(python))} -m macsima_pipeline.cli "
        f"preprocess-worker --config {shlex.quote(str(config_path))} "
        f"--jobs-csv {shlex.quote(str(jobs_csv))} --task-id \"$task_id\""
    )


def _merge_body_cmd(config_path: Path, jobs_csv: Path) -> str:
    python = Path(sys.executable)
    return (
        f"uv run --frozen --no-sync {shlex.quote(str(python))} -m macsima_pipeline.cli "
        f"preprocess-merge --config {shlex.quote(str(config_path))} "
        f"--jobs-csv {shlex.quote(str(jobs_csv))}"
    )


def plan(cfg: Config, config_path: Path) -> PreprocessPlan:
    """Render worker-array and merge sbatch scripts for concrete mcmicro outputs."""
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    ensure_dir(cfg.paths.work_dir / cfg.paths.jobs_dir)

    jobs = discover_jobs(cfg)
    jobs_csv = write_jobs_csv(cfg, jobs)
    work_dir = cfg.paths.work_dir.resolve()
    config_path = config_path.resolve()
    jobs_csv = jobs_csv.resolve()

    worker_body = f"cd {shlex.quote(str(work_dir))}\n{_worker_body_cmd(config_path, jobs_csv)}"
    worker_content = render_sbatch(cfg, "preprocess", array_size=len(jobs), body_cmd=worker_body)
    worker_sbatch = write_sbatch(cfg, "preprocess", worker_content)

    merge_body = f"cd {shlex.quote(str(work_dir))}\n{_merge_body_cmd(config_path, jobs_csv)}"
    merge_content = render_sbatch(
        cfg,
        "preprocess_merge",
        array_size=None,
        body_cmd=merge_body,
        template_stage="preprocess",
        slurm_stage="preprocess_merge",
        output_stage="preprocess_merge",
    )
    merge_sbatch = write_sbatch(cfg, "preprocess_merge", merge_content)

    max_workers = cfg.preprocess.parallel.max_workers
    array_limit = max_workers if max_workers > 0 else None
    return PreprocessPlan(
        jobs_csv=jobs_csv,
        worker_sbatch=worker_sbatch,
        merge_sbatch=merge_sbatch,
        n_jobs=len(jobs),
        array_limit=array_limit,
    )


def run(cfg: Config, config_path: Path, *, do_submit: bool, dependency: str | None = None) -> int | None:
    try:
        plan_obj = plan(cfg, config_path)
    except PreprocessPlanningDeferred as e:
        log.warning("[warn]preprocess planning deferred[/]: %s", e)
        if not do_submit:
            return None
        raise

    log.info(
        "preprocess plan: [count]%d[/] ROI worker(s), csv=[path]%s[/] worker=[path]%s[/] merge=[path]%s[/]",
        plan_obj.n_jobs,
        plan_obj.jobs_csv,
        plan_obj.worker_sbatch,
        plan_obj.merge_sbatch,
    )
    if not do_submit:
        suffix = f"%{plan_obj.array_limit}" if plan_obj.array_limit else ""
        log.warning(
            "[warn](dry-run)[/] sbatch --array=1-[count]%d%s [path]%s[/]; merge [path]%s[/]",
            plan_obj.n_jobs,
            suffix,
            plan_obj.worker_sbatch,
            plan_obj.merge_sbatch,
        )
        return None

    worker_id = submit(
        plan_obj.worker_sbatch,
        array_size=plan_obj.n_jobs,
        array_limit=plan_obj.array_limit,
        dependency=dependency,
    )
    merge_id = submit(plan_obj.merge_sbatch, array_size=None, dependency=str(worker_id))
    log.info("[ok]preprocess submitted[/]: workers=[count]%d[/] merge=[count]%d[/]", worker_id, merge_id)
    return merge_id
