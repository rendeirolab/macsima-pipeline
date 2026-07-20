"""Stage 3: build SpatialData from mcmicro outputs, segment cells, export AnnData.

The local/debug path still supports one in-process run over all ROIs. The SLURM
path fans out one GPU array task per (background variant, ROI image), then runs
a CPU merge job to assemble the final experiment-level SpatialData and AnnData.
"""

from __future__ import annotations

import csv
import logging
import shlex
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .slurm import render_sbatch, submit, write_sbatch
from .utils import ensure_dir, roi_name_from_mcmicro_stem

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessJob:
    job_id: int
    bg: bool
    variant: str
    roi_name: str
    image_path: Path
    part_zarr: Path
    part_h5ad: Path


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
    base = cfg.paths.work_dir / cfg.paths.mcmicro_out / cfg.experiment.name
    pattern = cfg.mcmicro.background_pattern if bg else cfg.mcmicro.registration_pattern
    images = sorted(base.rglob(pattern))
    if not images:
        raise FileNotFoundError(f"No images found under {base} matching {pattern}")
    return images


def _load_channel_info(cfg: Config, images: list[Path], bg: bool) -> pd.DataFrame:
    """Resolve markers CSV from the first mcmicro sample dir."""
    first_sample = images[0].parent.parent  # .../sample/registration/img.ome.tif -> sample
    if bg:
        csv_path = first_sample / cfg.mcmicro.markers_bs_csv
        ci = pd.read_csv(csv_path)
        ci["channel_index"] = range(len(ci))
        ci = ci.drop_duplicates(subset="marker_name").reset_index(drop=True)
    else:
        csv_path = first_sample / cfg.mcmicro.markers_csv
        ci = pd.read_csv(csv_path)
        # Capture the TIFF position before filtering. DataFrame row numbers after
        # filtering are not physical channel indices.
        ci["channel_index"] = range(len(ci))
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
        roi_name = roi_name_from_mcmicro_stem(img_path.stem)
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
    drop_cols = [c for c in ("cell_id", "region", "index", "slide") if c in adata.obs.columns]
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


def _run_variant(cfg: Config, bg: bool) -> tuple[Path, Path]:
    """Run the full pipeline for one background-subtraction variant."""
    label = _variant_label(bg)
    log.info("[ok]variant[/]: [stage]%s[/] (suffix=[path]%s[/])", label, cfg.suffix_for(bg) or "(none)")

    images = _find_images(cfg, bg)
    log.info("found [count]%d[/] image(s)", len(images))
    channel_info = _load_channel_info(cfg, images, bg)
    roi_labels = {
        roi_name_from_mcmicro_stem(img_path.stem): _roi_label_from_mcmicro_name(
            roi_name_from_mcmicro_stem(img_path.stem)
        )
        for img_path in images
    }

    sdata = _build_spatialdata_for_images(cfg, images, channel_info)
    _segment_spatialdata(cfg, sdata)

    zarr_path = cfg.zarr_path(bg)
    ensure_dir(zarr_path.parent)
    log.info("writing SpatialData -> [path]%s[/]", zarr_path)
    _write_sdata_atomic(sdata, zarr_path)

    adata = _adata_from_sdata_tables(sdata, roi_labels)
    _join_roi_metadata(cfg, adata)

    h5ad_path = cfg.h5ad_path(bg)
    ensure_dir(h5ad_path.parent)
    log.info("writing AnnData -> [path]%s[/]", h5ad_path)
    adata.write_h5ad(str(h5ad_path))
    return zarr_path, h5ad_path


def run_inproc(cfg: Config) -> list[tuple[Path, Path]]:
    """Execute the preprocessing pipeline for every configured bg-sub variant.

    Returns a list of (zarr_path, h5ad_path) tuples — one per variant. In
    "auto" mode this can be two entries (bg-sub + no-bg-sub) when both image
    sets exist; in explicit bool mode it's always a single entry.
    """
    modes = cfg.bg_modes()
    log.info(
        "preprocess variants to run: [count]%d[/] (%s)",
        len(modes),
        ", ".join(_variant_label(m) for m in modes),
    )
    results: list[tuple[Path, Path]] = []
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


def _part_paths(cfg: Config, bg: bool, roi_name: str) -> tuple[Path, Path]:
    base = cfg.preprocess_parts_path(bg) / roi_name
    return base / f"{roi_name}.zarr", base / f"{roi_name}_cell_expression.h5ad"


def discover_jobs(cfg: Config) -> list[PreprocessJob]:
    """Discover concrete preprocessing work items from completed mcmicro outputs."""
    jobs: list[PreprocessJob] = []
    missing: list[str] = []
    for bg in cfg.bg_modes():
        try:
            images = _find_images(cfg, bg)
        except FileNotFoundError as e:
            missing.append(f"{_variant_label(bg)}: {e}")
            continue

        for img_path in images:
            roi_name = roi_name_from_mcmicro_stem(img_path.stem)
            part_zarr, part_h5ad = _part_paths(cfg, bg, roi_name)
            jobs.append(
                PreprocessJob(
                    job_id=len(jobs) + 1,
                    bg=bg,
                    variant=_variant_label(bg),
                    roi_name=roi_name,
                    image_path=img_path,
                    part_zarr=part_zarr,
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
            fieldnames=["job_id", "bg", "variant", "roi_name", "image_path", "part_zarr", "part_h5ad"],
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
                    "part_zarr": str(job.part_zarr),
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
                part_zarr=Path(row["part_zarr"]),
                part_h5ad=Path(row["part_h5ad"]),
            )
        )
    return jobs


def read_job_for_task(jobs_csv: Path, task_id: int) -> PreprocessJob:
    for job in read_jobs_csv(jobs_csv):
        if job.job_id == task_id:
            return job
    raise RuntimeError(f"No row for task_id={task_id} in {jobs_csv}")


def _write_sdata_atomic(sdata, dest: Path) -> None:
    """Write a SpatialData Zarr store to ``dest``, safely replacing any existing store.

    spatialdata>=0.7 refuses to overwrite a Zarr store when the object has
    Dask-backed elements under the target path (scverse/spatialdata#520), so
    ``overwrite=True`` is not reliable. Write to a sibling temp store first, then
    swap it in: delete the old store and rename temp -> dest. The temp path
    shares ``dest``'s parent to keep the rename on a single filesystem.
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    tmp = dest.parent / (dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    sdata.write(str(tmp))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def process_job(cfg: Config, job: PreprocessJob) -> tuple[Path, Path]:
    """Run one ROI/variant worker and write intermediate part outputs."""
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

    ensure_dir(job.part_zarr.parent)
    log.info("writing ROI SpatialData part -> [path]%s[/]", job.part_zarr)
    _write_sdata_atomic(sdata, job.part_zarr)

    roi_labels = {job.roi_name: _roi_label_from_mcmicro_name(job.roi_name)}
    adata = _adata_from_sdata_tables(sdata, roi_labels)
    ensure_dir(job.part_h5ad.parent)
    log.info("writing ROI AnnData part -> [path]%s[/]", job.part_h5ad)
    adata.write_h5ad(str(job.part_h5ad))
    return job.part_zarr, job.part_h5ad


def run_worker(cfg: Config, jobs_csv: Path, task_id: int) -> tuple[Path, Path]:
    return process_job(cfg, read_job_for_task(jobs_csv, task_id))


def _merge_spatialdata_parts(part_paths: list[Path]):
    from spatialdata import SpatialData, read_zarr

    merged: dict[str, dict] = {
        "images": {},
        "labels": {},
        "points": {},
        "shapes": {},
        "tables": {},
    }
    for part_path in part_paths:
        sdata = read_zarr(str(part_path))
        for attr, dest in merged.items():
            src = getattr(sdata, attr, None)
            if src:
                dest.update(dict(src))
    return SpatialData(**{k: v for k, v in merged.items() if v})


def _validate_parts(jobs: list[PreprocessJob]) -> None:
    missing = []
    for job in jobs:
        if not job.part_zarr.exists():
            missing.append(str(job.part_zarr))
        if not job.part_h5ad.is_file():
            missing.append(str(job.part_h5ad))
    if missing:
        preview = "\n".join(f"  - {p}" for p in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing preprocess part output(s):\n{preview}{extra}")


def merge_preprocess_parts(cfg: Config, jobs_csv: Path) -> list[tuple[Path, Path]]:
    """Merge per-ROI worker outputs into final per-variant zarr/h5ad files."""
    import anndata as ad

    jobs = read_jobs_csv(jobs_csv)
    if not jobs:
        raise RuntimeError(f"No preprocess jobs found in {jobs_csv}")

    outputs: list[tuple[Path, Path]] = []
    grouped: dict[bool, list[PreprocessJob]] = {}
    for job in jobs:
        grouped.setdefault(job.bg, []).append(job)

    for bg, bg_jobs in grouped.items():
        label = _variant_label(bg)
        log.info("merging preprocess parts for [stage]%s[/] ([count]%d[/] ROI jobs)", label, len(bg_jobs))
        _validate_parts(bg_jobs)

        adatas = [ad.read_h5ad(job.part_h5ad) for job in bg_jobs]
        adata = ad.concat(adatas)
        adata.obs.index = pd.Index(range(len(adata.obs))).astype(str)
        _join_roi_metadata(cfg, adata)

        h5ad_path = cfg.h5ad_path(bg)
        ensure_dir(h5ad_path.parent)
        log.info("writing merged AnnData -> [path]%s[/]", h5ad_path)
        adata.write_h5ad(str(h5ad_path))

        sdata = _merge_spatialdata_parts([job.part_zarr for job in bg_jobs])
        zarr_path = cfg.zarr_path(bg)
        ensure_dir(zarr_path.parent)
        log.info("writing merged SpatialData -> [path]%s[/]", zarr_path)
        _write_sdata_atomic(sdata, zarr_path)

        outputs.append((zarr_path, h5ad_path))

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
