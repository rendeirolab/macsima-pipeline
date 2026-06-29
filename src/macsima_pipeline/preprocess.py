"""Stage 3: build SpatialData from mcmicro outputs, segment cells, export AnnData.

Port of `expr10_mcmicro_preprocessed.py`. Single GPU node; no SLURM array — one
process iterates all ROIs (sopa segmentation per ROI).
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path

import pandas as pd

from .config import Config
from .slurm import render_sbatch, submit, write_sbatch
from .utils import ensure_dir, roi_name_from_mcmicro_stem

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


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
        csv = first_sample / cfg.mcmicro.markers_bs_csv
        ci = pd.read_csv(csv)
        ci["channel_index"] = range(len(ci))
        ci = ci.drop_duplicates(subset="marker_name").reset_index(drop=True)
    else:
        csv = first_sample / cfg.mcmicro.markers_csv
        ci = pd.read_csv(csv)
        # Capture the TIFF position before filtering. DataFrame row numbers after
        # filtering are not physical channel indices.
        ci["channel_index"] = range(len(ci))
        ci = ci[ci["remove"] != True]  # noqa: E712 — pandas mask, not bool identity
        ci = ci.drop(columns=["remove"]).drop_duplicates(subset="marker_name").reset_index(drop=True)
    log.info("channel_info: [count]%d[/] markers from [path]%s[/]", len(ci), csv)
    return ci


def _segmentation_callable(cfg: Config):
    """Return a sopa-compatible callable that runs the configured segmentation model."""
    seg = cfg.preprocess.segmentation
    if seg.method != "cellpose4":
        raise NotImplementedError(f"Segmentation method {seg.method!r} not implemented yet")

    def _cellpose4(image):
        from cellpose import models

        m = models.CellposeModel(pretrained_model=seg.model, gpu=seg.gpu)
        masks, _, _ = m.eval(image)
        return masks

    return _cellpose4


# --------------------------------------------------------------------------- #
#  Pipeline entry                                                             #
# --------------------------------------------------------------------------- #


def _run_variant(cfg: Config, bg: bool) -> tuple[Path, Path]:
    """Run the full pipeline for one background-subtraction variant."""
    import anndata as ad
    import sopa
    from dask_image.imread import imread
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel

    label = "bg-sub" if bg else "no-bg-sub"
    log.info("[ok]variant[/]: [stage]%s[/] (suffix=[path]%s[/])", label, cfg.suffix_for(bg) or "(none)")

    images = _find_images(cfg, bg)
    log.info("found [count]%d[/] image(s)", len(images))
    channel_info = _load_channel_info(cfg, images, bg)

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

    sdata = SpatialData(images=imgs_data)
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

    zarr_path = cfg.zarr_path(bg)
    log.info("writing SpatialData -> [path]%s[/]", zarr_path)
    sdata.write(str(zarr_path))

    # Build per-cell AnnData
    cellexp_keys = [k for k in sdata.tables if "cell_expression" in k]
    adata = ad.concat([sdata[k] for k in cellexp_keys])
    adata.obs["ROI"] = adata.obs["slide"].map(lambda x: f"ROI{int(x.split('_')[0])}")
    adata.obs.reset_index(inplace=True)
    adata.obs.index = adata.obs.index.astype(str)
    drop_cols = [c for c in ("cell_id", "region", "index", "slide") if c in adata.obs.columns]
    adata.obs.drop(columns=drop_cols, inplace=True)

    # Optional ROI metadata join
    if cfg.experiment.roi_metadata_csv is not None:
        meta_path = cfg.paths.work_dir / cfg.experiment.roi_metadata_csv
        if meta_path.is_file():
            meta = pd.read_csv(meta_path)
            adata.obs = adata.obs.merge(meta, on="ROI", how="left").set_index(adata.obs.index)
            log.info("joined ROI metadata from [path]%s[/]", meta_path)
        else:
            log.warning("[bad]ROI metadata csv missing[/]: [path]%s[/]", meta_path)

    h5ad_path = cfg.h5ad_path(bg)
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
    log.info("preprocess variants to run: [count]%d[/] (%s)", len(modes),
             ", ".join("bg-sub" if m else "no-bg-sub" for m in modes))
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
#  SLURM plan/submit                                                          #
# --------------------------------------------------------------------------- #


def plan(cfg: Config, config_path: Path) -> Path:
    """Render an sbatch that calls back into the CLI to run this stage in-process."""
    ensure_dir(cfg.paths.work_dir / cfg.paths.logs_dir)
    python = Path(sys.executable)
    work_dir = cfg.paths.work_dir.resolve()
    body = (
        f"cd {shlex.quote(str(work_dir))}\n"
        f"uv run --frozen --no-sync "
        f"{shlex.quote(str(python))} -m macsima_pipeline.cli preprocess "
        f"--config {shlex.quote(str(config_path))} --inproc"
    )
    content = render_sbatch(cfg, "preprocess", array_size=None, body_cmd=body)
    return write_sbatch(cfg, "preprocess", content)


def run(cfg: Config, config_path: Path, *, do_submit: bool, dependency: str | None = None) -> int | None:
    sbatch = plan(cfg, config_path)
    log.info("preprocess plan: sbatch=[path]%s[/]", sbatch)
    if not do_submit:
        log.warning("[warn](dry-run)[/] sbatch [path]%s[/]", sbatch)
        return None
    return submit(sbatch, array_size=None, dependency=dependency)
