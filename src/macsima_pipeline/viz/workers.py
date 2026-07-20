"""joblib-parallel viz driver: marker grids, ROI grids, RGB combinations, and cell-map QC.

Reads channel_info from the first mcmicro sample (same convention as
`preprocess.py`), enumerates ROI files, and fans out figure rendering across
workers. Per-(roi, marker) percentiles are cached to a parquet so subsequent
runs skip the percentile recompute.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from ..config import Config
from . import cell_maps
from . import channel_qc
from . import render
from .render import RoiImage

log = logging.getLogger(__name__)


def _collect_images(cfg: Config, bg: bool) -> list[Path]:
    base = cfg.paths.work_dir / cfg.paths.mcmicro_out / cfg.experiment.name
    pattern = cfg.mcmicro.background_pattern if bg else cfg.mcmicro.registration_pattern
    images = sorted(base.rglob(pattern))
    if not images:
        raise FileNotFoundError(f"No images under {base} matching {pattern}")
    return images


def _load_channel_info(cfg: Config, first_sample: Path, bg: bool) -> pd.DataFrame:
    if bg:
        csv = first_sample / cfg.mcmicro.markers_bs_csv
        ci = pd.read_csv(csv)
        ci["channel_index"] = range(len(ci))
        ci = ci.drop_duplicates(subset="marker_name").reset_index(drop=True)
    else:
        csv = first_sample / cfg.mcmicro.markers_csv
        ci = pd.read_csv(csv)
        ci["channel_index"] = range(len(ci))
        ci = ci[ci["remove"] != True]  # noqa: E712
        ci = ci.drop(columns=["remove"]).drop_duplicates(subset="marker_name").reset_index(drop=True)
    log.info("loaded %d markers from %s", len(ci), csv)
    return ci


def _percentile_cache_path(cfg: Config, bg: bool) -> Path:
    return cfg.figures_dir() / "_cache" / f"percentiles{cfg.suffix_for(bg)}.parquet"


def _load_percentile_cache(cfg: Config, bg: bool) -> dict:
    if not cfg.viz.cache_percentiles:
        return {}
    path = _percentile_cache_path(cfg, bg)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {(r.roi, r.marker): (float(r.p1), float(r.p99)) for r in df.itertuples()}


def _save_percentile_cache(cfg: Config, cache: dict, bg: bool) -> None:
    if not cfg.viz.cache_percentiles or not cache:
        return
    path = _percentile_cache_path(cfg, bg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"roi": roi, "marker": m, "p1": p1, "p99": p99} for (roi, m), (p1, p99) in cache.items()]
    )
    df.to_parquet(path, index=False)
    log.info("wrote percentile cache (%d rows) -> %s", len(df), path)


def _run_variant(cfg: Config, bg: bool) -> None:
    """Render all marker grids + ROI grids + RGB combinations for one variant."""
    label = "bg-sub" if bg else "no-bg-sub"
    log.info("viz variant: %s (suffix=%s)", label, cfg.suffix_for(bg) or "(none)")

    images = _collect_images(cfg, bg)
    first_sample = images[0].parent.parent
    channel_info = _load_channel_info(cfg, first_sample, bg)

    log.info("resolving %d ROI(s) (pyramid level pick)...", len(images))
    rois: list[RoiImage] = [render.resolve_roi(p, cfg) for p in images]

    # ---- Pass 0: quantitative per-ROI/channel staining QC -----------------
    channel_qc.run_variant_qc(cfg, rois, channel_info, first_sample, bg)

    figdir = cfg.figures_dir() / "rois"
    figdir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.suffix_for(bg)
    fmt = cfg.viz.output_format

    cache = _load_percentile_cache(cfg, bg)

    # ---- Pass A: one marker across all ROIs (n_markers figures) -----------
    def _marker_job(ix: int, marker_name: str):
        out = figdir / f"{cfg.experiment.name}_mcmicro_marker_{marker_name}_intensity{suffix}.{fmt}"
        if render.is_valid_output(out, fmt):
            log.info("skipping completed output: %s", out)
            return {}
        local_cache: dict = {}
        render.plot_marker_across_rois(rois, ix, marker_name, cfg, out, percentile_cache=local_cache)
        return local_cache

    log.info("rendering marker grids (%d markers, %d workers)", len(channel_info), cfg.viz.parallel.workers)
    marker_caches = Parallel(
        n_jobs=cfg.viz.parallel.workers, backend=cfg.viz.parallel.backend, verbose=10
    )(
        delayed(_marker_job)(row.channel_index, row.marker_name)
        for row in channel_info.itertuples()
    )
    for c in marker_caches:
        cache.update(c)

    # ---- Pass B: one ROI across all markers (n_rois figures) --------------
    def _roi_job(roi: RoiImage):
        out = figdir / f"{cfg.experiment.name}_mcmicro_all_markers_ROI_{roi.name}{suffix}.{fmt}"
        if render.is_valid_output(out, fmt):
            log.info("skipping completed output: %s", out)
            return {}
        local_cache: dict = dict(cache)  # seed with what we already know
        render.plot_all_markers_for_roi(roi, channel_info, cfg, out, percentile_cache=local_cache)
        # Return only the newly-added keys
        return {k: v for k, v in local_cache.items() if k not in cache}

    roi_workers = cfg.viz.parallel.roi_workers or cfg.viz.parallel.workers
    log.info(
        "rendering ROI grids (%d ROIs, %d workers)", len(rois), roi_workers
    )
    # Pass B holds n_markers panels in a single figure (~1 GB per worker for
    # 59 markers @ 2048x2048 once matplotlib's AGG buffer is added). Cap
    # concurrency and recycle workers after every job so memory is released
    # back to the OS instead of being held by the Python allocator.
    roi_caches = Parallel(
        n_jobs=roi_workers,
        backend=cfg.viz.parallel.backend,
        verbose=10,
        max_nbytes=None,
    )(delayed(_roi_job)(r) for r in rois)
    for c in roi_caches:
        cache.update(c)

    # ---- Pass C: RGB combinations -----------------------------------------
    if cfg.viz.combinations:
        marker_to_ix = {
            row.marker_name: row.channel_index for row in channel_info.itertuples()
        }
        for comb in cfg.viz.combinations:
            try:
                ixs = [marker_to_ix[m] for m in comb.markers]
            except KeyError as e:
                log.warning("skipping combination %s: missing marker %s", comb.name, e)
                continue
            out = figdir / f"{cfg.experiment.name}_mcmicro_{comb.name}_rgb{suffix}.{fmt}"
            if render.is_valid_output(out, fmt):
                log.info("skipping completed output: %s", out)
                continue
            render.plot_rgb_combination(rois, ixs, comb.markers, comb.name, cfg, out)

    # ---- Pass D: multi-page cell-location QC summary ----------------------
    if cfg.viz.cell_maps:
        # Prefer the phenotyped h5ad (colors maps by cell type + adds a coherence
        # page); fall back to the raw preprocess output for backward compatibility.
        phenotyped = cfg.phenotype_h5ad_path(bg)
        h5ad_path = phenotyped if phenotyped.is_file() else cfg.h5ad_path(bg)
        qc_dir = cfg.figures_dir() / "qc"
        out = qc_dir / f"{cfg.experiment.name}_mcmicro_cell_maps_summary{suffix}.pdf"
        if render.is_valid_output(out, "pdf"):
            log.info("skipping completed output: %s", out)
        elif not h5ad_path.is_file():
            log.warning("skipping cell-map QC summary; AnnData missing: %s", h5ad_path)
        else:
            import anndata as ad

            log.info("rendering cell-map QC summary from %s", h5ad_path)
            adata = ad.read_h5ad(h5ad_path)
            cell_maps.plot_cell_map_qc_summary(adata, cfg, out, bg=bg, h5ad_path=h5ad_path)

    _save_percentile_cache(cfg, cache, bg)
    log.info("viz variant done: %s", label)


def run_inproc(cfg: Config) -> None:
    """Render figures for every configured background-subtraction variant."""
    modes = cfg.bg_modes()
    log.info(
        "viz variants to run: %d (%s)",
        len(modes),
        ", ".join("bg-sub" if m else "no-bg-sub" for m in modes),
    )
    ran = 0
    for bg in modes:
        try:
            _run_variant(cfg, bg)
            ran += 1
        except FileNotFoundError as e:
            if isinstance(cfg.mcmicro.background_subtraction, bool):
                raise
            log.warning("skipping viz variant bg=%s: %s", bg, e)
    if ran == 0:
        raise FileNotFoundError("No viz variants produced output; check mcmicro outputs")
    channel_qc.write_bg_comparison(cfg)
    log.info("viz done (%d variant(s))", ran)
