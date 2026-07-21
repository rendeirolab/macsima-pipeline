"""Matplotlib renderers for marker grids, ROI grids, and RGB combinations.

PDF @ 300 dpi by default; rasterized=True keeps panels as single embedded images
so PDFs stay small and fast to save.
"""

from __future__ import annotations

import gc
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from .io import PyramidLevel, clip_norm_u8, percentiles, pick_level, read_channel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoiImage:
    """Resolved per-ROI metadata: file path + chosen pyramid level."""

    name: str
    path: Path
    level: PyramidLevel


def _setup_mpl(cfg: Config) -> None:
    matplotlib.rcParams["pdf.compression"] = cfg.viz.pdf_compression
    matplotlib.rcParams["savefig.bbox"] = "tight"


def resolve_roi(path: Path, cfg: Config) -> RoiImage:
    from ..utils import roi_name_from_results_stem

    lvl = pick_level(path, cfg.viz.target_max_dim)
    return RoiImage(name=roi_name_from_results_stem(path.stem), path=path, level=lvl)


def _grid_dims(n: int, ncols: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("Cannot create a grid with no panels")
    if ncols <= 0:
        raise ValueError("grid_ncols must be greater than zero")
    ncols = min(n, ncols)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _new_grid(n_panels: int, cfg: Config):
    nrows, ncols = _grid_dims(n_panels, cfg.viz.grid_ncols)
    fw, fh = cfg.viz.fig_size_per_panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * fw, nrows * fh))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)
    return fig, axes


def _trim_axes(axes, n_panels: int) -> None:
    flat = axes.flatten()
    for i in range(n_panels, len(flat)):
        flat[i].remove()


def _layout_grid(fig, cfg: Config) -> None:
    """Apply compact, deterministic spacing while reserving one figure-title line."""
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.94,
        wspace=cfg.viz.grid_wspace,
        hspace=cfg.viz.grid_hspace,
    )


def is_valid_output(path: Path, output_format: str | None = None) -> bool:
    """Cheap integrity check used to make visualization reruns resumable."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    fmt = (output_format or path.suffix.lstrip(".")).lower()
    try:
        with path.open("rb") as fh:
            if fmt == "pdf":
                if fh.read(5) != b"%PDF-":
                    return False
                fh.seek(max(0, path.stat().st_size - 1024))
                return b"%%EOF" in fh.read()
            if fmt == "png":
                return fh.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False
    return True


def _save_figure_atomic(fig, out_path: Path, cfg: Config) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp{out_path.suffix}")
    try:
        fig.savefig(
            str(tmp),
            dpi=cfg.viz.dpi,
            format=cfg.viz.output_format,
            bbox_inches="tight",
            pad_inches=cfg.viz.output_pad_inches,
        )
        if not is_valid_output(tmp, cfg.viz.output_format):
            raise OSError(f"matplotlib produced an invalid {cfg.viz.output_format} file: {tmp}")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)


def plot_marker_across_rois(
    rois: list[RoiImage],
    marker_ix: int,
    marker_name: str,
    cfg: Config,
    out_path: Path,
    *,
    percentile_cache: dict | None = None,
) -> Path:
    """One panel per ROI for one marker."""
    _setup_mpl(cfg)
    fig = None
    img = display = None
    try:
        fig, axes = _new_grid(len(rois), cfg)
        flat = axes.flatten()
        p_lo, p_hi = cfg.viz.percentile_clip
        for i, roi in enumerate(rois):
            ax = flat[i]
            img = read_channel(roi.path, roi.level.level, marker_ix)
            ckey = (roi.name, marker_name)
            if percentile_cache is not None and ckey in percentile_cache:
                p1, p99 = percentile_cache[ckey]
            else:
                p1, p99 = percentiles(img, p_lo, p_hi)
                if percentile_cache is not None:
                    percentile_cache[ckey] = (p1, p99)
            display = clip_norm_u8(img, p1, p99)
            ax.imshow(display, cmap=cfg.viz.cmap, vmin=0, vmax=255,
                      interpolation="nearest", rasterized=cfg.viz.rasterized)
            ax.set_title(f"ROI {roi.name} | [{p1:.1f}, {p99:.1f}]",
                         fontsize=cfg.viz.panel_title_size)
            ax.axis("off")
            img = display = None
        _trim_axes(axes, len(rois))
        fig.suptitle(f"Marker: {marker_name}", fontsize=cfg.viz.figure_title_size)
        _layout_grid(fig, cfg)
        _save_figure_atomic(fig, out_path, cfg)
    finally:
        del img, display
        if fig is not None:
            plt.close(fig)
        gc.collect()
    return out_path


def plot_all_markers_for_roi(
    roi: RoiImage,
    channel_info,
    cfg: Config,
    out_path: Path,
    *,
    percentile_cache: dict | None = None,
) -> Path:
    """One panel per marker for one ROI."""
    _setup_mpl(cfg)
    n_markers = len(channel_info)
    fig = None
    img = display = None
    try:
        fig, axes = _new_grid(n_markers, cfg)
        flat = axes.flatten()
        p_lo, p_hi = cfg.viz.percentile_clip

        for i, row in enumerate(channel_info.itertuples()):
            ax = flat[i]
            marker_name = row.marker_name
            marker_ix = getattr(row, "channel_index", i)
            img = read_channel(roi.path, roi.level.level, marker_ix)
            ckey = (roi.name, marker_name)
            if percentile_cache is not None and ckey in percentile_cache:
                p1, p99 = percentile_cache[ckey]
            else:
                p1, p99 = percentiles(img, p_lo, p_hi)
                if percentile_cache is not None:
                    percentile_cache[ckey] = (p1, p99)
            display = clip_norm_u8(img, p1, p99)
            ax.imshow(display, cmap=cfg.viz.cmap, vmin=0, vmax=255,
                      interpolation="nearest", rasterized=cfg.viz.rasterized)
            ax.set_title(f"{marker_name} | [{p1:.1f}, {p99:.1f}]",
                         fontsize=cfg.viz.panel_title_size)
            ax.axis("off")
            img = display = None

        _trim_axes(axes, n_markers)
        fig.suptitle(f"ROI: {roi.name}", fontsize=cfg.viz.figure_title_size)
        _layout_grid(fig, cfg)
        _save_figure_atomic(fig, out_path, cfg)
    finally:
        del img, display
        if fig is not None:
            plt.close(fig)
        gc.collect()
    return out_path


def plot_rgb_combination(
    rois: list[RoiImage],
    marker_indices: list[int],
    marker_names: list[str],
    comb_name: str,
    cfg: Config,
    out_path: Path,
) -> Path:
    """One panel per ROI, mapping three markers to RGB."""
    if len(marker_indices) != 3:
        raise ValueError(f"RGB combination needs exactly 3 markers, got {len(marker_indices)}")
    _setup_mpl(cfg)
    fig = None
    ch = rgb = chans = None
    try:
        fig, axes = _new_grid(len(rois), cfg)
        flat = axes.flatten()
        p_lo, p_hi = cfg.viz.percentile_clip
        for i, roi in enumerate(rois):
            ax = flat[i]
            chans = []
            for mix in marker_indices:
                ch = read_channel(roi.path, roi.level.level, mix)
                p1, p99 = percentiles(ch, p_lo, p_hi)
                chans.append(clip_norm_u8(ch, p1, p99))
                ch = None
            rgb = np.stack(chans, axis=-1)
            chans = None
            ax.imshow(rgb, interpolation="nearest", rasterized=cfg.viz.rasterized)
            ax.set_title(f"ROI {roi.name}", fontsize=cfg.viz.panel_title_size)
            ax.axis("off")
            rgb = None
        _trim_axes(axes, len(rois))
        fig.suptitle(
            f"Combination {comb_name}: {' / '.join(marker_names)}",
            fontsize=cfg.viz.figure_title_size,
        )
        _layout_grid(fig, cfg)
        _save_figure_atomic(fig, out_path, cfg)
    finally:
        del ch, rgb, chans
        if fig is not None:
            plt.close(fig)
        gc.collect()
    return out_path
