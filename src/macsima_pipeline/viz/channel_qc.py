"""Quantitative per-ROI/channel staining QC for the viz stage."""

from __future__ import annotations

import gc
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages

from ..config import Config
from . import render
from .render import RoiImage

log = logging.getLogger(__name__)

EPS = 1e-6
QC_REQUIRED_COLUMNS = {
    "roi",
    "marker_name",
    "channel_index",
    "median",
    "p99",
    "snr_p99",
    "positive_fraction",
    "flag_count",
}


def qc_csv_path(cfg: Config, bg: bool) -> Path:
    return cfg.figures_dir() / "qc" / f"{cfg.experiment.name}_mcmicro_channel_qc{cfg.suffix_for(bg)}.csv"


def qc_pdf_path(cfg: Config, bg: bool) -> Path:
    return cfg.figures_dir() / "qc" / f"{cfg.experiment.name}_mcmicro_channel_qc_summary{cfg.suffix_for(bg)}.pdf"


def comparison_csv_path(cfg: Config) -> Path:
    return cfg.figures_dir() / "qc" / f"{cfg.experiment.name}_mcmicro_channel_qc_bg_comparison.csv"


def comparison_pdf_path(cfg: Config) -> Path:
    return cfg.figures_dir() / "qc" / f"{cfg.experiment.name}_mcmicro_channel_qc_bg_comparison.pdf"


def is_valid_qc_csv(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        columns = set(pd.read_csv(path, nrows=1).columns)
    except Exception:
        return False
    return QC_REQUIRED_COLUMNS.issubset(columns)


def _series_channel(series, level: int, channel_ix: int) -> np.ndarray:
    if hasattr(series, "levels"):
        arr = series.levels[level].asarray(key=channel_ix)
    else:
        arr = series.asarray(key=channel_ix)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _finite_values(img: np.ndarray) -> np.ndarray:
    values = np.asarray(img).reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    return values.astype(np.float64, copy=False)


def _robust_baseline(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), EPS
    baseline = float(np.nanmedian(values))
    noise = float(np.nanmedian(np.abs(values - baseline)) * 1.4826)
    if not math.isfinite(noise) or noise <= EPS:
        noise = float(np.nanstd(values))
    if not math.isfinite(noise) or noise <= EPS:
        noise = EPS
    return baseline, noise


def _saturated_fraction(img: np.ndarray) -> float:
    arr = np.asarray(img)
    if not np.issubdtype(arr.dtype, np.integer) or arr.size == 0:
        return 0.0
    return float(np.mean(arr == np.iinfo(arr.dtype).max))


def _tile_medians(img: np.ndarray, tile_grid: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.empty((0, 0), dtype=float)
    n_rows = max(1, min(int(tile_grid[0]), arr.shape[0]))
    n_cols = max(1, min(int(tile_grid[1]), arr.shape[1]))
    row_ixs = np.array_split(np.arange(arr.shape[0]), n_rows)
    col_ixs = np.array_split(np.arange(arr.shape[1]), n_cols)
    out = np.empty((n_rows, n_cols), dtype=float)
    for r, rows in enumerate(row_ixs):
        for c, cols in enumerate(col_ixs):
            tile = arr[np.ix_(rows, cols)]
            out[r, c] = float(np.nanmedian(tile)) if tile.size else float("nan")
    return out


def _spatial_metrics(img: np.ndarray, tile_grid: tuple[int, int]) -> dict[str, float]:
    medians = _tile_medians(img, tile_grid)
    values = medians[np.isfinite(medians)]
    if values.size == 0:
        return {"tile_median_cv": float("nan"), "edge_center_ratio": float("nan")}
    mean = float(np.nanmean(values))
    cv = float(np.nanstd(values) / abs(mean)) if abs(mean) > EPS else 0.0

    n_rows, n_cols = medians.shape
    edge_mask = np.zeros_like(medians, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    center_mask = ~edge_mask
    if not bool(center_mask.any()):
        center_mask = np.ones_like(edge_mask, dtype=bool)
    edge = float(np.nanmedian(medians[edge_mask]))
    center = float(np.nanmedian(medians[center_mask]))
    if abs(center) <= EPS:
        ratio = 1.0 if abs(edge) <= EPS else float("nan")
    else:
        ratio = edge / center
    return {"tile_median_cv": cv, "edge_center_ratio": float(ratio)}


def compute_channel_metrics(img: np.ndarray, qc_cfg: Any, background_img: np.ndarray | None = None) -> dict[str, float]:
    """Compute scalar intensity, sensitivity, and coarse spatial QC metrics."""
    values = _finite_values(img)
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p1": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "p99_9": float("nan"),
            "dynamic_range": float("nan"),
            "zero_fraction": float("nan"),
            "saturated_fraction": float("nan"),
            "background_median": float("nan"),
            "noise_mad": float("nan"),
            "snr_p99": float("nan"),
            "positive_fraction": float("nan"),
            "tile_median_cv": float("nan"),
            "edge_center_ratio": float("nan"),
        }

    p1, p5, median, p95, p99, p99_9 = [float(v) for v in np.percentile(values, [1, 5, 50, 95, 99, 99.9])]
    if background_img is not None:
        background_values = _finite_values(background_img)
        background_median, noise = _robust_baseline(background_values)
    else:
        lower = values[values <= median]
        background_median, noise = _robust_baseline(lower)

    threshold = background_median + float(qc_cfg.sigma) * noise
    spatial = _spatial_metrics(img, qc_cfg.tile_grid)
    return {
        "mean": float(np.nanmean(values)),
        "median": median,
        "p1": p1,
        "p5": p5,
        "p95": p95,
        "p99": p99,
        "p99_9": p99_9,
        "dynamic_range": p99 - p1,
        "zero_fraction": float(np.mean(values == 0)),
        "saturated_fraction": _saturated_fraction(img),
        "background_median": background_median,
        "noise_mad": noise,
        "snr_p99": float((p99 - background_median) / noise) if noise >= EPS else float("nan"),
        "positive_fraction": float(np.mean(values > threshold)),
        **spatial,
    }


def _as_clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _raw_marker_table(cfg: Config, first_sample: Path) -> pd.DataFrame:
    csv = first_sample / cfg.mcmicro.markers_csv
    if not csv.is_file():
        return pd.DataFrame()
    raw = pd.read_csv(csv)
    raw["channel_index"] = range(len(raw))
    return raw


def _background_index_by_name(raw_markers: pd.DataFrame) -> dict[str, int]:
    if raw_markers.empty or "marker_name" not in raw_markers.columns:
        return {}
    out: dict[str, int] = {}
    for row in raw_markers.itertuples(index=False):
        marker = _as_clean_text(getattr(row, "marker_name", None))
        ix = getattr(row, "channel_index", None)
        if marker is not None and ix is not None and marker not in out:
            out[marker] = int(ix)
    return out


def _qc_roi(
    roi: RoiImage,
    channel_rows: list[dict[str, Any]],
    bg_index_by_name: dict[str, int],
    cfg: Config,
    bg: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    background_cache: dict[int, np.ndarray] = {}
    with tifffile.TiffFile(str(roi.path)) as tf:
        series = tf.series[0]
        for i, marker_row in enumerate(channel_rows):
            marker = str(marker_row.get("marker_name", f"channel_{i}"))
            channel_ix = int(marker_row.get("channel_index", i))
            img = _series_channel(series, roi.level.level, channel_ix)

            background_name = _as_clean_text(_field(marker_row, "background", "background_marker"))
            background_ix = None if bg or background_name is None else bg_index_by_name.get(background_name)
            background_img = None
            if background_ix is not None:
                if background_ix not in background_cache:
                    background_cache[background_ix] = _series_channel(series, roi.level.level, background_ix)
                background_img = background_cache[background_ix]

            metrics = compute_channel_metrics(img, cfg.viz.channel_qc, background_img=background_img)
            exposure = _float_or_nan(_field(marker_row, "exposure", "Exposure"))
            p99 = metrics["p99"]
            p99_per_exposure = p99 / exposure if exposure and exposure > 0 else float("nan")
            rows.append(
                {
                    "experiment": cfg.experiment.name,
                    "variant": "bg-sub" if bg else "no-bg-sub",
                    "bg_subtracted": bool(bg),
                    "roi": roi.name,
                    "marker_name": marker,
                    "channel_index": channel_ix,
                    "channel_number": _field(marker_row, "channel_number", "Channel", "channel"),
                    "cycle_number": _field(marker_row, "cycle_number", "cycle"),
                    "filter": _field(marker_row, "Filter", "filter"),
                    "exposure": exposure,
                    "background_marker": background_name,
                    "background_channel_index": background_ix,
                    "p99_per_exposure": p99_per_exposure,
                    "pyramid_level": roi.level.level,
                    "image_path": str(roi.path),
                    **metrics,
                }
            )
            del img, background_img
    return pd.DataFrame(rows)


def _robust_z_by_marker(df: pd.DataFrame, value_col: str) -> pd.Series:
    out = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for _, idx in df.groupby("marker_name", sort=False).groups.items():
        values = pd.to_numeric(df.loc[idx, value_col], errors="coerce").astype(float)
        med = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - med)) * 1.4826)
        if not math.isfinite(mad) or mad <= EPS:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (values - med) / mad
    return out


def _flag_string(row) -> str:
    flags = []
    for name in ("low_intensity", "low_sensitivity", "saturated", "uneven"):
        if bool(getattr(row, name)):
            flags.append(name)
    return ";".join(flags)


def _apply_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if df.empty:
        return df
    qc = cfg.viz.channel_qc
    out = df.copy()
    out["log1p_p99"] = np.log1p(np.clip(pd.to_numeric(out["p99"], errors="coerce"), 0, None))
    out["intensity_z"] = _robust_z_by_marker(out, "log1p_p99")
    out["tile_cv_z"] = _robust_z_by_marker(out, "tile_median_cv")
    edge_ratio = pd.to_numeric(out["edge_center_ratio"], errors="coerce")
    out["edge_center_log2"] = np.log2(np.clip(edge_ratio, EPS, None))
    out["edge_center_z"] = _robust_z_by_marker(out, "edge_center_log2")

    out["low_intensity"] = (out["intensity_z"] < -qc.outlier_z) | (
        pd.to_numeric(out["dynamic_range"], errors="coerce") <= 0
    )
    out["low_sensitivity"] = (pd.to_numeric(out["snr_p99"], errors="coerce") < qc.min_snr) | (
        pd.to_numeric(out["positive_fraction"], errors="coerce") < qc.min_positive_fraction
    )
    out["saturated"] = pd.to_numeric(out["saturated_fraction"], errors="coerce") > qc.max_saturated_fraction
    out["uneven"] = (
        (pd.to_numeric(out["tile_median_cv"], errors="coerce") > 1.0)
        | (out["tile_cv_z"] > qc.outlier_z)
        | (edge_ratio > 2.5)
        | (edge_ratio < 0.4)
        | (out["edge_center_z"].abs() > qc.outlier_z)
    )
    flag_cols = ["low_intensity", "low_sensitivity", "saturated", "uneven"]
    out["flag_count"] = out[flag_cols].sum(axis=1).astype(int)
    out["flags"] = [_flag_string(row) for row in out.itertuples()]
    return out


def _sort_qc(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ("roi", "cycle_number", "channel_index", "marker_name") if c in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _compute_qc_table(
    cfg: Config,
    rois: list[RoiImage],
    channel_info: pd.DataFrame,
    first_sample: Path,
    bg: bool,
) -> pd.DataFrame:
    channel_rows = channel_info.to_dict("records")
    raw_markers = pd.DataFrame() if bg else _raw_marker_table(cfg, first_sample)
    bg_index_by_name = _background_index_by_name(raw_markers)
    workers = cfg.viz.channel_qc.workers or cfg.viz.parallel.workers
    workers = max(1, min(int(workers), len(rois)))
    log.info("computing channel QC (%d ROIs, %d markers, %d workers)", len(rois), len(channel_rows), workers)
    frames = Parallel(n_jobs=workers, backend=cfg.viz.parallel.backend, verbose=10, max_nbytes=None)(
        delayed(_qc_roi)(roi, channel_rows, bg_index_by_name, cfg, bg) for roi in rois
    )
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _sort_qc(_apply_flags(df, cfg))


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("wrote channel QC table (%d rows) -> %s", len(df), out_path)


def _marker_order(df: pd.DataFrame) -> list[str]:
    if "channel_index" in df.columns:
        ordered = df.sort_values("channel_index", kind="stable")
    else:
        ordered = df
    return list(dict.fromkeys(str(v) for v in ordered["marker_name"]))


def _roi_order(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(str(v) for v in df["roi"]))


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="roi", columns="marker_name", values=value, aggfunc="mean").reindex(
        index=_roi_order(df), columns=_marker_order(df)
    )


def _plot_heatmap(ax, data: pd.DataFrame, title: str, *, cmap: str = "viridis") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    values = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)
    if len(data.index) <= 40:
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(data.index)} ROIs")
    if len(data.columns) <= 30:
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=90, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"{len(data.columns)} markers")
    ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)


def _plot_flag_table(ax, df: pd.DataFrame, title: str, cfg: Config) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    flagged = df[df["flag_count"] > 0].sort_values(
        ["flag_count", "marker_name", "roi"], ascending=[False, True, True], kind="stable"
    )
    if flagged.empty:
        ax.text(0.0, 0.9, "No warning flags.", va="top", transform=ax.transAxes)
        return
    top = flagged.head(cfg.viz.channel_qc.report_top_n).copy()
    top["snr_p99"] = pd.to_numeric(top["snr_p99"], errors="coerce").map(lambda v: f"{v:.2f}")
    top["positive_fraction"] = pd.to_numeric(top["positive_fraction"], errors="coerce").map(lambda v: f"{v:.4f}")
    top["p99"] = pd.to_numeric(top["p99"], errors="coerce").map(lambda v: f"{v:.1f}")
    top["marker_name"] = top["marker_name"].astype(str).map(lambda v: v if len(v) <= 22 else f"{v[:19]}...")
    table = top[["roi", "marker_name", "p99", "snr_p99", "positive_fraction", "flags"]]
    mpl_table = ax.table(cellText=table.values, colLabels=table.columns, loc="upper left", cellLoc="left")
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(6)
    mpl_table.scale(1.0, 1.2)


def _plot_variant_pdf(df: pd.DataFrame, cfg: Config, bg: bool, out_path: Path) -> None:
    matplotlib.rcParams["pdf.compression"] = cfg.viz.pdf_compression
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp.pdf")
    variant = "bg-sub" if bg else "no-bg-sub"
    try:
        with PdfPages(str(tmp)) as pdf:
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            summary = [
                f"Experiment: {cfg.experiment.name}",
                f"Variant: {variant}",
                f"ROIs: {df['roi'].nunique()}",
                f"Markers: {df['marker_name'].nunique()}",
                f"ROI-channel rows: {len(df)}",
                f"Flagged rows: {int((df['flag_count'] > 0).sum())}",
                f"Low sensitivity: {int(df['low_sensitivity'].sum())}",
                f"Low intensity: {int(df['low_intensity'].sum())}",
                f"Saturated: {int(df['saturated'].sum())}",
                f"Uneven: {int(df['uneven'].sum())}",
            ]
            ax = fig.add_subplot(gs[0, 0])
            ax.axis("off")
            ax.set_title("Channel QC Summary", fontsize=11, fontweight="bold", loc="left")
            ax.text(0.0, 0.92, "\n".join(summary), va="top", transform=ax.transAxes)
            counts = df[["low_intensity", "low_sensitivity", "saturated", "uneven"]].sum()
            ax = fig.add_subplot(gs[0, 1])
            ax.bar(counts.index, counts.to_numpy(), color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"])
            ax.set_ylabel("Flagged ROI-channels")
            ax.tick_params(axis="x", labelrotation=25)
            _plot_heatmap(fig.add_subplot(gs[1, 0]), np.log1p(_pivot(df, "p99")), "log1p p99 intensity")
            _plot_heatmap(fig.add_subplot(gs[1, 1]), _pivot(df, "snr_p99"), "Sensitivity: SNR p99")
            fig.suptitle(f"Staining QC: {variant}", fontsize=14, fontweight="bold")
            pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            _plot_heatmap(fig.add_subplot(gs[0, 0]), _pivot(df, "positive_fraction"), "Positive fraction")
            _plot_heatmap(fig.add_subplot(gs[0, 1]), _pivot(df, "saturated_fraction"), "Saturated fraction")
            _plot_heatmap(fig.add_subplot(gs[1, 0]), _pivot(df, "tile_median_cv"), "Tile median CV")
            _plot_heatmap(fig.add_subplot(gs[1, 1]), _pivot(df, "edge_center_ratio"), "Edge / center median")
            fig.suptitle(f"Staining QC Details: {variant}", fontsize=14, fontweight="bold")
            pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            _plot_flag_table(fig.add_subplot(111), df, "Top flagged ROI-channels", cfg)
            pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            plt.close(fig)

        if not render.is_valid_output(tmp, "pdf"):
            raise OSError(f"matplotlib produced an invalid pdf file: {tmp}")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
        gc.collect()
    log.info("wrote channel QC summary -> %s", out_path)


def run_variant_qc(
    cfg: Config,
    rois: list[RoiImage],
    channel_info: pd.DataFrame,
    first_sample: Path,
    bg: bool,
) -> pd.DataFrame | None:
    """Write per-variant channel QC CSV/PDF, reusing valid completed outputs."""
    if not cfg.viz.channel_qc.enabled:
        return None
    csv_out = qc_csv_path(cfg, bg)
    pdf_out = qc_pdf_path(cfg, bg)
    if is_valid_qc_csv(csv_out):
        log.info("skipping completed channel QC table: %s", csv_out)
        df = pd.read_csv(csv_out)
    else:
        df = _compute_qc_table(cfg, rois, channel_info, first_sample, bg)
        _write_csv(df, csv_out)
    if render.is_valid_output(pdf_out, "pdf"):
        log.info("skipping completed channel QC summary: %s", pdf_out)
    else:
        _plot_variant_pdf(df, cfg, bg, pdf_out)
    return df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return num / den.where(den.abs() > EPS)


def _comparison_flags(row) -> str:
    flags = []
    if bool(getattr(row, "over_subtracted")):
        flags.append("over_subtracted")
    if bool(getattr(row, "weak_bg_removal")):
        flags.append("weak_bg_removal")
    return ";".join(flags)


def build_bg_comparison(no_bs: pd.DataFrame, bg: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    keep = [
        "roi",
        "marker_name",
        "median",
        "p99",
        "snr_p99",
        "positive_fraction",
        "flag_count",
        "flags",
    ]
    no = no_bs[keep].rename(columns={c: f"{c}_no_bs" for c in keep if c not in {"roi", "marker_name"}})
    bs = bg[keep].rename(columns={c: f"{c}_bg_sub" for c in keep if c not in {"roi", "marker_name"}})
    out = no.merge(bs, on=["roi", "marker_name"], how="inner")
    out["p50_reduction"] = 1.0 - _safe_ratio(out["median_bg_sub"], out["median_no_bs"])
    out["p99_retention"] = _safe_ratio(out["p99_bg_sub"], out["p99_no_bs"])
    out["snr_change"] = pd.to_numeric(out["snr_p99_bg_sub"], errors="coerce") - pd.to_numeric(
        out["snr_p99_no_bs"], errors="coerce"
    )
    out["positive_fraction_change"] = pd.to_numeric(out["positive_fraction_bg_sub"], errors="coerce") - pd.to_numeric(
        out["positive_fraction_no_bs"], errors="coerce"
    )
    out["over_subtracted"] = (out["p99_retention"] < 0.25) | (out["positive_fraction_change"] < -0.25)
    out["weak_bg_removal"] = (out["p50_reduction"].abs() < 0.05) & (out["snr_change"] <= 0)
    out["comparison_flag_count"] = out[["over_subtracted", "weak_bg_removal"]].sum(axis=1).astype(int)
    out["comparison_flags"] = [_comparison_flags(row) for row in out.itertuples()]
    return _sort_qc(out.rename(columns={"marker_name": "marker_name"}))


def _plot_comparison_pdf(df: pd.DataFrame, cfg: Config, out_path: Path) -> None:
    matplotlib.rcParams["pdf.compression"] = cfg.viz.pdf_compression
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp.pdf")
    try:
        with PdfPages(str(tmp)) as pdf:
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            summary = [
                f"Experiment: {cfg.experiment.name}",
                f"Paired ROI-channel rows: {len(df)}",
                f"Over-subtracted flags: {int(df['over_subtracted'].sum())}",
                f"Weak background-removal flags: {int(df['weak_bg_removal'].sum())}",
            ]
            ax = fig.add_subplot(gs[0, 0])
            ax.axis("off")
            ax.set_title("Background Subtraction QC", fontsize=11, fontweight="bold", loc="left")
            ax.text(0.0, 0.9, "\n".join(summary), va="top", transform=ax.transAxes)
            counts = df[["over_subtracted", "weak_bg_removal"]].sum()
            ax = fig.add_subplot(gs[0, 1])
            ax.bar(counts.index, counts.to_numpy(), color=["#e45756", "#4c78a8"])
            ax.tick_params(axis="x", labelrotation=20)
            ax.set_ylabel("Flagged ROI-channels")
            _plot_heatmap(fig.add_subplot(gs[1, 0]), _pivot(df, "p99_retention"), "p99 retention (bg / no-BS)")
            _plot_heatmap(fig.add_subplot(gs[1, 1]), _pivot(df, "snr_change"), "SNR change (bg - no-BS)")
            fig.suptitle("Channel QC: Background Subtraction Comparison", fontsize=14, fontweight="bold")
            pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            plt.close(fig)

            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            gs = fig.add_gridspec(2, 1)
            _plot_heatmap(fig.add_subplot(gs[0, 0]), _pivot(df, "p50_reduction"), "Median intensity reduction")
            _plot_heatmap(fig.add_subplot(gs[1, 0]), _pivot(df, "positive_fraction_change"), "Positive fraction change")
            pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            plt.close(fig)
        if not render.is_valid_output(tmp, "pdf"):
            raise OSError(f"matplotlib produced an invalid pdf file: {tmp}")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
        gc.collect()
    log.info("wrote channel QC background comparison -> %s", out_path)


def write_bg_comparison(cfg: Config) -> pd.DataFrame | None:
    if not cfg.viz.channel_qc.enabled:
        return None
    no_bs_csv = qc_csv_path(cfg, bg=False)
    bg_csv = qc_csv_path(cfg, bg=True)
    if not is_valid_qc_csv(no_bs_csv) or not is_valid_qc_csv(bg_csv):
        return None
    out_csv = comparison_csv_path(cfg)
    out_pdf = comparison_pdf_path(cfg)
    if out_csv.is_file() and out_csv.stat().st_size > 0 and render.is_valid_output(out_pdf, "pdf"):
        log.info("skipping completed channel QC background comparison: %s", out_csv)
        return pd.read_csv(out_csv)
    comparison = build_bg_comparison(pd.read_csv(no_bs_csv), pd.read_csv(bg_csv), cfg)
    _write_csv(comparison, out_csv)
    _plot_comparison_pdf(comparison, cfg, out_pdf)
    return comparison
