"""Cell-location and expression QC summary PDF rendering."""

from __future__ import annotations

import gc
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ..config import Config
from . import render

log = logging.getLogger(__name__)

XY_COLUMN_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("X", "Y"),
    ("centroid_x", "centroid_y"),
    ("x_centroid", "y_centroid"),
    ("center_x", "center_y"),
    ("CenterX", "CenterY"),
    ("X_centroid", "Y_centroid"),
)

_INTERNAL_OBS_COLUMNS = {
    "ROI",
    "cell_id",
    "region",
    "index",
    "slide",
    "instance_id",
    "label",
    # phenotype-stage outputs (never ROI-level metadata)
    "cell_type",
    "cell_type_coarse",
    "cell_type_confidence",
    "astir_celltype",
    "flowsom",
    "flowsom_celltype",
    "pheno_agree",
}


def resolve_xy_columns(obs: pd.DataFrame) -> tuple[str, str] | None:
    """Find usable cell-centroid columns in an AnnData obs table."""
    for x_col, y_col in XY_COLUMN_CANDIDATES:
        if x_col in obs.columns and y_col in obs.columns:
            x = pd.to_numeric(obs[x_col], errors="coerce")
            y = pd.to_numeric(obs[y_col], errors="coerce")
            if bool((x.notna() & y.notna()).any()):
                return x_col, y_col
    return None


def _roi_sort_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if match is None:
        return text, -1, text
    return text[: match.start()], int(match.group(1)), text


def _obs_with_roi(adata) -> pd.DataFrame:
    obs = adata.obs.copy()
    if "ROI" not in obs.columns:
        obs["ROI"] = "all"
    obs["ROI"] = obs["ROI"].fillna("missing").astype(str)
    return obs


def _metadata_columns(obs: pd.DataFrame, xy_cols: tuple[str, str] | None) -> list[str]:
    coord_cols = {col for pair in XY_COLUMN_CANDIDATES for col in pair}
    if xy_cols is not None:
        coord_cols.update(xy_cols)

    metadata_cols: list[str] = []
    for col in obs.columns:
        if col in _INTERNAL_OBS_COLUMNS or col in coord_cols:
            continue
        series = obs[col]
        if int(series.notna().sum()) == 0:
            continue
        try:
            max_values_per_roi = int(obs.groupby("ROI", observed=False)[col].nunique(dropna=True).max())
        except (TypeError, ValueError):
            continue
        if max_values_per_roi <= 3:
            metadata_cols.append(col)
    return metadata_cols


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value)
    return text if len(text) <= 32 else f"{text[:29]}..."


def _roi_metadata_lines(obs: pd.DataFrame, metadata_cols: list[str]) -> list[str]:
    lines: list[str] = []
    for col in metadata_cols:
        values = [_format_value(v) for v in pd.unique(obs[col].dropna())[:3]]
        if not values:
            values = ["missing"]
        suffix = "" if len(values) < 3 else " ..."
        lines.append(f"{col}: {', '.join(values)}{suffix}")
    return lines


def _marker_lines(markers: list[str], ncols: int) -> list[str]:
    if not markers:
        return ["No markers found"]
    ncols = max(1, ncols)
    nrows = math.ceil(len(markers) / ncols)
    clipped = [m if len(m) <= 24 else f"{m[:21]}..." for m in markers]
    columns = [clipped[i * nrows : (i + 1) * nrows] for i in range(ncols)]
    widths = [max([len(v) for v in col] or [1]) for col in columns]
    lines: list[str] = []
    for row in range(nrows):
        parts = []
        for col, width in zip(columns, widths, strict=False):
            parts.append((col[row] if row < len(col) else "").ljust(width))
        lines.append("  ".join(parts).rstrip())
    return lines


def _matrix_sum_by_cell(matrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([], dtype=float)
    return np.asarray(matrix.sum(axis=1)).reshape(-1).astype(float, copy=False)


def _matrix_detected_by_cell(matrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([], dtype=float)
    return np.asarray((matrix > 0).sum(axis=1)).reshape(-1).astype(float, copy=False)


def _dense_matrix(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _cell_qc(adata) -> tuple[np.ndarray, np.ndarray]:
    return _matrix_sum_by_cell(adata.X), _matrix_detected_by_cell(adata.X)


def _title(ax, text: str) -> None:
    ax.set_title(text, fontsize=10, fontweight="bold", loc="left")


def _text_panel(ax, title: str, lines: list[str], *, fontsize: float = 8.5, family: str | None = None) -> None:
    ax.axis("off")
    _title(ax, title)
    ax.text(
        0.0,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=fontsize,
        family=family,
        transform=ax.transAxes,
    )


def _plot_total_hist(ax, totals: np.ndarray, title: str) -> None:
    _title(ax, title)
    if totals.size == 0:
        ax.text(0.5, 0.5, "No cells", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    values = np.log10(np.clip(totals, 0, None) + 1)
    ax.hist(values, bins=min(50, max(8, int(np.sqrt(values.size)))), color="#4c78a8")
    ax.set_xlabel("log10(total expression + 1)")
    ax.set_ylabel("Cells")


def _plot_detected_hist(ax, detected: np.ndarray, title: str) -> None:
    _title(ax, title)
    if detected.size == 0:
        ax.text(0.5, 0.5, "No cells", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    upper = int(np.nanmax(detected)) if detected.size else 0
    bins = np.arange(-0.5, upper + 1.5, 1)
    ax.hist(detected, bins=bins, color="#59a14f")
    ax.set_xlabel("Detected markers per cell")
    ax.set_ylabel("Cells")


def _plot_roi_counts(ax, roi_counts: pd.Series) -> None:
    _title(ax, "Cells per ROI")
    if roi_counts.empty:
        ax.text(0.5, 0.5, "No ROI records", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    labels = [str(v) for v in roi_counts.index]
    ax.bar(range(len(roi_counts)), roi_counts.to_numpy(), color="#f28e2b")
    ax.set_ylabel("Cells")
    ax.set_xticks(range(len(labels)))
    if len(labels) <= 35:
        ax.set_xticklabels(labels, rotation=90 if len(labels) > 10 else 45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel(f"{len(labels)} ROIs")
    stats = roi_counts.describe()
    ax.set_title(
        "Cells per ROI "
        f"(min={int(stats['min'])}, median={int(roi_counts.median())}, max={int(stats['max'])})",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )


def _plot_cell_map(
    ax,
    obs: pd.DataFrame,
    xy_cols: tuple[str, str] | None,
    cfg: Config,
    label_col: str | None = None,
) -> None:
    title = f"Cell XY map ({label_col})" if label_col else "Cell XY map"
    _title(ax, title)
    if xy_cols is None:
        ax.text(0.5, 0.5, "No usable XY columns found", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    x_col, y_col = xy_cols
    x = pd.to_numeric(obs[x_col], errors="coerce")
    y = pd.to_numeric(obs[y_col], errors="coerce")
    finite = x.notna() & y.notna()
    if not bool(finite.any()):
        ax.text(0.5, 0.5, "No finite cell coordinates", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    if label_col and label_col in obs.columns:
        _scatter_by_label(ax, x[finite], y[finite], obs[label_col][finite], cfg)
    else:
        ax.scatter(
            x[finite], y[finite], s=cfg.viz.cell_map_point_size, c="#4c78a8",
            alpha=0.7, linewidths=0, rasterized=True,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def _scatter_by_label(ax, x, y, labels: pd.Series, cfg: Config) -> None:
    """Scatter cells colored by a categorical label, with a compact legend."""
    cats = [str(c) for c in pd.Categorical(labels.astype(str)).categories]
    cmap = matplotlib.colormaps["tab20"] if len(cats) > 10 else matplotlib.colormaps["tab10"]
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    label_str = labels.astype(str)
    for c in cats:
        m = (label_str == c).to_numpy()
        ax.scatter(
            x[m], y[m], s=cfg.viz.cell_map_point_size, color=colors[c],
            alpha=0.8, linewidths=0, rasterized=True, label=c,
        )
    ax.legend(markerscale=4, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def _plot_marker_qc(ax, adata_roi, cfg: Config) -> None:
    _title(ax, "Top marker QC")
    if adata_roi.n_obs == 0 or adata_roi.n_vars == 0:
        ax.text(0.5, 0.5, "No marker data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    matrix = _dense_matrix(adata_roi.X)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    medians = np.nanmedian(matrix, axis=0)
    if bool(np.nanmax(medians) > 0):
        values = medians
        xlabel = "Median expression"
    else:
        values = np.nanmean(matrix > 0, axis=0)
        xlabel = "Detection fraction"
    top_n = max(1, min(cfg.viz.cell_map_marker_top_n, len(values)))
    order = np.argsort(values)[-top_n:]
    names = np.asarray([str(v) for v in adata_roi.var_names])[order]
    shown = values[order]
    ax.barh(range(top_n), shown, color="#b279a2")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(xlabel)


def _summary_figure(
    adata,
    obs: pd.DataFrame,
    xy_cols: tuple[str, str] | None,
    metadata_cols: list[str],
    cfg: Config,
    bg: bool,
    h5ad_path: Path | None,
) -> plt.Figure:
    totals, detected = _cell_qc(adata)
    roi_counts = obs["ROI"].value_counts()
    roi_counts = roi_counts.loc[sorted(roi_counts.index, key=_roi_sort_key)]
    variant = "bg-sub" if bg else "no-bg-sub"
    marker_names = [str(v) for v in adata.var_names]
    notes: list[str] = []
    if xy_cols is None:
        notes.append("No usable XY coordinate columns found.")
    else:
        x = pd.to_numeric(obs[xy_cols[0]], errors="coerce")
        y = pd.to_numeric(obs[xy_cols[1]], errors="coerce")
        missing_xy = int((x.isna() | y.isna()).sum())
        if missing_xy:
            notes.append(f"{missing_xy} cells have missing XY coordinates.")
    if metadata_cols:
        gaps = [f"{col} ({int(obs[col].isna().sum())} missing)" for col in metadata_cols if obs[col].isna().any()]
        notes.append("Metadata gaps: " + (", ".join(gaps) if gaps else "none"))
    else:
        notes.append("No ROI-level metadata columns detected.")

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.1])
    info = [
        f"Experiment: {cfg.experiment.name}",
        f"Variant: {variant}",
        f"ROIs: {len(roi_counts)}",
        f"Cells: {adata.n_obs}",
        f"Markers: {adata.n_vars}",
        f"XY columns: {xy_cols[0]} / {xy_cols[1]}" if xy_cols else "XY columns: not found",
        f"AnnData: {h5ad_path.name if h5ad_path is not None else 'in-memory'}",
        "",
        "ROI metadata columns:",
        ", ".join(metadata_cols) if metadata_cols else "none",
        "",
        "Notes:",
        *notes,
    ]
    _text_panel(fig.add_subplot(gs[0, 0]), "Experiment summary", info)
    _text_panel(
        fig.add_subplot(gs[0, 1]),
        f"Markers ({len(marker_names)})",
        _marker_lines(marker_names, cfg.viz.cell_map_marker_columns),
        fontsize=7.0,
        family="monospace",
    )
    _plot_roi_counts(fig.add_subplot(gs[1, :]), roi_counts)
    _plot_total_hist(fig.add_subplot(gs[2, 0]), totals, "Expression per cell")
    _plot_detected_hist(fig.add_subplot(gs[2, 1]), detected, "Detected markers per cell")
    fig.suptitle("Cell Map QC Summary", fontsize=14, fontweight="bold")
    return fig


def _roi_figure(
    adata,
    obs: pd.DataFrame,
    roi: str,
    xy_cols: tuple[str, str] | None,
    metadata_cols: list[str],
    cfg: Config,
    label_col: str | None = None,
) -> plt.Figure:
    mask = obs["ROI"].eq(roi).to_numpy()
    roi_obs = obs.loc[mask]
    roi_adata = adata[mask]
    totals, detected = _cell_qc(roi_adata)
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0])
    _plot_cell_map(fig.add_subplot(gs[:, 0]), roi_obs, xy_cols, cfg, label_col=label_col)

    lines = [
        f"ROI: {roi}",
        f"Cells: {roi_adata.n_obs}",
    ]
    if xy_cols is not None and roi_adata.n_obs:
        x = pd.to_numeric(roi_obs[xy_cols[0]], errors="coerce")
        y = pd.to_numeric(roi_obs[xy_cols[1]], errors="coerce")
        missing_xy = int((x.isna() | y.isna()).sum())
        lines.append(f"Missing XY: {missing_xy}")
    if totals.size:
        lines.append(f"Median total expr: {float(np.nanmedian(totals)):.2f}")
    if detected.size:
        lines.append(f"Median markers detected: {float(np.nanmedian(detected)):.1f}")
    if metadata_cols:
        lines.extend(["", "Metadata:", *_roi_metadata_lines(roi_obs, metadata_cols)])
    else:
        lines.extend(["", "Metadata: none detected"])
    _text_panel(fig.add_subplot(gs[0, 1]), "ROI summary", lines)
    _plot_total_hist(fig.add_subplot(gs[0, 2]), totals, "Expression per cell")
    _plot_detected_hist(fig.add_subplot(gs[1, 1]), detected, "Detected markers per cell")
    _plot_marker_qc(fig.add_subplot(gs[1, 2]), roi_adata, cfg)
    fig.suptitle(f"Cell Map QC: {roi}", fontsize=14, fontweight="bold")
    return fig


def _phenotype_page(adata, cfg: Config) -> plt.Figure | None:
    """One-page phenotype overview from the precomputed uns['phenotype'] metrics."""
    pheno = adata.uns.get("phenotype", {})
    if not isinstance(pheno, dict):
        return None
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    _title(ax0, "Cells per cell type")
    if "cell_type" in adata.obs.columns:
        counts = adata.obs["cell_type"].astype(str).value_counts()
        ax0.bar(range(len(counts)), counts.to_numpy(), color="#4c78a8")
        ax0.set_xticks(range(len(counts)))
        ax0.set_xticklabels(counts.index, rotation=90, fontsize=6)

    lines = [
        f"Engines: {', '.join(pheno.get('engines', []))}",
        f"Primary: {pheno.get('primary_engine', '')}",
    ]
    agr = pheno.get("agreement", {})
    if isinstance(agr, dict) and agr:
        lines += [
            "",
            "Cross-engine agreement:",
            f"  accuracy: {float(agr.get('accuracy', float('nan'))):.3f}",
            f"  cohen_kappa: {float(agr.get('cohen_kappa', float('nan'))):.3f}",
            f"  adjusted_rand: {float(agr.get('adjusted_rand', float('nan'))):.3f}",
        ]
    sq = pheno.get("spatial_qc", {}) if isinstance(pheno.get("spatial_qc"), dict) else {}
    homo = sq.get("homophily", {}) if isinstance(sq.get("homophily"), dict) else {}
    if homo:
        lines += ["", f"Spatial homophily (overall): {float(homo.get('overall', float('nan'))):.3f}"]
    _text_panel(fig.add_subplot(gs[0, 1]), "Phenotype summary", lines)

    ax2 = fig.add_subplot(gs[1, 0])
    _title(ax2, "Composition per ROI")
    comp = pheno.get("composition")
    if not isinstance(comp, pd.DataFrame) and isinstance(comp, dict):
        try:
            comp = pd.DataFrame(comp)
        except Exception:  # noqa: BLE001
            comp = None
    if isinstance(comp, pd.DataFrame) and not comp.empty:
        im = ax2.imshow(comp.to_numpy(), aspect="auto", cmap="magma")
        ax2.set_xticks(range(comp.shape[1]))
        ax2.set_xticklabels([str(c) for c in comp.columns], rotation=90, fontsize=6)
        ax2.set_yticks(range(comp.shape[0]))
        ax2.set_yticklabels([str(i) for i in comp.index], fontsize=6)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 1])
    _title(ax3, "Neighborhood enrichment (z)")
    z = sq.get("nhood_zscore")
    if z is not None:
        z = np.asarray(z)
        labels = sq.get("labels", [str(i) for i in range(z.shape[0])])
        im = ax3.imshow(z, aspect="auto", cmap="coolwarm")
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=90, fontsize=6)
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle("Phenotype QC", fontsize=14, fontweight="bold")
    return fig


def plot_cell_map_qc_summary(
    adata,
    cfg: Config,
    out_path: Path,
    *,
    bg: bool,
    h5ad_path: Path | None = None,
) -> Path:
    """Render a multi-page PDF with experiment and per-ROI cell-map QC."""
    matplotlib.rcParams["pdf.compression"] = cfg.viz.pdf_compression
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp.pdf")
    obs = _obs_with_roi(adata)
    xy_cols = resolve_xy_columns(obs)
    metadata_cols = _metadata_columns(obs, xy_cols)
    rois = sorted(pd.unique(obs["ROI"]), key=_roi_sort_key)
    # Color spatial maps by phenotype label when the phenotype stage has run.
    label_col = "cell_type" if "cell_type" in obs.columns else None

    try:
        with PdfPages(str(tmp)) as pdf:
            fig = _summary_figure(adata, obs, xy_cols, metadata_cols, cfg, bg, h5ad_path)
            try:
                pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
            finally:
                plt.close(fig)

            if "phenotype" in getattr(adata, "uns", {}):
                fig = _phenotype_page(adata, cfg)
                if fig is not None:
                    try:
                        pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight",
                                    pad_inches=cfg.viz.output_pad_inches)
                    finally:
                        plt.close(fig)

            for roi in rois:
                fig = _roi_figure(adata, obs, str(roi), xy_cols, metadata_cols, cfg, label_col=label_col)
                try:
                    pdf.savefig(fig, dpi=cfg.viz.dpi, bbox_inches="tight", pad_inches=cfg.viz.output_pad_inches)
                finally:
                    plt.close(fig)

        if not render.is_valid_output(tmp, "pdf"):
            raise OSError(f"matplotlib produced an invalid pdf file: {tmp}")
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
        gc.collect()

    log.info("wrote cell-map QC summary -> %s", out_path)
    return out_path
