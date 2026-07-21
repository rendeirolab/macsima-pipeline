"""Multi-page phenotype QC PDF: composition, confidence, spatial coherence.

Written atomically (temp file -> os.replace), mirroring the viz QC reports. Reads the
already-computed metrics; it never recomputes phenotyping.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _heatmap(ax, matrix: np.ndarray, row_labels, col_labels, title: str, cmap: str = "viridis") -> None:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_title(title, fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def write_phenotype_report(cfg, adata, bg: bool, results: dict, qc: dict, comp: pd.DataFrame,
                           agreement: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    suffix = cfg.suffix_for(bg)
    out_dir = cfg.qc_dir() / "phenotype"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{cfg.experiment.name}_phenotype_summary{suffix}.pdf"
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.tmp")

    cats = adata.obs["cell_type"].astype("category")
    counts = cats.value_counts()

    try:
        with PdfPages(tmp) as pdf:
            # --- page 1: summary ---
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes[0, 0].bar(range(len(counts)), counts.to_numpy(), color="#4c78a8")
            axes[0, 0].set_xticks(range(len(counts)))
            axes[0, 0].set_xticklabels(counts.index, rotation=90, fontsize=6)
            axes[0, 0].set_title("cells per cell type", fontsize=9)

            axes[0, 1].hist(adata.obs["cell_type_confidence"].to_numpy(), bins=30, color="#72b7b2")
            axes[0, 1].set_title("primary-engine confidence", fontsize=9)
            axes[0, 1].set_xlabel("max probability")

            lines = [
                f"experiment: {cfg.experiment.name}",
                f"variant: {'bg-sub' if bg else 'no-bg-sub'}",
                f"cells: {adata.n_obs}",
                f"markers: {adata.n_vars}",
                f"cell types: {len(counts)}",
                f"engines: {', '.join(results)}",
            ]
            if agreement:
                lines += [
                    "",
                    "cross-engine agreement:",
                    f"  accuracy: {agreement.get('accuracy', float('nan')):.3f}",
                    f"  cohen_kappa: {agreement.get('cohen_kappa', float('nan')):.3f}",
                    f"  adjusted_rand: {agreement.get('adjusted_rand', float('nan')):.3f}",
                ]
            homo = qc.get("homophily", {})
            if homo:
                lines += ["", f"spatial homophily (overall): {homo.get('overall', float('nan')):.3f}"]
            axes[1, 0].axis("off")
            axes[1, 0].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

            per_type = (qc.get("homophily", {}) or {}).get("per_type", {})
            if per_type:
                keys = list(per_type)
                axes[1, 1].barh(range(len(keys)), [per_type[k] for k in keys], color="#e45756")
                axes[1, 1].set_yticks(range(len(keys)))
                axes[1, 1].set_yticklabels(keys, fontsize=6)
                axes[1, 1].set_title("same-type neighbor fraction", fontsize=9)
                axes[1, 1].set_xlim(0, 1)
            else:
                axes[1, 1].axis("off")
            fig.suptitle(f"Phenotype QC — {cfg.experiment.name}{suffix}", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

            # --- page 2: composition heatmap ---
            if comp is not None and not comp.empty:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                _heatmap(ax, comp.to_numpy(), list(comp.index), list(comp.columns),
                         "cell-type composition (fraction) per ROI", cmap="magma")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # --- page 3: neighborhood enrichment ---
            z = qc.get("nhood_zscore")
            if z is not None:
                labels = qc.get("labels", [str(i) for i in range(len(z))])
                fig, ax = plt.subplots(figsize=(9, 8))
                _heatmap(ax, np.asarray(z), labels, labels,
                         "neighborhood enrichment (z-score)", cmap="coolwarm")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    log.info("phenotype report: [path]%s[/]", dest)
    return dest
