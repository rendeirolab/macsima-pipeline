"""Spatial-coherence QC — formalizes the "do labels make sense on the map?" check.

Builds a per-ROI spatial graph and measures whether cell-type labels are spatially
coherent (neighborhood enrichment + same-type homophily) and how the two engines
agree. Incoherent maps or low agreement are the automatic red flags that replace
eyeballing spatial scatters.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import PhenotypeSpatialQCCfg

log = logging.getLogger(__name__)


def build_spatial(adata, x_col: str, y_col: str) -> None:
    """Populate `obsm['spatial']` from centroid columns (squidpy convention)."""
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].to_numpy(dtype=float)


def _homophily(adata, label_key: str) -> dict:
    """Fraction of same-type neighbors, overall and per cell type (vectorized)."""
    adj = adata.obsp["spatial_connectivities"].tocsr()
    cats = adata.obs[label_key].astype("category")
    codes = cats.cat.codes.to_numpy()
    n, k = adj.shape[0], len(cats.cat.categories)
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, k))
    neigh_by_type = adj @ onehot                     # (N, K) neighbor counts per type
    same = np.asarray(neigh_by_type.multiply(onehot).sum(axis=1)).ravel()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    frac = np.divide(same, deg, out=np.zeros(n), where=deg > 0)
    per_type = {
        str(t): float(frac[codes == i].mean())
        for i, t in enumerate(cats.cat.categories)
        if (codes == i).any()
    }
    return {"overall": float(frac[deg > 0].mean()) if (deg > 0).any() else 0.0, "per_type": per_type}


def compute_spatial_qc(adata, cfg: PhenotypeSpatialQCCfg, label_key: str, batch_key: str | None = None) -> dict:
    """Build the spatial graph and compute neighborhood enrichment + homophily."""
    if not cfg.enabled:
        return {}
    import squidpy as sq

    adata.obs[label_key] = adata.obs[label_key].astype("category")
    neigh_kwargs: dict = {"coord_type": cfg.coord_type, "n_neighs": cfg.n_neighs}
    if batch_key and batch_key in adata.obs.columns:
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
        neigh_kwargs["library_key"] = batch_key  # graphs never cross ROIs
    sq.gr.spatial_neighbors(adata, **neigh_kwargs)

    out: dict = {"labels": [str(c) for c in adata.obs[label_key].cat.categories]}
    if cfg.nhood_enrichment:
        try:
            sq.gr.nhood_enrichment(
                adata, cluster_key=label_key, n_perms=cfg.n_perms, seed=cfg.random_seed,
                show_progress_bar=False,
            )
            z = adata.uns[f"{label_key}_nhood_enrichment"]["zscore"]
            out["nhood_zscore"] = np.asarray(z, dtype=np.float32)
        except Exception as e:  # noqa: BLE001 - QC must not crash the stage
            log.warning("nhood_enrichment failed (%s); skipping", e)
    if cfg.homophily:
        out["homophily"] = _homophily(adata, label_key)
    return out


def composition_table(adata, label_key: str, batch_key: str | None = None) -> pd.DataFrame:
    """Per-ROI (rows) × cell-type (cols) composition fractions."""
    labels = adata.obs[label_key].astype("category")
    if batch_key and batch_key in adata.obs.columns:
        groups = adata.obs[batch_key].astype(str)
    else:
        groups = pd.Series(["all"] * adata.n_obs, index=adata.obs_names)
    return pd.crosstab(groups, labels, normalize="index")


def cross_engine_agreement(a: pd.Series, b: pd.Series) -> tuple[pd.Series, dict]:
    """Per-cell agreement flag + global metrics (accuracy, Cohen's κ, ARI)."""
    from sklearn.metrics import adjusted_rand_score, cohen_kappa_score

    a_arr = np.asarray(a.to_numpy(), dtype=object)
    b_arr = np.asarray(b.to_numpy(), dtype=object)
    agree = pd.Series(a_arr == b_arr, index=a.index, name="pheno_agree")
    metrics = {
        "accuracy": float(agree.mean()),
        "cohen_kappa": float(cohen_kappa_score(a_arr, b_arr)),
        "adjusted_rand": float(adjusted_rand_score(a_arr, b_arr)),
        "n_cells": int(len(a_arr)),
    }
    return agree, metrics
