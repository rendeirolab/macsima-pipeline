"""Per-marker normalization + batch handling for phenotyping.

Pure numpy, deterministic, sparse-safe. Operates in place on `adata.X`:

    stash_raw(adata)      # layers['counts'] <- raw (Astir reads this untouched)
    normalize(adata, cfg) # X <- winsorize -> transform -> z-score
    apply_batch(adata, cfg)  # X <- batch-corrected (per-ROI z-score by default)

The caller then copies the final X into `layers[normalized_layer]` (FlowSOM input).

Design note: normalization choice dominates cell-typing accuracy for imaging data,
and per-marker z-score is the most robust transform (Hickey et al. 2021). Astir does
NOT use this output — it re-derives its own transform from the raw counts layer.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from ..config import PhenotypeBatchCfg, PhenotypeNormalizeCfg

log = logging.getLogger(__name__)


def _to_dense(x) -> np.ndarray:
    """Dense float32 view of a possibly-sparse matrix."""
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def stash_raw(adata, layer: str = "counts") -> None:
    """Preserve raw intensities in `layers[layer]` before X is normalized."""
    if layer:
        adata.layers[layer] = _to_dense(adata.X)


# ---- transforms (per-marker == per-column) ---------------------------------


def _winsorize(m: np.ndarray, p_low: float | None, p_high: float | None) -> np.ndarray:
    out = m
    if p_high is not None:
        out = np.minimum(out, np.percentile(out, p_high, axis=0))
    if p_low is not None:
        out = np.maximum(out, np.percentile(out, p_low, axis=0))
    return out


def _arcsinh(m: np.ndarray, cofactor: float, cofactors: dict[str, float], var_names: list[str]) -> np.ndarray:
    cof = np.full(m.shape[1], float(cofactor), dtype=np.float64)
    for i, name in enumerate(var_names):
        if name in cofactors:
            cof[i] = float(cofactors[name])
    cof = np.where(cof <= 0, 1.0, cof)
    return np.arcsinh(m / cof)


def _percentile_scale(m: np.ndarray, p: float) -> np.ndarray:
    denom = np.percentile(m, p, axis=0)
    denom = np.where(denom <= 0, 1.0, denom)
    return np.clip(m / denom, 0.0, 1.0)


def _zscore(m: np.ndarray) -> np.ndarray:
    mean = m.mean(axis=0)
    std = m.std(axis=0)
    std = np.where(std <= 0, 1.0, std)  # constant columns -> 0, never NaN
    return (m - mean) / std


def normalize(adata, cfg: PhenotypeNormalizeCfg) -> None:
    """Winsorize -> transform -> z-score. Sets `adata.X` to a dense float32 matrix."""
    m = _to_dense(adata.X)
    m = _winsorize(m, cfg.clip_lower_percentile, cfg.clip_percentile)
    if cfg.transform == "arcsinh":
        m = _arcsinh(m, cfg.cofactor, cfg.cofactors, list(adata.var_names))
    elif cfg.transform == "percentile":
        m = _percentile_scale(m, cfg.percentile_norm_p)
    if cfg.zscore:
        m = _zscore(m)
    adata.X = m.astype(np.float32)


# ---- batch handling --------------------------------------------------------


def apply_batch(adata, cfg: PhenotypeBatchCfg) -> None:
    """Correct batch at the intensity stage (keeps markers interpretable)."""
    if cfg.method == "none":
        return
    if cfg.batch_key not in adata.obs.columns:
        log.warning("batch_key %r absent from obs; skipping batch correction", cfg.batch_key)
        return
    if cfg.method == "zscore_per_roi":
        _zscore_per_batch(adata, cfg.batch_key, cfg.min_cells_per_batch)
    elif cfg.method == "combat":
        import scanpy as sc

        sc.pp.combat(adata, key=cfg.batch_key)
    elif cfg.method == "quantile_reference":
        _quantile_reference(adata, cfg.batch_key, cfg.reference)


def _zscore_per_batch(adata, batch_key: str, min_cells: int) -> None:
    """Per-marker z-score within each batch; batches under `min_cells` use global stats."""
    m = _to_dense(adata.X)
    batches = np.asarray(adata.obs[batch_key].to_numpy())
    gmean = m.mean(axis=0)
    gstd = np.where(m.std(axis=0) <= 0, 1.0, m.std(axis=0))
    out = m.copy()
    for b in np.unique(batches):
        mask = batches == b
        if int(mask.sum()) < min_cells:
            mean, std = gmean, gstd
        else:
            mean = m[mask].mean(axis=0)
            std = np.where(m[mask].std(axis=0) <= 0, 1.0, m[mask].std(axis=0))
        out[mask] = (m[mask] - mean) / std
    adata.X = out.astype(np.float32)


def _quantile_reference(adata, batch_key: str, reference: str | None) -> None:
    """Map each batch's per-marker distribution onto a reference (or pooled) distribution."""
    m = _to_dense(adata.X)
    batches = np.asarray(adata.obs[batch_key].to_numpy())
    uniq = list(np.unique(batches))
    if reference is not None and reference in {str(u) for u in uniq}:
        ref = m[batches.astype(str) == str(reference)]
    else:
        ref = m
    ref_sorted = np.sort(ref, axis=0)
    n_ref = ref_sorted.shape[0]
    ref_q = (np.arange(n_ref) + 0.5) / n_ref
    out = m.copy()
    for b in uniq:
        mask = batches == b
        sub = m[mask]
        n = sub.shape[0]
        ranks = np.argsort(np.argsort(sub, axis=0), axis=0)
        q = (ranks + 0.5) / n
        for j in range(m.shape[1]):
            out[mask, j] = np.interp(q[:, j], ref_q, ref_sorted[:, j])
    adata.X = out.astype(np.float32)
