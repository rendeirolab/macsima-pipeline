"""Per-marker normalization + batch handling for phenotyping.

Pure numpy, deterministic, sparse-safe. Operates in place on `adata.X`:

    stash_raw(adata)      # layers['counts'] <- raw (kept for provenance)
    normalize(adata, cfg) # X <- winsorize -> transform -> z-score
    apply_batch(adata, cfg)  # X <- batch-corrected (per-ROI z-score by default)

The caller then copies the final X into `layers[normalized_layer]`; both engines
(scyan, Leiden) read that normalized layer.

Design note: normalization choice dominates cell-typing accuracy for imaging data,
and per-marker z-score is the most robust transform (Hickey et al. 2021).
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
    if cfg.embedding_key:
        # Embedding mode: write a corrected embedding, leave X (and the marker layer) alone.
        _harmony_embedding(adata, cfg)
        return
    if cfg.method == "zscore_per_roi":
        _zscore_per_batch(adata, cfg.batch_key, cfg.min_cells_per_batch)
    elif cfg.method == "combat":
        import scanpy as sc

        sc.pp.combat(adata, key=cfg.batch_key)
    elif cfg.method == "quantile_reference":
        _quantile_reference(adata, cfg.batch_key, cfg.reference)
    elif cfg.method == "harmony":
        _harmony(adata, cfg)


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


def _harmony_kwargs(cfg: PhenotypeBatchCfg) -> dict:
    kwargs: dict = {"sigma": cfg.harmony_sigma, "max_iter_harmony": cfg.harmony_max_iter, "verbose": False}
    if cfg.harmony_theta is not None:
        kwargs["theta"] = cfg.harmony_theta
    if cfg.harmony_nclust is not None:
        kwargs["nclust"] = cfg.harmony_nclust
    return kwargs


def _harmony_embedding(adata, cfg: PhenotypeBatchCfg) -> None:
    """PCA -> Harmony2 -> ``obsm[cfg.embedding_key]``; ``X`` is left untouched.

    This is harmony's intended use (correction on principal components). The embedding is
    for graph construction only -- marker-based labeling continues to read the uncorrected
    normalized layer, so integration cannot distort the intensities the signature reads.
    """
    import harmonypy
    import pandas as pd

    m = _to_dense(adata.X).astype(np.float64)
    batches = adata.obs[cfg.batch_key].astype(str).to_numpy()
    n_batches = len(np.unique(batches))
    if n_batches < 2:
        log.warning("harmony embedding: only %d level(s) of %r; storing uncorrected PCs",
                    n_batches, cfg.batch_key)

    n_pcs = int(min(cfg.embedding_n_pcs, m.shape[1] - 1, m.shape[0] - 1))
    # PCA on the z-scored markers (already centred per marker by `normalize`).
    m = m - m.mean(axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(m, full_matrices=False)
    pcs = m @ vt[:n_pcs].T
    var_frac = float((sv[:n_pcs] ** 2).sum() / (sv**2).sum())
    log.info(
        "harmony embedding: %d cells, %d PCs (%.1f%% variance), %d %r batches -> obsm[%r]",
        m.shape[0], n_pcs, 100 * var_frac, n_batches, cfg.batch_key, cfg.embedding_key,
    )

    if n_batches < 2:
        adata.obsm[cfg.embedding_key] = np.ascontiguousarray(pcs, dtype=np.float32)
        return

    res = harmonypy.run_harmony(
        pcs, pd.DataFrame({cfg.batch_key: batches}), [cfg.batch_key], **_harmony_kwargs(cfg)
    )
    z = np.asarray(res.Z_corr)
    if z.shape != pcs.shape:  # harmonypy 2.x returns (n_cells, n_dims); older transposed
        z = z.T
    if z.shape != pcs.shape:
        raise ValueError(f"harmony returned shape {np.asarray(res.Z_corr).shape}, expected {pcs.shape}")
    adata.obsm[cfg.embedding_key] = np.ascontiguousarray(z, dtype=np.float32)


def _harmony(adata, cfg: PhenotypeBatchCfg) -> None:
    """Harmony2 (harmonypy >= 2.0) applied directly in marker space.

    Harmony is normally run on principal components, but the correction it applies is a
    linear per-soft-cluster shift, so any embedding is a valid input space. Running it on
    the marker matrix keeps `X` interpretable and lets both engines (which read a
    marker-space layer) consume the result unchanged.
    """
    import harmonypy
    import pandas as pd

    m = _to_dense(adata.X).astype(np.float64)
    batches = adata.obs[cfg.batch_key].astype(str).to_numpy()
    n_batches = len(np.unique(batches))
    if n_batches < 2:
        log.warning("harmony: only %d level(s) of %r; skipping", n_batches, cfg.batch_key)
        return

    kwargs = _harmony_kwargs(cfg)

    log.info(
        "harmony: correcting %d cells x %d markers over %d %r batches",
        m.shape[0], m.shape[1], n_batches, cfg.batch_key,
    )
    res = harmonypy.run_harmony(m, pd.DataFrame({cfg.batch_key: batches}), [cfg.batch_key], **kwargs)

    z = np.asarray(res.Z_corr)
    # harmonypy 2.x returns (n_cells, n_markers); older releases returned it transposed.
    if z.shape != m.shape:
        z = z.T
    if z.shape != m.shape:
        raise ValueError(f"harmony returned shape {np.asarray(res.Z_corr).shape}, expected {m.shape}")
    adata.X = np.ascontiguousarray(z, dtype=np.float32)


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
