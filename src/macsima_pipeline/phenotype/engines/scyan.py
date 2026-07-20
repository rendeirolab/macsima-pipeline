"""Scyan engine adapter: probabilistic cell-type annotation via a normalizing flow.

Wraps the ``scyan`` package (Blampey et al.). Scyan consumes scaled expression (the
arcsinh + per-marker z-scored layer) plus a knowledge table (population x marker,
values -1/1/NaN) built from the signature matrix, and returns per-cell population
probabilities + hard labels, with unassigned / low-confidence cells marked "Unknown".

Input: the normalized z-scored layer (``cfg.use_layer``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..normalize import _to_dense
from ..signature import SignatureMatrix
from .base import EngineResult

log = logging.getLogger(__name__)


def _accelerator(device: str) -> str:
    """Map the config device to a lightning accelerator ("cpu"/"gpu")."""
    if device == "cuda":
        return "gpu"
    if device == "cpu":
        return "cpu"
    try:  # auto
        import torch

        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def run_scyan(adata, sig: SignatureMatrix, cfg, batch_key: str | None = None) -> EngineResult:
    import anndata as ad
    import scyan

    layer = cfg.use_layer
    if layer and layer in adata.layers:
        source = adata.layers[layer]
    else:
        log.warning("scyan use_layer=%r absent; falling back to X (must be scaled)", layer)
        source = adata.X
    z = _to_dense(source)
    panel = list(adata.var_names)

    # markers referenced by the signature that are present in the panel, in panel order
    referenced = set(sig.all_markers())
    markers = [m for m in panel if m in referenced]
    if not markers:
        raise ValueError("no signature markers present in the panel for scyan")
    col_idx = [panel.index(m) for m in markers]

    # knowledge table: population x marker, +1 positive / -1 negative / NaN otherwise
    names = sig.cell_type_names()
    table = pd.DataFrame(sig.score_matrix(markers), index=names, columns=markers).replace(0.0, np.nan)

    use_batch = bool(cfg.include_batch_covariate and batch_key and batch_key in adata.obs.columns)
    obs = pd.DataFrame(index=pd.Index([str(i) for i in range(adata.n_obs)]))
    if use_batch:
        obs[batch_key] = pd.Categorical(adata.obs[batch_key].astype(str).to_numpy())

    sub = ad.AnnData(
        X=np.ascontiguousarray(z[:, col_idx], dtype=np.float32),
        var=pd.DataFrame(index=pd.Index(markers, dtype=object)),
        obs=obs,
    )

    model = scyan.Scyan(
        sub,
        table,
        batch_key=(batch_key if use_batch else None),
        lr=cfg.lr,
        prior_std=cfg.prior_std,
        hidden_size=cfg.hidden_size,
        n_hidden_layers=cfg.n_hidden_layers,
        n_layers=cfg.n_layers,
        temperature=cfg.temperature,
        max_samples=cfg.max_samples,
    )
    model.fit(max_epochs=cfg.max_epochs, accelerator=_accelerator(cfg.device))
    pred = model.predict(key_added="scyan_pop", log_prob_th=cfg.log_prob_th)

    # per-population soft probabilities -> confidence = row max (drop derived columns)
    proba = model.predict_proba()
    pop_cols = [c for c in names if c in proba.columns]
    probs = pd.DataFrame(proba[pop_cols].to_numpy(), index=adata.obs_names, columns=pop_cols)
    conf = probs.max(axis=1)

    labels = pd.Series(np.asarray(pred.to_numpy(), dtype=object), index=adata.obs_names)
    unknown = labels.isna()
    if cfg.min_confidence > 0:
        unknown = unknown | (conf.to_numpy() < cfg.min_confidence)
    labels = labels.mask(unknown.to_numpy(), "Unknown").astype(str)

    return EngineResult(
        labels=labels.rename("scyan_celltype"),
        confidence=conf.rename("scyan_confidence"),
        probabilities=probs,
        cluster=None,
        uns={
            "engine": "scyan",
            "cell_types": names,
            "n_unassigned": int(unknown.to_numpy().sum()),
            "params": {
                "max_epochs": cfg.max_epochs,
                "lr": cfg.lr,
                "prior_std": cfg.prior_std,
                "temperature": cfg.temperature,
                "log_prob_th": cfg.log_prob_th,
                "min_confidence": cfg.min_confidence,
                "include_batch_covariate": cfg.include_batch_covariate,
                "use_layer": cfg.use_layer,
            },
        },
    )
