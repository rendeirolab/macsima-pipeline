"""Astir engine adapter: wraps the clean-room `lib.astir` for the phenotype stage.

Feeds RAW intensities (the model normalizes internally) and returns per-cell type
probabilities + hard labels, with low-confidence cells marked "Unknown".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..normalize import _to_dense
from ..signature import SignatureMatrix
from .base import EngineResult

log = logging.getLogger(__name__)


def run_astir(adata, sig: SignatureMatrix, cfg, batch_key: str | None = None) -> EngineResult:
    from ...lib import astir as astir_lib

    layer = cfg.use_layer
    if layer and layer in adata.layers:
        source = adata.layers[layer]
    else:
        log.warning("astir use_layer=%r absent; falling back to X (must be RAW intensities)", layer)
        source = adata.X
    matrix = _to_dense(source)
    marker_names = list(adata.var_names)

    batch = None
    if cfg.include_batch_covariate and batch_key and batch_key in adata.obs.columns:
        batch = np.asarray(adata.obs[batch_key].to_numpy())

    model = astir_lib.fit(
        matrix,
        marker_names,
        sig.to_marker_dict(),
        batch=batch,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        n_init=cfg.n_init,
        random_seed=cfg.random_seed,
        device=cfg.device,
        cofactor=cfg.cofactor,
        winsorize=tuple(cfg.winsorize),
        include_batch=cfg.include_batch_covariate,
        precision=cfg.precision,
    )

    probs = pd.DataFrame(model.predict_proba(), index=adata.obs_names, columns=model.classes_)
    conf = probs.max(axis=1)
    labels = probs.idxmax(axis=1).where(conf >= cfg.min_confidence, other="Unknown")
    return EngineResult(
        labels=labels.rename("astir_celltype"),
        confidence=conf.rename("astir_confidence"),
        probabilities=probs,
        cluster=None,
        uns={
            "engine": "astir",
            "cell_types": list(model.classes_),
            "losses": [float(x) for x in getattr(model, "losses", [])],
            "converged": bool(getattr(model, "converged", False)),
            "params": {
                "cofactor": cfg.cofactor,
                "max_epochs": cfg.max_epochs,
                "n_init": cfg.n_init,
                "min_confidence": cfg.min_confidence,
                "include_batch_covariate": cfg.include_batch_covariate,
                "random_seed": cfg.random_seed,
                "use_layer": cfg.use_layer,
            },
        },
    )
