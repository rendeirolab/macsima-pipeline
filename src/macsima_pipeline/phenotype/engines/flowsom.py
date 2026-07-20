"""FlowSOM engine: SOM + consensus metaclustering, then signature-based labeling.

Uses the saeyslab ``flowsom`` package (peer-reviewed; Numba SOM + consensus
metaclustering) for reproducible metaclusters — more stable than choosing a Leiden
resolution. Metaclusters are then auto-labeled against the signature matrix, so the
labels are directly comparable to Astir's.

Input: the arcsinh + per-marker z-scored layer (``cfg.use_layer``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..normalize import _to_dense
from ..signature import SignatureMatrix
from .base import EngineResult

log = logging.getLogger(__name__)


def _label_metaclusters(
    z: np.ndarray,
    cluster_ids: np.ndarray,
    score_matrix: np.ndarray,
    names: list[str],
    tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], np.ndarray]:
    """Score each metacluster's mean z-profile against the signature and label it.

    score(m, t) = mean_{g in pos_t} zbar[m, g] - mean_{g in neg_t} zbar[m, g]

    Returns (per-cell labels, per-cell confidence, {metacluster: label}, score matrix).
    Pure numpy — testable without the flowsom package.
    """
    metas = np.unique(cluster_ids)
    pos = (score_matrix > 0).astype(np.float64)  # (K, M)
    neg = (score_matrix < 0).astype(np.float64)
    npos = np.maximum(pos.sum(axis=1), 1.0)
    nneg = neg.sum(axis=1)
    nneg_safe = np.where(nneg <= 0, 1.0, nneg)

    zbar = np.vstack([z[cluster_ids == m].mean(axis=0) for m in metas])  # (n_meta, M)
    pos_term = (zbar @ pos.T) / npos[None, :]
    neg_term = (zbar @ neg.T) / nneg_safe[None, :]
    neg_term = np.where(nneg[None, :] > 0, neg_term, 0.0)
    scores = pos_term - neg_term  # (n_meta, K)

    best = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    ex = np.exp(scores - scores.max(axis=1, keepdims=True))
    conf_meta = (ex / ex.sum(axis=1, keepdims=True)).max(axis=1)

    label_map: dict[int, str] = {}
    conf_map: dict[int, float] = {}
    for i, m in enumerate(metas):
        label_map[int(m)] = names[best[i]] if best_score[i] >= tau else "Unknown"
        conf_map[int(m)] = float(conf_meta[i])

    labels = np.array([label_map[int(m)] for m in cluster_ids], dtype=object)
    conf = np.array([conf_map[int(m)] for m in cluster_ids], dtype=np.float64)
    return labels, conf, label_map, scores


def _fit_flowsom(matrix: np.ndarray, markers: list[str], cfg) -> np.ndarray:
    """Fit FlowSOM on (optionally subsampled) data; return a metacluster id per cell.

    Trains the SOM + consensus metaclustering on up to ``cfg.train_subsample`` cells,
    then assigns every cell to a metacluster via the fitted model (``model.predict``).

    Targets FlowSOM_Python 0.2.x: ``FlowSOM(inp, n_clusters, cols_to_use=, xdim=,
    ydim=, rlen=, seed=)``; per-cell metaclusters live in
    ``get_cell_data().obs['metaclustering']``. Requires pandas < 3 (FlowSOM 0.2.2
    trips over copy-on-write's read-only arrays otherwise).
    """
    import anndata as ad
    import flowsom as fs

    try:  # keep FlowSOM's verbose loguru output out of pipeline logs
        from loguru import logger as _loguru

        _loguru.disable("flowsom")
    except Exception:  # noqa: BLE001
        pass

    n = matrix.shape[0]
    rng = np.random.default_rng(cfg.random_seed)
    subsample = cfg.train_subsample and n > cfg.train_subsample
    train_idx = np.sort(rng.choice(n, size=cfg.train_subsample, replace=False)) if subsample else None

    var = pd.DataFrame(index=pd.Index(markers, dtype=object))
    train_x = matrix if train_idx is None else matrix[train_idx]
    train_ad = ad.AnnData(X=np.ascontiguousarray(train_x, dtype=np.float32), var=var.copy())
    fsom = fs.FlowSOM(
        train_ad,
        cfg.n_metaclusters,
        cols_to_use=list(markers),
        xdim=cfg.grid_size[0],
        ydim=cfg.grid_size[1],
        rlen=cfg.som_iterations,
        seed=cfg.random_seed,
    )
    if train_idx is None:
        meta = np.asarray(fsom.get_cell_data().obs["metaclustering"])
    else:
        # model.predict expects the cols_to_use columns (== all markers here), in order
        meta = np.asarray(fsom.model.predict(np.ascontiguousarray(matrix, dtype=np.float32)))
    return meta.astype(int)


def run_flowsom(adata, sig: SignatureMatrix, cfg, batch_key: str | None = None) -> EngineResult:
    """Run FlowSOM on the z-scored layer and label metaclusters via the signature."""
    layer = cfg.use_layer
    source = adata.layers[layer] if layer and layer in adata.layers else adata.X
    z = _to_dense(source)
    markers = list(adata.var_names)

    cluster_ids = _fit_flowsom(z, markers, cfg)
    labels, conf, label_map, scores = _label_metaclusters(
        z, cluster_ids, sig.score_matrix(markers), sig.cell_type_names(), cfg.tau
    )
    idx = adata.obs_names
    return EngineResult(
        labels=pd.Series(labels, index=idx, name="flowsom_celltype"),
        confidence=pd.Series(conf, index=idx, name="flowsom_confidence"),
        probabilities=None,
        cluster=pd.Series(cluster_ids.astype(str), index=idx, name="flowsom"),
        uns={
            "engine": "flowsom",
            "cell_types": sig.cell_type_names(),
            "label_map": {str(k): v for k, v in label_map.items()},
            "metacluster_scores": scores.astype(np.float32),
            "params": {
                "grid_size": list(cfg.grid_size),
                "n_metaclusters": cfg.n_metaclusters,
                "train_subsample": cfg.train_subsample,
                "tau": cfg.tau,
                "random_seed": cfg.random_seed,
                "use_layer": cfg.use_layer,
            },
        },
    )
