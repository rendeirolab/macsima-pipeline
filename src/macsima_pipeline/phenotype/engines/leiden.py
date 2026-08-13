"""Leiden engine: graph clustering (scanpy, ``flavor="igraph"``), then labeling.

Builds a kNN graph on the arcsinh + per-marker z-scored layer (``cfg.use_layer``),
runs Leiden community detection, then auto-labels each cluster against the signature
matrix (``label_clusters``) so labels are directly comparable to scyan's.

Input: the normalized z-scored layer (``cfg.use_layer``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..normalize import _to_dense
from ..signature import SignatureMatrix
from ._labeling import label_clusters
from .base import EngineResult

log = logging.getLogger(__name__)


def _leiden_clusters(z: np.ndarray, markers: list[str] | None, cfg) -> np.ndarray:
    """kNN graph + Leiden (igraph flavor) on `z`; cluster id per cell.

    `z` is the z-scored marker matrix by default, or a batch-corrected embedding when
    `cfg.use_embedding` is set (see `run_leiden`). `markers` names the columns when they
    are markers, and is None for an embedding.
    """
    import anndata as ad
    import scanpy as sc

    var = pd.DataFrame(index=pd.Index(markers, dtype=object)) if markers is not None else None
    sub = ad.AnnData(X=np.ascontiguousarray(z, dtype=np.float32), var=var)
    sc.pp.neighbors(sub, n_neighbors=cfg.n_neighbors, use_rep="X", random_state=cfg.random_seed)
    sc.tl.leiden(
        sub,
        resolution=cfg.resolution,
        flavor="igraph",
        n_iterations=cfg.n_iterations,
        directed=False,
        random_state=cfg.random_seed,
    )
    return sub.obs["leiden"].astype(int).to_numpy()


def run_leiden(adata, sig: SignatureMatrix, cfg, batch_key: str | None = None) -> EngineResult:
    """Cluster the z-scored layer with Leiden and label clusters via the signature."""
    layer = cfg.use_layer
    source = adata.layers[layer] if layer and layer in adata.layers else adata.X
    z = _to_dense(source)
    markers = list(adata.var_names)

    # Cluster on the batch-corrected embedding when one is available, but always LABEL from
    # the uncorrected marker matrix `z`: correcting the intensities the signature reads is
    # what mislabelled keratin-high host epithelium as Fibroblast CAF.
    emb_key = cfg.use_embedding
    if emb_key and emb_key in adata.obsm:
        graph_space = _to_dense(adata.obsm[emb_key])
        graph_cols = None
        log.info("leiden: clustering on obsm[%r] (%d dims), labeling on %r markers",
                 emb_key, graph_space.shape[1], layer)
    else:
        if emb_key:
            log.warning("leiden use_embedding=%r absent from obsm; clustering on %r", emb_key, layer)
        graph_space, graph_cols = z, markers

    cluster_ids = _leiden_clusters(graph_space, graph_cols, cfg)
    labels, conf, label_map, scores = label_clusters(
        z, cluster_ids, sig.score_matrix(markers), sig.cell_type_names(), cfg.tau
    )
    idx = adata.obs_names
    return EngineResult(
        labels=pd.Series(labels, index=idx, name="leiden_celltype"),
        confidence=pd.Series(conf, index=idx, name="leiden_confidence"),
        probabilities=None,
        cluster=pd.Series(cluster_ids.astype(str), index=idx, name="leiden"),
        uns={
            "engine": "leiden",
            "cell_types": sig.cell_type_names(),
            "n_clusters": int(np.unique(cluster_ids).size),
            "label_map": {str(k): v for k, v in label_map.items()},
            "cluster_scores": scores.astype(np.float32),
            "params": {
                "n_neighbors": cfg.n_neighbors,
                "resolution": cfg.resolution,
                "n_iterations": cfg.n_iterations,
                "tau": cfg.tau,
                "random_seed": cfg.random_seed,
                "use_layer": cfg.use_layer,
                "use_embedding": cfg.use_embedding,
                "clustered_on": ("embedding:" + emb_key) if graph_cols is None else "markers",
            },
        },
    )
