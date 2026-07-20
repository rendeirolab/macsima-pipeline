"""Signature-based labeling of unsupervised clusters.

Shared by clustering engines (e.g. Leiden): score each cluster's mean z-profile
against the signature matrix and assign the enriched cell type, so cluster labels
are directly comparable to the probabilistic engine (scyan).
"""

from __future__ import annotations

import numpy as np


def label_clusters(
    z: np.ndarray,
    cluster_ids: np.ndarray,
    score_matrix: np.ndarray,
    names: list[str],
    tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], np.ndarray]:
    """Score each cluster's mean z-profile against the signature and label it.

    score(c, t) = mean_{g in pos_t} zbar[c, g] - mean_{g in neg_t} zbar[c, g]

    Returns (per-cell labels, per-cell confidence, {cluster: label}, score matrix).
    Pure numpy — testable without any clustering package.
    """
    clusters = np.unique(cluster_ids)
    pos = (score_matrix > 0).astype(np.float64)  # (K, M)
    neg = (score_matrix < 0).astype(np.float64)
    npos = np.maximum(pos.sum(axis=1), 1.0)
    nneg = neg.sum(axis=1)
    nneg_safe = np.where(nneg <= 0, 1.0, nneg)

    zbar = np.vstack([z[cluster_ids == c].mean(axis=0) for c in clusters])  # (n_clusters, M)
    pos_term = (zbar @ pos.T) / npos[None, :]
    neg_term = (zbar @ neg.T) / nneg_safe[None, :]
    neg_term = np.where(nneg[None, :] > 0, neg_term, 0.0)
    scores = pos_term - neg_term  # (n_clusters, K)

    best = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    ex = np.exp(scores - scores.max(axis=1, keepdims=True))
    conf_cluster = (ex / ex.sum(axis=1, keepdims=True)).max(axis=1)

    label_map: dict[int, str] = {}
    conf_map: dict[int, float] = {}
    for i, c in enumerate(clusters):
        label_map[int(c)] = names[best[i]] if best_score[i] >= tau else "Unknown"
        conf_map[int(c)] = float(conf_cluster[i])

    labels = np.array([label_map[int(c)] for c in cluster_ids], dtype=object)
    conf = np.array([conf_map[int(c)] for c in cluster_ids], dtype=np.float64)
    return labels, conf, label_map, scores
