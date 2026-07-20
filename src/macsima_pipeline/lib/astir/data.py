"""Data prep for the Astir model: normalization, design matrix, parameter inits.

The Astir model consumes RAW intensities and derives its own normalization here:
  Y = winsorize(arcsinh(raw / cofactor))   -- the likelihood target (arcsinh units)
  X = z-score(Y)                            -- the recognition-network input only
"""

from __future__ import annotations

import numpy as np


def normalize_for_astir(
    raw: np.ndarray,
    cofactor: float,
    winsorize: tuple[float, float] = (0.0, 99.9),
) -> np.ndarray:
    """arcsinh(raw / cofactor) then per-feature winsorization -> Y (arcsinh units)."""
    y = np.arcsinh(np.asarray(raw, dtype=np.float64) / max(float(cofactor), 1e-12))
    lo, hi = winsorize
    if hi is not None and hi < 100:
        y = np.minimum(y, np.percentile(y, hi, axis=0))
    if lo is not None and lo > 0:
        y = np.maximum(y, np.percentile(y, lo, axis=0))
    return y.astype(np.float32)


def zscore(y: np.ndarray) -> np.ndarray:
    """Per-feature standardization (recognition-net input)."""
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std <= 0, 1.0, std)
    return ((y - mean) / std).astype(np.float32)


def build_design(
    n: int,
    batch: np.ndarray | None = None,
    include_batch: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Design matrix for the per-cell baseline.

    Default: a single all-ones intercept column (P=1). With a batch vector and
    ``include_batch``, a full one-hot over batches (no shared intercept), so `mu`
    becomes a per-batch baseline that absorbs ROI-level background shifts.
    """
    if batch is None or not include_batch:
        return np.ones((n, 1), dtype=np.float32), ["intercept"]
    codes = np.asarray(batch)
    uniq = list(dict.fromkeys(codes.tolist()))  # stable order, deterministic
    design = np.zeros((n, len(uniq)), dtype=np.float32)
    pos = {u: j for j, u in enumerate(uniq)}
    for i, c in enumerate(codes.tolist()):
        design[i, pos[c]] = 1.0
    return design, [str(u) for u in uniq]


def mu_sigma_init(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data-driven inits: mu = log(mean Y) so exp(mu) ~ background; log_sigma = log(std Y)."""
    mean = np.clip(y.mean(axis=0), 1e-6, None)
    std = np.clip(y.std(axis=0), 1e-6, None)
    return np.log(mean).astype(np.float32), np.log(std).astype(np.float32)
