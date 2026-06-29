"""Fast OME-TIFF readers that honor the existing mcmicro pyramid.

The original viz script rebuilt a multiscale in-memory via `multiscale_spatial_image`
per ROI, then called `.compute()` separately for every (marker, ROI) cell. That
re-reads the file once per cell. mcmicro already writes a pyramid; reading the
right level directly is 8-64x less I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PyramidLevel:
    level: int          # index into series.levels
    shape: tuple[int, ...]  # (C, Y, X) typically


def _level_max_xy(shape: tuple[int, ...]) -> int:
    # last two axes are Y, X for OME-TIFFs from mcmicro
    return max(shape[-2:])


def pick_level(path: Path, target_max_dim: int) -> PyramidLevel:
    """Pick the smallest pyramid level whose max(Y, X) >= target_max_dim, else the smallest available."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        levels = list(series.levels) if hasattr(series, "levels") else [series]
        # Levels are typically ordered finest -> coarsest. We want the coarsest level still >= target.
        # Iterate finest -> coarsest, remember the last one with max_dim >= target.
        chosen = 0
        for i, lvl in enumerate(levels):
            if _level_max_xy(lvl.shape) >= target_max_dim:
                chosen = i
            else:
                break
        shape = tuple(levels[chosen].shape)
    log.debug("pyramid pick: %s -> level=%d shape=%s (target=%d)", path.name, chosen, shape, target_max_dim)
    return PyramidLevel(level=chosen, shape=shape)


def read_channel(path: Path, level: int, channel_ix: int) -> np.ndarray:
    """Read a single channel from a given pyramid level. Returns 2D uint16/float ndarray."""
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        if hasattr(series, "levels"):
            arr = series.levels[level].asarray(key=channel_ix)
        else:
            arr = series.asarray(key=channel_ix)
    # Some OME writers nest (1, Y, X); squeeze leading 1s.
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def percentiles(img: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    p1, p99 = np.percentile(img, [p_low, p_high])
    return float(p1), float(p99)


def clip_norm(img: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip to [p1, p99]. Returns the clipped array (still in original units)."""
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip(img, p1, p99)


def clip_norm_u8(img: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip to [p1, p99] and rescale to uint8 [0, 255].

    Use this instead of `clip_norm` when many panels are composited into a
    single figure: matplotlib's AGG backend rasterizes uint8 directly without
    promoting to float64, which dramatically reduces peak memory.
    """
    if p99 <= p1:
        return np.zeros(img.shape, dtype=np.uint8)
    out = np.clip(img, p1, p99).astype(np.float32, copy=False)
    out -= np.float32(p1)
    out *= np.float32(255.0 / (p99 - p1))
    return out.astype(np.uint8, copy=False)
