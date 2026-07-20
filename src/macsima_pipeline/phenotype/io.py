"""AnnData IO for the phenotype stage: read input, atomic write, resume check."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


def read_cells(cfg, bg: bool | None = None):
    """Read the preprocess-stage AnnData for a background variant."""
    import anndata as ad

    return ad.read_h5ad(cfg.h5ad_path(bg))


def write_cells_atomic(adata, dest: str | Path) -> Path:
    """Write an h5ad via a temp file + `os.replace` (atomic on the same filesystem)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def phenotype_done(cfg, bg: bool | None = None) -> bool:
    """True if a phenotyped h5ad already exists and carries `obs['cell_type']`.

    Peeks the HDF5 structure with h5py to avoid a full AnnData read on resume.
    """
    dest = cfg.phenotype_h5ad_path(bg)
    if not dest.is_file() or dest.stat().st_size == 0:
        return False
    try:
        import h5py

        with h5py.File(dest, "r") as f:
            obs = f.get("obs")
            return bool(obs is not None and "cell_type" in obs)
    except Exception as e:  # noqa: BLE001 - resume check must never crash the stage
        log.warning("could not inspect %s for resume (%s); will recompute", dest, e)
        return False
