"""Signature matrix: the marker -> cell-type prior that drives phenotyping.

A small YAML artifact naming the expected positive (and optional negative) markers
per cell type, with an optional lineage `parent`. This is the single knowledge input
that replaces manually reading N markers across every cluster: both engines
(Astir, FlowSOM) consume it, so their labels are directly comparable.

Schema (``version: 1``)::

    version: 1
    markers: [DAPI, CD3, CD8, ...]        # optional panel restriction; default = all listed
    cell_types:
      T cell:     {positive: [CD3, CD45], negative: [CD19, CD68], parent: Immune}
      CD8 T cell: {positive: [CD3, CD8],  negative: [CD4],        parent: T cell}
      B cell:     [CD19, CD20]            # positive-only shorthand also accepted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellTypeSig:
    name: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    parent: str | None = None


@dataclass(frozen=True)
class SignatureMatrix:
    version: int
    cell_types: dict[str, CellTypeSig]
    markers: tuple[str, ...] | None = None  # optional explicit panel restriction

    # ---- accessors ----

    def cell_type_names(self) -> list[str]:
        return list(self.cell_types.keys())

    def all_markers(self) -> list[str]:
        """Every marker mentioned (positive or negative), first-seen order."""
        out: list[str] = []
        for ct in self.cell_types.values():
            for m in (*ct.positive, *ct.negative):
                if m not in out:
                    out.append(m)
        return out

    # ---- validation ----

    def validate_against(self, var_names: list[str]) -> list[str]:
        """Reconcile the signature with an actual marker panel.

        Warns on signature markers absent from the panel (they are dropped by the
        engines). Raises if any cell type loses *all* of its positive markers.
        Returns the usable marker list (panel markers referenced by the signature,
        in panel order).
        """
        var_set = set(var_names)
        missing = sorted({m for m in self.all_markers() if m not in var_set})
        if missing:
            log.warning("signature markers absent from panel (dropped): %s", ", ".join(missing))
        for name, ct in self.cell_types.items():
            usable_pos = [m for m in ct.positive if m in var_set]
            if ct.positive and not usable_pos:
                raise ValueError(
                    f"cell type {name!r} has no positive markers present in the panel "
                    f"(needs one of: {', '.join(ct.positive)})"
                )
        referenced = set(self.all_markers())
        return [m for m in var_names if m in referenced]

    # ---- engine inputs ----

    def to_marker_dict(self) -> dict[str, list[str]]:
        """cell_type -> positive markers (Astir ``rho`` construction)."""
        return {name: list(ct.positive) for name, ct in self.cell_types.items()}

    def score_matrix(self, markers: list[str]) -> np.ndarray:
        """(K, M) signed matrix aligned to `markers`: +1 positive, -1 negative, else 0.

        Used by FlowSOM metacluster scoring.
        """
        idx = {m: i for i, m in enumerate(markers)}
        mat = np.zeros((len(self.cell_types), len(markers)), dtype=np.float32)
        for k, ct in enumerate(self.cell_types.values()):
            for m in ct.positive:
                if m in idx:
                    mat[k, idx[m]] = 1.0
            for m in ct.negative:
                if m in idx:
                    mat[k, idx[m]] = -1.0
        return mat

    def coarse_map(self) -> dict[str, str]:
        """Map each leaf cell type to its root ancestor via the `parent` chain.

        A `parent` that is not itself a defined cell type (a pure lineage label,
        e.g. ``Immune``) terminates the walk and becomes the coarse label.
        """
        out: dict[str, str] = {}
        for name in self.cell_types:
            cur = name
            seen: set[str] = set()
            while True:
                ct = self.cell_types.get(cur)
                if ct is None or ct.parent is None or cur in seen:
                    break
                seen.add(cur)
                cur = ct.parent
            out[name] = cur
        return out


def _parse_cell_type(name: str, spec: object) -> CellTypeSig:
    if isinstance(spec, list):  # positive-only shorthand
        positive = [str(m) for m in spec]
        negative: list[str] = []
        parent: object = None
    elif isinstance(spec, dict):
        positive = [str(m) for m in (spec.get("positive") or [])]
        negative = [str(m) for m in (spec.get("negative") or [])]
        parent = spec.get("parent")
    else:
        raise ValueError(f"cell type {name!r} must be a list or mapping, got {type(spec).__name__}")
    if not positive:
        raise ValueError(f"cell type {name!r} has no positive markers")
    return CellTypeSig(
        name=name,
        positive=tuple(positive),
        negative=tuple(negative),
        parent=str(parent) if parent else None,
    )


def load_signature(path: str | Path) -> SignatureMatrix:
    """Load and validate a signature-matrix YAML."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    raw_types = data.get("cell_types")
    if not isinstance(raw_types, dict) or not raw_types:
        raise ValueError(f"{path}: a non-empty 'cell_types' mapping is required")
    cell_types = {str(name): _parse_cell_type(str(name), spec) for name, spec in raw_types.items()}
    markers = data.get("markers")
    markers_t = tuple(str(m) for m in markers) if markers else None
    return SignatureMatrix(version=int(data.get("version", 1)), cell_types=cell_types, markers=markers_t)
