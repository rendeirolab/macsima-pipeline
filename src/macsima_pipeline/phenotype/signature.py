"""Signature matrix: the marker -> cell-type prior that drives phenotyping.

A scyan-native CSV table. Rows are cell-type populations, columns are markers, and
each value is the expected relative expression in ``[-1, 1]`` (blank / NA = unknown,
i.e. not informative for that population). One reserved column, ``parent``, gives each
population a coarse/lineage label used for ``cell_type_coarse``. Both engines (scyan,
Leiden) consume this table, so their labels are directly comparable.

``#`` comment lines are allowed anywhere (read with pandas ``comment="#"``), so a
signature file can carry its own documentation.

Schema (CSV)::

    # optional comments (panel, rationale, caveats ...)
    population,parent,DAPI,CD3,CD45,CD8,...
    T cell,Immune,,1,1,,...
    CD8 T cell,T cell,,1,,1,...

Values are numeric in ``[-1, 1]`` or blank/NA. scyan uses them directly as prior modes
(NA is masked / "don't care"); Leiden uses only their sign (``> 0`` positive, ``< 0``
negative). A graded value such as ``0.5`` is a valid, weaker prior than ``1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Reserved (non-marker) column naming each population's coarse/lineage label.
PARENT_COLUMN = "parent"


@dataclass(frozen=True)
class SignatureMatrix:
    """Population x marker prior table plus per-population coarse/lineage parents."""

    table: pd.DataFrame  # index = populations, columns = markers, values in [-1, 1] or NaN
    parents: dict[str, str | None]  # population -> coarse/lineage label (or None)

    # ---- accessors ----

    def cell_type_names(self) -> list[str]:
        return [str(n) for n in self.table.index]

    def all_markers(self) -> list[str]:
        return [str(m) for m in self.table.columns]

    # ---- validation ----

    def validate_against(self, var_names: list[str]) -> list[str]:
        """Reconcile the signature with an actual marker panel.

        Warns on signature markers absent from the panel (they are dropped by the
        engines). Raises if any population has no informative (non-NaN) marker present
        in the panel. Returns the usable marker list (signature markers present in the
        panel, in panel order).
        """
        var_set = set(var_names)
        missing = sorted(m for m in self.all_markers() if m not in var_set)
        if missing:
            log.warning("signature markers absent from panel (dropped): %s", ", ".join(missing))
        present = [m for m in self.all_markers() if m in var_set]
        sub = self.table[present] if present else self.table.iloc[:, :0]
        for name in self.table.index:
            if present and sub.loc[name].notna().to_numpy().sum() > 0:
                continue
            raise ValueError(
                f"population {name!r} has no informative markers present in the panel "
                f"(signature markers: {', '.join(self.all_markers())})"
            )
        return [m for m in var_names if m in var_set & set(self.all_markers())]

    # ---- engine inputs ----

    def score_matrix(self, markers: list[str]) -> np.ndarray:
        """(K, M) prior matrix aligned to ``markers``: values in [-1, 1], NaN = unknown.

        Feeds the scyan knowledge table directly (values used as-is) and Leiden cluster
        scoring (which uses only the sign). Markers absent from the signature are NaN.
        Row order matches :meth:`cell_type_names`.
        """
        return self.table.reindex(columns=markers).to_numpy(dtype=np.float64)

    def coarse_map(self) -> dict[str, str]:
        """Map each population to its root ancestor via the ``parent`` chain.

        A ``parent`` that is not itself a defined population (a pure lineage label,
        e.g. ``Immune``) terminates the walk and becomes the coarse label. A population
        with no parent maps to itself.
        """
        out: dict[str, str] = {}
        for name in self.cell_type_names():
            cur = name
            seen: set[str] = set()
            while True:
                parent = self.parents.get(cur)
                if not parent or cur in seen:
                    break
                seen.add(cur)
                cur = parent
            out[name] = cur
        return out


def load_signature(path: str | Path) -> SignatureMatrix:
    """Load and validate a scyan-native signature CSV (see module docstring)."""
    path = Path(path)
    df = pd.read_csv(path, index_col=0, comment="#", skipinitialspace=True)

    # Clean the population index; drop fully-blank rows.
    df.index = [str(n).strip() for n in df.index]
    df = df[[bool(n) and n.lower() != "nan" for n in df.index]]
    if df.empty:
        raise ValueError(f"{path}: no populations found (need at least one data row)")
    if df.index.duplicated().any():
        dupes = sorted(df.index[df.index.duplicated()].unique())
        raise ValueError(f"{path}: duplicate population name(s): {', '.join(dupes)}")

    # Reserved parent column -> coarse/lineage mapping (case-insensitive header match).
    parents: dict[str, str | None] = {name: None for name in df.index}
    parent_cols = [c for c in df.columns if str(c).strip().lower() == PARENT_COLUMN]
    if parent_cols:
        raw = df.pop(parent_cols[0])
        for name, val in raw.items():
            s = "" if pd.isna(val) else str(val).strip()
            parents[str(name)] = s or None

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] == 0:
        raise ValueError(f"{path}: no marker columns found")

    # Coerce to float; validate range [-1, 1] or NaN.
    try:
        table = df.apply(pd.to_numeric).astype(np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{path}: signature values must be numeric in [-1, 1] or blank ({e})") from e
    vals = table.to_numpy()
    finite = vals[~np.isnan(vals)]
    if finite.size == 0:
        raise ValueError(f"{path}: signature has no values (every entry is blank)")
    if finite.min() < -1.0 or finite.max() > 1.0:
        raise ValueError(f"{path}: signature values must lie in [-1, 1] (blank/NA = unknown)")

    return SignatureMatrix(table=table, parents=parents)
