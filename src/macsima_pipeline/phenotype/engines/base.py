"""Common result type for phenotyping engines."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EngineResult:
    """Output of one phenotyping engine, indexed by cell (``obs_names``).

    labels        hard cell-type label per cell ("Unknown" when below threshold)
    confidence    per-cell confidence in [0, 1]
    probabilities per-cell × cell-type probabilities (Astir); None for FlowSOM
    cluster       metacluster id per cell (FlowSOM); None for Astir
    uns           engine params + diagnostics (serialized into uns['phenotype'])
    """

    labels: pd.Series
    confidence: pd.Series
    probabilities: pd.DataFrame | None = None
    cluster: pd.Series | None = None
    uns: dict = field(default_factory=dict)
