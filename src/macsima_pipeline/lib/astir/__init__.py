"""Clean-room implementation of the Astir cell-type model.

Original work implementing the model of Geuenich et al. (Cell Systems 2021) from the
published equations; NOT derived from the GPL-2.0 `astir` package. See NOTICE.
"""

from __future__ import annotations

from .model import AstirCellType, fit

__all__ = ["AstirCellType", "fit"]
