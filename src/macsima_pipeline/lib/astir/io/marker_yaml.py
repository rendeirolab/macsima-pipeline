"""Build the Astir marker matrix (rho) from a positive-marker signature."""

from __future__ import annotations

import numpy as np

OTHER_CLASS = "Other"


def build_rho(
    signature: dict[str, list[str]],
    marker_names: list[str],
    *,
    include_other: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Construct the binary marker matrix.

    Parameters
    ----------
    signature
        cell_type -> list of positive marker names.
    marker_names
        The feature panel (columns of the expression matrix), in order.
    include_other
        Append an all-zero "Other" class column (a cell matching no type's markers).

    Returns
    -------
    (rho, class_names)
        rho: float32 array of shape (G, C[+1]); rho[g, c] == 1 iff marker g is a
        positive marker of class c. The optional trailing "Other" column is all zeros.
        class_names: the class labels, in column order (with "Other" last if included).
    """
    class_names = list(signature.keys())
    idx = {m: i for i, m in enumerate(marker_names)}
    n_cols = len(class_names) + (1 if include_other else 0)
    rho = np.zeros((len(marker_names), n_cols), dtype=np.float32)
    for c, name in enumerate(class_names):
        for marker in signature[name]:
            j = idx.get(marker)
            if j is not None:
                rho[j, c] = 1.0
    if include_other:
        class_names = [*class_names, OTHER_CLASS]
    return rho, class_names
