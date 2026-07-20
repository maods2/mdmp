"""Sparsification helpers for pairwise distance matrices."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


def sparsify_distance_matrix(
    matrix: np.ndarray,
    *,
    knn: Optional[int] = None,
    threshold: Optional[float] = None,
) -> csr_matrix:
    """
    Build a sparse neighbourhood dissimilarity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Dense symmetric distance matrix (zero diagonal).
    knn : int, optional
        Keep each subject's ``k`` nearest neighbours (symmetrised).
    threshold : float, optional
        Drop pairs with distance strictly greater than this value.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    full = np.asarray(matrix, dtype=float)
    s = full.shape[0]
    mask = np.ones_like(full, dtype=bool)
    np.fill_diagonal(mask, False)

    if threshold is not None:
        mask &= full <= threshold

    if knn is not None:
        keep = np.zeros_like(mask)
        for i in range(s):
            order = np.argsort(full[i])
            order = order[order != i][:knn]
            keep[i, order] = True
        keep = keep | keep.T
        mask &= keep

    sparse = np.where(mask, full, 0.0)
    return csr_matrix(sparse)
