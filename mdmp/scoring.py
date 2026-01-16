"""
Scoring functions for MDM structure learning.

This module implements discount factor selection and log predictive likelihood
computation for MDM model scoring.
"""

from typing import Any, Dict, Optional

import numpy as np

from .dlm import dlm_filter
from .utils import DEFAULT_NBF, build_design_matrix, extract_target_series, get_default_delta


def select_discount_factors(
    data: np.ndarray,
    adj_mat: np.ndarray,
    nbf: int = DEFAULT_NBF,
    delta: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Select discount factors that maximize log predictive likelihood for each node.

    Parameters
    ----------
    data : np.ndarray
        Time series data (T x N).
    adj_mat : np.ndarray
        Adjacency matrix (N x N).
    nbf : int, optional
        Burn-in time point. Default is 15.
    delta : np.ndarray, optional
        Sequence of discount factors. Default is np.arange(0.5, 1.01, 0.01).

    Returns
    -------
    dict
        Dictionary containing:
        - lpldet : Log predictive likelihoods for each delta and node (nd, N)
        - DF_hat : Selected discount factors for each node (N,)
    
    Raises
    ------
    ValueError
        If data and adj_mat dimensions are incompatible.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mdmp.scoring import select_discount_factors
    >>> data = np.random.randn(100, 3)
    >>> adj_mat = np.zeros((3, 3))
    >>> adj_mat[0, 1] = 1
    >>> result = select_discount_factors(data, adj_mat)
    >>> print(result['DF_hat'])
    """
    if delta is None:
        delta = get_default_delta()

    nd = len(delta)
    Nn = data.shape[1]
    lpldet = np.zeros((nd, Nn))

    # Pre-compute design matrices and target series for each node
    # (they don't depend on delta, so we can cache them)
    design_matrices = {}
    target_series = {}
    for i in range(Nn):
        design_matrices[i], _ = build_design_matrix(data, adj_mat, i)
        target_series[i] = extract_target_series(data, i)

    # Evaluate log predictive likelihood for each delta and node
    for k in range(nd):
        for i in range(Nn):
            Ft = design_matrices[i]
            Yt = target_series[i]

            # Run DLM filter
            result = dlm_filter(Yt, Ft.T, delta=delta[k])
            lpldet[k, i] = np.sum(result['lpl'][nbf:])

    # Select best delta for each node (handling NaN values)
    DF_hat = _select_best_deltas(lpldet, delta, Nn)

    return {
        'lpldet': lpldet,
        'DF_hat': DF_hat
    }


def _select_best_deltas(
    lpldet: np.ndarray,
    delta: np.ndarray,
    num_nodes: int,
    default_delta: float = 0.9
) -> np.ndarray:
    """
    Select the best discount factor for each node based on log predictive likelihood.
    
    Parameters
    ----------
    lpldet : np.ndarray
        Log predictive likelihoods (nd, N).
    delta : np.ndarray
        Array of discount factors.
    num_nodes : int
        Number of nodes.
    default_delta : float, optional
        Default discount factor if all values are NaN. Default is 0.9.
    
    Returns
    -------
    np.ndarray
        Selected discount factors for each node (N,).
    """
    DF_hat = np.zeros(num_nodes)
    for i in range(num_nodes):
        valid_indices = ~np.isnan(lpldet[:, i])
        if np.any(valid_indices):
            max_idx = np.nanargmax(lpldet[:, i])
            DF_hat[i] = delta[max_idx]
        else:
            DF_hat[i] = default_delta
    return DF_hat


def compute_logpl(
    data: np.ndarray,
    adj_mat: np.ndarray,
    delta: float,
    node_idx: int,
    nbf: int = DEFAULT_NBF
) -> float:
    """
    Compute negative log predictive likelihood for a given node and discount factor.

    Used in optimization for structure learning.

    Parameters
    ----------
    data : np.ndarray
        Time series data (T x N).
    adj_mat : np.ndarray
        Adjacency matrix (N x N).
    delta : float
        Discount factor (must be between 0 and 1).
    node_idx : int
        Index of target node.
    nbf : int, optional
        Burn-in time point. Default is 15.

    Returns
    -------
    float
        Negative log predictive likelihood.
    
    Raises
    ------
    ValueError
        If node_idx is out of bounds or adj_mat dimensions don't match data.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mdmp.scoring import compute_logpl
    >>> data = np.random.randn(100, 3)
    >>> adj_mat = np.zeros((3, 3))
    >>> logpl = compute_logpl(data, adj_mat, delta=0.9, node_idx=0)
    >>> print(logpl)
    """
    if delta > 1 or delta < 0:
        return np.inf

    # Validate inputs
    if node_idx < 0 or node_idx >= data.shape[1]:
        raise ValueError(
            f"node_idx ({node_idx}) must be between 0 and {data.shape[1] - 1}"
        )

    if adj_mat.shape[0] != data.shape[1] or adj_mat.shape[1] != data.shape[1]:
        raise ValueError(
            f"adj_mat shape {adj_mat.shape} does not match data shape {data.shape}"
        )

    # Build design matrix and extract target series
    Ft, _ = build_design_matrix(data, adj_mat, node_idx)
    Yt = extract_target_series(data, node_idx)

    # Run DLM filter and compute log predictive likelihood
    result = dlm_filter(Yt, Ft.T, delta=delta)
    lpldet = np.sum(result['lpl'][nbf:])

    return -lpldet

