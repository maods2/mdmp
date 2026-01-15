"""
Scoring functions for MDM structure learning.

This module implements discount factor selection and log predictive likelihood
computation for MDM model scoring.
"""

import numpy as np
from typing import Optional, Dict, Any
from .dlm import dlm_filter
from .utils import (
    build_design_matrix,
    extract_target_series,
    get_default_delta,
    DEFAULT_NBF
)


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
    """
    if delta is None:
        delta = get_default_delta()

    nd = len(delta)
    Nn = data.shape[1]
    lpldet = np.zeros((nd, Nn))

    # Evaluate log predictive likelihood for each delta and node
    for k in range(nd):
        for i in range(Nn):
            Ft, _ = build_design_matrix(data, adj_mat, i)
            Yt = extract_target_series(data, i)
            
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
    """
    if delta > 1 or delta < 0:
        return np.inf

    # Build design matrix and extract target series
    Ft, _ = build_design_matrix(data, adj_mat, node_idx)
    Yt = extract_target_series(data, node_idx)

    # Run DLM filter and compute log predictive likelihood
    result = dlm_filter(Yt, Ft.T, delta=delta)
    lpldet = np.sum(result['lpl'][nbf:])

    return -lpldet

