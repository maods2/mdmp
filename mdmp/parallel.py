"""
Parallel processing utilities for MDM operations.

This module provides worker functions and utilities for parallelizing
MDM operations using multiprocessing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dlm import dlm_filter, dlm_smooth
from .utils import (
    build_design_matrix,
    build_parameter_names,
    extract_target_series,
)


def _get_n_jobs(n_jobs: Optional[int], default: int = -1) -> int:
    """
    Determine the number of parallel jobs to use.

    Parameters
    ----------
    n_jobs : int or None
        Number of jobs. If None, returns default.
        If -1, uses all available CPU cores.
        If 1, uses serial processing.
        If > 1, uses that many parallel workers.
    default : int, optional
        Default value if n_jobs is None. Default is -1.

    Returns
    -------
    int
        Number of jobs to use. Returns 1 for serial processing,
        or a positive integer for parallel processing.
    """
    if n_jobs is None:
        n_jobs = default

    if n_jobs == -1:
        return os.cpu_count() or 1
    elif n_jobs < 1:
        raise ValueError(f"n_jobs must be -1, None, or a positive integer, got {n_jobs}")
    else:
        return n_jobs


def _worker_select_delta_node(args: Tuple[int, int, np.ndarray, np.ndarray, float, int]) -> Tuple[int, int, float]:
    """
    Worker function to process a single (delta, node) combination.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - k : int, delta index
        - i : int, node index
        - Yt : np.ndarray, target time series
        - Ft : np.ndarray, design matrix (T x p)
        - delta_k : float, discount factor value
        - nbf : int, burn-in time point

    Returns
    -------
    tuple
        Tuple containing (k, i, lpl_sum) where lpl_sum is the
        sum of log predictive likelihoods from nbf onwards.
    """
    k, i, Yt, Ft, delta_k, nbf = args

    # Run DLM filter
    result = dlm_filter(Yt, Ft.T, delta=delta_k)

    # Sum log predictive likelihood from nbf onwards
    lpl_sum = np.sum(result['lpl'][nbf:])

    return (k, i, lpl_sum)


def _worker_filter_node(
    args: Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[str]]
) -> Tuple[int, Dict[str, Any], List[str]]:
    """
    Worker function to filter a single node.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - i : int, node index
        - data : np.ndarray, time series data (T x N)
        - adj_mat : np.ndarray, adjacency matrix (N x N)
        - DF_hat : np.ndarray, discount factors for each node (N,)
        - node_names : list of str, node names

    Returns
    -------
    tuple
        Tuple containing (i, result_dict, param_names) where:
        - i : node index
        - result_dict : dict with DLM filter results
        - param_names : list of parameter names for this node
    """
    i, data, adj_mat, DF_hat, node_names = args

    # Build design matrix and extract target series
    Ft, parent_list = build_design_matrix(data, adj_mat, i)
    Yt = extract_target_series(data, i)

    # Run DLM filter
    result = dlm_filter(Yt, Ft.T, delta=DF_hat[i])

    # Build parameter names
    param_names = build_parameter_names(i, adj_mat, node_names)

    # Prepare result dictionary
    result_dict = {
        'mt': result['mt'],
        'Ct': result['Ct'],
        'Rt': result['Rt'],
        'nt': result['nt'],
        'dt': result['dt'],
        'ft': result['ft'],
        'Qt': result['Qt'],
        'ets': result['ets'],
        'lpl': result['lpl'],
    }

    return (i, result_dict, param_names)


def _worker_smooth_node(
    args: Tuple[int, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function to smooth a single node.

    Parameters
    ----------
    args : tuple
        Tuple containing:
        - i : int, node index
        - mt : dict, filtered posterior means
        - Ct : dict, filtered posterior variances
        - Rt : dict, prior variances
        - nt : dict, precision hyperparameters
        - dt : dict, precision hyperparameters

    Returns
    -------
    tuple
        Tuple containing (i, result_dict) where result_dict contains
        smoothed estimates and standard errors.
    """
    i, mt, Ct, Rt, nt, dt = args

    # Run DLM smooth
    result = dlm_smooth(mt[i], Ct[i], Rt[i], nt[i], dt[i])

    smt = result['smt']
    sCt = result['sCt']

    # Compute standard errors
    from scipy import stats

    if sCt.ndim == 2:  # Single parameter case
        SE = stats.t.ppf(0.975, nt[i][-1]) * np.sqrt(sCt)
    else:  # Multiple parameters
        SE_array = np.zeros((sCt.shape[2], sCt.shape[0]))
        for j in range(sCt.shape[0]):
            SE_array[:, j] = (
                stats.t.ppf(0.975, nt[i][-1]) *
                np.sqrt(sCt[j, j, :])
            )
        import pandas as pd
        col_names = [f"SE_{name}" for name in range(sCt.shape[0])]
        SE = pd.DataFrame(SE_array, columns=col_names)

    result_dict = {
        'smt': smt,
        'sCt': sCt,
        'SE': SE
    }

    return (i, result_dict)
