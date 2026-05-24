"""
Node-level dispatch for parallel and serial MDM computations.

Provides three plain functions — filter_all_nodes, smooth_all_nodes, evaluate_lpl —
that apply the appropriate worker function to every node either serially or in parallel.
This replaces the processing/ class hierarchy (NodeProcessor ABC, factory, serial/parallel
subclasses, and the three domain-specific processor classes).
"""

from typing import Dict, List, Optional

import numpy as np

from .parallel import (
    _get_n_jobs,
    _worker_filter_node,
    _worker_select_delta_node,
    _worker_smooth_node,
)
from .progress import get_progress_bar, process_map_with_progress


def _parallel_map(worker_fn, args_list, n_jobs, desc, verbose=False):
    """Apply worker_fn to each item in args_list serially or in parallel."""
    n = _get_n_jobs(n_jobs, default=1)
    if n == 1:
        pbar = get_progress_bar(len(args_list), desc=desc, disable=not verbose)
        results = []
        try:
            for args in args_list:
                results.append(worker_fn(args))
                if hasattr(pbar, "update"):
                    pbar.update(1)
        finally:
            if hasattr(pbar, "close"):
                pbar.close()
        return results
    return process_map_with_progress(
        worker_fn, args_list, max_workers=n, desc=desc, disable=not verbose
    )


def filter_all_nodes(
    data: np.ndarray,
    adj_mat: np.ndarray,
    DF_hat: np.ndarray,
    node_names: List[str],
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run dlm_filter for every node and return aggregated results.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
    adj_mat : np.ndarray, shape (N, N)
    DF_hat : np.ndarray, shape (N,) — selected discount factor per node
    node_names : list of str
    n_jobs : int or None — None/1 for serial, -1 for all cores, >1 for that many workers
    verbose : bool — show progress bar

    Returns
    -------
    dict with keys mt, Ct, Rt, nt, dt, ft, Qt, ets, lpl, row_names — each a dict keyed by node index
    """
    Nn = data.shape[1]
    args_list = [(i, data, adj_mat, DF_hat, node_names) for i in range(Nn)]
    results = _parallel_map(_worker_filter_node, args_list, n_jobs, "Filtering nodes", verbose)

    mt, Ct, Rt, nt, dt, ft, Qt, ets, lpl, row_names = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for i, result_dict, param_names in results:
        mt[i] = result_dict["mt"]
        Ct[i] = result_dict["Ct"]
        Rt[i] = result_dict["Rt"]
        nt[i] = result_dict["nt"]
        dt[i] = result_dict["dt"]
        ft[i] = result_dict["ft"]
        Qt[i] = result_dict["Qt"]
        ets[i] = result_dict["ets"]
        lpl[i] = result_dict["lpl"]
        row_names[i] = param_names[: mt[i].shape[0]] if mt[i].ndim == 2 else param_names[:1]

    return {
        "mt": mt, "Ct": Ct, "Rt": Rt, "nt": nt, "dt": dt,
        "ft": ft, "Qt": Qt, "ets": ets, "lpl": lpl, "row_names": row_names,
    }


def smooth_all_nodes(
    mt: Dict[int, np.ndarray],
    Ct: Dict[int, np.ndarray],
    Rt: Dict[int, np.ndarray],
    nt: Dict[int, np.ndarray],
    dt: Dict[int, np.ndarray],
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Run dlm_smooth for every node and return aggregated results.

    Parameters
    ----------
    mt, Ct, Rt, nt, dt : dict — filtered posterior parameters, keyed by node index
    n_jobs : int or None
    verbose : bool

    Returns
    -------
    dict with keys smt, sCt, SE — each a dict keyed by node index
    """
    Nn = len(mt)
    args_list = [(i, mt, Ct, Rt, nt, dt) for i in range(Nn)]
    results = _parallel_map(_worker_smooth_node, args_list, n_jobs, "Smoothing nodes", verbose)

    smt_out, sCt_out, SE_out = {}, {}, {}
    for i, result_dict in results:
        smt_out[i] = result_dict["smt"]
        sCt_out[i] = result_dict["sCt"]
        SE_out[i] = result_dict["SE"]

    return {"smt": smt_out, "sCt": sCt_out, "SE": SE_out}


def evaluate_lpl(
    delta: np.ndarray,
    design_matrices: Dict[int, np.ndarray],
    target_series: Dict[int, np.ndarray],
    nbf: int,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Evaluate log predictive likelihood for all (delta, node) combinations.

    Parameters
    ----------
    delta : np.ndarray, shape (nd,) — discount factor grid
    design_matrices : dict — node index → design matrix (T, p)
    target_series : dict — node index → target array (T,)
    nbf : int — burn-in index
    n_jobs : int or None
    verbose : bool

    Returns
    -------
    np.ndarray, shape (nd, N) — lpl summed from nbf for each (delta, node)
    """
    nd = len(delta)
    Nn = len(design_matrices)
    args_list = [
        (k, i, target_series[i], design_matrices[i], delta[k], nbf)
        for k in range(nd)
        for i in range(Nn)
    ]
    results = _parallel_map(
        _worker_select_delta_node, args_list, n_jobs, "Selecting discount factors", verbose
    )
    lpldet = np.zeros((nd, Nn))
    for k, i, lpl_sum in results:
        lpldet[k, i] = lpl_sum
    return lpldet
