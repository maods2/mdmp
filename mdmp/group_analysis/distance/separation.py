"""Pairwise MDM separation matrix computation."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from ..._node_dispatch import _parallel_map
from ...scoring import select_discount_factors
from ...structure import StructureLearner
from ...utils import get_default_delta
from .coercion import coerce_subjects_for_distance, default_subject_ids
from .estimation import fit_individual_structures
from .metrics import MetricFn, joint_lpl, resolve_metric
from .types import MDMDistanceResult


def _fit_pairwise_common_structure_concat(
    data_i: np.ndarray,
    data_j: np.ndarray,
    method: str,
    nbf: int,
    delta_grid: np.ndarray,
    node_names: Optional[List[str]],
) -> np.ndarray:
    """Learn common DAG on time-wise concatenation (VTS-concat surrogate)."""
    pair = np.vstack([data_i, data_j])
    learner = StructureLearner(verbose=False)
    return learner.learn_structure(
        data=pair, method=method, nbf=nbf, delta=delta_grid, node_names=node_names
    )


def _is_acyclic(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = [i for i in range(n) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for child in np.where(adj[node] > 0)[0]:
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)
    return visited == n


def _joint_score_pair(
    adj: np.ndarray,
    data_i: np.ndarray,
    data_j: np.ndarray,
    nbf: int,
    delta_grid: np.ndarray,
) -> float:
    """Sum of joint LPL for two subjects under a shared adjacency (maximisation target)."""
    df_i = select_discount_factors(data_i, adj, nbf=nbf, delta=delta_grid)["DF_hat"]
    df_j = select_discount_factors(data_j, adj, nbf=nbf, delta=delta_grid)["DF_hat"]
    return joint_lpl(data_i, adj, df_i, nbf) + joint_lpl(data_j, adj, df_j, nbf)


def _fit_pairwise_common_structure_joint(
    data_i: np.ndarray,
    data_j: np.ndarray,
    adj_i: np.ndarray,
    adj_j: np.ndarray,
    nbf: int,
    delta_grid: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Greedy hill-climb maximising ``LPL_i(m) + LPL_j(m)`` over a shared DAG.

    Uses existing per-node ``compute_local_score`` primitives only.
    """
    n = adj_i.shape[0]
    # start from union of individual structures (restricted to acyclic subgraph)
    adj = ((adj_i + adj_j) > 0).astype(int)
    np.fill_diagonal(adj, 0)
    if not _is_acyclic(adj):
        adj = np.zeros((n, n), dtype=int)

    data_list = [data_i, data_j]
    best_score = _joint_score_pair(adj, data_i, data_j, nbf, delta_grid)

    for _ in range(max_iter):
        improved = False
        candidates: List[Tuple[np.ndarray, float]] = []

        # try adding edges
        for i in range(n):
            for j in range(n):
                if i == j or adj[i, j]:
                    continue
                trial = adj.copy()
                trial[i, j] = 1
                if _is_acyclic(trial):
                    sc = _joint_score_pair(trial, data_i, data_j, nbf, delta_grid)
                    candidates.append((trial, sc))

        # try removing edges
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    trial = adj.copy()
                    trial[i, j] = 0
                    sc = _joint_score_pair(trial, data_i, data_j, nbf, delta_grid)
                    candidates.append((trial, sc))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_trial, best_trial_score = candidates[0]
        if best_trial_score > best_score + 1e-9:
            adj = best_trial
            best_score = best_trial_score
            improved = True

        if not improved:
            break

    return adj


def _prepare_individuals(
    subjects: Sequence[Any],
    *,
    method: str,
    nbf: int,
    delta_grid: np.ndarray,
    node_names: Optional[List[str]],
    subject_ids: Sequence[Any],
    n_jobs: Optional[int],
    verbose: bool,
) -> List[Any]:
    """Return fitted MDM list, fitting only when raw arrays are supplied."""
    _, _, prefit = coerce_subjects_for_distance(subjects, node_names)
    if prefit is not None:
        return prefit
    return fit_individual_structures(
        subjects,
        method=method,
        nbf=nbf,
        delta_grid=delta_grid,
        node_names=node_names,
        subject_ids=subject_ids,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def _pair_worker(args: Tuple) -> Tuple[int, int, float]:
    """Worker for parallel pair loop."""
    (
        i,
        j,
        mdms,
        metric_fn,
        metric_needs_common,
        method,
        nbf,
        delta_grid,
        node_names,
        common_structure,
        self_lpl,
    ) = args

    ctx: Dict[str, Any] = {
        "self_lpl_i": self_lpl[i],
        "self_lpl_j": self_lpl[j],
        "nbf": nbf,
        "delta_grid": delta_grid,
    }

    if metric_needs_common:
        data_i = np.asarray(mdms[i].data, dtype=float)
        data_j = np.asarray(mdms[j].data, dtype=float)
        if common_structure == "joint":
            adj_i = np.asarray(mdms[i].adj_mat, dtype=int)
            adj_j = np.asarray(mdms[j].adj_mat, dtype=int)
            ctx["common_adj"] = _fit_pairwise_common_structure_joint(
                data_i, data_j, adj_i, adj_j, nbf, delta_grid
            )
        else:
            ctx["common_adj"] = _fit_pairwise_common_structure_concat(
                data_i, data_j, method, nbf, delta_grid, node_names
            )

    d = metric_fn(mdms[i], mdms[j], ctx=ctx)
    return i, j, float(d)


def compute_mdm_distance(
    subjects: Sequence[Union[np.ndarray, Any]],
    *,
    metric: Union[str, MetricFn] = "lpl_separation",
    common_structure: Literal["concat", "joint"] = "concat",
    method: Literal["hc", "tabu", "mmhc"] = "hc",
    nbf: int = 15,
    delta_grid: Optional[np.ndarray] = None,
    node_names: Optional[List[str]] = None,
    subject_ids: Optional[Sequence[Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> MDMDistanceResult:
    """
    Compute pairwise MDM dissimilarity matrix for a cohort (stages 2–3).

    Accepts raw time-series arrays or pre-fitted MDM objects from
    :func:`fit_individual_structures`.
    """
    if delta_grid is None:
        delta_grid = get_default_delta()
    delta_grid = np.minimum(np.asarray(delta_grid, dtype=float), 1.0)

    arrays, resolved_names, _ = coerce_subjects_for_distance(subjects, node_names)
    s = len(arrays)
    if s < 2:
        raise ValueError("Need at least 2 subjects to build a distance matrix.")
    if subject_ids is None:
        subject_ids = default_subject_ids(subjects)
    else:
        subject_ids = list(subject_ids)

    metric_name, metric_fn = resolve_metric(metric)
    metric_needs_common = metric_name == "lpl_separation"

    individuals = _prepare_individuals(
        subjects,
        method=method,
        nbf=nbf,
        delta_grid=delta_grid,
        node_names=resolved_names,
        subject_ids=subject_ids,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    ind_adj: List[np.ndarray] = []
    ind_delta: List[np.ndarray] = []
    self_lpl: List[float] = []
    for m in individuals:
        adj = np.asarray(m.adj_mat, dtype=int)
        df = np.asarray(m.DF["DF_hat"], dtype=float)
        data = np.asarray(m.data, dtype=float)
        ind_adj.append(adj)
        ind_delta.append(df)
        self_lpl.append(joint_lpl(data, adj, df, nbf))

    pair_indices = list(combinations(range(s), 2))
    args_list = [
        (
            i,
            j,
            individuals,
            metric_fn,
            metric_needs_common,
            method,
            nbf,
            delta_grid,
            resolved_names,
            common_structure,
            self_lpl,
        )
        for i, j in pair_indices
    ]

    if verbose and n_jobs != 1:
        print(f"[distance] computing {len(pair_indices)} pairs (n_jobs={n_jobs})")

    results = _parallel_map(
        _pair_worker,
        args_list,
        n_jobs,
        "Computing pairwise distances",
        verbose=verbose,
    )

    dist_sq = np.zeros((s, s), dtype=float)
    for i, j, d in results:
        dist_sq[i, j] = d
        dist_sq[j, i] = d

    from scipy.spatial.distance import squareform

    condensed = squareform(dist_sq, checks=False)

    if verbose:
        for i, j, d in results:
            print(f"[distance] d({subject_ids[i]},{subject_ids[j]}) = {d:.3f}")

    return MDMDistanceResult(
        condensed=condensed,
        subject_ids=subject_ids,
        metric=metric_name,
        method=method,
        individuals=individuals,
        metadata={
            "individual_adj": ind_adj,
            "individual_delta": ind_delta,
            "self_lpl": self_lpl,
            "nbf": nbf,
            "common_structure": common_structure,
            "node_names": resolved_names,
        },
    )
