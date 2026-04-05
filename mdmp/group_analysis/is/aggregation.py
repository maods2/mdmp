"""
Individual Structure (IS) aggregation: combine subject DAGs by edge voting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class ISAggregationResult:
    """
    Result of aggregating individual structures into one global DAG.

    Attributes
    ----------
    adj_mat : np.ndarray
        Binary (N, N) adjacency matrix; ``adj_mat[i, j] == 1`` means parent i → child j.
    node_names : list of str
        Variable names aligned with matrix rows/columns.
    n_subjects : int
        Number of subject graphs aggregated.
    tau : float
        Threshold used for inclusion (edge kept iff frequency > tau).
    metadata : dict
        Includes ``edge_counts``, ``edge_frequencies``,
        ``edges_removed_for_acyclicity`` (list of dicts with parent_idx, child_idx,
        frequency).
    """

    adj_mat: np.ndarray
    node_names: List[str]
    n_subjects: int
    tau: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_float_matrix(adj: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (N,N) float array and optional column names from DataFrame."""
    if isinstance(adj, pd.DataFrame):
        names = [str(c) for c in adj.columns.tolist()]
        return np.asarray(adj.values, dtype=float), names
    return np.asarray(adj, dtype=float), None


def _to_binary_adj(
    adj: Union[np.ndarray, pd.DataFrame],
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (N,N) int array and optional names from DataFrame."""
    arr, names = _as_float_matrix(adj)
    flat = arr.ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("adjacency matrices must contain only finite values")
    if not np.logical_or(flat == 0, flat == 1).all():
        raise ValueError(
            "adjacency must be binary (0/1); non-binary values are not allowed"
        )
    return arr.astype(int), names


def _validate_adj_list(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame]],
    node_names: Optional[Sequence[str]],
) -> Tuple[List[np.ndarray], List[str], int]:
    if len(adj_mats) == 0:
        raise ValueError("adj_mats must contain at least one adjacency matrix")

    parsed: List[Tuple[np.ndarray, Optional[List[str]]]] = [
        _to_binary_adj(a) for a in adj_mats
    ]
    shapes = {p[0].shape for p in parsed}
    if len(shapes) != 1:
        raise ValueError(
            f"All adjacency matrices must have the same shape, got {shapes}"
        )
    n, m = parsed[0][0].shape
    if n != m:
        raise ValueError(f"Adjacency matrices must be square, got shape {(n, m)}")

    arrays = [p[0] for p in parsed]

    names: List[str]
    if node_names is not None:
        if len(node_names) != n:
            raise ValueError(
                f"node_names length {len(node_names)} does not match N={n}"
            )
        names = [str(x) for x in node_names]
    else:
        first_df_names = parsed[0][1]
        if first_df_names is not None and len(first_df_names) == n:
            names = first_df_names
        else:
            for p in parsed:
                if p[1] is not None and len(p[1]) == n:
                    names = p[1]
                    break
            else:
                names = [f"V{i + 1}" for i in range(n)]

    return arrays, names, len(arrays)


def _adj_to_graph(adj: np.ndarray) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(adj.shape[0]))
    idx = np.argwhere(adj != 0)
    for i, j in idx:
        if i != j:
            g.add_edge(int(i), int(j))
    return g


def _remove_lowest_freq_cycle_edge(
    adj: np.ndarray,
    freq: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    If ``adj`` has a directed cycle, remove one cycle edge with minimum frequency.

    Tie-break: lexicographic (parent_idx, child_idx).

    Returns
    -------
    tuple or None
        (i, j) removed, or None if already acyclic.
    """
    g = _adj_to_graph(adj)
    if nx.is_directed_acyclic_graph(g):
        return None
    try:
        cyc = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return None

    # find_cycle returns (u, v, direction) with orientation='original'
    edges: List[Tuple[int, int]] = []
    for item in cyc:
        if len(item) == 3:
            u, v, _ = item
            edges.append((int(u), int(v)))
        else:
            u, v = item[0], item[1]
            edges.append((int(u), int(v)))

    def _sort_key(e: Tuple[int, int]) -> Tuple[float, int, int]:
        return (float(freq[e[0], e[1]]), e[0], e[1])

    i, j = min(edges, key=_sort_key)
    adj[i, j] = 0
    return (i, j)


def aggregate_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
) -> ISAggregationResult:
    """
    Aggregate subject-specific DAG adjacency matrices into one global DAG.

    For each directed edge (i → j), compute the fraction of subjects that
    include the edge. Include it in the pooled graph if frequency > ``tau``.
    If the thresholded graph has directed cycles, repeatedly remove the
    lowest-frequency edge on a detected cycle until the graph is acyclic.

    Parameters
    ----------
    adj_mats : sequence of array-like or DataFrame
        One (N, N) binary adjacency per subject; ``[i, j] == 1`` means i → j.
        All matrices must share the same N and variable ordering.
    tau : float, optional
        Strict threshold in (0, 1]. Default 0.5 (majority: more than half).
    node_names : sequence of str, optional
        Names of length N. If omitted, taken from the first DataFrame or ``V1``…``VN``.

    Returns
    -------
    ISAggregationResult

    Notes
    -----
    For weighted / fractional votes in the future, generalize the count tensor
    stored in ``metadata['edge_counts']``.
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")

    arrays, names, s = _validate_adj_list(adj_mats, node_names)
    n = arrays[0].shape[0]
    stack = np.stack(arrays, axis=0)
    edge_counts = np.sum(stack, axis=0).astype(np.int64)
    np.fill_diagonal(edge_counts, 0)
    edge_frequencies = edge_counts.astype(float) / float(s)

    adj = (edge_frequencies > tau).astype(np.int8)
    np.fill_diagonal(adj, 0)

    removed: List[Dict[str, Any]] = []
    while True:
        dropped = _remove_lowest_freq_cycle_edge(adj, edge_frequencies)
        if dropped is None:
            break
        i, j = dropped
        removed.append(
            {
                "parent_idx": i,
                "child_idx": j,
                "parent_name": names[i],
                "child_name": names[j],
                "frequency": float(edge_frequencies[i, j]),
            }
        )

    out_adj = adj.astype(np.float64)
    meta: Dict[str, Any] = {
        "edge_counts": edge_counts,
        "edge_frequencies": edge_frequencies.copy(),
        "edges_removed_for_acyclicity": removed,
    }
    return ISAggregationResult(
        adj_mat=out_adj,
        node_names=names,
        n_subjects=s,
        tau=tau,
        metadata=meta,
    )
