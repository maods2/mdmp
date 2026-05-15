"""
Edge-frequency voting and greedy acyclic repair for IS aggregation.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np

ThresholdMode = Literal["strict", "inclusive"]


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
    If ``adj`` has a directed cycle, remove one cycle edge with minimum empirical vote frequency.

    This **greedy** repair processes one cycle at a time (via
    :func:`networkx.find_cycle`) and need not yield a minimum feedback arc set
    for the full graph.

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


def _vote_threshold_and_repair_cycles(
    subject_adjs: List[np.ndarray],
    tau: float,
    node_names: List[str],
    *,
    threshold_mode: ThresholdMode = "strict",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Edge-wise vote against ``tau``, then greedy lowest-frequency cycle-edge removal until acyclic.

    Strict mode keeps edges with frequency **strictly greater** than ``tau``
    (historical default). Inclusive mode keeps edges with frequency **greater
    than or equal** to ``tau``.
    """
    s = len(subject_adjs)
    stack = np.stack(subject_adjs, axis=0)
    edge_counts = np.sum(stack, axis=0).astype(np.int64)
    np.fill_diagonal(edge_counts, 0)
    edge_frequencies = edge_counts.astype(float) / float(s)

    if threshold_mode == "strict":
        adj = (edge_frequencies > tau).astype(np.int8)
    elif threshold_mode == "inclusive":
        adj = (edge_frequencies >= tau).astype(np.int8)
    else:
        raise ValueError(f"threshold_mode must be 'strict' or 'inclusive', got {threshold_mode!r}")
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
                "parent_name": node_names[i],
                "child_name": node_names[j],
                "frequency": float(edge_frequencies[i, j]),
            }
        )

    out_adj = adj.astype(np.float64)
    meta: Dict[str, Any] = {
        "edge_counts": edge_counts,
        "edge_frequencies": edge_frequencies.copy(),
        "edges_removed_for_acyclicity": removed,
        "threshold_mode": threshold_mode,
    }
    return out_adj, meta
