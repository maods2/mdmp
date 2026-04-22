"""
Utility functions for structure learning algorithms.
"""

from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd


def extract_adjacency_from_model(
    model: Any,
    columns: List[str]
) -> np.ndarray:
    """
    Extract adjacency matrix from a pgmpy model or dict.

    Parameters
    ----------
    model : Any
        pgmpy model object or dictionary containing model information.
    columns : list of str
        List of column names in the correct order.

    Returns
    -------
    np.ndarray
        Adjacency matrix (N x N).
    """
    if isinstance(model, dict):
        adj = model.get("adjmat")
        if adj is not None:
            if isinstance(adj, pd.DataFrame):
                return adj.loc[columns, columns].to_numpy(dtype=int)
            return np.array(adj, dtype=int)
        if "model_edges" in model and model["model_edges"] is not None:
            edges = model["model_edges"]
        elif "edges" in model and model["edges"] is not None:
            edges = model["edges"]
        elif "model" in model and model["model"] is not None:
            edges = list(model["model"].edges())
        else:
            edges = []
    else:
        edges = list(model.edges())

    return edges_to_adjacency_matrix(edges, columns)


def edges_to_adjacency_matrix(
    edges: Iterable[Tuple[str, str]],
    columns: List[str]
) -> np.ndarray:
    """
    Convert edge list to adjacency matrix.

    Parameters
    ----------
    edges : iterable of tuple
        Iterable of (parent, child) tuples.
    columns : list of str
        List of column names in the correct order.

    Returns
    -------
    np.ndarray
        Adjacency matrix (N x N).
    """
    node_idx = {name: idx for idx, name in enumerate(columns)}
    N = len(columns)
    adj = np.zeros((N, N), dtype=int)
    for parent, child in edges:
        if parent in node_idx and child in node_idx:
            adj[node_idx[parent], node_idx[child]] = 1
    return adj
