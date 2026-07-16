"""Shared helpers for the MDMp vs MDMr HC reproducibility experiment."""

from __future__ import annotations

import os
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

# Pinned scenario constants (Michel TCC setting)
W = 0.01
V = 100.0
T = 200
N_REPLICATIONS = 300
BASE_SEED = 1564
NBF = 15
DELTA = np.linspace(0.5, 1.0, 51)
DAGS = ("3var", "5var")

NODE_NAMES = {
    "3var": ["Y1", "Y2", "Y3"],
    "5var": ["Y1", "Y2", "Y3", "Y4", "Y5"],
}

METRIC_NAMES = (
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "directional_accuracy",
    "computation_time",
    "n_edges",
)


def scenario_prefix(dag: str, w: float = W, v: float = V, t: int = T) -> str:
    """Return the filename prefix for a DAG/scenario combination."""
    return f"dag_{dag}_W{w}_V{v}_T{t}"


def data_filename(dag: str, dataset_id: int, w: float = W, v: float = V, t: int = T) -> str:
    """Return the simulated data CSV filename for one replication."""
    return f"{scenario_prefix(dag, w, v, t)}_ind{dataset_id}.csv"


def true_adj_filename(dag: str, w: float = W, v: float = V, t: int = T) -> str:
    """Return the true adjacency CSV filename for a DAG/scenario."""
    return f"{scenario_prefix(dag, w, v, t)}_true_adjacency.csv"


def build_connection_matrix(adj_mat: np.ndarray) -> dict:
    """Build symmetric connection matrix from directed adjacency matrix."""
    n_n = adj_mat.shape[0]
    k = np.zeros((n_n, n_n))

    pairs = list(combinations(range(n_n), 2))
    lower_ind = [(i, j) for i, j in pairs if i > j]
    upper_ind = [(i, j) for i, j in pairs if i < j]

    m_con = np.zeros(len(pairs))
    for idx, (i, j) in enumerate(pairs):
        m_con[idx] = adj_mat[i, j] + adj_mat[j, i]

    for idx, (i, j) in enumerate(pairs):
        k[i, j] = k[j, i] = 1 if m_con[idx] == 1 else 0

    return {
        "connection_matrix": k,
        "connection_vector": m_con,
        "lower_ind": lower_ind,
        "upper_ind": upper_ind,
    }


def compute_metrics(true_adj: np.ndarray, estimated_adj: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics for DAG structure learning."""
    n_n = true_adj.shape[0]

    true_con = build_connection_matrix(true_adj)
    est_con = build_connection_matrix(estimated_adj)

    m_con_true = true_con["connection_vector"]
    m_con_est = est_con["connection_vector"]

    accuracy = np.mean(m_con_est == m_con_true)

    if np.any(m_con_true == 1):
        sensitivity = np.mean(m_con_est[m_con_true == 1] == 1)
    else:
        sensitivity = np.nan

    if np.any(m_con_true == 0):
        specificity = np.mean(m_con_est[m_con_true == 0] == 0)
    else:
        specificity = np.nan

    if np.any(m_con_est == 1):
        ppv = np.mean(m_con_true[m_con_est == 1] == 1)
    else:
        ppv = np.nan

    if np.any(m_con_est == 0):
        npv = np.mean(m_con_true[m_con_est == 0] == 0)
    else:
        npv = np.nan

    k = true_con["connection_matrix"]
    pairs_with_connections = [
        (i, j) for i, j in combinations(range(n_n), 2) if k[i, j] == 1
    ]

    if pairs_with_connections:
        d_accuracy_vals = [
            (true_adj[i, j] == estimated_adj[i, j])
            & (true_adj[j, i] == estimated_adj[j, i])
            for i, j in pairs_with_connections
        ]
        d_accuracy = np.mean(d_accuracy_vals)
    else:
        d_accuracy = np.nan

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "directional_accuracy": d_accuracy,
    }


def load_true_adj(
    data_dir: str,
    dag: str,
    w: float = W,
    v: float = V,
    t: int = T,
) -> np.ndarray:
    """Load the true adjacency matrix for a DAG/scenario."""
    path = os.path.join(data_dir, true_adj_filename(dag, w, v, t))
    return pd.read_csv(path, index_col=0).values.astype(int)


def adjacency_to_long(
    dag: str,
    dataset_id: int,
    adj_mat: np.ndarray,
    node_names: Iterable[str] | None = None,
) -> list[dict]:
    """Convert an adjacency matrix to long-format rows for CSV export."""
    if node_names is None:
        node_names = NODE_NAMES[dag]
    node_names = list(node_names)
    rows = []
    for i, parent in enumerate(node_names):
        for j, child in enumerate(node_names):
            rows.append(
                {
                    "dag": dag,
                    "dataset_id": dataset_id,
                    "node_from": parent,
                    "node_to": child,
                    "edge": int(adj_mat[i, j]),
                }
            )
    return rows


def long_to_adjacency(
    df: pd.DataFrame,
    dataset_id: int,
    dag: str,
    node_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Reconstruct an adjacency matrix from long-format rows."""
    if node_names is None:
        node_names = NODE_NAMES[dag]
    node_names = list(node_names)

    subset = df[(df["dag"] == dag) & (df["dataset_id"] == dataset_id)]
    pivot = subset.pivot(index="node_from", columns="node_to", values="edge")
    pivot = pivot.reindex(index=node_names, columns=node_names, fill_value=0)
    return pivot.values.astype(int)
