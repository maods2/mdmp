"""
Utility functions for MDM operations.

This module contains helper functions that reduce code duplication and
improve readability across the MDM package.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import (
    DEFAULT_DELTA_MAX,
    DEFAULT_DELTA_MIN,
    DEFAULT_DELTA_STEP,
)


def build_design_matrix(
    data: np.ndarray,
    adj_mat: np.ndarray,
    node_idx: int
) -> Tuple[np.ndarray, List[int]]:
    """
    Build design matrix Ft for a given node based on adjacency matrix.

    The design matrix includes:
    - Intercept column (always)
    - Columns for each parent node (from adjacency matrix)

    Parameters
    ----------
    data : np.ndarray
        Time series data (T x N), where T is number of time points and N is number of nodes.
    adj_mat : np.ndarray
        Adjacency matrix (N x N) representing the DAG structure.
    node_idx : int
        Index of the target node.

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        Design matrix Ft with shape (T, p) where p is number of parameters,
        and list of parent node indices (excluding self).
    """
    Nt = data.shape[0]

    # Count number of parameters: intercept + number of parents
    # Cast to int so float adjacency (e.g. from refit paths) does not yield np.float64.
    num_parents = int(np.sum(adj_mat[:, node_idx]))
    if adj_mat[node_idx, node_idx] == 0:
        num_parents += 1  # Always include intercept

    # Get parent indices (excluding self-edge)
    parents = np.where(adj_mat[:, node_idx] > 0)[0]
    parent_list = [p for p in parents if p != node_idx]

    # Build design matrix: first column is intercept (ones)
    Ft = np.ones((Nt, num_parents))

    # Fill in parent data columns
    if len(parent_list) > 0:
        Ft[:, 1:1+len(parent_list)] = data[:, parent_list]

    return Ft, parent_list


def get_node_parameter_count(adj_mat: np.ndarray, node_idx: int) -> int:
    """
    Calculate the number of parameters for a node based on adjacency matrix.

    Parameters
    ----------
    adj_mat : np.ndarray
        Adjacency matrix (N x N).
    node_idx : int
        Index of the target node.

    Returns
    -------
    int
        Number of parameters (intercept + number of parent nodes).
    """
    num_params = np.sum(adj_mat[:, node_idx])
    if adj_mat[node_idx, node_idx] == 0:
        num_params += 1  # Always include intercept
    return num_params


def extract_target_series(data: np.ndarray, node_idx: int) -> np.ndarray:
    """
    Extract time series for a target node.

    Parameters
    ----------
    data : np.ndarray
        Time series data (T x N).
    node_idx : int
        Index of the target node.

    Returns
    -------
    np.ndarray
        Time series for the target node with shape (T,).
    """
    return data[:, node_idx]


def get_default_delta() -> np.ndarray:
    """
    Get default discount factor sequence for optimization.

    Returns
    -------
    np.ndarray
        Array of discount factors from 0.5 to 1.0 in steps of 0.01.
    """
    return np.arange(DEFAULT_DELTA_MIN, DEFAULT_DELTA_MAX, DEFAULT_DELTA_STEP)


def build_parameter_names(
    node_idx: int,
    adj_mat: np.ndarray,
    node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Build parameter names for a node based on its parents.

    Parameters
    ----------
    node_idx : int
        Index of the target node.
    adj_mat : np.ndarray
        Adjacency matrix (N x N).
    node_names : list of str, optional
        Names of nodes. If None, uses default names V1, V2, etc.

    Returns
    -------
    list of str
        List of parameter names: ['beta0_node', 'parent1->node', ...]
    """
    if node_names is None:
        node_names = [f"V{i+1}" for i in range(adj_mat.shape[0])]

    param_names = [f"beta0_{node_names[node_idx]}"]

    # Find connections pointing to this node
    connections = np.where(adj_mat == 1)
    for j in range(len(connections[0])):
        if connections[1][j] == node_idx:  # This is a parent of node_idx
            parent_name = node_names[connections[0][j]]
            param_names.append(f"{parent_name}->{node_names[node_idx]}")

    return param_names

