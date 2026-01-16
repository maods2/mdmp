"""
Pytest configuration and shared fixtures for MDMP tests.
"""


import numpy as np
import pytest


@pytest.fixture
def sample_data() -> np.ndarray:
    """
    Generate sample time series data for testing.
    
    Returns
    -------
    np.ndarray
        Time series data with shape (T, N) where T=100, N=3.
    """
    np.random.seed(42)
    T = 100
    N = 3
    data = np.random.randn(T, N)
    return data


@pytest.fixture
def small_dag_adjacency() -> np.ndarray:
    """
    Create a small known DAG structure for testing.
    
    Structure: 0 -> 1, 0 -> 2, 1 -> 2
    (Node 0 is parent of 1 and 2, Node 1 is parent of 2)
    
    Returns
    -------
    np.ndarray
        Adjacency matrix (3x3).
    """
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1  # 0 -> 1
    adj[0, 2] = 1  # 0 -> 2
    adj[1, 2] = 1  # 1 -> 2
    return adj


@pytest.fixture
def empty_dag_adjacency() -> np.ndarray:
    """
    Create an empty DAG (no edges) for testing.
    
    Returns
    -------
    np.ndarray
        Empty adjacency matrix (3x3).
    """
    return np.zeros((3, 3), dtype=int)


@pytest.fixture
def default_delta() -> np.ndarray:
    """
    Get default discount factor sequence for testing.
    
    Returns
    -------
    np.ndarray
        Array of discount factors from 0.5 to 1.0 in steps of 0.01.
    """
    return np.arange(0.5, 1.01, 0.01)
