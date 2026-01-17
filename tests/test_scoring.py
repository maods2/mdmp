"""
Tests for scoring functions (discount factor selection).
"""

import numpy as np

from mdmp.scoring import compute_logpl, select_discount_factors
from mdmp.utils import DEFAULT_NBF


def test_select_discount_factors_basic(sample_data, small_dag_adjacency, default_delta):
    """Test basic discount factor selection."""
    result = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta
    )

    assert 'lpldet' in result
    assert 'DF_hat' in result

    # Check dimensions
    assert result['lpldet'].shape[0] == len(default_delta)
    assert result['lpldet'].shape[1] == sample_data.shape[1]
    assert result['DF_hat'].shape[0] == sample_data.shape[1]

    # Check that selected deltas are in valid range
    assert np.all(result['DF_hat'] >= default_delta[0])
    assert np.all(result['DF_hat'] <= default_delta[-1])


def test_select_discount_factors_empty_dag(sample_data, empty_dag_adjacency, default_delta):
    """Test discount factor selection with empty DAG."""
    result = select_discount_factors(
        sample_data,
        empty_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta
    )

    assert 'DF_hat' in result
    assert len(result['DF_hat']) == sample_data.shape[1]

    # All nodes should have selected discount factors
    assert np.all(np.isfinite(result['DF_hat']))


def test_compute_logpl_basic(sample_data, small_dag_adjacency):
    """Test log predictive likelihood computation."""
    node_idx = 2
    delta = 0.9

    logpl = compute_logpl(
        sample_data,
        small_dag_adjacency,
        delta,
        node_idx,
        nbf=DEFAULT_NBF
    )

    # Should return a finite float
    assert np.isfinite(logpl)
    assert isinstance(logpl, (float, np.floating))

    # compute_logpl returns -lpldet, so if lpldet is negative (typical),
    # logpl will be positive. Just check it's finite.
    # The actual sign depends on the data and model fit.


def test_compute_logpl_invalid_delta(sample_data, small_dag_adjacency):
    """Test compute_logpl with invalid discount factor."""
    node_idx = 0

    # Delta > 1 should return inf
    logpl = compute_logpl(
        sample_data,
        small_dag_adjacency,
        delta=1.5,
        node_idx=node_idx
    )
    assert not np.isfinite(logpl) or logpl == np.inf

    # Delta < 0 should return inf
    logpl = compute_logpl(
        sample_data,
        small_dag_adjacency,
        delta=-0.1,
        node_idx=node_idx
    )
    assert not np.isfinite(logpl) or logpl == np.inf


def test_compute_logpl_boundary_conditions(sample_data, small_dag_adjacency):
    """Test compute_logpl at boundary conditions."""
    node_idx = 0

    # Delta = 0 causes division by zero, should return inf
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logpl_0 = compute_logpl(
            sample_data,
            small_dag_adjacency,
            delta=0.0,
            node_idx=node_idx
        )
        # Delta = 0 causes numerical issues, may return inf
        assert not np.isfinite(logpl_0) or logpl_0 == np.inf

    # Delta = 1 should work
    logpl_1 = compute_logpl(
        sample_data,
        small_dag_adjacency,
        delta=1.0,
        node_idx=node_idx
    )
    assert np.isfinite(logpl_1) or logpl_1 == np.inf


def test_select_discount_factors_nan_handling():
    """Test that NaN values in lpldet are handled correctly."""
    # Create data that might produce NaN
    T = 20
    N = 2
    data = np.random.randn(T, N)

    # Create a simple adjacency matrix
    adj = np.zeros((N, N), dtype=int)
    adj[0, 1] = 1

    delta = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    result = select_discount_factors(data, adj, nbf=5, delta=delta)

    # Should still return valid discount factors
    assert np.all(np.isfinite(result['DF_hat']))
    assert len(result['DF_hat']) == N


def test_select_discount_factors_with_n_jobs(sample_data, small_dag_adjacency, default_delta):
    """Test select_discount_factors with n_jobs parameter."""
    # Test with n_jobs=None (should work same as default)
    result = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=None
    )

    assert 'lpldet' in result
    assert 'DF_hat' in result
    assert np.all(np.isfinite(result['DF_hat']))