"""
Tests for parallel processing functionality in MDM.

This module tests that parallel processing produces identical results
to serial processing and validates edge cases.
"""

import numpy as np
import pytest

from mdmp.parallel import _get_n_jobs
from mdmp.scoring import select_discount_factors
from mdmp.constants import DEFAULT_NBF


def test_get_n_jobs():
    """Test _get_n_jobs utility function."""
    # Test None (should use default)
    assert _get_n_jobs(None, default=1) == 1
    assert _get_n_jobs(None, default=-1) > 0  # Should return number of cores

    # Test -1 (should use all cores)
    n_cores = _get_n_jobs(-1)
    assert n_cores > 0
    assert isinstance(n_cores, int)

    # Test 1 (serial)
    assert _get_n_jobs(1) == 1

    # Test specific number
    assert _get_n_jobs(4) == 4
    assert _get_n_jobs(8) == 8

    # Test invalid values
    with pytest.raises(ValueError):
        _get_n_jobs(0)
    with pytest.raises(ValueError):
        _get_n_jobs(-2)


def test_select_discount_factors_serial_vs_parallel(sample_data, small_dag_adjacency, default_delta):
    """Test that serial and parallel processing produce identical results."""
    # Run serial (n_jobs=None)
    result_serial = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=None
    )

    # Run serial (n_jobs=1)
    result_serial_1 = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=1
    )

    # Run parallel (n_jobs=2, if we have at least 2 cores)
    import os
    n_cores = os.cpu_count() or 1
    if n_cores >= 2:
        result_parallel = select_discount_factors(
            sample_data,
            small_dag_adjacency,
            nbf=DEFAULT_NBF,
            delta=default_delta,
            n_jobs=2
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(
            result_serial['lpldet'],
            result_parallel['lpldet'],
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            result_serial['DF_hat'],
            result_parallel['DF_hat'],
            decimal=10
        )

    # Serial with None and 1 should be identical
    np.testing.assert_array_almost_equal(
        result_serial['lpldet'],
        result_serial_1['lpldet'],
        decimal=10
    )
    np.testing.assert_array_almost_equal(
        result_serial['DF_hat'],
        result_serial_1['DF_hat'],
        decimal=10
    )


def test_select_discount_factors_n_jobs_minus_one(sample_data, small_dag_adjacency, default_delta):
    """Test select_discount_factors with n_jobs=-1 (all cores)."""
    result = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=-1
    )

    # Should produce valid results
    assert 'lpldet' in result
    assert 'DF_hat' in result
    assert result['lpldet'].shape[0] == len(default_delta)
    assert result['lpldet'].shape[1] == sample_data.shape[1]
    assert result['DF_hat'].shape[0] == sample_data.shape[1]
    assert np.all(np.isfinite(result['DF_hat']))


def test_select_discount_factors_n_jobs_specific(sample_data, small_dag_adjacency, default_delta):
    """Test select_discount_factors with specific n_jobs value."""
    import os
    n_cores = os.cpu_count() or 1
    n_jobs_test = min(2, n_cores)  # Use at most 2 workers

    result = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=n_jobs_test
    )

    # Should produce valid results
    assert 'lpldet' in result
    assert 'DF_hat' in result
    assert np.all(np.isfinite(result['DF_hat']))


def test_select_discount_factors_backward_compatibility(sample_data, small_dag_adjacency, default_delta):
    """Test that default behavior (n_jobs=None) matches original implementation."""
    # Call without n_jobs parameter (should default to serial)
    result_default = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta
    )

    # Call with n_jobs=None explicitly
    result_explicit = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        n_jobs=None
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(
        result_default['lpldet'],
        result_explicit['lpldet'],
        decimal=10
    )
    np.testing.assert_array_almost_equal(
        result_default['DF_hat'],
        result_explicit['DF_hat'],
        decimal=10
    )


def test_mdm_parallel_processing(sample_data):
    """Test MDM class with parallel processing."""
    import sys
    from unittest.mock import MagicMock, Mock, patch

    # Create fake pgmpy module for testing when pgmpy is not installed
    if 'pgmpy' not in sys.modules:
        pgmpy_mock = MagicMock()
        pgmpy_mock.estimators = MagicMock()
        sys.modules['pgmpy'] = pgmpy_mock
        sys.modules['pgmpy.estimators'] = pgmpy_mock.estimators

    from mdmp import MDM

    # Use small dataset for faster tests
    small_data = sample_data[:30, :2]  # 30 time points, 2 nodes

    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        # Mock the structure learning to return a simple DAG
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        # Test with n_jobs=None (serial)
        model_serial = MDM(small_data, method="hc", nbf=5, verbose=False, n_jobs=None)

        # Test with n_jobs=1 (serial)
        model_serial_1 = MDM(small_data, method="hc", nbf=5, verbose=False, n_jobs=1)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            model_serial.DF['DF_hat'],
            model_serial_1.DF['DF_hat'],
            decimal=10
        )

        # Test with parallel processing if we have multiple cores
        import os
        n_cores = os.cpu_count() or 1
        if n_cores >= 2:
            model_parallel = MDM(small_data, method="hc", nbf=5, verbose=False, n_jobs=2)

            # Results should be identical
            np.testing.assert_array_almost_equal(
                model_serial.DF['DF_hat'],
                model_parallel.DF['DF_hat'],
                decimal=10
            )

            # Check that filtered and smoothed results are also identical
            for key in ['mt', 'Ct', 'Rt', 'nt', 'dt']:
                for node_idx in model_serial.Filt[key].keys():
                    np.testing.assert_array_almost_equal(
                        model_serial.Filt[key][node_idx],
                        model_parallel.Filt[key][node_idx],
                        decimal=10
                    )
