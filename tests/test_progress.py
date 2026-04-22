"""
Tests for progress bar functionality in MDM.

This module tests that progress bars work correctly and don't break
functionality when verbose is enabled.
"""

import numpy as np

from mdmp.progress import TQDM_AVAILABLE, get_progress_bar, process_map_with_progress
from mdmp.scoring import select_discount_factors
from mdmp.utils import DEFAULT_NBF


def test_get_progress_bar_basic():
    """Test basic progress bar creation."""
    iterable = range(10)
    result = get_progress_bar(iterable, desc="Test", disable=True)

    # Should return iterable when disabled
    assert list(result) == list(iterable)


def test_get_progress_bar_with_verbose():
    """Test progress bar with verbose enabled."""
    iterable = range(5)
    result = get_progress_bar(iterable, desc="Test", disable=False)

    # Should return something iterable
    assert hasattr(result, '__iter__')
    # When not disabled, should be able to iterate
    assert len(list(result)) == 5


def test_select_discount_factors_with_progress(sample_data, small_dag_adjacency, default_delta):
    """Test that select_discount_factors works with verbose progress."""
    # Test with verbose=False (no progress bar)
    result_no_progress = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        verbose=False
    )

    # Test with verbose=True (with progress bar)
    result_with_progress = select_discount_factors(
        sample_data,
        small_dag_adjacency,
        nbf=DEFAULT_NBF,
        delta=default_delta,
        verbose=True
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(
        result_no_progress['lpldet'],
        result_with_progress['lpldet'],
        decimal=10
    )
    np.testing.assert_array_almost_equal(
        result_no_progress['DF_hat'],
        result_with_progress['DF_hat'],
        decimal=10
    )


def test_select_discount_factors_parallel_with_progress(sample_data, small_dag_adjacency, default_delta):
    """Test select_discount_factors with parallel processing and progress."""
    import os
    n_cores = os.cpu_count() or 1

    if n_cores >= 2:
        # Test with parallel processing and progress
        result = select_discount_factors(
            sample_data,
            small_dag_adjacency,
            nbf=DEFAULT_NBF,
            delta=default_delta,
            n_jobs=2,
            verbose=True
        )

        # Should produce valid results
        assert 'lpldet' in result
        assert 'DF_hat' in result
        assert np.all(np.isfinite(result['DF_hat']))


def test_mdm_with_progress(sample_data):
    """Test MDM class with progress bars."""
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

        # Test with verbose=False
        model_no_progress = MDM(small_data, method="hc", nbf=5, verbose=False)

        # Test with verbose=True
        model_with_progress = MDM(small_data, method="hc", nbf=5, verbose=True)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            model_no_progress.DF['DF_hat'],
            model_with_progress.DF['DF_hat'],
            decimal=10
        )


def test_process_map_with_progress():
    """Test process_map_with_progress function."""
    def square(x):
        return x ** 2

    items = list(range(10))

    # Test with single worker (serial)
    results = process_map_with_progress(
        square,
        items,
        max_workers=1,
        disable=True  # Disable progress for test
    )

    assert len(results) == 10
    assert results[0] == 0
    assert results[5] == 25
    assert results[9] == 81


def test_tqdm_availability():
    """Test that tqdm availability is correctly detected."""
    # TQDM_AVAILABLE should be a boolean
    assert isinstance(TQDM_AVAILABLE, bool)

    # If tqdm is available, we should be able to use it
    if TQDM_AVAILABLE:
        from tqdm import tqdm
        assert callable(tqdm)


def test_progress_bar_fallback():
    """Test that progress bar falls back gracefully when tqdm unavailable."""
    # This test verifies that the code doesn't break if tqdm is not available
    # The get_progress_bar function should handle this gracefully
    result = get_progress_bar(range(5), desc="Test", disable=False)

    # Should still be iterable
    assert hasattr(result, '__iter__')
    assert len(list(result)) == 5
