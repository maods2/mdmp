"""
Tests for utility functions.
"""

import numpy as np

from mdmp.utils import (
    DEFAULT_DELTA_MAX,
    DEFAULT_DELTA_MIN,
    DEFAULT_DELTA_STEP,
    build_design_matrix,
    build_parameter_names,
    extract_target_series,
    get_default_delta,
)


def test_build_design_matrix_basic(sample_data, small_dag_adjacency):
    """Test basic design matrix construction."""
    # Test for node 2 (has parents 0 and 1)
    Ft, parent_list = build_design_matrix(sample_data, small_dag_adjacency, 2)

    assert Ft.shape[0] == sample_data.shape[0]  # T rows
    assert Ft.shape[1] == 3  # intercept + 2 parents
    assert len(parent_list) == 2
    assert 0 in parent_list
    assert 1 in parent_list

    # First column should be all ones (intercept)
    assert np.allclose(Ft[:, 0], 1.0)

    # Second column should be data from parent 0
    assert np.allclose(Ft[:, 1], sample_data[:, 0])

    # Third column should be data from parent 1
    assert np.allclose(Ft[:, 2], sample_data[:, 1])


def test_build_design_matrix_no_parents(sample_data, empty_dag_adjacency):
    """Test design matrix for node with no parents (only intercept)."""
    Ft, parent_list = build_design_matrix(sample_data, empty_dag_adjacency, 0)

    assert Ft.shape[0] == sample_data.shape[0]
    assert Ft.shape[1] == 1  # Only intercept
    assert len(parent_list) == 0

    # Only column should be all ones
    assert np.allclose(Ft[:, 0], 1.0)


def test_extract_target_series(sample_data):
    """Test extracting target series."""
    for i in range(sample_data.shape[1]):
        series = extract_target_series(sample_data, i)
        assert series.shape == (sample_data.shape[0],)
        assert np.allclose(series, sample_data[:, i])


def test_get_default_delta():
    """Test default delta sequence generation."""
    delta = get_default_delta()

    assert isinstance(delta, np.ndarray)
    assert delta[0] == DEFAULT_DELTA_MIN
    assert delta[-1] < DEFAULT_DELTA_MAX
    assert np.allclose(np.diff(delta), DEFAULT_DELTA_STEP)


def test_build_parameter_names(small_dag_adjacency):
    """Test parameter name construction."""
    node_names = ["A", "B", "C"]
    param_names = build_parameter_names(2, small_dag_adjacency, node_names)

    # Should have intercept + 2 parents
    assert len(param_names) == 3
    assert param_names[0] == "beta0_C"
    assert "A->C" in param_names
    assert "B->C" in param_names


def test_build_parameter_names_default_names(small_dag_adjacency):
    """Test parameter name construction with default names."""
    param_names = build_parameter_names(2, small_dag_adjacency, None)

    assert len(param_names) == 3
    assert param_names[0] == "beta0_V3"
    assert "V1->V3" in param_names
    assert "V2->V3" in param_names


def test_build_parameter_names_no_parents(empty_dag_adjacency):
    """Test parameter names for node with no parents."""
    param_names = build_parameter_names(0, empty_dag_adjacency, None)

    assert len(param_names) == 1
    assert param_names[0] == "beta0_V1"
