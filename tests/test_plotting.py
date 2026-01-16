"""
Tests for plotting functions.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from mdmp import MDM

# Create fake pgmpy module for testing when pgmpy is not installed
if 'pgmpy' not in sys.modules:
    pgmpy_mock = MagicMock()
    pgmpy_mock.estimators = MagicMock()
    sys.modules['pgmpy'] = pgmpy_mock
    sys.modules['pgmpy.estimators'] = pgmpy_mock.estimators


@pytest.fixture
def mock_mdm_model(sample_data):
    """Create a mock MDM model for plotting tests."""
    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        small_data = sample_data[:30, :2]
        model = MDM(small_data, method="hc", nbf=5, verbose=False)
        return model


def test_plot_dag_does_not_crash(mock_mdm_model):
    """Test that plot_dag doesn't crash."""
    from mdmp.plotting import plot_dag

    # Test graph plot
    fig = plot_dag(mock_mdm_model, plot_type="graph")
    assert fig is not None

    # Test heatmap plot
    fig = plot_dag(mock_mdm_model, plot_type="heatmap")
    assert fig is not None


def test_plot_arcs_does_not_crash(mock_mdm_model):
    """Test that plot_arcs doesn't crash."""
    from mdmp.plotting import plot_arcs

    fig = plot_arcs(mock_mdm_model, plot_type="connections", distribution="filt")
    assert fig is not None


def test_plot_marginal_does_not_crash(mock_mdm_model):
    """Test that plot_marginal doesn't crash."""
    from mdmp.plotting import plot_marginal

    fig = plot_marginal(mock_mdm_model, target_node=0, distribution="filt")
    assert fig is not None


def test_plot_stream_does_not_crash(mock_mdm_model):
    """Test that plot_stream doesn't crash."""
    from mdmp.plotting import plot_stream

    fig = plot_stream(mock_mdm_model, child_node=0, distribution="filt")
    assert fig is not None


def test_plot_dag_parameter_validation(mock_mdm_model):
    """Test plot_dag parameter validation."""
    from mdmp.plotting import plot_dag

    # Should work with custom node labels
    fig = plot_dag(
        mock_mdm_model,
        node_labels=["Custom1", "Custom2"],
        plot_type="graph"
    )
    assert fig is not None
