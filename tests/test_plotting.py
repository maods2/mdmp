"""
Tests for plotting functions.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
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

    fig = plot_dag(mock_mdm_model, plot_type="graph", hierarchical=False, layout_seed=1)
    assert fig is not None


def test_plot_arcs_does_not_crash(mock_mdm_model):
    """Test that plot_arcs doesn't crash."""
    from mdmp.plotting import plot_arcs

    fig = plot_arcs(mock_mdm_model, plot_type="connections", distribution="filt")
    assert fig is not None


def test_plot_arcs_grid_shape_helpers():
    from mdmp.plotting.parameters import _default_plot_arcs_figsize, _grid_shape

    assert _grid_shape(0) == (1, 1)
    assert _grid_shape(1) == (1, 1)
    assert _grid_shape(4) == (1, 4)
    assert _grid_shape(5) == (2, 4)
    assert _grid_shape(8) == (2, 4)
    assert _grid_shape(9) == (3, 4)
    assert _grid_shape(5, max_cols=3) == (2, 3)
    w, h = _default_plot_arcs_figsize(2, 4)
    assert w > 0 and h > 0


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

    fig = plot_stream(mock_mdm_model, child_node=0, distribution="filt", smooth=False)
    assert fig is not None


def test_plot_anomalies_does_not_crash(mock_mdm_model):
    """Smoke test for plot_anomalies."""
    from mdmp.plotting import plot_anomalies

    fig = plot_anomalies(mock_mdm_model, series=0, ci_level=0.95)
    assert fig is not None
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.get_lines()) >= 2


def test_plot_marginal_smooth_and_labels(mock_mdm_model):
    from mdmp.plotting import plot_marginal
    from mdmp.plotting._style import format_param_label, upsample_curve

    assert format_param_label("beta0_Y1") == "intercept"
    assert format_param_label("Y1->Y2") == "Y1→Y2"

    t = np.arange(10, dtype=float)
    y = np.sin(t)
    fine_t, fine_y = upsample_curve(t, y, factor=4)
    assert len(fine_t) == 40
    assert fine_t[0] == 0.0 and fine_t[-1] == 9.0
    assert np.allclose(fine_y[0], y[0], atol=0.01)
    assert np.allclose(fine_y[-1], y[-1], atol=0.01)

    fig = plot_marginal(mock_mdm_model, target_node=0, distribution="filt", smooth=True)
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.get_lines()) >= 1


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


def test_plot_title_custom_and_none(mock_mdm_model):
    """Optional title= overrides defaults; None omits ax.set_title."""
    from mdmp.plotting import plot_dag, plot_marginal, plot_stream

    fig = plot_dag(mock_mdm_model, plot_type="graph", title="Custom")
    assert fig.axes[0].get_title() == "Custom"

    fig = plot_dag(mock_mdm_model, plot_type="graph", title=None)
    assert fig.axes[0].get_title() == ""

    fig = plot_dag(mock_mdm_model, plot_type="heatmap", title="Custom heat")
    assert fig.axes[0].get_title() == "Custom heat"

    fig = plot_dag(mock_mdm_model, plot_type="heatmap", title=None)
    assert fig.axes[0].get_title() == ""

    fig = plot_marginal(mock_mdm_model, target_node=0, title="Custom")
    assert fig.axes[0].get_title() == "Custom"

    fig = plot_marginal(mock_mdm_model, target_node=0, title=None)
    assert fig.axes[0].get_title() == ""

    fig = plot_stream(mock_mdm_model, child_node=0, title="Custom")
    assert fig.axes[0].get_title() == "Custom"

    fig = plot_stream(mock_mdm_model, child_node=0, title=None)
    assert fig.axes[0].get_title() == ""


def test_plot_dag_graphviz_style(mock_mdm_model):
    """Graphviz style renders when pydot + dot are available; else clear error."""
    from mdmp.plotting import plot_dag

    try:
        import pydot  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pydot"):
            plot_dag(mock_mdm_model, style="graphviz")
        return

    try:
        fig = plot_dag(
            mock_mdm_model,
            node_labels=["A", "B"],
            style="graphviz",
        )
    except RuntimeError as exc:
        # pydot installed but Graphviz binary missing
        assert "dot" in str(exc).lower() or "graphviz" in str(exc).lower()
        return

    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_dag_unknown_style(mock_mdm_model):
    from mdmp.plotting import plot_dag

    with pytest.raises(ValueError, match="Unknown style"):
        plot_dag(mock_mdm_model, style="plotly")  # type: ignore[arg-type]
