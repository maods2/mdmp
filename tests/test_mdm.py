"""
Integration tests for MDM class.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from mdmp import MDM

# Create fake pgmpy module for testing when pgmpy is not installed
if 'pgmpy' not in sys.modules:
    pgmpy_mock = MagicMock()
    pgmpy_mock.estimators = MagicMock()
    sys.modules['pgmpy'] = pgmpy_mock
    sys.modules['pgmpy.estimators'] = pgmpy_mock.estimators


def test_mdm_init_with_numpy_array(sample_data):
    """Test MDM initialization with numpy array."""
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

        model = MDM(small_data, method="hc", nbf=5, verbose=False)

        assert hasattr(model, 'adj_mat')
        assert hasattr(model, 'data')
        assert hasattr(model, 'DF')
        assert hasattr(model, 'Filt')
        assert hasattr(model, 'Smoo')
        assert hasattr(model, 'node_names')

        assert model.adj_mat.shape == (2, 2)
        assert model.data.shape == small_data.shape


def test_mdm_init_with_dataframe(sample_data):
    """Test MDM initialization with pandas DataFrame."""
    small_data = sample_data[:30, :2]
    df = pd.DataFrame(small_data, columns=["Node1", "Node2"])

    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        model = MDM(df, method="hc", nbf=5, verbose=False)

        assert model.node_names == ["Node1", "Node2"]


def test_mdm_init_invalid_data():
    """Test MDM initialization with invalid data."""
    # 1D array should raise error
    with pytest.raises(ValueError, match="must be a 2D array"):
        MDM(np.array([1, 2, 3]), method="hc", verbose=False)

    # Wrong type should raise error
    with pytest.raises(TypeError, match="must be a numpy array or pandas DataFrame"):
        MDM("not valid data", method="hc", verbose=False)


def test_mdm_init_different_methods(sample_data):
    """Test MDM with different structure learning methods."""
    small_data = sample_data[:30, :2]

    # Test tabu method (doesn't need pgmpy)
    model = MDM(small_data, method="tabu", nbf=5, verbose=False, max_iter=3)
    assert hasattr(model, 'adj_mat')


def test_mdm_repr(sample_data):
    """Test MDM string representation."""
    small_data = sample_data[:30, :2]

    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        model = MDM(small_data, method="hc", nbf=5, verbose=False)

        repr_str = repr(model)
        assert "MDM" in repr_str
        assert "nodes" in repr_str
        assert "time_points" in repr_str


def test_mdm_attributes(sample_data):
    """Test that MDM has all expected attributes after initialization."""
    small_data = sample_data[:30, :2]

    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        model = MDM(small_data, method="hc", nbf=5, verbose=False)

        # Check DF structure
        assert 'DF_hat' in model.DF
        assert 'lpldet' in model.DF

        # Check Filt structure
        assert 'mt' in model.Filt
        assert 'Ct' in model.Filt
        assert 'Rt' in model.Filt

        # Check Smoo structure
        assert 'smt' in model.Smoo
        assert 'sCt' in model.Smoo
        assert 'SE' in model.Smoo


def test_mdm_custom_delta(sample_data):
    """Test MDM with custom discount factor sequence."""
    small_data = sample_data[:30, :2]
    custom_delta = np.array([0.8, 0.9, 0.95, 1.0])

    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc, \
         patch('pgmpy.estimators.StructureScore'):
        mock_model = Mock()
        mock_model.edges.return_value = []

        mock_hc_instance = Mock()
        mock_hc_instance.estimate.return_value = mock_model
        mock_hc.return_value = mock_hc_instance

        model = MDM(small_data, method="hc", delta=custom_delta, nbf=5, verbose=False)

        # Check that custom delta was used
        assert model.delta is not None
        assert len(model.delta) == len(custom_delta)
