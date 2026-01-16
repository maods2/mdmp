"""
Tests for structure learning functions.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from mdmp.structure import StructureLearner

# Create fake pgmpy module for testing when pgmpy is not installed
if 'pgmpy' not in sys.modules:
    pgmpy_mock = MagicMock()
    pgmpy_mock.estimators = MagicMock()
    sys.modules['pgmpy'] = pgmpy_mock
    sys.modules['pgmpy.estimators'] = pgmpy_mock.estimators


def test_structure_learner_init():
    """Test StructureLearner initialization."""
    learner = StructureLearner(verbose=True)
    assert learner.verbose is True

    learner = StructureLearner(verbose=False)
    assert learner.verbose is False


def test_has_cycle_no_cycle(small_dag_adjacency):
    """Test cycle detection on acyclic graph."""
    learner = StructureLearner(verbose=False)
    assert not learner._has_cycle(small_dag_adjacency)


def test_has_cycle_with_cycle():
    """Test cycle detection on cyclic graph."""
    learner = StructureLearner(verbose=False)
    # Create a cyclic graph: 0 -> 1 -> 2 -> 0
    cyclic_adj = np.zeros((3, 3), dtype=int)
    cyclic_adj[0, 1] = 1
    cyclic_adj[1, 2] = 1
    cyclic_adj[2, 0] = 1

    assert learner._has_cycle(cyclic_adj)


def test_has_cycle_self_loop():
    """Test cycle detection with self-loop."""
    learner = StructureLearner(verbose=False)
    adj = np.zeros((2, 2), dtype=int)
    adj[0, 0] = 1  # Self-loop

    # Self-loop is considered a cycle
    assert learner._has_cycle(adj)


def test_generate_edge_operations(small_dag_adjacency):
    """Test edge operation generation."""
    learner = StructureLearner(verbose=False)
    N = small_dag_adjacency.shape[0]

    candidates = learner._generate_edge_operations(small_dag_adjacency, N)

    assert isinstance(candidates, list)
    assert len(candidates) > 0

    # Each candidate should be a tuple (adj_matrix, needs_cycle_check)
    for candidate_adj, needs_check in candidates:
        assert isinstance(candidate_adj, np.ndarray)
        assert candidate_adj.shape == (N, N)
        assert isinstance(needs_check, bool)


def test_compute_total_score(sample_data, small_dag_adjacency):
    """Test total score computation."""
    learner = StructureLearner(verbose=False)

    score = learner._compute_total_score(
        sample_data,
        small_dag_adjacency,
        nbf=15
    )

    assert np.isfinite(score)
    assert isinstance(score, (float, np.floating))


def test_learn_structure_invalid_method(sample_data):
    """Test that invalid method raises error."""
    learner = StructureLearner(verbose=False)

    with pytest.raises(ValueError, match="Unknown method"):
        learner.learn_structure(sample_data, method="invalid_method")


def test_learn_structure_ipa_not_implemented(sample_data):
    """Test that IPA method raises NotImplementedError."""
    learner = StructureLearner(verbose=False)

    with pytest.raises(NotImplementedError):
        learner.learn_structure(sample_data, method="ipa")


def test_learn_structure_hc_with_pgmpy(sample_data):
    """Test hill-climbing with pgmpy (mocked)."""
    # Patch where it's imported, not where it's defined
    with patch('pgmpy.estimators.HillClimbSearch') as mock_hc_class, \
         patch('pgmpy.estimators.StructureScore') as mock_score_class:

        learner = StructureLearner(verbose=False)

        # Mock pgmpy components
        mock_model = Mock()
        mock_model.edges.return_value = [("V1", "V2"), ("V2", "V3")]

        mock_hc = Mock()
        mock_hc.estimate.return_value = mock_model
        mock_hc_class.return_value = mock_hc

        # Mock the score class
        mock_score_instance = Mock()
        mock_score_class.return_value = mock_score_instance

        result = learner.learn_structure(sample_data, method="hc")

        assert isinstance(result, np.ndarray)
        assert result.shape == (sample_data.shape[1], sample_data.shape[1])


def test_learn_structure_tabu(sample_data):
    """Test tabu search method."""
    learner = StructureLearner(verbose=False)

    # Use a small dataset for faster execution
    small_data = sample_data[:30, :]

    result = learner.learn_structure(small_data, method="tabu", max_iter=5)

    assert isinstance(result, np.ndarray)
    assert result.shape == (small_data.shape[1], small_data.shape[1])
    assert not learner._has_cycle(result)  # Should be acyclic


def test_edges_to_adjmat():
    """Test conversion from edges to adjacency matrix."""
    learner = StructureLearner(verbose=False)

    # Edges should be tuples of node names (strings), not indices
    edges = [("V1", "V2"), ("V2", "V3")]
    columns = ["V1", "V2", "V3"]

    adj = learner._edges_to_adjmat(edges, columns)

    assert adj.shape == (3, 3)
    assert adj[0, 1] == 1  # V1 -> V2
    assert adj[1, 2] == 1  # V2 -> V3
    assert adj[0, 2] == 0  # No direct edge


def test_extract_adj_from_model_dict():
    """Test extracting adjacency from dict model."""
    learner = StructureLearner(verbose=False)

    # Test with dict containing adjmat
    adj_mat = np.array([[0, 1], [0, 0]])
    model = {"adjmat": adj_mat}
    columns = ["V1", "V2"]

    result = learner._extract_adj_from_model(model, columns)
    assert np.array_equal(result, adj_mat)


def test_extract_adj_from_model_edges():
    """Test extracting adjacency from model with edges."""
    learner = StructureLearner(verbose=False)

    # Test with dict containing edges (should be node names)
    model = {"edges": [("V1", "V2")]}
    columns = ["V1", "V2"]

    result = learner._extract_adj_from_model(model, columns)
    assert result[0, 1] == 1
    assert result[1, 0] == 0


def test_extract_adj_from_model_pgmpy():
    """Test extracting adjacency from pgmpy model."""
    learner = StructureLearner(verbose=False)

    # Mock pgmpy model (edges should be node names)
    mock_model = Mock()
    mock_model.edges.return_value = [("V1", "V2"), ("V2", "V3")]
    columns = ["V1", "V2", "V3"]

    result = learner._extract_adj_from_model(mock_model, columns)

    assert result.shape == (3, 3)
    assert result[0, 1] == 1  # V1 -> V2
    assert result[1, 2] == 1  # V2 -> V3
