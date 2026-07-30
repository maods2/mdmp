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
        # verbose=False defaults show_progress=False for pgmpy estimators
        _, kwargs = mock_hc.estimate.call_args
        assert kwargs.get("show_progress") is False


def test_pgmpy_verbosity_context_restores_logger_level():
    """verbose=False temporarily raises pgmpy logger; level restored after."""
    import logging

    from mdmp.structure.algorithms import _pgmpy_verbosity

    logger = logging.getLogger("pgmpy")
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        with _pgmpy_verbosity(False):
            assert logger.level == logging.WARNING
        assert logger.level == logging.INFO
    finally:
        logger.setLevel(previous)


def test_pgmpy_kwargs_respects_explicit_show_progress():
    from mdmp.structure.algorithms import HillClimbingAlgorithm

    algo = HillClimbingAlgorithm(verbose=False)
    out = algo._pgmpy_kwargs({"show_progress": True, "max_iter": 10})
    assert out["show_progress"] is True
    assert out["max_iter"] == 10