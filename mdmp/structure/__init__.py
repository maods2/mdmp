"""
Structure learning algorithms for MDM Bayesian networks.

This package contains the modular algorithm implementations for structure learning.
"""

from .algorithms import (
    BaseLearningAlgorithm,
    HillClimbingAlgorithm,
    IpaAlgorithm,
    TabuSearchAlgorithm,
)
from .learner import StructureLearner
from .registry import get_algorithm, list_algorithms, register_algorithm

# Register default algorithms
register_algorithm("hc", HillClimbingAlgorithm)
register_algorithm("tabu", TabuSearchAlgorithm)
register_algorithm("ipa", IpaAlgorithm)

__all__ = [
    "StructureLearner",
    "BaseLearningAlgorithm",
    "HillClimbingAlgorithm",
    "TabuSearchAlgorithm",
    "IpaAlgorithm",
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
]
