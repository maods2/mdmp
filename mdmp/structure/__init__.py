"""
Structure learning algorithms for MDM Bayesian networks.

This package contains the modular algorithm implementations for structure learning.
"""

from .algorithms import (
    BaseLearningAlgorithm,
    HillClimbingAlgorithm,
    IpaAlgorithm,
    MMHCAlgorithm,
    TabuAlgorithm,
)
from .learner import StructureLearner
from .registry import get_algorithm, list_algorithms, register_algorithm

# Register default algorithms
register_algorithm("hc", HillClimbingAlgorithm)
register_algorithm("tabu", TabuAlgorithm)
register_algorithm("ipa", IpaAlgorithm)  # Registered but not implemented
register_algorithm("mmhc", MMHCAlgorithm)

__all__ = [
    "StructureLearner",
    "BaseLearningAlgorithm",
    "HillClimbingAlgorithm",
    "TabuAlgorithm",
    # "IpaAlgorithm",
    "MMHCAlgorithm",
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
]
