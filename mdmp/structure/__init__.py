"""
Structure learning algorithms for MDM Bayesian networks.

This package contains the modular algorithm implementations for structure learning.
"""

from .algorithms import (
    BaseLearningAlgorithm,
    HillClimbingAlgorithm,
    IpaAlgorithm,
    MMHCAlgorithm,
)
from .learner import StructureLearner
from .registry import get_algorithm, list_algorithms, register_algorithm

# Register default algorithms
register_algorithm("hc", HillClimbingAlgorithm)
register_algorithm("ipa", IpaAlgorithm)
register_algorithm("mmhc", MMHCAlgorithm)

__all__ = [
    "StructureLearner",
    "BaseLearningAlgorithm",
    "HillClimbingAlgorithm",
    "IpaAlgorithm",
    "MMHCAlgorithm",
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
]
