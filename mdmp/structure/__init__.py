"""
Structure learning algorithms for MDM Bayesian networks.

This package contains the modular algorithm implementations for structure learning.
"""

from .algorithms import (
    BaseLearningAlgorithm,
    HillClimbingAlgorithm,
    IpaAlgorithm,
    MMHCAlgorithm,
    NotearsAlgorithm,
)
from .learner import StructureLearner
from .registry import get_algorithm, list_algorithms, register_algorithm

# Register default algorithms
register_algorithm("hc", HillClimbingAlgorithm)
register_algorithm("ipa", IpaAlgorithm)
register_algorithm("mmhc", MMHCAlgorithm)
register_algorithm("notears", NotearsAlgorithm)

__all__ = [
    "StructureLearner",
    "BaseLearningAlgorithm",
    "HillClimbingAlgorithm",
    "IpaAlgorithm",
    "MMHCAlgorithm",
    "NotearsAlgorithm",
    "register_algorithm",
    "get_algorithm",
    "list_algorithms",
]
