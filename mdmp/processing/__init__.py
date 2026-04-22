"""
Processing abstractions for MDM operations.

This module provides unified interfaces for serial and parallel processing
of nodes, eliminating code duplication.
"""

from .base import NodeProcessor
from .factory import create_processor
from .filtering import FilteringProcessor
from .scoring import ScoringProcessor
from .smoothing import SmoothingProcessor

__all__ = [
    "NodeProcessor",
    "create_processor",
    "FilteringProcessor",
    "SmoothingProcessor",
    "ScoringProcessor",
]
