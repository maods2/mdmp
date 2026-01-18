"""
Type definitions and dataclasses for MDM operations.

This module provides structured data types to replace nested dictionaries,
improving type safety and code clarity.
"""

from .common import ProcessingMode
from .filtering import FilteringResult, NodeFilteringResult
from .scoring import DiscountFactorResult, ScoreResult
from .smoothing import NodeSmoothingResult, SmoothingResult

__all__ = [
    "FilteringResult",
    "NodeFilteringResult",
    "SmoothingResult",
    "NodeSmoothingResult",
    "DiscountFactorResult",
    "ScoreResult",
    "ProcessingMode",
]
