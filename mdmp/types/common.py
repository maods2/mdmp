"""
Common types and enums for MDM operations.
"""

from enum import Enum


class ProcessingMode(Enum):
    """Processing mode for MDM operations."""
    SERIAL = "serial"
    PARALLEL = "parallel"
