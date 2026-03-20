"""
Type definitions for Virtual Typical Subject (VTS) results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class VTSResult:
    """
    Result of Virtual Typical Subject computation.

    Attributes
    ----------
    vts_data : np.ndarray
        The VTS representation. Shape (T x N) for time series or (N,) for
        scalar summary, depending on method.
    method : str
        Method used: "concatenation" or "mean".
    n_subjects : int
        Number of subjects used.
    metadata : dict
        Additional metadata (subject_lengths, node_names, etc.).
    """

    vts_data: np.ndarray
    method: str
    n_subjects: int
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Ensure vts_data is numpy array."""
        if not isinstance(self.vts_data, np.ndarray):
            self.vts_data = np.asarray(self.vts_data)


@dataclass
class ComparisonResult:
    """
    Result of comparing multiple VTS methods.

    Attributes
    ----------
    results : dict
        Mapping of method name to VTSResult.
    comparison_table : pd.DataFrame, optional
        Summary table with metrics per method (e.g., MSE).
    """

    results: Dict[str, VTSResult]
    comparison_table: Optional[pd.DataFrame] = field(default=None)
