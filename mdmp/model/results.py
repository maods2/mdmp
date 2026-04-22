"""
Result dataclasses for MDM model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class MDMResults:
    """
    Structured results from MDM model fitting.

    This class provides a structured interface to MDM results,
    while maintaining backward compatibility with dictionary access.

    Attributes
    ----------
    adj_mat : np.ndarray
        Adjacency matrix representing the learned DAG structure.
    data : np.ndarray
        Original input data (T x N).
    DF : dict
        Discount factor estimation results.
    Filt : dict
        Filtered dynamic parameters.
    Smoo : dict
        Smoothed dynamic parameters.
    node_names : List[str]
        Names of the nodes/variables.
    """
    adj_mat: np.ndarray
    data: np.ndarray
    DF: Dict[str, Any]
    Filt: Dict[str, Any]
    Smoo: Dict[str, Any]
    node_names: List[str]

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access for backward compatibility."""
        if key == 'adj_mat':
            return self.adj_mat
        elif key == 'data':
            return self.data
        elif key == 'DF':
            return self.DF
        elif key == 'Filt':
            return self.Filt
        elif key == 'Smoo':
            return self.Smoo
        elif key == 'node_names':
            return self.node_names
        else:
            raise KeyError(f"Unknown key: {key}")
