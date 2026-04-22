"""
Type definitions for smoothing results.
"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import pandas as pd


@dataclass
class NodeSmoothingResult:
    """
    Smoothing result for a single node.

    Attributes
    ----------
    smt : np.ndarray
        Smoothed means (p, T).
    sCt : np.ndarray
        Smoothed variances (p, p, T).
    SE : Union[np.ndarray, pd.DataFrame]
        Standard errors. Can be array or DataFrame depending on dimensions.
    """
    smt: np.ndarray
    sCt: np.ndarray
    SE: Union[np.ndarray, pd.DataFrame]


@dataclass
class SmoothingResult:
    """
    Smoothing results for all nodes.

    Attributes
    ----------
    nodes : Dict[int, NodeSmoothingResult]
        Dictionary mapping node index to smoothing result.
    """
    nodes: Dict[int, NodeSmoothingResult]

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format for backward compatibility.

        Returns
        -------
        dict
            Dictionary with keys: smt, sCt, SE
        """
        result = {
            'smt': {},
            'sCt': {},
            'SE': {}
        }

        for node_idx, node_result in self.nodes.items():
            result['smt'][node_idx] = node_result.smt
            result['sCt'][node_idx] = node_result.sCt
            result['SE'][node_idx] = node_result.SE

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "SmoothingResult":
        """
        Create from dictionary format.

        Parameters
        ----------
        data : dict
            Dictionary with keys: smt, sCt, SE

        Returns
        -------
        SmoothingResult
            SmoothingResult instance.
        """
        nodes = {}
        for node_idx in data['smt'].keys():
            nodes[node_idx] = NodeSmoothingResult(
                smt=data['smt'][node_idx],
                sCt=data['sCt'][node_idx],
                SE=data['SE'][node_idx]
            )
        return cls(nodes=nodes)
