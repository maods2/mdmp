"""
Type definitions for filtering results.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class NodeFilteringResult:
    """
    Filtering result for a single node.

    Attributes
    ----------
    mt : np.ndarray
        Posterior means (p, T).
    Ct : np.ndarray
        Posterior variances (p, p, T).
    Rt : np.ndarray
        Prior variances (p, p, T).
    nt : np.ndarray
        Hyperparameters of precision (T,).
    dt : np.ndarray
        Hyperparameters of precision (T,).
    ft : np.ndarray
        One-step forecasts (T,).
    Qt : np.ndarray
        Forecast variances (T,).
    ets : np.ndarray
        Standardized errors (T,).
    lpl : np.ndarray
        Log predictive likelihood (T,).
    param_names : List[str]
        Parameter names for this node.
    """
    mt: np.ndarray
    Ct: np.ndarray
    Rt: np.ndarray
    nt: np.ndarray
    dt: np.ndarray
    ft: np.ndarray
    Qt: np.ndarray
    ets: np.ndarray
    lpl: np.ndarray
    param_names: List[str]


@dataclass
class FilteringResult:
    """
    Filtering results for all nodes.

    Attributes
    ----------
    nodes : Dict[int, NodeFilteringResult]
        Dictionary mapping node index to filtering result.
    """
    nodes: Dict[int, NodeFilteringResult]

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format for backward compatibility.

        Returns
        -------
        dict
            Dictionary with keys: mt, Ct, Rt, nt, dt, ft, Qt, ets, lpl, row_names
        """
        result = {
            'mt': {},
            'Ct': {},
            'Rt': {},
            'nt': {},
            'dt': {},
            'ft': {},
            'Qt': {},
            'ets': {},
            'lpl': {},
            'row_names': {}
        }

        for node_idx, node_result in self.nodes.items():
            result['mt'][node_idx] = node_result.mt
            result['Ct'][node_idx] = node_result.Ct
            result['Rt'][node_idx] = node_result.Rt
            result['nt'][node_idx] = node_result.nt
            result['dt'][node_idx] = node_result.dt
            result['ft'][node_idx] = node_result.ft
            result['Qt'][node_idx] = node_result.Qt
            result['ets'][node_idx] = node_result.ets
            result['lpl'][node_idx] = node_result.lpl
            result['row_names'][node_idx] = node_result.param_names

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "FilteringResult":
        """
        Create from dictionary format.

        Parameters
        ----------
        data : dict
            Dictionary with keys: mt, Ct, Rt, nt, dt, ft, Qt, ets, lpl, row_names

        Returns
        -------
        FilteringResult
            FilteringResult instance.
        """
        nodes = {}
        for node_idx in data['mt'].keys():
            nodes[node_idx] = NodeFilteringResult(
                mt=data['mt'][node_idx],
                Ct=data['Ct'][node_idx],
                Rt=data['Rt'][node_idx],
                nt=data['nt'][node_idx],
                dt=data['dt'][node_idx],
                ft=data['ft'][node_idx],
                Qt=data['Qt'][node_idx],
                ets=data['ets'][node_idx],
                lpl=data['lpl'][node_idx],
                param_names=data.get('row_names', {}).get(node_idx, [])
            )
        return cls(nodes=nodes)
