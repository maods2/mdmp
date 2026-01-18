"""
MDM structure scoring for pgmpy algorithms.
"""

from typing import List

import numpy as np
import pandas as pd

from ..scoring import optimize_local_score


class MdmStructureScore:
    """
    MDM structure score for pgmpy algorithms.

    This class provides a scoring function that can be used with pgmpy's
    structure learning algorithms, replacing the nested class pattern.
    Note: This class does not inherit from StructureScore directly to avoid
    pgmpy dependency in this module. It is wrapped in algorithms.py.
    """

    def __init__(self, df_input: pd.DataFrame, nbf_value: int):
        """
        Initialize MDM structure score.

        Parameters
        ----------
        df_input : pd.DataFrame
            Input data frame.
        nbf_value : int
            Burn-in time point.
        """
        self._data_np = df_input.to_numpy()
        self._nbf = nbf_value
        self._node_to_idx = {
            name: idx for idx, name in enumerate(df_input.columns)
        }
        self._num_nodes = len(self._node_to_idx)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Compute local score for a variable given its parents.

        Parameters
        ----------
        variable : str
            Variable name.
        parents : list of str
            List of parent variable names.

        Returns
        -------
        float
            Optimized local score.
        """
        node_idx = self._node_to_idx[variable]
        adj = np.zeros((self._num_nodes, self._num_nodes), dtype=int)
        for parent in parents:
            adj[self._node_to_idx[parent], node_idx] = 1

        optimized_score, _ = optimize_local_score(
            self._data_np,
            adj,
            node_idx,
            nbf=self._nbf
        )
        return optimized_score
