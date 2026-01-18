"""
Structure learning pipeline for MDM.
"""

from typing import List, Literal, Optional

import numpy as np

from ..structure import StructureLearner
from ..utils import get_default_delta


class StructureLearningPipeline:
    """
    Pipeline for structure learning operations.

    Encapsulates structure learning logic, separating it from the main MDM class.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize structure learning pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose
        self.structure_learner = StructureLearner(verbose=verbose)

    def learn_structure(
        self,
        data: np.ndarray,
        method: Literal["hc", "tabu", "mmhc"] = "hc",
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Learn Bayesian network structure from data.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        method : {"hc", "tabu", "mmhc"}, optional
            Learning method. Default is "hc".
        nbf : int, optional
            Burn-in time point. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors. Default is np.arange(0.5, 1.01, 0.01).
        node_names : list of str, optional
            Node/variable names.
        **kwargs
            Additional arguments for structure learning algorithm.

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        if delta is None:
            delta = get_default_delta()

        if self.verbose:
            print(f"Learning structure using method: {method}")

        adj_mat = self.structure_learner.learn_structure(
            data=data,
            method=method,
            nbf=nbf,
            delta=delta,
            node_names=node_names,
            **kwargs
        )

        return adj_mat
