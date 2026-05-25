"""
Structure learning for MDM Bayesian networks.

This module implements the StructureLearner class that coordinates
various structure learning algorithms.
"""

from typing import Dict, List, Optional, Type

import numpy as np

from ..utils import get_default_delta
from .algorithms import (
    BaseLearningAlgorithm,
    HillClimbingAlgorithm,
    IpaAlgorithm,
    MMHCAlgorithm,
    NotearsAlgorithm,
    TabuAlgorithm,
)

METHODS: Dict[str, Type[BaseLearningAlgorithm]] = {
    "hc": HillClimbingAlgorithm,
    "tabu": TabuAlgorithm,
    "mmhc": MMHCAlgorithm,
    "notears": NotearsAlgorithm,
    "ipa": IpaAlgorithm,
}


class StructureLearner:
    """
    Structure learning for MDM Bayesian networks.
    Supports multiple learning algorithms compatible with MDM scoring functions.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize structure learner.
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    def learn_structure(
        self,
        data: np.ndarray,
        method: str = "hc",
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Learn Bayesian network structure from data.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        method : str, optional
            Learning method. Options: "hc", "tabu", "ipa", "mmhc", "notears".
            Default is "hc".

            - "hc": Hill-climbing using pgmpy (requires pgmpy)
            - "tabu": Tabu search using pgmpy (requires pgmpy)
            - "mmhc": Max-Min Hill-Climbing using pgmpy (requires pgmpy)
            - "notears": NOTEARS linear (requires notears)
        nbf : int, optional
            Burn-in time point. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors. Default is np.arange(0.5, 1.01, 0.01).
        node_names : list of str, optional
            Node/variable names to use when building labeled data frames.
        **kwargs
            Additional arguments (e.g., pgmpy hill-climbing options when method="hc").

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        if delta is None:
            delta = get_default_delta()

        try:
            algorithm_class = METHODS[method]

        except KeyError:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: {', '.join(sorted(METHODS))}. "
                "Note: Methods h2pc and rsmax2 are not yet implemented."
            ) from None

        algorithm = algorithm_class(verbose=self.verbose)

        return algorithm.learn(data=data, nbf=nbf, delta=delta, node_names=node_names, **kwargs)
