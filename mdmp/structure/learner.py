"""
Structure learning for MDM Bayesian networks.

This module implements the StructureLearner class that coordinates
various structure learning algorithms.
"""

from typing import List, Optional

import numpy as np

from ..utils import get_default_delta
from .registry import get_algorithm, list_algorithms


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
        **kwargs
    ) -> np.ndarray:
        """
        Learn Bayesian network structure from data.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        method : str, optional
            Learning method. Options: "hc", "tabu", "mmhc".
            Default is "hc".

            - "hc": Hill-climbing using pgmpy (requires pgmpy)
            - "tabu": Tabu search using pgmpy (requires pgmpy)
            - "mmhc": Max-Min Hill-Climbing using pgmpy (requires pgmpy)

            Note: Methods "ipa", "h2pc", and "rsmax2" are not yet implemented.
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

        # Get algorithm from registry
        try:
            algorithm_class = get_algorithm(method)
        except ValueError as e:
            # Provide helpful error message with available algorithms
            available = list_algorithms()
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: {', '.join(available) if available else 'none'}. "
                "Note: Methods h2pc and rsmax2 are not yet implemented."
            ) from e

        # Create algorithm instance and learn structure
        algorithm = algorithm_class(verbose=self.verbose)
        return algorithm.learn(
            data=data,
            nbf=nbf,
            delta=delta,
            node_names=node_names,
            **kwargs
        )

