"""
Structure learning for MDM Bayesian networks.

This module implements the StructureLearner class that coordinates
various structure learning algorithms.
"""

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..scoring import compute_structure_score
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
            Learning method. Options: "hc", "tabu", "ipa", "mmhc", "notears".
            Default is "hc".
            
            - "hc": Hill-climbing using pgmpy (requires pgmpy)
            - "tabu": Tabu search
            - "ipa": Integer Programming Approach (not yet implemented)
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

        # Get algorithm from registry
        try:
            algorithm_class = get_algorithm(method)
        except ValueError as e:
            # Provide helpful error message with available algorithms
            available = list_algorithms()
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: {', '.join(available) if available else 'none'}. "
                "Note: Methods mmhc, h2pc, and rsmax2 are not yet implemented."
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

    def _has_cycle(self, adj_mat: np.ndarray) -> bool:
        """
        Check if adjacency matrix contains cycles (detect DAG violations).

        This method is provided for backward compatibility with tests.

        Parameters
        ----------
        adj_mat : np.ndarray
            Adjacency matrix.

        Returns
        -------
        bool
            True if cycle exists, False otherwise.
        """
        N = adj_mat.shape[0]
        visited = [False] * N
        rec_stack = [False] * N

        def has_cycle_util(node):
            visited[node] = True
            rec_stack[node] = True

            for neighbor in range(N):
                if adj_mat[node, neighbor] == 1:
                    if not visited[neighbor]:
                        if has_cycle_util(neighbor):
                            return True
                    elif rec_stack[neighbor]:
                        return True

            rec_stack[node] = False
            return False

        for node in range(N):
            if not visited[node]:
                if has_cycle_util(node):
                    return True

        return False

    def _compute_total_score(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute total MDM score for a given structure.

        This method is provided for backward compatibility with tests.
        It uses the unified scoring function.

        Parameters
        ----------
        data : np.ndarray
            Time series data.
        adj_mat : np.ndarray
            Adjacency matrix.
        nbf : int
            Burn-in time point.
        delta : np.ndarray, optional
            Sequence of discount factors.

        Returns
        -------
        float
            Total score (sum of log predictive likelihoods).
        """
        if delta is None:
            delta = get_default_delta()
        return compute_structure_score(data, adj_mat, nbf=nbf, delta=delta)

    def _generate_edge_operations(
        self,
        adj_mat: np.ndarray,
        N: int
    ) -> List[Tuple[np.ndarray, bool]]:
        """
        Generate candidate adjacency matrices by trying all edge operations.

        This method is provided for backward compatibility with tests.

        Parameters
        ----------
        adj_mat : np.ndarray
            Current adjacency matrix.
        N : int
            Number of nodes.

        Returns
        -------
        List[Tuple[np.ndarray, bool]]
            List of (candidate_adjacency_matrix, needs_cycle_check) tuples.
        """
        candidates = []

        # Try adding edges
        for i in range(N):
            for j in range(N):
                if i != j and adj_mat[i, j] == 0:
                    test_adj = adj_mat.copy()
                    test_adj[i, j] = 1
                    candidates.append((test_adj, True))  # Needs cycle check

        # Try removing edges
        for i in range(N):
            for j in range(N):
                if adj_mat[i, j] == 1:
                    test_adj = adj_mat.copy()
                    test_adj[i, j] = 0
                    candidates.append((test_adj, False))  # No cycle check needed

        # Try reversing edges
        for i in range(N):
            for j in range(N):
                if adj_mat[i, j] == 1:
                    test_adj = adj_mat.copy()
                    test_adj[i, j] = 0
                    test_adj[j, i] = 1
                    candidates.append((test_adj, True))  # Needs cycle check

        return candidates

    def _extract_adj_from_model(
        self,
        model: Any,
        columns: List[str]
    ) -> np.ndarray:
        """
        Extract adjacency matrix from a pgmpy model or dict.

        This method is provided for backward compatibility with tests.

        Parameters
        ----------
        model : Any
            pgmpy model or dict containing model information.
        columns : List[str]
            Column names for the nodes.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        if isinstance(model, dict):
            adj = model.get("adjmat")
            if adj is not None:
                if isinstance(adj, pd.DataFrame):
                    return adj.loc[columns, columns].to_numpy(dtype=int)
                return np.array(adj, dtype=int)
            if "model_edges" in model and model["model_edges"] is not None:
                edges = model["model_edges"]
            elif "edges" in model and model["edges"] is not None:
                edges = model["edges"]
            elif "model" in model and model["model"] is not None:
                edges = list(model["model"].edges())
            else:
                edges = []
        else:
            edges = list(model.edges())

        return self._edges_to_adjmat(edges, columns)

    def _edges_to_adjmat(
        self,
        edges: Iterable[Tuple[str, str]],
        columns: List[str]
    ) -> np.ndarray:
        """
        Convert edges to adjacency matrix.

        This method is provided for backward compatibility with tests.

        Parameters
        ----------
        edges : Iterable[Tuple[str, str]]
            List of edges as (parent, child) tuples.
        columns : List[str]
            Column names for the nodes.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        node_idx = {name: idx for idx, name in enumerate(columns)}
        N = len(columns)
        adj = np.zeros((N, N), dtype=int)
        for parent, child in edges:
            if parent in node_idx and child in node_idx:
                adj[node_idx[parent], node_idx[child]] = 1
        return adj
