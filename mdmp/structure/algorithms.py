"""
Structure learning algorithms for MDM Bayesian networks.

This module implements the base class and concrete algorithm implementations
using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..scoring import compute_local_score, compute_structure_score, optimize_local_score
from ..utils import get_default_delta


class BaseLearningAlgorithm(ABC):
    """
    Base class for structure learning algorithms.

    All learning algorithms should inherit from this class and implement
    the `learn` method. The `compute_score` method provides a default
    implementation that can be overridden if needed.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the learning algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    @abstractmethod
    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure and return adjacency matrix.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray
            Sequence of discount factors.
        node_names : list of str, optional
            Node/variable names.
        **kwargs
            Additional algorithm-specific arguments.

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        pass

    def compute_score(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        cache: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute structure score (default implementation).

        This method provides a default implementation using the unified
        scoring function. Algorithms can override this if they need
        custom scoring behavior.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        adj_mat : np.ndarray
            Adjacency matrix (N x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray
            Sequence of discount factors.
        cache : dict, optional
            Optional cache for score computations.

        Returns
        -------
        float
            Total structure score.
        """
        return compute_structure_score(data, adj_mat, nbf=nbf, delta=delta, cache=cache)

    def _has_cycle(self, adj_mat: np.ndarray) -> bool:
        """
        Check if adjacency matrix contains cycles (detect DAG violations).

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


class HillClimbingAlgorithm(BaseLearningAlgorithm):
    """
    Hill-climbing structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's HillClimbSearch with a custom MDM scoring
    function that optimizes the log predictive likelihood per node.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using pgmpy's hill-climbing with a custom MDM score.

        This mirrors the R implementation using bnlearn::hc with a custom score
        function by optimizing the MDM log predictive likelihood per node.
        """
        try:
            from pgmpy.estimators import HillClimbSearch, StructureScore
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for hill-climbing algorithm. "
                "Install with `pip install pgmpy`."
            ) from exc

        N = data.shape[1]
        if node_names is not None:
            if len(node_names) != N:
                raise ValueError(
                    f"node_names length ({len(node_names)}) must match number of columns "
                    f"in data ({N})"
                )
            columns = list(node_names)
        else:
            columns = [f"V{i+1}" for i in range(N)]
        df = pd.DataFrame(data, columns=columns)

        class _MdmStructureScore(StructureScore):
            def __init__(self, df_input: pd.DataFrame, nbf_value: int):
                super().__init__(df_input)
                self._data_np = df_input.to_numpy()
                self._nbf = nbf_value
                self._node_to_idx = {
                    name: idx for idx, name in enumerate(df_input.columns)
                }
                self._num_nodes = len(self._node_to_idx)

            def local_score(self, variable, parents):
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

        mdm_score = _MdmStructureScore(df, nbf_value=nbf)

        hc_kwargs = dict(kwargs)
        hc_kwargs.pop("methodtype", None)
        hc_kwargs.pop("scoretype", None)
        hc_kwargs.pop("scoring_method", None)

        hc = HillClimbSearch(df)
        model = hc.estimate(scoring_method=mdm_score, **hc_kwargs)

        return self._extract_adj_from_model(model, columns)

    def _extract_adj_from_model(
        self,
        model: Any,
        columns: List[str]
    ) -> np.ndarray:
        """
        Extract adjacency matrix from a pgmpy model or dict.
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
        node_idx = {name: idx for idx, name in enumerate(columns)}
        N = len(columns)
        adj = np.zeros((N, N), dtype=int)
        for parent, child in edges:
            if parent in node_idx and child in node_idx:
                adj[node_idx[parent], node_idx[child]] = 1
        return adj


class TabuSearchAlgorithm(BaseLearningAlgorithm):
    """
    Tabu search structure learning algorithm.

    This algorithm uses tabu search to explore the space of DAG structures,
    avoiding recently visited solutions to escape local optima.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using tabu search.
        """
        max_iter = kwargs.get("max_iter", 100)
        tabu_size = kwargs.get("tabu_size", 10)

        N = data.shape[1]
        adj_mat = np.zeros((N, N), dtype=int)
        current_score = self.compute_score(data, adj_mat, nbf, delta)
        best_adj = adj_mat.copy()
        best_score = current_score

        tabu_list = []
        score_cache = {}  # Cache for score computations

        for iteration in range(max_iter):
            neighbors = []
            neighbor_scores = []

            # Generate neighbors
            for i in range(N):
                for j in range(N):
                    if i != j:
                        test_adj = adj_mat.copy()
                        if adj_mat[i, j] == 0:
                            test_adj[i, j] = 1
                            if not self._has_cycle(test_adj):
                                neighbor_key = (i, j, 'add')
                                if neighbor_key not in tabu_list:
                                    score = self.compute_score(
                                        data, test_adj, nbf, delta, cache=score_cache
                                    )
                                    neighbors.append(test_adj)
                                    neighbor_scores.append(score)
                        else:
                            test_adj[i, j] = 0
                            neighbor_key = (i, j, 'remove')
                            if neighbor_key not in tabu_list:
                                score = self.compute_score(
                                    data, test_adj, nbf, delta, cache=score_cache
                                )
                                neighbors.append(test_adj)
                                neighbor_scores.append(score)

            if not neighbors:
                break

            # Select best neighbor (even if worse than current)
            best_idx = np.argmax(neighbor_scores)
            new_adj = neighbors[best_idx]
            current_score = neighbor_scores[best_idx]

            # Update tabu list with the move that was made
            diff = new_adj - adj_mat
            changed = np.where(diff != 0)
            if len(changed[0]) > 0:
                i_move, j_move = changed[0][0], changed[1][0]
                move_type = 'add' if new_adj[i_move, j_move] == 1 else 'remove'
                move_made = (i_move, j_move, move_type)

                if len(tabu_list) >= tabu_size:
                    tabu_list.pop(0)
                tabu_list.append(move_made)

            adj_mat = new_adj

            # Update best if improved
            if current_score > best_score:
                best_score = current_score
                best_adj = adj_mat.copy()

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Score = {current_score:.2f}")

        return best_adj


class IpaAlgorithm(BaseLearningAlgorithm):
    """
    Integer Programming Approach (IPA) structure learning algorithm.

    This algorithm uses GOBNILP for exact structure learning via integer
    programming. Currently not implemented.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using Integer Programming Approach (IPA) via GOBNILP.

        Note: This requires GOBNILP to be installed and configured.
        """
        raise NotImplementedError(
            "IPA method with GOBNILP is not yet implemented in Python. "
            "Please use one of: hc, tabu"
        )
