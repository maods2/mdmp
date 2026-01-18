"""
Structure learning algorithms for MDM Bayesian networks.

This module implements the base class and concrete algorithm implementations
using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


class TabuAlgorithm(BaseLearningAlgorithm):
    """
    Tabu search structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's HillClimbSearch with tabu search enabled
    (via tabu_length parameter) and a custom MDM scoring function.
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
        Learn structure using pgmpy's hill-climbing with tabu search and custom MDM score.

        This uses the same implementation as HillClimbingAlgorithm but enables
        tabu search by setting tabu_length in the kwargs.

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
            Additional arguments passed to HillClimbSearch.estimate().
            Common parameters:
            - tabu_length: int, default 100. Length of tabu list.
            - max_iter: int, default 1000000. Maximum iterations.
            - epsilon: float, default 0.0001. Convergence threshold.
            - show_progress: bool, default True. Show progress.

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        try:
            from pgmpy.estimators import HillClimbSearch, StructureScore
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for tabu search algorithm. "
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

        # Set default tabu_length if not provided
        tabu_kwargs = dict(kwargs)
        tabu_kwargs.pop("methodtype", None)
        tabu_kwargs.pop("scoretype", None)
        tabu_kwargs.pop("scoring_method", None)
        
        # Set default tabu_length if not specified
        if "tabu_length" not in tabu_kwargs:
            tabu_kwargs["tabu_length"] = 100

        hc = HillClimbSearch(df)
        model = hc.estimate(scoring_method=mdm_score, **tabu_kwargs)

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


class MMHCAlgorithm(BaseLearningAlgorithm):
    """
    Max-Min Hill-Climbing (MMHC) structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's MmhcEstimator with a custom MDM scoring
    function. MMHC first learns an undirected skeleton via MMPC (Max-Min
    Parents and Children), then orients edges using hill-climbing with
    the custom MDM score.
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
        Learn structure using pgmpy's MMHC with a custom MDM score.

        MMHC first learns a skeleton via conditional independence tests (MMPC),
        then orients edges using hill-climbing with the custom MDM scoring
        function that optimizes the log predictive likelihood per node.
        """
        try:
            from pgmpy.estimators import MmhcEstimator, StructureScore
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for MMHC algorithm. "
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

        # Extract MMHC-specific kwargs
        mmhc_kwargs = dict(kwargs)
        mmhc_kwargs.pop("methodtype", None)
        mmhc_kwargs.pop("scoretype", None)
        mmhc_kwargs.pop("scoring_method", None)

        mmhc = MmhcEstimator(df)
        model = mmhc.estimate(scoring_method=mdm_score, **mmhc_kwargs)

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
