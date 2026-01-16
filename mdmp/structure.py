"""
Structure learning for MDM Bayesian networks.

This module implements various structure learning algorithms including
hill-climbing, tabu search, and other heuristics compatible with MDM scoring.
"""

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .scoring import compute_logpl, select_discount_factors
from .utils import get_default_delta


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
            Learning method. Options: "hc", "tabu", "ipa".
            Default is "hc".
            
            Note: Methods "mmhc", "h2pc", and "rsmax2" are not yet implemented.
            Use "hc" (hill-climbing) or "tabu" (tabu search) for structure learning.
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

        if method not in ["hc", "tabu", "ipa"]:
            raise ValueError(
                f"Unknown method: {method}. "
                "Choose from: hc, tabu, ipa. "
                "Note: Methods mmhc, h2pc, and rsmax2 are not yet implemented."
            )

        if method == "ipa":
            return self._learn_ipa(data, nbf=nbf, delta=delta, **kwargs)
        else:
            return self._learn_heuristic(
                data,
                method=method,
                nbf=nbf,
                delta=delta,
                node_names=node_names,
                **kwargs
            )

    def _learn_heuristic(
        self,
        data: np.ndarray,
        method: str = "hc",
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using heuristic search (hill-climbing or tabu search).

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        method : str
            Learning method ("hc" or "tabu").
        nbf : int
            Burn-in time point.
        delta : np.ndarray, optional
            Sequence of discount factors.
        **kwargs
            Additional arguments passed to the underlying bnlearn call when
            method="hc".

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        N = data.shape[1]
        adj_mat = np.zeros((N, N), dtype=int)

        if self.verbose:
            print(f"Learning structure using {method}...")

        # Hill-climbing algorithm
        if method == "hc":
            adj_mat = self._learn_hc_pgmpy(
                data, nbf=nbf, delta=delta, node_names=node_names, **kwargs
            )

        elif method == "tabu":
            adj_mat = self._tabu_search(data, nbf=nbf, delta=delta)
        else:
            raise ValueError(
                f"Method '{method}' is not supported in _learn_heuristic. "
                "This should not happen - please report this as a bug."
            )

        return adj_mat

    def _learn_hc_pgmpy(
        self,
        data: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
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
                "pgmpy is required for method='hc'. Install with `pip install pgmpy`."
            ) from exc

        if delta is None:
            delta = get_default_delta()

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
                self._node_to_idx = {name: idx for idx, name in enumerate(df_input.columns)}
                self._num_nodes = len(self._node_to_idx)

            def local_score(self, variable, parents):
                node_idx = self._node_to_idx[variable]
                adj = np.zeros((self._num_nodes, self._num_nodes), dtype=int)
                for parent in parents:
                    adj[self._node_to_idx[parent], node_idx] = 1

                def objective(delta_value: float) -> float:
                    return compute_logpl(
                        self._data_np,
                        adj,
                        delta_value,
                        node_idx,
                        nbf=self._nbf
                    )

                result = minimize_scalar(
                    objective,
                    bounds=(0.0, 1.0),
                    method="bounded"
                )
                if not result.success or not np.isfinite(result.fun):
                    return -np.inf
                return -result.fun

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

    def _hill_climbing(
        self,
        data: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        max_iter: int = 100
    ) -> np.ndarray:
        """
        Hill-climbing structure learning algorithm.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray, optional
            Sequence of discount factors.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        N = data.shape[1]
        adj_mat = np.zeros((N, N), dtype=int)
        current_score = self._compute_total_score(data, adj_mat, nbf=nbf, delta=delta)

        improved = True
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            best_adj, best_score = adj_mat.copy(), current_score

            # Try all edge operations: add, remove, reverse
            candidates = self._generate_edge_operations(adj_mat, N)

            for candidate_adj, needs_cycle_check in candidates:
                if needs_cycle_check and self._has_cycle(candidate_adj):
                    continue

                score = self._compute_total_score(data, candidate_adj, nbf=nbf, delta=delta)
                if score > best_score:
                    best_score = score
                    best_adj = candidate_adj.copy()
                    improved = True

            adj_mat = best_adj
            current_score = best_score
            iteration += 1

            if self.verbose and improved:
                print(f"Iteration {iteration}: Score = {current_score:.2f}")

        return adj_mat

    def _tabu_search(
        self,
        data: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tabu_size: int = 10
    ) -> np.ndarray:
        """
        Tabu search structure learning algorithm.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray, optional
            Sequence of discount factors.
        max_iter : int
            Maximum number of iterations.
        tabu_size : int
            Size of tabu list.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        N = data.shape[1]
        adj_mat = np.zeros((N, N), dtype=int)
        current_score = self._compute_total_score(data, adj_mat, nbf=nbf, delta=delta)
        best_adj = adj_mat.copy()
        best_score = current_score

        tabu_list = []

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
                                    score = self._compute_total_score(data, test_adj, nbf=nbf, delta=delta)
                                    neighbors.append(test_adj)
                                    neighbor_scores.append(score)
                        else:
                            test_adj[i, j] = 0
                            neighbor_key = (i, j, 'remove')
                            if neighbor_key not in tabu_list:
                                score = self._compute_total_score(data, test_adj, nbf=nbf, delta=delta)
                                neighbors.append(test_adj)
                                neighbor_scores.append(score)

            if not neighbors:
                break

            # Select best neighbor (even if worse than current)
            best_idx = np.argmax(neighbor_scores)
            new_adj = neighbors[best_idx]
            current_score = neighbor_scores[best_idx]

            # Update tabu list with the move that was made
            # Determine what changed
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

    def _compute_total_score(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute total MDM score for a given structure.

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

        df_result = select_discount_factors(data, adj_mat, nbf=nbf, delta=delta)
        total_score = np.sum([np.max(df_result['lpldet'][:, i]) for i in range(data.shape[1])])

        return total_score

    def _generate_edge_operations(
        self,
        adj_mat: np.ndarray,
        N: int
    ) -> List[Tuple[np.ndarray, bool]]:
        """
        Generate candidate adjacency matrices by trying all edge operations.
        
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

    def _learn_ipa(
        self,
        data: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        gobnilp_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Learn structure using Integer Programming Approach (IPA) via GOBNILP.

        Note: This requires GOBNILP to be installed and configured.

        Parameters
        ----------
        data : np.ndarray
            Time series data.
        nbf : int
            Burn-in time point.
        delta : np.ndarray, optional
            Sequence of discount factors.
        gobnilp_path : str, optional
            Path to GOBNILP binary.

        Returns
        -------
        np.ndarray
            Adjacency matrix.
        """
        raise NotImplementedError(
            "IPA method with GOBNILP is not yet implemented in Python. "
            "Please use one of: hc, tabu"
        )

