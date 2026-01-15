"""
Structure learning for MDM Bayesian networks.

This module implements various structure learning algorithms including
hill-climbing, tabu search, and other heuristics compatible with MDM scoring.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from scipy.optimize import minimize
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
        **kwargs
    ) -> np.ndarray:
        """
        Learn Bayesian network structure from data.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        method : str, optional
            Learning method. Options: "hc", "tabu", "mmhc", "h2pc", "rsmax2", "ipa".
            Default is "hc".
        nbf : int, optional
            Burn-in time point. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors. Default is np.arange(0.5, 1.01, 0.01).
        **kwargs
            Additional arguments (e.g., gobnilp_path for IPA method).

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        if delta is None:
            delta = get_default_delta()

        if method not in ["hc", "tabu", "mmhc", "h2pc", "rsmax2", "ipa"]:
            raise ValueError(
                f"Unknown method: {method}. "
                "Choose from: hc, tabu, mmhc, h2pc, rsmax2, ipa"
            )

        if method == "ipa":
            return self._learn_ipa(data, nbf=nbf, delta=delta, **kwargs)
        else:
            return self._learn_heuristic(data, method=method, nbf=nbf, delta=delta)

    def _learn_heuristic(
        self,
        data: np.ndarray,
        method: str = "hc",
        nbf: int = 15,
        delta: Optional[np.ndarray] = None
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
            adj_mat = self._hill_climbing(data, nbf=nbf, delta=delta)

        elif method == "tabu":
            adj_mat = self._tabu_search(data, nbf=nbf, delta=delta)

        else:
            # For other methods, use a simplified hill-climbing
            # (Full implementation would require external libraries like pgmpy)
            if self.verbose:
                print(f"Note: {method} not fully implemented, using hill-climbing instead")
            adj_mat = self._hill_climbing(data, nbf=nbf, delta=delta)

        return adj_mat

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

