"""
Discount factor selection pipeline for MDM.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..scoring import select_discount_factors


class DiscountFactorSelector:
    """
    Pipeline for discount factor selection operations.

    Encapsulates discount factor selection logic.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize discount factor selector.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    def select_discount_factors(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select discount factors that maximize log predictive likelihood.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        adj_mat : np.ndarray
            Adjacency matrix (N x N).
        nbf : int, optional
            Burn-in time point. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors. Default is np.arange(0.5, 1.01, 0.01).
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.

        Returns
        -------
        dict
            Dictionary containing:
            - lpldet : Log predictive likelihoods for each delta and node (nd, N)
            - DF_hat : Selected discount factors for each node (N,)
        """
        if self.verbose:
            print("Selecting discount factors...")

        return select_discount_factors(
            data=data,
            adj_mat=adj_mat,
            nbf=nbf,
            delta=delta,
            n_jobs=n_jobs,
            verbose=self.verbose
        )
