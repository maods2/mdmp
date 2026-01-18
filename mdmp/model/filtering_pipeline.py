"""
Filtering pipeline for MDM.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..processing.filtering import FilteringProcessor


class FilteringPipeline:
    """
    Pipeline for filtering operations.

    Encapsulates filtering logic, separating it from the main MDM class.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize filtering pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    def filter_nodes(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        DF_hat: np.ndarray,
        node_names: List[str],
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Filter all nodes.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        adj_mat : np.ndarray
            Adjacency matrix (N x N).
        DF_hat : np.ndarray
            Selected discount factors for each node (N,).
        node_names : list of str
            Names of nodes.
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.

        Returns
        -------
        dict
            Filtered estimates for all nodes.
        """
        if self.verbose:
            print("Computing filtered estimates...")

        processor = FilteringProcessor(n_jobs=n_jobs, verbose=self.verbose)
        return processor.process_nodes(
            data=data,
            adj_mat=adj_mat,
            DF_hat=DF_hat,
            node_names=node_names
        )
