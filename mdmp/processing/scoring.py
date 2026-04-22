"""
Scoring processor for MDM discount factor selection.
"""

from typing import Dict, Optional

import numpy as np

from ..dlm import dlm_filter
from ..parallel import _worker_select_delta_node
from .factory import create_processor


class ScoringProcessor:
    """
    Processor for evaluating log predictive likelihood.

    Handles both serial and parallel scoring operations.
    """

    def __init__(self, n_jobs: Optional[int] = None, verbose: bool = False):
        """
        Initialize scoring processor.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.
        verbose : bool, optional
            Whether to show progress bars. Default is False.
        """
        self.processor = create_processor(n_jobs=n_jobs, verbose=verbose, default_n_jobs=1)
        self.verbose = verbose

    def evaluate_lpl(
        self,
        delta: np.ndarray,
        design_matrices: Dict[int, np.ndarray],
        target_series: Dict[int, np.ndarray],
        nbf: int
    ) -> np.ndarray:
        """
        Evaluate log predictive likelihood for all (delta, node) combinations.

        Parameters
        ----------
        delta : np.ndarray
            Array of discount factors.
        design_matrices : dict
            Dictionary mapping node index to design matrix.
        target_series : dict
            Dictionary mapping node index to target series.
        nbf : int
            Burn-in time point.

        Returns
        -------
        np.ndarray
            Log predictive likelihoods for each delta and node (nd, Nn).
        """
        nd = len(delta)
        Nn = len(design_matrices)
        lpldet = np.zeros((nd, Nn))

        # Prepare arguments for all (delta, node) combinations
        args_list = []
        for k in range(nd):
            for i in range(Nn):
                args_list.append((
                    k, i, target_series[i], design_matrices[i], delta[k], nbf
                ))

        # Process combinations
        from .parallel import ParallelProcessor
        if isinstance(self.processor, ParallelProcessor):
            # Parallel processing
            results = self.processor.process(
                args_list,
                process_func=_worker_select_delta_node,
                desc="Selecting discount factors (parallel)",
                unit="combinations"
            )
        else:
            # Serial processing
            def process_combination(args):
                k, i, Yt, Ft, delta_k, nbf = args
                # Run DLM filter
                result = dlm_filter(Yt, Ft.T, delta=delta_k)
                lpl_sum = np.sum(result['lpl'][nbf:])
                return (k, i, lpl_sum)

            results = self.processor.process(
                args_list,
                process_func=process_combination,
                desc="Selecting discount factors",
                unit="combinations"
            )

        # Aggregate results
        for k, i, lpl_sum in results:
            lpldet[k, i] = lpl_sum

        return lpldet
