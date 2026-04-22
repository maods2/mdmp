"""
Smoothing pipeline for MDM.
"""

from typing import Any, Dict, Optional

import numpy as np

from ..processing.smoothing import SmoothingProcessor


class SmoothingPipeline:
    """
    Pipeline for smoothing operations.

    Encapsulates smoothing logic, separating it from the main MDM class.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize smoothing pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    def smooth_nodes(
        self,
        mt: Dict[int, np.ndarray],
        Ct: Dict[int, np.ndarray],
        Rt: Dict[int, np.ndarray],
        nt: Dict[int, np.ndarray],
        dt: Dict[int, np.ndarray],
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Smooth all nodes.

        Parameters
        ----------
        mt : dict
            Filtered posterior means.
        Ct : dict
            Filtered posterior variances.
        Rt : dict
            Prior variances.
        nt : dict
            Hyperparameters of precision.
        dt : dict
            Hyperparameters of precision.
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.

        Returns
        -------
        dict
            Smoothed estimates for all nodes.
        """
        if self.verbose:
            print("Computing smoothed estimates...")

        processor = SmoothingProcessor(n_jobs=n_jobs, verbose=self.verbose)
        return processor.process_nodes(
            mt=mt,
            Ct=Ct,
            Rt=Rt,
            nt=nt,
            dt=dt
        )
