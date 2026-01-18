"""
Smoothing processor for MDM nodes.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..dlm import dlm_smooth
from ..parallel import _worker_smooth_node
from .factory import create_processor


class SmoothingProcessor:
    """
    Processor for smoothing MDM nodes.

    Handles both serial and parallel smoothing operations.
    """

    def __init__(self, n_jobs: Optional[int] = None, verbose: bool = False):
        """
        Initialize smoothing processor.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.
        verbose : bool, optional
            Whether to show progress bars. Default is False.
        """
        self.processor = create_processor(n_jobs=n_jobs, verbose=verbose, default_n_jobs=1)
        self.verbose = verbose

    def process_nodes(
        self,
        mt: Dict[int, np.ndarray],
        Ct: Dict[int, np.ndarray],
        Rt: Dict[int, np.ndarray],
        nt: Dict[int, np.ndarray],
        dt: Dict[int, np.ndarray]
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

        Returns
        -------
        dict
            Smoothed estimates for all nodes.
        """
        Nn = len(mt)

        # Prepare arguments for all nodes
        args_list = [
            (i, mt, Ct, Rt, nt, dt)
            for i in range(Nn)
        ]

        # Process nodes
        from .parallel import ParallelProcessor
        if isinstance(self.processor, ParallelProcessor):
            # Parallel processing
            results = self.processor.process(
                args_list,
                process_func=_worker_smooth_node,
                desc="Smoothing nodes (parallel)",
                unit="nodes"
            )
        else:
            # Serial processing
            def process_node(args):
                i, mt, Ct, Rt, nt, dt = args
                # Run DLM smooth
                result = dlm_smooth(mt[i], Ct[i], Rt[i], nt[i], dt[i])

                smt = result['smt']
                sCt = result['sCt']

                # Compute standard errors
                if sCt.ndim == 2:  # Single parameter case
                    SE = stats.t.ppf(0.975, nt[i][-1]) * np.sqrt(sCt)
                else:  # Multiple parameters
                    SE_array = np.zeros((sCt.shape[2], sCt.shape[0]))
                    for j in range(sCt.shape[0]):
                        SE_array[:, j] = (
                            stats.t.ppf(0.975, nt[i][-1]) *
                            np.sqrt(sCt[j, j, :])
                        )
                    col_names = [f"SE_{name}" for name in range(sCt.shape[0])]
                    SE = pd.DataFrame(SE_array, columns=col_names)

                result_dict = {
                    'smt': smt,
                    'sCt': sCt,
                    'SE': SE
                }

                return (i, result_dict)

            results = self.processor.process(
                args_list,
                process_func=process_node,
                desc="Smoothing nodes",
                unit="nodes"
            )

        # Reorganize results into dictionaries
        smt = {}
        sCt = {}
        SE = {}

        for i, result_dict in results:
            smt[i] = result_dict['smt']
            sCt[i] = result_dict['sCt']
            SE[i] = result_dict['SE']

        return {
            'smt': smt,
            'sCt': sCt,
            'SE': SE
        }
