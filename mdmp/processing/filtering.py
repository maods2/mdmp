"""
Filtering processor for MDM nodes.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..dlm import dlm_filter
from ..parallel import _worker_filter_node
from ..utils import build_design_matrix, build_parameter_names, extract_target_series
from .factory import create_processor


class FilteringProcessor:
    """
    Processor for filtering MDM nodes.

    Handles both serial and parallel filtering operations.
    """

    def __init__(self, n_jobs: Optional[int] = None, verbose: bool = False):
        """
        Initialize filtering processor.

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
        data: np.ndarray,
        adj_mat: np.ndarray,
        DF_hat: np.ndarray,
        node_names: List[str]
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

        Returns
        -------
        dict
            Filtered estimates for all nodes.
        """
        Nn = data.shape[1]

        # Prepare arguments for all nodes
        args_list = [
            (i, data, adj_mat, DF_hat, node_names)
            for i in range(Nn)
        ]

        # Process nodes
        from .parallel import ParallelProcessor
        if isinstance(self.processor, ParallelProcessor):
            # Parallel processing
            results = self.processor.process(
                args_list,
                process_func=_worker_filter_node,
                desc="Filtering nodes (parallel)",
                unit="nodes"
            )
        else:
            # Serial processing
            def process_node(args):
                i, data, adj_mat, DF_hat, node_names = args
                # Build design matrix and extract target series
                Ft, parent_list = build_design_matrix(data, adj_mat, i)
                Yt = extract_target_series(data, i)

                # Run DLM filter
                result = dlm_filter(Yt, Ft.T, delta=DF_hat[i])

                # Build parameter names
                param_names = build_parameter_names(i, adj_mat, node_names)

                # Prepare result dictionary
                result_dict = {
                    'mt': result['mt'],
                    'Ct': result['Ct'],
                    'Rt': result['Rt'],
                    'nt': result['nt'],
                    'dt': result['dt'],
                    'ft': result['ft'],
                    'Qt': result['Qt'],
                    'ets': result['ets'],
                    'lpl': result['lpl'],
                }

                return (i, result_dict, param_names)

            results = self.processor.process(
                args_list,
                process_func=process_node,
                desc="Filtering nodes",
                unit="nodes"
            )

        # Reorganize results into dictionaries
        mt = {}
        Ct = {}
        Rt = {}
        nt = {}
        dt = {}
        ft = {}
        Qt = {}
        ets = {}
        lpl = {}
        row_names = {}

        for i, result_dict, param_names in results:
            mt[i] = result_dict['mt']
            Ct[i] = result_dict['Ct']
            Rt[i] = result_dict['Rt']
            nt[i] = result_dict['nt']
            dt[i] = result_dict['dt']
            ft[i] = result_dict['ft']
            Qt[i] = result_dict['Qt']
            ets[i] = result_dict['ets']
            lpl[i] = result_dict['lpl']
            row_names[i] = param_names[:mt[i].shape[0]] if mt[i].ndim == 2 else param_names[:1]

        return {
            'mt': mt,
            'Ct': Ct,
            'Rt': Rt,
            'nt': nt,
            'dt': dt,
            'ft': ft,
            'Qt': Qt,
            'ets': ets,
            'lpl': lpl,
            'row_names': row_names
        }
