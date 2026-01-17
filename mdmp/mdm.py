"""
Multiregression Dynamic Model (MDM) - Main interface and model class.

This module implements the main MDM class that coordinates structure learning,
discount factor selection, filtering, and smoothing.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .dlm import dlm_filter, dlm_smooth
from .parallel import _get_n_jobs, _worker_filter_node, _worker_smooth_node
from .scoring import select_discount_factors
from .structure import StructureLearner
from .utils import (
    build_design_matrix,
    build_parameter_names,
    extract_target_series,
    get_default_delta,
)


class MDM:
    """
    Multiregression Dynamic Model (MDM) for Bayesian network structure learning
    and dynamic parameter estimation from multivariate time series data.

    Attributes
    ----------
    adj_mat : np.ndarray
        Adjacency matrix representing the learned DAG structure.
    data : np.ndarray
        Original input data (T x N).
    DF : dict
        Discount factor estimation results.
    Filt : dict
        Filtered dynamic parameters.
    Smoo : dict
        Smoothed dynamic parameters.
    node_names : List[str]
        Names of the nodes/variables.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: str = "hc",
        nbf: int = 15,
        delta: Optional[np.ndarray] = None,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize and fit MDM model.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Multivariate time series data. Rows represent time points,
            columns represent nodes. Must be complete (no missing values).
        method : str, optional
            Method for structure learning. Options: "hc", "tabu", "ipa".
            Default is "hc".
            
            Note: Methods "mmhc", "h2pc", and "rsmax2" are not yet implemented.
        nbf : int, optional
            Burn-in time point for log predictive likelihood calculation. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors for optimization. Default is np.arange(0.5, 1.01, 0.01).
            All values must be between 0 and 1.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        n_jobs : int, optional
            Number of parallel jobs for discount factor selection, filtering, and smoothing.
            If None or 1, uses serial processing. If -1, uses all available CPU cores.
            If > 1, uses that many parallel workers. Default is None (serial processing).
        **kwargs
            Additional arguments passed to StructureLearner.
        
        Raises
        ------
        TypeError
            If data is not a numpy array or pandas DataFrame.
        ValueError
            If data dimensions are invalid, delta values are out of range, or
            structure learning method is invalid.
        ImportError
            If method="hc" and pgmpy is not installed (install with: pip install mdmp[hc]).
        NotImplementedError
            If method="ipa" (not yet implemented).
        
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mdmp import MDM
        >>> # Create sample data
        >>> data = np.random.randn(100, 3)
        >>> # Fit MDM model with parallel processing
        >>> model = MDM(data, method="hc", nbf=15, verbose=False, n_jobs=-1)
        >>> # Access results
        >>> print(model.adj_mat.shape)
        (3, 3)
        >>> print(model.DF['DF_hat'])
        """
        # Convert input to numpy array
        if isinstance(data, pd.DataFrame):
            self.node_names = list(data.columns)
            data = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array (T x N), got {data.ndim}D array with shape {data.shape}"
                )
            if data.shape[0] < 2:
                raise ValueError(
                    f"data must have at least 2 time points, got {data.shape[0]}"
                )
            if data.shape[1] < 1:
                raise ValueError(
                    f"data must have at least 1 variable, got {data.shape[1]}"
                )
            self.node_names = [f"V{i+1}" for i in range(data.shape[1])]
        else:
            raise TypeError(
                f"data must be a numpy array or pandas DataFrame, got {type(data).__name__}"
            )

        self.data = data
        self.verbose = verbose
        self.nbf = nbf

        if delta is None:
            delta = get_default_delta()
        else:
            if not isinstance(delta, np.ndarray):
                raise TypeError(f"delta must be a numpy array, got {type(delta).__name__}")
            if len(delta) == 0:
                raise ValueError("delta must not be empty")
            if np.any(delta < 0) or np.any(delta > 1):
                raise ValueError("delta values must be between 0 and 1")
        self.delta = delta

        # Learn structure
        if self.verbose:
            print(f"Learning structure using method: {method}")
        structure_learner = StructureLearner(verbose=verbose)
        self.adj_mat = structure_learner.learn_structure(
            data,
            method=method,
            nbf=nbf,
            delta=delta,
            node_names=self.node_names,
            **kwargs
        )

        # Set node names in adjacency matrix
        if self.node_names:
            n = len(self.node_names)
            if self.adj_mat.shape[0] != n or self.adj_mat.shape[1] != n:
                raise ValueError(
                    f"adjacency matrix shape {self.adj_mat.shape} does not match "
                    f"number of nodes {n}"
                )
            self.adj_mat = pd.DataFrame(
                self.adj_mat, index=self.node_names, columns=self.node_names
            ).values

        # Select discount factors
        if self.verbose:
            print("Selecting discount factors...")
        df_result = select_discount_factors(self.data, self.adj_mat, nbf=nbf, delta=delta, n_jobs=n_jobs)
        self.DF = df_result

        # Filter
        if self.verbose:
            print("Computing filtered estimates...")
        self.Filt = self._mdm_filter(self.data, self.adj_mat, df_result['DF_hat'], n_jobs=n_jobs)

        # Smooth
        if self.verbose:
            print("Computing smoothed estimates...")
        self.Smoo = self._mdm_smooth(
            self.Filt['mt'], self.Filt['Ct'], self.Filt['Rt'],
            self.Filt['nt'], self.Filt['dt'], n_jobs=n_jobs
        )

    def _mdm_filter(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        DF_hat: np.ndarray,
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute MDM filtering for all nodes.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        adj_mat : np.ndarray
            Adjacency matrix (N x N).
        DF_hat : np.ndarray
            Selected discount factors for each node (N,).
        n_jobs : int, optional
            Number of parallel jobs. If None or 1, uses serial processing.
            If -1, uses all available CPU cores. If > 1, uses that many workers.
            Default is None (serial processing).

        Returns
        -------
        dict
            Filtered estimates for all nodes.
        """
        Nn = data.shape[1]  # Number of nodes
        Nt = data.shape[0]  # Number of time points

        # Determine number of jobs
        n_jobs_actual = _get_n_jobs(n_jobs, default=1)

        if n_jobs_actual == 1:
            # Serial processing (original code)
            mt = {}
            Ct = {}
            Rt = {}
            nt = {}
            dt = {}
            ft = {}
            Qt = {}
            ets = {}
            lpl = {}
            row_names = {}  # Store parameter names for each node

            # Find connections
            connections = np.where(adj_mat == 1)

            for i in range(Nn):
                # Build design matrix and extract target series
                Ft, parent_list = build_design_matrix(data, adj_mat, i)
                Yt = extract_target_series(data, i)

                # Run DLM filter
                result = dlm_filter(Yt, Ft.T, delta=DF_hat[i])

                # Store results
                mt[i] = result['mt']
                Ct[i] = result['Ct']
                Rt[i] = result['Rt']
                nt[i] = result['nt']
                dt[i] = result['dt']
                ft[i] = result['ft']
                Qt[i] = result['Qt']
                ets[i] = result['ets']
                lpl[i] = result['lpl']

                # Store parameter names for later use (will be used in plotting)
                param_names = build_parameter_names(i, adj_mat, self.node_names)
                row_names[i] = param_names[:mt[i].shape[0]] if mt[i].ndim == 2 else param_names[:1]
        else:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor

            # Prepare arguments for all nodes
            args_list = [
                (i, data, adj_mat, DF_hat, self.node_names)
                for i in range(Nn)
            ]

            # Process in parallel
            with ProcessPoolExecutor(max_workers=n_jobs_actual) as executor:
                results = list(executor.map(_worker_filter_node, args_list))

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
            'row_names': row_names  # Store parameter names
        }

    def _mdm_smooth(
        self,
        mt: Dict[int, np.ndarray],
        Ct: Dict[int, np.ndarray],
        Rt: Dict[int, np.ndarray],
        nt: Dict[int, np.ndarray],
        dt: Dict[int, np.ndarray],
        n_jobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute MDM smoothing for all nodes.

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
            If -1, uses all available CPU cores. If > 1, uses that many workers.
            Default is None (serial processing).

        Returns
        -------
        dict
            Smoothed estimates for all nodes.
        """
        Nn = len(mt)

        # Determine number of jobs
        n_jobs_actual = _get_n_jobs(n_jobs, default=1)

        if n_jobs_actual == 1:
            # Serial processing (original code)
            smt = {}
            sCt = {}
            SE = {}

            for i in range(Nn):
                # Run DLM smooth
                result = dlm_smooth(mt[i], Ct[i], Rt[i], nt[i], dt[i])

                smt[i] = result['smt']
                sCt[i] = result['sCt']

                # Compute standard errors
                if sCt[i].ndim == 2:  # Single parameter case
                    SE[i] = stats.t.ppf(0.975, nt[i][-1]) * np.sqrt(sCt[i])
                else:  # Multiple parameters
                    SE_array = np.zeros((sCt[i].shape[2], sCt[i].shape[0]))
                    for j in range(sCt[i].shape[0]):
                        SE_array[:, j] = (
                            stats.t.ppf(0.975, nt[i][-1]) *
                            np.sqrt(sCt[i][j, j, :])
                        )
                    col_names = [f"SE_{name}" for name in range(sCt[i].shape[0])]
                    SE[i] = pd.DataFrame(SE_array, columns=col_names)
        else:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor

            # Prepare arguments for all nodes
            args_list = [
                (i, mt, Ct, Rt, nt, dt)
                for i in range(Nn)
            ]

            # Process in parallel
            with ProcessPoolExecutor(max_workers=n_jobs_actual) as executor:
                results = list(executor.map(_worker_smooth_node, args_list))

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

    def __repr__(self) -> str:
        """String representation of MDM object."""
        return (
            f"MDM(nodes={len(self.node_names)}, "
            f"time_points={self.data.shape[0]}, "
            f"edges={np.sum(self.adj_mat)}/{self.adj_mat.size})"
        )

