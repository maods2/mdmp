"""
Multiregression Dynamic Model (MDM) - Main interface and model class.

This module implements the main MDM class that coordinates structure learning,
discount factor selection, filtering, and smoothing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional, List, Dict, Any
from .dlm import dlm_filter, dlm_smooth
from .structure import StructureLearner
from .scoring import select_discount_factors, compute_logpl
from .utils import (
    build_design_matrix,
    extract_target_series,
    build_parameter_names,
    get_default_delta,
    DEFAULT_NBF
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
            Method for structure learning. Options: "hc", "tabu", "mmhc", "h2pc", "rsmax2", "ipa".
            Default is "hc".
        nbf : int, optional
            Burn-in time point for log predictive likelihood calculation. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors for optimization. Default is np.arange(0.5, 1.01, 0.01).
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        **kwargs
            Additional arguments passed to StructureLearner (e.g., gobnilp_path for IPA method).
        """
        # Convert input to numpy array
        if isinstance(data, pd.DataFrame):
            self.node_names = list(data.columns)
            data = data.values
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("data must be a 2D array (T x N)")
            self.node_names = [f"V{i+1}" for i in range(data.shape[1])]
        else:
            raise TypeError("data must be a numpy array or pandas DataFrame")

        self.data = data
        self.verbose = verbose
        self.nbf = nbf

        if delta is None:
            delta = get_default_delta()
        self.delta = delta

        # Learn structure
        if self.verbose:
            print(f"Learning structure using method: {method}")
        structure_learner = StructureLearner(verbose=verbose)
        self.adj_mat = structure_learner.learn_structure(
            data, method=method, nbf=nbf, delta=delta, **kwargs
        )

        # Set node names in adjacency matrix
        if self.node_names:
            n = len(self.node_names)
            if self.adj_mat.shape[0] == n:
                self.adj_mat = pd.DataFrame(
                    self.adj_mat, index=self.node_names, columns=self.node_names
                ).values

        # Select discount factors
        if self.verbose:
            print("Selecting discount factors...")
        df_result = select_discount_factors(self.data, self.adj_mat, nbf=nbf, delta=delta)
        self.DF = df_result

        # Filter
        if self.verbose:
            print("Computing filtered estimates...")
        self.Filt = self._mdm_filter(self.data, self.adj_mat, df_result['DF_hat'])

        # Smooth
        if self.verbose:
            print("Computing smoothed estimates...")
        self.Smoo = self._mdm_smooth(
            self.Filt['mt'], self.Filt['Ct'], self.Filt['Rt'],
            self.Filt['nt'], self.Filt['dt']
        )

    def _mdm_filter(
        self,
        data: np.ndarray,
        adj_mat: np.ndarray,
        DF_hat: np.ndarray
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

        Returns
        -------
        dict
            Filtered estimates for all nodes.
        """
        Nn = data.shape[1]  # Number of nodes
        Nt = data.shape[0]  # Number of time points

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
        dt: Dict[int, np.ndarray]
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

        Returns
        -------
        dict
            Smoothed estimates for all nodes.
        """
        Nn = len(mt)
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

