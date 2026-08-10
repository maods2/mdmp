"""
MDM model — orchestrates structure learning, discount selection, filtering, and smoothing.
"""

import time
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from .._node_dispatch import filter_all_nodes, smooth_all_nodes
from ..scoring import select_discount_factors
from ..structure import StructureLearner
from ..utils import get_default_delta
from ..validation import validate_data, validate_delta


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
    node_names : list of str
        Names of the nodes/variables.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        method: Literal["hc", "tabu", "mmhc"] = "hc",
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
            Multivariate time series data. Rows are time points, columns are nodes.
        method : {"hc", "tabu", "mmhc"}, optional
            Method for structure learning. Default is "hc".
        nbf : int, optional
            Burn-in time point for log predictive likelihood. Default is 15.
        delta : np.ndarray, optional
            Discount factor grid. Default is np.arange(0.5, 1.01, 0.01).
        verbose : bool, optional
            Print progress messages. Default is True.
            When False, also raises the ``pgmpy`` logger to WARNING during
            structure learning (silencing INFO messages such as datatype
            inference) and defaults ``show_progress=False`` for pgmpy estimators.
        n_jobs : int, optional
            Parallel workers. None/1 = serial, -1 = all cores, >1 = that many workers.
        **kwargs
            Passed to StructureLearner and algorithm-specific parameters.

        Raises
        ------
        TypeError
            If data is not a numpy array or pandas DataFrame.
        ValueError
            If data dimensions are invalid, delta values are out of range, or
            structure learning method is invalid.
        """
        self.data, self.node_names = validate_data(data)

        if delta is None:
            delta = get_default_delta()
        else:
            validate_delta(delta)
        self.delta = delta

        self.verbose = verbose
        self.nbf = nbf

        start_time = time.time()

        # Learn structure
        if verbose:
            print(f"Learning structure using method: {method}")
        structure_learner = StructureLearner(verbose=verbose)
        self.adj_mat = structure_learner.learn_structure(
            data=self.data,
            method=method,
            nbf=nbf,
            delta=delta,
            node_names=self.node_names,
            **kwargs,
        )

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
        if verbose:
            print("Selecting discount factors...")
        self.DF = select_discount_factors(
            data=self.data,
            adj_mat=self.adj_mat,
            nbf=nbf,
            delta=delta,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Filter
        if verbose:
            print("Computing filtered estimates...")
        self.Filt = filter_all_nodes(
            data=self.data,
            adj_mat=self.adj_mat,
            DF_hat=self.DF["DF_hat"],
            node_names=self.node_names,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Smooth
        if verbose:
            print("Computing smoothed estimates...")
        self.Smoo = smooth_all_nodes(
            mt=self.Filt["mt"],
            Ct=self.Filt["Ct"],
            Rt=self.Filt["Rt"],
            nt=self.Filt["nt"],
            dt=self.Filt["dt"],
            n_jobs=n_jobs,
            verbose=verbose,
        )

        elapsed = time.time() - start_time
        if verbose:
            print()
            if elapsed < 60:
                print(f"MDM processing completed in {elapsed:.2f} seconds")
            elif elapsed < 3600:
                minutes, seconds = int(elapsed // 60), elapsed % 60
                print(f"MDM processing completed in {minutes}m {seconds:.2f}s")
            else:
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = elapsed % 60
                print(f"MDM processing completed in {hours}h {minutes}m {seconds:.2f}s")

    def __repr__(self) -> str:
        return (
            f"MDM(nodes={len(self.node_names)}, "
            f"time_points={self.data.shape[0]}, "
            f"edges={np.sum(self.adj_mat)}/{self.adj_mat.size})"
        )
