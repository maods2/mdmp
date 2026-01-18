"""
Main MDM model class - orchestrator only.

This module contains the main MDM class that coordinates all pipeline components.
The class is now focused on orchestration rather than implementation details.
"""

import time
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from ..utils import get_default_delta
from ..validation import validate_data, validate_delta
from .discount_selection import DiscountFactorSelector
from .filtering_pipeline import FilteringPipeline
from .smoothing_pipeline import SmoothingPipeline
from .structure import StructureLearningPipeline


class MDM:
    """
    Multiregression Dynamic Model (MDM) for Bayesian network structure learning
    and dynamic parameter estimation from multivariate time series data.

    This class orchestrates the MDM pipeline by delegating to specialized
    pipeline components for structure learning, discount factor selection,
    filtering, and smoothing.

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
            Multivariate time series data. Rows represent time points,
            columns represent nodes. Must be complete (no missing values).
        method : {"hc", "tabu", "mmhc"}, optional
            Method for structure learning. Default is "hc".
        nbf : int, optional
            Burn-in time point for log predictive likelihood calculation. Default is 15.
        delta : np.ndarray, optional
            Sequence of discount factors for optimization. Default is np.arange(0.5, 1.01, 0.01).
            All values must be between 0 and 1.
        verbose : bool, optional
            Whether to print progress messages and show progress bars. Default is True.
        n_jobs : int, optional
            Number of parallel jobs for discount factor selection, filtering, and smoothing.
            If None or 1, uses serial processing. If -1, uses all available CPU cores.
            If > 1, uses that many parallel workers. Default is None (serial processing).
        **kwargs
            Additional arguments passed to StructureLearner and algorithm-specific parameters.

        Raises
        ------
        TypeError
            If data is not a numpy array or pandas DataFrame.
        ValueError
            If data dimensions are invalid, delta values are out of range, or
            structure learning method is invalid.
        """
        # Validate and prepare data
        self.data, self.node_names = validate_data(data)

        # Validate delta
        if delta is None:
            delta = get_default_delta()
        else:
            validate_delta(delta)
        self.delta = delta

        self.verbose = verbose
        self.nbf = nbf

        # Start timing
        start_time = time.time()

        # Initialize pipeline components
        structure_pipeline = StructureLearningPipeline(verbose=verbose)
        discount_selector = DiscountFactorSelector(verbose=verbose)
        filtering_pipeline = FilteringPipeline(verbose=verbose)
        smoothing_pipeline = SmoothingPipeline(verbose=verbose)

        # Learn structure
        self.adj_mat = structure_pipeline.learn_structure(
            data=self.data,
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
        df_result = discount_selector.select_discount_factors(
            data=self.data,
            adj_mat=self.adj_mat,
            nbf=nbf,
            delta=delta,
            n_jobs=n_jobs
        )
        self.DF = df_result

        # Filter
        self.Filt = filtering_pipeline.filter_nodes(
            data=self.data,
            adj_mat=self.adj_mat,
            DF_hat=df_result['DF_hat'],
            node_names=self.node_names,
            n_jobs=n_jobs
        )

        # Smooth
        self.Smoo = smoothing_pipeline.smooth_nodes(
            mt=self.Filt['mt'],
            Ct=self.Filt['Ct'],
            Rt=self.Filt['Rt'],
            nt=self.Filt['nt'],
            dt=self.Filt['dt'],
            n_jobs=n_jobs
        )

        # Log total processing time
        elapsed_time = time.time() - start_time
        if self.verbose:
            print()
            if elapsed_time < 60:
                print(f"MDM processing completed in {elapsed_time:.2f} seconds")
            elif elapsed_time < 3600:
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60
                print(f"MDM processing completed in {minutes}m {seconds:.2f}s")
            else:
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = elapsed_time % 60
                print(f"MDM processing completed in {hours}h {minutes}m {seconds:.2f}s")

# Removed: _prepare_data and _validate_delta methods
# These are now in validation.py module

    def __repr__(self) -> str:
        """String representation of MDM object."""
        return (
            f"MDM(nodes={len(self.node_names)}, "
            f"time_points={self.data.shape[0]}, "
            f"edges={np.sum(self.adj_mat)}/{self.adj_mat.size})"
        )
