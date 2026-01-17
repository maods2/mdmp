"""
MDMP: Bayesian Dynamic Regression Model (MDM) for Python

A Python package for learning Bayesian network structures from multivariate time series
and estimating time-varying dynamic parameters using Kalman filtering and smoothing.

This package is a Python port of the R package 'mdmr'.
"""

from .dlm import dlm_filter, dlm_smooth
from .mdm import MDM
from .plotting import plot_arcs, plot_dag, plot_idag, plot_marginal, plot_stream
from .scoring import compute_logpl, select_discount_factors
from .structure import StructureLearner
from .datasets import (
    load_dataset,
    list_datasets,
)

# Aliases to match R package function names
CDELT = select_discount_factors  # R: CDELT
dlm_filt = dlm_filter            # R: dlm_filt
dlm_smoo = dlm_smooth            # R: dlm_smoo

__version__ = "0.6.2"
__all__ = [
    # Main class
    "MDM",
    "StructureLearner",

    # Core DLM functions (R exports)
    "dlm_filter",
    "dlm_smooth",
    "dlm_filt",  # Alias for R compatibility
    "dlm_smoo",  # Alias for R compatibility

    # Scoring functions (R exports)
    "select_discount_factors",
    "CDELT",  # Alias for R compatibility
    "compute_logpl",  # Internal but available

    # Plotting functions (R exports)
    "plot_dag",
    "plot_arcs",
    "plot_marginal",
    "plot_stream",
    "plot_idag",

    # Dataset loading functions
    "load_dataset",
    "list_datasets",
]

