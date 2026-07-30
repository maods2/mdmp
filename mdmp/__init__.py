"""
MDMP: Bayesian Dynamic Regression Model (MDM) for Python

A Python package for learning Bayesian network structures from multivariate time series
and estimating time-varying dynamic parameters using Kalman filtering and smoothing.

This package is a Python port of the R package 'mdmr'.
"""

from ._version import __version__
from .anomaly import AnomalyDetectionResult, detect_anomalies
from .datasets import (
    list_datasets,
    load_dataset,
)
from .dlm import dlm_filter, dlm_smooth
from .group_analysis import (
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    ISAggregationResult,
    MCPosteriorSource,
    MDMDistanceResult,
    VTSResult,
    aggregate_individual_structures,
    bayes_factor_cut,
    compute_mdm_distance,
    compute_vts,
    fit_individual_structures,
    nearest_neighbours,
    silhouette,
    suggest_clusters,
)
from .model import MDM, refit_mdm_on_structure
from .plotting import (
    plot_anomalies,
    plot_arcs,
    plot_dag,
    plot_dendrogram,
    plot_group_embedding,
    plot_idag,
    plot_marginal,
    plot_projection,
    plot_stream,
    project_distance,
)
from .scoring import compute_logpl, select_discount_factors
from .structure import StructureLearner
from .validation import validate_multi_subject_data

# Aliases to match R package function names
CDELT = select_discount_factors  # R: CDELT
dlm_filt = dlm_filter            # R: dlm_filt
dlm_smoo = dlm_smooth            # R: dlm_smoo

__all__ = [
    "__version__",
    # Main class
    "MDM",
    "refit_mdm_on_structure",
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

    # Anomaly detection
    "detect_anomalies",
    "AnomalyDetectionResult",

    # Plotting functions (R exports)
    "plot_dag",
    "plot_arcs",
    "plot_anomalies",
    "plot_marginal",
    "plot_stream",
    "plot_idag",

    # Dataset loading functions
    "load_dataset",
    "list_datasets",

    # Group analysis (VTS + IS aggregation)
    "compute_vts",
    "VTSResult",
    "aggregate_individual_structures",
    "GlobalBetaMCResult",
    "ISAggregationResult",
    "ISAggregatedMDMView",
    "MCPosteriorSource",
    "validate_multi_subject_data",

    # Group-structure distance + projection
    "fit_individual_structures",
    "compute_mdm_distance",
    "MDMDistanceResult",
    "nearest_neighbours",
    "silhouette",
    "suggest_clusters",
    "bayes_factor_cut",
    "project_distance",
    "plot_projection",
    "plot_dendrogram",
    "plot_group_embedding",
]

