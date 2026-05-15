"""
Group-level analyses for multi-subject multivariate time series.

Subpackages
-----------
vts
    Virtual Typical Subject (VTS): build a representative time series across
    subjects (concatenation, mean-based, or median-based).
is
    Individual Structure (IS) aggregation: combine subject-specific DAGs into
    one global DAG via edge-frequency thresholding and acyclic repair; optional
    Monte Carlo pooling of filtered DLM edge coefficients. The return value
    (:class:`ISAggregatedMDMView`) mirrors key :class:`mdmp.model.MDM` attributes
    for :mod:`mdmp.plotting` when ``plot_data`` / ``plot_filt`` / etc. are supplied.
    Import the ``is`` subpackage via ``importlib.import_module('mdmp.group_analysis.is')``
    or use the re-exports below (``is`` is a keyword, so ``from ...is import`` is invalid).
"""

import importlib

from .vts import (
    VTSResult,
    align_subjects,
    compute_vts,
    get_estimator,
    global_mean,
    global_median,
    list_estimators,
    prepare_multi_subject_data,
    ConcatenationStrategy,
    MeanBasedStrategy,
    MedianBasedStrategy,
)

_is = importlib.import_module("mdmp.group_analysis.is")
aggregate_individual_structures = _is.aggregate_individual_structures
aggregate_with_options = _is.aggregate_with_options
ISAggregationResult = _is.ISAggregationResult
ISAggregatedMDMView = _is.ISAggregatedMDMView
ISAggregateOptions = _is.ISAggregateOptions
GlobalBetaMCResult = _is.GlobalBetaMCResult
MCContributorMode = _is.MCContributorMode
MCPosteriorSource = _is.MCPosteriorSource
PoolingMode = _is.PoolingMode
ThresholdMode = _is.ThresholdMode
build_plot_filt_from_subjects = _is.build_plot_filt_from_subjects

__all__ = [
    "compute_vts",
    "prepare_multi_subject_data",
    "align_subjects",
    "VTSResult",
    "ConcatenationStrategy",
    "MeanBasedStrategy",
    "MedianBasedStrategy",
    "get_estimator",
    "global_mean",
    "global_median",
    "list_estimators",
    "aggregate_individual_structures",
    "aggregate_with_options",
    "ISAggregationResult",
    "ISAggregatedMDMView",
    "ISAggregateOptions",
    "GlobalBetaMCResult",
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "ThresholdMode",
    "build_plot_filt_from_subjects",
]
