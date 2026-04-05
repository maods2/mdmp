"""
Group-level analyses for multi-subject multivariate time series.

Subpackages
-----------
vts
    Virtual Typical Subject (VTS): build a representative time series across
    subjects (concatenation or mean-based).
is
    Individual Structure (IS) aggregation: combine subject-specific DAGs into
    one global DAG via edge-frequency thresholding and acyclic repair.
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
)

_is = importlib.import_module("mdmp.group_analysis.is")
aggregate_individual_structures = _is.aggregate_individual_structures
ISAggregationResult = _is.ISAggregationResult

__all__ = [
    "compute_vts",
    "prepare_multi_subject_data",
    "align_subjects",
    "VTSResult",
    "ConcatenationStrategy",
    "MeanBasedStrategy",
    "get_estimator",
    "global_mean",
    "global_median",
    "list_estimators",
    "aggregate_individual_structures",
    "ISAggregationResult",
]
