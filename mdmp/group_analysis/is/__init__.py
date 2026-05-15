"""
Individual Structure (IS) aggregation across subjects.

Because ``is`` is a Python keyword, prefer::

    from mdmp.group_analysis import aggregate_individual_structures, ISAggregationResult

or::

    import importlib
    is_mod = importlib.import_module("mdmp.group_analysis.is")
"""

from .aggregation import (
    GlobalBetaMCResult,
    ISAggregateOptions,
    ISAggregatedMDMView,
    ISAggregationResult,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
    aggregate_individual_structures,
    aggregate_with_options,
    build_plot_filt_from_subjects,
)
from .voting import ThresholdMode

__all__ = [
    "GlobalBetaMCResult",
    "ISAggregateOptions",
    "ISAggregatedMDMView",
    "ISAggregationResult",
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "ThresholdMode",
    "aggregate_individual_structures",
    "aggregate_with_options",
    "build_plot_filt_from_subjects",
]
