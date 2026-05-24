"""
Individual Structure (IS) aggregation across subjects.

Because ``is`` is a Python keyword, prefer::

    from mdmp.group_analysis import aggregate_individual_structures, ISAggregationResult

or::

    import importlib
    is_mod = importlib.import_module("mdmp.group_analysis.is")
"""

from .aggregation import (
    ConditionalEdgePosteriorResult,
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    ISAggregateOptions,
    ISAggregationResult,
    ISPlotAdapter,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
    aggregate_individual_structures,
    aggregate_with_options,
    build_plot_filt_from_subjects,
    compute_individual_structure_consensus,
)
from .voting import ThresholdMode

__all__ = [
    # Inference result types
    "GlobalBetaMCResult",
    "ConditionalEdgePosteriorResult",   # canonical alias for GlobalBetaMCResult
    "ISAggregationResult",
    # Plot-adapter types
    "ISAggregatedMDMView",
    "ISPlotAdapter",                    # canonical alias for ISAggregatedMDMView
    # Options
    "ISAggregateOptions",
    # Type aliases
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "ThresholdMode",
    # Functions
    "aggregate_individual_structures",
    "aggregate_with_options",
    "compute_individual_structure_consensus",  # canonical alias
    "build_plot_filt_from_subjects",
]
