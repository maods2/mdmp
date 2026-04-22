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
    ISAggregatedMDMView,
    ISAggregationResult,
    aggregate_individual_structures,
    build_plot_filt_from_subjects,
)

__all__ = [
    "GlobalBetaMCResult",
    "ISAggregatedMDMView",
    "ISAggregationResult",
    "aggregate_individual_structures",
    "build_plot_filt_from_subjects",
]
