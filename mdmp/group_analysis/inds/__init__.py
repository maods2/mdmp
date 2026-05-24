"""
Individual Structure (inds) aggregation across subjects.

Import from here or from :mod:`mdmp.group_analysis` / :mod:`mdmp` re-exports::

    from mdmp.group_analysis.inds import vote_individual_structures, ISAggregationResult
"""

from .aggregation import (
    ConditionalEdgePosteriorResult,
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    ISAggregateOptions,
    ISAggregationResult,
    ISPlotAdapter,
    IndsRefitResult,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
    aggregate_individual_structures,
    aggregate_with_options,
    as_inds_mdm_view,
    build_plot_filt_from_subjects,
    compute_individual_structure_consensus,
    pool_conditional_filtered_states,
    refit_on_consensus,
    run_inds_global_beta_mc,
    vote_individual_structures,
)
from .results import (
    ISMDMViewOptions,
    ISMonteCarloOptions,
    ISVoteOptions,
    merge_aggregate_options,
)
from .voting import ThresholdMode, repair_dag_to_acyclic, vote_edge_frequencies

__all__ = [
    "GlobalBetaMCResult",
    "ConditionalEdgePosteriorResult",
    "ISAggregationResult",
    "ISAggregatedMDMView",
    "ISPlotAdapter",
    "ISAggregateOptions",
    "ISVoteOptions",
    "ISMonteCarloOptions",
    "ISMDMViewOptions",
    "merge_aggregate_options",
    "IndsRefitResult",
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "ThresholdMode",
    "aggregate_individual_structures",
    "aggregate_with_options",
    "compute_individual_structure_consensus",
    "vote_individual_structures",
    "refit_on_consensus",
    "run_inds_global_beta_mc",
    "pool_conditional_filtered_states",
    "as_inds_mdm_view",
    "build_plot_filt_from_subjects",
    "vote_edge_frequencies",
    "repair_dag_to_acyclic",
]
