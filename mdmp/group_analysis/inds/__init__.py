"""

Individual Structure (inds) aggregation across subjects.



Import from here or from :mod:`mdmp.group_analysis` / :mod:`mdmp` re-exports::



    from mdmp import aggregate_individual_structures, ISAggregatedMDMView

"""



from .pipeline import aggregate_individual_structures

from .results import (

    ConditionalEdgePosteriorResult,

    GlobalBetaMCResult,

    ISAggregatedMDMView,

    ISAggregationResult,

    ISPlotAdapter,

    MCPosteriorSource,

)



__all__ = [

    "GlobalBetaMCResult",

    "ConditionalEdgePosteriorResult",

    "ISAggregationResult",

    "ISAggregatedMDMView",

    "ISPlotAdapter",

    "MCPosteriorSource",

    "aggregate_individual_structures",

]

