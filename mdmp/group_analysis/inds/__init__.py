"""

Individual Structure (inds) aggregation across subjects.



Import from here or from :mod:`mdmp.group_analysis`::



    from mdmp.group_analysis import compute_is, ISAggregatedMDMView

"""



from .pipeline import compute_is

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

    "compute_is",

]

