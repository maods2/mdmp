"""

Group-level analyses for multi-subject multivariate time series.



Subpackages

-----------

vts

    Virtual Typical Subject (VTS): build a representative time series across

    subjects (concatenation, mean-based, or median-based).

inds

    Individual Structure aggregation: combine subject-specific DAGs into

    one global DAG via edge-frequency thresholding and acyclic repair; optional

    Monte Carlo pooling of filtered DLM edge coefficients. The return value

    (:class:`ISAggregatedMDMView`) mirrors key :class:`mdmp.model.MDM` attributes

    for :mod:`mdmp.plotting` when ``time_series`` / ``plot_filt`` / etc. are supplied.

"""



from .vts import (

    ConcatenationStrategy,

    MeanBasedStrategy,

    MedianBasedStrategy,

    VTSResult,

    align_subjects,

    compute_vts,

    get_estimator,

    global_mean,

    global_median,

    list_estimators,

    prepare_multi_subject_data,

)



from .inds import (

    ConditionalEdgePosteriorResult,

    GlobalBetaMCResult,

    ISAggregatedMDMView,

    ISAggregationResult,

    ISPlotAdapter,

    MCPosteriorSource,

    aggregate_individual_structures,

)



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

    "ISAggregationResult",

    "ISAggregatedMDMView",

    "ISPlotAdapter",

    "GlobalBetaMCResult",

    "ConditionalEdgePosteriorResult",

    "MCPosteriorSource",

]

