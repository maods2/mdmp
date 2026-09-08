"""
Group-level analyses for multi-subject multivariate time series.

A *subject* (unit, individual) is one observational unit contributing a
multivariate series of N nodes. Paper names and Python entry points:

- Compute VTS — :func:`compute_vts` (``vts``)
- Compute IS — :func:`compute_is` (``inds``)
- Compute GS — :func:`fit_individual_structures` then
  :func:`compute_gs` (``distance``)
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

    compute_is,

)



from .distance import (

    MDMDistanceResult,

    METRIC_REGISTRY,

    bayes_factor_cut,

    compute_gs,

    fit_individual_structures,

    nearest_neighbours,

    silhouette,

    suggest_clusters,

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

    "compute_is",

    "ISAggregationResult",

    "ISAggregatedMDMView",

    "ISPlotAdapter",

    "GlobalBetaMCResult",

    "ConditionalEdgePosteriorResult",

    "MCPosteriorSource",

    "fit_individual_structures",

    "compute_gs",

    "MDMDistanceResult",

    "METRIC_REGISTRY",

    "nearest_neighbours",

    "silhouette",

    "suggest_clusters",

    "bayes_factor_cut",

]

