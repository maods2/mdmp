"""
Shared checks for objects passed to plotting (MDM or IS aggregation view).
"""


def require_filt_for_plot(obj: object, *, plot_kw: str) -> object:
    filt = getattr(obj, "Filt", None)
    if filt is None:
        raise TypeError(
            "This plot requires a model with Filt set. For an IS-aggregated graph, "
            "pass fitted MDM instances to mdmp.group_analysis.aggregate_individual_structures, "
            "or use mdmp.model.MDM after fitting."
        )
    return filt


def require_data_for_plot(obj: object, *, plot_kw: str) -> object:
    data = getattr(obj, "data", None)
    if data is None:
        raise TypeError(
            "This plot requires model.data. For an IS-aggregated graph, pass fitted "
            "MDM instances to mdmp.group_analysis.aggregate_individual_structures, "
            "or use mdmp.model.MDM after fitting."
        )
    return data


def require_smoo_for_plot(obj: object) -> object:
    sm = getattr(obj, "Smoo", None)
    if sm is None:
        raise TypeError(
            "This plot requires smoothed output (model.Smoo). Use a fitted "
            "mdmp.model.MDM, or pass fitted MDMs to aggregate_individual_structures."
        )
    return sm
