"""
Refit MDM pipelines on a fixed DAG (no structure learning).
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..utils import get_default_delta
from ..validation import validate_data, validate_delta
from .discount_selection import DiscountFactorSelector
from .filtering_pipeline import FilteringPipeline
from .results import MDMResults
from .smoothing_pipeline import SmoothingPipeline


def refit_mdm_on_structure(
    data: Union[np.ndarray, pd.DataFrame],
    adj_mat: Union[np.ndarray, pd.DataFrame],
    *,
    node_names: Optional[Sequence[str]] = None,
    nbf: int = 15,
    delta: Optional[np.ndarray] = None,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> MDMResults:
    """
    Run discount-factor selection, filtering, and smoothing on fixed binary ``adj_mat``.

    Same numerical pipeline as :class:`mdmp.model.MDM` after structure learning, without
    calling :class:`mdmp.model.StructureLearningPipeline` or duplicating ``dlm_filter``
    internals (delegates to :class:`DiscountFactorSelector`, :class:`FilteringPipeline`,
    :class:`SmoothingPipeline`).

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Multivariate time series ``(T, N)``.
    adj_mat : np.ndarray or pd.DataFrame
        Binary ``(N, N)`` directed adjacency; ``[i, j] == 1`` means parent ``i`` → child ``j``.
    node_names : sequence of str, optional
        Names of length ``N``. If omitted, taken from ``data`` columns or ``V1``…``VN``.
    nbf : int, optional
        Burn-in index for log predictive likelihood during discount selection.
    delta : np.ndarray, optional
        Discount-factor grid; defaults match :class:`mdmp.model.MDM`.
    verbose : bool, optional
        Progress messages from pipeline components.
    n_jobs : int, optional
        Parallel workers for discount selection, filtering, and smoothing.

    Returns
    -------
    MDMResults
        ``adj_mat``, ``data``, ``DF``, ``Filt``, ``Smoo``, ``node_names`` aligned with :class:`mdmp.model.MDM`.
    """
    data_arr, inferred_names = validate_data(data)
    arr = np.asarray(data_arr, dtype=float)
    t, n = arr.shape

    if isinstance(adj_mat, pd.DataFrame):
        am = np.asarray(adj_mat.values, dtype=float)
        adj_cols = [str(c) for c in adj_mat.columns.tolist()]
        adj_rows = [str(r) for r in adj_mat.index.tolist()]
        if adj_cols != adj_rows:
            raise ValueError("adj_mat DataFrame index and columns must match for refit")
        names_eff: List[str]
        if node_names is not None:
            names_eff = [str(x) for x in node_names]
            col_order = [adj_cols.index(nm) for nm in names_eff]
            am = am[np.ix_(col_order, col_order)]
        else:
            names_eff = adj_cols
            if names_eff != inferred_names:
                raise ValueError(
                    "node_names inferred from data must match adj_mat DataFrame columns; "
                    "pass node_names=... explicitly if reordering is intended."
                )
    else:
        am = np.asarray(adj_mat, dtype=float)
        if node_names is not None:
            names_eff = [str(x) for x in node_names]
        else:
            names_eff = inferred_names
        if len(names_eff) != n:
            raise ValueError(
                f"node_names length {len(names_eff)} does not match data with N={n}"
            )

    if am.ndim != 2 or am.shape[0] != am.shape[1]:
        raise ValueError(f"adj_mat must be square 2D, got shape {getattr(am, 'shape', None)}")
    if am.shape[0] != n:
        raise ValueError(f"adj_mat shape {am.shape} incompatible with data columns N={n}")

    flat = am.ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("adj_mat must contain only finite values")
    if not np.logical_or(flat == 0, flat == 1).all():
        raise ValueError("adj_mat must be binary (0/1) for refit_mdm_on_structure")
    adj_bin = am.astype(float)
    np.fill_diagonal(adj_bin, 0)

    if delta is None:
        delta = get_default_delta()
    else:
        validate_delta(delta)

    discount_selector = DiscountFactorSelector(verbose=verbose)
    filtering_pipeline = FilteringPipeline(verbose=verbose)
    smoothing_pipeline = SmoothingPipeline(verbose=verbose)

    df_result = discount_selector.select_discount_factors(
        data=arr,
        adj_mat=adj_bin,
        nbf=nbf,
        delta=delta,
        n_jobs=n_jobs,
    )
    filt = filtering_pipeline.filter_nodes(
        data=arr,
        adj_mat=adj_bin,
        DF_hat=df_result["DF_hat"],
        node_names=list(names_eff),
        n_jobs=n_jobs,
    )
    smoo = smoothing_pipeline.smooth_nodes(
        mt=filt["mt"],
        Ct=filt["Ct"],
        Rt=filt["Rt"],
        nt=filt["nt"],
        dt=filt["dt"],
        n_jobs=n_jobs,
    )

    return MDMResults(
        adj_mat=adj_bin,
        data=arr,
        DF=df_result,
        Filt=filt,
        Smoo=smoo,
        node_names=list(names_eff),
    )
