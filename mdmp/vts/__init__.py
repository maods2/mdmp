"""
Virtual Typical Subject (VTS) module.

Provides methods for computing a representative subject from multi-subject
multivariate time series via concatenation-based or mean-based approaches.
"""

from typing import Literal, Union

import numpy as np
import pandas as pd

from .data import align_subjects, prepare_multi_subject_data
from .evaluation import (
    compare_vts_methods,
    evaluate_vts_representation,
    subject_vs_vts_metrics,
)
from .estimators import get_estimator, global_mean, global_median, list_estimators
from .strategies import ConcatenationStrategy, MeanBasedStrategy
from .types import ComparisonResult, VTSResult


def compute_vts(
    data: Union[list, np.ndarray, pd.DataFrame],
    method: Literal["concatenation", "mean"] = "mean",
    estimator: str = "mean",
    return_series: bool = True,
    align_method: Literal["truncate", "pad", "interpolate"] = "truncate",
    **kwargs,
) -> VTSResult:
    """
    Compute Virtual Typical Subject from multi-subject time series.

    Parameters
    ----------
    data : list of np.ndarray, np.ndarray, or pd.DataFrame
        Multi-subject data:
        - List of (T_s x N) arrays
        - 3D array (S x T x N)
        - DataFrame with subject_id column (long format)
    method : {"concatenation", "mean"}, optional
        VTS computation method. Default "mean".
    estimator : str, optional
        For concatenation when return_series=False: "mean" or "median".
        Default "mean".
    return_series : bool, optional
        For concatenation: if True, return (T_total x N) for MDM fitting.
        If False, apply estimator over time to get (N,) summary.
        Default True.
    align_method : {"truncate", "pad", "interpolate"}, optional
        For mean method when subjects have different T. Default "truncate".
    **kwargs
        Additional options (reserved for future use).

    Returns
    -------
    VTSResult
        Result with vts_data, method, n_subjects, metadata.

    Examples
    --------
    >>> import numpy as np
    >>> from mdmp.vts import compute_vts
    >>> data = [np.random.randn(50, 3), np.random.randn(50, 3)]
    >>> result = compute_vts(data, method="mean")
    >>> result.vts_data.shape
    (50, 3)
    """
    arrays, metadata = prepare_multi_subject_data(data)

    if method == "concatenation":
        strategy = ConcatenationStrategy(
            estimator=estimator,
            return_series=return_series,
        )
    elif method == "mean":
        strategy = MeanBasedStrategy(align_method=align_method)
    else:
        raise ValueError(
            f"method must be 'concatenation' or 'mean', got {method!r}"
        )

    return strategy.compute(arrays, metadata, **kwargs)


__all__ = [
    "compute_vts",
    "compare_vts_methods",
    "evaluate_vts_representation",
    "subject_vs_vts_metrics",
    "prepare_multi_subject_data",
    "align_subjects",
    "VTSResult",
    "ComparisonResult",
    "ConcatenationStrategy",
    "MeanBasedStrategy",
    "get_estimator",
    "global_mean",
    "global_median",
    "list_estimators",
]
