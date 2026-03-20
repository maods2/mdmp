"""
Data utilities for Virtual Typical Subject (VTS) computation.

Handles multi-subject data preparation and alignment for mean-based VTS
when subjects have different time series lengths.
"""

from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd

from ..validation import validate_multi_subject_data


def prepare_multi_subject_data(
    data: Union[List[np.ndarray], np.ndarray, pd.DataFrame],
) -> Tuple[List[np.ndarray], dict]:
    """
    Prepare and normalize multi-subject time series data for VTS computation.

    Converts various input formats to a normalized list of (T_s x N) arrays
    and returns metadata. Delegates to validate_multi_subject_data.

    Parameters
    ----------
    data : list of np.ndarray, np.ndarray, or pd.DataFrame
        Multi-subject data in any supported format.

    Returns
    -------
    tuple
        (data_list, metadata) where:
        - data_list: list of (T_s x N) numpy arrays
        - metadata: dict with n_subjects, node_names, subject_lengths

    Examples
    --------
    >>> import numpy as np
    >>> from mdmp.vts.data import prepare_multi_subject_data
    >>> data = [np.random.randn(50, 3), np.random.randn(60, 3)]
    >>> arrays, meta = prepare_multi_subject_data(data)
    >>> len(arrays)
    2
    >>> meta["n_subjects"]
    2
    """
    return validate_multi_subject_data(data)


def align_subjects(
    data: List[np.ndarray],
    method: Literal["truncate", "pad", "interpolate"] = "truncate",
) -> List[np.ndarray]:
    """
    Align subject time series to equal length for mean-based VTS.

    When subjects have different T, alignment is required before
    computing the mean across subjects.

    Parameters
    ----------
    data : list of np.ndarray
        List of (T_s x N) arrays. All must have same N.
    method : {"truncate", "pad", "interpolate"}, optional
        - "truncate": truncate all to min(T_s). Default.
        - "pad": pad shorter series with NaN (not recommended for MDM).
        - "interpolate": interpolate shorter series to max(T_s) length.
        Default is "truncate".

    Returns
    -------
    list of np.ndarray
        List of (T_aligned x N) arrays with equal T_aligned.

    Raises
    ------
    ValueError
        If data is empty or arrays have incompatible N.
    """
    if len(data) == 0:
        raise ValueError("data list must not be empty")

    n_vars = data[0].shape[1]
    for i, arr in enumerate(data):
        if arr.shape[1] != n_vars:
            raise ValueError(
                f"Subject {i} has {arr.shape[1]} variables, expected {n_vars}"
            )

    lengths = [arr.shape[0] for arr in data]
    if len(set(lengths)) == 1:
        return [arr.copy() for arr in data]

    if method == "truncate":
        min_len = min(lengths)
        return [arr[:min_len, :].copy() for arr in data]

    if method == "pad":
        max_len = max(lengths)
        result = []
        for arr in data:
            if arr.shape[0] < max_len:
                padded = np.full((max_len, n_vars), np.nan)
                padded[: arr.shape[0], :] = arr
                result.append(padded)
            else:
                result.append(arr.copy())
        return result

    if method == "interpolate":
        max_len = max(lengths)
        result = []
        for arr in data:
            if arr.shape[0] == max_len:
                result.append(arr.copy())
            else:
                interp_arr = np.zeros((max_len, n_vars))
                for j in range(n_vars):
                    x_old = np.linspace(0, 1, arr.shape[0])
                    x_new = np.linspace(0, 1, max_len)
                    interp_arr[:, j] = np.interp(x_new, x_old, arr[:, j])
                result.append(interp_arr)
        return result

    raise ValueError(
        f"method must be 'truncate', 'pad', or 'interpolate', got {method!r}"
    )
