"""
Validation functions for MDM inputs.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def validate_data(data: Union[np.ndarray, pd.DataFrame]) -> tuple:
    """
    Validate input data for MDM.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.

    Returns
    -------
    tuple
        Tuple of (data_array, node_names).

    Raises
    ------
    TypeError
        If data is not a numpy array or pandas DataFrame.
    ValueError
        If data dimensions are invalid.
    """
    if isinstance(data, pd.DataFrame):
        node_names = list(data.columns)
        data = data.values
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2D array (T x N), got {data.ndim}D array with shape {data.shape}"
            )
        if data.shape[0] < 2:
            raise ValueError(
                f"data must have at least 2 time points, got {data.shape[0]}"
            )
        if data.shape[1] < 1:
            raise ValueError(
                f"data must have at least 1 variable, got {data.shape[1]}"
            )
        node_names = [f"V{i+1}" for i in range(data.shape[1])]
    else:
        raise TypeError(
            f"data must be a numpy array or pandas DataFrame, got {type(data).__name__}"
        )

    return data, node_names


def validate_delta(delta: np.ndarray) -> None:
    """
    Validate discount factor array.

    Parameters
    ----------
    delta : np.ndarray
        Discount factor array.

    Raises
    ------
    TypeError
        If delta is not a numpy array.
    ValueError
        If delta is empty or contains invalid values.
    """
    if not isinstance(delta, np.ndarray):
        raise TypeError(f"delta must be a numpy array, got {type(delta).__name__}")
    if len(delta) == 0:
        raise ValueError("delta must not be empty")
    if np.any(delta < 0) or np.any(delta > 1):
        raise ValueError("delta values must be between 0 and 1")


def validate_multi_subject_data(
    data: Union[List[np.ndarray], np.ndarray, pd.DataFrame],
) -> Tuple[List[np.ndarray], dict]:
    """
    Validate and normalize multi-subject time series data for VTS computation.

    Accepts list of arrays, 3D array (S x T x N), or DataFrame with subject_id column.
    Reuses validate_data() logic for per-subject 2D validation.

    Parameters
    ----------
    data : list of np.ndarray, np.ndarray, or pd.DataFrame
        - List of (T_s x N) arrays: one per subject, variable lengths allowed
        - 3D array (S x T x N): S subjects, T time points, N variables
        - DataFrame: long format with subject_id column and variable columns

    Returns
    -------
    tuple
        (data_list, metadata) where:
        - data_list: list of (T_s x N) numpy arrays
        - metadata: dict with n_subjects, node_names, subject_lengths

    Raises
    ------
    TypeError
        If data format is not supported.
    ValueError
        If data dimensions or structure are invalid.
    """
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("data list must not be empty")
        arrays = []
        node_names = None
        for i, arr in enumerate(data):
            arr_val, names = validate_data(arr)
            arrays.append(arr_val)
            if node_names is None:
                node_names = names
            elif names != node_names:
                raise ValueError(
                    f"Subject {i} has node names {names}, expected {node_names}"
                )
        subject_lengths = [a.shape[0] for a in arrays]
        n_vars = arrays[0].shape[1]
        for i, a in enumerate(arrays[1:], 1):
            if a.shape[1] != n_vars:
                raise ValueError(
                    f"Subject {i} has {a.shape[1]} variables, expected {n_vars}"
                )

    elif isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError(
                f"3D array must have shape (S x T x N), got {data.ndim}D "
                f"with shape {data.shape}"
            )
        S, T, N = data.shape
        if S < 1:
            raise ValueError("Must have at least 1 subject")
        if T < 2:
            raise ValueError("Must have at least 2 time points per subject")
        if N < 1:
            raise ValueError("Must have at least 1 variable")
        arrays = [data[s, :, :] for s in range(S)]
        node_names = [f"V{i+1}" for i in range(N)]
        subject_lengths = [T] * S

    elif isinstance(data, pd.DataFrame):
        subject_cols = ["subject_id", "subject", "id"]
        subject_col = None
        for col in subject_cols:
            if col in data.columns:
                subject_col = col
                break
        if subject_col is None:
            raise ValueError(
                "DataFrame must have a subject identifier column "
                f"(one of {subject_cols})"
            )
        var_cols = [c for c in data.columns if c != subject_col]
        if len(var_cols) == 0:
            raise ValueError("DataFrame must have at least one variable column")
        groups = data.groupby(subject_col)
        arrays = []
        for _, grp in groups:
            sub_df = grp[var_cols]
            arr, _ = validate_data(sub_df)
            arrays.append(arr)
        node_names = var_cols
        subject_lengths = [a.shape[0] for a in arrays]

    else:
        raise TypeError(
            f"data must be a list of arrays, 3D numpy array, or DataFrame, "
            f"got {type(data).__name__}"
        )

    metadata = {
        "n_subjects": len(arrays),
        "node_names": node_names,
        "subject_lengths": subject_lengths,
    }
    return arrays, metadata
