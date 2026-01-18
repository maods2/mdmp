"""
Validation functions for MDM inputs.
"""

from typing import Union

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
