"""
Estimators for group-level statistics in Virtual Typical Subject computation.

Provides pluggable estimators (mean, median, etc.) applied over concatenated
or aggregated data to produce the VTS representation.
"""

from typing import Callable, Optional

import numpy as np

# Registry of estimator names to functions
_ESTIMATOR_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}


def _register_estimator(name: str) -> Callable:
    """Decorator to register an estimator function."""

    def decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        _ESTIMATOR_REGISTRY[name] = func
        return func

    return decorator


def get_estimator(name: str) -> Callable[..., np.ndarray]:
    """
    Get estimator function by name.

    Parameters
    ----------
    name : str
        Estimator name: "mean", "median", etc.

    Returns
    -------
    callable
        Estimator function with signature (data, axis) -> np.ndarray.

    Raises
    ------
    ValueError
        If estimator name is not registered.
    """
    if name not in _ESTIMATOR_REGISTRY:
        available = ", ".join(_ESTIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown estimator {name!r}. Available: {available}"
        )
    return _ESTIMATOR_REGISTRY[name]


def list_estimators() -> list[str]:
    """List available estimator names."""
    return list(_ESTIMATOR_REGISTRY.keys())


@_register_estimator("mean")
def global_mean(
    data: np.ndarray,
    axis: Optional[int] = 0,
) -> np.ndarray:
    """
    Compute global mean along specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input data (e.g., T x N for time series).
    axis : int, optional
        Axis along which to compute mean. Default 0 (over time points).

    Returns
    -------
    np.ndarray
        Mean values. Shape is data.shape with axis removed.
    """
    return np.mean(data, axis=axis)


@_register_estimator("median")
def global_median(
    data: np.ndarray,
    axis: Optional[int] = 0,
) -> np.ndarray:
    """
    Compute global median along specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input data (e.g., T x N for time series).
    axis : int, optional
        Axis along which to compute median. Default 0 (over time points).

    Returns
    -------
    np.ndarray
        Median values. Shape is data.shape with axis removed.
    """
    return np.median(data, axis=axis)
