"""Validation helpers for IS aggregation input arguments.

These functions guard invariants at two pipeline stages: before coercion
(top-level argument checks) and after coercion (shape / length consistency
checks that depend on the materialized subject arrays).

No statistical logic lives here; these are purely defensive guards.
"""

from typing import Any, Optional, Sequence

import numpy as np


def validate_aggregate_args(tau: float) -> None:
    """Validate top-level call arguments before any coercion or computation."""
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")


def validate_after_coercion(
    resolved_posterior_per_subject: Optional[Sequence[Any]],
    n_subjects: int,
    resolved_posterior_len: Optional[int],
    resolved_time_series: Optional[np.ndarray],
    n_nodes: int,
) -> None:
    """Validate coerced subject arrays (shapes and filtered-state alignment)."""
    if resolved_posterior_per_subject is not None and resolved_posterior_len != n_subjects:
        raise ValueError(
            f"posterior_per_subject length {resolved_posterior_len} must match "
            f"number of adjacency matrices {n_subjects}"
        )
    if resolved_time_series is None:
        return
    pd_arr = np.asarray(resolved_time_series)
    if pd_arr.ndim != 2 or pd_arr.shape[1] != n_nodes:
        raise ValueError(
            f"time_series must have shape (T, {n_nodes}), got {getattr(pd_arr, 'shape', None)}"
        )
