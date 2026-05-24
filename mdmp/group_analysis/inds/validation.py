"""Validation helpers for IS aggregation input arguments.

These functions guard invariants at two pipeline stages: before coercion
(top-level argument checks) and after coercion (shape / length consistency
checks that depend on the materialized subject arrays).

No statistical logic lives here; these are purely defensive guards.
"""

from typing import Any, Optional, Sequence

import numpy as np

from .results import MCContributorMode


def validate_aggregate_args(
    tau: float,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
) -> None:
    """Validate top-level call arguments before any coercion or computation."""
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if mc_contributors == "all_subjects" and not mc_refit_global_structure:
        raise ValueError(
            "mc_contributors='all_subjects' requires mc_refit_global_structure=True"
        )


def validate_after_coercion(
    mc_n_samples: int,
    resolved_filtered_per_subject: Optional[Sequence[Any]],
    mc_refit_global_structure: bool,
    rng: Optional[Any],
    n_subjects: int,
    resolved_filtered_len: Optional[int],
    resolved_time_series: Optional[np.ndarray],
    n_nodes: int,
    *,
    mc_requested: bool,
) -> None:
    """Validate arguments whose feasibility depends on the coerced inputs."""
    if (
        mc_requested
        and resolved_filtered_per_subject is None
        and not mc_refit_global_structure
    ):
        raise ValueError(
            "Monte Carlo requires per-subject filtered states "
            "(pass fitted MDM instances with Filt, or use run_inds_global_beta_mc "
            "with filtered_per_subject=..., or mc_refit_global_structure=True "
            "with per-subject data from MDMs)"
        )
    if resolved_filtered_per_subject is not None and resolved_filtered_len != n_subjects:
        raise ValueError(
            f"filtered_per_subject length {resolved_filtered_len} must match "
            f"number of adjacency matrices {n_subjects}"
        )
    if resolved_time_series is None:
        return
    pd_arr = np.asarray(resolved_time_series)
    if pd_arr.ndim != 2 or pd_arr.shape[1] != n_nodes:
        raise ValueError(
            f"time_series must have shape (T, {n_nodes}), got {getattr(pd_arr, 'shape', None)}"
        )
