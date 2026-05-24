"""Validation helpers for IS aggregation input arguments.

These functions guard invariants at two pipeline stages: before coercion
(top-level argument checks) and after coercion (shape / length consistency
checks that depend on the materialized subject arrays).

No statistical logic lives here; these are purely defensive guards.
"""

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .results import MCContributorMode


def validate_aggregate_args(
    tau: float,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    plot_filt: Optional[Mapping[str, Any]],
    pool_filt_for_plotting: bool,
) -> None:
    """Validate top-level call arguments before any coercion or computation."""
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if mc_contributors == "all_subjects" and not mc_refit_global_structure:
        raise ValueError(
            "mc_contributors='all_subjects' requires mc_refit_global_structure=True"
        )
    if plot_filt is not None and pool_filt_for_plotting:
        raise ValueError("pass only one of plot_filt=... or pool_filt_for_plotting=True")


def validate_after_coercion(
    n_draws: int,
    filtered_eff: Optional[Sequence[Mapping[str, Any]]],
    mc_refit_global_structure: bool,
    rng: Optional[Any],
    pool_filt_for_plotting: bool,
    n_subjects: int,
    filtered_len: Optional[int],
    plot_data_eff: Optional[np.ndarray],
    n_nodes: int,
) -> None:
    """Validate arguments whose feasibility depends on the coerced inputs."""
    if n_draws > 0 and filtered_eff is None and not mc_refit_global_structure:
        raise ValueError(
            "filtered_per_subject is required when n_draws > 0 "
            "(unless adj_mats are fitted MDM instances with Filt "
            "or mc_refit_global_structure=True with per-subject data)"
        )
    if n_draws > 0 and rng is None:
        raise ValueError("rng is required when n_draws > 0")
    if pool_filt_for_plotting and filtered_eff is None:
        raise ValueError(
            "filtered_per_subject is required when pool_filt_for_plotting=True "
            "(unless adj_mats are fitted MDM instances with Filt)"
        )
    if filtered_eff is not None and filtered_len != n_subjects:
        raise ValueError(
            f"filtered_per_subject length {filtered_len} must match "
            f"number of adjacency matrices {n_subjects}"
        )
    if plot_data_eff is None:
        return
    pd_arr = np.asarray(plot_data_eff)
    if pd_arr.ndim != 2 or pd_arr.shape[1] != n_nodes:
        raise ValueError(
            f"plot_data must have shape (T, {n_nodes}), got {getattr(pd_arr, 'shape', None)}"
        )
