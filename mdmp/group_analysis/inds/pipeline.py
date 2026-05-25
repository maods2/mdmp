"""
Individual Structure (inds) aggregation pipeline.

Public entry point: :func:`aggregate_individual_structures`.

Stages: validate → coerce → vote → repair DAG → optional refit → MC → assemble result.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .coercion import (
    _PreparedSubjects,
    _coerce_subjects_for_aggregation,
    _normalize_first_argument,
    _validate_adj_list,
)
from .monte_carlo import _monte_carlo_global_edge_beta
from .pooled_filtering import build_plot_filt_from_subjects
from .refit import build_mc_inputs
from .results import (
    ISAggregatedMDMView,
    ISAggregationResult,
    MCPosteriorSource,
)
from .validation import validate_after_coercion, validate_aggregate_args
from .voting import _vote_threshold_and_repair_cycles


# ---------------------------------------------------------------------------
# Stage 1 — prepare
# ---------------------------------------------------------------------------


def prepare_and_validate(
    adj_mats: Any,
    tau: float,
    node_names: Optional[Sequence[str]],
) -> _PreparedSubjects:
    """
    Stage 1: validate arguments, coerce subjects, validate coerced shapes.

    Filtered states come from fitted MDM ``Filt`` fields when present.
    Monte Carlo argument checks run later in
    :func:`aggregate_individual_structures`.
    """
    adj_mats_norm = _normalize_first_argument(adj_mats)
    (
        resolved_adj_mats,
        resolved_node_names,
        resolved_posterior_per_subject,
        resolved_time_series,
        mdm_data_per_subject,
    ) = _coerce_subjects_for_aggregation(adj_mats_norm, node_names, None)
    arrays, names, s = _validate_adj_list(resolved_adj_mats, resolved_node_names)
    n = arrays[0].shape[0]
    resolved_posterior_len = (
        len(resolved_posterior_per_subject) if resolved_posterior_per_subject is not None else None
    )
    prepared = _PreparedSubjects(
        arrays=arrays,
        names=names,
        n_subjects=s,
        n_nodes=n,
        posterior_per_subject=resolved_posterior_per_subject,
        time_series=resolved_time_series,
        mdm_data_per_subject=mdm_data_per_subject,
    )
    validate_aggregate_args(tau)
    validate_after_coercion(
        resolved_posterior_per_subject,
        s,
        resolved_posterior_len,
        resolved_time_series,
        n,
    )
    return prepared


# ---------------------------------------------------------------------------
# Stage 2 — consensus
# ---------------------------------------------------------------------------


def vote_and_repair(
    prepared: _PreparedSubjects,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Edge-frequency vote (strict threshold) and greedy acyclic repair.

    Parameters
    ----------
    prepared
        Coerced subject adjacencies from :func:`prepare_and_validate`.
    tau
        Vote threshold in (0, 1).

    Returns
    -------
    tuple of (ndarray, dict)
        Consensus adjacency and vote/repair metadata.
    """
    return _vote_threshold_and_repair_cycles(
        prepared.arrays, tau, prepared.names, threshold_mode="strict"
    )


def make_aggregation_result(
    prepared: _PreparedSubjects,
    out_adj: np.ndarray,
    meta: Dict[str, Any],
    tau: float,
) -> ISAggregatedMDMView:
    """
    Wrap vote/repair output into an :class:`ISAggregatedMDMView` (plot fields unset).

    Parameters
    ----------
    prepared
        Coerced subjects (for ``node_names`` and ``n_subjects``).
    out_adj
        Consensus adjacency from :func:`vote_and_repair`.
    meta
        Vote/repair metadata.
    tau
        Threshold used for voting.

    Returns
    -------
    ISAggregatedMDMView
        Consensus DAG without Monte Carlo or plot adapter fields.
    """
    return ISAggregatedMDMView(
        adj_mat=out_adj,
        node_names=list(prepared.names),
        n_subjects=prepared.n_subjects,
        tau=tau,
        metadata=meta,
        global_beta_mc=None,
    )


def _aggregate_by_vote(
    prepared: _PreparedSubjects,
    tau: float,
) -> ISAggregatedMDMView:
    """
    Run stage 2: vote, repair, and package the consensus DAG.

    Returns
    -------
    ISAggregatedMDMView
    """
    out_adj, meta = vote_and_repair(prepared, tau)
    return make_aggregation_result(prepared, out_adj, meta, tau)


# ---------------------------------------------------------------------------
# Stage 3 — Monte Carlo
# ---------------------------------------------------------------------------


def _resolve_mc_refit(
    flag: Optional[bool],
    prepared: _PreparedSubjects,
) -> bool:
    """
    Resolve ``mc_refit_global_structure`` after coercion.

    ``None`` (auto): ``True`` when MDM-like per-subject ``data`` was coerced,
    otherwise ``False``.  Explicit ``True`` / ``False`` are preserved.
    """
    if flag is not None:
        return flag
    return prepared.mdm_data_per_subject is not None


def _should_run_mc(prepared: _PreparedSubjects, mc_n_samples: int) -> bool:
    """True when MDM filtered states exist and MC sample count is positive."""
    return prepared.posterior_per_subject is not None and mc_n_samples > 0


def _resolve_mc_rng(
    rng: Optional[np.random.Generator],
    run_mc: bool,
) -> Optional[np.random.Generator]:
    """
    Return ``rng`` or a default generator when MC will run.

    Parameters
    ----------
    rng
        User-supplied generator, or ``None``.
    run_mc
        Whether the MC stage is scheduled.

    Returns
    -------
    Generator or None
        ``None`` when ``run_mc`` is false.
    """
    if not run_mc:
        return None
    if rng is not None:
        return rng
    return np.random.default_rng()


def run_mc_path(
    prepared: _PreparedSubjects,
    view: ISAggregatedMDMView,
    *,
    mc_n_samples: int,
    rng: np.random.Generator,
    mc_quantiles: Optional[Sequence[float]],
    mc_posterior: MCPosteriorSource,
    mc_refit_global_structure: bool,
    mc_refit_n_jobs: Optional[int],
    mc_n_jobs: Optional[int],
    data_per_subject: Optional[Sequence[np.ndarray]] = None,
) -> ISAggregatedMDMView:
    """
    Refit on G*, then Monte Carlo population-mean edge coefficients.

    Parameters
    ----------
    prepared
        Coerced subjects from stage 1.
    view
        Consensus view carrying ``adj_mat`` (G*).
    mc_n_samples, rng, mc_quantiles
        Monte Carlo configuration (all filter time steps ``0 … T-1``).
    mc_posterior, mc_refit_global_structure
        Posterior source; refit on G* must be enabled.
    mc_refit_n_jobs
        Parallel jobs for optional smoothing during MC setup.
    mc_n_jobs
        Parallel jobs over filter time steps during Monte Carlo sampling
        (``None`` or ``1`` = serial; ``-1`` = all cores).
    data_per_subject
        Optional override for refit data; defaults to ``prepared.mdm_data_per_subject``.

    Returns
    -------
    ISAggregatedMDMView
        View with ``global_beta_mc`` and optional refit outputs attached.
    """
    mc = build_mc_inputs(
        mc_refit_global_structure=mc_refit_global_structure,
        mc_posterior=mc_posterior,
        arrays=prepared.arrays,
        names=prepared.names,
        n_subjects=prepared.n_subjects,
        n_nodes=prepared.n_nodes,
        out_adj=view.adj_mat,
        resolved_posterior_per_subject=prepared.posterior_per_subject,
        mdm_data_per_subject=prepared.mdm_data_per_subject,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )
    gb = _monte_carlo_global_edge_beta(
        posterior_per_subject=mc.filt_per_subject,
        subject_adjacency_matrices=mc.design_adjs,
        consensus_view=view,
        mc_n_samples=mc_n_samples,
        rng=rng,
        mc_quantiles=mc_quantiles,
        mc_posterior=mc_posterior,
        smoothed_per_subject=mc.smoothed_per_subject,
        mc_n_jobs=mc_n_jobs,
    )
    return replace(
        view,
        global_beta_mc=gb,
        refit_filt_per_subject=mc.refit_filt_per_subject,
        refit_smoo_per_subject=mc.refit_smoo_per_subject,
    )


# ---------------------------------------------------------------------------
# Stage 4 — view assembly
# ---------------------------------------------------------------------------


def _finalize_mdm_view(
    prepared: _PreparedSubjects,
    view: ISAggregatedMDMView,
) -> ISAggregatedMDMView:
    """
    Attach pooled ``Filt`` and group-mean ``data`` for plotting.

    Parameters
    ----------
    prepared
        Coerced subjects (filtered states and optional time series).
    view
        Consensus view, optionally already carrying MC outputs.

    Returns
    -------
    ISAggregatedMDMView
    """
    filt_final = None
    if prepared.posterior_per_subject is not None:
        filt_final = build_plot_filt_from_subjects(
            view.adj_mat,
            prepared.posterior_per_subject,
            prepared.arrays,
            prepared.names,
        )
    return replace(
        view,
        data=(
            None if prepared.time_series is None else np.asarray(prepared.time_series, dtype=float)
        ),
        Filt=filt_final,
    )


def aggregate_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    mc_n_samples: int = 20_000,
    rng: Optional[np.random.Generator] = None,
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    mc_refit_global_structure: Optional[bool] = None,
    mc_refit_n_jobs: Optional[int] = None,
    mc_n_jobs: Optional[int] = None,
) -> ISAggregatedMDMView:
    """
    Aggregate subject-specific DAGs into one consensus DAG.

    Pass binary adjacency matrices / DataFrames, or fitted :class:`~mdmp.model.MDM`
    instances.  MDM inputs run Monte Carlo (``mc_n_samples``, ``rng``) and build
    pooled ``Filt`` for :func:`~mdmp.plotting.plot_arcs`.  Adjacency-only inputs
    return the consensus graph only.

    Monte Carlo (when enabled) refits each subject on the consensus DAG G*, then
    for each replicate ``b`` and time ``t`` forms
    :math:`\\bar\\theta_t^{(b)} = \\frac{1}{S}\\sum_{i=1}^S \\theta_{it}^{(b)}`
    from marginal posterior draws :math:`\\theta_{it}^{(b)}`.
    """
    prepared = prepare_and_validate(adj_mats, tau, node_names)
    resolved_refit = _resolve_mc_refit(mc_refit_global_structure, prepared)
    view = _aggregate_by_vote(prepared, tau)

    run_mc = _should_run_mc(prepared, mc_n_samples)
    if run_mc:
        if not resolved_refit:
            raise ValueError(
                "Monte Carlo requires mc_refit_global_structure=True (refit each "
                "subject on the consensus DAG so all S subjects contribute)"
            )
        mc_rng = _resolve_mc_rng(rng, True)
        assert mc_rng is not None
        view = run_mc_path(
            prepared,
            view,
            mc_n_samples=mc_n_samples,
            rng=mc_rng,
            mc_quantiles=mc_quantiles,
            mc_posterior=mc_posterior,
            mc_refit_global_structure=resolved_refit,
            mc_refit_n_jobs=mc_refit_n_jobs,
            mc_n_jobs=mc_n_jobs,
        )

    if prepared.posterior_per_subject is not None:
        return _finalize_mdm_view(prepared, view)
    return view
