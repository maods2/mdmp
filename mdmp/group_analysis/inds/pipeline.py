"""
Individual Structure (inds) aggregation — public API and pipeline stages.

validate → coerce → vote → repair DAG → optional refit → MC → assemble result
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

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
    ISAggregateOptions,
    ISAggregationResult,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
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
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
    mc_n_samples: int,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: Optional[bool],
    rng: Optional[np.random.Generator],
    mc_requested: bool,
    rng_defaults: bool = False,
) -> _PreparedSubjects:
    """
    Stage 1: validate arguments, coerce subjects, validate coerced shapes.

    Parameters
    ----------
    adj_mats
        Subject adjacencies, DataFrames, or fitted MDM objects.
    tau
        Vote threshold in (0, 1).
    node_names
        Optional shared node labels.
    filtered_per_subject
        Optional explicit filtered states (overrides MDM ``Filt`` extraction).
    mc_n_samples, mc_contributors
        Monte Carlo flags validated when MC is requested.
    mc_refit_global_structure
        ``None`` selects auto mode after coercion (MDM inputs → refit on G*).
    rng
        Random generator; required when ``mc_requested`` and ``mc_n_samples > 0``.
    mc_requested
        When true, enforce MC prerequisites (filtered states, ``rng``).
    rng_defaults
        If true and ``rng`` is missing, use ``np.random.default_rng()``.

    Returns
    -------
    _PreparedSubjects
        Coerced adjacency arrays and optional MDM side data.
    """
    adj_mats_norm = _normalize_first_argument(adj_mats)
    (
        resolved_adj_mats,
        resolved_node_names,
        resolved_filtered_per_subject,
        resolved_time_series,
        mdm_data_per_subject,
    ) = _coerce_subjects_for_aggregation(
        adj_mats_norm,
        node_names,
        filtered_per_subject,
    )
    arrays, names, s = _validate_adj_list(resolved_adj_mats, resolved_node_names)
    n = arrays[0].shape[0]
    resolved_filtered_len = (
        len(resolved_filtered_per_subject)
        if resolved_filtered_per_subject is not None
        else None
    )
    prepared = _PreparedSubjects(
        arrays=arrays,
        names=names,
        n_subjects=s,
        n_nodes=n,
        filtered_per_subject=resolved_filtered_per_subject,
        time_series=resolved_time_series,
        mdm_data_per_subject=mdm_data_per_subject,
    )
    resolved_refit = _resolve_mc_refit(mc_refit_global_structure, prepared)
    validate_aggregate_args(tau, mc_contributors, resolved_refit)
    validate_mc = mc_requested and mc_n_samples > 0
    mc_rng = rng
    if validate_mc and mc_rng is None:
        if rng_defaults:
            mc_rng = np.random.default_rng()
        else:
            raise ValueError("rng is required when mc_n_samples > 0")
    validate_after_coercion(
        mc_n_samples,
        resolved_filtered_per_subject,
        resolved_refit,
        mc_rng,
        s,
        resolved_filtered_len,
        resolved_time_series,
        n,
        mc_requested=validate_mc,
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
    return prepared.filtered_per_subject is not None and mc_n_samples > 0


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
    pooling: PoolingMode,
    mc_quantiles: Optional[Sequence[float]],
    mc_posterior: MCPosteriorSource,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    mc_refit_n_jobs: Optional[int],
    data_per_subject: Optional[Sequence[np.ndarray]] = None,
) -> ISAggregatedMDMView:
    """
    Optional refit on G*, then Monte Carlo global edge coefficients.

    Parameters
    ----------
    prepared
        Coerced subjects from stage 1.
    view
        Consensus view carrying ``adj_mat`` (G*).
    mc_n_samples, rng, pooling, mc_quantiles
        Monte Carlo configuration (all filter time steps ``0 … T-1``).
    mc_posterior, mc_contributors, mc_refit_global_structure
        Posterior source and contributor semantics (refit flag must be resolved).
    mc_refit_n_jobs
        Parallel jobs for optional smoothing during MC setup.
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
        resolved_filtered_per_subject=prepared.filtered_per_subject,
        mdm_data_per_subject=prepared.mdm_data_per_subject,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )
    gb = _monte_carlo_global_edge_beta(
        mc.filt_per_subject,
        prepared.arrays,
        mc.design_adjs,
        view,
        mc_n_samples,
        rng,
        pooling,
        mc_quantiles=mc_quantiles,
        mc_contributors=mc_contributors,
        mc_posterior=mc_posterior,
        smoothed_per_subject=mc.smoothed_per_subject,
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


def _consensus_to_view(
    consensus: Union[ISAggregationResult, ISAggregatedMDMView],
) -> ISAggregatedMDMView:
    """
    Convert a consensus result to :class:`ISAggregatedMDMView` if needed.

    Returns
    -------
    ISAggregatedMDMView
        Pass-through when ``consensus`` is already a view.
    """
    if isinstance(consensus, ISAggregatedMDMView):
        return consensus
    return ISAggregatedMDMView(
        adj_mat=consensus.adj_mat,
        node_names=list(consensus.node_names),
        n_subjects=consensus.n_subjects,
        tau=consensus.tau,
        metadata=dict(consensus.metadata),
        global_beta_mc=consensus.global_beta_mc,
    )


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
    if prepared.filtered_per_subject is not None:
        filt_final = build_plot_filt_from_subjects(
            view.adj_mat,
            prepared.filtered_per_subject,
            prepared.arrays,
            prepared.names,
        )
    return replace(
        view,
        data=(
            None
            if prepared.time_series is None
            else np.asarray(prepared.time_series, dtype=float)
        ),
        Filt=filt_final,
    )


def assemble_view(
    prepared: _PreparedSubjects,
    consensus: Union[ISAggregationResult, ISAggregatedMDMView],
) -> ISAggregatedMDMView:
    """
    Build :class:`ISAggregatedMDMView` with pooled ``Filt`` and MDM ``data``.

    Thin wrapper around :func:`_finalize_mdm_view` for consensus-only inputs.
    """
    return _finalize_mdm_view(prepared, _consensus_to_view(consensus))


def _assert_prepared_matches(
    prepared: _PreparedSubjects,
    consensus: ISAggregationResult,
) -> None:
    """
    Ensure coerced subjects align with an existing consensus result.

    Raises
    ------
    ValueError
        If subject or node counts disagree.
    """
    if prepared.n_subjects != consensus.n_subjects:
        raise ValueError(
            f"subject count {prepared.n_subjects} does not match consensus "
            f"n_subjects={consensus.n_subjects}"
        )
    if prepared.n_nodes != consensus.adj_mat.shape[0]:
        raise ValueError(
            f"node count {prepared.n_nodes} does not match consensus adjacency "
            f"shape {consensus.adj_mat.shape}"
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def vote_individual_structures(
    adj_mats: Any,
    tau: float,
    node_names: Optional[Sequence[str]] = None,
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ISAggregationResult:
    """Vote subject DAGs into one consensus DAG (strict threshold + acyclic repair)."""
    prepared = prepare_and_validate(
        adj_mats,
        tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        mc_n_samples=0,
        mc_contributors="individual_edge",
        mc_refit_global_structure=False,
        rng=None,
        mc_requested=False,
    )
    return _aggregate_by_vote(prepared, tau)


def aggregate_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    mc_n_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
    pooling: PoolingMode = "mean_with_edge",
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    mc_contributors: MCContributorMode = "individual_edge",
    mc_refit_global_structure: Optional[bool] = None,
    mc_refit_n_jobs: Optional[int] = None,
) -> ISAggregatedMDMView:
    """
    Aggregate subject-specific DAGs into one consensus DAG.

    Pass binary adjacency matrices / DataFrames, or fitted :class:`~mdmp.model.MDM`
    instances.  MDM inputs run Monte Carlo (``mc_n_samples``, ``rng``) and build
    pooled ``Filt`` for :func:`~mdmp.plotting.plot_arcs`.  Adjacency-only inputs
    return the consensus graph only.

    With ``mc_refit_global_structure=None`` (default), MDM-like inputs refit each
    subject on the consensus DAG G* before Monte Carlo; pass ``False`` to use
    per-subject filtered states under individual DAGs instead.
    """
    prepared = prepare_and_validate(
        adj_mats,
        tau,
        node_names,
        filtered_per_subject=None,
        mc_n_samples=mc_n_samples,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
        rng=rng,
        mc_requested=False,
        rng_defaults=False,
    )
    resolved_refit = _resolve_mc_refit(mc_refit_global_structure, prepared)
    view = _aggregate_by_vote(prepared, tau)

    run_mc = _should_run_mc(prepared, mc_n_samples)
    if run_mc:
        mc_rng = _resolve_mc_rng(rng, True)
        assert mc_rng is not None
        view = run_mc_path(
            prepared,
            view,
            mc_n_samples=mc_n_samples,
            rng=mc_rng,
            pooling=pooling,
            mc_quantiles=mc_quantiles,
            mc_posterior=mc_posterior,
            mc_contributors=mc_contributors,
            mc_refit_global_structure=resolved_refit,
            mc_refit_n_jobs=mc_refit_n_jobs,
        )

    if prepared.filtered_per_subject is not None:
        return _finalize_mdm_view(prepared, view)
    return view


def pool_conditional_filtered_states(
    consensus: ISAggregationResult,
    adj_mats: Any,
    filtered_per_subject: Sequence[Mapping[str, Any]],
    *,
    node_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Pool per-subject filtered states onto the consensus DAG (conditional on edges).

    Visualization helper for :func:`mdmp.plotting.plot_arcs` — not a joint posterior.
    """
    prepared = prepare_and_validate(
        adj_mats,
        consensus.tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        mc_n_samples=0,
        mc_contributors="individual_edge",
        mc_refit_global_structure=False,
        rng=None,
        mc_requested=False,
    )
    _assert_prepared_matches(prepared, consensus)
    assert prepared.filtered_per_subject is not None
    return build_plot_filt_from_subjects(
        consensus.adj_mat,
        prepared.filtered_per_subject,
        prepared.arrays,
        prepared.names,
    )


def run_inds_global_beta_mc(
    result: Union[ISAggregationResult, ISAggregatedMDMView],
    adj_mats: Any,
    *,
    mc_n_samples: int,
    rng: np.random.Generator,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
    pooling: PoolingMode = "mean_with_edge",
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    mc_contributors: MCContributorMode = "individual_edge",
    mc_refit_global_structure: Optional[bool] = None,
    data_per_subject: Optional[Sequence[np.ndarray]] = None,
    mc_refit_n_jobs: Optional[int] = None,
    node_names: Optional[Sequence[str]] = None,
) -> ISAggregatedMDMView:
    """Monte Carlo global edge coefficients conditional on the consensus DAG G*."""
    if mc_n_samples <= 0:
        raise ValueError("mc_n_samples must be > 0")
    prepared = prepare_and_validate(
        adj_mats,
        result.tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        mc_n_samples=mc_n_samples,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
        rng=rng,
        mc_requested=True,
    )
    resolved_refit = _resolve_mc_refit(mc_refit_global_structure, prepared)
    if prepared.n_subjects != result.n_subjects:
        raise ValueError(
            f"subject count {prepared.n_subjects} does not match result "
            f"n_subjects={result.n_subjects}"
        )
    base_view = _consensus_to_view(result)
    view = run_mc_path(
        prepared,
        base_view,
        mc_n_samples=mc_n_samples,
        rng=rng,
        pooling=pooling,
        mc_quantiles=mc_quantiles,
        mc_posterior=mc_posterior,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=resolved_refit,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )
    if isinstance(result, ISAggregatedMDMView):
        return replace(
            view,
            data=result.data,
            Filt=result.Filt,
            Smoo=result.Smoo,
            DF=result.DF,
        )
    return view


def as_inds_mdm_view(
    consensus: ISAggregationResult,
    *,
    filt: Optional[Mapping[str, Any]] = None,
    data: Optional[np.ndarray] = None,
    smoo: Optional[Mapping[str, Any]] = None,
    df: Optional[Mapping[str, Any]] = None,
) -> ISAggregatedMDMView:
    """Attach MDM-shaped fields for :mod:`mdmp.plotting` (does not run inference)."""
    base = _consensus_to_view(consensus)
    return replace(
        base,
        data=None if data is None else np.asarray(data, dtype=float),
        Filt=None if filt is None else dict(filt),
        Smoo=None if smoo is None else dict(smoo),
        DF=None if df is None else dict(df),
    )


def aggregate_with_options(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    options: ISAggregateOptions,
) -> ISAggregatedMDMView:
    """Aggregate with bundled :class:`ISAggregateOptions` (delegates to :func:`aggregate_individual_structures`)."""
    return aggregate_individual_structures(
        adj_mats, tau, node_names, **asdict(options)
    )
