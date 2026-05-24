"""
Individual Structure (inds) aggregation — public API and pipeline stages.

validate → coerce → vote → repair DAG → optional refit → MC → assemble result
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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


def prepare_and_validate(
    adj_mats: Any,
    tau: float,
    node_names: Optional[Sequence[str]],
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
    mc_n_samples: int,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    rng: Optional[np.random.Generator],
    mc_requested: bool,
    rng_defaults: bool = False,
) -> _PreparedSubjects:
    """Validate arguments, coerce subjects, validate coerced shapes."""
    validate_aggregate_args(tau, mc_contributors, mc_refit_global_structure)
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
        mc_refit_global_structure,
        mc_rng,
        s,
        resolved_filtered_len,
        resolved_time_series,
        n,
        mc_requested=validate_mc,
    )
    return _PreparedSubjects(
        arrays=arrays,
        names=names,
        n_subjects=s,
        n_nodes=n,
        filtered_per_subject=resolved_filtered_per_subject,
        time_series=resolved_time_series,
        mdm_data_per_subject=mdm_data_per_subject,
    )


def vote_and_repair(
    prepared: _PreparedSubjects,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Edge-frequency vote (strict threshold) and greedy acyclic repair."""
    return _vote_threshold_and_repair_cycles(
        prepared.arrays, tau, prepared.names, threshold_mode="strict"
    )


def run_mc_path(
    prepared: _PreparedSubjects,
    consensus: ISAggregationResult,
    *,
    mc_n_samples: int,
    rng: np.random.Generator,
    pooling: PoolingMode,
    time_index: int,
    time_indices: Optional[Sequence[int]],
    mc_quantiles: Optional[Sequence[float]],
    mc_posterior: MCPosteriorSource,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    data_per_subject: Optional[Sequence[np.ndarray]],
    mc_refit_n_jobs: Optional[int],
) -> ISAggregatedMDMView:
    """Optional refit on G*, then Monte Carlo global edge coefficients."""
    base_view = _consensus_to_view(consensus)
    mc = build_mc_inputs(
        mc_refit_global_structure=mc_refit_global_structure,
        mc_posterior=mc_posterior,
        arrays=prepared.arrays,
        names=prepared.names,
        n_subjects=prepared.n_subjects,
        n_nodes=prepared.n_nodes,
        out_adj=consensus.adj_mat,
        resolved_filtered_per_subject=prepared.filtered_per_subject,
        mdm_data_per_subject=prepared.mdm_data_per_subject,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )
    gb = _monte_carlo_global_edge_beta(
        mc.filt_per_subject,
        prepared.arrays,
        mc.design_adjs,
        base_view,
        time_index,
        mc_n_samples,
        rng,
        pooling,
        time_indices=time_indices,
        mc_quantiles=mc_quantiles,
        mc_contributors=mc_contributors,
        mc_posterior=mc_posterior,
        smoothed_per_subject=mc.smoothed_per_subject,
    )
    return replace(
        base_view,
        global_beta_mc=gb,
        refit_filt_per_subject=mc.refit_filt_per_subject,
        refit_smoo_per_subject=mc.refit_smoo_per_subject,
    )


def assemble_view(
    prepared: _PreparedSubjects,
    consensus: ISAggregationResult,
) -> ISAggregatedMDMView:
    """Build :class:`ISAggregatedMDMView` with pooled ``Filt`` and MDM ``data``."""
    filt_final = None
    if prepared.filtered_per_subject is not None:
        filt_final = build_plot_filt_from_subjects(
            consensus.adj_mat,
            prepared.filtered_per_subject,
            prepared.arrays,
            prepared.names,
        )
    base = _consensus_to_view(consensus)
    return replace(
        base,
        data=(
            None
            if prepared.time_series is None
            else np.asarray(prepared.time_series, dtype=float)
        ),
        Filt=filt_final,
    )


def _consensus_to_view(consensus: ISAggregationResult) -> ISAggregatedMDMView:
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


def build_consensus_result(
    prepared: _PreparedSubjects,
    out_adj: np.ndarray,
    meta: Dict[str, Any],
    tau: float,
) -> ISAggregationResult:
    return ISAggregationResult(
        adj_mat=out_adj,
        node_names=list(prepared.names),
        n_subjects=prepared.n_subjects,
        tau=tau,
        metadata=meta,
        global_beta_mc=None,
    )


def vote_individual_structures(
    adj_mats: Any,
    tau: float,
    node_names: Optional[Sequence[str]] = None,
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ISAggregationResult:
    """Vote subject DAGs into one consensus DAG (strict threshold + acyclic repair)."""
    validate_aggregate_args(tau, "individual_edge", False)
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
    out_adj, meta = vote_and_repair(prepared, tau)
    return build_consensus_result(prepared, out_adj, meta, tau)


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
    mc_refit_global_structure: bool = False,
    mc_refit_n_jobs: Optional[int] = None,
) -> ISAggregatedMDMView:
    """
    Aggregate subject-specific DAGs into one consensus DAG.

    Pass binary adjacency matrices / DataFrames, or fitted :class:`~mdmp.model.MDM`
    instances.  MDM inputs run Monte Carlo (``mc_n_samples``, ``rng``) and build
    pooled ``Filt`` for :func:`~mdmp.plotting.plot_arcs`.  Adjacency-only inputs
    return the consensus graph only.
    """
    validate_aggregate_args(tau, mc_contributors, mc_refit_global_structure)
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
    out_adj, meta = vote_and_repair(prepared, tau)
    consensus = build_consensus_result(prepared, out_adj, meta, tau)

    run_mc = (
        prepared.filtered_per_subject is not None
        and mc_n_samples > 0
    )
    mc_rng = (
        rng
        if rng is not None
        else (np.random.default_rng() if run_mc else None)
    )

    if run_mc:
        assert mc_rng is not None
        result: Union[ISAggregationResult, ISAggregatedMDMView] = run_mc_path(
            prepared,
            consensus,
            mc_n_samples=mc_n_samples,
            rng=mc_rng,
            pooling=pooling,
            time_index=0,
            time_indices=None,
            mc_quantiles=mc_quantiles,
            mc_posterior=mc_posterior,
            mc_contributors=mc_contributors,
            mc_refit_global_structure=mc_refit_global_structure,
            data_per_subject=None,
            mc_refit_n_jobs=mc_refit_n_jobs,
        )
    else:
        result = consensus

    if prepared.filtered_per_subject is not None:
        return assemble_view(prepared, result)
    return _consensus_to_view(result)


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
    time_index: int = 0,
    time_indices: Optional[Sequence[int]] = None,
    pooling: PoolingMode = "mean_with_edge",
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    mc_contributors: MCContributorMode = "individual_edge",
    mc_refit_global_structure: bool = False,
    data_per_subject: Optional[Sequence[np.ndarray]] = None,
    mc_refit_n_jobs: Optional[int] = None,
    node_names: Optional[Sequence[str]] = None,
) -> ISAggregatedMDMView:
    """Monte Carlo global edge coefficients conditional on the consensus DAG G*."""
    if mc_n_samples <= 0:
        raise ValueError("mc_n_samples must be > 0")
    validate_aggregate_args(result.tau, mc_contributors, mc_refit_global_structure)
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
    if prepared.n_subjects != result.n_subjects:
        raise ValueError(
            f"subject count {prepared.n_subjects} does not match result "
            f"n_subjects={result.n_subjects}"
        )
    view = run_mc_path(
        prepared,
        result,
        mc_n_samples=mc_n_samples,
        rng=rng,
        pooling=pooling,
        time_index=time_index,
        time_indices=time_indices,
        mc_quantiles=mc_quantiles,
        mc_posterior=mc_posterior,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
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
    """Aggregate with bundled :class:`ISAggregateOptions`."""
    return aggregate_individual_structures(
        adj_mats, tau, node_names, **asdict(options)
    )


compute_individual_structure_consensus = aggregate_with_options
