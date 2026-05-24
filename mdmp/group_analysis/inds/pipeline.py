"""
Individual Structure aggregation pipeline stages.

validate → coerce → vote → repair DAG → optional refit → MC → assemble result
"""

from __future__ import annotations

from dataclasses import replace
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
    ISAggregationResult,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
)
from .validation import validate_after_coercion, validate_aggregate_args
from .voting import ThresholdMode, _vote_threshold_and_repair_cycles


def prepare_and_validate(
    adj_mats: Any,
    tau: float,
    node_names: Optional[Sequence[str]],
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]],
    plot_data: Optional[np.ndarray],
    pool_filt_for_plotting: bool,
    n_draws: int,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    plot_filt: Optional[Mapping[str, Any]],
    rng: Optional[np.random.Generator],
) -> _PreparedSubjects:
    """Validate arguments, coerce subjects, validate coerced shapes."""
    validate_aggregate_args(
        tau, mc_contributors, mc_refit_global_structure, plot_filt, pool_filt_for_plotting
    )
    adj_mats_norm = _normalize_first_argument(adj_mats)
    (
        adj_mats_eff,
        node_names_eff,
        filtered_eff,
        plot_data_eff,
        mdm_data_per_subject,
    ) = _coerce_subjects_for_aggregation(
        adj_mats_norm,
        node_names,
        filtered_per_subject,
        plot_data,
        pool_filt_for_plotting=pool_filt_for_plotting,
        n_draws=n_draws,
    )
    arrays, names, s = _validate_adj_list(adj_mats_eff, node_names_eff)
    n = arrays[0].shape[0]
    fe_len = len(filtered_eff) if filtered_eff is not None else None
    validate_after_coercion(
        n_draws,
        filtered_eff,
        mc_refit_global_structure,
        rng,
        pool_filt_for_plotting,
        s,
        fe_len,
        plot_data_eff,
        n,
    )
    return _PreparedSubjects(
        arrays=arrays,
        names=names,
        n_subjects=s,
        n_nodes=n,
        filtered=filtered_eff,
        plot_data=plot_data_eff,
        mdm_data_per_subject=mdm_data_per_subject,
    )


def vote_and_repair(
    prepared: _PreparedSubjects,
    tau: float,
    *,
    threshold_mode: ThresholdMode,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Edge-frequency vote and greedy acyclic repair → consensus adjacency + metadata."""
    return _vote_threshold_and_repair_cycles(
        prepared.arrays, tau, prepared.names, threshold_mode=threshold_mode
    )


def run_mc_path(
    prepared: _PreparedSubjects,
    consensus: ISAggregationResult,
    *,
    n_draws: int,
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
        filtered_eff=prepared.filtered,
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
        n_draws,
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


def resolve_pooled_filt(
    prepared: _PreparedSubjects,
    consensus_adj: np.ndarray,
    *,
    plot_filt: Optional[Mapping[str, Any]],
    pool_filt_for_plotting: bool,
) -> Optional[Dict[str, Any]]:
    if plot_filt is not None:
        return dict(plot_filt)
    if not pool_filt_for_plotting:
        return None
    assert prepared.filtered is not None
    return build_plot_filt_from_subjects(
        consensus_adj, prepared.filtered, prepared.arrays, prepared.names
    )


def assemble_view(
    prepared: _PreparedSubjects,
    consensus: ISAggregationResult,
    *,
    plot_filt: Optional[Mapping[str, Any]] = None,
    plot_smoo: Optional[Mapping[str, Any]] = None,
    plot_df: Optional[Mapping[str, Any]] = None,
    pool_filt_for_plotting: bool = False,
) -> ISAggregatedMDMView:
    """Build :class:`ISAggregatedMDMView` with optional plot-adapter fields."""
    filt_final = resolve_pooled_filt(
        prepared,
        consensus.adj_mat,
        plot_filt=plot_filt,
        pool_filt_for_plotting=pool_filt_for_plotting,
    )
    base = _consensus_to_view(consensus)
    return replace(
        base,
        data=(
            None
            if prepared.plot_data is None
            else np.asarray(prepared.plot_data, dtype=float)
        ),
        Filt=filt_final,
        Smoo=None if plot_smoo is None else dict(plot_smoo),
        DF=None if plot_df is None else dict(plot_df),
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


def run_full(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float,
    node_names: Optional[Sequence[str]],
    *,
    threshold_mode: ThresholdMode = "strict",
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
    time_index: int = 0,
    time_indices: Optional[Sequence[int]] = None,
    n_draws: int = 0,
    rng: Optional[np.random.Generator] = None,
    pooling: PoolingMode = "mean_with_edge",
    plot_data: Optional[np.ndarray] = None,
    plot_filt: Optional[Mapping[str, Any]] = None,
    plot_smoo: Optional[Mapping[str, Any]] = None,
    plot_df: Optional[Mapping[str, Any]] = None,
    pool_filt_for_plotting: bool = False,
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    mc_contributors: MCContributorMode = "individual_edge",
    mc_refit_global_structure: bool = False,
    data_per_subject: Optional[Sequence[np.ndarray]] = None,
    mc_refit_n_jobs: Optional[int] = None,
) -> ISAggregatedMDMView:
    """Full pipeline: validate → coerce → vote → repair → refit? → MC? → assemble."""
    prepared = prepare_and_validate(
        adj_mats,
        tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        plot_data=plot_data,
        pool_filt_for_plotting=pool_filt_for_plotting,
        n_draws=n_draws,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
        plot_filt=plot_filt,
        rng=rng,
    )
    out_adj, meta = vote_and_repair(prepared, tau, threshold_mode=threshold_mode)
    consensus = build_consensus_result(prepared, out_adj, meta, tau)

    if n_draws > 0:
        assert rng is not None
        result: Union[ISAggregationResult, ISAggregatedMDMView] = run_mc_path(
            prepared,
            consensus,
            n_draws=n_draws,
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
    else:
        result = consensus

    any_plot = (
        plot_data is not None
        or plot_filt is not None
        or plot_smoo is not None
        or plot_df is not None
        or pool_filt_for_plotting
        or prepared.plot_data is not None
    )
    if any_plot:
        return assemble_view(
            prepared,
            result,
            plot_filt=plot_filt,
            plot_smoo=plot_smoo,
            plot_df=plot_df,
            pool_filt_for_plotting=pool_filt_for_plotting,
        )
    if isinstance(result, ISAggregatedMDMView):
        return result
    return _consensus_to_view(result)
