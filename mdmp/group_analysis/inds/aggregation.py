"""
Individual Structure (inds) aggregation — thin public façade.

Start with :func:`vote_individual_structures` for a consensus DAG only.
For global edge posteriors use :func:`run_inds_global_beta_mc` (refit on G*
via :func:`refit_on_consensus` is recommended when subject data are available).
For plotting use :func:`as_inds_mdm_view` after optional
:func:`pool_conditional_filtered_states`.

The full legacy path remains :func:`aggregate_individual_structures` →
:func:`~mdmp.group_analysis.inds.pipeline.run_full`.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .pipeline import (
    assemble_view,
    build_consensus_result,
    prepare_and_validate,
    run_full,
    run_mc_path,
    vote_and_repair,
)
from .pooled_filtering import build_plot_filt_from_subjects
from .refit import IndsRefitResult, refit_on_consensus
from .results import (
    ConditionalEdgePosteriorResult,
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    ISAggregateOptions,
    ISAggregationResult,
    ISPlotAdapter,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
)
from .validation import validate_aggregate_args
from .voting import ThresholdMode

__all__ = [
    "GlobalBetaMCResult",
    "ConditionalEdgePosteriorResult",
    "ISAggregationResult",
    "ISAggregatedMDMView",
    "ISPlotAdapter",
    "ISAggregateOptions",
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "ThresholdMode",
    "IndsRefitResult",
    "aggregate_individual_structures",
    "aggregate_with_options",
    "compute_individual_structure_consensus",
    "vote_individual_structures",
    "refit_on_consensus",
    "run_inds_global_beta_mc",
    "pool_conditional_filtered_states",
    "as_inds_mdm_view",
    "build_plot_filt_from_subjects",
]


def vote_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    threshold_mode: ThresholdMode = "strict",
) -> ISAggregationResult:
    """
    Vote subject DAGs into one consensus DAG (threshold + greedy acyclic repair).

    Only ``adj_mats`` and ``tau`` are required.  No Monte Carlo or plot fields.
    """
    validate_aggregate_args(tau, "individual_edge", False, None, False)
    prepared = prepare_and_validate(
        adj_mats,
        tau,
        node_names,
        filtered_per_subject=None,
        plot_data=None,
        pool_filt_for_plotting=False,
        n_draws=0,
        mc_contributors="individual_edge",
        mc_refit_global_structure=False,
        plot_filt=None,
        rng=None,
    )
    out_adj, meta = vote_and_repair(prepared, tau, threshold_mode=threshold_mode)
    return build_consensus_result(prepared, out_adj, meta, tau)


def pool_conditional_filtered_states(
    consensus: ISAggregationResult,
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    filtered_per_subject: Sequence[Mapping[str, Any]],
    *,
    node_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Pool per-subject filtered states onto the consensus DAG (conditional on edges).

    Visualization helper for :func:`mdmp.plotting.plot_arcs` — **not** a joint
    posterior.  For edge-coefficient uncertainty use :func:`run_inds_global_beta_mc`.
    """
    prepared = prepare_and_validate(
        adj_mats,
        consensus.tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        plot_data=None,
        pool_filt_for_plotting=False,
        n_draws=0,
        mc_contributors="individual_edge",
        mc_refit_global_structure=False,
        plot_filt=None,
        rng=None,
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
    assert prepared.filtered is not None
    return build_plot_filt_from_subjects(
        consensus.adj_mat, prepared.filtered, prepared.arrays, prepared.names
    )


def as_inds_mdm_view(
    consensus: ISAggregationResult,
    *,
    filt: Optional[Mapping[str, Any]] = None,
    data: Optional[np.ndarray] = None,
    smoo: Optional[Mapping[str, Any]] = None,
    df: Optional[Mapping[str, Any]] = None,
) -> ISAggregatedMDMView:
    """
    Attach MDM-shaped fields for :mod:`mdmp.plotting` (does not run inference).
    """
    from .pipeline import _consensus_to_view

    base = _consensus_to_view(consensus)
    return replace(
        base,
        data=None if data is None else np.asarray(data, dtype=float),
        Filt=None if filt is None else dict(filt),
        Smoo=None if smoo is None else dict(smoo),
        DF=None if df is None else dict(df),
    )


def run_inds_global_beta_mc(
    result: Union[ISAggregationResult, ISAggregatedMDMView],
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    *,
    n_draws: int,
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
    """
    Monte Carlo global edge coefficients conditional on the consensus DAG G*.

    When per-subject ``(T, N)`` data are available, set
    ``mc_refit_global_structure=True`` (standard workflow: refit on G* then draw).
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be > 0")
    validate_aggregate_args(
        result.tau, mc_contributors, mc_refit_global_structure, None, False
    )
    prepared = prepare_and_validate(
        adj_mats,
        result.tau,
        node_names,
        filtered_per_subject=filtered_per_subject,
        plot_data=None,
        pool_filt_for_plotting=False,
        n_draws=n_draws,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
        plot_filt=None,
        rng=rng,
    )
    if prepared.n_subjects != result.n_subjects:
        raise ValueError(
            f"subject count {prepared.n_subjects} does not match result "
            f"n_subjects={result.n_subjects}"
        )
    view = run_mc_path(
        prepared,
        result,
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
    if isinstance(result, ISAggregatedMDMView):
        return replace(
            view,
            data=result.data,
            Filt=result.Filt,
            Smoo=result.Smoo,
            DF=result.DF,
        )
    return view


def aggregate_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
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
    """
    Aggregate subject-specific DAG adjacency matrices into one consensus DAG.

    Thin wrapper around :func:`~mdmp.group_analysis.inds.pipeline.run_full`.
    See that module for the staged pipeline and
    :func:`vote_individual_structures` / :func:`run_inds_global_beta_mc` for
    stepwise use.
    """
    return run_full(
        adj_mats,
        tau,
        node_names,
        threshold_mode=threshold_mode,
        filtered_per_subject=filtered_per_subject,
        time_index=time_index,
        time_indices=time_indices,
        n_draws=n_draws,
        rng=rng,
        pooling=pooling,
        plot_data=plot_data,
        plot_filt=plot_filt,
        plot_smoo=plot_smoo,
        plot_df=plot_df,
        pool_filt_for_plotting=pool_filt_for_plotting,
        mc_quantiles=mc_quantiles,
        mc_posterior=mc_posterior,
        mc_contributors=mc_contributors,
        mc_refit_global_structure=mc_refit_global_structure,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )


def aggregate_with_options(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    options: ISAggregateOptions,
) -> ISAggregatedMDMView:
    """Same as :func:`aggregate_individual_structures` with bundled options."""
    return aggregate_individual_structures(
        adj_mats, tau, node_names, **asdict(options)
    )


compute_individual_structure_consensus = aggregate_with_options
