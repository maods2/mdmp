"""
Individual Structure (IS) aggregation: combine subject DAGs by edge voting.

Statistical Interpretation
--------------------------
The pipeline produces a **consensus DAG** G* via empirical edge-frequency
voting, then optionally runs **Monte Carlo aggregation** of independent
per-subject DLM posteriors conditioned on G*.

Key assumptions that must be explicit:

1. **Conditional edge effects, not population effects.**
   Default pooling (``pooling='mean_with_edge'`` / canonical name
   ``'conditional_mean_among_edge_subjects'``) estimates

       E[θ_{pc,t} | edge_{pc} = 1]

   using only subjects that expressed the edge.  The divisor is the number
   of contributing subjects A, not total subject count S.  This is **not**
   an unconditional population average.

2. **Fixed-structure inference.**
   All posterior inference is conditioned on the fixed consensus DAG G*:

       p(θ | G*)

   Structural uncertainty is not propagated; the resulting intervals do not
   encode uncertainty about G* itself.

3. **Independent subject posteriors.**
   Subjects are drawn independently; the result is not a joint hierarchical
   posterior with population-level random effects.

Pipeline (high level)
---------------------
1. Normalize the first argument (optional single MDM / single 2D adjacency).
2. Coerce MDM-like inputs to per-subject adjacency + optional Filt / plot_data.
3. Validate adjacency list and optional plot arrays.
4. **Vote**: edge frequency above ``tau`` (strict ``>`` or inclusive ``>=``;
   see ``threshold_mode``) → candidate global DAG; **repair** cycles by
   **greedy** removal (one directed cycle at a time: drop its
   lowest-frequency edge, repeat) — not necessarily a minimum feedback arc
   set (global FAS).
5. Optionally build a pooled ``Filt`` for plotting, or run Monte Carlo on DLM
   states for global edge coefficients (individual-filter posteriors by
   default; optional refit on the consensus DAG for global-structure
   posteriors).
"""

from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .coercion import (
    _coerce_subjects_for_aggregation,
    _normalize_first_argument,
    _validate_adj_list,
)
from .monte_carlo import (
    _monte_carlo_global_edge_beta,
    _smooth_filtered_sequence,
)
from .pooled_filtering import build_plot_filt_from_subjects
from .refit import _MCInputs, build_mc_inputs
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
from .validation import validate_after_coercion, validate_aggregate_args
from .voting import ThresholdMode, _vote_threshold_and_repair_cycles

__all__ = [
    # Inference result types
    "GlobalBetaMCResult",
    "ConditionalEdgePosteriorResult",  # canonical alias for GlobalBetaMCResult
    "ISAggregationResult",
    # Plot-adapter type
    "ISAggregatedMDMView",
    "ISPlotAdapter",                   # canonical alias for ISAggregatedMDMView
    # Options
    "ISAggregateOptions",
    # Type aliases
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    # Functions
    "aggregate_individual_structures",
    "aggregate_with_options",
    "compute_individual_structure_consensus",  # canonical alias
    "build_plot_filt_from_subjects",
]


def _resolve_aggregated_view_filt(
    plot_filt: Optional[Mapping[str, Any]],
    pool_filt_for_plotting: bool,
    out_adj: np.ndarray,
    filtered_eff: Optional[Sequence[Mapping[str, Any]]],
    arrays: List[np.ndarray],
    names: List[str],
) -> Optional[Dict[str, Any]]:
    if plot_filt is not None:
        return dict(plot_filt)
    if not pool_filt_for_plotting:
        return None
    assert filtered_eff is not None
    return build_plot_filt_from_subjects(out_adj, filtered_eff, arrays, names)


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

    Only ``adj_mats`` is required; ``tau`` defaults to ``0.5``.  All parameters
    after ``node_names`` are keyword-only with defaults.  For a single options
    object instead of many keywords, see :func:`aggregate_with_options` and
    :class:`ISAggregateOptions`.

    Statistical Interpretation
    --------------------------
    **Inference is conditional on the consensus DAG G*.**
    The pipeline first fixes G* via edge-frequency voting and greedy repair,
    then draws from per-subject DLM posteriors given G*:

        p(θ | G*)

    Structural uncertainty is **not** propagated; G* is treated as known.
    Monte Carlo intervals reflect posterior uncertainty of individual DLM
    states, not uncertainty about which graph generated the data.

    **Pooling is conditional on edge membership.**
    With ``pooling='mean_with_edge'`` (canonical:
    ``'conditional_mean_among_edge_subjects'``), the group mean at each
    edge uses only subjects that expressed that edge.  The divisor is A
    (number of contributors), not S (all subjects).  Absent subjects are
    excluded, not averaged in as zeros.  This estimates:

        E[θ_{pc,t} | edge_{pc} = 1]

    not the unconditional population mean.

    **Subjects are independent.**
    No joint hierarchical model is fitted.  Draws are independent across
    subjects; the resulting intervals are not hierarchical credible intervals
    and do not encode between-subject covariance structure.

    Voting and DAG repair
    ---------------------
    For each directed edge (i → j), compute the fraction of subjects that
    include the edge.  Include it in the pooled graph using ``threshold_mode``:
    ``strict`` keeps edges with frequency **>** ``tau`` (default); ``inclusive``
    keeps edges with frequency **≥** ``tau``.  If the thresholded graph has
    directed cycles, **greedy** repair removes one lowest-frequency edge from a
    detected directed cycle at a time until the graph is acyclic (not
    necessarily a minimum feedback arc set).

    Monte Carlo
    -----------
    When ``n_draws > 0``, pools regression-state draws onto global edges.  By
    default draws use each subject's **filtered** posterior under its
    **individual** DAG, and only subjects with that edge contribute
    (``mc_contributors='individual_edge'``).  Set
    ``mc_refit_global_structure=True`` to refit :class:`mdmp.model.MDM`-style
    filtering (fixed consensus DAG) per subject via
    :func:`mdmp.model.refit_mdm_on_structure`, then draw from those
    posteriors.  ``mc_contributors='all_subjects'`` requires
    ``mc_refit_global_structure=True``.

    Use ``mc_posterior='smoothed'`` to sample using smoothed ``smt``/``sCt``
    with filter ``nt``/``dt`` at the same time index — a pragmatic
    approximation (see :mod:`mdmp.group_analysis.is.monte_carlo`).

    Parameters
    ----------
    adj_mats : sequence of array-like, DataFrame, or fitted MDM
        One (N, N) binary adjacency per subject; ``[i, j] == 1`` means i → j.
        Alternatively, pass a sequence of fitted :class:`mdmp.model.MDM`
        instances (duck-typed: objects with ``adj_mat``, ``Filt``,
        ``node_names``).
    tau : float, optional
        Threshold in (0, 1].  Default 0.5 (majority under strict mode).
    threshold_mode : {'strict', 'inclusive'}, optional
        ``strict``: keep edge if frequency ``> tau``.
        ``inclusive``: keep if ``>= tau``.
    node_names : sequence of str, optional
        Names of length N.
    filtered_per_subject : sequence of dict-like, optional
        One dict per subject with keys ``'mt'``, ``'Ct'``, ``'nt'``, ``'dt'``,
        ``'Rt'`` per child index.  Required when ``n_draws > 0`` and inputs
        are not MDM instances.
    time_index : int, optional
        Time index t for posterior draws when ``time_indices`` is not set.
    time_indices : sequence of int, optional
        If set, run Monte Carlo at each listed t; ``beta_draws`` shape
        ``(n_draws, n_edges, T)``.
    n_draws : int, optional
        If > 0, build :attr:`ISAggregationResult.global_beta_mc`.
    rng : numpy.random.Generator, optional
        Required when ``n_draws > 0``.
    pooling : PoolingMode, optional
        How to pool coefficients across contributing subjects.
        Canonical names: ``'conditional_mean_among_edge_subjects'``,
        ``'conditional_sum_among_edge_subjects'``.  Legacy aliases
        ``'mean_with_edge'`` / ``'sum_with_edge'`` are identical in effect.
    plot_data : np.ndarray, optional
        ``(T, N)`` series aligned with ``node_names`` for plotting.
    plot_filt, plot_smoo, plot_df : mapping, optional
        Populate ``Filt`` / ``Smoo`` / ``DF`` on the returned
        :class:`ISAggregatedMDMView` (plot-adapter fields only).
    pool_filt_for_plotting : bool, optional
        If True, build ``Filt`` via :func:`build_plot_filt_from_subjects`.
    mc_quantiles : sequence of float, optional
        Store empirical quantiles of ``beta_draws`` in
        ``global_beta_mc.beta_quantiles``.
    mc_posterior : {'filtered', 'smoothed'}, optional
        Whether Monte Carlo draws use filtered or smoothed state moments.
    mc_contributors : {'individual_edge', 'all_subjects'}, optional
        Contributor set per edge.
    mc_refit_global_structure : bool, optional
        If True, refit per subject on the consensus DAG before Monte Carlo.
    data_per_subject : sequence of array-like, optional
        Per-subject ``(T, N)`` series when ``mc_refit_global_structure=True``.
    mc_refit_n_jobs : int, optional
        Parallel workers for refit pipelines.

    Returns
    -------
    ISAggregatedMDMView
        Subclass of :class:`ISAggregationResult` with optional plot-adapter
        fields ``data`` / ``Filt`` / ``Smoo`` / ``DF``.

    Notes
    -----
    The Student-*t* marginal of the DLM state is sampled via a Gamma–Normal
    mixture (:math:`\\phi \\sim \\mathrm{Gamma}(n_t/2, d_t/2)`,
    :math:`\\theta\\mid\\phi \\sim \\mathcal{N}(m_t, C_t/\\phi)`), matching
    ``mdmp.dlm`` filtering when ``mc_posterior='filtered'``.  Monte Carlo
    output does not encode dependence across subjects beyond independent
    filtering/smoothing.
    """
    # Stage 1: validate top-level arguments
    validate_aggregate_args(
        tau, mc_contributors, mc_refit_global_structure, plot_filt, pool_filt_for_plotting
    )

    # Stage 2: normalize and coerce inputs
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

    # Stage 3: validate coerced inputs
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

    # Stage 4: vote + greedy DAG repair
    out_adj, meta = _vote_threshold_and_repair_cycles(
        arrays, tau, names, threshold_mode=threshold_mode
    )

    # Stage 5: optional plot-adapter Filt assembly
    filt_final = _resolve_aggregated_view_filt(
        plot_filt, pool_filt_for_plotting, out_adj, filtered_eff, arrays, names
    )

    # Stage 6: assemble base result
    result = ISAggregatedMDMView(
        adj_mat=out_adj,
        node_names=names,
        n_subjects=s,
        tau=tau,
        metadata=meta,
        global_beta_mc=None,
        data=None if plot_data_eff is None else np.asarray(plot_data_eff, dtype=float),
        Filt=filt_final,
        Smoo=None if plot_smoo is None else dict(plot_smoo),
        DF=None if plot_df is None else dict(plot_df),
    )

    if n_draws == 0:
        return result

    # Stage 7: optional Monte Carlo aggregation
    assert rng is not None
    mc = build_mc_inputs(
        mc_refit_global_structure=mc_refit_global_structure,
        mc_posterior=mc_posterior,
        arrays=arrays,
        names=names,
        n_subjects=s,
        n_nodes=n,
        out_adj=out_adj,
        filtered_eff=filtered_eff,
        mdm_data_per_subject=mdm_data_per_subject,
        data_per_subject=data_per_subject,
        mc_refit_n_jobs=mc_refit_n_jobs,
    )

    gb = _monte_carlo_global_edge_beta(
        mc.filt_per_subject,
        arrays,
        mc.design_adjs,
        result,
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

    # Stage 8: assemble final result with MC outputs
    return replace(
        result,
        global_beta_mc=gb,
        refit_filt_per_subject=mc.refit_filt_per_subject,
        refit_smoo_per_subject=mc.refit_smoo_per_subject,
    )


def aggregate_with_options(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    options: ISAggregateOptions,
) -> ISAggregatedMDMView:
    """
    Same as :func:`aggregate_individual_structures` but with keyword-only
    arguments supplied via :class:`ISAggregateOptions` (defaults match the
    flat API).

    See also :func:`compute_individual_structure_consensus` (canonical alias).
    """
    return aggregate_individual_structures(
        adj_mats, tau, node_names, **asdict(options)
    )


# Canonical name that makes the scientific purpose of the function explicit.
# ``aggregate_with_options`` is retained as a backward-compatible alias.
compute_individual_structure_consensus = aggregate_with_options
