"""
Individual Structure (IS) aggregation: combine subject DAGs by edge voting.

Pipeline (high level)
---------------------
1. Normalize the first argument (optional single MDM / single 2D adjacency).
2. Coerce MDM-like inputs to per-subject adjacency + optional ``Filt`` / ``plot_data``.
3. Validate adjacency list and optional plot arrays.
4. **Vote**: edge frequency above ``tau`` (strict ``>`` or inclusive ``>=``; see
   ``threshold_mode``) → candidate global DAG; **repair** cycles by **greedy**
   removal (one directed cycle at a time: drop its lowest-frequency edge, repeat),
   which need not coincide with a minimum feedback arc set (global FAS).
5. Optionally build a pooled ``Filt`` for plotting, or run Monte Carlo on DLM
   states for global edge coefficients (individual-filter posteriors by default;
   optional refit on the aggregated DAG for global-structure posteriors).
"""

from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ...model.refit import refit_mdm_on_structure
from ..._node_dispatch import smooth_all_nodes
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
from .results import (
    GlobalBetaMCResult,
    ISAggregateOptions,
    ISAggregatedMDMView,
    ISAggregationResult,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
)
from .voting import ThresholdMode, _vote_threshold_and_repair_cycles

__all__ = [
    "GlobalBetaMCResult",
    "ISAggregateOptions",
    "ISAggregatedMDMView",
    "ISAggregationResult",
    "MCContributorMode",
    "MCPosteriorSource",
    "PoolingMode",
    "aggregate_individual_structures",
    "aggregate_with_options",
    "build_plot_filt_from_subjects",
]


class _MCInputs(NamedTuple):
    """Bundled Monte Carlo inputs after the global DAG is fixed."""

    filt_per_subject: List[Mapping[str, Any]]
    design_adjs: List[np.ndarray]
    smoothed_per_subject: Optional[List[Dict[str, Any]]]
    refit_filt_per_subject: Optional[List[Dict[str, Any]]]
    refit_smoo_per_subject: Optional[List[Dict[str, Any]]]


def _validate_aggregate_args(
    tau: float,
    mc_contributors: MCContributorMode,
    mc_refit_global_structure: bool,
    plot_filt: Optional[Mapping[str, Any]],
    pool_filt_for_plotting: bool,
) -> None:
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if mc_contributors == "all_subjects" and not mc_refit_global_structure:
        raise ValueError(
            "mc_contributors='all_subjects' requires mc_refit_global_structure=True"
        )
    if plot_filt is not None and pool_filt_for_plotting:
        raise ValueError("pass only one of plot_filt=... or pool_filt_for_plotting=True")


def _validate_after_coercion(
    n_draws: int,
    filtered_eff: Optional[Sequence[Mapping[str, Any]]],
    mc_refit_global_structure: bool,
    rng: Optional[np.random.Generator],
    pool_filt_for_plotting: bool,
    n_subjects: int,
    filtered_len: Optional[int],
    plot_data_eff: Optional[np.ndarray],
    n_nodes: int,
) -> None:
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


def _per_subject_data_for_refit(
    n_subjects: int,
    n_nodes: int,
    data_per_subject: Optional[Sequence[np.ndarray]],
    mdm_data_per_subject: Optional[List[np.ndarray]],
) -> List[np.ndarray]:
    if data_per_subject is not None:
        if len(data_per_subject) != n_subjects:
            raise ValueError(
                f"data_per_subject length {len(data_per_subject)} must match "
                f"number of subjects {n_subjects}"
            )
        return [np.asarray(x, dtype=float) for x in data_per_subject]
    if mdm_data_per_subject is not None:
        return mdm_data_per_subject
    raise ValueError(
        "mc_refit_global_structure=True requires per-subject data "
        "(fitted MDM inputs with .data, or data_per_subject=...)"
    )


def _refit_each_subject_on_global_adj(
    datas: List[np.ndarray],
    global_adj: np.ndarray,
    names: List[str],
    n_nodes: int,
    mc_refit_n_jobs: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Mapping[str, Any]]]:
    refit_filt: List[Dict[str, Any]] = []
    refit_smoo: List[Dict[str, Any]] = []
    filt_mc: List[Mapping[str, Any]] = []
    for di in datas:
        if di.ndim != 2 or di.shape[1] != n_nodes:
            raise ValueError(
                f"each per-subject data array must have shape (T, {n_nodes}); "
                f"got {di.shape}"
            )
        rfit = refit_mdm_on_structure(
            di,
            global_adj,
            node_names=names,
            verbose=False,
            n_jobs=mc_refit_n_jobs,
        )
        refit_filt.append(dict(rfit.Filt))
        smoo_raw = getattr(rfit, "Smoo", None)
        if smoo_raw is None:
            smoo_raw = smooth_all_nodes(
                mt=rfit.Filt["mt"],
                Ct=rfit.Filt["Ct"],
                Rt=rfit.Filt["Rt"],
                nt=rfit.Filt["nt"],
                dt=rfit.Filt["dt"],
                n_jobs=mc_refit_n_jobs,
            )
        refit_smoo.append(dict(smoo_raw))
        filt_mc.append(rfit.Filt)
    return refit_filt, refit_smoo, filt_mc


def _build_mc_inputs(
    *,
    mc_refit_global_structure: bool,
    mc_posterior: MCPosteriorSource,
    arrays: List[np.ndarray],
    names: List[str],
    n_subjects: int,
    n_nodes: int,
    out_adj: np.ndarray,
    filtered_eff: Optional[Sequence[Mapping[str, Any]]],
    mdm_data_per_subject: Optional[List[np.ndarray]],
    data_per_subject: Optional[Sequence[np.ndarray]],
    mc_refit_n_jobs: Optional[int],
) -> _MCInputs:
    global_adj = np.asarray(out_adj, dtype=int)

    if mc_refit_global_structure:
        datas = _per_subject_data_for_refit(
            n_subjects, n_nodes, data_per_subject, mdm_data_per_subject
        )
        refit_filt, refit_smoo, filt_mc = _refit_each_subject_on_global_adj(
            datas, global_adj, names, n_nodes, mc_refit_n_jobs
        )
        design = [global_adj.copy() for _ in range(n_subjects)]
        smoo_seq = refit_smoo if mc_posterior == "smoothed" else None
        return _MCInputs(filt_mc, design, smoo_seq, refit_filt, refit_smoo)

    assert filtered_eff is not None
    filt_mc = list(filtered_eff)
    smoo_seq = (
        _smooth_filtered_sequence(filt_mc, mc_refit_n_jobs)
        if mc_posterior == "smoothed"
        else None
    )
    return _MCInputs(filt_mc, arrays, smoo_seq, None, None)


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
    Aggregate subject-specific DAG adjacency matrices into one global DAG.

    Only ``adj_mats`` is required; ``tau`` defaults to ``0.5``. All parameters
    after ``node_names`` are keyword-only with defaults. For a single options
    object instead of many keywords, see :func:`aggregate_with_options` and
    :class:`ISAggregateOptions`.

    For each directed edge (i → j), compute the fraction of subjects that
    include the edge. Include it in the pooled graph using ``threshold_mode``:
    ``strict`` keeps edges with frequency **>** ``tau`` (default); ``inclusive``
    keeps edges with frequency **≥** ``tau``. If the thresholded graph has
    directed cycles, **greedy** repair removes one lowest-frequency edge from a
    detected directed cycle at a time until the graph is acyclic (not necessarily
    a minimum feedback arc set).

    Monte Carlo (when ``n_draws > 0``) pools regression-state draws onto global
    edges. By default draws use each subject's **filtered** posterior under its
    **individual** DAG, and only subjects with that edge contribute
    (``mc_contributors='individual_edge'``). With ``pooling='mean_with_edge'``,
    the implemented group mean divides by the number of **contributors**
    :math:`A` at that edge (not by total subject count :math:`S` when some
    subjects lack the edge on their individual DAG). Set
    ``mc_refit_global_structure=True`` to refit :class:`mdmp.model.MDM`-style
    filtering (fixed aggregated DAG) per subject via
    :func:`mdmp.model.refit_mdm_on_structure`, then draw from those posteriors.
    ``mc_contributors='all_subjects'`` pools across all subjects at each global
    edge and requires ``mc_refit_global_structure=True``. Draws are **independent**
    across subjects; stored intervals reflect that assumption, not a joint
    hierarchical population model. Use ``mc_posterior='smoothed'`` to sample using
    smoothed ``smt``/``sCt`` with filter ``nt``/``dt`` at the same time index
    (Gamma–Normal step reusing filter degrees of freedom—a pragmatic approximation).

    Parameters
    ----------
    adj_mats : sequence of array-like, DataFrame, or fitted MDM
        One (N, N) binary adjacency per subject; ``[i, j] == 1`` means i → j.
        Alternatively, pass a sequence of fitted :class:`mdmp.model.MDM` instances
        (duck-typed: objects with ``adj_mat``, ``Filt``, ``node_names``): adjacency
        is taken as ``(adj_mat > 0)`` (off-diagonal), and when ``filtered_per_subject``
        is omitted, ``Filt`` is read from each model for ``n_draws > 0`` or
        ``pool_filt_for_plotting=True``. With MDM inputs, if ``plot_data`` is omitted
        and ``pool_filt_for_plotting=True``, a cross-subject mean of ``data`` is used
        when all models share the same ``(T, N)`` shape. Do not mix MDMs and raw
        adjacency matrices in one call.
    tau : float, optional
        Threshold in (0, 1]. Default 0.5 (majority under strict mode).
    threshold_mode : {'strict', 'inclusive'}, optional
        ``strict``: keep edge if frequency ``> tau``. ``inclusive``: keep if ``>= tau``.
    node_names : sequence of str, optional
        Names of length N. If omitted, taken from the first DataFrame, MDM
        ``node_names``, or ``V1``…``VN``.
    filtered_per_subject : sequence of dict-like, optional
        One dict per subject (same length and order as ``adj_mats``) with keys
        ``'mt'``, ``'Ct'``, ``'nt'``, ``'dt'``, ``'Rt'`` per child index, as from filtering.
        Required when ``n_draws > 0`` and inputs are **not** MDM instances (MDM path
        fills this from ``m.Filt`` when omitted). Explicit values override MDM ``Filt``.
    time_index : int, optional
        Time index :math:`t` for posterior draws when ``time_indices`` is not set
        (default 0).
    time_indices : sequence of int, optional
        If set, run the Monte Carlo procedure at **each** listed time :math:`t`
        (in order). ``beta_draws`` then has shape ``(n\\_draws, n\\_edges, T)``.
        When omitted, only ``time_index`` is used (shape ``(n\\_draws, n\\_edges)``).
    n_draws : int, optional
        If > 0, build :attr:`ISAggregationResult.global_beta_mc` using this
        many Monte Carlo draws.
    rng : numpy.random.Generator, optional
        Required when ``n_draws > 0``.
    pooling : {'mean_with_edge', 'sum_with_edge'}, optional
        How to pool coefficients across contributing subjects for each global edge.
        ``mean_with_edge`` uses :math:`1/A` over contributors at that edge, not
        :math:`1/S` over all subjects when ``mc_contributors='individual_edge'``.
    plot_data : np.ndarray, optional
        ``(T, N)`` series aligned with ``node_names`` for :mod:`mdmp.plotting`
        routines that need ``MDM.data``.
    plot_filt, plot_smoo, plot_df : mapping, optional
        Populate ``Filt`` / ``Smoo`` / ``DF`` on the returned
        :class:`ISAggregatedMDMView` (same key conventions as :class:`mdmp.model.MDM`).
    pool_filt_for_plotting : bool, optional
        If True (with ``filtered_per_subject`` set), build ``Filt`` via
        :func:`build_plot_filt_from_subjects` for use with :mod:`mdmp.plotting`.
        Ignored if ``plot_filt`` is provided.
    mc_quantiles : sequence of float, optional
        If given (e.g. ``(0.025, 0.5, 0.975)``), store empirical quantiles of
        ``beta_draws`` along the draw axis in ``global_beta_mc.beta_quantiles``.
    mc_posterior : {'filtered', 'smoothed'}, optional
        Whether Monte Carlo draws use filtered or smoothed state means/covariances.
    mc_contributors : {'individual_edge', 'all_subjects'}, optional
        ``individual_edge``: only subjects whose **individual** DAG had the edge
        contribute to that edge's pool (and to the ``mean_with_edge`` divisor
        :math:`A`). ``all_subjects``: after global-structure refit, typically every
        subject contributes at each edge (requires ``mc_refit_global_structure``).
    mc_refit_global_structure : bool, optional
        If True, run :func:`mdmp.model.refit_mdm_on_structure` per subject
        on the aggregated DAG before Monte Carlo (requires per-subject ``data``).
    data_per_subject : sequence of array-like, optional
        ``(T, N)`` series per subject when ``mc_refit_global_structure=True`` with
        plain adjacency inputs. With MDM inputs, subject ``data`` is used automatically
        unless overridden here.
    mc_refit_n_jobs : int, optional
        Parallel workers forwarded to refit pipelines (per-subject refits run serially).

    Returns
    -------
    ISAggregatedMDMView
        Subclass of :class:`ISAggregationResult` with optional ``data``/``Filt``/
        ``Smoo``/``DF`` for plotting.

    Notes
    -----
    For weighted / fractional votes in the future, generalize the count tensor
    stored in ``metadata['edge_counts']``.

    The Student-*t* marginal of the DLM state is sampled via a Gamma–Normal
    mixture (:math:`\\phi \\sim \\mathrm{Gamma}(n_t/2, d_t/2)`,
    :math:`\\theta\\mid\\phi \\sim \\mathcal{N}(m_t, C_t/\\phi)`), matching
    ``mdmp.dlm`` filtering when ``mc_posterior='filtered'``. With
    ``mc_posterior='smoothed'``, using the same :math:`(n_t,d_t)` with smoothed
    :math:`(m_t,C_t)` is a pragmatic sampling shortcut (see
    :mod:`mdmp.group_analysis.is.mc_global_beta`). Monte Carlo output does not
    encode dependence across subjects beyond independent filtering/smoothing.
    """
    _validate_aggregate_args(
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
    _validate_after_coercion(
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

    out_adj, meta = _vote_threshold_and_repair_cycles(
        arrays, tau, names, threshold_mode=threshold_mode
    )

    filt_final = _resolve_aggregated_view_filt(
        plot_filt, pool_filt_for_plotting, out_adj, filtered_eff, arrays, names
    )

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

    assert rng is not None
    mc = _build_mc_inputs(
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
    """
    return aggregate_individual_structures(
        adj_mats, tau, node_names, **asdict(options)
    )
