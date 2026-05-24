"""Dataclasses and type aliases for IS aggregation and global-beta Monte Carlo.

Statistical Interpretation
--------------------------
The objects in this module represent outputs of a two-stage pipeline:

1. **Empirical edge voting** + greedy DAG repair → a single consensus DAG G*.
2. **Monte Carlo aggregation** of independent per-subject DLM posteriors,
   conditioned on G*.

Key distinctions to keep in mind:

* **Conditional edge effects, not population averages.**  The default pooling
  averages only subjects that express each edge, estimating

      E[θ_{pc,t} | edge_{pc} = 1]

  *not* the unconditional population mean

      E[θ_{pc,t}] = (1/S) Σ_i θ_{i,pc,t}.

  Subjects without the edge are excluded from both the numerator and the
  divisor (they do not enter as zeros).

* **Fixed-structure inference.**  All posterior draws are conditioned on G*:

      p(θ | G*)

  Structural uncertainty p(G) is not propagated; Monte Carlo intervals do
  not encode graph uncertainty.

* **Independent subject posteriors.**  Subjects are sampled independently;
  the resulting intervals are not from a joint hierarchical model with
  population-level random effects (no shrinkage, no between-subject
  covariance structure).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from .voting import ThresholdMode

MCContributorMode = Literal["individual_edge", "all_subjects"]
MCPosteriorSource = Literal["filtered", "smoothed"]

# Canonical pooling mode names that make the conditional semantics explicit.
# The old names ("mean_with_edge", "sum_with_edge") are retained as
# backward-compatible aliases and continue to behave identically.
PoolingMode = Literal[
    "conditional_mean_among_edge_subjects",   # canonical name
    "conditional_sum_among_edge_subjects",    # canonical name
    "mean_with_edge",                         # backward-compatible alias
    "sum_with_edge",                          # backward-compatible alias
]


@dataclass
class GlobalBetaMCResult:
    """
    Monte Carlo draws of pooled edge coefficients aligned to a consensus DAG.

    Statistical Interpretation
    --------------------------
    **What these intervals represent.**
    ``beta_draws`` encodes uncertainty propagated from each subject's DLM
    state posterior through the align-and-pool transform, conditioned on the
    fixed consensus DAG G*.  Specifically:

    * Draws are taken **independently** across subjects (no joint hierarchical
      model, no shrinkage, no between-subject covariance structure).
    * Pooling with ``pooling='conditional_mean_among_edge_subjects'`` (or its
      backward-compatible alias ``'mean_with_edge'``) estimates:

          E[θ_{pc,t} | edge_{pc} = 1]

      This is the conditional posterior mean among subjects that expressed the
      edge, **not** an unconditional population average (absent subjects are
      omitted, not averaged in as zeros).

    **What these intervals do NOT represent.**

    * They are not hierarchical credible intervals from a population model.
    * They do not encode structural uncertainty (graph G* is fixed).
    * Inference is p(θ | G*), not the model average p(θ) = Σ_G p(θ|G) p(G).

    Procedure
    ---------
    1. For each replicate :math:`b=1,\\ldots,B`, draw a regression state vector
       per subject at the chosen time index(es) using :math:`(m_{i,t}, C_{i,t},
       n_t, d_t)` from **filtered** outputs, or :math:`(m,C)` from **smoothed**
       ``smt``/``sCt`` with :math:`n_t,d_t` taken from the filter at the same
       time when ``mc_posterior='smoothed'``.

       .. note::
           Smoothed Monte Carlo draws use smoothed state moments together with
           filtered variance parameters at the same time index.  This is a
           pragmatic approximation to the full smoothed Student-t posterior,
           not an exact draw from the joint smoothing distribution.

    2. For each global edge (p → c), align the local coefficient for parent p
       in each contributing subject and pool.  With
       ``pooling='conditional_mean_among_edge_subjects'``:

       .. math::
           \\bar{\\theta}_t^{(b)} =
               \\frac{1}{A}\\sum_{i \\in \\mathcal{A}} \\theta_{i,t}^{(b)}

       where :math:`\\mathcal{A}` is the contributor set and :math:`A =
       |\\mathcal{A}|`.  For ``mc_contributors='individual_edge'``,
       :math:`\\mathcal{A}` = subjects whose **individual** DAG contained that
       edge; :math:`A \\le S`.  For ``mc_contributors='all_subjects'`` after
       global refit, typically :math:`A = S`.

    3. Columns of ``beta_draws`` (and optional quantile / nan-mean summaries)
       are the empirical distribution of
       :math:`\\{\\bar{\\theta}_t^{(b)}\\}_{b=1}^B`.

    Attributes
    ----------
    beta_draws : np.ndarray
        Shape ``(n_draws, n_edges)`` for a single time index, or
        ``(n_draws, n_edges, n_times)`` when multiple times are requested.
    edges : list of tuple
        ``(parent_idx, child_idx)`` in the same column order as ``beta_draws``.
    n_contributors : np.ndarray
        Per-edge count :math:`A` of subjects that entered the mean or sum for
        that edge.
    time_index : int
        First / reference time slice.
    pooling : str
        Pooling policy label (``'conditional_mean_among_edge_subjects'`` or
        ``'conditional_sum_among_edge_subjects'``; legacy names
        ``'mean_with_edge'`` / ``'sum_with_edge'`` are identical in effect).
    metadata : dict
        Extra diagnostics including ``n_subjects``, ``mc_posterior``,
        ``mc_contributors``, ``pooling_semantics`` (human-readable description
        of what the pooling computes), and ``contributors_per_edge`` (mapping
        ``(parent_idx, child_idx)`` → contributor count).
    time_indices_mc : tuple of int, optional
        Time indices when ``beta_draws`` has a time dimension.
    beta_quantiles : np.ndarray, optional
        Empirical quantiles of ``beta_draws`` along the draw axis.
    quantile_levels : tuple of float, optional
        Probability levels for ``beta_quantiles``.
    beta_mean : np.ndarray, optional
        ``numpy.nanmean`` over draws (axis 0).
    beta_var : np.ndarray, optional
        ``numpy.nanvar`` over draws (axis 0).
    """

    beta_draws: np.ndarray
    edges: List[Tuple[int, int]]
    n_contributors: np.ndarray
    time_index: int
    pooling: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_indices_mc: Optional[Tuple[int, ...]] = None
    beta_quantiles: Optional[np.ndarray] = None
    quantile_levels: Optional[Tuple[float, ...]] = None
    beta_mean: Optional[np.ndarray] = None
    beta_var: Optional[np.ndarray] = None


# Canonical name that makes inferential conditioning explicit.
# ``GlobalBetaMCResult`` is retained as a backward-compatible alias.
ConditionalEdgePosteriorResult = GlobalBetaMCResult


@dataclass
class ISAggregationResult:
    """
    Result of aggregating individual structures into one consensus DAG.

    Statistical Interpretation
    --------------------------
    The ``adj_mat`` is the output of empirical edge-frequency voting followed
    by greedy acyclic repair.  It is a **consensus summary** of the observed
    subject DAGs, not a posterior distribution over graphs.

    When ``global_beta_mc`` is present, all Monte Carlo inference is
    **conditional on this fixed consensus DAG** (``metadata["conditioning"]``
    will be ``"fixed_consensus_dag"``).  Structural uncertainty is not
    propagated; p(G) does not enter the interval computation.

    Attributes
    ----------
    adj_mat : np.ndarray
        Binary (N, N) adjacency matrix; ``adj_mat[i, j] == 1`` means
        parent i → child j.
    node_names : list of str
        Variable names aligned with matrix rows/columns.
    n_subjects : int
        Number of subject graphs aggregated.
    tau : float
        Threshold used for voting (see ``threshold_mode`` in ``metadata``).
    metadata : dict
        Includes ``edge_counts``, ``edge_frequencies``,
        ``edges_removed_for_acyclicity`` (greedy cycle breaking; not necessarily
        a global minimum feedback arc set), ``threshold_mode``
        (``'strict'`` vs ``'inclusive'``), ``graph_repair_strategy``
        (``'greedy_lowest_frequency_edge'``), and ``conditioning``
        (``'fixed_consensus_dag'``).
    global_beta_mc : GlobalBetaMCResult, optional
        Present when ``n_draws > 0``.
    """

    adj_mat: np.ndarray
    node_names: List[str]
    n_subjects: int
    tau: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    global_beta_mc: Optional[GlobalBetaMCResult] = None


@dataclass
class ISAggregatedMDMView(ISAggregationResult):
    """
    IS aggregation output structured like :class:`mdmp.model.MDM` for plotting.

    Concern separation
    ------------------
    This object has two distinct roles that should be reasoned about separately:

    * **Inference outputs** (inherited from :class:`ISAggregationResult`):
      ``adj_mat``, ``metadata``, ``global_beta_mc`` — the statistical results.
    * **Plot-ready adapters** (fields below): ``data``, ``Filt``, ``Smoo``,
      ``DF`` — convenience attributes populated only when plotting routines
      are enabled.  They do not affect inference.
    * **Refit artifacts**: ``refit_filt_per_subject``,
      ``refit_smoo_per_subject`` — auxiliary outputs from optional per-subject
      refits; stored for diagnostics, not required for interpreting
      ``global_beta_mc``.

    Inherits edge-voting metadata from :class:`ISAggregationResult` and adds
    the same optional attributes that :class:`mdmp.model.MDM` exposes so
    :mod:`mdmp.plotting` functions can be reused when those fields are set
    (e.g. ``plot_filt`` / ``plot_data`` passed to
    :func:`aggregate_individual_structures`).

    Attributes
    ----------
    data : np.ndarray, optional
        Time series ``(T, N)`` aligned with ``node_names`` (for
        ``plot_marginal``, ``plot_idag``).
    Filt : dict, optional
        Same layout as ``MDM.Filt`` (``mt``, ``Ct``, ``nt``, ``dt``, optional
        ``row_names``) for the **global** DAG.  This is a plotting adapter,
        not a joint posterior on the global graph.
    Smoo : dict, optional
        Same layout as ``MDM.Smoo`` when using smoothed plots.
    DF : dict, optional
        Discount-factor outputs, same role as ``MDM.DF`` if needed downstream.
    refit_filt_per_subject : list of dict, optional
        When ``mc_refit_global_structure=True`` and ``n_draws > 0``, one
        ``Filt`` dict per subject after refitting the consensus DAG.
    refit_smoo_per_subject : list of dict, optional
        Parallel ``Smoo`` dicts when refitting was run.
    """

    data: Optional[np.ndarray] = None
    Filt: Optional[Dict[str, Any]] = None
    Smoo: Optional[Dict[str, Any]] = None
    DF: Optional[Dict[str, Any]] = None
    refit_filt_per_subject: Optional[List[Dict[str, Any]]] = None
    refit_smoo_per_subject: Optional[List[Dict[str, Any]]] = None

    def __repr__(self) -> str:
        n_edge = int(np.sum(self.adj_mat != 0))
        return (
            f"ISAggregatedMDMView(nodes={len(self.node_names)}, "
            f"time_points={self.data.shape[0] if self.data is not None else 0}, "
            f"edges={n_edge}/{self.adj_mat.size}, n_subjects={self.n_subjects})"
        )


# Canonical name that makes the plotting-adapter role explicit.
# ``ISAggregatedMDMView`` is retained as a backward-compatible alias.
ISPlotAdapter = ISAggregatedMDMView


@dataclass
class ISAggregateOptions:
    """
    Bundles keyword-only arguments for :func:`aggregate_individual_structures`.

    Only :class:`ISAggregateOptions` fields are optional configuration; the
    graph list and ``tau`` stay on :func:`aggregate_with_options` /
    :func:`compute_individual_structure_consensus` / the main aggregate
    function.  Defaults match :func:`aggregate_individual_structures` so you
    can construct ``ISAggregateOptions(n_draws=100, rng=...)`` and omit
    unrelated fields.

    Notes
    -----
    **Nothing here is strictly required** for a minimal global-DAG vote: you
    only need subject adjacency matrices (or MDMs) and ``tau`` on the aggregate
    call.  Enable plotting, Monte Carlo, or refit by setting the corresponding
    fields on this dataclass (or passing the same keywords to
    :func:`aggregate_individual_structures`).

    The ``pooling`` field accepts the canonical conditional names
    ``'conditional_mean_among_edge_subjects'`` and
    ``'conditional_sum_among_edge_subjects'``, as well as the legacy aliases
    ``'mean_with_edge'`` and ``'sum_with_edge'`` (identical behaviour).
    """

    threshold_mode: ThresholdMode = "strict"
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None
    time_index: int = 0
    time_indices: Optional[Sequence[int]] = None
    n_draws: int = 0
    rng: Optional[Any] = None
    pooling: PoolingMode = "mean_with_edge"
    plot_data: Optional[np.ndarray] = None
    plot_filt: Optional[Mapping[str, Any]] = None
    plot_smoo: Optional[Mapping[str, Any]] = None
    plot_df: Optional[Mapping[str, Any]] = None
    pool_filt_for_plotting: bool = False
    mc_quantiles: Optional[Sequence[float]] = None
    mc_posterior: MCPosteriorSource = "filtered"
    mc_contributors: MCContributorMode = "individual_edge"
    mc_refit_global_structure: bool = False
    data_per_subject: Optional[Sequence[np.ndarray]] = None
    mc_refit_n_jobs: Optional[int] = None
