"""Dataclasses and type aliases for IS aggregation and global-beta Monte Carlo.

Statistical Interpretation
--------------------------
The objects in this module represent outputs of a two-stage pipeline:

1. **Empirical edge voting** + greedy DAG repair → a single consensus DAG G*.
2. **Monte Carlo aggregation** of independent per-subject DLM posteriors,
   conditioned on G*.

Key distinctions to keep in mind:

* **Population mean over subjects.**  For each MC replicate :math:`b`,

      \\bar{\\theta}_t^{(b)} = \\frac{1}{S}\\sum_{i=1}^{S} \\theta_{it}^{(b)}

  with :math:`\\theta_{it}^{(b)}` drawn from each subject's marginal posterior at
  time :math:`t` (after refit on the consensus DAG G*).

* **Fixed-structure inference.**  All posterior samples are conditioned on G*:

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

MCPosteriorSource = Literal["filtered", "smoothed"]


@dataclass
class GlobalBetaMCResult:
    """
    Monte Carlo samples of pooled edge coefficients aligned to a consensus DAG.

    Statistical Interpretation
    --------------------------
    **What these intervals represent.**
    ``beta_samples`` encodes uncertainty propagated from each subject's DLM
    state posterior through the align-and-pool transform, conditioned on the
    fixed consensus DAG G*.  Specifically:

    * Samples are taken **independently** across subjects (no joint hierarchical
      model, no shrinkage, no between-subject covariance structure).
    * For each replicate :math:`b` and time :math:`t`,

          \\bar{\\theta}_t^{(b)} = \\frac{1}{S}\\sum_{i=1}^{S} \\theta_{it}^{(b)}

      with :math:`\\theta_{it}^{(b)}` from each subject's marginal filtered posterior.

    **What these intervals do NOT represent.**

    * They are not hierarchical credible intervals from a population model.
    * They do not encode structural uncertainty (graph G* is fixed).
    * Inference is p(θ | G*), not the model average p(θ) = Σ_G p(θ|G) p(G).

    Procedure
    ---------
    1. For each replicate :math:`b=1,\\ldots,B`, sample a regression state vector
       per subject at every filter time index using :math:`(m_{i,t}, C_{i,t},
       n_t, d_t)` from **filtered** outputs, or :math:`(m,C)` from **smoothed**
       ``smt``/``sCt`` with :math:`n_t,d_t` taken from the filter at the same
       time when ``mc_posterior='smoothed'``.

       .. note::
           Smoothed Monte Carlo samples use smoothed state moments together with
           filtered variance parameters at the same time index.  This is a
           pragmatic approximation to the full smoothed Student-t posterior,
           not an exact sample from the joint smoothing distribution.

    2. For each global edge (p → c), align the local coefficient for parent p
       and form the population mean at each replicate (divisor :math:`S`).

    3. Columns of ``beta_samples`` (and optional quantile / nan-mean summaries)
       are the empirical distribution of
       :math:`\\{\\bar{\\theta}_t^{(b)}\\}_{b=1}^B`.

    Attributes
    ----------
    beta_samples : np.ndarray
        Shape ``(mc_n_samples, n_edges, T)`` with ``T`` the filter series length.
    edges : list of tuple
        ``(parent_idx, child_idx)`` in the same column order as ``beta_samples``.
    n_contributors : np.ndarray
        Per-edge subject count :math:`S` (always the total number of subjects).
    time_index : int
        Always ``0`` (legacy field; use ``time_indices_mc`` for the full index
        range ``0 … T-1``).
    pooling : str
        Always ``'population_mean'``.
    metadata : dict
        Extra diagnostics including ``n_subjects``, ``mc_posterior``, and
        ``pooling_semantics``.
    time_indices_mc : tuple of int
        Time indices ``(0, 1, …, T-1)`` aligned with the last axis of ``beta_samples``.
    beta_quantiles : np.ndarray, optional
        Empirical quantiles of ``beta_samples`` along the sample axis.
    quantile_levels : tuple of float, optional
        Probability levels for ``beta_quantiles``.
    beta_mean : np.ndarray, optional
        ``numpy.nanmean`` over samples (axis 0).
    beta_var : np.ndarray, optional
        ``numpy.nanvar`` over samples (axis 0).
    """

    beta_samples: np.ndarray
    edges: List[Tuple[int, int]]
    n_contributors: np.ndarray
    time_index: int
    pooling: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_indices_mc: Tuple[int, ...] = ()
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
        Present when ``mc_n_samples > 0``.
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
    (populated automatically when aggregating fitted MDM instances).

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
        When ``mc_refit_global_structure=True`` and ``mc_n_samples > 0``, one
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

    Mirrors keyword-only arguments of
    :func:`~mdmp.group_analysis.inds.pipeline.aggregate_individual_structures`
    (not exported from the public package API).

    ``mc_refit_global_structure=None`` selects auto mode: refit on G* for
    MDM-like inputs, individual filtered posteriors otherwise.
    """

    mc_n_samples: int = 500
    rng: Optional[Any] = None
    mc_quantiles: Optional[Sequence[float]] = None
    mc_posterior: MCPosteriorSource = "filtered"
    mc_refit_global_structure: Optional[bool] = None
    mc_refit_n_jobs: Optional[int] = None
    mc_n_jobs: Optional[int] = None


@dataclass
class ISVoteOptions:
    """Voting-stage options for Individual Structure aggregation."""

    threshold_mode: ThresholdMode = "strict"


@dataclass
class ISMonteCarloOptions:
    """Monte Carlo / refit options (global edge posterior given G*)."""

    posterior_per_subject: Optional[Sequence[Mapping[str, Any]]] = None
    mc_n_samples: int = 0
    rng: Optional[Any] = None
    mc_quantiles: Optional[Sequence[float]] = None
    mc_posterior: MCPosteriorSource = "filtered"
    mc_refit_global_structure: Optional[bool] = None
    data_per_subject: Optional[Sequence[np.ndarray]] = None
    mc_refit_n_jobs: Optional[int] = None
    mc_n_jobs: Optional[int] = None


@dataclass
class ISMDMViewOptions:
    """Plot-adapter fields only (not inferential)."""

    time_series: Optional[np.ndarray] = None
    plot_filt: Optional[Mapping[str, Any]] = None
    plot_smoo: Optional[Mapping[str, Any]] = None
    plot_df: Optional[Mapping[str, Any]] = None
    pool_filt_for_plotting: bool = False
    posterior_per_subject: Optional[Sequence[Mapping[str, Any]]] = None


def merge_aggregate_options(
    vote: Optional[ISVoteOptions] = None,
    mc: Optional[ISMonteCarloOptions] = None,
    view: Optional[ISMDMViewOptions] = None,
) -> ISAggregateOptions:
    """
    Build a flat :class:`ISAggregateOptions` from grouped stage options.

    ``vote`` and ``view`` are accepted for API compatibility; only ``mc`` fields
    are merged into :class:`ISAggregateOptions` (aggregate no longer exposes
    voting or plot-adapter kwargs).
    """
    del vote, view  # voting/plot options are not on the aggregate entry point
    opts = ISAggregateOptions()
    if mc is not None:
        for field_name in (
            "mc_n_samples",
            "rng",
            "mc_quantiles",
            "mc_posterior",
            "mc_refit_global_structure",
            "mc_refit_n_jobs",
            "mc_n_jobs",
        ):
            setattr(opts, field_name, getattr(mc, field_name))
    return opts
