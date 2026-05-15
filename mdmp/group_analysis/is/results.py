"""Dataclasses and type aliases for IS aggregation and global-beta Monte Carlo."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from .voting import ThresholdMode

MCContributorMode = Literal["individual_edge", "all_subjects"]
MCPosteriorSource = Literal["filtered", "smoothed"]
PoolingMode = Literal["mean_with_edge", "sum_with_edge"]


@dataclass
class GlobalBetaMCResult:
    """
    Monte Carlo draws of pooled edge coefficients aligned to a global DAG.

    By default each replicate draws from each subject's **filtered** DLM state
    posterior under that subject's **individual** DAG (with coefficients aligned
    to the voted global edges). With ``mc_refit_global_structure=True``,
    posteriors come from a **global-structure refit** (same filtered/smoothed
    pipeline as :class:`mdmp.model.MDM`, fixed aggregated adjacency) per subject.

    Procedure (group pooling of draws):

    1. For each replicate :math:`b=1,\\ldots,B`, draw a regression state vector
       per subject at the chosen time index(es) using :math:`(m_{i,t}, C_{i,t},
       n_t, d_t)` from **filtered** outputs, or :math:`(m,C)` from **smoothed**
       ``smt``/``sCt`` with :math:`n_t,d_t` taken from the filter at the same
       time when ``mc_posterior='smoothed'``. The latter reuses the Gamma–Normal
       draw with smoothed moments—a **pragmatic approximation**, not an exact
       draw from the joint smoothing posterior.
    2. For each global edge, align the local coefficient for that parent on
       each contributing subject; with ``pooling='mean_with_edge'``, set
       :math:`\\bar{\\theta}_t^{(b)} = \\frac{1}{A}\\sum_{i \\in \\mathcal{A}}
       \\theta_{i,t}^{(b)}` where :math:`A=|\\mathcal{A}|`. This is a mean over
       **contributors only**, not :math:`\\frac{1}{S}\\sum_{i=1}^S` over all
       subjects (non-contributors are omitted, not averaged in as zeros). With
       ``mc_contributors='individual_edge'``, :math:`\\mathcal{A}` is subjects
       whose **individual** DAG contained that edge (before optional refit), so
       :math:`A\\le S`. With ``mc_contributors='all_subjects'`` after refit on the
       global DAG, typically all subjects contribute (:math:`A=S`).
    3. Columns of ``beta_draws`` (and optional quantiles / nan-mean summaries)
       summarize the empirical distribution of :math:`\\{\\bar{\\theta}_t^{(b)}\\}`.

    Subjects are treated as **independent** at the draw step; ``beta_draws``
    describe uncertainty propagated through the pool of independent per-subject
    posteriors, **not** a full joint hierarchical model over the population.

    Attributes
    ----------
    beta_draws : np.ndarray
        Shape ``(n_draws, n_edges)`` for a single time index, or
        ``(n_draws, n_edges, n_times)`` when multiple times are requested.
        One column (or column-time slice) per global directed edge
        ``parent -> child`` (excluding self-loops).
    edges : list of tuple
        ``(parent_idx, child_idx)`` in the same column order as ``beta_draws``.
    n_contributors : np.ndarray
        Per-edge count :math:`A` of subjects that entered the mean or sum for
        that edge (divisor for ``mean_with_edge`` when implemented as
        :math:`1/A` over contributors; see ``mc_contributors`` in
        :func:`aggregate_individual_structures`).
    time_index : int
        First / reference time slice (when a single ``t`` is used, this is it).
    pooling : str
        Pooling policy label (see :func:`aggregate_individual_structures`).
    metadata : dict
        Extra diagnostics (e.g. ``n_subjects``, ``mc_posterior``).
    time_indices_mc : tuple of int, optional
        If ``beta_draws`` has a time dimension, these are the time indices in
        axis order (length matches ``beta_draws.shape[2]``).
    beta_quantiles : np.ndarray, optional
        Empirical quantiles of ``beta_draws`` along the draw axis (axis 0):
        shape ``(n_levels, n_edges)`` or ``(n_levels, n_edges, n_times)``.
    quantile_levels : tuple of float, optional
        Probability levels used for ``beta_quantiles`` (same order as axis 0).
    beta_mean : np.ndarray, optional
        ``numpy.nanmean`` over draws (axis 0); same trailing shape as one slice
        of ``beta_draws``.
    beta_var : np.ndarray, optional
        ``numpy.nanvar`` over draws (axis 0); same shape convention as ``beta_mean``.
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


@dataclass
class ISAggregationResult:
    """
    Result of aggregating individual structures into one global DAG.

    Attributes
    ----------
    adj_mat : np.ndarray
        Binary (N, N) adjacency matrix; ``adj_mat[i, j] == 1`` means parent i → child j.
    node_names : list of str
        Variable names aligned with matrix rows/columns.
    n_subjects : int
        Number of subject graphs aggregated.
    tau : float
        Threshold used for voting (see ``threshold_mode`` in ``metadata``).
    metadata : dict
        Includes ``edge_counts``, ``edge_frequencies``,
        ``edges_removed_for_acyclicity`` (greedy cycle breaking; not necessarily a
        global minimum feedback arc set), ``threshold_mode`` (``'strict'`` vs
        ``'inclusive'``).
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

    Inherits edge voting metadata from :class:`ISAggregationResult` and adds
    the same optional attributes that :class:`mdmp.model.MDM` exposes so
    :mod:`mdmp.plotting` functions can be reused when those fields are set
    (e.g. ``plot_filt`` / ``plot_data`` passed to
    :func:`aggregate_individual_structures`).

    Attributes
    ----------
    data : np.ndarray, optional
        Time series ``(T, N)`` aligned with ``node_names`` (for ``plot_marginal``,
        ``plot_idag``).
    Filt : dict, optional
        Same layout as ``MDM.Filt`` (``mt``, ``Ct``, ``nt``, ``dt``, optional
        ``row_names``) for the **global** DAG (for ``plot_arcs``, etc.).
    Smoo : dict, optional
        Same layout as ``MDM.Smoo`` when using smoothed plots.
    DF : dict, optional
        Discount-factor outputs, same role as ``MDM.DF`` if needed downstream.
    refit_filt_per_subject : list of dict, optional
        When ``mc_refit_global_structure=True`` and ``n_draws > 0``, one ``Filt`` dict
        per subject after refitting the voted global DAG.
    refit_smoo_per_subject : list of dict, optional
        Parallel ``Smoo`` dicts when refitting was run (empty dicts if smoothing unused).
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


@dataclass
class ISAggregateOptions:
    """
    Bundles keyword-only arguments for :func:`aggregate_individual_structures`.

    Only :class:`ISAggregateOptions` fields are optional configuration; the graph
    list and ``tau`` stay on :func:`aggregate_with_options` / the main aggregate
    function. Defaults match :func:`aggregate_individual_structures` so you can
    construct ``ISAggregateOptions(n_draws=100, rng=...)`` and omit unrelated
    fields.

    Notes
    -----
    **Nothing here is strictly required** for a minimal global-DAG vote: you
    only need subject adjacency matrices (or MDMs) and ``tau`` on the aggregate
    call. Enable plotting, Monte Carlo, or refit by setting the corresponding
    fields on this dataclass (or passing the same keywords to
    :func:`aggregate_individual_structures`).
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

