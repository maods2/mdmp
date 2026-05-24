"""Monte Carlo pooling of DLM regression states onto consensus DAG edges.

Statistical Interpretation
--------------------------
For each time :math:`t` and replicate :math:`b = 1,\\ldots,B`, draw
:math:`\\theta_{it}^{(b)}` from subject :math:`i`'s marginal filtered posterior
(Student-t via the Gamma–Normal scale mixture), then form the group mean

    θ̄_t^{(b)} = (1/S) Σ_{i=1}^S θ_{it}^{(b)}

The empirical distribution of :math:`\\{\\bar\\theta_t^{(b)}\\}` propagates
uncertainty through the **average** transform (quantiles / credible summaries).

Requires refit on the consensus DAG G* so every subject contributes a coefficient
for each global edge.  Inference is conditional on fixed G*; subjects are sampled
independently (no hierarchical population model).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..._node_dispatch import smooth_all_nodes
from ...utils import build_design_matrix
from .results import GlobalBetaMCResult, ISAggregatedMDMView, MCPosteriorSource

POOLING_POPULATION_MEAN = "population_mean"


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def _sample_dlm_state_posterior(
    mt: np.ndarray,
    Ct: np.ndarray,
    nt: float,
    dt: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """
    Sample one DLM state vector from the marginal Student-t posterior.

    Implements :math:`\\theta \\sim t_{\\nu}(m, C)` via
    :math:`\\phi \\sim \\mathrm{Gamma}(n_t/2, 2/d_t)` and
    :math:`\\theta \\mid \\phi \\sim \\mathcal{N}(m, C/\\phi)`.

    Parameters
    ----------
    mt
        Posterior mean vector ``m_t`` for one child node's regression
        coefficients at a single time index, shape ``(p,)``.
    Ct
        Posterior covariance matrix ``C_t`` matching ``mt``, shape ``(p, p)``.
    nt
        Filtered hyperparameter ``n_t`` (drives the Student-t degrees of
        freedom) at the same time index; must be positive.
    dt
        Filtered hyperparameter ``d_t`` (scale in the Gamma mixing distribution)
        at the same time index; must be positive.
    gen
        NumPy random generator used for the Gamma and multivariate-normal draws.

    Returns
    -------
    np.ndarray
        One posterior draw of the coefficient vector, shape ``(p,)``.
    """
    m = np.asarray(mt, dtype=float).reshape(-1)
    p = m.shape[0]
    c = np.asarray(Ct, dtype=float).reshape(p, p)
    if nt <= 0.0 or dt <= 0.0:
        raise ValueError(f"nt and dt must be positive, got nt={nt}, dt={dt}")
    phi = float(gen.gamma(shape=nt / 2.0, scale=2.0 / dt))
    cov = c / phi
    return gen.multivariate_normal(m, cov)


# ---------------------------------------------------------------------------
# DAG / alignment helpers
# ---------------------------------------------------------------------------


def _global_dag_edges(adj: np.ndarray) -> List[Tuple[int, int]]:
    """Directed edges (parent, child) in column-major child order."""
    a = np.asarray(adj, dtype=int)
    nn = a.shape[0]
    out: List[Tuple[int, int]] = []
    for c in range(nn):
        parents = np.where(a[:, c] != 0)[0]
        for p in sorted(int(x) for x in parents if int(x) != c):
            out.append((p, c))
    return out


def _align_child_local_to_global(
    local_theta: np.ndarray,
    local_parent_list: Sequence[int],
    global_parents: Sequence[int],
) -> np.ndarray:
    """Map a child's local regression coefficients to global parent ordering."""
    lt = np.asarray(local_theta, dtype=float).reshape(-1)
    lp = [int(x) for x in local_parent_list]
    gp = [int(x) for x in global_parents]
    if lt.size != 1 + len(lp):
        raise ValueError(
            f"local_theta length {lt.size} != 1 + len(local_parent_list) ({1 + len(lp)})"
        )
    pos = {par: k for k, par in enumerate(lp)}
    aligned = np.empty(len(gp), dtype=float)
    for j, par in enumerate(gp):
        k = pos.get(par)
        if k is None:
            aligned[j] = np.nan
        else:
            aligned[j] = lt[1 + k]
    return aligned


def _infer_filter_time_length(filtered: Sequence[Mapping[str, Any]]) -> int:
    """Infer filter time length ``T`` from the first non-scalar ``mt`` entry."""
    for filt in filtered:
        mt0 = filt["mt"][0]
        if hasattr(mt0, "shape") and mt0.ndim >= 1:
            return int(mt0.shape[-1])
    raise ValueError("Could not infer time length from filtered outputs")


# ---------------------------------------------------------------------------
# Setup for MC sampling
# ---------------------------------------------------------------------------


def _build_parent_lists(
    subject_adjacency_matrices: List[np.ndarray],
    n_subjects: int,
    n_nodes: int,
    n_times: int,
) -> List[List[List[int]]]:
    """Per-subject parent lists derived from each subject's design DAG."""
    dummy_data = np.zeros((n_times, n_nodes), dtype=float)
    parent_lists: List[List[List[int]]] = []
    for si in range(n_subjects):
        pl_si: List[List[int]] = []
        for c in range(n_nodes):
            _, pl = build_design_matrix(dummy_data, subject_adjacency_matrices[si], c)
            pl_si.append(list(pl))
        parent_lists.append(pl_si)
    return parent_lists


def _population_mean_over_subjects(vals: List[float], n_subjects: int) -> float:
    """(1/S) sum_i theta_i for one MC replicate; ``nan`` if any subject is missing."""
    if len(vals) != n_subjects or not all(np.isfinite(v) for v in vals):
        return np.nan
    return float(sum(vals)) / float(n_subjects)


def _sample_subject_states_at_time(
    filtered: Sequence[Mapping[str, Any]],
    nn: int,
    t_index: int,
    gen: np.random.Generator,
    *,
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[List[np.ndarray]]:
    """One MC replicate: sample all subjects, all child nodes at ``t_index``."""
    subject_samples: List[List[np.ndarray]] = []
    for si, filt in enumerate(filtered):
        row: List[np.ndarray] = []
        for c in range(nn):
            nt_c = filt["nt"][c]
            dt_c = filt["dt"][c]
            nt_t = float(nt_c[t_index])
            dt_t = float(dt_c[t_index])
            if smoothed_per_subject is None:
                mt_c = filt["mt"][c]
                ct_c = filt["Ct"][c]
            else:
                smo = smoothed_per_subject[si]
                mt_c = smo["smt"][c]
                ct_c = smo["sCt"][c]
            m_col = mt_c[:, t_index]
            c_slice = ct_c[:, :, t_index]
            row.append(_sample_dlm_state_posterior(m_col, c_slice, nt_t, dt_t, gen))
        subject_samples.append(row)
    return subject_samples


# ---------------------------------------------------------------------------
# Per-time sample + population mean
# ---------------------------------------------------------------------------


def _monte_carlo_beta_samples_at_time(
    filtered: Sequence[Mapping[str, Any]],
    edges: List[Tuple[int, int]],
    parent_lists: List[List[List[int]]],
    ss: int,
    nn: int,
    t_index: int,
    n_mc: int,
    gen: np.random.Generator,
    *,
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> np.ndarray:
    """
    One timestep: for each replicate ``b``, sample per subject then
    ``bar_theta_t^(b) = (1/S) sum_i theta_it^(b)`` per edge.
    """
    e_ct = len(edges)
    beta_samples = np.empty((n_mc, e_ct), dtype=float)

    for b in range(n_mc):
        subject_samples = _sample_subject_states_at_time(
            filtered,
            nn,
            t_index,
            gen,
            smoothed_per_subject=smoothed_per_subject,
        )
        for e, (p, cc) in enumerate(edges):
            vals: List[float] = []
            for si in range(ss):
                theta = subject_samples[si][cc]
                aligned = _align_child_local_to_global(
                    theta,
                    parent_lists[si][cc],
                    [p],
                )
                v = aligned[0]
                if not np.isfinite(v):
                    vals = []
                    break
                vals.append(float(v))
            beta_samples[b, e] = _population_mean_over_subjects(vals, ss)

    return beta_samples


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _build_global_beta_metadata(
    consensus_view: ISAggregatedMDMView,
    n_subjects: int,
    mc_posterior: MCPosteriorSource,
) -> Dict[str, Any]:
    """Metadata dict for :class:`GlobalBetaMCResult`."""
    return {
        "edges_removed_for_acyclicity": consensus_view.metadata.get(
            "edges_removed_for_acyclicity", []
        ),
        "n_subjects": n_subjects,
        "mc_posterior": mc_posterior,
        "conditioning": "fixed_consensus_dag",
        "pooling_semantics": (
            "population_mean: bar_theta_t^(b) = (1/S) sum_i theta_it^(b); "
            "Monte Carlo over B replicates propagates uncertainty through the mean"
        ),
    }


def _empty_global_beta_result(
    n_mc: int,
    t_len: int,
    base_meta: Dict[str, Any],
) -> GlobalBetaMCResult:
    """Return a zero-edge :class:`GlobalBetaMCResult` with consistent metadata."""
    t_list = list(range(t_len))
    beta_empty = np.empty((n_mc, 0, t_len), dtype=float)
    return GlobalBetaMCResult(
        beta_samples=beta_empty,
        edges=[],
        n_contributors=np.zeros(0, dtype=int),
        time_index=0,
        pooling=POOLING_POPULATION_MEAN,
        metadata=base_meta,
        time_indices_mc=tuple(t_list),
        beta_mean=np.empty((0, t_len), dtype=float),
        beta_var=np.empty((0, t_len), dtype=float),
    )


def _summarize_beta_samples(
    beta_samples: np.ndarray,
    mc_quantiles: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Tuple[float, ...]]]:
    """Mean, variance, and optional quantiles along the MC axis."""
    beta_mean = np.nanmean(beta_samples, axis=0)
    beta_var = np.nanvar(beta_samples, axis=0)
    beta_q: Optional[np.ndarray] = None
    q_tuple: Optional[Tuple[float, ...]] = None
    if mc_quantiles is not None:
        q_list = [float(x) for x in mc_quantiles]
        if q_list:
            q_tuple = tuple(q_list)
            beta_q = np.nanquantile(beta_samples, np.asarray(q_list, dtype=float), axis=0)
    return beta_mean, beta_var, beta_q, q_tuple


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _monte_carlo_global_edge_beta(
    posterior_per_subject: Sequence[Mapping[str, Any]],
    subject_adjacency_matrices: List[np.ndarray],
    consensus_view: ISAggregatedMDMView,
    mc_n_samples: int,
    rng: np.random.Generator,
    *,
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_posterior: MCPosteriorSource = "filtered",
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> GlobalBetaMCResult:
    """
    Monte Carlo global edge coefficients conditional on the consensus DAG G*.

    For each filter time ``t ∈ {0, …, T-1}`` and replicate ``b = 1, …, B``,
    draw ``θ_{it}^{(b)}`` from each subject's marginal posterior, form the
    population mean ``θ̄_t^{(b)} = (1/S) Σ_i θ_{it}^{(b)}`` on each global edge,
    and summarize the empirical distribution over ``b``.

    Parameters
    ----------
    posterior_per_subject
        Length-``S`` sequence of per-subject filtered DLM outputs (dicts with
        ``mt``, ``Ct``, ``nt``, ``dt`` keyed by child node index).  After refit
        on G*, these are the posteriors used for sampling.
    subject_adjacency_matrices
        Length-``S`` list of ``N×N`` binary adjacency matrices—one per
        subject—defining local parent sets when mapping regression coefficients
        onto global edges.  With refit on the consensus DAG, entries are
        typically identical copies of ``consensus_view.adj_mat``.
    consensus_view
        :class:`~mdmp.group_analysis.inds.results.ISAggregatedMDMView` whose
        ``adj_mat`` is the fixed global DAG G*.  Directed edges in this matrix
        are the Monte Carlo targets.
    mc_n_samples
        Number of Monte Carlo replicates ``B`` (first axis of ``beta_samples``).
    rng
        NumPy random generator used for Student-t posterior draws.
    mc_quantiles
        Optional quantile levels in ``(0, 1)``; when set, ``beta_quantiles`` is
        computed along the replicate axis.
    mc_posterior
        ``"filtered"`` samples from filtered moments ``(mt, Ct)``;
        ``"smoothed"`` uses ``(smt, sCt)`` with ``nt``/``dt`` from the filter
        at the same time index.
    smoothed_per_subject
        Required when ``mc_posterior="smoothed"``: length-``S`` smoothed-output
        dicts (``smt``, ``sCt`` per child) aligned with ``posterior_per_subject``.

    Returns
    -------
    GlobalBetaMCResult
        ``beta_samples`` with shape ``(B, n_edges, T)``, plus ``beta_mean``,
        ``beta_var``, and optional ``beta_quantiles``.
    """
    if mc_n_samples < 1:
        raise ValueError("mc_n_samples must be at least 1")
    n_subjects = len(posterior_per_subject)
    if n_subjects == 0:
        raise ValueError("posterior_per_subject must be non-empty")
    if len(subject_adjacency_matrices) != n_subjects:
        raise ValueError(
            f"subject_adjacency_matrices length {len(subject_adjacency_matrices)} "
            f"!= posterior_per_subject length {n_subjects}"
        )

    n_times = _infer_filter_time_length(posterior_per_subject)
    time_indices = list(range(n_times))

    global_adj = np.asarray(consensus_view.adj_mat, dtype=int)
    n_nodes = global_adj.shape[0]
    edges = _global_dag_edges(global_adj)

    base_meta = _build_global_beta_metadata(consensus_view, n_subjects, mc_posterior)

    if len(edges) == 0:
        return _empty_global_beta_result(mc_n_samples, n_times, base_meta)

    parent_lists = _build_parent_lists(
        subject_adjacency_matrices, n_subjects, n_nodes, n_times
    )

    if mc_posterior == "smoothed" and smoothed_per_subject is None:
        raise ValueError("smoothed_per_subject is required when mc_posterior='smoothed'")

    blocks = [
        _monte_carlo_beta_samples_at_time(
            posterior_per_subject,
            edges,
            parent_lists,
            n_subjects,
            n_nodes,
            tix,
            mc_n_samples,
            rng,
            smoothed_per_subject=smoothed_per_subject,
        )
        for tix in time_indices
    ]
    beta_samples = np.stack(blocks, axis=2)

    beta_mean, beta_var, beta_q, q_tuple = _summarize_beta_samples(beta_samples, mc_quantiles)

    return GlobalBetaMCResult(
        beta_samples=beta_samples,
        edges=edges,
        n_contributors=np.full(len(edges), n_subjects, dtype=int),
        time_index=0,
        pooling=POOLING_POPULATION_MEAN,
        metadata=base_meta,
        time_indices_mc=tuple(time_indices),
        beta_quantiles=beta_q,
        quantile_levels=q_tuple,
        beta_mean=beta_mean,
        beta_var=beta_var,
    )


# ---------------------------------------------------------------------------
# Smoothing helper (imported by refit.build_mc_inputs)
# ---------------------------------------------------------------------------


def _smooth_filtered_sequence(
    filtered: Sequence[Mapping[str, Any]],
    n_jobs: Optional[int],
) -> List[Dict[str, Any]]:
    """Run ``smooth_all_nodes`` on each subject's filtered output."""
    return [
        smooth_all_nodes(
            mt=f["mt"],
            Ct=f["Ct"],
            Rt=f["Rt"],
            nt=f["nt"],
            dt=f["dt"],
            n_jobs=n_jobs,
        )
        for f in filtered
    ]
