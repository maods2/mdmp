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

from ..._node_dispatch import _parallel_map, smooth_all_nodes
from ...utils import build_design_matrix
from .results import GlobalBetaMCResult, ISAggregatedMDMView, MCPosteriorSource

POOLING_POPULATION_MEAN = "population_mean"


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


def _sample_dlm_state_posterior_batch(
    n_mc: int,
    mt: np.ndarray,
    Ct: np.ndarray,
    nt: float,
    dt: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """
    Sample ``n_mc`` DLM state vectors from the marginal Student-t posterior.

    Implements :math:`\\theta^{(b)} \\sim t_{\\nu}(m, C)` via independent
    :math:`\\phi_b \\sim \\mathrm{Gamma}(n_t/2, 2/d_t)` and
    :math:`\\theta^{(b)} \\mid \\phi_b \\sim \\mathcal{N}(m, C/\\phi_b)`.

    Returns
    -------
    np.ndarray
        Posterior draws, shape ``(n_mc, p)``.
    """
    m = np.asarray(mt, dtype=float).reshape(-1)
    p = m.shape[0]
    c = np.asarray(Ct, dtype=float).reshape(p, p)
    if nt <= 0.0 or dt <= 0.0:
        raise ValueError(f"nt and dt must be positive, got nt={nt}, dt={dt}")
    phi = gen.gamma(shape=nt / 2.0, scale=2.0 / dt, size=n_mc)
    chol = np.linalg.cholesky(c)
    z = gen.standard_normal(size=(n_mc, p))
    return m + (z @ chol.T) / np.sqrt(phi)[:, np.newaxis]


def _sample_dlm_state_posterior(
    mt: np.ndarray,
    Ct: np.ndarray,
    nt: float,
    dt: float,
    gen: np.random.Generator,
) -> np.ndarray:
    """One draw from the Student-t posterior; shape ``(p,)``."""
    return _sample_dlm_state_posterior_batch(1, mt, Ct, nt, dt, gen)[0]


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


def _build_edge_coef_index(
    edges: List[Tuple[int, int]],
    parent_lists: List[List[List[int]]],
    n_subjects: int,
) -> Tuple[List[int], List[List[Optional[int]]]]:
    """
    Children that appear on global edges and per-edge local coefficient indices.

    For edge ``(p, cc)`` and subject ``si``, index ``1 + k`` into the sampled
    state at child ``cc``, or ``None`` if parent ``p`` is absent locally.
    """
    children_needed = sorted({cc for _, cc in edges})
    edge_coef_index: List[List[Optional[int]]] = []
    for p, cc in edges:
        row: List[Optional[int]] = []
        for si in range(n_subjects):
            lp = parent_lists[si][cc]
            pos = {int(par): k for k, par in enumerate(lp)}
            k = pos.get(int(p))
            row.append(None if k is None else 1 + k)
        edge_coef_index.append(row)
    return children_needed, edge_coef_index


def _population_mean_nanrule_batch(vals: np.ndarray, n_subjects: int) -> np.ndarray:
    """
    Population mean along subjects for each MC replicate.

    ``vals`` has shape ``(B, S)``.  Row ``b`` is ``nan`` unless all ``S`` entries
    are finite (same rule as the former per-replicate list aggregation).
    """
    if vals.shape[1] != n_subjects:
        raise ValueError(f"vals.shape[1] {vals.shape[1]} != n_subjects {n_subjects}")
    ok = np.all(np.isfinite(vals), axis=1)
    out = np.full(vals.shape[0], np.nan, dtype=float)
    if np.any(ok):
        out[ok] = np.mean(vals[ok], axis=1)
    return out


def _mc_root_seed_sequence(rng: np.random.Generator) -> np.random.SeedSequence:
    """Root entropy for per-time RNG spawns (one draw from ``rng``)."""
    return np.random.SeedSequence(int(rng.integers(0, 2**63, dtype=np.uint64)))


def _moments_at_time(
    filt: Mapping[str, Any],
    child: int,
    t_index: int,
    smoothed: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Posterior moments for one child at one filter time index."""
    nt_c = filt["nt"][child]
    dt_c = filt["dt"][child]
    nt_t = float(nt_c[t_index])
    dt_t = float(dt_c[t_index])
    if smoothed is None:
        mt_c = filt["mt"][child]
        ct_c = filt["Ct"][child]
    else:
        mt_c = smoothed["smt"][child]
        ct_c = smoothed["sCt"][child]
    m_col = np.asarray(mt_c[:, t_index], dtype=float)
    c_slice = np.asarray(ct_c[:, :, t_index], dtype=float)
    return m_col, c_slice, nt_t, dt_t


# ---------------------------------------------------------------------------
# Per-time sample + population mean
# ---------------------------------------------------------------------------


def _monte_carlo_beta_samples_at_time(
    posterior_per_subject: Sequence[Mapping[str, Any]],
    edges: List[Tuple[int, int]],
    edge_coef_index: List[List[Optional[int]]],
    children_needed: List[int],
    n_subjects: int,
    time_index: int,
    mc_n_samples: int,
    rng: np.random.Generator,
    *,
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> np.ndarray:
    """
    Monte Carlo pooled edge coefficients at a single filter time index.

    For each replicate ``b = 1, …, B``, draw regression states from each
    subject's marginal posterior on the consensus DAG, then form the population
    mean :math:`\\bar\\theta_t^{(b)} = \\frac{1}{S}\\sum_i \\theta_{it}^{(b)}`
    on every global edge.  Sampling is vectorized over ``B``.

    Parameters
    ----------
    posterior_per_subject
        Length-``S`` filtered (or refit) DLM outputs with ``mt``, ``Ct``,
        ``nt``, ``dt`` per child node.
    edges
        Global consensus edges ``(parent, child)`` in column-major order.
    edge_coef_index
        ``edge_coef_index[e][i]`` is the index into subject ``i``'s sampled
        state vector at child ``edges[e][1]`` for parent ``edges[e][0]``, or
        ``None`` if that parent is absent in subject ``i``'s local DAG.
    children_needed
        Sorted child node indices that appear as endpoints of ``edges``.
    n_subjects
        Number of subjects ``S``.
    time_index
        Filter time index ``t`` (column into ``mt`` / ``Ct``).
    mc_n_samples
        Number of Monte Carlo replicates ``B``.
    rng
        NumPy generator for Student-t posterior draws at this time index.
    smoothed_per_subject
        When set, use ``smt`` / ``sCt`` from these dicts with ``nt`` / ``dt``
        from ``posterior_per_subject`` (``mc_posterior='smoothed'``).

    Returns
    -------
    np.ndarray
        ``beta_samples`` slice for this time, shape ``(mc_n_samples, n_edges)``.
    """
    n_edges = len(edges)
    batch: Dict[Tuple[int, int], np.ndarray] = {}

    for subject_idx, filt in enumerate(posterior_per_subject):
        smo = (
            None
            if smoothed_per_subject is None
            else smoothed_per_subject[subject_idx]
        )
        for child in children_needed:
            m_col, c_slice, nt_t, dt_t = _moments_at_time(
                filt, child, time_index, smo
            )
            batch[(subject_idx, child)] = _sample_dlm_state_posterior_batch(
                mc_n_samples, m_col, c_slice, nt_t, dt_t, rng
            )

    beta_samples = np.empty((mc_n_samples, n_edges), dtype=float)
    for edge_idx, (_, child) in enumerate(edges):
        vals = np.full((mc_n_samples, n_subjects), np.nan, dtype=float)
        for subject_idx in range(n_subjects):
            coef_idx = edge_coef_index[edge_idx][subject_idx]
            if coef_idx is not None:
                vals[:, subject_idx] = batch[(subject_idx, child)][:, coef_idx]
        beta_samples[:, edge_idx] = _population_mean_nanrule_batch(vals, n_subjects)

    return beta_samples


def _worker_mc_beta_at_time(args: Tuple[Any, ...]) -> np.ndarray:
    """Picklable worker: one filter time index (see ``_monte_carlo_beta_samples_at_time``)."""
    (
        time_index,
        posterior_per_subject,
        edges,
        edge_coef_index,
        children_needed,
        n_subjects,
        mc_n_samples,
        time_entropy,
        smoothed_per_subject,
    ) = args
    rng = np.random.default_rng(np.random.SeedSequence(time_entropy))
    return _monte_carlo_beta_samples_at_time(
        posterior_per_subject,
        edges,
        edge_coef_index,
        children_needed,
        n_subjects,
        time_index,
        mc_n_samples,
        rng,
        smoothed_per_subject=smoothed_per_subject,
    )


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
    mc_n_jobs: Optional[int] = None,
) -> GlobalBetaMCResult:
    """
    Monte Carlo global edge coefficients conditional on the consensus DAG G*.

    For each filter time ``t ∈ {0, …, T-1}`` and replicate ``b = 1, …, B``,
    draw ``θ_{it}^{(b)}`` from each subject's marginal posterior, form the
    population mean ``θ̄_t^{(b)} = (1/S) Σ_i θ_{it}^{(b)}`` on each global edge,
    and summarize the empirical distribution over ``b``.
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
    children_needed, edge_coef_index = _build_edge_coef_index(
        edges, parent_lists, n_subjects
    )

    if mc_posterior == "smoothed" and smoothed_per_subject is None:
        raise ValueError("smoothed_per_subject is required when mc_posterior='smoothed'")

    # One child RNG per filter time: derived from ``rng`` via SeedSequence.spawn so
    # serial (mc_n_jobs=1) and parallel (mc_n_jobs>1) runs match, and times do not
    # share random draws when workers run out of order.
    root_ss = _mc_root_seed_sequence(rng)
    time_seed_seqs = root_ss.spawn(n_times)

    worker_args = [
        (
            time_index,
            posterior_per_subject,
            edges,
            edge_coef_index,
            children_needed,
            n_subjects,
            mc_n_samples,
            time_seed_seqs[time_index].entropy,
            smoothed_per_subject,
        )
        for time_index in time_indices
    ]

    blocks = _parallel_map(
        _worker_mc_beta_at_time,
        worker_args,
        mc_n_jobs,
        "Monte Carlo time steps",
    )
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
