"""Monte Carlo pooling of DLM regression states onto consensus DAG edges.

Statistical Interpretation
--------------------------
**What this module computes.**
For each global edge (p → c) in the consensus DAG G*, samples from per-subject
DLM state posteriors are aligned to that edge and pooled across contributing
subjects.  The resulting empirical distribution over pooled samples summarises:

    E[θ_{pc,t} | edge_{pc} = 1]  (with pooling='conditional_mean_among_edge_subjects')

This is the conditional posterior mean among subjects that expressed the edge,
**not** an unconditional population-average effect.

**What this module does NOT compute.**

* No hierarchical population model is fitted; subjects are sampled
  **independently** — no shrinkage, no between-subject covariance.
* Credible intervals from ``beta_samples`` are *not* hierarchical credible
  intervals from a joint population model.
* Structural uncertainty is not propagated; inference is p(θ | G*), not the
  full Bayesian model average p(θ) = Σ_G p(θ|G) p(G).

**Pooling denominator.**
With ``pooling='conditional_mean_among_edge_subjects'`` (alias:
``'mean_with_edge'``), the code averages only contributing subjects A at that
edge.  For ``mc_contributors='individual_edge'``, contributors are subjects
whose individual DAG had the edge, so the divisor is A ≤ S, not a fixed 1/S
over all S subjects (missing edges are excluded, not averaged in as zeros).
With ``mc_contributors='all_subjects'`` after a global refit, typically A = S.

**Smoothed samples.**
When ``mc_posterior='smoothed'``, samples use smoothed ``(m, C)`` with the
filter's ``(n_t, d_t)`` at the same time in the Gamma–Normal step — a
pragmatic reuse of the filtered sampling machinery, not a claim of exact
posterior sampling from the joint smoothing distribution.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..._node_dispatch import smooth_all_nodes
from ...utils import build_design_matrix
from .results import (
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    MCContributorMode,
    MCPosteriorSource,
    PoolingMode,
)

# Pooling modes that implement conditional_mean_among_edge_subjects semantics
# (old name 'mean_with_edge' is a backward-compatible alias).
_CONDITIONAL_MEAN_MODES = frozenset({"conditional_mean_among_edge_subjects", "mean_with_edge"})
_CONDITIONAL_SUM_MODES = frozenset({"conditional_sum_among_edge_subjects", "sum_with_edge"})


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
    Sample one DLM state vector from a Gamma–Normal posterior.

    Parameters
    ----------
    mt, Ct
        Posterior mean and covariance at one time index.
    nt, dt
        Filter precision hyperparameters (must be positive).
    gen
        NumPy random generator.

    Returns
    -------
    ndarray
        Sampled regression state vector.
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
    """
    Map a child's local regression coefficients to global parent ordering.

    Parameters
    ----------
    local_theta
        Local state vector (intercept + parent effects).
    local_parent_list
        Parent node indices in the subject's design matrix.
    global_parents
        Target parent ordering on the consensus DAG.

    Returns
    -------
    ndarray
        Aligned parent effects; ``nan`` where a global parent is absent locally.
    """
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
    """
    Infer filter time length ``T`` from the first non-scalar ``mt`` entry.

    Returns
    -------
    int
        Number of time steps in filtered outputs.

    Raises
    ------
    ValueError
        If no suitable ``mt`` array is found.
    """
    for filt in filtered:
        mt0 = filt["mt"][0]
        if hasattr(mt0, "shape") and mt0.ndim >= 1:
            return int(mt0.shape[-1])
    raise ValueError("Could not infer time length from filtered outputs")


# ---------------------------------------------------------------------------
# Setup for MC sampling
# ---------------------------------------------------------------------------


def _build_parent_lists(
    design_adjs: List[np.ndarray],
    ss: int,
    nn: int,
    t_len: int,
) -> List[List[List[int]]]:
    """
    Per-subject parent lists for each child node (for coefficient alignment).

    Uses a dummy data matrix; only the adjacency structure affects parent lists.
    """
    dummy_data = np.zeros((t_len, nn), dtype=float)
    parent_lists: List[List[List[int]]] = []
    for si in range(ss):
        pl_si: List[List[int]] = []
        for c in range(nn):
            _, pl = build_design_matrix(dummy_data, design_adjs[si], c)
            pl_si.append(list(pl))
        parent_lists.append(pl_si)
    return parent_lists


def _count_contributors_per_edge(
    edges: List[Tuple[int, int]],
    adjs_individual: List[np.ndarray],
    ss: int,
    mc_contributors: MCContributorMode,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Count contributing subjects A per consensus edge.

    For ``mc_contributors='individual_edge'``, A is the number of subjects whose
    individual DAG contains the edge; for ``'all_subjects'``, A = S.
    """
    e_ct = len(edges)
    n_contrib = np.zeros(e_ct, dtype=int)
    contributors_per_edge: Dict[Tuple[int, int], int] = {}
    for e, (p, cc) in enumerate(edges):
        if mc_contributors == "all_subjects":
            n_contrib[e] = ss
        else:
            n_contrib[e] = int(sum(1 for a in adjs_individual if a[p, cc] != 0))
        contributors_per_edge[(p, cc)] = int(n_contrib[e])
    return n_contrib, contributors_per_edge


def _pooling_semantics_label(pool: PoolingMode) -> str:
    """Human-readable description of what the pooling mode computes."""
    if pool in _CONDITIONAL_MEAN_MODES:
        return (
            "conditional_mean_E[theta|edge=1]: mean over subjects expressing the "
            "edge (divisor = n_contributors, not total S subjects); subjects "
            "without the edge are excluded from both numerator and divisor"
        )
    if pool in _CONDITIONAL_SUM_MODES:
        return (
            "conditional_sum: sum over subjects expressing the edge; subjects "
            "without the edge are excluded"
        )
    return f"unknown pooling: {pool!r}"


def _pool_contributor_values(vals: List[float], pool: PoolingMode) -> float:
    """
    Pool finite contributor samples for one edge and one MC replicate.

    Returns
    -------
    float
        Pooled value, or ``nan`` when ``vals`` is empty.
    """
    if not vals:
        return np.nan
    if pool in _CONDITIONAL_MEAN_MODES:
        return float(sum(vals)) / float(len(vals))
    if pool in _CONDITIONAL_SUM_MODES:
        return float(sum(vals))
    raise ValueError(f"unknown pooling mode: {pool!r}")


def _sample_subject_states_at_time(
    filtered: Sequence[Mapping[str, Any]],
    nn: int,
    t_index: int,
    gen: np.random.Generator,
    *,
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[List[np.ndarray]]:
    """
    Sample one MC replicate: all subjects, all child nodes at ``t_index``.

    See the module docstring for the smoothed-sample approximation caveat.
    """
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
# Per-time sample + pool
# ---------------------------------------------------------------------------


def _monte_carlo_beta_samples_at_time(
    filtered: Sequence[Mapping[str, Any]],
    adjs_edge_mask: List[np.ndarray],
    edges: List[Tuple[int, int]],
    parent_lists: List[List[List[int]]],
    ss: int,
    nn: int,
    t_index: int,
    n_mc: int,
    gen: np.random.Generator,
    pool: PoolingMode,
    *,
    mc_contributors: MCContributorMode = "individual_edge",
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> np.ndarray:
    """
    One timestep: for each MC replicate, sample per subject then pool per edge.

    See the module docstring for conditional-mean pooling semantics and the
    smoothed-sample approximation caveat.
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
                if mc_contributors == "individual_edge":
                    if adjs_edge_mask[si][p, cc] == 0:
                        continue
                theta = subject_samples[si][cc]
                aligned = _align_child_local_to_global(
                    theta,
                    parent_lists[si][cc],
                    [p],
                )
                v = aligned[0]
                if np.isfinite(v):
                    vals.append(float(v))
            beta_samples[b, e] = _pool_contributor_values(vals, pool)

    return beta_samples


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _build_global_beta_metadata(
    is_res: ISAggregatedMDMView,
    ss: int,
    mc_posterior: MCPosteriorSource,
    mc_contributors: MCContributorMode,
    pool: PoolingMode,
    contributors_per_edge: Dict[Tuple[int, int], int],
) -> Dict[str, Any]:
    """Metadata dict for :class:`GlobalBetaMCResult`."""
    return {
        "edges_removed_for_acyclicity": is_res.metadata.get("edges_removed_for_acyclicity", []),
        "n_subjects": ss,
        "mc_posterior": mc_posterior,
        "mc_contributors": mc_contributors,
        "conditioning": "fixed_consensus_dag",
        "pooling_semantics": _pooling_semantics_label(pool),
        "contributors_per_edge": contributors_per_edge,
    }


def _empty_global_beta_result(
    n_mc: int,
    t_len: int,
    pool: PoolingMode,
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
        pooling=pool,
        metadata=base_meta,
        time_indices_mc=tuple(t_list),
        beta_mean=np.empty((0, t_len), dtype=float),
        beta_var=np.empty((0, t_len), dtype=float),
    )


def _summarize_beta_samples(
    beta_samples: np.ndarray,
    mc_quantiles: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Tuple[float, ...]]]:
    """
    Compute mean, variance, and optional quantiles along the MC axis.

    Returns
    -------
    beta_mean, beta_var, beta_quantiles, quantile_levels
    """
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
    filtered: Sequence[Mapping[str, Any]],
    adjs_individual: List[np.ndarray],
    design_adjs: List[np.ndarray],
    is_res: ISAggregatedMDMView,
    n_mc: int,
    gen: np.random.Generator,
    pool: PoolingMode,
    *,
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_contributors: MCContributorMode = "individual_edge",
    mc_posterior: MCPosteriorSource = "filtered",
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> GlobalBetaMCResult:
    """
    Monte Carlo over DLM posteriors + conditional pooling on the **consensus** DAG.

    Samples are computed at **every** filter time index ``t = 0, …, T-1`` inferred
    from ``filtered``.  ``beta_samples`` has shape ``(B, n_edges, T)``.
    """
    if n_mc < 1:
        raise ValueError("mc_n_samples must be at least 1")
    ss = len(filtered)
    if ss == 0:
        raise ValueError("filtered_per_subject must be non-empty")
    if len(adjs_individual) != ss:
        raise ValueError(
            f"adjacency list length {len(adjs_individual)} != filtered_per_subject length {ss}"
        )
    if len(design_adjs) != ss:
        raise ValueError(
            f"design_adjs length {len(design_adjs)} != filtered_per_subject length {ss}"
        )

    t_len = _infer_filter_time_length(filtered)
    t_list = list(range(t_len))

    global_adj = np.asarray(is_res.adj_mat, dtype=int)
    nn = global_adj.shape[0]
    edges = _global_dag_edges(global_adj)

    n_contrib, contributors_per_edge = _count_contributors_per_edge(
        edges, adjs_individual, ss, mc_contributors
    )
    base_meta = _build_global_beta_metadata(
        is_res, ss, mc_posterior, mc_contributors, pool, contributors_per_edge
    )

    if len(edges) == 0:
        return _empty_global_beta_result(n_mc, t_len, pool, base_meta)

    parent_lists = _build_parent_lists(design_adjs, ss, nn, t_len)

    if mc_posterior == "smoothed" and smoothed_per_subject is None:
        raise ValueError("smoothed_per_subject is required when mc_posterior='smoothed'")

    blocks = [
        _monte_carlo_beta_samples_at_time(
            filtered,
            adjs_individual,
            edges,
            parent_lists,
            ss,
            nn,
            tix,
            n_mc,
            gen,
            pool,
            mc_contributors=mc_contributors,
            smoothed_per_subject=smoothed_per_subject,
        )
        for tix in t_list
    ]
    beta_samples = np.stack(blocks, axis=2)

    beta_mean, beta_var, beta_q, q_tuple = _summarize_beta_samples(beta_samples, mc_quantiles)

    return GlobalBetaMCResult(
        beta_samples=beta_samples,
        edges=edges,
        n_contributors=n_contrib,
        time_index=0,
        pooling=pool,
        metadata=base_meta,
        time_indices_mc=tuple(t_list),
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
