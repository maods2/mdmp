"""Monte Carlo pooling of DLM regression states onto global DAG edges.

Each replicate draws **independently** across subjects from that subject's
filter (or smoother) state; pooled edge summaries are the pushforward of those
**independent** posteriors through the align-and-pool map. Credible intervals
from ``beta_draws`` are **not** from a fully joint hierarchical population
model over subjects.

**Pooling denominator.** With ``pooling='mean_with_edge'``, the code averages
only **contributing** subjects :math:`A` at that edge. For
``mc_contributors='individual_edge'``, contributors are subjects whose
individual DAG had the edge, so the divisor is :math:`A\\le S`, not a fixed
:math:`1/S` over all :math:`S` subjects (missing edges are excluded rather than
averaged in as zeros). With ``mc_contributors='all_subjects'`` after a global
refit, typically :math:`A=S``.

**Smoothed draws.** When ``mc_posterior='smoothed'``, samples use smoothed
``(m,C)`` with the filter's ``(n_t,d_t)`` at the same time in the Gamma–Normal
step—a pragmatic reuse of the filtered sampling machinery, not a claim of exact
posterior sampling from the joint smoothing distribution.
"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...model.smoothing_pipeline import SmoothingPipeline
from ...utils import build_design_matrix
from .results import (
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    MCContributorMode,
    MCPosteriorSource,
)


def _sample_dlm_state_posterior(
    mt: np.ndarray,
    Ct: np.ndarray,
    nt: float,
    dt: float,
    gen: np.random.Generator,
) -> np.ndarray:
    m = np.asarray(mt, dtype=float).reshape(-1)
    p = m.shape[0]
    c = np.asarray(Ct, dtype=float).reshape(p, p)
    if nt <= 0.0 or dt <= 0.0:
        raise ValueError(f"nt and dt must be positive, got nt={nt}, dt={dt}")
    phi = float(gen.gamma(shape=nt / 2.0, scale=2.0 / dt))
    cov = c / phi
    return gen.multivariate_normal(m, cov)


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
    for filt in filtered:
        mt0 = filt["mt"][0]
        if hasattr(mt0, "shape") and mt0.ndim >= 1:
            return int(mt0.shape[-1])
    raise ValueError("Could not infer time length from filtered outputs")


def _monte_carlo_beta_draws_at_time(
    filtered: Sequence[Mapping[str, Any]],
    adjs_edge_mask: List[np.ndarray],
    edges: List[Tuple[int, int]],
    parent_lists: List[List[List[int]]],
    ss: int,
    nn: int,
    t_index: int,
    n_mc: int,
    gen: np.random.Generator,
    pool: Literal["mean_with_edge", "sum_with_edge"],
    *,
    mc_contributors: MCContributorMode = "individual_edge",
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> np.ndarray:
    """
    One timestep: for each MC replicate b, sample per subject then pool per edge.

    ``mean_with_edge`` implements
    :math:`\\bar{\\theta}^{(b)} = \\frac{1}{A}\\sum_{i\\in\\mathcal{A}} \\theta_i^{(b)}`
    with :math:`\\mathcal{A}` the contributor set at that edge (for
    ``individual_edge``, subjects with the edge on their DAG), not necessarily
    all :math:`S` subjects.

    Hot path: a batched draw layout (vectorized over ``b``) could reduce Python
    loop overhead here without changing the pooling definition.
    """
    e_ct = len(edges)
    beta_draws = np.empty((n_mc, e_ct), dtype=float)

    for b in range(n_mc):
        draws: List[List[np.ndarray]] = []
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
                    m_col = mt_c[:, t_index]
                    c_slice = ct_c[:, :, t_index]
                else:
                    smo = smoothed_per_subject[si]
                    mt_c = smo["smt"][c]
                    ct_c = smo["sCt"][c]
                    m_col = mt_c[:, t_index]
                    c_slice = ct_c[:, :, t_index]
                row.append(
                    _sample_dlm_state_posterior(m_col, c_slice, nt_t, dt_t, gen)
                )
            draws.append(row)

        for e, (p, cc) in enumerate(edges):
            vals: List[float] = []
            for si in range(ss):
                if mc_contributors == "individual_edge":
                    if adjs_edge_mask[si][p, cc] == 0:
                        continue
                theta = draws[si][cc]
                aligned = _align_child_local_to_global(
                    theta,
                    parent_lists[si][cc],
                    [p],
                )
                v = aligned[0]
                if np.isfinite(v):
                    vals.append(float(v))
            if not vals:
                beta_draws[b, e] = np.nan
            elif pool == "mean_with_edge":
                # Explicit group mean: (1/A) * sum_i theta_i^(b), A = len(vals)
                a = len(vals)
                beta_draws[b, e] = float(sum(vals)) / float(a)
            elif pool == "sum_with_edge":
                beta_draws[b, e] = float(sum(vals))
            else:
                raise ValueError(f"unknown pooling: {pool}")

    return beta_draws


def _monte_carlo_global_edge_beta(
    filtered: Sequence[Mapping[str, Any]],
    adjs_individual: List[np.ndarray],
    design_adjs: List[np.ndarray],
    is_res: ISAggregatedMDMView,
    time_index: int,
    n_mc: int,
    gen: np.random.Generator,
    pool: Literal["mean_with_edge", "sum_with_edge"],
    *,
    time_indices: Optional[Sequence[int]] = None,
    mc_quantiles: Optional[Sequence[float]] = None,
    mc_contributors: MCContributorMode = "individual_edge",
    mc_posterior: MCPosteriorSource = "filtered",
    smoothed_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
) -> GlobalBetaMCResult:
    """
    Monte Carlo over DLM posteriors + group pooling on the **global** DAG.

    See module docstring for pooling divisor :math:`A` vs total subjects :math:`S`,
    independence / non-hierarchical interpretation, and the smoothed-sampling
    caveat.

    If ``time_indices`` is set, steps (1)–(2) are repeated for each :math:`t`
    and ``beta_draws`` has shape ``(B, n_edges, n_times)``; otherwise a single
    ``time_index`` gives shape ``(B, n_edges)``.
    """
    if n_mc < 1:
        raise ValueError("n_draws must be at least 1")
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

    if time_indices is not None:
        t_list = [int(t) for t in time_indices]
    else:
        t_list = [int(time_index)]

    t_len = _infer_filter_time_length(filtered)
    for tix in t_list:
        if not (0 <= tix < t_len):
            raise ValueError(f"time index {tix} out of range for T={t_len}")

    global_adj = np.asarray(is_res.adj_mat, dtype=int)
    nn = global_adj.shape[0]
    edges = _global_dag_edges(global_adj)
    e_ct = len(edges)

    multi_t = len(t_list) > 1
    if e_ct == 0:
        meta_empty: Dict[str, Any] = {
            "edges_removed_for_acyclicity": is_res.metadata.get(
                "edges_removed_for_acyclicity", []
            ),
            "mc_posterior": mc_posterior,
            "mc_contributors": mc_contributors,
        }
        if multi_t:
            beta_empty = np.empty((n_mc, 0, len(t_list)), dtype=float)
        else:
            beta_empty = np.empty((n_mc, 0), dtype=float)
        return GlobalBetaMCResult(
            beta_draws=beta_empty,
            edges=[],
            n_contributors=np.zeros(0, dtype=int),
            time_index=t_list[0],
            pooling=pool,
            metadata=meta_empty,
            time_indices_mc=tuple(t_list) if multi_t else None,
            beta_mean=np.empty(0, dtype=float),
            beta_var=np.empty(0, dtype=float),
        )

    n_contrib = np.zeros(e_ct, dtype=int)
    for e, (p, cc) in enumerate(edges):
        if mc_contributors == "all_subjects":
            n_contrib[e] = ss
        else:
            n_contrib[e] = int(sum(1 for a in adjs_individual if a[p, cc] != 0))

    edge_mask_adjs = adjs_individual

    dummy_data = np.zeros((t_len, nn), dtype=float)
    parent_lists: List[List[List[int]]] = []
    for si in range(ss):
        pl_si: List[List[int]] = []
        for c in range(nn):
            _, pl = build_design_matrix(dummy_data, design_adjs[si], c)
            pl_si.append(list(pl))
        parent_lists.append(pl_si)

    if mc_posterior == "smoothed" and smoothed_per_subject is None:
        raise ValueError("smoothed_per_subject is required when mc_posterior='smoothed'")

    if multi_t:
        blocks = [
            _monte_carlo_beta_draws_at_time(
                filtered,
                edge_mask_adjs,
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
        beta_draws = np.stack(blocks, axis=2)
    else:
        beta_draws = _monte_carlo_beta_draws_at_time(
            filtered,
            edge_mask_adjs,
            edges,
            parent_lists,
            ss,
            nn,
            t_list[0],
            n_mc,
            gen,
            pool,
            mc_contributors=mc_contributors,
            smoothed_per_subject=smoothed_per_subject,
        )

    beta_mean = np.nanmean(beta_draws, axis=0)
    beta_var = np.nanvar(beta_draws, axis=0)

    beta_q: Optional[np.ndarray] = None
    q_tuple: Optional[Tuple[float, ...]] = None
    if mc_quantiles is not None:
        q_list = [float(x) for x in mc_quantiles]
        if q_list:
            q_tuple = tuple(q_list)
            beta_q = np.nanquantile(beta_draws, np.asarray(q_list, dtype=float), axis=0)

    meta_mc: Dict[str, Any] = {
        "edges_removed_for_acyclicity": is_res.metadata.get(
            "edges_removed_for_acyclicity", []
        ),
        "n_subjects": ss,
        "mc_posterior": mc_posterior,
        "mc_contributors": mc_contributors,
    }
    return GlobalBetaMCResult(
        beta_draws=beta_draws,
        edges=edges,
        n_contributors=n_contrib,
        time_index=t_list[0],
        pooling=pool,
        metadata=meta_mc,
        time_indices_mc=tuple(t_list) if multi_t else None,
        beta_quantiles=beta_q,
        quantile_levels=q_tuple,
        beta_mean=beta_mean,
        beta_var=beta_var,
    )


def _smooth_filtered_sequence(
    filtered: Sequence[Mapping[str, Any]],
    n_jobs: Optional[int],
) -> List[Dict[str, Any]]:
    sp = SmoothingPipeline(verbose=False)
    smoothed: List[Dict[str, Any]] = []
    for f in filtered:
        smoothed.append(
            sp.smooth_nodes(
                mt=f["mt"],
                Ct=f["Ct"],
                Rt=f["Rt"],
                nt=f["nt"],
                dt=f["dt"],
                n_jobs=n_jobs,
            )
        )
    return smoothed
