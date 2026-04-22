"""
Individual Structure (IS) aggregation: combine subject DAGs by edge voting.

Pipeline (high level)
---------------------
1. Normalize the first argument (optional single MDM / single 2D adjacency).
2. Coerce MDM-like inputs to per-subject adjacency + optional ``Filt`` / ``plot_data``.
3. Validate adjacency list and optional plot arrays.
4. **Vote**: edge frequency > ``tau`` → candidate global DAG; **repair** cycles by
   dropping the lowest-frequency edge on each cycle until acyclic.
5. Optionally build a pooled ``Filt`` for plotting, or run Monte Carlo on filtered
   states for global edge coefficients.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from ...utils import build_design_matrix, build_parameter_names

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GlobalBetaMCResult:
    """
    Monte Carlo draws of pooled edge coefficients aligned to a global DAG.

    Procedure (matches propagating uncertainty through a **group mean**):

    1. For each replicate :math:`b=1,\\ldots,B`, draw a regression state vector
       from each subject's filtered **multivariate** Student-*t*-type posterior
       (Gamma–Normal mixture on :math:`(m_{i,t}, C_{i,t}, n_t, d_t)`) at the
       chosen time index(es).
    2. For each global edge, align the local coefficient for that parent on
       each subject graph; with ``pooling='mean_with_edge'``, set
       :math:`\\bar{\\theta}_t^{(b)} = \\frac{1}{A}\\sum_{i \\in \\mathcal{A}}
       \\theta_{i,t}^{(b)}` where :math:`\\mathcal{A}` is the set of subjects
       whose individual DAG contains that edge and :math:`A=|\\mathcal{A}|`.
    3. The columns of ``beta_draws`` (and optional ``beta_quantiles``) summarize
       the empirical distribution of :math:`\\{\\bar{\\theta}_t^{(b)}\\}_{b=1}^B`
       (quantiles = step 3 in the usual Monte Carlo workflow).

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
        Per-edge count of subjects whose *individual* DAG contained that edge
        (the denominator :math:`A` for ``mean_with_edge``).
    time_index : int
        First / reference time slice (when a single ``t`` is used, this is it).
    pooling : str
        Pooling policy label (see :func:`aggregate_individual_structures`).
    metadata : dict
        Extra diagnostics (e.g. ``n_subjects``).
    time_indices_mc : tuple of int, optional
        If ``beta_draws`` has a time dimension, these are the time indices in
        axis order (length matches ``beta_draws.shape[2]``).
    beta_quantiles : np.ndarray, optional
        Empirical quantiles of ``beta_draws`` along the draw axis (axis 0):
        shape ``(n_levels, n_edges)`` or ``(n_levels, n_edges, n_times)``.
    quantile_levels : tuple of float, optional
        Probability levels used for ``beta_quantiles`` (same order as axis 0).
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
        Threshold used for inclusion (edge kept iff frequency > tau).
    metadata : dict
        Includes ``edge_counts``, ``edge_frequencies``,
        ``edges_removed_for_acyclicity`` (list of dicts with parent_idx, child_idx,
        frequency).
    global_beta_mc : GlobalBetaMCResult, optional
        Present when :func:`aggregate_individual_structures` was called with
        ``filtered_per_subject`` and ``n_draws > 0``.
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
    """

    data: Optional[np.ndarray] = None
    Filt: Optional[Dict[str, Any]] = None
    Smoo: Optional[Dict[str, Any]] = None
    DF: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        n_edge = int(np.sum(self.adj_mat != 0))
        return (
            f"ISAggregatedMDMView(nodes={len(self.node_names)}, "
            f"time_points={self.data.shape[0] if self.data is not None else 0}, "
            f"edges={n_edge}/{self.adj_mat.size}, n_subjects={self.n_subjects})"
        )


# ---------------------------------------------------------------------------
# Pooled Filt for plotting (mean across subjects with matching edges)
# ---------------------------------------------------------------------------


def build_plot_filt_from_subjects(
    global_adj: np.ndarray,
    filtered_per_subject: Sequence[Mapping[str, Any]],
    adj_per_subject: Sequence[Union[np.ndarray, pd.DataFrame]],
    node_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Build a ``Filt``-shaped dict on a **global** adjacency by pooling per-subject
    filtered posteriors (mean of ``mt`` / diagonal ``Ct``, mean of ``nt`` / ``dt``).

    For each child node and each regression coefficient aligned with the global
    parent ordering, only subjects whose individual DAG contains the same
    directed parent edge contribute to that coefficient's pooled series.

    This is a plug-in summary for visualization (e.g. :func:`mdmp.plotting.plot_arcs`);
    it is not a joint Bayesian posterior on the global graph.
    """
    arrays: List[np.ndarray] = []
    for raw in adj_per_subject:
        a, _ = _to_binary_adj(raw)
        arrays.append(a)

    ga = np.asarray(global_adj, dtype=int)
    n = ga.shape[0]
    if arrays[0].shape != (n, n):
        raise ValueError(
            f"global_adj shape {ga.shape} must match subject adjacencies {arrays[0].shape}"
        )

    s_sub = len(filtered_per_subject)
    if len(arrays) != s_sub:
        raise ValueError(
            f"adj_per_subject length {len(arrays)} must match filtered_per_subject length {s_sub}"
        )

    T = int(np.asarray(filtered_per_subject[0]["mt"][0]).shape[-1])
    dummy = np.zeros((T, n), dtype=float)
    str_names: List[str] = [str(x) for x in node_names]
    if len(str_names) != n:
        raise ValueError(f"node_names length {len(str_names)} must match N={n}")

    mt: Dict[int, np.ndarray] = {}
    Ct: Dict[int, np.ndarray] = {}
    nt: Dict[int, np.ndarray] = {}
    dt: Dict[int, np.ndarray] = {}
    row_names: Dict[int, List[str]] = {}

    for c in range(n):
        Ft, pl_g = build_design_matrix(dummy, ga, c)
        p = Ft.shape[1]
        m_arr = np.zeros((p, T))
        c_arr = np.zeros((p, p, T))
        n_vec = np.zeros(T)
        d_vec = np.zeros(T)

        for t in range(T):
            n_vec[t] = float(
                np.mean([float(f["nt"][c][t]) for f in filtered_per_subject])
            )
            d_vec[t] = float(
                np.mean([float(f["dt"][c][t]) for f in filtered_per_subject])
            )
            for j in range(p):
                mvals: List[float] = []
                cvals: List[float] = []
                for si, filt in enumerate(filtered_per_subject):
                    adj_s = arrays[si]
                    _, pl_s = build_design_matrix(dummy, adj_s, c)
                    mt_s = np.asarray(filt["mt"][c], dtype=float)
                    Ct_s = np.asarray(filt["Ct"][c], dtype=float)
                    if mt_s.ndim == 1:
                        mt_s = mt_s.reshape(-1, T)
                    if j == 0:
                        idx = 0
                    else:
                        par = pl_g[j - 1]
                        if par not in pl_s:
                            continue
                        idx = 1 + pl_s.index(par)
                    if mt_s.shape[0] <= idx:
                        continue
                    mvals.append(float(mt_s[idx, t]))
                    if Ct_s.ndim == 3:
                        cvals.append(float(Ct_s[idx, idx, t]))
                    elif Ct_s.ndim == 2 and Ct_s.shape[-1] == T:
                        cvals.append(float(Ct_s[idx, idx, t]))
                    else:
                        cvals.append(float(Ct_s[idx, idx]))
                if mvals:
                    m_arr[j, t] = float(np.mean(mvals))
                if cvals:
                    c_arr[j, j, t] = float(np.mean(cvals))

        mt[c] = m_arr
        Ct[c] = c_arr
        nt[c] = n_vec
        dt[c] = d_vec
        row_names[c] = build_parameter_names(c, ga, str_names)

    return {"mt": mt, "Ct": Ct, "nt": nt, "dt": dt, "row_names": row_names}


# ---------------------------------------------------------------------------
# Adjacency parsing & validation
# ---------------------------------------------------------------------------


def _as_float_matrix(adj: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (N,N) float array and optional column names from DataFrame."""
    if isinstance(adj, pd.DataFrame):
        names = [str(c) for c in adj.columns.tolist()]
        return np.asarray(adj.values, dtype=float), names
    return np.asarray(adj, dtype=float), None


def _to_binary_adj(
    adj: Union[np.ndarray, pd.DataFrame],
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Return (N,N) int array and optional names from DataFrame."""
    arr, names = _as_float_matrix(adj)
    flat = arr.ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("adjacency matrices must contain only finite values")
    if not np.logical_or(flat == 0, flat == 1).all():
        raise ValueError(
            "adjacency must be binary (0/1); non-binary values are not allowed"
        )
    return arr.astype(int), names


def _validate_adj_list(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame]],
    node_names: Optional[Sequence[str]],
) -> Tuple[List[np.ndarray], List[str], int]:
    if len(adj_mats) == 0:
        raise ValueError("adj_mats must contain at least one adjacency matrix")

    parsed: List[Tuple[np.ndarray, Optional[List[str]]]] = [
        _to_binary_adj(a) for a in adj_mats
    ]
    shapes = {p[0].shape for p in parsed}
    if len(shapes) != 1:
        raise ValueError(
            f"All adjacency matrices must have the same shape, got {shapes}"
        )
    n, m = parsed[0][0].shape
    if n != m:
        raise ValueError(f"Adjacency matrices must be square, got shape {(n, m)}")

    arrays = [p[0] for p in parsed]

    names: List[str]
    if node_names is not None:
        if len(node_names) != n:
            raise ValueError(
                f"node_names length {len(node_names)} does not match N={n}"
            )
        names = [str(x) for x in node_names]
    else:
        first_df_names = parsed[0][1]
        if first_df_names is not None and len(first_df_names) == n:
            names = first_df_names
        else:
            for p in parsed:
                if p[1] is not None and len(p[1]) == n:
                    names = p[1]
                    break
            else:
                names = [f"V{i + 1}" for i in range(n)]

    return arrays, names, len(arrays)


# ---------------------------------------------------------------------------
# MDM-like inputs → adjacency + optional Filt / plot_data
# ---------------------------------------------------------------------------


def _is_fitted_mdm_like(obj: Any) -> bool:
    """
    True for fitted :class:`mdmp.model.MDM`-style objects (duck-typed).

    Plain adjacency inputs are ``ndarray`` / ``DataFrame`` and are excluded.
    """
    if obj is None or isinstance(obj, (np.ndarray, np.generic)):
        return False
    if isinstance(obj, pd.DataFrame):
        return False
    try:
        from ...model import MDM as _MDM

        if isinstance(obj, _MDM):
            return True
    except Exception:
        pass
    try:
        if getattr(obj, "adj_mat", None) is None:
            return False
        if getattr(obj, "Filt", None) is None:
            return False
        if getattr(obj, "node_names", None) is None:
            return False
    except Exception:
        return False
    return True


def _subject_sequence_kind(subjects: Sequence[Any]) -> Literal["adj", "mdm"]:
    if len(subjects) == 0:
        raise ValueError("adj_mats must contain at least one element")
    flags = [_is_fitted_mdm_like(x) for x in subjects]
    if all(flags):
        return "mdm"
    if not any(flags):
        return "adj"
    raise TypeError(
        "aggregate_individual_structures: pass either only fitted MDM instances "
        "(with adj_mat, Filt, node_names) or only adjacency matrices / DataFrames, "
        "not a mix."
    )


def _materialize_subjects_list(subjects: Sequence[Any]) -> List[Any]:
    if isinstance(subjects, (list, tuple)):
        return list(subjects)
    if isinstance(subjects, np.ndarray) and subjects.ndim == 2:
        return [subjects]
    if isinstance(subjects, pd.DataFrame):
        return [subjects]
    return list(subjects)


def _coerce_subjects_for_aggregation(
    subjects: Sequence[Any],
    node_names: Optional[Sequence[str]],
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]],
    plot_data: Optional[np.ndarray],
    *,
    pool_filt_for_plotting: bool,
    n_draws: int,
) -> Tuple[
    Sequence[Union[np.ndarray, pd.DataFrame]],
    Optional[Sequence[str]],
    Optional[Sequence[Mapping[str, Any]]],
    Optional[np.ndarray],
]:
    """
    If ``subjects`` are MDM-like, build adjacency list and optionally fill
    ``filtered_per_subject`` / ``plot_data`` from each model.
    """
    subjects_list = _materialize_subjects_list(subjects)

    kind = _subject_sequence_kind(subjects_list)
    if kind == "adj":
        return subjects_list, node_names, filtered_per_subject, plot_data

    mdms: List[Any] = subjects_list
    names_ref = [str(x) for x in mdms[0].node_names]
    for mi, m in enumerate(mdms[1:], start=1):
        other = [str(x) for x in m.node_names]
        if other != names_ref:
            raise ValueError(
                "All fitted MDM objects must share the same node_names in the same "
                f"order; index 0 vs {mi} differ."
            )
    if node_names is not None:
        if [str(x) for x in node_names] != names_ref:
            raise ValueError(
                "node_names=... does not match the node_names on the MDM objects."
            )

    adjs: List[np.ndarray] = []
    for m in mdms:
        a = np.asarray(m.adj_mat, dtype=float)
        b = (a > 0).astype(np.int64)
        np.fill_diagonal(b, 0)
        adjs.append(b)

    filt_eff = filtered_per_subject
    if filt_eff is None and (pool_filt_for_plotting or n_draws > 0):
        filt_eff = [m.Filt for m in mdms]

    plot_eff = plot_data
    if plot_eff is None and pool_filt_for_plotting:
        datas = [np.asarray(m.data, dtype=float) for m in mdms]
        shapes = {d.shape for d in datas}
        if len(shapes) == 1:
            plot_eff = np.mean(np.stack(datas, axis=0), axis=0)

    return adjs, node_names if node_names is not None else names_ref, filt_eff, plot_eff


def _normalize_first_argument(adj_mats: Any) -> Any:
    """Wrap a single MDM or single 2D adjacency matrix as a one-element sequence."""
    try:
        from ...model import MDM as _MDM
    except Exception:  # pragma: no cover
        _MDM = None

    if _MDM is not None and isinstance(adj_mats, _MDM):
        return [adj_mats]
    if isinstance(adj_mats, np.ndarray) and adj_mats.ndim == 2:
        return [adj_mats]
    return adj_mats


# ---------------------------------------------------------------------------
# Majority vote + acyclic repair
# ---------------------------------------------------------------------------


def _adj_to_graph(adj: np.ndarray) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(adj.shape[0]))
    idx = np.argwhere(adj != 0)
    for i, j in idx:
        if i != j:
            g.add_edge(int(i), int(j))
    return g


def _remove_lowest_freq_cycle_edge(
    adj: np.ndarray,
    freq: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    If ``adj`` has a directed cycle, remove one cycle edge with minimum frequency.

    Tie-break: lexicographic (parent_idx, child_idx).

    Returns
    -------
    tuple or None
        (i, j) removed, or None if already acyclic.
    """
    g = _adj_to_graph(adj)
    if nx.is_directed_acyclic_graph(g):
        return None
    try:
        cyc = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return None

    edges: List[Tuple[int, int]] = []
    for item in cyc:
        if len(item) == 3:
            u, v, _ = item
            edges.append((int(u), int(v)))
        else:
            u, v = item[0], item[1]
            edges.append((int(u), int(v)))

    def _sort_key(e: Tuple[int, int]) -> Tuple[float, int, int]:
        return (float(freq[e[0], e[1]]), e[0], e[1])

    i, j = min(edges, key=_sort_key)
    adj[i, j] = 0
    return (i, j)


def _vote_threshold_and_repair_cycles(
    subject_adjs: List[np.ndarray],
    tau: float,
    node_names: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Edge-wise majority vote (frequency > tau), then drop lowest-frequency cycle
    edges until the graph is a DAG.
    """
    s = len(subject_adjs)
    stack = np.stack(subject_adjs, axis=0)
    edge_counts = np.sum(stack, axis=0).astype(np.int64)
    np.fill_diagonal(edge_counts, 0)
    edge_frequencies = edge_counts.astype(float) / float(s)

    adj = (edge_frequencies > tau).astype(np.int8)
    np.fill_diagonal(adj, 0)

    removed: List[Dict[str, Any]] = []
    while True:
        dropped = _remove_lowest_freq_cycle_edge(adj, edge_frequencies)
        if dropped is None:
            break
        i, j = dropped
        removed.append(
            {
                "parent_idx": i,
                "child_idx": j,
                "parent_name": node_names[i],
                "child_name": node_names[j],
                "frequency": float(edge_frequencies[i, j]),
            }
        )

    out_adj = adj.astype(np.float64)
    meta: Dict[str, Any] = {
        "edge_counts": edge_counts,
        "edge_frequencies": edge_frequencies.copy(),
        "edges_removed_for_acyclicity": removed,
    }
    return out_adj, meta


# ---------------------------------------------------------------------------
# Monte Carlo: pool filtered regression states onto global edges
# ---------------------------------------------------------------------------


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
    adjs: List[np.ndarray],
    edges: List[Tuple[int, int]],
    parent_lists: List[List[List[int]]],
    ss: int,
    nn: int,
    t_index: int,
    n_mc: int,
    gen: np.random.Generator,
    pool: Literal["mean_with_edge", "sum_with_edge"],
) -> np.ndarray:
    """
    One timestep: for each MC replicate b, sample per subject then pool per edge.

    ``mean_with_edge`` implements
    :math:`\\bar{\\theta}^{(b)} = \\frac{1}{A}\\sum_{i\\in\\mathcal{A}} \\theta_i^{(b)}`.
    """
    e_ct = len(edges)
    beta_draws = np.empty((n_mc, e_ct), dtype=float)

    for b in range(n_mc):
        draws: List[List[np.ndarray]] = []
        for si, filt in enumerate(filtered):
            row: List[np.ndarray] = []
            for c in range(nn):
                mt_c = filt["mt"][c]
                ct_c = filt["Ct"][c]
                nt_c = filt["nt"][c]
                dt_c = filt["dt"][c]
                m_col = mt_c[:, t_index]
                c_slice = ct_c[:, :, t_index]
                nt_t = float(nt_c[t_index])
                dt_t = float(dt_c[t_index])
                row.append(
                    _sample_dlm_state_posterior(m_col, c_slice, nt_t, dt_t, gen)
                )
            draws.append(row)

        for e, (p, cc) in enumerate(edges):
            vals: List[float] = []
            for si in range(ss):
                if adjs[si][p, cc] == 0:
                    continue
                theta = draws[si][cc]
                aligned = _align_child_local_to_global(
                    theta, parent_lists[si][cc], [p]
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
    adjs: List[np.ndarray],
    is_res: ISAggregatedMDMView,
    time_index: int,
    n_mc: int,
    gen: np.random.Generator,
    pool: Literal["mean_with_edge", "sum_with_edge"],
    *,
    time_indices: Optional[Sequence[int]] = None,
    mc_quantiles: Optional[Sequence[float]] = None,
) -> GlobalBetaMCResult:
    """
    Monte Carlo over filtered posteriors + group pooling on the **global** DAG.

    If ``time_indices`` is set, steps (1)–(2) are repeated for each :math:`t`
    and ``beta_draws`` has shape ``(B, n_edges, n_times)``; otherwise a single
    ``time_index`` gives shape ``(B, n_edges)``.
    """
    if n_mc < 1:
        raise ValueError("n_draws must be at least 1")
    ss = len(filtered)
    if ss == 0:
        raise ValueError("filtered_per_subject must be non-empty")
    if len(adjs) != ss:
        raise ValueError(
            f"adjacency list length {len(adjs)} != filtered_per_subject length {ss}"
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
        )

    n_contrib = np.zeros(e_ct, dtype=int)
    for e, (p, cc) in enumerate(edges):
        n_contrib[e] = int(sum(1 for a in adjs if a[p, cc] != 0))

    dummy_data = np.zeros((t_len, nn), dtype=float)
    parent_lists: List[List[List[int]]] = []
    for si in range(ss):
        pl_si: List[List[int]] = []
        for c in range(nn):
            _, pl = build_design_matrix(dummy_data, adjs[si], c)
            pl_si.append(list(pl))
        parent_lists.append(pl_si)

    if multi_t:
        blocks = [
            _monte_carlo_beta_draws_at_time(
                filtered,
                adjs,
                edges,
                parent_lists,
                ss,
                nn,
                tix,
                n_mc,
                gen,
                pool,
            )
            for tix in t_list
        ]
        beta_draws = np.stack(blocks, axis=2)
    else:
        beta_draws = _monte_carlo_beta_draws_at_time(
            filtered,
            adjs,
            edges,
            parent_lists,
            ss,
            nn,
            t_list[0],
            n_mc,
            gen,
            pool,
        )

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
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_individual_structures(
    adj_mats: Sequence[Union[np.ndarray, pd.DataFrame, Any]],
    tau: float = 0.5,
    node_names: Optional[Sequence[str]] = None,
    *,
    filtered_per_subject: Optional[Sequence[Mapping[str, Any]]] = None,
    time_index: int = 0,
    time_indices: Optional[Sequence[int]] = None,
    n_draws: int = 0,
    rng: Optional[np.random.Generator] = None,
    pooling: Literal["mean_with_edge", "sum_with_edge"] = "mean_with_edge",
    plot_data: Optional[np.ndarray] = None,
    plot_filt: Optional[Mapping[str, Any]] = None,
    plot_smoo: Optional[Mapping[str, Any]] = None,
    plot_df: Optional[Mapping[str, Any]] = None,
    pool_filt_for_plotting: bool = False,
    mc_quantiles: Optional[Sequence[float]] = None,
) -> ISAggregatedMDMView:
    """
    Aggregate subject-specific DAG adjacency matrices into one global DAG.

    For each directed edge (i → j), compute the fraction of subjects that
    include the edge. Include it in the pooled graph if frequency > ``tau``.
    If the thresholded graph has directed cycles, repeatedly remove the
    lowest-frequency edge on a detected cycle until the graph is acyclic.

    Optional keyword-only arguments run Monte Carlo pooling of regression
    coefficients (from each subject's filtered DLM) onto the **same** global
    DAG built from ``adj_mats``, using subject graphs identical to those in
    ``adj_mats``.

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
        Strict threshold in (0, 1]. Default 0.5 (majority: more than half).
    node_names : sequence of str, optional
        Names of length N. If omitted, taken from the first DataFrame, MDM
        ``node_names``, or ``V1``…``VN``.
    filtered_per_subject : sequence of dict-like, optional
        One dict per subject (same length and order as ``adj_mats``) with keys
        ``'mt'``, ``'Ct'``, ``'nt'``, ``'dt'`` per child index, as from filtering.
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
        How to pool coefficients across subjects that have each global edge.
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
    ``mdmp.dlm`` filtering.

    **Monte Carlo (resumo):** para cada :math:`b`, amostrar estados filtrados por
    indivíduo; para cada aresta global, calcular a média grupal dos coeficientes
    alinhados (passo ``mean_with_edge``); repetir gera :math:`\\{\\bar{\\theta}^{(b)}\\}`,
    da qual quantis empíricos opcionais vêm de ``mc_quantiles``.
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if plot_filt is not None and pool_filt_for_plotting:
        raise ValueError("pass only one of plot_filt=... or pool_filt_for_plotting=True")

    adj_mats_norm = _normalize_first_argument(adj_mats)
    adj_mats_eff, node_names_eff, filtered_eff, plot_data_eff = _coerce_subjects_for_aggregation(
        adj_mats_norm,
        node_names,
        filtered_per_subject,
        plot_data,
        pool_filt_for_plotting=pool_filt_for_plotting,
        n_draws=n_draws,
    )

    if n_draws > 0 and filtered_eff is None:
        raise ValueError(
            "filtered_per_subject is required when n_draws > 0 "
            "(unless adj_mats are fitted MDM instances with Filt)"
        )
    if filtered_eff is not None and n_draws > 0 and rng is None:
        raise ValueError("rng is required when n_draws > 0 and filtered_per_subject is set")
    if pool_filt_for_plotting and filtered_eff is None:
        raise ValueError(
            "filtered_per_subject is required when pool_filt_for_plotting=True "
            "(unless adj_mats are fitted MDM instances with Filt)"
        )

    arrays, names, s = _validate_adj_list(adj_mats_eff, node_names_eff)
    if filtered_eff is not None and len(filtered_eff) != s:
        raise ValueError(
            f"filtered_per_subject length {len(filtered_eff)} must match "
            f"number of adjacency matrices {s}"
        )

    n = arrays[0].shape[0]
    if plot_data_eff is not None:
        pd_arr = np.asarray(plot_data_eff)
        if pd_arr.ndim != 2 or pd_arr.shape[1] != n:
            raise ValueError(
                f"plot_data must have shape (T, {n}), got {getattr(pd_arr, 'shape', None)}"
            )

    out_adj, meta = _vote_threshold_and_repair_cycles(arrays, tau, names)

    filt_final: Optional[Dict[str, Any]] = None
    if plot_filt is not None:
        filt_final = dict(plot_filt)
    elif pool_filt_for_plotting:
        assert filtered_eff is not None
        filt_final = build_plot_filt_from_subjects(out_adj, filtered_eff, arrays, names)

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

    if filtered_eff is not None and n_draws > 0:
        assert rng is not None
        gb = _monte_carlo_global_edge_beta(
            filtered_eff,
            arrays,
            result,
            time_index,
            n_draws,
            rng,
            pooling,
            time_indices=time_indices,
            mc_quantiles=mc_quantiles,
        )
        result = replace(result, global_beta_mc=gb)

    return result
