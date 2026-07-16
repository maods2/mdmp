"""Pluggable pairwise metrics between fitted MDM subjects."""

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from ...scoring import compute_logpl, select_discount_factors
from ...utils import get_default_delta

MetricFn = Callable[..., float]


def _clip_delta(delta: float) -> float:
    return min(float(delta), 1.0)


def joint_lpl(
    data: np.ndarray,
    adj_mat: np.ndarray,
    delta: np.ndarray,
    nbf: int,
) -> float:
    """Joint log predictive likelihood (sum of per-node LPLs)."""
    n_nodes = data.shape[1]
    total = 0.0
    for r in range(n_nodes):
        d_r = _clip_delta(delta[r])
        total += -compute_logpl(data, adj_mat, d_r, r, nbf)
    return total


def _extract_strength_matrix(mdm: Any) -> np.ndarray:
    """Vectorise smoothed edge coefficients into an (N, N) strength matrix."""
    adj = np.asarray(mdm.adj_mat, dtype=float)
    n = adj.shape[0]
    strength = np.zeros((n, n), dtype=float)
    smoo = mdm.Smoo
    smt_list = smoo["smt"]
    for j in range(n):
        smt = np.asarray(smt_list[j])
        if smt.ndim == 1:
            # intercept only — no parent edges
            continue
        # smt shape (p, T); row 0 is intercept, rows 1: are parent coefficients
        parents = np.where(adj[:, j] > 0)[0]
        if len(parents) == 0:
            continue
        # use time-averaged smoothed means for parent coefficients
        coefs = np.mean(smt[1:, :], axis=1)
        for k, parent in enumerate(parents):
            if k < len(coefs):
                strength[parent, j] = coefs[k]
    return strength


def lpl_separation(mdm_i: Any, mdm_j: Any, *, ctx: Optional[dict] = None) -> float:
    """Log-Bayes-factor separation (Paper #1, default metric)."""
    ctx = ctx or {}
    self_lpl_i = ctx["self_lpl_i"]
    self_lpl_j = ctx["self_lpl_j"]
    data_i = np.asarray(mdm_i.data, dtype=float)
    data_j = np.asarray(mdm_j.data, dtype=float)
    m_ij = ctx["common_adj"]
    nbf = ctx["nbf"]
    delta_grid = ctx.get("delta_grid")
    if delta_grid is None:
        delta_grid = get_default_delta()

    df_i = select_discount_factors(data_i, m_ij, nbf=nbf, delta=delta_grid)["DF_hat"]
    df_j = select_discount_factors(data_j, m_ij, nbf=nbf, delta=delta_grid)["DF_hat"]
    lpl_i_shared = joint_lpl(data_i, m_ij, df_i, nbf)
    lpl_j_shared = joint_lpl(data_j, m_ij, df_j, nbf)
    d = (self_lpl_i + self_lpl_j) - (lpl_i_shared + lpl_j_shared)
    if d < -1e-6:
        warnings.warn(
            f"lpl_separation: d={d:.4g} < 0 (possível inconsistência de "
            f"score/busca; M_i deveria ser ótimo para i)."
        )
    return max(float(d), 0.0)


def structural_hamming(mdm_i: Any, mdm_j: Any, *, ctx: Optional[dict] = None) -> float:
    """Edit distance between individual adjacency matrices."""
    del ctx
    a = (np.asarray(mdm_i.adj_mat) > 0).astype(int)
    b = (np.asarray(mdm_j.adj_mat) > 0).astype(int)
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    return float(np.sum(np.abs(a - b)))


def strength_frobenius(mdm_i: Any, mdm_j: Any, *, ctx: Optional[dict] = None) -> float:
    """Frobenius distance between smoothed connectivity strength matrices."""
    del ctx
    si = _extract_strength_matrix(mdm_i)
    sj = _extract_strength_matrix(mdm_j)
    return float(np.linalg.norm(si - sj, ord="fro"))


METRIC_REGISTRY: Dict[str, MetricFn] = {
    "lpl_separation": lpl_separation,
    "structural_hamming": structural_hamming,
    "strength_frobenius": strength_frobenius,
}


def resolve_metric(metric: Union[str, MetricFn]) -> tuple[str, MetricFn]:
    """Return ``(name, callable)`` for a registry name or user callable."""
    if callable(metric):
        name = getattr(metric, "__name__", "custom")
        return name, metric
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric {metric!r}; choose from {list(METRIC_REGISTRY)} or pass a callable."
        )
    return metric, METRIC_REGISTRY[metric]
