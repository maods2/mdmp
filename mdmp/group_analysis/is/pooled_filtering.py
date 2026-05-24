"""Pooled filter dict for plotting on a global DAG."""

from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from ...utils import build_design_matrix, build_parameter_names
from .coercion import _to_binary_adj


def build_plot_filt_from_subjects(
    global_adj: np.ndarray,
    filtered_per_subject: Sequence[Mapping[str, Any]],
    adj_per_subject: Sequence[Union[np.ndarray, pd.DataFrame]],
    node_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Build a ``Filt``-shaped dict on the consensus DAG by conditionally pooling
    per-subject filtered posteriors (mean of ``mt`` / diagonal ``Ct``, mean of
    ``nt`` / ``dt``).

    **Conditional pooling.**  For each child node and each regression coefficient
    aligned with the global parent ordering, only subjects whose individual DAG
    contains the same directed parent edge contribute to that coefficient's
    pooled series.  **Subjects without the edge do not contribute to the pooled
    coefficient and are excluded from the divisor.**  This is a conditional
    mean, analogous to ``pooling='conditional_mean_among_edge_subjects'`` in
    Monte Carlo aggregation.

    **Visualization only.**  The returned dict is a plug-in summary for
    :func:`mdmp.plotting.plot_arcs` and related routines.  It is **not** a
    joint Bayesian posterior on the global graph, and it does not propagate
    structural uncertainty.
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
