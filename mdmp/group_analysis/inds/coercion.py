"""Adjacency normalization, validation, and MDM-like subject coercion."""

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class _PreparedSubjects:
    """
    Normalized per-subject inputs after coercion (one row per subject).

    Fields here are the **resolved** values used by the pipeline: caller
    arguments may be ``None`` and are filled from fitted MDM objects when
    applicable (see ``_coerce_subjects_for_aggregation``).
    """

    arrays: List[np.ndarray]
    names: List[str]
    n_subjects: int
    n_nodes: int
    posterior_per_subject: Optional[Sequence[Mapping[str, Any]]]
    time_series: Optional[np.ndarray]
    mdm_data_per_subject: Optional[List[np.ndarray]]


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
    except ImportError:
        pass
    # Duck-typed fallback: check for MDM-like attributes
    if getattr(obj, "adj_mat", None) is None:
        return False
    if getattr(obj, "Filt", None) is None:
        return False
    if getattr(obj, "node_names", None) is None:
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
    posterior_per_subject: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[
    Sequence[Union[np.ndarray, pd.DataFrame]],
    Optional[Sequence[str]],
    Optional[Sequence[Mapping[str, Any]]],
    Optional[np.ndarray],
    Optional[List[np.ndarray]],
]:
    """
    If ``subjects`` are MDM-like, build adjacency list and fill
    ``posterior_per_subject`` / mean ``time_series`` from each model.

    ``posterior_per_subject`` is only accepted on split APIs (e.g.
    :func:`aggregate_individual_structures`
    derives filters from MDMs only.

    Returns adjacency list, node names, **resolved** filtered states,
    **resolved** group ``(T, N)`` time series, and per-subject data arrays
    (``None`` for plain adjacency-only inputs).
    """
    subjects_list = _materialize_subjects_list(subjects)

    kind = _subject_sequence_kind(subjects_list)
    if kind == "adj":
        return subjects_list, node_names, posterior_per_subject, None, None

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

    resolved_posterior = (
        posterior_per_subject if posterior_per_subject is not None else [m.Filt for m in mdms]
    )

    datas = [np.asarray(m.data, dtype=float) for m in mdms]
    shapes = {d.shape for d in datas}
    resolved_time_series: Optional[np.ndarray] = None
    if len(shapes) == 1:
        resolved_time_series = np.mean(np.stack(datas, axis=0), axis=0)

    data_per_subject = datas

    return (
        adjs,
        node_names if node_names is not None else names_ref,
        resolved_posterior,
        resolved_time_series,
        data_per_subject,
    )


def _normalize_first_argument(adj_mats: Any) -> Any:
    """Wrap a single MDM or single 2D adjacency matrix as a one-element sequence."""
    try:
        from ...model import MDM as _MDM
        if isinstance(adj_mats, _MDM):
            return [adj_mats]
    except ImportError:  # pragma: no cover
        pass
    if isinstance(adj_mats, np.ndarray) and adj_mats.ndim == 2:
        return [adj_mats]
    return adj_mats
