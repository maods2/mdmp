"""Subject coercion for the distance subpackage (mirrors ``inds/coercion``)."""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _is_fitted_mdm_like(obj: Any) -> bool:
    """True for fitted :class:`mdmp.model.MDM`-style objects (duck-typed)."""
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
    if getattr(obj, "adj_mat", None) is None:
        return False
    if getattr(obj, "Filt", None) is None:
        return False
    if getattr(obj, "node_names", None) is None:
        return False
    return True


def _subject_sequence_kind(subjects: Sequence[Any]) -> Literal["array", "mdm"]:
    if len(subjects) == 0:
        raise ValueError("subjects must contain at least one element")
    flags = [_is_fitted_mdm_like(x) for x in subjects]
    if all(flags):
        return "mdm"
    if not any(flags):
        return "array"
    raise TypeError(
        "fit_individual_structures / compute_mdm_distance: pass either only fitted "
        "MDM instances or only raw time-series arrays, not a mix."
    )


_SUBJECT_ID_ATTRS = ("subject_id", "subject", "id")


def _infer_mdm_subject_id(obj: Any) -> Any:
    """Return a subject identifier stored on a fitted MDM, if any."""
    for attr in _SUBJECT_ID_ATTRS:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                return val
    return None


def default_subject_ids(subjects: Sequence[Any]) -> List[Any]:
    """
    Default subject IDs for a cohort.

    When ``subjects`` are pre-fitted MDMs and every object carries a
    ``subject_id`` (or ``subject`` / ``id``) attribute, those values are
    reused; otherwise fall back to ``0 .. S-1``.
    """
    subjects_list = _materialize_subjects_list(subjects)
    s = len(subjects_list)
    if s == 0:
        return []
    try:
        if _subject_sequence_kind(subjects_list) != "mdm":
            return list(range(s))
    except ValueError:
        return list(range(s))
    inferred = [_infer_mdm_subject_id(m) for m in subjects_list]
    if all(x is not None for x in inferred):
        return inferred
    return list(range(s))


def _materialize_subjects_list(subjects: Sequence[Any]) -> List[Any]:
    if isinstance(subjects, (list, tuple)):
        return list(subjects)
    if isinstance(subjects, np.ndarray) and subjects.ndim == 2:
        return [subjects]
    if isinstance(subjects, pd.DataFrame):
        return [subjects]
    return list(subjects)


def coerce_subjects_for_distance(
    subjects: Sequence[Any],
    node_names: Optional[Sequence[str]] = None,
) -> Tuple[List[np.ndarray], Optional[List[str]], Optional[List[Any]]]:
    """
    Normalize ``subjects`` to data arrays and optionally fitted MDM objects.

    Returns
    -------
    arrays, node_names, mdms_or_none
    """
    subjects_list = _materialize_subjects_list(subjects)
    kind = _subject_sequence_kind(subjects_list)

    if kind == "array":
        arrays = [np.asarray(s, dtype=float) for s in subjects_list]
        shapes = {a.shape[1] for a in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"All subject arrays must share the same node count, got {shapes}"
            )
        n_nodes = next(iter(shapes))
        if node_names is not None:
            if len(node_names) != n_nodes:
                raise ValueError(
                    f"node_names length {len(node_names)} does not match N={n_nodes}"
                )
            names = [str(x) for x in node_names]
        else:
            names = [f"V{i + 1}" for i in range(n_nodes)]
        return arrays, names, None

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

    arrays = [np.asarray(m.data, dtype=float) for m in mdms]
    return arrays, names_ref, mdms
