"""Stage 1 — per-subject individual structure estimation."""

from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ...model import MDM
from ...utils import get_default_delta
from .coercion import (
    _materialize_subjects_list,
    _subject_sequence_kind,
    default_subject_ids,
)


def _clip_delta_grid(delta_grid: Optional[np.ndarray]) -> np.ndarray:
    grid = get_default_delta() if delta_grid is None else np.asarray(delta_grid, dtype=float)
    return np.minimum(grid, 1.0)


def _as_mdm_input(data: np.ndarray, node_names: Optional[List[str]]) -> Union[np.ndarray, pd.DataFrame]:
    if node_names is None:
        return data
    return pd.DataFrame(data, columns=node_names)


def fit_individual_structures(
    subjects: Sequence[Union[np.ndarray, MDM]],
    *,
    method: Literal["hc", "tabu", "mmhc"] = "hc",
    nbf: int = 15,
    delta_grid: Optional[np.ndarray] = None,
    node_names: Optional[List[str]] = None,
    subject_ids: Optional[Sequence[Any]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> List[MDM]:
    """
    Fit one :class:`mdmp.model.MDM` per subject (stage 1 of the GS workflow).

    Already-fitted MDM objects are passed through unchanged.
    """
    subjects_list = _materialize_subjects_list(subjects)
    if len(subjects_list) == 0:
        raise ValueError("subjects must contain at least one element")

    kind = _subject_sequence_kind(subjects_list)
    if subject_ids is None:
        subject_ids = default_subject_ids(subjects_list)
    else:
        subject_ids = list(subject_ids)

    delta_grid = _clip_delta_grid(delta_grid)

    if kind == "mdm":
        if verbose:
            for sid in subject_ids:
                print(f"[distance] reusing fitted MDM for subject {sid}")
        return list(subjects_list)  # type: ignore[return-value]

    models: List[MDM] = []
    for idx, data in enumerate(subjects_list):
        if verbose:
            print(f"[distance] fitting individual structure for subject {subject_ids[idx]}")
        m = MDM(
            _as_mdm_input(np.asarray(data, dtype=float), node_names),
            method=method,
            nbf=nbf,
            delta=delta_grid,
            verbose=False,
            n_jobs=n_jobs,
        )
        m.subject_id = subject_ids[idx]
        models.append(m)
    return models
