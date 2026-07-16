"""Proximity analysis helpers over :class:`MDMDistanceResult` objects."""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import silhouette_score

from .types import MDMDistanceResult

_KASS_RAFTERY_LOG_BF = {
    "weak": 2.0,
    "positive": 6.0,
    "strong": 10.0,
    "very_strong": 15.0,
}


def nearest_neighbours(
    dist: MDMDistanceResult,
    subject: Union[int, str, Any],
    k: int = 5,
) -> List[Tuple[Any, float]]:
    """
    Return the ``k`` nearest neighbours of ``subject`` as ``(id, distance)`` pairs.

    ``subject`` may be an index or a value in ``dist.subject_ids``.
    """
    if isinstance(subject, int):
        idx = subject
    else:
        try:
            idx = dist.subject_ids.index(subject)
        except ValueError as exc:
            raise ValueError(f"subject {subject!r} not found in subject_ids") from exc

    d_row = dist.matrix[idx].copy()
    d_row[idx] = np.inf
    order = np.argsort(d_row)
    k = min(k, len(order))
    return [(dist.subject_ids[i], float(d_row[i])) for i in order[:k]]


def silhouette(dist: MDMDistanceResult, labels: Sequence[int]) -> float:
    """Cluster-separation quality for a label assignment (precomputed metric)."""
    labels_arr = np.asarray(labels)
    if len(np.unique(labels_arr)) < 2:
        raise ValueError("silhouette requires at least 2 clusters")
    return float(
        silhouette_score(dist.matrix, labels_arr, metric="precomputed")
    )


def suggest_clusters(
    dist: MDMDistanceResult,
    criterion: Literal["silhouette"] = "silhouette",
    max_clusters: Optional[int] = None,
    linkage_method: str = "average",
) -> np.ndarray:
    """
    Pick a dendrogram cut that maximises silhouette (default).

    Returns flat cluster labels.
    """
    if criterion != "silhouette":
        raise ValueError(f"Unknown criterion: {criterion!r}")

    s = len(dist.subject_ids)
    if s < 3:
        return dist.cluster_labels(2, linkage_method=linkage_method)

    max_k = max_clusters if max_clusters is not None else min(s - 1, 10)
    max_k = max(2, max_k)

    best_labels: Optional[np.ndarray] = None
    best_score = -np.inf
    for k in range(2, max_k + 1):
        labels = dist.cluster_labels(k, linkage_method=linkage_method)
        try:
            sc = silhouette(dist, labels)
        except ValueError:
            continue
        if sc > best_score:
            best_score = sc
            best_labels = labels

    if best_labels is None:
        return dist.cluster_labels(2, linkage_method=linkage_method)
    return best_labels


def bayes_factor_cut(
    dist: MDMDistanceResult,
    strength: Literal["weak", "positive", "strong", "very_strong"] = "strong",
    linkage_method: str = "average",
) -> np.ndarray:
    """
    Cut the dendrogram at a log-Bayes-factor threshold (Kass–Raftery scale).

    Because ``d(i,j)`` is a log Bayes factor, larger merge heights indicate
    weaker evidence of separation.
    """
    if strength not in _KASS_RAFTERY_LOG_BF:
        raise ValueError(
            f"Unknown strength {strength!r}; choose from {list(_KASS_RAFTERY_LOG_BF)}"
        )
    threshold = _KASS_RAFTERY_LOG_BF[strength]
    from scipy.cluster.hierarchy import fcluster

    z = dist.to_linkage(linkage_method)
    return fcluster(z, t=threshold, criterion="distance")
