"""
Pairwise MDM distance matrix (Group Structure method) and proximity analysis.

Stages of the individual-comparison workflow:

1. :func:`fit_individual_structures` — per-subject MDM estimates
2–3. :func:`compute_mdm_distance` — pairwise dissimilarity matrix
5. :mod:`mdmp.group_analysis.distance.analysis` — proximity helpers
"""

from .analysis import (
    bayes_factor_cut,
    nearest_neighbours,
    silhouette,
    suggest_clusters,
)
from .estimation import fit_individual_structures
from .metrics import METRIC_REGISTRY
from .separation import compute_mdm_distance
from .types import MDMDistanceResult

__all__ = [
    "MDMDistanceResult",
    "fit_individual_structures",
    "compute_mdm_distance",
    "METRIC_REGISTRY",
    "nearest_neighbours",
    "silhouette",
    "suggest_clusters",
    "bayes_factor_cut",
]
