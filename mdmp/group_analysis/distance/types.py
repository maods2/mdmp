"""Type definitions for MDM pairwise distance results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix

from .sparse import sparsify_distance_matrix


@dataclass
class MDMDistanceResult:
    """
    Pairwise MDM dissimilarity matrix between subjects.

    Attributes
    ----------
    condensed : np.ndarray
        Upper-triangular distance vector (``scipy.spatial.distance.squareform`` layout).
    subject_ids : list
        Row/column labels matching the square matrix.
    metric : str
        Name of the pairwise metric that produced the matrix.
    method : str
        Structure-learning method used for individual/common DAGs.
    individuals : list
        Per-subject fitted :class:`mdmp.model.MDM` objects (stage 1 artifact).
    metadata : dict
        Extra info (discount factors, self-LPL, common_structure, nbf, ...).
    """

    condensed: np.ndarray
    subject_ids: List[Any]
    metric: str
    method: str
    individuals: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def matrix(self) -> np.ndarray:
        """Dense symmetric (S x S) distance matrix with zero diagonal."""
        return squareform(self.condensed)

    def to_similarity(self, scale: Optional[float] = None) -> np.ndarray:
        """
        Convert distances to similarities via ``exp(-d / scale)``.

        Default ``scale`` is the median of off-diagonal distances.
        """
        d = self.matrix
        if scale is None:
            off = d[np.triu_indices(d.shape[0], k=1)]
            scale = float(np.median(off)) if off.size else 1.0
            if scale == 0.0:
                scale = 1.0
        s = np.exp(-d / scale)
        np.fill_diagonal(s, 1.0)
        return s

    def to_frame(self) -> pd.DataFrame:
        """Labelled S x S distance matrix for inspection or CSV export."""
        ids = [str(x) for x in self.subject_ids]
        return pd.DataFrame(self.matrix, index=ids, columns=ids)

    def to_sparse(
        self,
        *,
        knn: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> csr_matrix:
        """Return a sparse neighbourhood dissimilarity matrix (see ``sparse.py``)."""
        return sparsify_distance_matrix(self.matrix, knn=knn, threshold=threshold)

    def to_linkage(self, linkage_method: str = "average") -> np.ndarray:
        """SciPy linkage matrix for dendrograms / cluster cutting."""
        return linkage(self.condensed, method=linkage_method)

    def cluster_labels(self, n_clusters: int, linkage_method: str = "average") -> np.ndarray:
        """Flat cluster assignment by cutting the dendrogram into ``n_clusters``."""
        z = self.to_linkage(linkage_method)
        return fcluster(z, t=n_clusters, criterion="maxclust")
