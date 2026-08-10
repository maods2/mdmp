"""Multidimensional projection and dendrogram plots for MDM distance matrices."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from scipy.sparse import issparse, spmatrix

from ..group_analysis.distance.types import MDMDistanceResult

_DENSE_ONLY = frozenset({"mds", "nmds"})
_SPARSE_OK = frozenset({"tsne", "isomap", "umap"})


def _maybe_add_normalized_stress(mds_kw: dict) -> None:
    """Use ``normalized_stress`` when supported (scikit-learn >= 1.2)."""
    import inspect

    from sklearn.manifold import MDS

    params = inspect.signature(MDS.__init__).parameters
    if "normalized_stress" in params:
        mds_kw.setdefault("normalized_stress", "auto")
    if "n_init" in params:
        mds_kw.setdefault("n_init", 4)


def _maybe_set_tsne_learning_rate(tsne_kw: dict) -> None:
    """Avoid FutureWarning on scikit-learn >= 1.2."""
    import inspect

    from sklearn.manifold import TSNE

    if (
        "learning_rate" in inspect.signature(TSNE.__init__).parameters
        and "learning_rate" not in tsne_kw
    ):
        tsne_kw["learning_rate"] = "auto"


def _check_sparse_guard(technique: str, dist_input: Any) -> None:
    is_sparse_input = issparse(dist_input)
    if technique in _DENSE_ONLY and is_sparse_input:
        raise ValueError(
            f"technique={technique!r} requires a dense distance matrix; sparse inputs "
            "silently treat absent pairs as distance zero. Use a dense MDMDistanceResult "
            "or pass technique='tsne'/'isomap'/'umap' for neighbour-graph projectors."
        )
    if technique in _SPARSE_OK and not is_sparse_input and technique == "umap":
        pass  # umap accepts dense precomputed too


def project_distance(
    dist: Union[MDMDistanceResult, np.ndarray, spmatrix],
    *,
    technique: Literal["mds", "nmds", "tsne", "isomap", "umap"] = "mds",
    n_components: int = 2,
    random_state: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Embed a precomputed dissimilarity matrix into R^{n_components}.

    Returns an (S x n_components) coordinate array.

    Metric MDS / non-metric MDS require dense input; t-SNE, Isomap, and UMAP
    accept sparse neighbour graphs from :meth:`MDMDistanceResult.to_sparse`.
    """
    _check_sparse_guard(technique, dist)

    if issparse(dist):
        d = dist.toarray()
    elif isinstance(dist, MDMDistanceResult):
        d = dist.matrix
    else:
        d = np.asarray(dist, dtype=float)

    if technique == "mds":
        from sklearn.manifold import MDS

        mds_kw = dict(
            n_components=n_components,
            dissimilarity="precomputed",
            metric=True,
            random_state=random_state,
        )
        mds_kw.update(kwargs)
        _maybe_add_normalized_stress(mds_kw)
        model = MDS(**mds_kw)
        return model.fit_transform(d)

    if technique == "nmds":
        from sklearn.manifold import MDS

        mds_kw = dict(
            n_components=n_components,
            dissimilarity="precomputed",
            metric=False,
            random_state=random_state,
        )
        mds_kw.update(kwargs)
        _maybe_add_normalized_stress(mds_kw)
        model = MDS(**mds_kw)
        return model.fit_transform(d)

    if technique == "tsne":
        from sklearn.manifold import TSNE

        n = d.shape[0]
        perplexity = kwargs.pop(
            "perplexity", min(30.0, max(2.0, (n - 1) / 3.0))
        )
        tsne_kw = dict(
            n_components=n_components,
            metric="precomputed",
            init="random",
            perplexity=perplexity,
            random_state=random_state,
        )
        tsne_kw.update(kwargs)
        _maybe_set_tsne_learning_rate(tsne_kw)
        model = TSNE(**tsne_kw)
        return model.fit_transform(d)

    if technique == "isomap":
        from sklearn.manifold import Isomap

        n_neighbors = kwargs.pop("n_neighbors", min(5, d.shape[0] - 1))
        model = Isomap(
            n_components=n_components,
            metric="precomputed",
            n_neighbors=n_neighbors,
            **kwargs,
        )
        return model.fit_transform(d)

    if technique == "umap":
        try:
            import umap as umap_mod
        except ImportError as exc:
            raise ImportError(
                "technique='umap' requires the optional 'umap-learn' package. "
                "Install with: pip install 'mdmp[umap]'"
            ) from exc
        model = umap_mod.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            **kwargs,
        )
        return model.fit_transform(d)

    raise ValueError(f"Unknown technique: {technique!r}")


def plot_projection(
    dist: Union[MDMDistanceResult, np.ndarray, spmatrix],
    *,
    technique: Literal["mds", "nmds", "tsne", "isomap", "umap"] = "mds",
    labels: Optional[Sequence[Any]] = None,
    n_clusters: Optional[int] = None,
    subject_ids: Optional[Sequence[Any]] = None,
    ax: Optional[Any] = None,
    random_state: int = 0,
    show_legend: bool = False,
    **kwargs: Any,
) -> Any:
    """Scatter plot of a 2D projection, coloured by cluster or supplied labels."""
    import matplotlib.pyplot as plt

    coords = project_distance(
        dist,
        technique=technique,
        n_components=2,
        random_state=random_state,
        **kwargs,
    )

    if labels is None and n_clusters is not None and isinstance(dist, MDMDistanceResult):
        labels = dist.cluster_labels(n_clusters)
    if subject_ids is None:
        if isinstance(dist, MDMDistanceResult):
            subject_ids = dist.subject_ids
        else:
            subject_ids = list(range(coords.shape[0]))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=80)
    else:
        labels_arr = np.asarray(labels)
        for lab in np.unique(labels_arr):
            m = labels_arr == lab
            ax.scatter(coords[m, 0], coords[m, 1], s=80, label=f"cluster {lab}")
        if show_legend:
            ax.legend(frameon=False, fontsize=8)

    for k, sid in enumerate(subject_ids):
        ax.annotate(
            str(sid),
            (coords[k, 0], coords[k, 1]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_title(f"MDM subject projection ({technique.upper()})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    return ax


def plot_dendrogram(
    dist: MDMDistanceResult,
    *,
    linkage_method: str = "average",
    color_threshold: Optional[float] = None,
    leaf_rotation: float = 45.0,
    ax: Optional[Any] = None,
) -> Any:
    """Hierarchical-clustering dendrogram of the subject dissimilarity matrix."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    z = dist.to_linkage(linkage_method)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    dendrogram(
        z,
        labels=[str(s) for s in dist.subject_ids],
        color_threshold=color_threshold,
        leaf_rotation=leaf_rotation,
        leaf_font_size=9,
        ax=ax,
    )
    # Align rotated tick labels so long category names do not overlap.
    for label in ax.get_xticklabels():
        label.set_rotation(leaf_rotation)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.set_title(f"MDM subject dendrogram ({linkage_method} linkage)")
    ax.set_ylabel("separation d(i, j)")
    return ax


def plot_group_embedding(
    dist: MDMDistanceResult,
    *,
    technique: Literal["mds", "nmds", "tsne", "isomap", "umap"] = "mds",
    n_clusters: int = 2,
    linkage_method: str = "average",
    random_state: int = 0,
    show_legend: bool = False,
    figsize: Optional[tuple] = None,
) -> Any:
    """Side-by-side projection scatter and dendrogram (Paper #2 style)."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (13, 5.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_projection(
        dist,
        technique=technique,
        n_clusters=n_clusters,
        ax=ax1,
        random_state=random_state,
        show_legend=show_legend,
    )
    plot_dendrogram(dist, linkage_method=linkage_method, ax=ax2)
    fig.tight_layout()
    return fig
