"""Multidimensional projection and dendrogram plots for MDM distance matrices."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from scipy.sparse import issparse, spmatrix

from ..group_analysis.distance.types import MDMDistanceResult
from ._style import OKABE_ITO

_DENSE_ONLY = frozenset({"mds", "nmds"})
_SPARSE_OK = frozenset({"tsne", "isomap", "umap"})

# Mixed/root dendrogram links (not a cluster) — avoid C0 so it cannot collide.
MIXED_CLUSTER_COLOR = OKABE_ITO["gray"]


def _label_key(lab: Any) -> Any:
    """Hashable cluster-id key (unwrap NumPy scalars)."""
    if isinstance(lab, np.generic):
        return lab.item()
    return lab


def _cluster_color_map(labels: np.ndarray) -> dict[Any, str]:
    """Map each unique cluster id to a stable color from the matplotlib cycle."""
    import matplotlib.pyplot as plt

    unique = [_label_key(lab) for lab in np.unique(labels)]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {lab: cycle[i % len(cycle)] for i, lab in enumerate(unique)}


def _subtree_cluster(
    node: int,
    n: int,
    z: np.ndarray,
    labels: np.ndarray,
    cache: dict[int, Any],
) -> Any:
    """Cluster id shared by all leaves under ``node``, or None if mixed."""
    if node in cache:
        return cache[node]
    if node < n:
        cache[node] = _label_key(labels[node])
        return cache[node]
    merge = z[node - n]
    left = _subtree_cluster(int(merge[0]), n, z, labels, cache)
    right = _subtree_cluster(int(merge[1]), n, z, labels, cache)
    cache[node] = left if left == right else None
    return cache[node]


def _dendrogram_merge_order(z: np.ndarray) -> list[int]:
    """Post-order merge node ids matching SciPy ``dendrogram`` ``icoord`` order."""
    n = int(z.shape[0]) + 1
    order: list[int] = []

    def visit(i: int) -> None:
        if i < n:
            return
        visit(int(z[i - n, 0]))
        visit(int(z[i - n, 1]))
        order.append(i)

    visit(2 * n - 2)
    return order


def _color_for_cluster(
    cluster: Any,
    color_map: dict[Any, str],
    mixed: str = MIXED_CLUSTER_COLOR,
) -> str:
    if cluster is None:
        return mixed
    return color_map[cluster]


def _draw_cluster_colored_links(
    ax: Any,
    dendro: dict,
    z: np.ndarray,
    labels: np.ndarray,
    color_map: dict[Any, str],
) -> None:
    """Draw each U-link as three segments: pure-cluster arms keep cluster color."""
    from matplotlib.collections import LineCollection

    for coll in list(ax.collections):
        if isinstance(coll, LineCollection):
            coll.remove()

    n = len(labels)
    cache: dict[int, Any] = {}
    merge_order = _dendrogram_merge_order(z)
    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    if len(icoord) != len(merge_order):
        raise RuntimeError("dendrogram layout does not match linkage merge order")

    lw = 1.5
    for xs, ys, node in zip(icoord, dcoord, merge_order):
        left = int(z[node - n, 0])
        right = int(z[node - n, 1])
        c_left = _subtree_cluster(left, n, z, labels, cache)
        c_right = _subtree_cluster(right, n, z, labels, cache)
        col_left = _color_for_cluster(c_left, color_map)
        col_right = _color_for_cluster(c_right, color_map)
        col_bar = (
            col_left
            if (c_left is not None and c_left == c_right)
            else MIXED_CLUSTER_COLOR
        )
        ax.plot(
            [xs[0], xs[1]],
            [ys[0], ys[1]],
            color=col_left,
            lw=lw,
            solid_capstyle="round",
        )
        ax.plot(
            [xs[1], xs[2]],
            [ys[1], ys[2]],
            color=col_bar,
            lw=lw,
            solid_capstyle="round",
        )
        ax.plot(
            [xs[2], xs[3]],
            [ys[2], ys[3]],
            color=col_right,
            lw=lw,
            solid_capstyle="round",
        )


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
        perplexity = kwargs.pop("perplexity", min(30.0, max(2.0, (n - 1) / 3.0)))
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
        color_map = _cluster_color_map(labels_arr)
        for lab in np.unique(labels_arr):
            m = labels_arr == lab
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=80,
                color=color_map[_label_key(lab)],
                label=f"cluster {lab}",
            )
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
    n_clusters: Optional[int] = None,
    labels: Optional[Sequence[Any]] = None,
    leaf_rotation: float = 45.0,
    ax: Optional[Any] = None,
) -> Any:
    """Hierarchical-clustering dendrogram of the subject dissimilarity matrix.

    Pass ``n_clusters`` or ``labels`` to color links (and leaf ticks) by the
    same cluster ids used in :func:`plot_projection`. Mixed / root joins use a
    neutral gray. Without either, SciPy's default ``color_threshold`` coloring
    is kept.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    z = dist.to_linkage(linkage_method)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    cluster_labels = labels
    if cluster_labels is None and n_clusters is not None:
        cluster_labels = dist.cluster_labels(n_clusters, linkage_method)

    dendro_kw: dict[str, Any] = {
        "labels": [str(s) for s in dist.subject_ids],
        "leaf_rotation": leaf_rotation,
        "leaf_font_size": 9,
        "ax": ax,
    }
    color_map: dict[Any, str] | None = None
    cluster_arr: np.ndarray | None = None
    if cluster_labels is not None:
        cluster_arr = np.asarray(cluster_labels)
        color_map = _cluster_color_map(cluster_arr)
        dendro_kw["above_threshold_color"] = MIXED_CLUSTER_COLOR
    else:
        dendro_kw["color_threshold"] = color_threshold

    dendro = dendrogram(z, **dendro_kw)
    if color_map is not None and cluster_arr is not None:
        _draw_cluster_colored_links(ax, dendro, z, cluster_arr, color_map)
    # Align rotated tick labels so long category names do not overlap.
    leaves = dendro.get("leaves") or list(range(len(dist.subject_ids)))
    for tick, leaf_idx in zip(ax.get_xticklabels(), leaves):
        tick.set_rotation(leaf_rotation)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        if color_map is not None and cluster_arr is not None:
            tick.set_color(color_map[_label_key(cluster_arr[leaf_idx])])
    ax.set_title(f"MDM subject dendrogram ({linkage_method} linkage)")
    ax.set_ylabel("separation d(i, j)")
    return ax


def plot_mdp(
    dist: MDMDistanceResult,
    *,
    technique: Literal["mds", "nmds", "tsne", "isomap", "umap"] = "mds",
    n_clusters: int = 2,
    linkage_method: str = "average",
    random_state: int = 0,
    show_legend: bool = False,
    figsize: Optional[tuple] = None,
) -> Any:
    """Side-by-side projection scatter and dendrogram (MDP / Compute GS results view)."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (13, 5.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cluster_labels = dist.cluster_labels(n_clusters, linkage_method)
    plot_projection(
        dist,
        technique=technique,
        labels=cluster_labels,
        ax=ax1,
        random_state=random_state,
        show_legend=show_legend,
    )
    plot_dendrogram(
        dist,
        linkage_method=linkage_method,
        labels=cluster_labels,
        ax=ax2,
    )
    fig.tight_layout()
    return fig
