"""Tests for multidimensional projection and dendrogram plots."""

from __future__ import annotations

import importlib.util
from typing import List

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from scipy.spatial.distance import squareform

from mdmp.group_analysis import compute_gs
from mdmp.group_analysis.distance.types import MDMDistanceResult
from mdmp.plotting import plot_dendrogram, plot_mdp, plot_projection, project_distance
from mdmp.plotting.projection import (
    MIXED_CLUSTER_COLOR,
    _cluster_color_map,
    _label_key,
)


def _small_cohort() -> List[np.ndarray]:
    rng = np.random.default_rng(3)
    subjects = []
    for _ in range(4):
        e = rng.normal(size=(70, 3))
        x = np.zeros((70, 3))
        x[:, 0] = e[:, 0]
        x[:, 1] = 0.6 * x[:, 0] + e[:, 1]
        x[:, 2] = e[:, 2]
        subjects.append(x)
    return subjects


@pytest.fixture
def dist_result():
    return compute_gs(_small_cohort(), nbf=10, verbose=False)


@pytest.mark.parametrize("technique", ["mds", "nmds", "tsne", "isomap"])
def test_project_distance_shape(dist_result, technique):
    coords = project_distance(dist_result, technique=technique, random_state=0)
    assert coords.shape == (4, 2)


def test_sparse_guard_mds_raises(dist_result):
    sparse = dist_result.to_sparse(knn=2)
    with pytest.raises(ValueError, match="requires a dense"):
        project_distance(sparse, technique="mds")
    with pytest.raises(ValueError, match="requires a dense"):
        project_distance(sparse, technique="nmds")


def test_sparse_tsne_isomap_work(dist_result):
    sparse = dist_result.to_sparse(knn=2)
    coords_tsne = project_distance(sparse, technique="tsne", random_state=0)
    assert coords_tsne.shape == (4, 2)
    coords_iso = project_distance(sparse, technique="isomap")
    assert coords_iso.shape == (4, 2)


def test_plot_projection_returns_axes(dist_result):
    ax = plot_projection(dist_result, technique="mds", random_state=0)
    assert isinstance(ax, Axes)


def test_plot_dendrogram_returns_axes(dist_result):
    ax = plot_dendrogram(dist_result)
    assert isinstance(ax, Axes)


def test_plot_mdp_returns_figure(dist_result):
    fig = plot_mdp(dist_result, technique="mds", n_clusters=2)
    assert isinstance(fig, Figure)


def test_plot_projection_external_labels(dist_result):
    labels = [1, 1, 2, 2]
    ax = plot_projection(dist_result, labels=labels, technique="mds", random_state=0)
    assert isinstance(ax, Axes)


def test_umap_import_error_when_missing(dist_result):
    if importlib.util.find_spec("umap") is not None:
        pytest.skip("umap-learn is installed")
    with pytest.raises(ImportError, match="umap"):
        project_distance(dist_result, technique="umap")


def _two_cluster_dist() -> MDMDistanceResult:
    """Two tight pairs so both clusters have below-root dendrogram links."""
    mat = np.array(
        [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ]
    )
    return MDMDistanceResult(
        condensed=squareform(mat),
        subject_ids=["A", "B", "C", "D"],
        metric="test",
        method="hc",
    )


def test_cluster_color_map_stable_by_id():
    import matplotlib.pyplot as plt

    labels = np.array([2, 1, 2, 1])
    color_map = _cluster_color_map(labels)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert color_map[1] == cycle[0]
    assert color_map[2] == cycle[1]


def test_cluster_color_map_wraps_cycle():
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n = len(cycle) + 3
    labels = np.arange(1, n + 1)
    color_map = _cluster_color_map(labels)
    assert color_map[1] == cycle[0]
    assert color_map[n] == cycle[(n - 1) % len(cycle)]


def _line_rgbs(ax) -> list:
    return [to_rgba(line.get_color())[:3] for line in ax.get_lines()]


def test_plot_mdp_cluster_colors_aligned():
    dist = _two_cluster_dist()
    labels = dist.cluster_labels(2)
    color_map = _cluster_color_map(np.asarray(labels))
    fig = plot_mdp(dist, technique="mds", n_clusters=2, random_state=0)
    ax_scatter, ax_dendo = fig.axes

    unique = [_label_key(lab) for lab in np.unique(labels)]
    scatter_colls = [c for c in ax_scatter.collections if len(np.asarray(c.get_offsets())) > 0]
    scatter_by_cluster = {}
    for coll, lab in zip(scatter_colls, unique):
        scatter_by_cluster[lab] = to_rgba(coll.get_facecolor()[0])

    for tick in ax_dendo.get_xticklabels():
        sid = tick.get_text()
        idx = list(dist.subject_ids).index(sid)
        lab = _label_key(labels[idx])
        expected = to_rgba(color_map[lab])
        np.testing.assert_allclose(to_rgba(tick.get_color())[:3], expected[:3])
        np.testing.assert_allclose(scatter_by_cluster[lab][:3], expected[:3])

    mixed = to_rgba(MIXED_CLUSTER_COLOR)[:3]
    line_rgbs = _line_rgbs(ax_dendo)
    assert any(np.allclose(rgb, mixed) for rgb in line_rgbs)
    for color in color_map.values():
        cluster_rgb = to_rgba(color)[:3]
        assert not np.allclose(cluster_rgb, mixed)
        assert any(np.allclose(rgb, cluster_rgb) for rgb in line_rgbs)


def test_plot_dendrogram_n_clusters_colors_leaves():
    dist = _two_cluster_dist()
    labels = dist.cluster_labels(2)
    color_map = _cluster_color_map(np.asarray(labels))
    ax = plot_dendrogram(dist, n_clusters=2)
    for tick in ax.get_xticklabels():
        sid = tick.get_text()
        idx = list(dist.subject_ids).index(sid)
        lab = _label_key(labels[idx])
        np.testing.assert_allclose(
            to_rgba(tick.get_color())[:3], to_rgba(color_map[lab])[:3]
        )


def _singleton_cluster_dist() -> MDMDistanceResult:
    """README-shaped tree: {0,1,2} vs singleton 3."""
    mat = np.array(
        [
            [0.00, 0.65, 0.23, 0.99],
            [0.65, 0.00, 0.65, 0.99],
            [0.23, 0.65, 0.00, 0.99],
            [0.99, 0.99, 0.99, 0.00],
        ]
    )
    return MDMDistanceResult(
        condensed=squareform(mat),
        subject_ids=[0, 1, 2, 3],
        metric="test",
        method="hc",
    )


def test_plot_dendrogram_singleton_arm_matches_label():
    dist = _singleton_cluster_dist()
    labels = dist.cluster_labels(2)
    color_map = _cluster_color_map(np.asarray(labels))
    singleton_lab = _label_key(labels[3])
    orange = to_rgba(color_map[singleton_lab])[:3]
    mixed = to_rgba(MIXED_CLUSTER_COLOR)[:3]

    ax = plot_dendrogram(dist, n_clusters=2)
    tick_x = None
    for tick in ax.get_xticklabels():
        if tick.get_text() == "3":
            tick_x = tick.get_position()[0]
            np.testing.assert_allclose(to_rgba(tick.get_color())[:3], orange)
    assert tick_x is not None

    found_arm = False
    found_mixed_bar = False
    for line in ax.get_lines():
        xd, yd = line.get_data()
        if len(xd) != 2:
            continue
        rgb = to_rgba(line.get_color())[:3]
        vertical = np.allclose(xd[0], xd[1], atol=1e-6)
        from_leaf = min(yd) <= 1e-8
        if vertical and from_leaf and np.allclose(xd[0], tick_x, atol=1.5):
            np.testing.assert_allclose(rgb, orange)
            found_arm = True
        horizontal = np.allclose(yd[0], yd[1], atol=1e-6)
        if horizontal and max(yd) > 0.9:
            np.testing.assert_allclose(rgb, mixed)
            found_mixed_bar = True
    assert found_arm
    assert found_mixed_bar


def _three_cluster_dist() -> MDMDistanceResult:
    n = 6
    mat = np.ones((n, n))
    np.fill_diagonal(mat, 0.0)
    for i, j in ((0, 1), (2, 3), (4, 5)):
        mat[i, j] = mat[j, i] = 0.1
    return MDMDistanceResult(
        condensed=squareform(mat),
        subject_ids=list(range(n)),
        metric="test",
        method="hc",
    )


def test_plot_mdp_three_clusters_colors_aligned():
    import matplotlib.pyplot as plt

    dist = _three_cluster_dist()
    labels = dist.cluster_labels(3)
    assert len(np.unique(labels)) == 3
    color_map = _cluster_color_map(np.asarray(labels))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    unique = [_label_key(lab) for lab in np.unique(labels)]
    for i, lab in enumerate(unique):
        assert color_map[lab] == cycle[i]

    fig = plot_mdp(dist, technique="mds", n_clusters=3, random_state=0)
    ax_scatter, ax_dendo = fig.axes
    scatter_colls = [c for c in ax_scatter.collections if len(np.asarray(c.get_offsets())) > 0]
    scatter_by_cluster = {}
    for coll, lab in zip(scatter_colls, unique):
        scatter_by_cluster[lab] = to_rgba(coll.get_facecolor()[0])

    for tick in ax_dendo.get_xticklabels():
        sid = int(tick.get_text())
        lab = _label_key(labels[sid])
        expected = to_rgba(color_map[lab])[:3]
        np.testing.assert_allclose(to_rgba(tick.get_color())[:3], expected)
        np.testing.assert_allclose(scatter_by_cluster[lab][:3], expected)

    line_rgbs = _line_rgbs(ax_dendo)
    for lab in unique:
        cluster_rgb = to_rgba(color_map[lab])[:3]
        assert any(np.allclose(rgb, cluster_rgb) for rgb in line_rgbs)
