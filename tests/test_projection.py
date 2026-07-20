"""Tests for multidimensional projection and dendrogram plots."""

from __future__ import annotations

import importlib.util
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mdmp import compute_mdm_distance, project_distance
from mdmp.plotting import plot_dendrogram, plot_group_embedding, plot_projection


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
    return compute_mdm_distance(_small_cohort(), nbf=10, verbose=False)


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


def test_plot_group_embedding_returns_figure(dist_result):
    fig = plot_group_embedding(dist_result, technique="mds", n_clusters=2)
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
