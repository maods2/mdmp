"""Tests for proximity analysis helpers over MDMDistanceResult."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from mdmp.group_analysis.distance import (
    bayes_factor_cut,
    compute_mdm_distance,
    nearest_neighbours,
    silhouette,
    suggest_clusters,
)


def _two_group_cohort() -> Tuple[List[np.ndarray], List[int]]:
    rng = np.random.default_rng(7)
    subjects = []
    groups = []

    def gen(kind: str) -> np.ndarray:
        e = rng.normal(size=(80, 3))
        x = np.zeros((80, 3))
        x[:, 0] = e[:, 0]
        if kind == "A":
            x[:, 1] = 0.8 * x[:, 0] + 0.4 * e[:, 1]
            x[:, 2] = 0.8 * x[:, 1] + 0.4 * e[:, 2]
        else:
            x[:, 2] = 0.8 * x[:, 0] + 0.4 * e[:, 2]
            x[:, 1] = 0.8 * x[:, 2] + 0.4 * e[:, 1]
        return x

    for _ in range(2):
        subjects.append(gen("A"))
        groups.append(0)
    for _ in range(2):
        subjects.append(gen("B"))
        groups.append(1)
    return subjects, groups


@pytest.fixture
def dist_result():
    subjects, _ = _two_group_cohort()
    ids = [f"S{i}" for i in range(len(subjects))]
    return compute_mdm_distance(subjects, subject_ids=ids, nbf=10, verbose=False)


def test_nearest_neighbours(dist_result):
    nn = nearest_neighbours(dist_result, "S0", k=2)
    assert len(nn) == 2
    assert all(isinstance(x, tuple) and len(x) == 2 for x in nn)


def test_silhouette(dist_result):
    labels = dist_result.cluster_labels(2)
    sc = silhouette(dist_result, labels)
    assert -1.0 <= sc <= 1.0


def test_suggest_clusters(dist_result):
    labels = suggest_clusters(dist_result)
    assert labels.shape == (len(dist_result.subject_ids),)


def test_bayes_factor_cut(dist_result):
    labels = bayes_factor_cut(dist_result, strength="strong")
    assert labels.shape == (len(dist_result.subject_ids),)
    assert len(np.unique(labels)) >= 1
