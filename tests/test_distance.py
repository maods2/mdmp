"""Tests for MDM pairwise distance matrix (group-structure method)."""

from __future__ import annotations

import inspect
from typing import List, Tuple

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from mdmp import MDM, compute_mdm_distance, fit_individual_structures
from mdmp.group_analysis.distance import MDMDistanceResult
from mdmp.scoring import compute_logpl
from mdmp.utils import get_default_delta


def _make_synthetic_cohort(
    n_per_group: int = 2,
    n_nodes: int = 3,
    n_time: int = 80,
    seed: int = 7,
) -> Tuple[List[np.ndarray], List[int]]:
    """Two latent groups with different causal wiring."""
    rng = np.random.default_rng(seed)
    subjects: List[np.ndarray] = []
    groups: List[int] = []

    def gen(kind: str) -> np.ndarray:
        e = rng.normal(0, 1, size=(n_time, n_nodes))
        x = np.zeros((n_time, n_nodes))
        x[:, 0] = e[:, 0]
        if kind == "A":
            x[:, 1] = 0.8 * x[:, 0] + 0.4 * e[:, 1]
            x[:, 2] = 0.8 * x[:, 1] + 0.4 * e[:, 2]
        else:
            x[:, 2] = 0.8 * x[:, 0] + 0.4 * e[:, 2]
            x[:, 1] = 0.8 * x[:, 2] + 0.4 * e[:, 1]
        return x

    for _ in range(n_per_group):
        subjects.append(gen("A"))
        groups.append(0)
    for _ in range(n_per_group):
        subjects.append(gen("B"))
        groups.append(1)
    return subjects, groups


@pytest.fixture
def three_subject_cohort() -> List[np.ndarray]:
    rng = np.random.default_rng(1)
    out = []
    for _ in range(3):
        e = rng.normal(size=(60, 3))
        x = np.zeros((60, 3))
        x[:, 0] = e[:, 0]
        x[:, 1] = 0.5 * x[:, 0] + e[:, 1]
        x[:, 2] = e[:, 2]
        out.append(x)
    return out


@pytest.fixture
def two_group_cohort() -> Tuple[List[np.ndarray], List[int]]:
    return _make_synthetic_cohort(n_per_group=2, n_time=80, n_nodes=3)


def test_basic_distance_properties(three_subject_cohort):
    dist = compute_mdm_distance(
        three_subject_cohort, method="hc", nbf=10, verbose=False
    )
    d = dist.matrix
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0)
    np.testing.assert_allclose(d, d.T)
    assert np.all(d >= -1e-9)


def test_two_group_separation(two_group_cohort):
    subjects, groups = two_group_cohort
    dist = compute_mdm_distance(subjects, method="hc", nbf=10, verbose=False)
    d = dist.matrix
    within = []
    across = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            if groups[i] == groups[j]:
                within.append(d[i, j])
            else:
                across.append(d[i, j])
    assert max(within) <= max(across) + 1e-6 or np.mean(within) <= np.mean(across)
    labels = dist.cluster_labels(2)
    assert len(labels) == len(subjects)


def test_condensed_roundtrip(three_subject_cohort):
    dist = compute_mdm_distance(three_subject_cohort, nbf=10, verbose=False)
    assert len(dist.condensed) == 3 * 2 // 2
    np.testing.assert_allclose(squareform(dist.condensed), dist.matrix)


def test_to_sparse_knn(three_subject_cohort):
    dist = compute_mdm_distance(three_subject_cohort, nbf=10, verbose=False)
    k = 2
    sparse = dist.to_sparse(knn=k)
    assert sparse.nnz <= 3 * k * 2  # symmetrised upper bound
    assert sparse.diagonal().sum() == 0.0


def test_delta_clip_self_distance_finite(three_subject_cohort):
    """Regression: delta grid values slightly above 1 must not yield inf self-LPL."""
    delta_grid = get_default_delta()
    dist = compute_mdm_distance(
        three_subject_cohort, delta_grid=delta_grid, nbf=10, verbose=False
    )
    for lpl in dist.metadata["self_lpl"]:
        assert np.isfinite(lpl)


def test_fit_individual_structures_returns_mdms(two_group_cohort):
    subjects, _ = two_group_cohort
    inds = fit_individual_structures(subjects, method="hc", nbf=10, verbose=False)
    assert len(inds) == len(subjects)
    assert all(isinstance(m, MDM) for m in inds)


def test_prefitted_mdms_inherit_subject_ids(two_group_cohort):
    subjects, _ = two_group_cohort
    ids = ["S0", "S1", "S2", "S3"]
    inds = fit_individual_structures(subjects, subject_ids=ids, nbf=10, verbose=False)
    dist = compute_mdm_distance(inds, nbf=10, verbose=False)
    assert dist.subject_ids == ids


def test_prefitted_mdms_same_as_raw(two_group_cohort):
    subjects, _ = two_group_cohort
    inds = fit_individual_structures(subjects, method="hc", nbf=10, verbose=False)
    d_raw = compute_mdm_distance(subjects, nbf=10, verbose=False)
    d_prefit = compute_mdm_distance(inds, nbf=10, verbose=False)
    np.testing.assert_allclose(d_raw.matrix, d_prefit.matrix, rtol=1e-5, atol=1e-5)


def test_result_individuals_populated(two_group_cohort):
    subjects, _ = two_group_cohort
    dist = compute_mdm_distance(subjects, nbf=10, verbose=False)
    assert len(dist.individuals) == len(subjects)
    assert all(isinstance(m, MDM) for m in dist.individuals)


def test_structural_hamming_metric(two_group_cohort):
    subjects, _ = two_group_cohort
    dist = compute_mdm_distance(
        subjects, metric="structural_hamming", nbf=10, verbose=False
    )
    assert dist.metric == "structural_hamming"
    assert np.all(dist.matrix >= 0)


def test_custom_metric_callable(two_group_cohort):
    subjects, _ = two_group_cohort

    def always_one(m_i, m_j, *, ctx=None):
        del m_i, m_j, ctx
        return 1.0

    dist = compute_mdm_distance(subjects, metric=always_one, nbf=10, verbose=False)
    off = dist.matrix[np.triu_indices(dist.matrix.shape[0], k=1)]
    np.testing.assert_allclose(off, 1.0)


def test_strength_frobenius_same_topology(two_group_cohort):
    subjects, _ = two_group_cohort
    inds = fit_individual_structures(subjects, nbf=10, verbose=False)
    adj = inds[0].adj_mat.copy()
    for m in inds[1:]:
        m.adj_mat = adj.copy()

    dist_lpl = compute_mdm_distance(inds, metric="lpl_separation", nbf=10, verbose=False)
    dist_str = compute_mdm_distance(inds, metric="strength_frobenius", nbf=10, verbose=False)
    assert np.max(dist_lpl.matrix) < 1e-3 or np.mean(dist_lpl.matrix) < np.mean(dist_str.matrix)


def test_to_similarity_and_to_frame(two_group_cohort):
    subjects, _ = two_group_cohort
    ids = ["S0", "S1", "S2", "S3"]
    dist = compute_mdm_distance(subjects, subject_ids=ids, nbf=10, verbose=False)
    sim = dist.to_similarity()
    np.testing.assert_allclose(np.diag(sim), 1.0)
    frame = dist.to_frame()
    assert list(frame.index) == ids
    assert list(frame.columns) == ids


def test_joint_common_structure_runs(two_group_cohort):
    subjects, _ = two_group_cohort
    dist = compute_mdm_distance(
        subjects, common_structure="joint", nbf=10, verbose=False
    )
    assert isinstance(dist, MDMDistanceResult)
    assert dist.metadata["common_structure"] == "joint"
    assert np.all(np.isfinite(dist.matrix))


def test_additive_scope_existing_apis_unchanged():
    """Smoke test: core public entry points still exist with expected signatures."""
    from mdmp import dlm_filter, compute_logpl, select_discount_factors, StructureLearner

    assert callable(dlm_filter)
    assert callable(compute_logpl)
    assert callable(select_discount_factors)
    learner = StructureLearner
    assert "learn_structure" in dir(learner)
    sig = inspect.signature(compute_logpl)
    assert list(sig.parameters)[:4] == ["data", "adj_mat", "delta", "node_idx"]

    # compute_logpl still returns finite for clipped delta
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    delta = get_default_delta()[-1]
    score = compute_logpl(data, adj, min(float(delta), 1.0), 1, 10)
    assert np.isfinite(score)


def test_lpl_separation_warns_and_clips_strongly_negative():
    """Strongly negative d emits a warning and is clipped to 0."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from mdmp.group_analysis.distance.metrics import lpl_separation

    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 3))
    adj = np.zeros((3, 3), dtype=int)
    m_i = SimpleNamespace(data=data)
    m_j = SimpleNamespace(data=data)
    ctx = {
        "self_lpl_i": 0.0,
        "self_lpl_j": 0.0,
        "common_adj": adj,
        "nbf": 10,
        "delta_grid": np.array([0.9, 0.9, 0.9]),
    }

    with patch(
        "mdmp.group_analysis.distance.metrics.select_discount_factors",
        return_value={"DF_hat": np.array([0.9, 0.9, 0.9])},
    ), patch(
        "mdmp.group_analysis.distance.metrics.joint_lpl",
        return_value=10.0,
    ):
        # d = (0+0) - (10+10) = -20 → warn + clip to 0
        with pytest.warns(UserWarning, match="lpl_separation"):
            d = lpl_separation(m_i, m_j, ctx=ctx)
    assert d == 0.0
