"""Smoke tests for ``refit_mdm_on_structure``."""

import numpy as np

from mdmp.model import refit_mdm_on_structure


def test_refit_mdm_on_structure_smoke():
    rng = np.random.default_rng(42)
    T, N = 45, 3
    data = rng.normal(size=(T, N))
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int)
    res = refit_mdm_on_structure(
        data,
        adj,
        node_names=["a", "b", "c"],
        verbose=False,
        n_jobs=None,
    )
    assert res.adj_mat.shape == (N, N)
    assert res.data.shape == (T, N)
    assert "DF_hat" in res.DF
    assert "mt" in res.Filt and "smt" in res.Smoo
