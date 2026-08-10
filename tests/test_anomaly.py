"""Tests for MDM predictive-interval anomaly detection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mdmp.anomaly import AnomalyDetectionResult, detect_anomalies
from mdmp.model import refit_mdm_on_structure


def _mock_mdm(
    y: np.ndarray,
    *,
    ft: Optional[np.ndarray] = None,
    Qt: Optional[np.ndarray] = None,
    nt: Optional[np.ndarray] = None,
    ets: Optional[np.ndarray] = None,
    node_names: Optional[list] = None,
) -> SimpleNamespace:
    """Build a minimal MDM-like object with controllable Filt arrays."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    t, n = y.shape
    if ft is None:
        ft = {j: np.zeros(t) for j in range(n)}
    elif not isinstance(ft, dict):
        ft = {0: np.asarray(ft, dtype=float)}
    if Qt is None:
        Qt = {j: np.ones(t) for j in range(n)}
    elif not isinstance(Qt, dict):
        Qt = {0: np.asarray(Qt, dtype=float)}
    if nt is None:
        # Posterior df after update; predictive df = nt - 1.
        nt = {j: np.full(t, 20.0) for j in range(n)}
    elif not isinstance(nt, dict):
        nt = {0: np.asarray(nt, dtype=float)}
    if ets is None:
        ets = {
            j: (y[:, j] - ft[j]) / np.sqrt(Qt[j])
            for j in range(n)
        }
    elif not isinstance(ets, dict):
        ets = {0: np.asarray(ets, dtype=float)}
    names = node_names or [f"V{i + 1}" for i in range(n)]
    return SimpleNamespace(
        data=y,
        node_names=names,
        Filt={"ft": ft, "Qt": Qt, "nt": nt, "ets": ets},
    )


def test_no_anomalies_inside_band():
    t = 30
    ft = np.zeros(t)
    Qt = np.ones(t)
    nt = np.full(t, 21.0)  # df = 20
    half = stats.t.ppf(0.975, 20) * np.sqrt(Qt)
    y = 0.1 * np.ones(t)  # well inside ±half
    model = _mock_mdm(y, ft=ft, Qt=Qt, nt=nt)
    res = detect_anomalies(model, ci_level=0.95, series=0)
    assert isinstance(res, AnomalyDetectionResult)
    assert not np.any(res.is_anomaly)
    assert res.observed.shape == (t,)
    assert np.allclose(res.upper - res.lower, 2 * half)


def test_one_and_multiple_anomalies():
    t = 20
    ft = np.zeros(t)
    Qt = np.full(t, 0.25)
    nt = np.full(t, 31.0)  # df = 30
    half = float(stats.t.ppf(0.975, 30) * 0.5)
    y = np.zeros(t)
    y[5] = half + 1.0
    y[10] = -(half + 2.0)
    y[11] = -(half + 2.0)
    model = _mock_mdm(y, ft=ft, Qt=Qt, nt=nt)
    res = detect_anomalies(model, ci_level=0.95, series=0)
    assert res.is_anomaly[5] and res.is_anomaly[10] and res.is_anomaly[11]
    assert res.is_anomaly.sum() == 3
    assert np.allclose(res.score[5], abs(y[5] / 0.5))


def test_anomaly_only_on_selected_variable():
    t = 25
    y = np.zeros((t, 2))
    y[7, 1] = 50.0  # huge outlier on node 1 only
    ft = {0: np.zeros(t), 1: np.zeros(t)}
    Qt = {0: np.ones(t), 1: np.ones(t)}
    nt = {0: np.full(t, 20.0), 1: np.full(t, 20.0)}
    model = _mock_mdm(y, ft=ft, Qt=Qt, nt=nt, node_names=["a", "b"])
    all_res = detect_anomalies(model, series=None)
    assert all_res.is_anomaly.shape == (t, 2)
    assert not np.any(all_res.is_anomaly[:, 0])
    assert all_res.is_anomaly[7, 1]
    only_a = detect_anomalies(model, series="a")
    assert only_a.is_anomaly.shape == (t,)
    assert not np.any(only_a.is_anomaly)
    only_b = detect_anomalies(model, series="b")
    assert only_b.is_anomaly[7]


def test_wider_ci_level_flags_fewer_or_equal():
    t = 40
    rng = np.random.default_rng(0)
    ft = np.zeros(t)
    Qt = np.ones(t)
    nt = np.full(t, 16.0)
    y = rng.normal(scale=1.2, size=t)
    model = _mock_mdm(y, ft=ft, Qt=Qt, nt=nt)
    narrow = detect_anomalies(model, ci_level=0.80)
    wide = detect_anomalies(model, ci_level=0.99)
    assert wide.is_anomaly.sum() <= narrow.is_anomaly.sum()


def test_dataframe_output_and_invalid_qt():
    t = 10
    y = np.linspace(-1, 1, t)
    Qt = np.ones(t)
    Qt[3] = np.nan
    model = _mock_mdm(y, Qt=Qt, nt=np.full(t, 10.0))
    df = detect_anomalies(model, output="dataframe", time_index=list(range(t)))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "time",
        "node",
        "observed",
        "fitted_mean",
        "lower",
        "upper",
        "score",
        "is_anomaly",
        "ci_level",
    }
    row = df.loc[df["time"] == 3].iloc[0]
    assert not bool(row["is_anomaly"])
    assert np.isnan(row["lower"]) and np.isnan(row["upper"])


def test_pathological_small_df_is_non_comparable():
    """Vague-prior first steps (df = n0 ≪ 1) must not produce ±1e100 bands."""
    t = 8
    y = np.full(t, 100.0)
    ft = np.zeros(t)
    Qt = np.full(t, 7.0)
    # Mimic default n0=0.001 after updates: nt = n0+1, n0+2, ...
    nt = 0.001 + np.arange(1, t + 1, dtype=float)
    model = _mock_mdm(y, ft=ft, Qt=Qt, nt=nt)
    res = detect_anomalies(model, series=0, ci_level=0.95)
    # df = nt-1 <= 2 for the first two indices → NaN bands, no flags.
    assert np.all(np.isnan(res.lower[:2]))
    assert np.all(np.isnan(res.upper[:2]))
    assert not np.any(res.is_anomaly[:2])
    assert np.all(np.isfinite(res.lower[2:]))
    assert np.nanmax(np.abs(res.upper)) < 1e6


def test_detect_anomalies_on_refit_model():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(50, 3))
    data[20, 0] = 25.0
    adj = np.zeros((3, 3), dtype=int)
    adj[0, 1] = 1
    model = refit_mdm_on_structure(
        data, adj, node_names=["a", "b", "c"], verbose=False, n_jobs=None
    )
    res = detect_anomalies(model, ci_level=0.95)
    assert res.is_anomaly.shape == (50, 3)
    assert res.node_names == ["a", "b", "c"]
    # The injected spike should be flagged on node a for a tight-enough model.
    assert res.is_anomaly[20, 0]
