"""
Tests for IS-aligned Monte Carlo global edge coefficients.
"""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from mdmp.group_analysis import (
    GlobalBetaMCResult,
    ISAggregatedMDMView,
    ISAggregationResult,
    aggregate_individual_structures,
)


def _mc_aggregate(adjs, filt, *, tau: float = 0.5, T: int = 5, n_nodes: int = 2, **kwargs):
    """Run MC via aggregate_individual_structures with mocked refit on G*."""
    filt_list = list(filt)
    call_idx = [0]
    node_names = kwargs.pop("node_names", None) or [f"n{i}" for i in range(n_nodes)]

    def _fake_refit(data, adj_mat, **kw):
        i = call_idx[0]
        call_idx[0] += 1
        filt_i = _add_rt_to_filt(filt_list[i])
        smoo = {
            "smt": {c: np.asarray(filt_i["mt"][c]) for c in filt_i["mt"]},
            "sCt": {c: np.asarray(filt_i["Ct"][c]) for c in filt_i["Ct"]},
        }
        return SimpleNamespace(Filt=filt_i, Smoo=smoo)

    subjects = [
        _mdm_like(
            adj_mat=np.asarray(adj, dtype=int),
            Filt=filt_list[i],
            node_names=list(node_names),
            data=np.zeros((T, n_nodes), dtype=float),
        )
        for i, adj in enumerate(adjs)
    ]
    with patch(
        "mdmp.group_analysis.inds.refit.refit_mdm_on_structure",
        side_effect=_fake_refit,
    ):
        return aggregate_individual_structures(
            subjects,
            tau=tau,
            mc_refit_global_structure=True,
            **kwargs,
        )


def test_aggregate_global_beta_mc_none_for_adj_only():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    r = aggregate_individual_structures([e01], tau=0.5, mc_n_samples=0)
    assert isinstance(r, ISAggregatedMDMView)
    assert isinstance(r, ISAggregationResult)
    assert r.global_beta_mc is None
    assert r.Filt is None


def test_aggregate_global_beta_posterior_mean_approximation():
    """Many MC samples with moderate Ct → empirical mean near mt (marginal t mean)."""
    rng = np.random.default_rng(0)
    T = 8
    tix = 3
    n = 2
    mt = np.array([0.25, 0.75])
    Ct = np.diag([0.02, 0.02])
    cc = np.zeros((2, 2, T))
    for ti in range(T):
        cc[:, :, ti] = Ct

    def filt_for_child1() -> dict:
        mt_d: dict = {}
        ct_d: dict = {}
        nt_d: dict = {}
        dt_d: dict = {}
        for c in range(n):
            if c == 0:
                mt_d[c] = np.zeros((1, T))
                ct_d[c] = np.ones((1, 1, T)) * 1e-6
            else:
                m = np.zeros((2, T))
                m[:, tix] = mt
                mt_d[c] = m
                ct_d[c] = cc.copy()
            nt_d[c] = np.full(T, 25.0)
            dt_d[c] = np.full(T, 25.0)
        return {"mt": mt_d, "Ct": ct_d, "nt": nt_d, "dt": dt_d}

    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = filt_for_child1()
    r = _mc_aggregate(
        [e01],
        [filt],
        mc_n_samples=12_000,
        rng=rng,
        T=T,
    )
    assert r.global_beta_mc is not None
    gb = r.global_beta_mc
    assert isinstance(gb, GlobalBetaMCResult)
    col = gb.beta_samples[:, 0, tix]
    assert np.nanmean(col) == pytest.approx(mt[1], abs=0.08)


def _add_rt_to_filt(filt: dict) -> dict:
    out = dict(filt)
    rt = {}
    for c in filt["Ct"]:
        Cc = np.asarray(filt["Ct"][c], dtype=float)
        rt[c] = Cc.copy()
    out["Rt"] = rt
    return out


def _synth_filtered_n2_t5(
    parent_coef_at_t2: Tuple[float, float, float],
) -> List[dict]:
    """Three subjects, N=2 nodes, T=5; only child 1 has a parent edge from 0."""
    T = 5
    tix = 2
    n = 2

    def one_filt(beta_parent: float) -> dict:
        mt = {}
        Ct = {}
        nt = {}
        dt = {}
        for c in range(n):
            if c == 0:
                mt[c] = np.zeros((1, T))
                Ct[c] = np.ones((1, 1, T)) * 1e-6
            else:
                m = np.zeros((2, T))
                m[1, tix] = beta_parent
                mt[c] = m
                C = np.eye(2) * 1e-6
                cc = np.zeros((2, 2, T))
                for ti in range(T):
                    cc[:, :, ti] = C
                Ct[c] = cc
            nt[c] = np.full(T, 80.0)
            dt[c] = np.full(T, 80.0)
        return {"mt": mt, "Ct": Ct, "nt": nt, "dt": dt}

    b0, b1, b2 = parent_coef_at_t2
    return [one_filt(b0), one_filt(b1), one_filt(b2)]


def test_aggregate_global_beta_population_mean():
    """bar_theta^(b) = (1/S) sum_i theta_i^(b) with S subjects on consensus DAG."""
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    adjs = [e01.copy(), e01.copy()]
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))[:2]
    rng = np.random.default_rng(42)
    tix = 2
    r = _mc_aggregate(
        adjs,
        filt,
        mc_n_samples=200,
        rng=rng,
    )
    assert r.global_beta_mc is not None
    out = r.global_beta_mc
    assert isinstance(out, GlobalBetaMCResult)
    assert out.edges == [(0, 1)]
    assert out.pooling == "population_mean"
    assert out.n_contributors[0] == 2
    assert np.nanmean(out.beta_samples[:, 0, tix]) == pytest.approx(3.0, abs=0.05)


def test_aggregate_global_beta_pooled_samples_wider_with_heterogeneous_subject_variance():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    T, tix, beta = 5, 2, 2.0

    def filt_child1(c_diag: float) -> dict:
        n = 2
        mt, Ct, nt, dt = {}, {}, {}, {}
        for c in range(n):
            if c == 0:
                mt[c] = np.zeros((1, T))
                Ct[c] = np.ones((1, 1, T)) * 1e-6
            else:
                m = np.zeros((2, T))
                m[1, tix] = beta
                mt[c] = m
                C = np.eye(2) * c_diag
                cc = np.zeros((2, 2, T))
                for ti in range(T):
                    cc[:, :, ti] = C
                Ct[c] = cc
            nt[c] = np.full(T, 80.0)
            dt[c] = np.full(T, 80.0)
        return {"mt": mt, "Ct": Ct, "nt": nt, "dt": dt}

    tight = 1e-6
    hom = [filt_child1(tight), filt_child1(tight)]
    het = [filt_child1(tight), filt_child1(4.0)]
    n_mc = 12_000
    kw = {"mc_n_samples": n_mc}
    r_hom = _mc_aggregate(
        [e01, e01],
        hom,
        rng=np.random.default_rng(101),
        **kw,
    )
    r_het = _mc_aggregate(
        [e01, e01],
        het,
        rng=np.random.default_rng(101),
        **kw,
    )
    col_hom = r_hom.global_beta_mc.beta_samples[:, 0, tix]
    col_het = r_het.global_beta_mc.beta_samples[:, 0, tix]
    assert np.nanmean(col_hom) == pytest.approx(beta, abs=0.06)
    assert np.nanmean(col_het) == pytest.approx(beta, abs=0.06)
    v_hom = float(np.nanvar(col_hom))
    v_het = float(np.nanvar(col_het))
    assert v_het > v_hom * 8.0


def test_aggregate_global_beta_rng_determinism():
    adjs = [np.array([[0, 1], [0, 0]], dtype=int)]
    filt = _synth_filtered_n2_t5((1.5, 0.0, 0.0))[:1]
    kw = {"mc_n_samples": 5}
    a = _mc_aggregate(adjs, filt, rng=np.random.default_rng(123), **kw)
    b = _mc_aggregate(adjs, filt, rng=np.random.default_rng(123), **kw)
    assert np.allclose(a.global_beta_mc.beta_samples, b.global_beta_mc.beta_samples)


def test_aggregate_global_beta_empty_graph():
    filt = _synth_filtered_n2_t5((1.0, 1.0, 1.0))[:1]
    adjs = [np.zeros((2, 2), dtype=int)]
    r = _mc_aggregate(
        adjs,
        filt,
        mc_n_samples=3,
        rng=np.random.default_rng(0),
    )
    assert r.global_beta_mc is not None
    assert r.global_beta_mc.beta_samples.shape == (3, 0, 5)


def test_aggregate_global_beta_all_filter_times():
    """MC runs at every filter time step 0 … T-1."""
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    rng = np.random.default_rng(0)
    r = _mc_aggregate(
        [e01, e01],
        filt[:2],
        mc_n_samples=50,
        rng=rng,
    )
    gb = r.global_beta_mc
    assert gb is not None
    assert gb.beta_samples.shape == (50, 1, 5)
    assert gb.time_indices_mc == (0, 1, 2, 3, 4)


def test_aggregate_global_beta_beta_mean_var():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    rng = np.random.default_rng(3)
    r = _mc_aggregate(
        [e01, e01],
        filt[:2],
        mc_n_samples=200,
        rng=rng,
    )
    gb = r.global_beta_mc
    assert gb is not None
    assert gb.beta_mean is not None and gb.beta_var is not None
    assert gb.beta_mean.shape == (1, 5)
    assert gb.beta_var.shape == (1, 5)
    assert np.isfinite(float(gb.beta_mean[0, 2]))


def test_aggregate_global_beta_smoothed_posterior():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt0 = _add_rt_to_filt(_synth_filtered_n2_t5((1.5, 0.0, 0.0))[0])
    rng = np.random.default_rng(11)
    r = _mc_aggregate(
        [e01],
        [filt0],
        mc_n_samples=30,
        rng=rng,
        mc_posterior="smoothed",
    )
    gb = r.global_beta_mc
    assert gb is not None
    assert gb.metadata.get("mc_posterior") == "smoothed"
    assert gb.beta_samples.shape == (30, 1, 5)
    assert np.all(np.isfinite(gb.beta_samples[:, 0, 2]))


def test_mc_requires_refit_on_consensus():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((1.0, 1.0, 1.0))[:1]
    m = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((5, 2)),
    )
    with pytest.raises(ValueError, match="mc_refit_global_structure"):
        aggregate_individual_structures(
            [m],
            tau=0.5,
            mc_n_samples=5,
            rng=np.random.default_rng(0),
            mc_refit_global_structure=False,
        )


@patch("mdmp.group_analysis.inds.refit.refit_mdm_on_structure")
def test_mdm_aggregate_population_mean_after_refit(mock_refit):
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    shared = _add_rt_to_filt(_synth_filtered_n2_t5((3.0, 3.0, 3.0))[0])

    def _fake(data, adj_mat, **kwargs):
        return SimpleNamespace(Filt=shared, Smoo={})

    mock_refit.side_effect = _fake
    T, n = 5, 2
    rng = np.random.default_rng(99)
    subjects = [
        _mdm_like(adj_mat=e01.copy(), Filt=shared, node_names=["a", "b"], data=np.zeros((T, n))),
        _mdm_like(adj_mat=e01.copy(), Filt=shared, node_names=["a", "b"], data=np.ones((T, n))),
    ]
    r = aggregate_individual_structures(
        subjects,
        tau=0.5,
        mc_n_samples=40,
        rng=rng,
        mc_refit_global_structure=True,
    )
    gb = r.global_beta_mc
    assert gb.n_contributors[0] == 2
    assert gb.pooling == "population_mean"


def test_aggregate_global_beta_mc_quantiles():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    rng = np.random.default_rng(1)
    r = _mc_aggregate(
        [e01, e01],
        filt[:2],
        mc_n_samples=400,
        rng=rng,
        mc_quantiles=(0.25, 0.5, 0.75),
    )
    gb = r.global_beta_mc
    assert gb is not None
    assert gb.beta_quantiles is not None
    assert gb.beta_quantiles.shape == (3, 1, 5)
    assert gb.quantile_levels == (0.25, 0.5, 0.75)
    qcol = gb.beta_quantiles[:, 0, 2]
    assert np.all(np.diff(qcol) >= 0)


def test_aggregate_invalid_precision_raises():
    T = 3
    tix = 1
    mt = {0: np.zeros((1, T)), 1: np.zeros((2, T))}
    cc = np.zeros((2, 2, T))
    for ti in range(T):
        cc[:, :, ti] = np.eye(2) * 1e-6
    Ct = {0: np.ones((1, 1, T)) * 1e-6, 1: cc}
    nt_bad = {0: np.full(T, 5.0), 1: np.array([5.0, -1.0, 5.0])}
    dt = {0: np.full(T, 5.0), 1: np.full(T, 5.0)}
    filt = {"mt": mt, "Ct": Ct, "nt": nt_bad, "dt": dt}
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    with pytest.raises(ValueError, match="nt and dt"):
        _mc_aggregate(
            [e01],
            [filt],
            mc_n_samples=2,
            rng=np.random.default_rng(0),
        )


def test_mdm_aggregate_auto_pools_filt():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    adjs = [e01.copy(), e01.copy(), np.zeros((2, 2), dtype=int)]
    r = aggregate_individual_structures(
        [
            _mdm_like(adj_mat=adjs[0], Filt=filt[0], node_names=["a", "b"], data=np.zeros((5, 2))),
            _mdm_like(adj_mat=adjs[1], Filt=filt[1], node_names=["a", "b"], data=np.zeros((5, 2))),
            _mdm_like(adj_mat=adjs[2], Filt=filt[2], node_names=["a", "b"], data=np.zeros((5, 2))),
        ],
        tau=0.5,
        mc_n_samples=10,
        rng=np.random.default_rng(0),
    )
    assert r.Filt is not None
    assert set(r.Filt.keys()) >= {"mt", "Ct", "nt", "dt"}
    assert 1 in r.Filt["mt"]
    assert r.Filt["mt"][1].ndim == 2


def test_plot_arcs_on_mdm_aggregate_view():
    from mdmp.plotting import plot_arcs

    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((1.0, 2.0, 0.0))[:1]
    T = 5
    m = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((T, 2), dtype=float),
    )
    r = aggregate_individual_structures(
        [m],
        tau=0.5,
        mc_n_samples=5,
        rng=np.random.default_rng(0),
    )
    fig = plot_arcs(r, plot_type="connections")
    assert fig is not None


def _mdm_like(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def test_aggregate_from_mdm_like_objects_pool_filt():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    T = 5
    m0 = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((T, 2), dtype=float),
    )
    m1 = _mdm_like(
        adj_mat=e01.copy(),
        Filt=filt[1],
        node_names=["a", "b"],
        data=np.ones((T, 2), dtype=float),
    )
    r = aggregate_individual_structures(
        [m0, m1],
        tau=0.5,
        mc_n_samples=10,
        rng=np.random.default_rng(0),
    )
    assert r.Filt is not None
    assert r.data is not None and r.data.shape == (T, 2)
    assert float(r.data[0, 0]) == pytest.approx(0.5)


def test_aggregate_accepts_generator_of_mdm_like():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    T = 5

    def gen():
        yield _mdm_like(
            adj_mat=e01,
            Filt=filt[0],
            node_names=["a", "b"],
            data=np.zeros((T, 2), dtype=float),
        )

    r = aggregate_individual_structures(
        gen(), tau=0.5, mc_n_samples=5, rng=np.random.default_rng(0)
    )
    assert r.Filt is not None


@patch("mdmp.group_analysis.inds.refit.refit_mdm_on_structure")
def test_mdm_aggregate_auto_refit_on_consensus(mock_refit):
    """mc_refit_global_structure=None refits each MDM subject on G* before MC."""
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    T = 5
    shared = _add_rt_to_filt(filt[0])

    def _fake(data, adj_mat, **kwargs):
        return SimpleNamespace(Filt=shared, Smoo={})

    mock_refit.side_effect = _fake
    m0 = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((T, 2), dtype=float),
    )
    m1 = _mdm_like(
        adj_mat=e01.copy(),
        Filt=filt[1],
        node_names=["a", "b"],
        data=np.ones((T, 2), dtype=float),
    )
    r = aggregate_individual_structures(
        [m0, m1],
        tau=0.5,
        mc_n_samples=10,
        rng=np.random.default_rng(0),
    )
    assert mock_refit.call_count == 2
    assert r.global_beta_mc is not None
    assert r.refit_filt_per_subject is not None


def test_mdm_aggregate_mc_rejects_no_refit():
    """Monte Carlo requires refit on the consensus DAG."""
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    T = 5
    m = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((T, 2), dtype=float),
    )
    with pytest.raises(ValueError, match="mc_refit_global_structure"):
        aggregate_individual_structures(
            [m],
            tau=0.5,
            mc_n_samples=10,
            rng=np.random.default_rng(0),
            mc_refit_global_structure=False,
        )


def test_aggregate_mdm_mc_covers_all_filter_times():
    """aggregate_individual_structures MC uses every filter time index."""
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((2.0, 4.0, 100.0))
    T = 5
    m = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((T, 2), dtype=float),
    )
    with patch(
        "mdmp.group_analysis.inds.refit.refit_mdm_on_structure",
        return_value=SimpleNamespace(Filt=_add_rt_to_filt(filt[0]), Smoo={}),
    ):
        r = aggregate_individual_structures(
            [m],
            tau=0.5,
            mc_n_samples=20,
            rng=np.random.default_rng(1),
            mc_refit_global_structure=True,
        )
    assert r.global_beta_mc is not None
    assert r.global_beta_mc.beta_samples.shape == (20, 1, T)
    assert r.global_beta_mc.time_indices_mc == tuple(range(T))


def test_aggregate_mixed_mdm_and_adj_raises():
    e01 = np.array([[0, 1], [0, 0]], dtype=int)
    filt = _synth_filtered_n2_t5((1.0, 0.0, 0.0))[:1]
    m = _mdm_like(
        adj_mat=e01,
        Filt=filt[0],
        node_names=["a", "b"],
        data=np.zeros((3, 2)),
    )
    with pytest.raises(TypeError, match="pass either only fitted MDM"):
        aggregate_individual_structures([m, e01], tau=0.5)
