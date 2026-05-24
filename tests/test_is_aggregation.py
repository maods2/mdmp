"""Tests for Individual Structure (IS) aggregation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from mdmp.group_analysis import (
    ISAggregatedMDMView,
    ISAggregateOptions,
    ISAggregationResult,
    ISMDMViewOptions,
    ISMonteCarloOptions,
    ISVoteOptions,
    aggregate_individual_structures,
    aggregate_with_options,
    as_inds_mdm_view,
    merge_aggregate_options,
    vote_individual_structures,
)
from mdmp.group_analysis.inds.voting import repair_dag_to_acyclic, vote_edge_frequencies
from mdmp.plotting import plot_dag


def _is_dag(adj: np.ndarray) -> bool:
    g = nx.DiGraph()
    n = adj.shape[0]
    g.add_nodes_from(range(n))
    idx = np.argwhere(adj != 0)
    for i, j in idx:
        if i != j:
            g.add_edge(int(i), int(j))
    return nx.is_directed_acyclic_graph(g)


def test_identical_dags_preserved():
    """All subjects share the same DAG → aggregate matches."""
    dag = np.zeros((3, 3), dtype=int)
    dag[0, 1] = 1
    dag[1, 2] = 1
    mats = [dag.copy() for _ in range(4)]
    r = aggregate_individual_structures(mats, tau=0.5)
    np.testing.assert_array_equal(r.adj_mat, dag.astype(float))
    assert r.metadata["edge_frequencies"][0, 1] == 1.0
    assert r.metadata["edge_frequencies"][1, 2] == 1.0
    assert r.n_subjects == 4


def test_threshold_inclusive_boundary():
    """6/10 at τ=0.6: strict excludes edge (=), inclusive keeps edge."""
    edge01 = np.zeros((2, 2), dtype=int)
    edge01[0, 1] = 1
    no_edge = np.zeros((2, 2), dtype=int)
    mats = [edge01.copy() for _ in range(6)] + [no_edge.copy() for _ in range(4)]
    rs = aggregate_individual_structures(mats, tau=0.6, threshold_mode="strict")
    ri = aggregate_individual_structures(mats, tau=0.6, threshold_mode="inclusive")
    assert rs.metadata["threshold_mode"] == "strict"
    assert rs.metadata["edge_frequencies"][0, 1] == pytest.approx(0.6)
    assert rs.adj_mat[0, 1] == 0.0
    assert ri.adj_mat[0, 1] == 1.0


def test_aggregate_with_options_matches_flat_keywords():
    edge01 = np.zeros((2, 2), dtype=int)
    edge01[0, 1] = 1
    no_edge = np.zeros((2, 2), dtype=int)
    mats = [edge01.copy() for _ in range(6)] + [no_edge.copy() for _ in range(4)]
    opts = ISAggregateOptions(threshold_mode="inclusive")
    flat = aggregate_individual_structures(mats, tau=0.6, threshold_mode="inclusive")
    wrapped = aggregate_with_options(mats, tau=0.6, options=opts)
    np.testing.assert_array_equal(wrapped.adj_mat, flat.adj_mat)
    assert wrapped.metadata["threshold_mode"] == flat.metadata["threshold_mode"]


def test_majority_threshold():
    """Edge in 2/3 subjects kept at tau=0.5, dropped at tau=2/3 (strict >)."""
    n = 2
    e01 = np.zeros((n, n), dtype=int)
    e01[0, 1] = 1
    no_edge = np.zeros((n, n), dtype=int)
    mats = [e01.copy(), e01.copy(), no_edge]
    r = aggregate_individual_structures(mats, tau=0.5)
    assert r.adj_mat[0, 1] == 1.0
    r2 = aggregate_individual_structures(mats, tau=2 / 3)
    assert r2.adj_mat[0, 1] == 0.0


def test_cycle_broken_lowest_frequency():
    """
    Four 3-node DAGs whose union above tau=0.4 forms a 3-cycle; lowest-freq edge removed.
    """
    z = np.zeros((3, 3), dtype=int)
    g1 = z.copy()
    g1[0, 1] = 1
    g1[1, 2] = 1
    g2 = z.copy()
    g2[1, 2] = 1
    g2[2, 0] = 1
    g3 = z.copy()
    g3[2, 0] = 1
    g3[0, 1] = 1
    g4 = z.copy()
    g4[0, 1] = 1
    g4[1, 2] = 1
    mats = [g1, g2, g3, g4]
    r = aggregate_individual_structures(mats, tau=0.4)
    assert _is_dag(r.adj_mat)
    assert len(r.metadata["edges_removed_for_acyclicity"]) >= 1
    rem = r.metadata["edges_removed_for_acyclicity"][0]
    assert rem["parent_idx"] == 2 and rem["child_idx"] == 0
    assert rem["frequency"] == 0.5


def test_dataframe_node_names():
    cols = ["a", "b"]
    z = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    d1 = z.copy()
    d1.loc["a", "b"] = 1
    d2 = z.copy()
    d2.loc["a", "b"] = 1
    r = aggregate_individual_structures([d1, d2], tau=0.5)
    assert r.node_names == ["a", "b"]
    assert r.adj_mat[0, 1] == 1.0


def test_empty_adj_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_individual_structures([], tau=0.5)


def test_shape_mismatch_raises():
    a = np.zeros((2, 2), dtype=int)
    b = np.zeros((3, 3), dtype=int)
    with pytest.raises(ValueError, match="same shape"):
        aggregate_individual_structures([a, b], tau=0.5)


def test_invalid_tau_raises():
    a = np.zeros((2, 2), dtype=int)
    with pytest.raises(ValueError, match="tau"):
        aggregate_individual_structures([a], tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        aggregate_individual_structures([a], tau=1.5)


def test_non_binary_raises():
    a = np.zeros((2, 2), dtype=float)
    a[0, 1] = 0.5
    with pytest.raises(ValueError, match="binary"):
        aggregate_individual_structures([a], tau=0.5)


def test_result_type():
    a = np.zeros((2, 2), dtype=int)
    r = aggregate_individual_structures([a], tau=0.5)
    assert isinstance(r, ISAggregatedMDMView)
    assert isinstance(r, ISAggregationResult)
    assert r.global_beta_mc is None


def test_plot_data_shape_validates():
    with pytest.raises(ValueError, match="plot_data"):
        aggregate_individual_structures(
            [np.zeros((2, 2), dtype=int)],
            tau=0.5,
            plot_data=np.zeros((5, 3)),
        )


def test_vote_individual_structures_matches_aggregate_no_mc():
    edge01 = np.zeros((2, 2), dtype=int)
    edge01[0, 1] = 1
    mats = [edge01.copy(), edge01.copy(), np.zeros((2, 2), dtype=int)]
    v = vote_individual_structures(mats, tau=0.5)
    full = aggregate_individual_structures(mats, tau=0.5, n_draws=0)
    np.testing.assert_array_equal(v.adj_mat, full.adj_mat)
    assert v.metadata["threshold_mode"] == full.metadata["threshold_mode"]


def test_vote_and_repair_split_matches_combined():
    edge01 = np.zeros((2, 2), dtype=int)
    edge01[0, 1] = 1
    mats = [edge01.copy(), edge01.copy(), np.zeros((2, 2), dtype=int)]
    names = ["a", "b"]
    cand, _counts, freq = vote_edge_frequencies(mats, 0.5, threshold_mode="strict")
    out, removed = repair_dag_to_acyclic(cand, freq, names)
    from mdmp.group_analysis.inds.voting import _vote_threshold_and_repair_cycles

    out2, meta2 = _vote_threshold_and_repair_cycles(mats, 0.5, names, threshold_mode="strict")
    np.testing.assert_array_equal(out, out2)
    assert removed == meta2["edges_removed_for_acyclicity"]


def test_merge_aggregate_options_matches_flat():
    opts = merge_aggregate_options(
        vote=ISVoteOptions(threshold_mode="inclusive"),
        view=ISMDMViewOptions(pool_filt_for_plotting=False),
    )
    flat = ISAggregateOptions(threshold_mode="inclusive")
    assert opts.threshold_mode == flat.threshold_mode


def test_run_inds_global_beta_mc_matches_aggregate():
    """Split MC path matches aggregate_individual_structures for the same inputs."""
    from mdmp.group_analysis import run_inds_global_beta_mc

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
    consensus = vote_individual_structures([e01], tau=0.5)
    full = aggregate_individual_structures(
        [e01],
        tau=0.5,
        filtered_per_subject=[filt],
        time_index=tix,
        n_draws=500,
        rng=np.random.default_rng(42),
    )
    split = run_inds_global_beta_mc(
        consensus,
        [e01],
        n_draws=500,
        rng=np.random.default_rng(42),
        filtered_per_subject=[filt],
        time_index=tix,
    )
    assert split.global_beta_mc is not None
    assert full.global_beta_mc is not None
    np.testing.assert_allclose(
        split.global_beta_mc.beta_mean, full.global_beta_mc.beta_mean
    )


def test_as_inds_mdm_view_preserves_consensus():
    dag = np.zeros((2, 2), dtype=int)
    dag[0, 1] = 1
    c = vote_individual_structures([dag, dag], tau=0.5)
    view = as_inds_mdm_view(c, data=np.zeros((10, 2)))
    assert view.data.shape == (10, 2)
    np.testing.assert_array_equal(view.adj_mat, c.adj_mat)


def test_plot_dag_accepts_is_aggregated_view():
    dag = np.zeros((3, 3), dtype=int)
    dag[0, 1] = 1
    dag[1, 2] = 1
    mats = [dag.copy() for _ in range(2)]
    r = aggregate_individual_structures(mats, tau=0.5, node_names=["a", "b", "c"])
    fig = plot_dag(r, plot_type="graph")
    assert fig is not None
    plt.close(fig)
    fig2 = plot_dag(r, plot_type="heatmap")
    plt.close(fig2)
