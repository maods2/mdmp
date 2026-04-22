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
    ISAggregationResult,
    aggregate_individual_structures,
)
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
