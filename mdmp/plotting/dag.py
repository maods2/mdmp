"""
DAG visualization functions for MDM models.
"""

from typing import TYPE_CHECKING, Any, List, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    pass


def _hierarchical_layout(
    G: nx.DiGraph,
    *,
    level_gap: float = 1.0,
    layout_seed: Optional[int] = 0,
) -> dict:
    """
    Layered positions for a directed graph: Graphviz ``dot`` when available,
    otherwise coordinates from :func:`networkx.topological_generations`.
    Non-DAGs use :func:`networkx.spring_layout`.
    """
    if G.number_of_nodes() == 0:
        return {}

    if not nx.is_directed_acyclic_graph(G):
        return nx.spring_layout(G, k=1.5, iterations=50, seed=layout_seed)

    try:
        return nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pass

    pos: dict = {}
    for level, gen in enumerate(nx.topological_generations(G)):
        nodes = list(gen)
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            if n_nodes > 1:
                x = (i - (n_nodes - 1) / 2.0) * level_gap
            else:
                x = 0.0
            y = -float(level) * level_gap
            pos[node] = (x, y)
    return pos


def plot_dag(
    mdm_object: Any,
    node_labels: Optional[List[str]] = None,
    plot_type: Literal["graph", "heatmap"] = "graph",
    show_legend: bool = False,
    edge_color: str = "black",
    node_color: str = "steelblue",
    label_color: str = "white",
    arrow_size: float = 4.0,
    figsize: Optional[tuple] = None,
    layout_seed: Optional[int] = 0,
    *,
    hierarchical: bool = True,
    level_gap: float = 1.0,
    node_size: float = 2000.0,
    font_size: int = 10,
    edge_width: float = 2.0,
) -> Figure:
    """
    Plot DAG structure as a graph or heatmap.

    Parameters
    ----------
    mdm_object
        Typically :class:`mdmp.model.MDM`, or :class:`mdmp.group_analysis.inds.aggregation.ISAggregatedMDMView`
        / the return value of :func:`mdmp.group_analysis.aggregate_individual_structures`
        (needs ``adj_mat`` and optionally ``node_names``).
    node_labels : list of str, optional
        Custom node labels. If None, uses MDM node names.
    plot_type : {"graph", "heatmap"}, optional
        Type of plot. Default is "graph".
    show_legend : bool, optional
        Whether to show legend. Default is False.
    edge_color : str, optional
        Color of edges. Default is "black".
    node_color : str, optional
        Color of nodes. Default is "steelblue".
    label_color : str, optional
        Color of node labels. Default is "white".
    arrow_size : float, optional
        Size of arrow heads. Default is 4.0.
    figsize : tuple, optional
        Figure size. Default scales lightly with node count for ``graph``.
    layout_seed : int, optional
        Random seed for spring layout (cycles / ``hierarchical=False``). Use None for nondeterministic layout.
    hierarchical : bool, optional
        If True (default), use layered layout (Graphviz ``dot`` when ``pygraphviz``
        and Graphviz are available, else topological generations). If False, use
        ``spring_layout``.
    level_gap : float, optional
        Horizontal/vertical spacing between layers in the topological fallback
        (ignored when Graphviz layout succeeds).
    node_size : float, optional
        Node diameter passed to :func:`networkx.draw_networkx_nodes`.
    font_size : int, optional
        Font size for node labels on the graph.
    edge_width : float, optional
        Width of directed edges.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    adj_mat = np.asarray(mdm_object.adj_mat)
    n = int(adj_mat.shape[0])

    if node_labels is None:
        if hasattr(mdm_object, "node_names") and mdm_object.node_names:
            node_labels = list(mdm_object.node_names)
        else:
            node_labels = [f"V{i + 1}" for i in range(n)]

    if plot_type == "heatmap":
        if figsize is None:
            figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(adj_mat, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(node_labels, rotation=45, ha="right")
        ax.set_yticklabels(node_labels)

        plt.colorbar(im, ax=ax, label="Edge")

        ax.set_xlabel("Child", fontsize=12)
        ax.set_ylabel("Parent", fontsize=12)
        ax.set_title("DAG Structure (Adjacency Matrix)", fontsize=14)

        plt.tight_layout()
        return fig

    if figsize is None:
        side = min(12.0, 5.5 + 0.45 * n)
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if adj_mat[i, j] != 0:
                G.add_edge(i, j)

    if hierarchical:
        pos = _hierarchical_layout(G, level_gap=level_gap, layout_seed=layout_seed)
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=layout_seed)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        alpha=0.9,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_color,
        arrows=True,
        arrowsize=arrow_size * 10,
        arrowstyle="->",
        width=edge_width,
        alpha=0.7,
    )

    labels = {i: node_labels[i] for i in range(n)}
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        ax=ax,
        font_color=label_color,
        font_weight="bold",
        font_size=font_size,
    )

    if show_legend:
        ax.legend()

    ax.set_title("DAG Structure", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    return fig
