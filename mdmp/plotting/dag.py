"""
DAG visualization functions for MDM models.
"""

from typing import TYPE_CHECKING, List, Literal, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..model import MDM


def plot_dag(
    mdm_object: "MDM",
    node_labels: Optional[List[str]] = None,
    plot_type: Literal["graph", "heatmap"] = "graph",
    show_legend: bool = False,
    edge_color: str = "black",
    node_color: str = "steelblue",
    label_color: str = "white",
    arrow_size: float = 4.0,
    figsize: Optional[tuple] = None,
    layout_seed: Optional[int] = 0
) -> Figure:
    """
    Plot DAG structure as a graph or heatmap.

    Parameters
    ----------
    mdm_object : MDM
        MDM model object.
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
        Figure size. Default is (8, 8).
    layout_seed : int, optional
        Random seed for graph layout. Use None for non-deterministic layout.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    adj_mat = mdm_object.adj_mat
    n = adj_mat.shape[0]

    if node_labels is None:
        if hasattr(mdm_object, 'node_names') and mdm_object.node_names:
            node_labels = mdm_object.node_names
        else:
            node_labels = [f"V{i+1}" for i in range(n)]

    if figsize is None:
        figsize = (8, 8)

    if plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(adj_mat, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(node_labels, rotation=45, ha='right')
        ax.set_yticklabels(node_labels)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Edge')

        ax.set_xlabel('Child', fontsize=12)
        ax.set_ylabel('Parent', fontsize=12)
        ax.set_title('DAG Structure (Adjacency Matrix)', fontsize=14)

        plt.tight_layout()
        return fig

    else:  # graph
        fig, ax = plt.subplots(figsize=figsize)

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Add edges
        edges = []
        for i in range(n):
            for j in range(n):
                if adj_mat[i, j] == 1:
                    G.add_edge(i, j)
                    edges.append((i, j))

        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=layout_seed)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_color,
            node_size=2000, alpha=0.9
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=edge_color,
            arrows=True, arrowsize=arrow_size*10,
            arrowstyle='->', width=2, alpha=0.7
        )

        # Draw labels
        labels = {i: node_labels[i] for i in range(n)}
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_color=label_color, font_weight='bold', font_size=5
        )

        ax.set_title('DAG Structure', fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        return fig
