"""
DAG visualization functions for MDM models.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, List, Literal, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    pass

# Publication Graphviz defaults (match Soft Impacts / retail experiment figures).
_GRAPHVIZ_NODE_COLOR = "#4682B4"  # steelblue
_GRAPHVIZ_EDGE_COLOR = "#333333"
_GRAPHVIZ_LABEL_COLOR = "white"


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

    try:
        return nx.nx_pydot.graphviz_layout(G, prog="dot")
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


def _build_pydot_dag(
    adj_mat: np.ndarray,
    node_labels: List[str],
    *,
    node_color: str,
    edge_color: str,
    label_color: str,
) -> Any:
    """Build a Graphviz digraph via pydot (circle nodes, curved splines)."""
    try:
        import pydot
    except ImportError as exc:
        raise ImportError(
            "style='graphviz' requires the pydot package and a Graphviz "
            "installation (the `dot` binary on PATH). "
            "Install with `pip install pydot` and Graphviz from "
            "https://graphviz.org/download/."
        ) from exc

    n = len(node_labels)
    graph = pydot.Dot(
        graph_type="digraph",
        rankdir="TB",
        splines="true",
        overlap="false",
        concentrate="true",
        nodesep="0.55" if n > 10 else "0.7",
        ranksep="0.9" if n > 10 else "1.0",
        bgcolor="white",
        pad="0.35",
    )
    fontsize = "11" if n <= 8 else ("8" if n > 12 else "9")
    width = "1.15" if n <= 8 else ("0.95" if n > 12 else "1.0")

    for i, lab in enumerate(node_labels):
        # Numeric ids avoid Graphviz issues with special characters in labels.
        graph.add_node(
            pydot.Node(
                str(i),
                label=str(lab),
                shape="circle",
                style="filled",
                fillcolor=node_color,
                fontcolor=label_color,
                fontsize=fontsize,
                fontname="Helvetica-Bold",
                width=width,
                height=width,
                fixedsize="true",
            )
        )

    for i in range(n):
        for j in range(n):
            if adj_mat[i, j] != 0:
                graph.add_edge(
                    pydot.Edge(
                        str(i),
                        str(j),
                        color=edge_color,
                        penwidth="1.7",
                        arrowsize="0.85",
                    )
                )
    return graph


def _plot_dag_graphviz(
    adj_mat: np.ndarray,
    node_labels: List[str],
    *,
    node_color: str,
    edge_color: str,
    label_color: str,
    figsize: Optional[tuple],
) -> Figure:
    """Render a Graphviz DAG into a matplotlib figure."""
    graph = _build_pydot_dag(
        adj_mat,
        node_labels,
        node_color=node_color,
        edge_color=edge_color,
        label_color=label_color,
    )
    try:
        png_bytes = graph.create_png()
    except Exception as exc:
        raise RuntimeError(
            "Graphviz failed to render the DAG. Ensure the `dot` binary is "
            "installed and on PATH (https://graphviz.org/download/)."
        ) from exc

    img = mpimg.imread(BytesIO(png_bytes), format="png")
    n = len(node_labels)
    if figsize is None:
        side = min(14.0, 6.0 + 0.35 * n)
        figsize = (side, side * 0.85)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("DAG Structure", fontsize=14)
    fig.tight_layout()
    return fig


def plot_dag(
    mdm_object: Any,
    node_labels: Optional[List[str]] = None,
    plot_type: Literal["graph", "heatmap"] = "graph",
    show_legend: bool = False,
    edge_color: Optional[str] = None,
    node_color: Optional[str] = None,
    label_color: Optional[str] = None,
    arrow_size: float = 4.0,
    figsize: Optional[tuple] = None,
    layout_seed: Optional[int] = 0,
    *,
    style: Literal["networkx", "graphviz"] = "networkx",
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
        Typically :class:`mdmp.model.MDM`, or :class:`mdmp.group_analysis.ISAggregatedMDMView`
        / the return value of :func:`mdmp.group_analysis.aggregate_individual_structures`
        (needs ``adj_mat`` and optionally ``node_names``).
    node_labels : list of str, optional
        Custom node labels. If None, uses MDM node names.
    plot_type : {"graph", "heatmap"}, optional
        Type of plot. Default is "graph".
    show_legend : bool, optional
        Whether to show legend. Default is False.
    edge_color : str, optional
        Color of edges. Defaults depend on ``style`` (black for ``networkx``,
        ``#333333`` for ``graphviz``).
    node_color : str, optional
        Color of nodes. Defaults depend on ``style`` (``steelblue`` /
        ``#4682B4``).
    label_color : str, optional
        Color of node labels. Default is white.
    arrow_size : float, optional
        Size of arrow heads (``networkx`` style only). Default is 4.0.
    figsize : tuple, optional
        Figure size. Default scales lightly with node count for ``graph``.
    layout_seed : int, optional
        Random seed for spring layout (cycles / ``hierarchical=False``). Use None for nondeterministic layout.
    style : {"networkx", "graphviz"}, optional
        Graph rendering backend when ``plot_type="graph"``.

        - ``"networkx"`` (default): Matplotlib + NetworkX drawing.
        - ``"graphviz"``: Graphviz ``dot`` layout with circular filled nodes and
          curved edge routing (publication style). Requires ``pydot`` and the
          Graphviz ``dot`` binary.
    hierarchical : bool, optional
        If True (default), use layered layout for ``style="networkx"``
        (Graphviz ``dot`` positions when available via ``pygraphviz`` or
        ``pydot``, else topological generations). If False, use
        ``spring_layout``. Ignored when ``style="graphviz"``.
    level_gap : float, optional
        Horizontal/vertical spacing between layers in the topological fallback
        (ignored when Graphviz layout succeeds).
    node_size : float, optional
        Node diameter passed to :func:`networkx.draw_networkx_nodes`
        (``networkx`` style only).
    font_size : int, optional
        Font size for node labels on the graph (``networkx`` style only).
    edge_width : float, optional
        Width of directed edges (``networkx`` style only).

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

    if style == "graphviz":
        return _plot_dag_graphviz(
            adj_mat,
            node_labels,
            node_color=node_color or _GRAPHVIZ_NODE_COLOR,
            edge_color=edge_color or _GRAPHVIZ_EDGE_COLOR,
            label_color=label_color or _GRAPHVIZ_LABEL_COLOR,
            figsize=figsize,
        )

    if style != "networkx":
        raise ValueError(f"Unknown style {style!r}; use 'networkx' or 'graphviz'.")

    if edge_color is None:
        edge_color = "black"
    if node_color is None:
        node_color = "steelblue"
    if label_color is None:
        label_color = "white"

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
        arrowstyle="-|>",
        width=edge_width,
        alpha=0.75,
        connectionstyle="arc3,rad=0.06",
        min_source_margin=12,
        min_target_margin=14,
        node_size=node_size,
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
