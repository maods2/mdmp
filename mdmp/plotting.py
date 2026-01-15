"""
Plotting functions for MDM visualization.

This module provides visualization tools for MDM models including DAG structure,
dynamic parameters, marginal posteriors, stream plots, and animated heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
from typing import Optional, Union, List, Literal
from pathlib import Path


def plot_dag(
    mdm_object,
    node_labels: Optional[List[str]] = None,
    plot_type: Literal["graph", "heatmap"] = "graph",
    show_legend: bool = False,
    edge_color: str = "black",
    node_color: str = "steelblue",
    label_color: str = "white",
    arrow_size: float = 4.0,
    figsize: Optional[tuple] = None
    ):
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
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        
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
            font_color=label_color, font_weight='bold', font_size=10
        )
        
        ax.set_title('DAG Structure', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig


def plot_arcs(
    mdm_object,
    plot_type: Literal["connections", "intercepts", "all"] = "connections",
    distribution: Literal["filt", "smoo"] = "filt",
    ci_level: float = 0.95,
    figsize: Optional[tuple] = None
):
    """
    Plot dynamic parameters over time.

    Parameters
    ----------
    mdm_object : MDM
        MDM model object.
    plot_type : {"connections", "intercepts", "all"}, optional
        Which parameters to plot. Default is "connections".
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    ci_level : float, optional
        Confidence interval level. Default is 0.95.
    figsize : tuple, optional
        Figure size. Default is (12, 8).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    from scipy import stats

    if distribution == "filt":
        mt_list = mdm_object.Filt['mt']
        Ct_list = mdm_object.Filt['Ct']
        nt_list = mdm_object.Filt['nt']
        use_se = False
    else:
        mt_list = mdm_object.Smoo['smt']
        Ct_list = mdm_object.Smoo['sCt']
        SE_list = mdm_object.Smoo['SE']
        use_se = True

    if figsize is None:
        figsize = (12, 8)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    plot_idx = 0
    for node in range(len(mt_list)):
        mt_node = mt_list[node]
        Ct_node = Ct_list[node]

        if mt_node.ndim == 1:
            mt_node = mt_node.reshape(1, -1)

        if distribution == "filt" and Ct_node.ndim == 1:
            Ct_node = Ct_node.reshape(1, 1, -1)

        # Get parameter names
        if 'row_names' in mdm_object.Filt and node in mdm_object.Filt['row_names']:
            param_names = mdm_object.Filt['row_names'][node]
        else:
            param_names = [f"beta0_{mdm_object.node_names[node] if hasattr(mdm_object, 'node_names') else node}"]

        T = mt_node.shape[1]
        time = np.arange(T)

        for param in range(mt_node.shape[0]):
            if plot_idx >= len(axes):
                break

            name = param_names[param] if param < len(param_names) else f"param_{param}"
            is_intercept = "beta0" in name
            is_connection = "->" in str(name)

            if (plot_type == "connections" and not is_connection) or \
               (plot_type == "intercepts" and not is_intercept) or \
               (plot_type == "all" and not (is_connection or is_intercept)):
                continue

            ax = axes[plot_idx]
            mean_vals = mt_node[param, :]

            if distribution == "filt":
                if Ct_node.ndim == 3:
                    var_vals = Ct_node[param, param, :]
                else:
                    var_vals = Ct_node[param, param] if Ct_node.ndim == 2 else Ct_node
                nt_node = nt_list[node]
                t_crit = stats.t.ppf((1 + ci_level) / 2, nt_node) * np.sqrt(var_vals)
            else:
                if use_se:
                    if isinstance(SE_list[node], np.ndarray):
                        if SE_list[node].ndim == 1:
                            t_crit = SE_list[node]
                        else:
                            t_crit = SE_list[node][:, param]
                    else:
                        t_crit = np.zeros(T)
                else:
                    t_crit = np.zeros(T)

            # Plot
            ax.plot(time, mean_vals, label=name, linewidth=2)
            ax.fill_between(
                time, mean_vals - t_crit, mean_vals + t_crit,
                alpha=0.3, label=f"{ci_level*100:.0f}% CI"
            )
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Parameter Value', fontsize=10)
            ax.set_title(str(name), fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            plot_idx += 1
            if plot_idx >= len(axes):
                break

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_marginal(
    mdm_object,
    target_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    scale_series: bool = False,
    figsize: Optional[tuple] = None
):
    """
    Plot marginal posterior for a target node.

    Parameters
    ----------
    mdm_object : MDM
        MDM model object.
    target_node : int
        Index of target node.
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    scale_series : bool, optional
        Whether to scale time series. Default is False.
    figsize : tuple, optional
        Figure size. Default is (10, 6).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    if distribution == "filt":
        mt_node = mdm_object.Filt['mt'][target_node]
    else:
        mt_node = mdm_object.Smoo['smt'][target_node]

    if mt_node.ndim == 1:
        mt_node = mt_node.reshape(1, -1)

    T = mt_node.shape[1]
    time = np.arange(T)

    # Plot observed data
    observed = mdm_object.data[:, target_node]
    if scale_series:
        observed = (observed - np.mean(observed)) / np.std(observed)
    ax.plot(time, observed, 'k-', alpha=0.5, label='Observed', linewidth=1)

    # Plot parameters
    for param in range(mt_node.shape[0]):
        param_vals = mt_node[param, :]
        if scale_series:
            param_vals = (param_vals - np.mean(param_vals)) / np.std(param_vals)
        ax.plot(time, param_vals, label=f'Parameter {param}', linewidth=2)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(
        f'Marginal Posterior: Node {mdm_object.node_names[target_node] if hasattr(mdm_object, "node_names") else target_node}',
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stream(
    mdm_object,
    child_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    figsize: Optional[tuple] = None
):
    """
    Plot stream plot showing parent contributions to a child node.

    Parameters
    ----------
    mdm_object : MDM
        MDM model object.
    child_node : int
        Index of child node.
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    figsize : tuple, optional
        Figure size. Default is (12, 6).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if figsize is None:
        figsize = (12, 6)

    fig, ax = plt.subplots(figsize=figsize)

    if distribution == "filt":
        mt_node = mdm_object.Filt['mt'][child_node]
    else:
        mt_node = mdm_object.Smoo['smt'][child_node]

    if mt_node.ndim == 1:
        mt_node = mt_node.reshape(1, -1)

    T = mt_node.shape[1]
    time = np.arange(T)

    # Get parents
    parents = np.where(mdm_object.adj_mat[:, child_node] > 0)[0]

    # Stack plot
    ax.stackplot(
        time, *[mt_node[i, :] for i in range(mt_node.shape[0])],
        labels=[f'Param {i}' for i in range(mt_node.shape[0])],
        alpha=0.7
    )

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Contribution', fontsize=12)
    ax.set_title(
        f'Parent Contributions to Node {mdm_object.node_names[child_node] if hasattr(mdm_object, "node_names") else child_node}',
        fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_idag(
    mdm_object,
    output_gif: str = "mdm_dynamic.gif",
    fps: int = 10,
    width: int = 6,
    height: int = 6,
    dpi: int = 100,
    distribution: Literal["filt", "smoo"] = "filt"
):
    """
    Create animated heatmap of dynamic parameters over time.

    Parameters
    ----------
    mdm_object : MDM
        MDM model object.
    output_gif : str, optional
        Output GIF filename. Default is "mdm_dynamic.gif".
    fps : int, optional
        Frames per second. Default is 10.
    width : int, optional
        Frame width in inches. Default is 6.
    height : int, optional
        Frame height in inches. Default is 6.
    dpi : int, optional
        Resolution. Default is 100.
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object.
    """
    if distribution == "filt":
        mt_list = mdm_object.Filt['mt']
    else:
        mt_list = mdm_object.Smoo['smt']

    # Get number of time points
    T = len(mdm_object.data)

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Get adjacency matrix dimensions
    n = mdm_object.adj_mat.shape[0]

    # Prepare data for animation
    frames_data = []
    for t in range(T):
        # Extract parameter values at time t
        param_mat = np.zeros((n, n))
        for node in range(n):
            mt_node = mt_list[node]
            if mt_node.ndim == 1:
                mt_node = mt_node.reshape(1, -1)
            
            # Map parameters to adjacency structure
            # (Simplified: just use intercepts and connections)
            for param_idx in range(mt_node.shape[0]):
                if param_idx == 0:  # Intercept
                    param_mat[node, node] = mt_node[param_idx, t]
                else:
                    # Find which parent this parameter corresponds to
                    parents = np.where(mdm_object.adj_mat[:, node] > 0)[0]
                    if param_idx - 1 < len(parents):
                        parent = parents[param_idx - 1]
                        param_mat[parent, node] = mt_node[param_idx, t]

        frames_data.append(param_mat)

    # Find value range
    vmin = min([np.min(data) for data in frames_data])
    vmax = max([np.max(data) for data in frames_data])

    im = ax.imshow(frames_data[0], cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Time: 0', fontsize=12)
    plt.colorbar(im, ax=ax)

    def animate(frame):
        ax.clear()
        im = ax.imshow(frames_data[frame], cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f'Time: {frame}', fontsize=12)
        return im

    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=1000/fps, blit=False, repeat=True
    )

    # Save animation
    if output_gif:
        anim.save(output_gif, writer='pillow', fps=fps, dpi=dpi)

    return anim

