"""
Parameter plotting functions for MDM models.
"""

from typing import TYPE_CHECKING, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats

if TYPE_CHECKING:
    from ..model import MDM


def plot_arcs(
    mdm_object: "MDM",
    plot_type: Literal["connections", "intercepts", "all"] = "connections",
    distribution: Literal["filt", "smoo"] = "filt",
    ci_level: float = 0.95,
    figsize: Optional[tuple] = None
) -> Figure:
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
    mdm_object: "MDM",
    target_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    scale_series: bool = False,
    figsize: Optional[tuple] = None
) -> Figure:
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
    mdm_object: "MDM",
    child_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    figsize: Optional[tuple] = None
) -> Figure:
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
