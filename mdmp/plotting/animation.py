"""
Animation plotting functions for MDM models.
"""

from typing import TYPE_CHECKING, Any, List, Literal, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ._input_checks import require_data_for_plot, require_filt_for_plot, require_smoo_for_plot

if TYPE_CHECKING:
    pass


def _resolve_node_labels(mdm_object: Any, n: int) -> List[str]:
    if hasattr(mdm_object, "node_names") and mdm_object.node_names:
        return list(mdm_object.node_names)
    return [f"V{i + 1}" for i in range(n)]


def _style_idag_axes(ax: plt.Axes, node_labels: List[str]) -> None:
    ax.set_xticks(range(len(node_labels)))
    ax.set_yticks(range(len(node_labels)))
    ax.set_xticklabels(node_labels, rotation=45, ha="right")
    ax.set_yticklabels(node_labels)
    ax.set_xlabel("Child", fontsize=12)
    ax.set_ylabel("Parent", fontsize=12)


def _row_names_for_idag(mdm_object: Any, distribution: Literal["filt", "smoo"]) -> dict:
    """Parameter names: smoothed runs use ``Filt`` names when ``Smoo`` has none."""
    filt_names = (getattr(mdm_object, "Filt", None) or {}).get("row_names") or {}
    if distribution == "filt":
        return filt_names
    smoo_names = (getattr(mdm_object, "Smoo", None) or {}).get("row_names") or {}
    return smoo_names or filt_names


def _param_matrix_at_time(
    mdm_object: Any,
    mt_list: dict,
    row_names: dict,
    node_labels: List[str],
    t: int,
    *,
    show_intercepts: bool,
) -> np.ndarray:
    """Build parent×child matrix at time ``t`` using ``row_names`` (same as ``plot_arcs``)."""
    adj = np.asarray(mdm_object.adj_mat)
    n = adj.shape[0]
    name_to_idx = {name: i for i, name in enumerate(node_labels)}
    param_mat = np.full((n, n), np.nan)

    for node in range(n):
        mt_node = mt_list[node]
        if mt_node.ndim == 1:
            mt_node = mt_node.reshape(1, -1)

        if node in row_names:
            param_names = row_names[node]
        else:
            param_names = [f"beta0_{node_labels[node]}"]

        for param_idx in range(mt_node.shape[0]):
            name = (
                param_names[param_idx]
                if param_idx < len(param_names)
                else f"param_{param_idx}"
            )
            name_str = str(name)
            if "->" in name_str:
                parent_s, child_s = name_str.split("->", 1)
                if parent_s not in name_to_idx or child_s not in name_to_idx:
                    continue
                i, j = name_to_idx[parent_s], name_to_idx[child_s]
                if adj[i, j] > 0:
                    param_mat[i, j] = mt_node[param_idx, t]
            elif show_intercepts and "beta0" in name_str:
                param_mat[node, node] = mt_node[param_idx, t]

    return param_mat


def plot_idag(
    mdm_object: Any,
    output_gif: str = "mdm_dynamic.gif",
    fps: int = 10,
    width: int = 6,
    height: int = 6,
    dpi: int = 100,
    distribution: Literal["filt", "smoo"] = "filt",
    node_labels: Optional[List[str]] = None,
    colorbar_label: Optional[str] = None,
    show_intercepts: bool = False,
) -> animation.FuncAnimation:
    """
    Create animated heatmap of dynamic parameters over time.

    Parameters
    ----------
    mdm_object
        :class:`mdmp.model.MDM` or an IS aggregation view with ``data`` and ``Filt``/``Smoo``.
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
    node_labels : list of str, optional
        Axis tick labels. If None, uses ``mdm_object.node_names`` or ``V1``, ….
    colorbar_label : str, optional
        Label for the color scale. Default depends on ``distribution``.
    show_intercepts : bool, optional
        If True, place intercept (``beta0``) coefficients on the diagonal.
        Default is False so the heatmap shows only parent→child edges from
        ``adj_mat`` (empty cells are masked).

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object.
    """
    require_data_for_plot(mdm_object, plot_kw="time_series=...")
    if distribution == "filt":
        require_filt_for_plot(mdm_object, plot_kw="plot_filt=...")
        mt_list = mdm_object.Filt['mt']
    else:
        require_smoo_for_plot(mdm_object)
        mt_list = mdm_object.Smoo['smt']

    # Get number of time points
    T = len(mdm_object.data)

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    n = mdm_object.adj_mat.shape[0]
    labels = list(node_labels) if node_labels is not None else _resolve_node_labels(mdm_object, n)
    if len(labels) != n:
        raise ValueError(f"node_labels length {len(labels)} != number of nodes {n}")

    if colorbar_label is None:
        colorbar_label = (
            "Filtered dynamic parameter" if distribution == "filt" else "Smoothed dynamic parameter"
        )

    row_names = _row_names_for_idag(mdm_object, distribution)

    frames_data = [
        _param_matrix_at_time(
            mdm_object, mt_list, row_names, labels, t, show_intercepts=show_intercepts
        )
        for t in range(T)
    ]

    finite = np.concatenate(
        [d[np.isfinite(d)].ravel() for d in frames_data if np.any(np.isfinite(d))]
    )
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmin == vmax:
            vmax = vmin + 1.0 if vmin == 0 else vmin * 1.01

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#f5f5f5")

    im = ax.imshow(
        frames_data[0],
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    _style_idag_axes(ax, labels)
    ax.set_title("Time: 0", fontsize=12)
    fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()

    def animate(frame: int):
        im.set_data(frames_data[frame])
        ax.set_title(f"Time: {frame}", fontsize=12)
        return [im]

    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=1000 / fps, blit=False, repeat=True
    )

    # Save animation
    if output_gif:
        anim.save(output_gif, writer='pillow', fps=fps, dpi=dpi)

    return anim
