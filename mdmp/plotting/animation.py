"""
Animation plotting functions for MDM models.
"""

from typing import TYPE_CHECKING, Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..model import MDM


def plot_idag(
    mdm_object: "MDM",
    output_gif: str = "mdm_dynamic.gif",
    fps: int = 10,
    width: int = 6,
    height: int = 6,
    dpi: int = 100,
    distribution: Literal["filt", "smoo"] = "filt"
) -> animation.FuncAnimation:
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
    ax.set_title('Time: 0', fontsize=12)
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
