"""
Shared styling and display helpers for time-series plots.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from scipy.interpolate import PchipInterpolator

# Okabe–Ito (colour-blind friendly)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#757575",
}

PARAM_COLORS: tuple[str, ...] = (
    OKABE_ITO["orange"],
    OKABE_ITO["blue"],
    OKABE_ITO["green"],
    OKABE_ITO["red"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
)

OBSERVED_COLOR = OKABE_ITO["gray"]
STREAM_ALPHA = 0.78
PARAM_BAND_ALPHA = 0.12
DEFAULT_SMOOTH_FACTOR = 5


def _param_names(
    mdm_object: Any,
    node_idx: int,
    distribution: Literal["filt", "smoo"],
) -> list[str]:
    """Return raw parameter names for one node."""
    name_src = mdm_object.Filt if distribution == "filt" else mdm_object.Smoo
    row_names = name_src.get("row_names") or {}
    if node_idx in row_names:
        return [str(n) for n in row_names[node_idx]]
    node_label = (
        mdm_object.node_names[node_idx]
        if hasattr(mdm_object, "node_names")
        else str(node_idx)
    )
    return [f"beta0_{node_label}"]


def format_param_label(name: str) -> str:
    """Human-readable legend label for a parameter name."""
    if "beta0" in name or name.lower() == "intercept":
        return "intercept"
    if "->" in name:
        return name.replace("->", "→")
    return name


def param_legend_labels(
    mdm_object: Any,
    node_idx: int,
    distribution: Literal["filt", "smoo"],
) -> list[str]:
    """Legend labels for all parameters of a node."""
    return [format_param_label(n) for n in _param_names(mdm_object, node_idx, distribution)]


def upsample_curve(
    time: np.ndarray,
    values: np.ndarray,
    *,
    factor: int = DEFAULT_SMOOTH_FACTOR,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample a time series for smooth display (visual only; does not alter estimates).

    Uses shape-preserving PCHIP interpolation so sharp turns are not overshot.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if not smooth or factor <= 1 or len(time) < 4:
        return time, values
    fine_time = np.linspace(time[0], time[-1], len(time) * factor)
    interp = PchipInterpolator(time, values)
    return fine_time, interp(fine_time)


def style_time_series_ax(
    ax: Axes,
    *,
    xlabel: str = "Time  $t$",
    ylabel: str,
    show_zero_line: bool = True,
    grid_alpha: float = 0.25,
    legend: bool = True,
) -> None:
    """Apply consistent, publication-friendly axis styling."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if show_zero_line:
        ax.axhline(
            0,
            color=OKABE_ITO["gray"],
            lw=0.8,
            ls="--",
            alpha=0.6,
            zorder=0,
        )
    ax.grid(True, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)


def param_color(index: int) -> str:
    """Cycle through the default parameter colour palette."""
    return PARAM_COLORS[index % len(PARAM_COLORS)]
