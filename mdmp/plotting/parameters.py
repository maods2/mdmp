"""
Parameter plotting functions for MDM models.
"""

from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

from ._input_checks import (
    require_data_for_plot,
    require_filt_for_plot,
    require_smoo_for_plot,
)
from ._style import (
    OBSERVED_COLOR,
    OKABE_ITO,
    STREAM_ALPHA,
    PARAM_BAND_ALPHA,
    param_color,
    param_legend_labels,
    style_time_series_ax,
    upsample_curve,
)
from ..anomaly import detect_anomalies

if TYPE_CHECKING:
    pass

_UNSET = object()  # sentinel: use built-in default title


_MAX_ARC_COLS = 4


def _grid_shape(n_panels: int, *, max_cols: int = _MAX_ARC_COLS) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for ``n_panels`` subplots with at most ``max_cols`` per row."""
    if n_panels <= 0:
        return (1, 1)
    ncols = min(max_cols, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    return (nrows, ncols)


def _default_plot_arcs_figsize(nrows: int, ncols: int) -> tuple[float, float]:
    """Scale default figure size with the subplot grid."""
    width = min(18.0, 3.2 * ncols + 2.0)
    height = min(22.0, 2.8 * nrows + 1.5)
    return (width, height)


def _count_matching_arc_panels(
    mdm_object: Any,
    mt_list: list,
    distribution: Literal["filt", "smoo"],
    plot_type: Literal["connections", "intercepts", "all"],
) -> int:
    """Count parameters that would be drawn for the given ``plot_type`` filter."""
    n = 0
    for node in range(len(mt_list)):
        mt_node = mt_list[node]
        if mt_node.ndim == 1:
            mt_node = mt_node.reshape(1, -1)

        name_src = mdm_object.Filt if distribution == "filt" else mdm_object.Smoo
        row_names = name_src.get("row_names") or {}
        if node in row_names:
            param_names = row_names[node]
        else:
            param_names = [
                f"beta0_{mdm_object.node_names[node] if hasattr(mdm_object, 'node_names') else node}"
            ]

        for param in range(mt_node.shape[0]):
            name = param_names[param] if param < len(param_names) else f"param_{param}"
            is_intercept = "beta0" in name
            is_connection = "->" in str(name)

            if (plot_type == "connections" and not is_connection) or (
                plot_type == "intercepts" and not is_intercept
            ) or (plot_type == "all" and not (is_connection or is_intercept)):
                continue
            n += 1
    return n


def plot_arcs(
    mdm_object: Any,
    plot_type: Literal["connections", "intercepts", "all"] = "connections",
    distribution: Literal["filt", "smoo"] = "filt",
    ci_level: float = 0.95,
    figsize: Optional[tuple] = None
) -> Figure:
    """
    Plot dynamic parameters over time.

    Parameters
    ----------
    mdm_object
        :class:`mdmp.model.MDM` or an IS aggregation view with ``Filt`` / ``Smoo``.
    plot_type : {"connections", "intercepts", "all"}, optional
        Which parameters to plot. Default is "connections".
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    ci_level : float, optional
        Confidence interval level. Default is 0.95.
    figsize : tuple, optional
        Figure size. If omitted, a default is chosen from the subplot grid
        (at most four columns; unused axes are hidden).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if distribution == "filt":
        require_filt_for_plot(mdm_object, plot_kw="plot_filt=...")
        mt_list = mdm_object.Filt['mt'] # Posterior means
        Ct_list = mdm_object.Filt['Ct'] # Posterior variances
        nt_list = mdm_object.Filt['nt'] # Precision hyperparameters
        use_se = False
    else:
        require_smoo_for_plot(mdm_object)
        mt_list = mdm_object.Smoo['smt'] # Smoothed means
        Ct_list = mdm_object.Smoo['sCt'] # Smoothed variances
        SE_list = mdm_object.Smoo['SE'] # Standard errors
        use_se = True

    n_panels = _count_matching_arc_panels(mdm_object, mt_list, distribution, plot_type)
    nrows, ncols = _grid_shape(n_panels)
    if figsize is None:
        figsize = _default_plot_arcs_figsize(nrows, ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    plot_idx = 0
    for node in range(len(mt_list)):
        mt_node = mt_list[node]
        Ct_node = Ct_list[node]

        if mt_node.ndim == 1:
            mt_node = mt_node.reshape(1, -1)

        if distribution == "filt" and Ct_node.ndim == 1:
            Ct_node = Ct_node.reshape(1, 1, -1)

        # Get parameter names
        name_src = mdm_object.Filt if distribution == "filt" else mdm_object.Smoo
        row_names = name_src.get("row_names") or {}
        if node in row_names:
            param_names = row_names[node]
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

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_marginal(
    mdm_object: Any,
    target_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    scale_series: bool = False,
    figsize: Optional[tuple] = None,
    smooth: bool = True,
    smooth_factor: int = 5,
    *,
    title: Optional[str] = _UNSET,  # type: ignore[assignment]
) -> Figure:
    """
    Plot marginal posterior for a target node.

    Parameters
    ----------
    mdm_object
        :class:`mdmp.model.MDM` or an IS aggregation view with ``data`` and ``Filt``/``Smoo``.
    target_node : int
        Index of target node.
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    scale_series : bool, optional
        Whether to scale time series. Default is False.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    smooth : bool, optional
        Upsample parameter curves for smooth display (visual only). Default is True.
    smooth_factor : int, optional
        Upsampling factor when ``smooth`` is True. Default is 5.
    title : str or None, optional
        Axes title. Defaults to ``"Marginal posterior: node <name>"``.
        Pass ``None`` to omit a title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if figsize is None:
        figsize = (10, 6)

    require_data_for_plot(mdm_object, plot_kw="time_series=...")
    if distribution == "filt":
        require_filt_for_plot(mdm_object, plot_kw="plot_filt=...")
    else:
        require_smoo_for_plot(mdm_object)

    fig, ax = plt.subplots(figsize=figsize)

    if distribution == "filt":
        mt_node = mdm_object.Filt['mt'][target_node]
    else:
        mt_node = mdm_object.Smoo['smt'][target_node]

    if mt_node.ndim == 1:
        mt_node = mt_node.reshape(1, -1)

    T = mt_node.shape[1]
    time = np.arange(T)
    labels = param_legend_labels(mdm_object, target_node, distribution)

    # Observed series (raw data — not smoothed)
    observed = mdm_object.data[:, target_node]
    if scale_series:
        observed = (observed - np.mean(observed)) / np.std(observed)
    ax.plot(
        time,
        observed,
        color=OBSERVED_COLOR,
        alpha=0.45,
        label="observed",
        linewidth=1.0,
        zorder=1,
    )

    # Posterior parameter means
    for param in range(mt_node.shape[0]):
        param_vals = mt_node[param, :]
        if scale_series:
            param_vals = (param_vals - np.mean(param_vals)) / np.std(param_vals)
        plot_t, plot_y = upsample_curve(
            time, param_vals, factor=smooth_factor, smooth=smooth
        )
        color = param_color(param)
        label = labels[param] if param < len(labels) else f"param {param}"
        y_span = float(np.nanmax(plot_y) - np.nanmin(plot_y))
        band = max(0.02 * y_span, 0.05 * np.nanstd(plot_y))
        ax.fill_between(
            plot_t,
            plot_y - band,
            plot_y + band,
            color=color,
            alpha=PARAM_BAND_ALPHA,
            linewidth=0,
            zorder=param + 1,
        )
        ax.plot(
            plot_t,
            plot_y,
            color=color,
            label=label,
            linewidth=2.0,
            zorder=param + 2,
        )

    node_label = (
        mdm_object.node_names[target_node]
        if hasattr(mdm_object, "node_names")
        else target_node
    )
    if title is _UNSET:
        title = f"Marginal posterior: node {node_label}"
    if title is not None:
        ax.set_title(title, fontsize=12, pad=10)
    style_time_series_ax(ax, ylabel="Parameter")

    plt.tight_layout()
    return fig


def plot_stream(
    mdm_object: Any,
    child_node: int,
    distribution: Literal["filt", "smoo"] = "filt",
    figsize: Optional[tuple] = None,
    smooth: bool = True,
    smooth_factor: int = 5,
    *,
    title: Optional[str] = _UNSET,  # type: ignore[assignment]
) -> Figure:
    """
    Plot stream plot showing parent contributions to a child node.

    Parameters
    ----------
    mdm_object
        :class:`mdmp.model.MDM` or an IS aggregation view with ``Filt`` / ``Smoo``.
    child_node : int
        Index of child node.
    distribution : {"filt", "smoo"}, optional
        Use filtered or smoothed estimates. Default is "filt".
    figsize : tuple, optional
        Figure size. Default is (12, 6).
    smooth : bool, optional
        Upsample contribution curves for smooth display (visual only). Default is True.
    smooth_factor : int, optional
        Upsampling factor when ``smooth`` is True. Default is 5.
    title : str or None, optional
        Axes title. Defaults to ``"Parent contributions to node <name>"``.
        Pass ``None`` to omit a title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if figsize is None:
        figsize = (12, 6)

    if distribution == "filt":
        require_filt_for_plot(mdm_object, plot_kw="plot_filt=...")
    else:
        require_smoo_for_plot(mdm_object)

    fig, ax = plt.subplots(figsize=figsize)

    if distribution == "filt":
        mt_node = mdm_object.Filt['mt'][child_node]
    else:
        mt_node = mdm_object.Smoo['smt'][child_node]

    if mt_node.ndim == 1:
        mt_node = mt_node.reshape(1, -1)

    T = mt_node.shape[1]
    time = np.arange(T, dtype=float)
    labels = param_legend_labels(mdm_object, child_node, distribution)
    n_params = mt_node.shape[0]

    plot_time = time
    layers = []
    for i in range(n_params):
        plot_t, plot_y = upsample_curve(
            time, mt_node[i, :], factor=smooth_factor, smooth=smooth
        )
        if i == 0:
            plot_time = plot_t
        layers.append(plot_y)

    colors = [param_color(i) for i in range(n_params)]
    ax.stackplot(
        plot_time,
        *layers,
        colors=colors,
        alpha=STREAM_ALPHA,
        linewidth=0,
        edgecolor="none",
    )

    node_label = (
        mdm_object.node_names[child_node]
        if hasattr(mdm_object, "node_names")
        else child_node
    )
    if title is _UNSET:
        title = f"Parent contributions to node {node_label}"
    if title is not None:
        ax.set_title(title, fontsize=12, pad=10)
    style_time_series_ax(
        ax,
        ylabel="Contribution",
        show_zero_line=False,
        grid_alpha=0.2,
        legend=False,
    )
    ax.legend(
        handles=[
            Patch(
                facecolor=colors[i],
                edgecolor="none",
                alpha=STREAM_ALPHA,
                label=labels[i] if i < len(labels) else f"param {i}",
            )
            for i in range(n_params)
        ],
        fontsize=9,
        loc="upper right",
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig


def plot_anomalies(
    mdm_object: Any,
    series: Union[int, str] = 0,
    *,
    ci_level: float = 0.95,
    figsize: Optional[tuple] = None,
    ax: Optional[Axes] = None,
    time_index: Optional[Any] = None,
    show_observed_markers: bool = True,
    observed_lw: float = 1.0,
    mean_lw: float = 1.6,
    anomaly_size: float = 36.0,
    band_alpha: Optional[float] = None,
) -> Figure:
    """
    Plot observed series against the MDM predictive mean and interval.

    Calls :func:`mdmp.anomaly.detect_anomalies` for the selected node and draws:

    - observed values as a connected polyline (optional circular markers;
      no smoothing or upsampling)
    - one-step predictive mean and ``ci_level`` band
    - anomalies marked with ``x``

    Parameters
    ----------
    mdm_object
        Fitted :class:`~mdmp.model.MDM` (or compatible) with ``data`` and ``Filt``.
    series : int or str, optional
        Node index or name. Default is ``0``.
    ci_level : float, optional
        Predictive interval level. Default is ``0.95``.
    figsize : tuple, optional
        Figure size when ``ax`` is not provided. Default is ``(10, 4)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If omitted, a new figure is created.
    time_index : sequence, optional
        Optional x-axis labels of length ``T`` (e.g. dates).
    show_observed_markers : bool, optional
        Draw circular markers on the observed series. Default is ``True``.
        Set ``False`` for dense multi-panel figures.
    observed_lw, mean_lw : float, optional
        Line widths for observed and predictive mean.
    anomaly_size : float, optional
        Marker size for anomaly ``x`` marks.
    band_alpha : float, optional
        Opacity of the predictive interval fill. Default uses the shared
        style constant.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the axes.

    Examples
    --------
    >>> from mdmp import MDM, plot_anomalies  # doctest: +SKIP
    >>> fig = plot_anomalies(model, series=0, ci_level=0.95)  # doctest: +SKIP
    """
    if series is None:
        raise TypeError("plot_anomalies requires a single series (int or str).")

    result = detect_anomalies(
        mdm_object,
        ci_level=ci_level,
        series=series,
        output="result",
        time_index=time_index,
    )
    y = np.asarray(result.observed, dtype=float).ravel()
    mean = np.asarray(result.fitted_mean, dtype=float).ravel()
    lower = np.asarray(result.lower, dtype=float).ravel()
    upper = np.asarray(result.upper, dtype=float).ravel()
    is_anom = np.asarray(result.is_anomaly, dtype=bool).ravel()
    t = y.shape[0]
    if result.time_index is not None:
        time = np.asarray(result.time_index)
    else:
        time = np.arange(t)
    node_label = result.node_names[0] if result.node_names else str(series)

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (10, 4)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mean_color = OKABE_ITO["blue"]
    anom_color = OKABE_ITO["red"]
    band_label = f"{ci_level * 100:.0f}% predictive interval"
    fill_alpha = PARAM_BAND_ALPHA if band_alpha is None else float(band_alpha)

    band_ok = np.isfinite(lower) & np.isfinite(upper)
    mean_plot = np.asarray(mean, dtype=float).copy()
    mean_plot[~np.isfinite(mean_plot) | ~band_ok] = np.nan

    ax.fill_between(
        time,
        lower,
        upper,
        where=band_ok,
        color=mean_color,
        alpha=fill_alpha,
        label=band_label,
        zorder=1,
        interpolate=False,
    )
    ax.plot(
        time,
        mean_plot,
        color=mean_color,
        lw=mean_lw,
        label="Predictive mean",
        zorder=2,
    )
    observed_kwargs: dict[str, Any] = {
        "color": OBSERVED_COLOR,
        "lw": observed_lw,
        "alpha": 0.9,
        "label": "Observed",
        "zorder": 3,
    }
    if show_observed_markers:
        observed_kwargs.update(
            marker="o",
            markersize=3.2,
            markerfacecolor=OBSERVED_COLOR,
            markeredgewidth=0.0,
        )
    ax.plot(time, y, **observed_kwargs)
    if np.any(is_anom):
        ax.scatter(
            time[is_anom],
            y[is_anom],
            marker="x",
            s=anomaly_size,
            color=anom_color,
            linewidths=1.35,
            zorder=4,
            label="Anomaly",
        )

    # Scale y primarily from the observed series so wide early predictive
    # bands do not flatten the panel; expand slightly for finite bands that
    # stay within a few times the observed range.
    y_finite = y[np.isfinite(y)]
    if y_finite.size:
        y_min = float(np.min(y_finite))
        y_max = float(np.max(y_finite))
        y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        if np.any(band_ok):
            band_vals = np.concatenate(
                [lower[band_ok], upper[band_ok], mean[band_ok]]
            )
            band_vals = band_vals[np.isfinite(band_vals)]
            # Ignore extreme band tails relative to the observed span.
            lo_clip = y_min - 1.5 * y_span
            hi_clip = y_max + 1.5 * y_span
            band_vals = band_vals[(band_vals >= lo_clip) & (band_vals <= hi_clip)]
            if band_vals.size:
                y_min = min(y_min, float(np.min(band_vals)))
                y_max = max(y_max, float(np.max(band_vals)))
                y_span = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0)
        pad = 0.08 * y_span
        ax.set_ylim(y_min - pad, y_max + pad)

    if created_fig:
        ax.set_title(str(node_label), fontsize=11)
        style_time_series_ax(
            ax,
            ylabel="Value",
            show_zero_line=False,
            grid_alpha=0.25,
            legend=True,
        )
        plt.tight_layout()
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
    return fig
