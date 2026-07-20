"""
Plotting functions for MDM visualization.

This module provides visualization tools for MDM models including DAG structure,
dynamic parameters, marginal posteriors, stream plots, and animated heatmaps.
"""

from .animation import plot_idag
from .dag import plot_dag
from .parameters import plot_arcs, plot_marginal, plot_stream
from .projection import (
    plot_dendrogram,
    plot_group_embedding,
    plot_projection,
    project_distance,
)

__all__ = [
    "plot_dag",
    "plot_arcs",
    "plot_marginal",
    "plot_stream",
    "plot_idag",
    "plot_projection",
    "plot_dendrogram",
    "plot_group_embedding",
    "project_distance",
]
