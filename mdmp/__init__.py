"""
MDMP: Bayesian Dynamic Regression Model (MDM) for Python

A Python package for learning Bayesian network structures from multivariate time series
and estimating time-varying dynamic parameters using Kalman filtering and smoothing.

This package is a Python port of the R package 'mdmr'.

Public entry points (import from the submodule that matches the job):

- ``mdmp.model`` — ``MDM``, ``refit_mdm_on_structure``, ``MDMResults``
- ``mdmp.retail`` — supermarket panel helpers (``load_retail``, labels, aggregation)
- ``mdmp.datasets`` — generic bundled loaders (``list_datasets``, ``load_dataset``)
- ``mdmp.plotting`` — DAG, arcs, stream, projection, and related plots
- ``mdmp.anomaly`` — ``detect_anomalies``, ``AnomalyDetectionResult``
- ``mdmp.group_analysis`` — VTS, IS, GS (``compute_vts``, ``compute_is``, ``compute_gs``, ...)
- ``mdmp.structure`` / ``mdmp.dlm`` / ``mdmp.scoring`` — advanced / R-compat APIs
"""

from ._version import __version__

__all__ = [
    "__version__",
]
