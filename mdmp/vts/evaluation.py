"""
Evaluation and comparison tools for Virtual Typical Subject methods.
"""

from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd

from .data import prepare_multi_subject_data
from .strategies import ConcatenationStrategy, MeanBasedStrategy
from .types import ComparisonResult, VTSResult


def compare_vts_methods(
    data: Union[List[np.ndarray], np.ndarray, pd.DataFrame],
    methods: List[str] = None,
    metrics: List[str] = None,
) -> ComparisonResult:
    """
    Compare multiple VTS methods on the same data.

    Parameters
    ----------
    data : list of np.ndarray, np.ndarray, or pd.DataFrame
        Multi-subject data.
    methods : list of str, optional
        Methods to compare: "concatenation", "mean". Default both.
    metrics : list of str, optional
        Metrics for comparison table: "mse", "correlation". Default ["mse"].

    Returns
    -------
    ComparisonResult
        Results for each method and optional comparison table.
    """
    if methods is None:
        methods = ["concatenation", "mean"]
    if metrics is None:
        metrics = ["mse"]

    arrays, metadata = prepare_multi_subject_data(data)
    results: Dict[str, VTSResult] = {}

    if "concatenation" in methods:
        strat = ConcatenationStrategy(return_series=True)
        results["concatenation"] = strat.compute(arrays, metadata)

    if "mean" in methods:
        strat = MeanBasedStrategy(align_method="truncate")
        results["mean"] = strat.compute(arrays, metadata)

    comparison_table = None
    if metrics:
        rows = []
        for name, vts_res in results.items():
            row = {"method": name}
            for m in metrics:
                val = evaluate_vts_representation(
                    arrays, vts_res, metric=m, metadata=metadata
                )
                row[m] = val
            rows.append(row)
        comparison_table = pd.DataFrame(rows)

    return ComparisonResult(
        results=results,
        comparison_table=comparison_table,
    )


def evaluate_vts_representation(
    data: List[np.ndarray],
    vts_result: VTSResult,
    metric: Literal["mse", "correlation"] = "mse",
    metadata: dict = None,
) -> float:
    """
    Evaluate how well the VTS represents the population.

    Compares subject-level means (over time) to VTS mean (over time).
    Works for both concatenation and mean-based VTS.

    Parameters
    ----------
    data : list of np.ndarray
        Original subject data (T_s x N).
    vts_result : VTSResult
        Computed VTS.
    metric : {"mse", "correlation"}, optional
        Evaluation metric. Default "mse".
    metadata : dict, optional
        Metadata from prepare_multi_subject_data. Reserved for future use.

    Returns
    -------
    float
        Aggregate metric value.
    """
    vts = vts_result.vts_data
    if vts.ndim == 1:
        vts_summary = vts
    else:
        vts_summary = np.mean(vts, axis=0)

    subject_summaries = [np.mean(arr, axis=0) for arr in data]

    if metric == "mse":
        mses = [
            float(np.mean((s - vts_summary) ** 2))
            for s in subject_summaries
        ]
        return float(np.mean(mses))

    if metric == "correlation":
        cors = []
        for s in subject_summaries:
            if np.std(s) > 0 and np.std(vts_summary) > 0:
                cors.append(np.corrcoef(s, vts_summary)[0, 1])
            else:
                cors.append(0.0)
        return float(np.mean(cors))

    raise ValueError(
        f"metric must be 'mse' or 'correlation', got {metric!r}"
    )


def subject_vs_vts_metrics(
    data: List[np.ndarray],
    vts_result: VTSResult,
    metric: Literal["mse", "correlation"] = "mse",
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Per-subject distance/metrics to VTS.

    For VTS with matching length (mean-based), compares per-time-point.
    For VTS with different length (concatenation), compares subject mean
    to VTS mean.

    Parameters
    ----------
    data : list of np.ndarray
        Original subject data.
    vts_result : VTSResult
        Computed VTS.
    metric : {"mse", "correlation"}, optional
        Per-subject metric. Default "mse".

    Returns
    -------
    dict
        - "per_subject": array of metric value per subject
        - "mean": mean across subjects
        - "std": std across subjects
    """
    from .data import align_subjects

    vts = vts_result.vts_data
    aligned = align_subjects(data, method="truncate")
    min_len = min(arr.shape[0] for arr in aligned)
    vts_can_match = vts.ndim == 2 and vts.shape[0] == min_len

    if vts.ndim == 1:
        vts_summary = vts
    else:
        vts_summary = np.mean(vts, axis=0)

    per_subject = []
    for arr in aligned:
        if vts_can_match:
            a = arr[:min_len]
            b = vts[:min_len]
        else:
            a = np.mean(arr, axis=0)
            b = vts_summary

        if metric == "mse":
            per_subject.append(float(np.mean((a - b) ** 2)))
        elif metric == "correlation":
            a_flat = np.asarray(a).flatten()
            b_flat = np.asarray(b).flatten()
            if np.std(a_flat) > 0 and np.std(b_flat) > 0:
                per_subject.append(float(np.corrcoef(a_flat, b_flat)[0, 1]))
            else:
                per_subject.append(0.0)
        else:
            raise ValueError(
                f"metric must be 'mse' or 'correlation', got {metric!r}"
            )

    arr = np.array(per_subject)
    return {
        "per_subject": arr,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
