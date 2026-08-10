"""
MDM predictive-interval anomaly detection.

Flags observations that fall outside the one-step Student-t predictive
interval from the fitted DLM: ``Y_t | y_{t-1} ~ t_{n_{t-1}}[f_t, Q_t]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AnomalyDetectionResult:
    """
    Predictive-interval anomaly flags for one or more MDM nodes.

    Attributes
    ----------
    observed : np.ndarray
        Observed series, shape ``(T, N)`` or ``(T,)`` when a single series
        was requested.
    fitted_mean : np.ndarray
        One-step predictive means ``f_t``, same shape as ``observed``.
    lower, upper : np.ndarray
        Predictive interval bounds at ``ci_level``.
    is_anomaly : np.ndarray
        Boolean mask; ``True`` where the observation lies outside the band.
        Non-comparable times (non-finite ``Q_t`` or predictive ``df <= 2``)
        are ``False``.
    score : np.ndarray
        Absolute standardized forecast error ``|e*_t|``.
    node_names : list of str
        Names of the columns/nodes included in the result.
    ci_level : float
        Credible / predictive interval level used.
    time_index : sequence, optional
        Optional time labels of length ``T`` (e.g. dates).
    """

    observed: np.ndarray
    fitted_mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    is_anomaly: np.ndarray
    score: np.ndarray
    node_names: list[str]
    ci_level: float
    time_index: Optional[Sequence[Any]] = None


def _require_filt(mdm_object: Any) -> dict:
    filt = getattr(mdm_object, "Filt", None)
    if filt is None:
        raise TypeError(
            "detect_anomalies requires a fitted model with Filt "
            "(use MDM(...) or refit_mdm_on_structure)."
        )
    for key in ("ft", "Qt", "nt", "ets"):
        if key not in filt:
            raise TypeError(f"mdm_object.Filt is missing required key '{key}'.")
    return filt


def _require_data(mdm_object: Any) -> np.ndarray:
    data = getattr(mdm_object, "data", None)
    if data is None:
        raise TypeError(
            "detect_anomalies requires model.data "
            "(use MDM(...) or refit_mdm_on_structure)."
        )
    return np.asarray(data, dtype=float)


def _node_names(mdm_object: Any, n: int) -> list[str]:
    names = getattr(mdm_object, "node_names", None)
    if names is None:
        return [f"V{i + 1}" for i in range(n)]
    return [str(x) for x in names]


def _resolve_series_indices(
    series: Union[int, str, None],
    node_names: Sequence[str],
) -> list[int]:
    n = len(node_names)
    if series is None:
        return list(range(n))
    if isinstance(series, int):
        if series < 0 or series >= n:
            raise IndexError(
                f"series index {series} out of range for {n} nodes."
            )
        return [series]
    if isinstance(series, str):
        try:
            return [list(node_names).index(series)]
        except ValueError as exc:
            raise ValueError(
                f"Unknown series name {series!r}; "
                f"expected one of {list(node_names)}."
            ) from exc
    raise TypeError("series must be int, str, or None.")


# Predictive Student-t uses df = n_{t-1} = nt[t] - 1. With the default vague
# prior n0 ≈ 0.001, early steps have df << 1 and t.ppf returns ~1e100, which
# both invalidates anomaly flags and explodes plot y-limits. Require df > 2 so
# the predictive variance exists and critical values stay finite/usable.
_MIN_PREDICTIVE_DF = 2.0


def _predictive_band(
    ft: np.ndarray,
    Qt: np.ndarray,
    nt: np.ndarray,
    ci_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, upper, and valid-mask for the predictive interval."""
    ft = np.asarray(ft, dtype=float).ravel()
    Qt = np.asarray(Qt, dtype=float).ravel()
    nt = np.asarray(nt, dtype=float).ravel()
    t = ft.shape[0]
    lower = np.full(t, np.nan)
    upper = np.full(t, np.nan)
    df = nt - 1.0
    valid = (
        np.isfinite(Qt)
        & (Qt > 0)
        & np.isfinite(df)
        & (df > _MIN_PREDICTIVE_DF)
    )
    if not np.any(valid):
        return lower, upper, valid

    half = np.full(t, np.nan)
    # Student-t critical value uses predictive df = n_{t-1} = nt[t] - 1.
    half[valid] = stats.t.ppf((1.0 + ci_level) / 2.0, df[valid]) * np.sqrt(
        Qt[valid]
    )
    lower[valid] = ft[valid] - half[valid]
    upper[valid] = ft[valid] + half[valid]
    return lower, upper, valid


def detect_anomalies(
    mdm_object: Any,
    *,
    ci_level: float = 0.95,
    series: Union[int, str, None] = None,
    output: Literal["result", "dataframe"] = "result",
    time_index: Optional[Sequence[Any]] = None,
) -> Union[AnomalyDetectionResult, pd.DataFrame]:
    """
    Flag observations outside the MDM one-step predictive interval.

    For each node and time ``t``, the predictive distribution is the Student-t
    used by the DLM filter, ``Y_t | y_{t-1} ~ t_{n_{t-1}}[f_t, Q_t]``, with
    degrees of freedom ``nt[t] - 1`` on the returned (t=0-stripped) arrays.
    An observation is an anomaly when it lies outside
    ``f_t ± t_{1-α/2, df} √Q_t``. The anomaly score is ``|ets[t]|``.

    Times with non-finite ``Q_t`` or predictive ``df <= 2`` (typical of the
    vague DLM prior in the first few steps) are treated as non-comparable:
    bounds are ``NaN`` and ``is_anomaly`` is ``False``.

    Parameters
    ----------
    mdm_object
        Fitted :class:`~mdmp.model.MDM` or :class:`~mdmp.model.MDMResults`
        (or compatible object) with ``data`` and ``Filt``.
    ci_level : float, optional
        Predictive interval level in ``(0, 1)``. Default is ``0.95``.
    series : int or str or None, optional
        Node index or name. ``None`` (default) selects all nodes.
    output : {"result", "dataframe"}, optional
        Return an :class:`AnomalyDetectionResult` or a long-form DataFrame.
    time_index : sequence, optional
        Optional length-``T`` labels stored on the result / DataFrame.

    Returns
    -------
    AnomalyDetectionResult or pandas.DataFrame
        Arrays / columns: ``observed``, ``fitted_mean``, ``lower``, ``upper``,
        ``is_anomaly``, ``score``.

    Examples
    --------
    >>> from mdmp import MDM, detect_anomalies
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(40, 2))
    >>> model = MDM(y, method="hc", nbf=5, verbose=False)  # doctest: +SKIP
    >>> res = detect_anomalies(model, ci_level=0.95)  # doctest: +SKIP
    >>> bool(res.is_anomaly.any())  # doctest: +SKIP
    """
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must be in (0, 1).")

    filt = _require_filt(mdm_object)
    data = _require_data(mdm_object)
    if data.ndim != 2:
        raise ValueError("model.data must be a 2-D array of shape (T, N).")
    t, n = data.shape
    names = _node_names(mdm_object, n)
    idxs = _resolve_series_indices(series, names)
    selected_names = [names[i] for i in idxs]

    observed_cols = []
    mean_cols = []
    lower_cols = []
    upper_cols = []
    anom_cols = []
    score_cols = []

    for j in idxs:
        ft = np.asarray(filt["ft"][j], dtype=float).ravel()
        Qt = np.asarray(filt["Qt"][j], dtype=float).ravel()
        nt = np.asarray(filt["nt"][j], dtype=float).ravel()
        ets = np.asarray(filt["ets"][j], dtype=float).ravel()
        if ft.shape[0] != t or Qt.shape[0] != t or nt.shape[0] != t:
            raise ValueError(
                f"Filt arrays for node {j} have length incompatible with data "
                f"(expected T={t})."
            )
        y = data[:, j]
        lower, upper, valid = _predictive_band(ft, Qt, nt, float(ci_level))
        is_anom = np.zeros(t, dtype=bool)
        comparable = valid & np.isfinite(y)
        is_anom[comparable] = (y[comparable] < lower[comparable]) | (
            y[comparable] > upper[comparable]
        )
        score = np.abs(ets)
        if score.shape[0] != t:
            score = np.full(t, np.nan)
            score[: min(t, ets.shape[0])] = np.abs(ets[:t])

        observed_cols.append(y)
        mean_cols.append(ft)
        lower_cols.append(lower)
        upper_cols.append(upper)
        anom_cols.append(is_anom)
        score_cols.append(score)

    squeeze = series is not None
    observed = np.column_stack(observed_cols)
    fitted_mean = np.column_stack(mean_cols)
    lower = np.column_stack(lower_cols)
    upper = np.column_stack(upper_cols)
    is_anomaly = np.column_stack(anom_cols)
    score = np.column_stack(score_cols)
    if squeeze:
        observed = observed[:, 0]
        fitted_mean = fitted_mean[:, 0]
        lower = lower[:, 0]
        upper = upper[:, 0]
        is_anomaly = is_anomaly[:, 0]
        score = score[:, 0]

    if time_index is not None and len(time_index) != t:
        raise ValueError(
            f"time_index length {len(time_index)} does not match T={t}."
        )

    result = AnomalyDetectionResult(
        observed=observed,
        fitted_mean=fitted_mean,
        lower=lower,
        upper=upper,
        is_anomaly=is_anomaly,
        score=score,
        node_names=selected_names,
        ci_level=float(ci_level),
        time_index=time_index,
    )

    if output == "result":
        return result
    if output != "dataframe":
        raise ValueError("output must be 'result' or 'dataframe'.")

    rows = []
    for col, name in enumerate(selected_names):
        y = observed if squeeze else observed[:, col]
        fm = fitted_mean if squeeze else fitted_mean[:, col]
        lo = lower if squeeze else lower[:, col]
        hi = upper if squeeze else upper[:, col]
        flag = is_anomaly if squeeze else is_anomaly[:, col]
        sc = score if squeeze else score[:, col]
        for i in range(t):
            row = {
                "time": time_index[i] if time_index is not None else i,
                "node": name,
                "observed": float(y[i]),
                "fitted_mean": float(fm[i]),
                "lower": float(lo[i]) if np.isfinite(lo[i]) else np.nan,
                "upper": float(hi[i]) if np.isfinite(hi[i]) else np.nan,
                "score": float(sc[i]) if np.isfinite(sc[i]) else np.nan,
                "is_anomaly": bool(flag[i]),
                "ci_level": float(ci_level),
            }
            rows.append(row)
    return pd.DataFrame(rows)
