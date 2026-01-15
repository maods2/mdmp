"""
Dynamic Linear Models (DLM) - Core filtering and smoothing functions.

This module implements the core DLM filtering and smoothing algorithms
used in the Multiregression Dynamic Model (MDM).
"""

import numpy as np
from typing import Tuple, Optional
from scipy import special
from .utils import (
    DEFAULT_N0,
    DEFAULT_D0,
    DEFAULT_CS0_SCALE
)


def dlm_filter(
    Yt: np.ndarray,
    Ft: np.ndarray,
    delta: float,
    Gt: Optional[np.ndarray] = None,
    m0: Optional[np.ndarray] = None,
    CS0: Optional[np.ndarray] = None,
    n0: float = DEFAULT_N0,
    d0: float = DEFAULT_D0
) -> dict:
    """
    Dynamic Linear Model (DLM) filtering for unknown observational and state variances.

    Parameters
    ----------
    Yt : np.ndarray
        Vector of observed time series with length T.
    Ft : np.ndarray
        Matrix of covariates with dimension (p, T) where p is the number of parameters.
    delta : float
        Discount factor. Wt = Ct * (1 - delta) / delta.
    Gt : np.ndarray, optional
        Matrix of state equation with dimension (p, p, T). Default is identity matrix.
    m0 : np.ndarray, optional
        Vector of prior mean at time t=0 with length p. Default is zero vector.
    CS0 : np.ndarray, optional
        Prior variance matrix C*0 with dimension (p, p). Default is 3 * I.
    n0 : float, optional
        Prior hyperparameter of precision phi ~ G(n0/2, d0/2). Default is 0.001.
    d0 : float, optional
        Prior hyperparameter of precision phi ~ G(n0/2, d0/2). Default is 0.001.

    Returns
    -------
    dict
        Dictionary containing:
        - mt : Posterior means (p, T)
        - Ct : Posterior variances (p, p, T)
        - Rt : Prior variances (p, p, T)
        - nt : Hyperparameters of precision (T,)
        - dt : Hyperparameters of precision (T,)
        - ft : One-step forecasts (T,)
        - Qt : Forecast variances (T,)
        - ets : Standardized errors (T,)
        - lpl : Log predictive likelihood (T,)
    """
    p = Ft.shape[0]  # Number of parameters
    Nt = len(Yt) + 1  # Sample size + t=0

    # Validate and set default n0
    if n0 == 0:
        n0 = DEFAULT_N0
        import warnings
        warnings.warn(f"n0 is set to {DEFAULT_N0}")

    # Initialize time series arrays (with t=0 padding)
    Y, F, G = _initialize_dlm_arrays(Yt, Ft, Gt, p, Nt)

    # Initialize prior parameters
    m0_init = m0 if m0 is not None else np.zeros(p)
    CS0_init = CS0 if CS0 is not None else DEFAULT_CS0_SCALE * np.eye(p)
    
    mt, Ct = _initialize_posterior_parameters(m0_init, CS0_init, p, Nt, n0, d0)

    # Initialize filtering arrays
    nt, dt, ft, Qt, ets, lpl = _initialize_filtering_arrays(Nt, n0, d0)
    Rt = np.zeros((p, p, Nt))

    # Main filtering loop
    for i in range(1, Nt):
        # Update filtering step
        _update_filtering_step(
            i, G, mt, Ct, Rt, nt, dt, F, Y, ft, Qt, ets, lpl, delta
        )

    return {
        'mt': mt[:, 1:],
        'Ct': Ct[:, :, 1:],
        'Rt': Rt[:, :, 1:],
        'nt': nt[1:],
        'dt': dt[1:],
        'ft': ft[1:],
        'Qt': Qt[1:],
        'ets': ets[1:],
        'lpl': lpl[1:]
    }


def dlm_smooth(
    mt: np.ndarray,
    Ct: np.ndarray,
    Rt: np.ndarray,
    nt: np.ndarray,
    dt: np.ndarray,
    Gt: Optional[np.ndarray] = None
) -> dict:
    """
    Dynamic Linear Model (DLM) smoothing for unknown observational and state variances.

    Parameters
    ----------
    mt : np.ndarray
        Matrix of posterior means with dimension (p, T).
    Ct : np.ndarray
        Matrix of posterior variances with dimension (p, p, T).
    Rt : np.ndarray
        Matrix of prior variances with dimension (p, p, T).
    nt : np.ndarray
        Vector of hyperparameters of precision with length T.
    dt : np.ndarray
        Vector of hyperparameters of precision with length T.
    Gt : np.ndarray, optional
        Matrix of state equation with dimension (p, p, T). Default is identity matrix.

    Returns
    -------
    dict
        Dictionary containing:
        - smt : Smoothed means (p, T)
        - sCt : Smoothed variances (p, p, T)
    """
    # Handle vector case
    if mt.ndim == 1:
        mt = mt.reshape(1, -1)
        Ct = Ct.reshape(1, 1, -1)
        Rt = Rt.reshape(1, 1, -1)

    if Gt is None:
        p = mt.shape[0]
        T = mt.shape[1]
        # Create identity matrices for each time point: shape (T, p, p)
        Gt_list = [np.eye(p) for _ in range(T)]
        # Convert to array and transpose to (p, p, T)
        Gt = np.array(Gt_list).transpose(1, 2, 0)

    p = mt.shape[0]
    Nt = mt.shape[1]

    smt = np.zeros((p, Nt))
    sCt = np.zeros((p, p, Nt))

    # At the last time point
    smt[:, Nt - 1] = mt[:, Nt - 1]
    sCt[:, :, Nt - 1] = Ct[:, :, Nt - 1]

    # Backward smoothing
    for i in range(Nt - 2, -1, -1):
        RSt = Rt[:, :, i + 1] * nt[i] / dt[i]
        CSt = Ct[:, :, i] * nt[i] / dt[i]
        
        # Inverse of RSt
        try:
            inv_sR = np.linalg.inv(RSt)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            inv_sR = np.linalg.pinv(RSt)
        
        B = CSt @ Gt[:, :, i + 1].T @ inv_sR
        smt[:, i] = mt[:, i] + B @ (smt[:, i + 1] - Gt[:, :, i + 1] @ mt[:, i])
        sCS = CSt + B @ (sCt[:, :, i + 1] * nt[Nt - 1] / dt[Nt - 1] - RSt) @ B.T
        sCt[:, :, i] = sCS * dt[Nt - 1] / nt[Nt - 1]

    return {
        'smt': smt,
        'sCt': sCt
    }


def _initialize_dlm_arrays(
    Yt: np.ndarray,
    Ft: np.ndarray,
    Gt: Optional[np.ndarray],
    p: int,
    Nt: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize DLM arrays with t=0 padding.
    
    Parameters
    ----------
    Yt : np.ndarray
        Observed time series.
    Ft : np.ndarray
        Design matrix (p, T).
    Gt : np.ndarray, optional
        State transition matrix.
    p : int
        Number of parameters.
    Nt : int
        Total number of time points including t=0.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Initialized Y, F, G arrays with t=0 padding.
    """
    # Initialize Y with t=0 padding
    Y = np.zeros(Nt)
    Y[1:] = Yt
    
    # Initialize F with t=0 padding
    F = np.zeros((p, Nt))
    F[:, 1:] = Ft
    
    # Initialize G with identity matrices if not provided
    if Gt is None:
        Gt_list = [np.eye(p) for _ in range(len(Yt))]
        Gt = np.array(Gt_list).transpose(1, 2, 0)
    
    G = np.zeros((p, p, Nt))
    G[:, :, 1:] = Gt
    
    return Y, F, G


def _initialize_posterior_parameters(
    m0: np.ndarray,
    CS0: np.ndarray,
    p: int,
    Nt: int,
    n0: float,
    d0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize posterior mean and variance parameters.
    
    Parameters
    ----------
    m0 : np.ndarray
        Prior mean at t=0.
    CS0 : np.ndarray
        Prior variance matrix C*0.
    p : int
        Number of parameters.
    Nt : int
        Total number of time points.
    n0 : float
        Prior hyperparameter n0.
    d0 : float
        Prior hyperparameter d0.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Initialized mt and Ct arrays.
    """
    mt = np.zeros((p, Nt))
    mt[:, 0] = m0
    
    Ct = np.zeros((p, p, Nt))
    Ct[:, :, 0] = CS0 * d0 / n0
    
    return mt, Ct


def _initialize_filtering_arrays(
    Nt: int,
    n0: float,
    d0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize filtering arrays.
    
    Parameters
    ----------
    Nt : int
        Total number of time points.
    n0 : float
        Prior hyperparameter n0.
    d0 : float
        Prior hyperparameter d0.
    
    Returns
    -------
    Tuple of arrays
        Initialized nt, dt, ft, Qt, ets, lpl arrays.
    """
    nt = np.zeros(Nt)
    nt[0] = n0
    dt = np.zeros(Nt)
    dt[0] = d0
    ft = np.zeros(Nt)
    Qt = np.zeros(Nt)
    ets = np.zeros(Nt)
    lpl = np.zeros(Nt)
    
    return nt, dt, ft, Qt, ets, lpl


def _update_filtering_step(
    i: int,
    G: np.ndarray,
    mt: np.ndarray,
    Ct: np.ndarray,
    Rt: np.ndarray,
    nt: np.ndarray,
    dt: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    ft: np.ndarray,
    Qt: np.ndarray,
    ets: np.ndarray,
    lpl: np.ndarray,
    delta: float
) -> None:
    """
    Update one step of the DLM filtering algorithm.
    
    Parameters
    ----------
    i : int
        Current time index.
    G : np.ndarray
        State transition matrices (p, p, Nt).
    mt : np.ndarray
        Posterior means (p, Nt), modified in place.
    Ct : np.ndarray
        Posterior variances (p, p, Nt), modified in place.
    Rt : np.ndarray
        Prior variances (p, p, Nt), modified in place.
    nt : np.ndarray
        Precision hyperparameters (Nt,), modified in place.
    dt : np.ndarray
        Precision hyperparameters (Nt,), modified in place.
    F : np.ndarray
        Design matrices (p, Nt).
    Y : np.ndarray
        Observed time series (Nt,).
    ft : np.ndarray
        Forecasts (Nt,), modified in place.
    Qt : np.ndarray
        Forecast variances (Nt,), modified in place.
    ets : np.ndarray
        Standardized errors (Nt,), modified in place.
    lpl : np.ndarray
        Log predictive likelihood (Nt,), modified in place.
    delta : float
        Discount factor.
    """
    from scipy import special
    p = mt.shape[0]
    
    # Prior at t: (theta_t | y_{t-1}) ~ t_{n_{t-1}}[a_t, R_t]
    at = G[:, :, i] @ mt[:, i - 1]
    RSt = (G[:, :, i] @ (Ct[:, :, i - 1] * nt[i - 1] / dt[i - 1]) @ G[:, :, i].T) / delta
    Rt[:, :, i] = RSt * dt[i - 1] / nt[i - 1]

    # One-step forecast: (Y_t | y_{t-1}) ~ t_{n_{t-1}}[f_t, Q_t]
    ft[i] = F[:, i].T @ at
    QSt = F[:, i].T @ RSt @ F[:, i] + 1
    Qt[i] = QSt * dt[i - 1] / nt[i - 1]
    et = Y[i] - ft[i]
    ets[i] = et / np.sqrt(Qt[i])

    # Posterior at t: (theta_t | y_t) ~ t_{n_t}[m_t, C_t]
    At = Rt[:, :, i] @ F[:, i] / Qt[i]
    mt[:, i] = at + At * et
    nt[i] = nt[i - 1] + 1
    dt[i] = dt[i - 1] + (et ** 2) / QSt
    CSt = RSt - np.outer(At, At) * QSt
    Ct[:, :, i] = CSt * dt[i] / nt[i]

    # Log Predictive Likelihood
    lpl[i] = (
        special.gammaln((nt[i - 1] + 1) / 2) -
        special.gammaln(nt[i - 1] / 2) -
        0.5 * np.log(np.pi * nt[i - 1] * Qt[i]) -
        ((nt[i - 1] + 1) / 2) * np.log(1 + (1 / nt[i - 1]) * et ** 2 / Qt[i])
    )

