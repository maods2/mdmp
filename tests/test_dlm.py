"""
Tests for DLM filtering and smoothing functions.
"""

import numpy as np

from mdmp.dlm import dlm_filter, dlm_smooth


def test_dlm_filter_basic():
    """Test basic DLM filtering."""
    T = 50
    p = 2  # 2 parameters

    # Create simple time series
    np.random.seed(42)
    Yt = np.random.randn(T)

    # Create design matrix (intercept + one covariate)
    Ft = np.ones((p, T))
    Ft[1, :] = np.linspace(0, 1, T)

    # Run filter
    result = dlm_filter(Yt, Ft, delta=0.9)

    # Check output structure
    assert 'mt' in result
    assert 'Ct' in result
    assert 'Rt' in result
    assert 'nt' in result
    assert 'dt' in result
    assert 'ft' in result
    assert 'Qt' in result
    assert 'ets' in result
    assert 'lpl' in result

    # Check dimensions
    assert result['mt'].shape == (p, T)
    assert result['Ct'].shape == (p, p, T)
    assert result['Rt'].shape == (p, p, T)
    assert result['nt'].shape == (T,)
    assert result['dt'].shape == (T,)
    assert result['ft'].shape == (T,)
    assert result['Qt'].shape == (T,)
    assert result['ets'].shape == (T,)
    assert result['lpl'].shape == (T,)

    # Check that nt and dt increase
    assert np.all(result['nt'] > 0)
    assert np.all(result['dt'] > 0)
    assert np.allclose(np.diff(result['nt']), 1.0)  # nt increases by 1 each step


def test_dlm_filter_single_parameter():
    """Test DLM filter with single parameter (intercept only)."""
    T = 20
    Yt = np.random.randn(T)
    Ft = np.ones((1, T))  # Only intercept

    result = dlm_filter(Yt, Ft, delta=0.95)

    assert result['mt'].shape == (1, T)
    assert result['Ct'].shape == (1, 1, T)


def test_dlm_filter_custom_prior():
    """Test DLM filter with custom prior parameters."""
    T = 30
    Yt = np.random.randn(T)
    Ft = np.ones((1, T))

    m0 = np.array([1.0])
    CS0 = 5.0 * np.eye(1)
    n0 = 1.0
    d0 = 1.0

    result = dlm_filter(Yt, Ft, delta=0.9, m0=m0, CS0=CS0, n0=n0, d0=d0)

    # Check that initial values are set correctly
    # Note: nt[0] will be n0 + 1 after first update, dt[0] will be d0 + (error^2/QSt)
    assert result['nt'][0] > n0
    assert result['dt'][0] >= d0


def test_dlm_filter_invalid_delta():
    """Test DLM filter with invalid discount factor."""
    T = 10
    Yt = np.random.randn(T)
    Ft = np.ones((1, T))

    # Delta < 0 will cause numerical issues but may not raise
    # Just check it doesn't crash (may produce warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dlm_filter(Yt, Ft, delta=-0.1)
        # Should still return a result (may contain NaN/Inf)
        assert 'mt' in result


def test_dlm_smooth_basic():
    """Test basic DLM smoothing."""
    T = 30
    p = 2

    # Create filtered results
    np.random.seed(42)
    mt = np.random.randn(p, T)
    Ct = np.random.randn(p, p, T)
    # Make Ct positive definite
    for t in range(T):
        Ct[:, :, t] = Ct[:, :, t] @ Ct[:, :, t].T + 0.1 * np.eye(p)

    Rt = Ct.copy()
    nt = np.linspace(10, 20, T)
    dt = np.linspace(5, 15, T)

    result = dlm_smooth(mt, Ct, Rt, nt, dt)

    assert 'smt' in result
    assert 'sCt' in result
    assert result['smt'].shape == (p, T)
    assert result['sCt'].shape == (p, p, T)

    # Smoothed values at last time point should equal filtered
    assert np.allclose(result['smt'][:, -1], mt[:, -1])


def test_dlm_smooth_single_parameter():
    """Test DLM smooth with single parameter."""
    T = 20
    mt = np.random.randn(T)
    Ct = np.abs(np.random.randn(T)) + 0.1
    Rt = Ct.copy()
    nt = np.ones(T) * 10
    dt = np.ones(T) * 5

    result = dlm_smooth(mt, Ct, Rt, nt, dt)

    assert result['smt'].shape == (1, T)
    assert result['sCt'].shape == (1, 1, T)


def test_dlm_smooth_consistency():
    """Test that smoothing is consistent with filtering."""
    T = 25
    p = 1

    Yt = np.random.randn(T)
    Ft = np.ones((p, T))

    # Run filter
    filt_result = dlm_filter(Yt, Ft, delta=0.9)

    # Run smooth on filtered results
    smooth_result = dlm_smooth(
        filt_result['mt'],
        filt_result['Ct'],
        filt_result['Rt'],
        filt_result['nt'],
        filt_result['dt']
    )

    # Last time point should match
    assert np.allclose(
        smooth_result['smt'][:, -1],
        filt_result['mt'][:, -1]
    )
