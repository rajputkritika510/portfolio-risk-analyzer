import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analytics import returns as ret_engine
from src.analytics import risk as risk_engine


def test_cagr_simple_doubling():
    dates = pd.date_range("2023-01-01", periods=2, freq="365D")
    series = pd.Series([100, 200], index=dates)
    result = ret_engine.cagr(series)
    assert abs(result - 1.0) < 0.01  # ~100% annual growth


def test_daily_returns_basic():
    series = pd.Series([100, 110, 121])
    r = ret_engine.daily_returns(series)
    assert abs(r.iloc[0] - 0.10) < 1e-9
    assert abs(r.iloc[1] - 0.10) < 1e-9


def test_max_drawdown_known_case():
    series = pd.Series([100, 120, 90, 100])
    mdd = risk_engine.max_drawdown(series)
    # Peak 120 -> trough 90 = -25%
    assert abs(mdd - (-0.25)) < 1e-9


def test_hhi_equal_weights():
    weights = pd.Series([0.25, 0.25, 0.25, 0.25])
    hhi = risk_engine.herfindahl_index(weights)
    assert abs(hhi - 0.25) < 1e-9


def test_hhi_concentrated():
    weights = pd.Series([1.0])
    hhi = risk_engine.herfindahl_index(weights)
    assert abs(hhi - 1.0) < 1e-9


def test_volatility_zero_for_constant_returns():
    returns = pd.Series([0.0] * 10)
    vol = risk_engine.volatility(returns)
    assert vol == 0
