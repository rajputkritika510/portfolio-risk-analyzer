"""
returns.py
-----------
Performance Engine: return calculations.

Includes simple daily returns, cumulative returns, CAGR and a basic
XIRR (money-weighted return) implementation for portfolios that had
a single lump-sum investment (buy-and-hold, V1 scope). Multi-cash-flow
XIRR is listed in the roadmap as a V5 feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def daily_returns(price_or_value_series: pd.Series) -> pd.Series:
    """Simple daily percentage returns."""
    return price_or_value_series.pct_change().dropna()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative growth of ₹1 invested, based on a return series."""
    return (1 + returns).cumprod() - 1


def cagr(value_series: pd.Series) -> float:
    """
    Compound Annual Growth Rate between the first and last value
    of a value/price series.
    """
    if len(value_series) < 2:
        return np.nan
    start_val = value_series.iloc[0]
    end_val = value_series.iloc[-1]
    n_years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    if n_years <= 0 or start_val <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / n_years) - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize a mean daily return."""
    mean_daily = returns.mean()
    return (1 + mean_daily) ** periods_per_year - 1


def xirr(cash_flows: list[tuple[pd.Timestamp, float]]) -> float:
    """
    Money-weighted return (XIRR) for a list of (date, amount) cash flows.
    Convention: investments are negative, final value / withdrawals positive.

    Example:
        xirr([(2024-01-01, -50000), (2025-01-01, 60000)])
    """
    if len(cash_flows) < 2:
        return np.nan

    dates = [cf[0] for cf in cash_flows]
    amounts = np.array([cf[1] for cf in cash_flows], dtype=float)
    t0 = dates[0]
    years = np.array([(d - t0).days / 365.25 for d in dates])

    def npv(rate):
        return np.sum(amounts / (1 + rate) ** years)

    try:
        return brentq(npv, -0.9999, 10)
    except ValueError:
        return np.nan
