"""
risk.py
--------
Risk Engine: volatility, drawdown, Sharpe/Sortino/Treynor, Beta,
VaR (historical / parametric / Monte Carlo), CVaR, correlation and
concentration (HHI) and per-asset risk contribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS = 252


# ---------------------------------------------------------------------
# Volatility & drawdown
# ---------------------------------------------------------------------
def volatility(returns: pd.Series, annualize: bool = True) -> float:
    vol = returns.std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def drawdown_series(value_series: pd.Series) -> pd.Series:
    """Running drawdown (%) from the running peak."""
    running_max = value_series.cummax()
    return (value_series - running_max) / running_max


def max_drawdown(value_series: pd.Series) -> float:
    return drawdown_series(value_series).min()


# ---------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------
def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """risk_free_rate is an ANNUAL rate, e.g. 0.06 for 6%."""
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1])
    return cov / var if var != 0 else np.nan


def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
          risk_free_rate: float = 0.06) -> float:
    """Annualized Jensen's alpha (CAPM)."""
    b = beta(portfolio_returns, benchmark_returns)
    if np.isnan(b):
        return np.nan
    port_ann = annualized_from_daily(portfolio_returns)
    bench_ann = annualized_from_daily(benchmark_returns)
    return port_ann - (risk_free_rate + b * (bench_ann - risk_free_rate))


def treynor_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                   risk_free_rate: float = 0.06) -> float:
    b = beta(portfolio_returns, benchmark_returns)
    if np.isnan(b) or b == 0:
        return np.nan
    port_ann = annualized_from_daily(portfolio_returns)
    return (port_ann - risk_free_rate) / b


def annualized_from_daily(returns: pd.Series) -> float:
    return (1 + returns.mean()) ** TRADING_DAYS - 1


# ---------------------------------------------------------------------
# Value at Risk / CVaR
# ---------------------------------------------------------------------
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    return np.percentile(returns, (1 - confidence) * 100)


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return mu + z * sigma


def var_monte_carlo(returns: pd.Series, confidence: float = 0.95,
                     n_simulations: int = 10_000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    mu, sigma = returns.mean(), returns.std()
    simulated = rng.normal(mu, sigma, n_simulations)
    return np.percentile(simulated, (1 - confidence) * 100)


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    var = var_historical(returns, confidence)
    tail_losses = returns[returns <= var]
    return tail_losses.mean() if len(tail_losses) > 0 else np.nan


# ---------------------------------------------------------------------
# Diversification / concentration
# ---------------------------------------------------------------------
def herfindahl_index(weights: pd.Series) -> float:
    """HHI = sum(w_i^2). Ranges 0 (fully diversified) to 1 (single asset)."""
    return float((weights ** 2).sum())


def concentration_label(hhi: float) -> str:
    if hhi < 0.15:
        return "Low"
    elif hhi < 0.25:
        return "Moderate"
    else:
        return "High"


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


def covariance_matrix(returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    cov = returns_df.cov()
    return cov * TRADING_DAYS if annualize else cov


def risk_contribution(weights: pd.Series, returns_df: pd.DataFrame) -> pd.Series:
    """
    Percentage contribution of each asset to total portfolio volatility,
    using the standard marginal-contribution-to-risk (MCTR) decomposition.
    """
    cov = covariance_matrix(returns_df)
    w = weights.reindex(cov.columns).fillna(0).values
    port_var = w @ cov.values @ w
    port_vol = np.sqrt(port_var)
    if port_vol == 0:
        return pd.Series(0, index=cov.columns)

    marginal_contrib = cov.values @ w
    contrib = w * marginal_contrib / port_vol
    pct_contrib = contrib / contrib.sum() * 100
    return pd.Series(pct_contrib, index=cov.columns).round(2)
