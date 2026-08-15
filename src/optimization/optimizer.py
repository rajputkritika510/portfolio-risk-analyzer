"""
optimizer.py
-------------
Optimization Engine: Mean-Variance Optimization using scipy.optimize.

Provides:
- minimum_variance_portfolio
- maximum_sharpe_portfolio
- target_return_portfolio
- efficient_frontier (a set of risk/return points)

Constraints supported: weights sum to 1, min/max weight per asset
(default 0%-100%, but can be tightened, e.g. max 30% per stock,
to avoid unrealistic concentrated solutions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def _portfolio_perf(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(weights.T @ cov_matrix @ weights * TRADING_DAYS)
    return ret, vol


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    ret, vol = _portfolio_perf(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol if vol > 0 else 1e6


def _portfolio_vol(weights, mean_returns, cov_matrix):
    return _portfolio_perf(weights, mean_returns, cov_matrix)[1]


def _bounds_and_x0(n_assets, min_weight, max_weight):
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    x0 = np.array([1 / n_assets] * n_assets)
    return bounds, x0


def maximum_sharpe_portfolio(returns_df: pd.DataFrame, risk_free_rate: float = 0.06,
                              min_weight: float = 0.0, max_weight: float = 1.0) -> dict:
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n, min_weight, max_weight)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(_neg_sharpe, x0, args=(mean_returns.values, cov_matrix.values, risk_free_rate),
                       method="SLSQP", bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=returns_df.columns).round(4)
    ret, vol = _portfolio_perf(result.x, mean_returns.values, cov_matrix.values)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return {"weights": weights, "expected_return": ret, "volatility": vol, "sharpe": sharpe}


def minimum_variance_portfolio(returns_df: pd.DataFrame,
                                min_weight: float = 0.0, max_weight: float = 1.0) -> dict:
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n, min_weight, max_weight)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(_portfolio_vol, x0, args=(mean_returns.values, cov_matrix.values),
                       method="SLSQP", bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=returns_df.columns).round(4)
    ret, vol = _portfolio_perf(result.x, mean_returns.values, cov_matrix.values)
    return {"weights": weights, "expected_return": ret, "volatility": vol}


def target_return_portfolio(returns_df: pd.DataFrame, target_return: float,
                             min_weight: float = 0.0, max_weight: float = 1.0) -> dict:
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n, min_weight, max_weight)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns.values) * TRADING_DAYS - target_return},
    )

    result = minimize(_portfolio_vol, x0, args=(mean_returns.values, cov_matrix.values),
                       method="SLSQP", bounds=bounds, constraints=constraints)

    weights = pd.Series(result.x, index=returns_df.columns).round(4)
    ret, vol = _portfolio_perf(result.x, mean_returns.values, cov_matrix.values)
    return {"weights": weights, "expected_return": ret, "volatility": vol, "success": result.success}


def efficient_frontier(returns_df: pd.DataFrame, n_points: int = 30,
                        min_weight: float = 0.0, max_weight: float = 1.0) -> pd.DataFrame:
    """Return a DataFrame of (volatility, return) points along the efficient frontier."""
    mean_returns = returns_df.mean()
    min_ret = mean_returns.min() * TRADING_DAYS
    max_ret = mean_returns.max() * TRADING_DAYS
    target_returns = np.linspace(min_ret, max_ret, n_points)

    points = []
    for tr in target_returns:
        res = target_return_portfolio(returns_df, tr, min_weight, max_weight)
        if res["success"]:
            points.append({"Return": res["expected_return"], "Volatility": res["volatility"]})

    return pd.DataFrame(points)
