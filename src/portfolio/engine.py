"""
engine.py
----------
Portfolio Engine: turns raw holdings (ticker, quantity, buy price)
plus a price history matrix into portfolio value, weights and P&L.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Holding:
    ticker: str
    quantity: float
    buy_price: float


class Portfolio:
    """
    Represents a buy-and-hold portfolio built from a list of Holdings
    and a matrix of historical adjusted-close prices.
    """

    def __init__(self, holdings: list[Holding], prices: pd.DataFrame):
        self.holdings = holdings
        self.tickers = [h.ticker for h in holdings]
        self.prices = prices[self.tickers].copy()
        self.quantities = pd.Series({h.ticker: h.quantity for h in holdings})
        self.buy_prices = pd.Series({h.ticker: h.buy_price for h in holdings})
        self.invested_amount = float((self.quantities * self.buy_prices).sum())

    # ---------------------------------------------------------------
    # Valuation
    # ---------------------------------------------------------------
    def position_values(self) -> pd.DataFrame:
        """Daily market value of every holding (Date x Ticker)."""
        return self.prices.mul(self.quantities, axis=1)

    def portfolio_value(self) -> pd.Series:
        """Total daily portfolio value (sum across holdings)."""
        return self.position_values().sum(axis=1)

    def weights_over_time(self) -> pd.DataFrame:
        """Daily weight of each holding (naturally drifting, buy & hold)."""
        values = self.position_values()
        total = values.sum(axis=1)
        return values.div(total, axis=0)

    def current_weights(self) -> pd.Series:
        return self.weights_over_time().iloc[-1]

    def latest_snapshot(self) -> pd.DataFrame:
        """
        Holdings table for the 'Holdings' tab:
        Ticker | Qty | Buy Price | Current Price | Value | Weight | P&L | Return %
        """
        current_price = self.prices.iloc[-1]
        value = current_price * self.quantities
        cost = self.buy_prices * self.quantities
        pnl = value - cost
        ret_pct = (pnl / cost) * 100
        weight = value / value.sum() * 100

        df = pd.DataFrame({
            "Ticker": self.tickers,
            "Quantity": self.quantities.values,
            "Buy Price": self.buy_prices.values,
            "Current Price": current_price.values,
            "Current Value": value.values,
            "Weight (%)": weight.values,
            "P&L (₹)": pnl.values,
            "Return (%)": ret_pct.values,
        })
        return df.round(2)

    # ---------------------------------------------------------------
    # P&L summary
    # ---------------------------------------------------------------
    def total_pnl(self) -> float:
        current_value = float(self.portfolio_value().iloc[-1])
        return current_value - self.invested_amount

    def total_return_pct(self) -> float:
        current_value = float(self.portfolio_value().iloc[-1])
        return ((current_value - self.invested_amount) / self.invested_amount) * 100
