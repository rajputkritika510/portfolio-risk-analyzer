"""
market_data.py
----------------
Data Engine: responsible for fetching, validating and cleaning
historical price data for portfolio holdings and the benchmark.

All prices returned are ADJUSTED CLOSE prices (splits + dividends
already accounted for), which is the correct convention for
total-return analysis.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(show_spinner=False, ttl=60 * 60)  # cache for 1 hour
def fetch_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]   e.g. ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    start   : str          "YYYY-MM-DD"
    end     : str          "YYYY-MM-DD"

    Returns
    -------
    pd.DataFrame  (index = Date, columns = tickers) of adjusted close prices.
    Invalid/unreachable tickers are silently dropped (see validate_tickers
    for surfacing that information to the user).
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives split/dividend-adjusted OHLC
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    # Newer yfinance versions can return MultiIndex columns even for a
    # single ticker, so we detect the shape instead of trusting len(tickers).
    if isinstance(raw.columns, pd.MultiIndex):
        top_level = raw.columns.get_level_values(0)
        if "Close" in top_level and set(top_level) <= {"Open", "High", "Low", "Close", "Volume", "Adj Close"}:
            # shape: (field, ticker) -- single-ticker MultiIndex sometimes comes this way
            prices = raw["Close"].copy()
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
        else:
            # shape: (ticker, field)
            available = [t for t in tickers if t in top_level]
            prices = pd.DataFrame({t: raw[t]["Close"] for t in available})
    else:
        # Flat columns: single ticker, no grouping
        if "Close" in raw.columns:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            return pd.DataFrame()

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    return prices


def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """
    Quick validity check for a list of tickers.
    Returns (valid_tickers, invalid_tickers).
    """
    valid, invalid = [], []
    for t in tickers:
        try:
            info = yf.Ticker(t).history(period="5d")
            if info is None or info.empty:
                invalid.append(t)
            else:
                valid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - forward-fill small gaps (non-trading days, holidays mismatch)
    - drop columns that are entirely NaN
    - drop leading rows before all assets have data
    """
    if prices.empty:
        return prices

    cleaned = prices.copy()
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.ffill().bfill()
    cleaned = cleaned.dropna(how="any")
    return cleaned


def fetch_benchmark(benchmark_ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch benchmark (e.g. NIFTY 50 -> '^NSEI') adjusted close as a Series.
    """
    df = fetch_price_history([benchmark_ticker], start, end)
    if df.empty:
        return pd.Series(dtype=float)
    return df[benchmark_ticker]
