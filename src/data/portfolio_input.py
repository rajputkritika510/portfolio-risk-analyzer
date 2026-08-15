"""
portfolio_input.py
--------------------
Robust parser for user-uploaded portfolio files (CSV or Excel).

Real-world files are messy: column headers vary ("Buy Price (₹)",
"Purchase Price", "Price"...), numbers come with currency symbols and
thousands separators ("₹3,050.00"), and extra columns (Company name,
Investment total, Buy Date) may be present. This module normalizes all
of that into a clean DataFrame with exactly: Ticker, Quantity, Buy Price.
"""

from __future__ import annotations

import io
import re
import pandas as pd

# Column-name aliases we recognise, in priority order.
TICKER_ALIASES = ["ticker", "symbol", "stock", "scrip"]
QUANTITY_ALIASES = ["quantity", "qty", "shares", "units", "no of shares", "no. of shares"]
PRICE_ALIASES = ["buy price", "purchase price", "avg price", "average price",
                  "cost price", "price", "buy rate"]


class PortfolioFileError(Exception):
    """Raised when a portfolio file can't be parsed into the required columns."""


def _normalize_col(col: str) -> str:
    """Lowercase, strip currency symbols/parentheses/extra spaces for matching."""
    col = str(col).lower()
    col = re.sub(r"\(.*?\)", "", col)     # remove "(₹)", "(rs.)", etc.
    col = re.sub(r"[^a-z0-9. ]", "", col)  # drop currency symbols/punctuation
    return col.strip()


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {c: _normalize_col(c) for c in columns}
    # exact match first
    for alias in aliases:
        for original, norm in normalized.items():
            if norm == alias:
                return original
    # then partial/contains match
    for alias in aliases:
        for original, norm in normalized.items():
            if alias in norm:
                return original
    return None


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Strip currency symbols (₹, $, Rs, etc.), thousands separators, and
    stray whitespace, then convert to float. Handles both string and
    already-numeric columns.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)  # keep digits, dot, minus only
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_portfolio_file(uploaded_file) -> pd.DataFrame:
    """
    Read a Streamlit UploadedFile (CSV or Excel) and return a DataFrame
    with exactly the columns: Ticker, Quantity, Buy Price.

    Raises PortfolioFileError with a user-friendly message on failure.
    """
    filename = getattr(uploaded_file, "name", "uploaded_file")
    raw_bytes = uploaded_file.read()

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            # Try common encodings; portfolio exports are often Excel-saved CSVs (cp1252/latin-1)
            df = None
            last_err = None
            for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin-1"]:
                try:
                    df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                    break
                except Exception as e:
                    last_err = e
            if df is None:
                raise PortfolioFileError(f"Could not read the CSV file ({last_err}).")
    except PortfolioFileError:
        raise
    except Exception as e:
        raise PortfolioFileError(f"Could not read '{filename}': {e}")

    if df is None or df.empty:
        raise PortfolioFileError(f"'{filename}' appears to be empty.")

    columns = list(df.columns)
    ticker_col = _find_column(columns, TICKER_ALIASES)
    qty_col = _find_column(columns, QUANTITY_ALIASES)
    price_col = _find_column(columns, PRICE_ALIASES)

    missing = []
    if ticker_col is None:
        missing.append("Ticker")
    if qty_col is None:
        missing.append("Quantity")
    if price_col is None:
        missing.append("Buy Price")

    if missing:
        raise PortfolioFileError(
            f"Could not find column(s) {missing} in '{filename}'. "
            f"Found columns: {columns}. "
            f"Expected something like: Ticker, Quantity, Buy Price."
        )

    result = pd.DataFrame({
        "Ticker": df[ticker_col].astype(str).str.strip(),
        "Quantity": _clean_numeric_series(df[qty_col]),
        "Buy Price": _clean_numeric_series(df[price_col]),
    })

    result = result.dropna(subset=["Ticker", "Quantity", "Buy Price"])
    result = result[result["Ticker"] != ""]

    if result.empty:
        raise PortfolioFileError(
            f"No valid rows found in '{filename}' after cleaning. "
            f"Please check that Quantity and Buy Price contain numbers."
        )

    return result.reset_index(drop=True)
