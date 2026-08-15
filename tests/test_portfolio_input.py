import sys
import os
import io
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.portfolio_input import read_portfolio_file, PortfolioFileError


class _FakeUpload:
    """Mimics a Streamlit UploadedFile for testing."""
    def __init__(self, data: bytes, name: str):
        self.name = name
        self._data = data

    def read(self):
        return self._data


def test_clean_csv_with_currency_symbols():
    csv_text = (
        "Ticker,Company,Quantity,Buy Date,Buy Price (₹),Investment (₹)\n"
        "TCS.NS,TCS,10,01-01-2026,\"₹3,050.00\",\"₹30,500.00\"\n"
        "INFY.NS,Infosys,15,01-01-2026,\"₹1,580.00\",\"₹23,700.00\"\n"
    )
    upload = _FakeUpload(csv_text.encode("utf-8"), "portfolio.csv")
    df = read_portfolio_file(upload)
    assert list(df.columns) == ["Ticker", "Quantity", "Buy Price"]
    assert df.loc[0, "Ticker"] == "TCS.NS"
    assert df.loc[0, "Quantity"] == 10
    assert df.loc[0, "Buy Price"] == 3050.0


def test_alternate_column_names():
    csv_text = "Symbol,Shares,Purchase Price\nTCS.NS,10,3050\n"
    upload = _FakeUpload(csv_text.encode("utf-8"), "portfolio.csv")
    df = read_portfolio_file(upload)
    assert df.loc[0, "Ticker"] == "TCS.NS"
    assert df.loc[0, "Quantity"] == 10
    assert df.loc[0, "Buy Price"] == 3050.0


def test_missing_required_column_raises():
    csv_text = "Ticker,Quantity\nTCS.NS,10\n"
    upload = _FakeUpload(csv_text.encode("utf-8"), "portfolio.csv")
    with pytest.raises(PortfolioFileError):
        read_portfolio_file(upload)


def test_excel_file_supported():
    df = pd.DataFrame({
        "Symbol": ["TCS.NS", "INFY.NS"],
        "Shares": [10, 15],
        "Purchase Price (Rs.)": ["₹3,050.00", "₹1,580.00"],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    upload = _FakeUpload(buf.read(), "portfolio.xlsx")
    result = read_portfolio_file(upload)
    assert result.loc[0, "Ticker"] == "TCS.NS"
    assert result.loc[0, "Buy Price"] == 3050.0
