# 📈 Portfolio Performance & Risk Analyzer

A Python-based **portfolio analytics and risk-management platform** that evaluates
portfolio performance, risk-adjusted returns, drawdowns, Value-at-Risk (VaR/CVaR),
diversification, benchmark comparison, and constrained mean-variance optimization —
all through an interactive Streamlit dashboard.

> Built as a Finance + Python analytics project. Educational / research tool — **not
> investment advice**.

---

## ✨ Features

| Category | What it does |
|---|---|
| **Portfolio Engine** | Tracks holdings, live valuation, weights, and P&L from historical adjusted-close prices |
| **Performance Engine** | Cumulative returns, CAGR, XIRR (money-weighted return) |
| **Risk Engine** | Volatility, Sharpe, Sortino, Beta, Alpha, Treynor, Max Drawdown |
| **Tail Risk** | VaR — Historical, Parametric & Monte Carlo methods, plus CVaR (Expected Shortfall) |
| **Diversification** | Herfindahl-Hirschman Index (HHI), sector-style concentration, correlation heatmap, per-asset **risk contribution** |
| **Benchmark Comparison** | Portfolio vs. NIFTY 50 / SENSEX / S&P 500 / NASDAQ |
| **Optimization Engine** | Minimum Variance, Maximum Sharpe, and Efficient Frontier (constrained mean-variance optimization via SciPy) |
| **Dashboard** | 6-tab interactive Streamlit UI — Overview, Holdings, Allocation, Risk, Performance, Optimization |

---

## 🖼️ Dashboard Preview

_Add your own screenshots here after running the app locally:_

```
assets/screenshots/overview.png
assets/screenshots/risk.png
assets/screenshots/optimization.png
```

---

## 🏗️ Architecture

```
USER PORTFOLIO
     │
     ▼
Portfolio Input (Ticker / Quantity / Buy Price)
     │
     ▼
Market Data Engine  ── yfinance (adjusted close, splits/dividends handled)
     │
     ▼
Data Cleaning & Validation
     │
     ▼
Portfolio Engine  ── value, weights, P&L
     │
     ├── Performance Engine  (CAGR, returns)
     └── Risk Engine  (volatility, Sharpe, VaR, CVaR, drawdown, correlation, HHI)
     │
     ▼
Benchmark Engine
     │
     ▼
Optimization Engine  ── SciPy constrained mean-variance optimization
     │
     ▼
Streamlit Dashboard
```

The app follows a **separation of concerns** design: `app.py` only handles UI —
all data fetching and math live in `/src`, which makes the logic independently
testable (see `/tests`).

---

## 📁 Project Structure

```
portfolio-risk-analyzer/
│
├── app.py                        # Streamlit dashboard (UI orchestration only)
│
├── src/
│   ├── data/
│   │   └── market_data.py        # yfinance fetch, validation, cleaning
│   ├── portfolio/
│   │   └── engine.py             # Portfolio & Holding classes — valuation, weights, P&L
│   ├── analytics/
│   │   ├── returns.py            # CAGR, cumulative returns, XIRR
│   │   └── risk.py               # Volatility, Sharpe, Sortino, Beta, VaR, CVaR, HHI, risk contribution
│   └── optimization/
│       └── optimizer.py          # Min-variance / Max-Sharpe / Efficient Frontier (SciPy)
│
├── data/
│   └── sample_portfolio.csv      # Demo portfolio to try the app instantly
│
├── tests/
│   └── test_returns.py           # Unit tests for core math (pytest)
│
├── .streamlit/
│   └── config.toml               # Dark theme configuration
│
├── assets/screenshots/           # Put dashboard screenshots here for the README
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

### 5. Try it instantly
In the sidebar, choose **"Use sample portfolio"** and click **Analyze Portfolio** —
no manual data entry needed.

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive dashboard
- **yfinance** — market data (unofficial Yahoo Finance API wrapper — research/educational use)
- **Pandas / NumPy** — data processing
- **SciPy** — constrained portfolio optimization
- **Plotly** — interactive charts

---

## 📌 Roadmap

This is **V1–V4** of an 8-version plan. Planned future versions:

- [ ] **V5 — Backtesting Engine**: periodic rebalancing, transaction costs, walk-forward evaluation, look-ahead-bias-free strategy simulation
- [ ] **V6 — Multi-cash-flow XIRR**: support deposits/withdrawals over time (Time-Weighted vs. Money-Weighted return)
- [ ] **V7 — Scenario & Stress Testing**: market shock simulations (e.g. "NIFTY -10%")
- [ ] **V8 — Automated PDF/CSV Reports**: one-click portfolio report export

Contributions and suggestions are welcome — feel free to open an issue.

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes**. It does not
constitute financial or investment advice. Market data is sourced via `yfinance`,
an unofficial, community-maintained package — always verify figures against an
official source before making real financial decisions.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 🙋 About

Built as a Finance + Python + Data Analytics portfolio project, demonstrating:
portfolio management, risk management, quantitative analysis, benchmarking, and
portfolio optimization — implemented with a clean, testable Python architecture
and an interactive Streamlit dashboard.

If you find this useful, consider ⭐ starring the repo!
