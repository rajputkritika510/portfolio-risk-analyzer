"""
app.py
-------
Portfolio Performance & Risk Analyzer
Streamlit dashboard — UI orchestration only. All calculations live in /src.

Run with:  streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data.market_data import fetch_price_history, fetch_benchmark, clean_price_data
from src.data.portfolio_input import read_portfolio_file, PortfolioFileError
from src.portfolio.engine import Holding, Portfolio
from src.analytics import returns as ret_engine
from src.analytics import risk as risk_engine
from src.optimization import optimizer as opt_engine

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Portfolio Performance & Risk Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CUSTOM CSS — dark finance-terminal theme + subtle animations
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Animated gradient header */
.hero-header {
    padding: 1.6rem 2rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364, #1a2980);
    background-size: 300% 300%;
    animation: gradientShift 10s ease infinite;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.hero-title { color: #ffffff; font-size: 1.9rem; font-weight: 800; margin: 0; }
.hero-sub { color: #d7e4f0; font-size: 0.95rem; margin-top: 4px; }

/* KPI cards */
.kpi-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color .25s ease;
    animation: fadeInUp 0.6s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.35);
    border-color: rgba(120,180,255,0.5);
}
.kpi-label { font-size: 0.78rem; color: #9fb3c8; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 1.55rem; font-weight: 800; margin-top: 2px; }
.kpi-delta-pos { color: #29d391; font-weight: 600; font-size: 0.85rem; }
.kpi-delta-neg { color: #ff5c6c; font-weight: 600; font-size: 0.85rem; }

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(14px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Section headers */
.section-title {
    font-size: 1.15rem; font-weight: 700; margin: 1.2rem 0 0.6rem 0;
    border-left: 4px solid #4f8ef7; padding-left: 10px;
}

/* Badges */
.badge-low { background:#123b28; color:#29d391; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600;}
.badge-mod { background:#3b3512; color:#f7c948; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600;}
.badge-high{ background:#3b1212; color:#ff5c6c; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600;}

/* Tabs a bit bigger */
button[data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ======================================================================
# HEADER
# ======================================================================
st.markdown("""
<div class="hero-header">
    <p class="hero-title">📈 Portfolio Performance & Risk Analyzer</p>
    <p class="hero-sub">Market Analytics • Risk Engine • Benchmark Comparison • Portfolio Optimization</p>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# SIDEBAR — PORTFOLIO INPUT
# ======================================================================
st.sidebar.header("⚙️ Portfolio Setup")

input_mode = st.sidebar.radio("How do you want to enter holdings?", ["Upload File", "Enter manually", "Use sample portfolio"])

default_df = pd.DataFrame({
    "Ticker": ["TCS.NS", "INFY.NS", "RELIANCE.NS"],
    "Quantity": [10, 15, 12],
    "Buy Price": [3200, 1450, 2300],
})

if input_mode == "Upload File":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel (Ticker, Quantity, Buy Price)",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is not None:
        try:
            holdings_df = read_portfolio_file(uploaded)
            st.sidebar.success(f"Loaded {len(holdings_df)} holding(s) from '{uploaded.name}'.")
        except PortfolioFileError as e:
            st.sidebar.error(str(e))
            holdings_df = default_df
    else:
        holdings_df = default_df
elif input_mode == "Use sample portfolio":
    holdings_df = pd.read_csv("data/sample_portfolio.csv")[["Ticker", "Quantity", "Buy Price"]]
else:
    holdings_df = default_df

st.sidebar.caption("Tickers should follow Yahoo Finance format, e.g. **TCS.NS**, **AAPL**, **RELIANCE.NS**")
holdings_df = st.sidebar.data_editor(holdings_df, num_rows="dynamic", use_container_width=True, key="holdings_editor")

benchmark_options = {
    "NIFTY 50 (India)": "^NSEI",
    "SENSEX (India)": "^BSESN",
    "S&P 500 (US)": "^GSPC",
    "NASDAQ (US)": "^IXIC",
}
benchmark_label = st.sidebar.selectbox("Benchmark", list(benchmark_options.keys()))
benchmark_ticker = benchmark_options[benchmark_label]

risk_free_rate = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 12.0, 6.0, 0.25) / 100

period_label = st.sidebar.selectbox("Analysis Period", ["6M", "1Y", "2Y", "3Y", "5Y", "Max"], index=1)
period_map = {"6M": 182, "1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "Max": 3650}
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=period_map[period_label])

max_weight_pct = st.sidebar.slider("Optimizer: max weight per asset (%)", 20, 100, 40, 5)

analyze = st.sidebar.button("🔍 Analyze Portfolio", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • yfinance • SciPy • Plotly")

# ======================================================================
# MAIN LOGIC
# ======================================================================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if analyze:
    st.session_state.analyzed = True

if not st.session_state.analyzed:
    st.info("👈 Enter your holdings in the sidebar and click **Analyze Portfolio** to get started.")
    st.stop()

holdings_df = holdings_df.dropna(subset=["Ticker", "Quantity", "Buy Price"])
holdings_df = holdings_df[holdings_df["Ticker"].astype(str).str.strip() != ""]

if holdings_df.empty:
    st.warning("Please add at least one valid holding.")
    st.stop()

tickers = holdings_df["Ticker"].astype(str).str.strip().tolist()

with st.spinner("Fetching market data..."):
    prices = fetch_price_history(tickers, str(start_date), str(end_date))
    prices = clean_price_data(prices)
    benchmark_prices = fetch_benchmark(benchmark_ticker, str(start_date), str(end_date))

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.error(f"⚠️ Could not retrieve data for: {', '.join(missing)}. Please check the ticker symbol(s).")

valid_holdings_df = holdings_df[holdings_df["Ticker"].isin(prices.columns)]
if valid_holdings_df.empty:
    st.error("No valid tickers found. Please correct the ticker symbols and try again.")
    st.stop()

holdings = [Holding(r["Ticker"], float(r["Quantity"]), float(r["Buy Price"])) for _, r in valid_holdings_df.iterrows()]
portfolio = Portfolio(holdings, prices)

port_value = portfolio.portfolio_value()
port_returns = ret_engine.daily_returns(port_value)

bench_aligned = benchmark_prices.reindex(port_value.index).ffill().bfill() if not benchmark_prices.empty else pd.Series(dtype=float)
bench_returns = ret_engine.daily_returns(bench_aligned) if not bench_aligned.empty else pd.Series(dtype=float)

asset_returns = prices[valid_holdings_df["Ticker"].tolist()].pct_change().dropna()

# ======================================================================
# TABS
# ======================================================================
tab_overview, tab_holdings, tab_alloc, tab_risk, tab_perf, tab_opt = st.tabs(
    ["🏠 Overview", "📊 Holdings", "🥧 Allocation", "⚠️ Risk", "📈 Performance", "🧠 Optimization"]
)

# ---------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------------------
with tab_overview:
    current_value = float(port_value.iloc[-1])
    total_pnl = portfolio.total_pnl()
    total_return = portfolio.total_return_pct()
    port_cagr = ret_engine.cagr(port_value) * 100 if len(port_value) > 5 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Current Value", f"₹{current_value:,.0f}")
    with c2: kpi_card("Total Invested", f"₹{portfolio.invested_amount:,.0f}")
    with c3: kpi_card("Total P&L", f"₹{total_pnl:,.0f}", f"{total_return:.2f}%", total_pnl >= 0)
    with c4: kpi_card("CAGR", f"{port_cagr:.2f}%" if not np.isnan(port_cagr) else "N/A")

    section("Portfolio Growth vs Benchmark")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_value.index, y=(port_value / port_value.iloc[0] - 1) * 100,
        name="Portfolio", line=dict(color="#4f8ef7", width=3),
    ))
    if not bench_aligned.empty:
        fig.add_trace(go.Scatter(
            x=bench_aligned.index, y=(bench_aligned / bench_aligned.iloc[0] - 1) * 100,
            name=benchmark_label, line=dict(color="#f7a44f", width=2, dash="dot"),
        ))
    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified",
                       yaxis_title="Cumulative Return (%)", margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if not bench_returns.empty:
        port_total_ret = (port_value.iloc[-1] / port_value.iloc[0] - 1) * 100
        bench_total_ret = (bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1) * 100
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Portfolio Return", f"{port_total_ret:.2f}%")
        with c2: kpi_card(f"{benchmark_label} Return", f"{bench_total_ret:.2f}%")
        with c3:
            outperf = port_total_ret - bench_total_ret
            kpi_card("Outperformance", f"{outperf:.2f}%", delta_positive=outperf >= 0)

# ---------------------------------------------------------------------
# TAB 2 — HOLDINGS
# ---------------------------------------------------------------------
with tab_holdings:
    section("Current Holdings")
    snapshot = portfolio.latest_snapshot()
    st.dataframe(
        snapshot.style.format({
            "Buy Price": "₹{:.2f}", "Current Price": "₹{:.2f}", "Current Value": "₹{:.2f}",
            "Weight (%)": "{:.2f}%", "P&L (₹)": "₹{:.2f}", "Return (%)": "{:.2f}%",
        }).map(lambda v: "color:#29d391" if isinstance(v, (int, float)) and v > 0 else "color:#ff5c6c",
               subset=["P&L (₹)", "Return (%)"]),
        use_container_width=True, height=320,
    )

    section("Individual Holding Returns")
    fig = px.bar(snapshot, x="Ticker", y="Return (%)", color="Return (%)",
                 color_continuous_scale=["#ff5c6c", "#f7c948", "#29d391"])
    fig.update_layout(template="plotly_dark", height=380, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3 — ALLOCATION
# ---------------------------------------------------------------------
with tab_alloc:
    weights = portfolio.current_weights()

    c1, c2 = st.columns([1, 1])
    with c1:
        section("Portfolio Allocation")
        fig = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.55,
                                marker=dict(line=dict(color="#0e1117", width=2))))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Concentration")
        hhi = risk_engine.herfindahl_index(weights)
        label = risk_engine.concentration_label(hhi)
        badge_class = {"Low": "badge-low", "Moderate": "badge-mod", "High": "badge-high"}[label]
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Herfindahl-Hirschman Index (HHI)</div>
                <div class="kpi-value">{hhi:.3f}</div>
                <span class="{badge_class}">{label} concentration</span>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        top_holding = weights.sort_values(ascending=False)
        st.markdown("**Top holdings**")
        for tkr, w in top_holding.head(5).items():
            st.progress(min(float(w), 1.0), text=f"{tkr} — {w*100:.1f}%")

    section("Risk Contribution by Asset")
    st.caption("Which holding actually drives your portfolio's total risk (not just its weight).")
    contrib = risk_engine.risk_contribution(weights, asset_returns)
    fig = px.bar(contrib.sort_values(ascending=False), orientation="v",
                 labels={"value": "Risk Contribution (%)", "index": "Asset"},
                 color=contrib.sort_values(ascending=False).values,
                 color_continuous_scale="Blues")
    fig.update_layout(template="plotly_dark", height=380, showlegend=False, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4 — RISK
# ---------------------------------------------------------------------
with tab_risk:
    vol = risk_engine.volatility(port_returns) * 100
    sharpe = risk_engine.sharpe_ratio(port_returns, risk_free_rate)
    sortino = risk_engine.sortino_ratio(port_returns, risk_free_rate)
    mdd = risk_engine.max_drawdown(port_value) * 100
    var_hist = risk_engine.var_historical(port_returns) * 100
    cvar_hist = risk_engine.cvar_historical(port_returns) * 100
    b = risk_engine.beta(port_returns, bench_returns) if not bench_returns.empty else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Volatility (ann.)", f"{vol:.2f}%")
    with c2: kpi_card("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
    with c3: kpi_card("Sortino Ratio", f"{sortino:.2f}" if not np.isnan(sortino) else "N/A")
    with c4: kpi_card("Beta", f"{b:.2f}" if not np.isnan(b) else "N/A")
    with c5: kpi_card("Max Drawdown", f"{mdd:.2f}%")
    with c6: kpi_card("VaR 95% (1-day)", f"{var_hist:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        section("Drawdown Over Time")
        dd = risk_engine.drawdown_series(port_value) * 100
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                    line=dict(color="#ff5c6c"), fillcolor="rgba(255,92,108,0.25)"))
        fig.update_layout(template="plotly_dark", height=360, yaxis_title="Drawdown (%)",
                           margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("VaR & CVaR — Method Comparison (95%)")
        var_p = risk_engine.var_parametric(port_returns) * 100
        var_mc = risk_engine.var_monte_carlo(port_returns) * 100
        cmp_df = pd.DataFrame({
            "Method": ["Historical", "Parametric", "Monte Carlo"],
            "VaR (%)": [var_hist, var_p, var_mc],
        })
        fig = px.bar(cmp_df, x="Method", y="VaR (%)", color="Method",
                     color_discrete_sequence=["#4f8ef7", "#f7a44f", "#29d391"])
        fig.update_layout(template="plotly_dark", height=360, showlegend=False, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"CVaR (Expected Shortfall) 95%: **{cvar_hist:.2f}%** — average loss on the worst 5% of days.")

    section("Correlation Heatmap")
    corr = risk_engine.correlation_matrix(asset_returns)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(template="plotly_dark", height=420, margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Volatility** — annualized standard deviation of daily returns; a measure of total risk.
- **Sharpe Ratio** — excess return earned per unit of total volatility.
- **Sortino Ratio** — like Sharpe, but only penalizes *downside* volatility.
- **Beta** — sensitivity of the portfolio to benchmark movements (1 = moves with the market).
- **Max Drawdown** — largest peak-to-trough decline over the selected period.
- **VaR (Value at Risk)** — estimated loss threshold at the given confidence level that returns are not expected to exceed on a given day, under the chosen methodology.
- **CVaR (Expected Shortfall)** — the average loss *given* that the VaR threshold was breached — a better view of tail risk.
        """)

# ---------------------------------------------------------------------
# TAB 5 — PERFORMANCE
# ---------------------------------------------------------------------
with tab_perf:
    section("Rolling 30-Day Volatility")
    rolling_vol = port_returns.rolling(30).std() * np.sqrt(252) * 100
    fig = go.Figure(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, line=dict(color="#f7c948")))
    fig.update_layout(template="plotly_dark", height=340, yaxis_title="Volatility (%)",
                       margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    section("Monthly Return Heatmap")
    monthly = port_value.resample("ME").last().pct_change().dropna() * 100
    if len(monthly) > 0:
        m_df = pd.DataFrame({"Return": monthly.values}, index=monthly.index)
        m_df["Year"] = m_df.index.year
        m_df["Month"] = m_df.index.strftime("%b")
        pivot = m_df.pivot(index="Year", columns="Month", values="Return")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn", aspect="auto",
                         labels=dict(color="Return (%)"))
        fig.update_layout(template="plotly_dark", height=320, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Not enough history yet for a monthly heatmap — try a longer analysis period.")

# ---------------------------------------------------------------------
# TAB 6 — OPTIMIZATION
# ---------------------------------------------------------------------
with tab_opt:
    st.caption("Mean-Variance Optimization (Modern Portfolio Theory) — for education/research purposes, not investment advice.")

    max_w = max_weight_pct / 100

    with st.spinner("Running optimizer..."):
        try:
            max_sharpe = opt_engine.maximum_sharpe_portfolio(asset_returns, risk_free_rate, 0.0, max_w)
            min_var = opt_engine.minimum_variance_portfolio(asset_returns, 0.0, max_w)
            frontier = opt_engine.efficient_frontier(asset_returns, 30, 0.0, max_w)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()

    current_ret = (port_returns.mean() * 252)
    current_vol = risk_engine.volatility(port_returns)
    current_sharpe = risk_engine.sharpe_ratio(port_returns, risk_free_rate)

    c1, c2 = st.columns(2)
    with c1:
        section("Current vs. Max-Sharpe Portfolio")
        cmp_df = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility", "Sharpe"],
            "Current": [f"{current_ret*100:.2f}%", f"{current_vol*100:.2f}%", f"{current_sharpe:.2f}"],
            "Optimized (Max Sharpe)": [f"{max_sharpe['expected_return']*100:.2f}%",
                                        f"{max_sharpe['volatility']*100:.2f}%",
                                        f"{max_sharpe['sharpe']:.2f}"],
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    with c2:
        section("Suggested Weights (Max Sharpe)")
        fig = go.Figure(go.Bar(x=max_sharpe["weights"].index, y=max_sharpe["weights"].values * 100,
                                marker_color="#4f8ef7"))
        fig.update_layout(template="plotly_dark", height=300, yaxis_title="Weight (%)",
                           margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    section("Efficient Frontier")
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(go.Scatter(x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
                                  mode="lines", name="Efficient Frontier", line=dict(color="#4f8ef7", width=3)))
    fig.add_trace(go.Scatter(x=[current_vol * 100], y=[current_ret * 100], mode="markers",
                              name="Current Portfolio", marker=dict(size=14, color="#f7a44f", symbol="diamond")))
    fig.add_trace(go.Scatter(x=[max_sharpe["volatility"] * 100], y=[max_sharpe["expected_return"] * 100],
                              mode="markers", name="Max Sharpe", marker=dict(size=16, color="#29d391", symbol="star")))
    fig.add_trace(go.Scatter(x=[min_var["volatility"] * 100], y=[min_var["expected_return"] * 100],
                              mode="markers", name="Min Variance", marker=dict(size=14, color="#ff5c6c", symbol="x")))
    fig.update_layout(template="plotly_dark", height=460, xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
                       margin=dict(t=20, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚠️ Important note on optimization"):
        st.markdown("""
Mean-variance optimization uses **historical returns as a proxy for expected future returns**,
which is a strong assumption. Results can be sensitive to the estimation period and may suggest
concentrated allocations. The **max weight per asset** slider in the sidebar constrains this.
Treat these outputs as a *starting point for analysis*, not a recommendation to trade.
        """)

st.markdown("---")
st.caption("⚠️ This tool is for educational and research purposes only and does not constitute financial advice. "
           "Market data via Yahoo Finance (yfinance) — unofficial, community-maintained package.")
