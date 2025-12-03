"""
Streamlit Dashboard for Portfolio Construction and Analysis.

This dashboard allows users to:
- Input company names/tickers
- Select portfolio construction methods
- View performance metrics and visualizations
- Compare different optimization strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import math

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from src.data import (
    download_data,
    get_returns,
    get_company_info,
    prepare_portfolio_data,
    compute_features,
    get_cache_info,
    clear_cache,
)
from src.risk import get_covariance_matrix
from src.optimize import optimize_portfolio
from src.backtest import walk_forward_backtest, simple_backtest
from src.metrics import (
    calculate_all_metrics,
    rolling_sharpe,
    rolling_volatility,
    maximum_drawdown,
    value_at_risk,
    conditional_value_at_risk,
)
from src.forecast import (
    forecast_portfolio_returns,
    ensemble_forecast,
)

# Quick-reference list of popular tickers that many users ask for.
# Keeping this data structure close to the top allows other modules to import it later if needed.
COMMON_COMPANY_TICKERS = [
    {"Company": "Apple Inc.", "Ticker": "AAPL", "Sector": "Technology"},
    {"Company": "Microsoft Corp.", "Ticker": "MSFT", "Sector": "Technology"},
    {"Company": "Alphabet (Google)", "Ticker": "GOOGL", "Sector": "Communication Services"},
    {"Company": "Amazon.com Inc.", "Ticker": "AMZN", "Sector": "Consumer Discretionary"},
    {"Company": "Meta Platforms", "Ticker": "META", "Sector": "Communication Services"},
    {"Company": "NVIDIA Corp.", "Ticker": "NVDA", "Sector": "Technology"},
    {"Company": "Tesla Inc.", "Ticker": "TSLA", "Sector": "Consumer Discretionary"},
    {"Company": "Netflix Inc.", "Ticker": "NFLX", "Sector": "Communication Services"},
    {"Company": "JPMorgan Chase", "Ticker": "JPM", "Sector": "Financials"},
    {"Company": "Johnson & Johnson", "Ticker": "JNJ", "Sector": "Healthcare"},
    {"Company": "Bank of America", "Ticker": "BAC", "Sector": "Financials"},
    {"Company": "The Walt Disney Company", "Ticker": "DIS", "Sector": "Communication Services"},
    {"Company": "The Home Depot", "Ticker": "HD", "Sector": "Consumer Discretionary"},
    {"Company": "Mastercard Inc.", "Ticker": "MA", "Sector": "Financials"},
    {"Company": "Nike Inc.", "Ticker": "NKE", "Sector": "Consumer Discretionary"},
    {"Company": "Procter & Gamble", "Ticker": "PG", "Sector": "Consumer Staples"},
    {"Company": "Visa Inc.", "Ticker": "V", "Sector": "Financials"},
    {"Company": "Walmart Inc.", "Ticker": "WMT", "Sector": "Consumer Staples"},
]

MODEL_REFERENCE = [
    {"Model": "Ledoit–Wolf", "Category": "Analyze", "Purpose": "Covariance estimation"},
    {"Model": "GLASSO", "Category": "Analyze", "Purpose": "Sparse inverse covariance"},
    {"Model": "GARCH(1,1)", "Category": "Predict", "Purpose": "Volatility forecasts"},
    {"Model": "DCC", "Category": "Predict", "Purpose": "Correlation forecasts"},
    {"Model": "Markowitz", "Category": "Analyze/Optimize", "Purpose": "Portfolio construction"},
    {"Model": "Minimum Variance", "Category": "Analyze/Optimize", "Purpose": "Risk minimization"},
    {"Model": "Sharpe Maximization", "Category": "Analyze/Optimize", "Purpose": "Risk-adjusted returns"},
    {"Model": "Black–Litterman", "Category": "Adjusts/Analyzes", "Purpose": "Blended expected returns"},
    {"Model": "CVaR Optimization", "Category": "Analyze/Optimize", "Purpose": "Tail-risk minimization"},
]

MODEL_REFERENCE_DF = pd.DataFrame(MODEL_REFERENCE)

# Friendly labels that describe what each optimization / covariance choice does.
OPTIMIZATION_OPTIONS = [
    ("Balance growth vs risk (diversified blend)", "markowitz"),
    ("Keep volatility at a minimum", "min_variance"),
    ("Maximize risk-adjusted payoff", "sharpe"),
    ("Blend market consensus with your views", "black_litterman"),
    ("Protect the tail outcomes", "cvar"),
]

RISK_MODEL_OPTIONS = [
    ("Stabilize noisy covariance estimates", "ledoit_wolf"),
    ("Use raw historical relationships", "sample"),
    ("Detect sparse dependency structure", "glasso"),
    ("Forecast time-varying volatility", "garch"),
]

INSIGHT_SYSTEM_PROMPT = (
    "You are a concise financial coach explaining charts to beginners. "
    "Keep explanations to 2-3 sentences, avoid jargon, and connect numbers to practical meaning. "
    "Close with a plain-language cue on whether to lean in or stay cautious, framed as education—not advice."
)


def format_pct(value: float, decimals: int = 1) -> str:
    """Format percentages safely."""
    try:
        return f"{value * 100:.{decimals}f}%"
    except Exception:
        return "n/a"


def generate_ai_insight(title: str, stats_text: str, default_text: str) -> str:
    """
    Return an LLM-generated explanation when enabled, otherwise fall back to a deterministic summary.
    """
    ai_enabled = st.session_state.get("enable_ai_insights", False)
    if not ai_enabled:
        return default_text
    
    api_key = st.session_state.get("llm_api_key")
    if not api_key:
        return default_text + " (Add an OpenAI API key in the sidebar to enable AI commentary.)"
    
    model = st.session_state.get("llm_model", "gpt-3.5-turbo")
    prompt = (
        f"Chart: {title}\n"
        f"Context stats: {stats_text}\n"
        "Explain what the chart says about the portfolio for a novice investor, then finish with either "
        "'Lean in because …' or 'Stay cautious because …'. This is educational, not investment advice."
    )
    
    try:
        from openai import OpenAI  # type: ignore
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return default_text + f" (AI unavailable: {e})"


def render_insight_box(title: str, stats_text: str, default_text: str):
    """Render a styled insight card."""
    insight_text = generate_ai_insight(title, stats_text, default_text)
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">AI insight — {title}</div>
            <div class="insight-body">{insight_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Page configuration
st.set_page_config(
    page_title="FinLove - Portfolio Construction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    body {
        background: radial-gradient(circle at 10% 20%, #f8fbff 0%, #f3f6ff 25%, #eef3ff 50%, #fdfdff 100%);
        color: #1f2933;
    }
    .main-header {
        font-size: 2.7rem;
        font-weight: 800;
        color: #0f4c81;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: linear-gradient(135deg, #f8fbff 0%, #eef3ff 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(15, 76, 129, 0.15);
        box-shadow: 0 4px 12px rgba(15, 76, 129, 0.08);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(0,0,0,0.05);
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        font-size: 0.95rem;
        color: #4a4a4a;
        margin-bottom: 0.4rem;
    }
    .metric-card p {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0f4c81;
        margin: 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #0f4c81 0%, #0a2f57 100%);
        color: #f7fbff;
        padding: 1rem 1.1rem;
        border-radius: 0.85rem;
        box-shadow: 0 8px 24px rgba(10, 47, 87, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.12);
        height: 100%;
    }
    .insight-title {
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        font-size: 0.75rem;
        opacity: 0.85;
    }
    .insight-body {
        font-size: 0.95rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 FinLove Portfolio Construction Dashboard</div>', unsafe_allow_html=True)

# Quick reference so users immediately see available companies/tickers.
with st.container():
    st.subheader("📋 Quick Company → Ticker Reference")
    quick_ticker_df = pd.DataFrame(COMMON_COMPANY_TICKERS)
    st.dataframe(quick_ticker_df, use_container_width=True, hide_index=True)
    st.caption("Curated from the latest dataset additions so newcomers instantly know which tickers they can try.")

# Explain each visualization once so first-time users understand the insights.
with st.expander("ℹ️ What each chart tells you", expanded=False):
    st.markdown("""
    - **Cumulative Returns** – tracks how a dollar invested in the portfolio compares with an equal-weight benchmark.
    - **Rolling Sharpe Ratio** – shows how risk-adjusted performance evolves (higher = better risk/return balance).
    - **Drawdown Chart** – highlights peak-to-trough losses so you can assess downside risk.
    - **Current Portfolio Weights** – displays the latest allocation so you can quickly review exposures.
    - **Weights Over Time** – reveals how allocations shifted during the backtest, surfacing turnover or drift.
    - **VaR / CVaR Metrics** – quantify worst-case daily losses at 95% confidence and the expected loss beyond that threshold.
    - **Rolling Volatility** – indicates how the portfolio’s realized risk is trending over time.
    - **Returns Distribution** – helps diagnose skew/kurtosis or fat tails relative to assumptions.
    """)

st.markdown("---")

# Data Management Section (at top of sidebar)
with st.sidebar.expander("📦 Data Management", expanded=False):
    cache_info = get_cache_info()
    st.write("**Cache Status:**")
    st.write(f"- Files: {cache_info['total_files']}")
    st.write(f"- Size: {cache_info['total_size_mb']:.2f} MB")
    if cache_info['newest_cache']:
        st.write(f"- Newest: {cache_info['newest_cache'].strftime('%Y-%m-%d %H:%M')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Cache Info"):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Cache"):
            cleared = clear_cache()
            st.success(f"Cleared {cleared} files")
            st.rerun()
    
    st.info("💡 **Tip:** Run `python download_data.py` to pre-download datasets for faster performance.")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Company/Ticker input
st.sidebar.subheader("1. Select Assets")
input_method = st.sidebar.radio(
    "Input Method",
    ["Company Ticker", "Sector ETFs (Default)"],
    help="Choose to input individual company tickers or use default sector ETFs"
)

if input_method == "Company Ticker":
    # Allow multiple company inputs
    ticker_input = st.sidebar.text_input(
        "Enter Company Ticker(s)",
        value="AAPL",
        help="Enter one or more ticker symbols separated by commas (e.g., AAPL,MSFT,GOOGL)"
    )
    
    if ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    else:
        tickers = ["AAPL"]
else:
    # Default sector ETFs
    default_etfs = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    tickers = default_etfs
    st.sidebar.info(f"Using default sector ETFs: {', '.join(tickers)}")

# Date range
st.sidebar.subheader("2. Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2015, 1, 1),
        max_value=datetime.today() - timedelta(days=30)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.today() - timedelta(days=1),
        max_value=datetime.today()
    )

# Optimization settings
st.sidebar.subheader("3. Portfolio Objective")
opt_labels = [label for label, _ in OPTIMIZATION_OPTIONS]
opt_label_selection = st.sidebar.selectbox(
    "How should the engine allocate your capital?",
    opt_labels,
    help="Pick the behavior that matches your goal—steady risk, tail protection, or conviction blending."
)
optimization_method = dict(OPTIMIZATION_OPTIONS)[opt_label_selection]

# Risk model
st.sidebar.subheader("4. Risk Engine")
risk_labels = [label for label, _ in RISK_MODEL_OPTIONS]
risk_label_selection = st.sidebar.selectbox(
    "How should relationships between assets be estimated?",
    risk_labels,
    help="Choose the description that best matches how you want covariance and volatility to be inferred."
)
risk_model = dict(RISK_MODEL_OPTIONS)[risk_label_selection]

# Capital amount – used in the Investment tab to convert optimal weights into dollar allocations.
st.sidebar.subheader("5. Investment Capital")
investment_amount = st.sidebar.number_input(
    "Total capital to allocate (USD)",
    min_value=0.0,
    value=10000.0,
    step=100.0,
    help=(
        "This amount will be distributed across assets according to the optimized weights. "
        "Set it to the notional money you plan to invest."
    ),
)

# Additional parameters related to risk and preferences.
st.sidebar.subheader("6. Risk Appetite")
risk_aversion = st.sidebar.slider(
    "Dial down risk-taking",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Higher values make the optimizer far more defensive."
)

# Backtest type
st.sidebar.subheader("7. Testing Style")
backtest_type = st.sidebar.radio(
    "Choose how to test the portfolio",
    ["Simple (One-time optimization)", "Walk-Forward (Rolling window)"]
)

# AI explanations
st.sidebar.subheader("8. AI Explanations")
enable_ai_insights = st.sidebar.checkbox(
    "Turn on AI chart explanations",
    value=st.session_state.get("enable_ai_insights", False),
    help="Requires an OpenAI API key. If off, you will see deterministic summaries."
)
st.session_state["enable_ai_insights"] = enable_ai_insights

llm_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=st.session_state.get("llm_api_key", ""),
    type="password",
    placeholder="sk-...",
    help="Key is stored only in your local Streamlit session."
)
if llm_api_key:
    st.session_state["llm_api_key"] = llm_api_key

llm_model = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-3.5-turbo"],
    index=0,
    help="Short explanations work well on lighter models."
)
st.session_state["llm_model"] = llm_model

# Advanced (optional) knobs so the main UI stays lightweight.
transaction_cost = 0.001
rebalance_band = 0.05
train_window = 36
test_window = 1

with st.sidebar.expander("Advanced Controls", expanded=False):
    transaction_cost = st.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=transaction_cost * 100,
        step=0.05,
        help="Proportional round-trip execution cost"
    ) / 100
    
    rebalance_band = st.slider(
        "Rebalance Band (%)",
        min_value=0.0,
        max_value=20.0,
        value=rebalance_band * 100,
        step=0.5,
        help="How much drift you tolerate before trading again"
    ) / 100
    
    if backtest_type == "Walk-Forward (Rolling window)":
        train_window = st.slider(
            "Training Window (months)",
            min_value=12,
            max_value=60,
            value=train_window,
            step=6
        )
        test_window = st.slider(
            "Test Window (months)",
            min_value=1,
            max_value=12,
            value=test_window,
            step=1
        )

# Main content area
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Downloading data and running analysis..."):
        try:
            # Download data
            returns, prices = prepare_portfolio_data(
                tickers,
                start_date=str(start_date),
                end_date=str(end_date)
            )
            
            if returns.empty or len(returns) < 60:
                st.error(f"❌ Insufficient data for tickers: {', '.join(tickers)}. Please try different tickers or date range.")
                st.stop()
            
            # Filter out assets with too many missing values
            returns = returns.dropna(axis=1, thresh=len(returns) * 0.8)
            
            if returns.empty:
                st.error("❌ No valid data after cleaning. Please check your inputs.")
                st.stop()
            
            # Update tickers to valid ones
            tickers = list(returns.columns)
            
            # Run backtest
            if backtest_type == "Simple (One-time optimization)":
                portfolio_returns, weights, metrics = simple_backtest(
                    returns,
                    optimization_method=optimization_method,
                    risk_model=risk_model,
                    transaction_cost=transaction_cost,
                    constraints={'long_only': True},
                    risk_aversion=risk_aversion
                )
                weights_history = pd.DataFrame([weights], index=[returns.index[-1]], columns=tickers)
            else:
                portfolio_returns, weights_history, metrics = walk_forward_backtest(
                    returns,
                    train_window=train_window,
                    test_window=test_window,
                    optimization_method=optimization_method,
                    risk_model=risk_model,
                    transaction_cost=transaction_cost,
                    rebalance_band=rebalance_band,
                    rebalance_frequency='monthly',
                    constraints={'long_only': True},
                    risk_aversion=risk_aversion
                )
                weights = weights_history.iloc[-1]
            
            # Store results in session state
            st.session_state['returns'] = returns
            st.session_state['prices'] = prices
            st.session_state['portfolio_returns'] = portfolio_returns
            st.session_state['weights'] = weights
            st.session_state['weights_history'] = weights_history
            st.session_state['metrics'] = metrics
            st.session_state['tickers'] = tickers
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.stop()

# Display results if available
if 'metrics' in st.session_state:
    metrics = st.session_state['metrics']
    portfolio_returns = st.session_state['portfolio_returns']
    weights = st.session_state['weights']
    weights_history = st.session_state['weights_history']
    returns = st.session_state['returns']
    prices = st.session_state['prices']
    tickers = st.session_state['tickers']
    
    st.success(f"✅ Analysis complete for {len(tickers)} assets: {', '.join(tickers)}")
    
    # Three main tabs matching the requested workflow:
    # 1) Investment Plan – turn optimized weights into a dollar investment plan.
    # 2) Analyze – in-depth risk and performance analysis using risk models from src/risk.py.
    # 3) Prediction – time-series models projecting future portfolio behavior.
    investment_tab, analyze_tab, predict_tab = st.tabs(
        ["💰 Investment Plan", "🔍 Analyze", "🔮 Prediction"]
    )
    
    # ------------------------------------------------------------------
    # 💰 Investment tab: convert optimal weights into a money allocation.
    # ------------------------------------------------------------------
    with investment_tab:
        st.subheader("Capital deployment based on optimized weights")
        
        if investment_amount <= 0:
            # If the user sets capital to zero, clearly explain why no plan is shown.
            st.warning(
                "Please enter a positive investment amount in the sidebar to see a dollar-based allocation plan."
            )
        else:
            # Compute dollar allocation for each asset from the optimized weights.
            dollar_allocation = weights * investment_amount
            
            # Build a human-readable table summarizing the investment plan.
            allocation_df = pd.DataFrame(
                {
                    "Asset": weights.index,
                    "Weight (%)": weights.values * 100,
                    "Allocation (USD)": dollar_allocation.values,
                }
            ).sort_values("Allocation (USD)", ascending=False)
            
            # High-level summary card.
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Capital", f"${investment_amount:,.2f}")
            with col_b:
                st.metric("Number of Assets", f"{len(weights.index)}")
            with col_c:
                st.metric(
                    "Optimization / Risk Engine",
                    f"{optimization_method}  |  {risk_model}",
                    help=(
                        "The weights were produced by the selected optimization method "
                        "combined with the chosen risk model from src/risk.py."
                    ),
                )
            
            st.markdown("#### Recommended allocation by asset")
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
            
            # Visualize the dollar allocations in a bar chart for quick interpretation.
            alloc_bar_fig = px.bar(
                allocation_df,
                x="Asset",
                y="Allocation (USD)",
                title="Dollar Allocation per Asset",
                text_auto=".2s",
            )
            alloc_bar_fig.update_layout(
                xaxis_title="Asset",
                yaxis_title="Allocation (USD)",
                height=400,
            )
            st.plotly_chart(alloc_bar_fig, use_container_width=True)
            
            st.caption(
                "This plan distributes your specified capital according to the optimized portfolio "
                "weights. You can change both the investment amount and configuration in the sidebar."
            )
    
    # ----------------------------------------------------------------------
    # 🔍 Analyze tab: performance, allocations, and risk model diagnostics.
    # ----------------------------------------------------------------------
    with analyze_tab:
        st.subheader("Key takeaways")
        metric_cols = st.columns(4)
        metric_cards = [
            ("Annualized Return", f"{metrics['annualized_return']*100:.2f}%"),
            ("Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%"),
            ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
            ("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%"),
        ]
        for col, (title, value) in zip(metric_cols, metric_cards):
            col.markdown(f"<div class='metric-card'><h3>{title}</h3><p>{value}</p></div>", unsafe_allow_html=True)
        
        st.markdown("#### Portfolio path")
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_benchmark = (1 + returns.mean(axis=1)).cumprod()
        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        perf_fig.add_trace(go.Scatter(
            x=cumulative_benchmark.index,
            y=cumulative_benchmark.values * 100,
            mode='lines',
            name='Equal-Weight Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        perf_fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=400
        )
        perf_cols = st.columns([3, 1.2])
        with perf_cols[0]:
            st.plotly_chart(perf_fig, use_container_width=True)
        port_total = cumulative_returns.iloc[-1] - 1
        bench_total = cumulative_benchmark.iloc[-1] - 1
        perf_gap = port_total - bench_total
        perf_default = (
            f"Portfolio delivered {format_pct(port_total, 2)} vs benchmark {format_pct(bench_total, 2)}, "
            f"for a spread of {format_pct(perf_gap, 2)}. "
            "Lean in if you believe this edge is durable; stay cautious if the lead came from a few lucky bursts."
        )
        perf_stats = (
            f"Portfolio total return {port_total:.4f}; benchmark {bench_total:.4f}; "
            f"best day {portfolio_returns.max():.4f}; worst day {portfolio_returns.min():.4f}."
        )
        with perf_cols[1]:
            render_insight_box("Cumulative Returns", perf_stats, perf_default)
        
        rolling_sharpe_series = rolling_sharpe(portfolio_returns, window=252)
        sharpe_fig = go.Figure()
        sharpe_fig.add_trace(go.Scatter(
            x=rolling_sharpe_series.index,
            y=rolling_sharpe_series.values,
            mode='lines',
            name='Rolling Sharpe (252 days)',
            line=dict(color='green', width=2)
        ))
        sharpe_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        sharpe_fig.update_layout(
            title="Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=300
        )
        sharpe_cols = st.columns([3, 1.2])
        with sharpe_cols[0]:
            st.plotly_chart(sharpe_fig, use_container_width=True)
        sharpe_latest = rolling_sharpe_series.dropna().iloc[-1] if not rolling_sharpe_series.dropna().empty else 0.0
        sharpe_mean = rolling_sharpe_series.mean()
        sharpe_above_zero = (rolling_sharpe_series > 0).mean()
        sharpe_default = (
            f"Latest rolling Sharpe is {sharpe_latest:.2f} vs average {sharpe_mean:.2f}; "
            f"{sharpe_above_zero*100:.0f}% of the window stays above zero. "
            "Lean in if Sharpe stays stable above zero; stay cautious if it’s chopping around flat or negative."
        )
        sharpe_stats = (
            f"Latest Sharpe {sharpe_latest:.3f}; mean {sharpe_mean:.3f}; "
            f"pct positive {sharpe_above_zero:.3f}; sample size {rolling_sharpe_series.count()}."
        )
        with sharpe_cols[1]:
            render_insight_box("Rolling Sharpe", sharpe_stats, sharpe_default)
        
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#d62728', width=1.5),
            fillcolor='rgba(214,39,40,0.25)'
        ))
        drawdown_fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300
        )
        draw_cols = st.columns([3, 1.2])
        with draw_cols[0]:
            st.plotly_chart(drawdown_fig, use_container_width=True)
        max_dd = drawdown.min()
        latest_dd = drawdown.iloc[-1]
        dd_default = (
            f"Deepest drawdown reached {max_dd:.2f}% and the latest drawdown sits at {latest_dd:.2f}%. "
            "Lean in if you’re comfortable with that depth; stay cautious if this pain level is beyond your tolerance."
        )
        dd_stats = (
            f"Max drawdown {max_dd:.3f}%; latest {latest_dd:.3f}%; "
            f"number of drawdown days {drawdown.count()}."
        )
        with draw_cols[1]:
            render_insight_box("Drawdown", dd_stats, dd_default)
        
        st.markdown("#### Allocation & risk snapshot")
        alloc_cols = st.columns((2.2, 3, 1.6))
        with alloc_cols[0]:
            pie_fig = go.Figure(data=[go.Pie(
                labels=weights.index,
                values=weights.values * 100,
                hole=0.35
            )])
            pie_fig.update_layout(title="Current Portfolio Weights", height=350)
            st.plotly_chart(pie_fig, use_container_width=True)
        with alloc_cols[1]:
            if len(weights_history) > 1:
                stack_fig = go.Figure()
                for asset in weights_history.columns:
                    stack_fig.add_trace(go.Scatter(
                        x=weights_history.index,
                        y=weights_history[asset].values * 100,
                        mode='lines',
                        name=asset,
                        stackgroup='one'
                    ))
                stack_fig.update_layout(
                    title="Allocation Drift Over Time",
                    xaxis_title="Date",
                    yaxis_title="Weight (%)",
                    hovermode='x unified',
                    height=350
                )
                st.plotly_chart(stack_fig, use_container_width=True)
        top_weights = weights.sort_values(ascending=False).head(3)
        turnover = metrics.get("avg_turnover")
        weight_stats = (
            f"Top weights: {', '.join([f'{idx} {format_pct(val)}' for idx, val in top_weights.items()])}. "
            f"Turnover: {format_pct(turnover) if turnover is not None else 'n/a'}."
        )
        weight_default = (
            "Largest tilts sit in the top three names; turnover figures hint at how active rebalancing was. "
            "Lean in if you agree with the top convictions; stay cautious if concentration feels too high."
        )
        with alloc_cols[2]:
            render_insight_box("Allocations", weight_stats, weight_default)
        
        weights_df = pd.DataFrame({
            'Asset': weights.index,
            'Weight (%)': weights.values * 100
        }).sort_values('Weight (%)', ascending=False)
        st.dataframe(weights_df, use_container_width=True)
        
        risk_cols = st.columns(4)
        risk_cols[0].metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
        risk_cols[1].metric("CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
        if 'avg_turnover' in metrics:
            risk_cols[2].metric("Avg Turnover", f"{metrics['avg_turnover']*100:.2f}%")
        if 'weight_stability' in metrics:
            risk_cols[3].metric("Weight Stability", f"{metrics['weight_stability']:.3f}")
        
        st.markdown("#### Detailed metrics")
        metrics_df = pd.DataFrame([
            ["Annualized Return", f"{metrics['annualized_return']*100:.2f}%"],
            ["Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%"],
            ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"],
            ["Maximum Drawdown", f"{metrics['max_drawdown']*100:.2f}%"],
            ["Total Return", f"{metrics['total_return']*100:.2f}%"],
            ["VaR (95%)", f"{metrics['var_95']*100:.2f}%"],
            ["CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%"],
        ])
        if 'avg_turnover' in metrics:
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame([
                    ["Average Turnover", f"{metrics['avg_turnover']*100:.2f}%"],
                    ["Weight Stability", f"{metrics['weight_stability']:.3f}"]
                ])
            ], ignore_index=True)
        metrics_df.columns = ["Metric", "Value"]
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # --- Per-asset analysis: analyze each input company side by side ---
        st.markdown(f"#### Per-asset performance snapshots (analyzing all {len(tickers)} companies)")
        st.caption(f"Below are detailed metrics and charts for each of the {len(tickers)} companies you selected: {', '.join(tickers)}")
        per_asset_rows = []
        for ticker in tickers:
            series = returns[ticker].dropna()
            if len(series) < 2:
                continue
            # Compute simple per-asset metrics so the user can compare companies directly.
            ann_ret = series.mean() * 252
            ann_vol = series.std() * np.sqrt(252)
            sharpe_i = ann_ret / ann_vol if ann_vol > 0 else np.nan
            per_asset_rows.append(
                [
                    ticker,
                    f"{ann_ret*100:.2f}%",
                    f"{ann_vol*100:.2f}%",
                    f"{sharpe_i:.3f}" if not np.isnan(sharpe_i) else "n/a",
                ]
            )
        
        if per_asset_rows:
            per_asset_df = pd.DataFrame(
                per_asset_rows,
                columns=["Ticker", "Annualized Return", "Annualized Volatility", "Sharpe"],
            ).sort_values("Ticker")
            st.dataframe(per_asset_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"#### Per-asset cumulative return charts (all {len(tickers)} companies)")
        st.caption("Each chart below shows the historical cumulative return performance for one company. All companies are displayed side by side.")
        # Arrange individual company charts in a grid so multiple tickers show next to each other.
        display_tickers = tickers  # Show all tickers
        n_cols = min(3, max(1, len(display_tickers)))
        for i in range(0, len(display_tickers), n_cols):
            row_tickers = display_tickers[i : i + n_cols]
            cols = st.columns(len(row_tickers))
            for col_idx, ticker in enumerate(row_tickers):
                with cols[col_idx]:
                    st.markdown(f"**{ticker}**")
                    # Use cumulative returns for a comparable performance view across assets.
                    asset_ret = returns[ticker].dropna()
                    if asset_ret.empty:
                        st.caption("No sufficient data to plot.")
                        continue
                    asset_cum = (1 + asset_ret).cumprod() * 100
                    fig_asset = go.Figure()
                    fig_asset.add_trace(
                        go.Scatter(
                            x=asset_cum.index,
                            y=asset_cum.values,
                            mode="lines",
                            name=f"{ticker} cumulative",
                        )
                    )
                    fig_asset.update_layout(
                        margin=dict(l=10, r=10, t=30, b=20),
                        height=240,
                        xaxis_title=None,
                        yaxis_title="Cumulative Return (%)",
                    )
                    st.plotly_chart(fig_asset, use_container_width=True)
        
        # --- Risk model diagnostics using src/risk.py ---
        st.markdown("#### Risk engine view (covariance & correlation)")
        try:
            # Use the selected risk model to estimate the covariance matrix, then derive correlations.
            cov_matrix = get_covariance_matrix(returns, method=risk_model)
            corr_matrix = cov_matrix.copy()
            # Convert covariance to correlation for a more interpretable heatmap.
            d = np.sqrt(np.diag(cov_matrix))
            # Avoid division by zero by clipping small values.
            d = np.clip(d, 1e-8, None)
            corr_values = cov_matrix.values / np.outer(d, d)
            corr_matrix.iloc[:, :] = corr_values
            
            risk_cols2 = st.columns(2)
            with risk_cols2[0]:
                st.write(f"**Covariance model:** `{risk_model}` from `src/risk.py`")
                cov_fig = px.imshow(
                    cov_matrix,
                    x=cov_matrix.columns,
                    y=cov_matrix.index,
                    color_continuous_scale="Blues",
                    labels=dict(color="Annualized Covariance"),
                    title="Annualized Covariance Matrix",
                )
                cov_fig.update_layout(height=420)
                st.plotly_chart(cov_fig, use_container_width=True)
            with risk_cols2[1]:
                corr_fig = px.imshow(
                    corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    labels=dict(color="Correlation"),
                    title="Correlation Structure Implied by Risk Model",
                )
                corr_fig.update_layout(height=420)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            st.caption(
                "The chosen risk model transforms historical returns into a covariance and correlation matrix. "
                "These matrices drive the optimizer that produced your portfolio weights."
            )
        except Exception as e:
            st.warning(f"Could not compute risk-model diagnostics: {e}")
        
        # --- Model map and configuration summary (moved from the old Information tab) ---
        st.markdown("---")
        st.subheader("Model map")
        st.dataframe(MODEL_REFERENCE_DF, use_container_width=True, hide_index=True)
        
        st.subheader("Configuration summary")
        config_df = pd.DataFrame(
            [
                ["Portfolio Objective", opt_label_selection],
                ["Risk Engine", risk_label_selection],
                ["Risk Appetite Dial", f"{risk_aversion:.2f}"],
                ["Transaction Cost", f"{transaction_cost*100:.2f}%"],
                ["Rebalance Band", f"{rebalance_band*100:.2f}%"],
                ["Backtest Type", backtest_type],
            ]
        )
        config_df.columns = ["Parameter", "Value"]
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        st.subheader("Company snapshots")
        for ticker in tickers[:10]:
            try:
                info = get_company_info(ticker)
                with st.expander(f"📊 {ticker} - {info.get('name', ticker)}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                        if not np.isnan(info.get('market_cap', np.nan)):
                            st.write(f"**Market Cap:** ${info.get('market_cap', 0):,.0f}")
                        if not np.isnan(info.get('pe_ratio', np.nan)):
                            st.write(f"**P/E Ratio:** {info.get('pe_ratio', 0):.2f}")
                    with col2:
                        if not np.isnan(info.get('beta', np.nan)):
                            st.write(f"**Beta:** {info.get('beta', 0):.2f}")
                        if not np.isnan(info.get('dividend_yield', np.nan)):
                            st.write(f"**Dividend Yield:** {info.get('dividend_yield', 0):.2f}%")
                        if not np.isnan(info.get('52_week_high', np.nan)):
                            st.write(f"**52W High:** ${info.get('52_week_high', 0):.2f}")
                        if not np.isnan(info.get('52_week_low', np.nan)):
                            st.write(f"**52W Low:** ${info.get('52_week_low', 0):.2f}")
            except Exception as e:
                st.warning(f"Could not fetch info for {ticker}: {str(e)}")
    
    # ----------------------------------------------------------------------
    # 🔮 Prediction tab: time-series based forward-looking portfolio view.
    # ----------------------------------------------------------------------
    with predict_tab:
        st.subheader("Forecasted portfolio behavior")
        
        # Model selection: provide multiple forecasting engines with clear labels.
        forecast_method_label = st.selectbox(
            "Forecasting model",
            [
                "Ensemble (ARIMA + Exponential Smoothing + Moving Average)",
                "ARIMA (Auto-Regressive Integrated Moving Average)",
                "Exponential Smoothing (Holt-Winters)",
                "Moving Average (Baseline)",
            ],
            help=(
                "Different models capture different patterns: ARIMA models trends and autocorrelations, "
                "Exponential Smoothing adapts to recent changes, Ensemble combines multiple models for robustness. "
                "If advanced models fail, the app will fall back to Moving Average."
            ),
        )
        method_map = {
            "Ensemble (ARIMA + Exponential Smoothing + Moving Average)": "ensemble",
            "ARIMA (Auto-Regressive Integrated Moving Average)": "arima",
            "Exponential Smoothing (Holt-Winters)": "exponential_smoothing",
            "Moving Average (Baseline)": "ma",
        }
        internal_forecast_method = method_map[forecast_method_label]
        
        forecast_horizon = st.slider(
            "Forecast horizon (days)",
            min_value=5,
            max_value=90,
            value=30,
            step=5,
            help="Number of future trading days to project based on the chosen forecasting model.",
        )
        
        # Run the forecasting step using the historical portfolio returns produced by the backtest.
        forecast_series = None
        forecast_model_used = internal_forecast_method
        fallback_used = False
        try:
            if internal_forecast_method == "ensemble":
                # Use an ensemble of ARIMA, exponential smoothing, and moving average.
                ensemble_result = ensemble_forecast(
                    portfolio_returns,
                    methods=["arima", "exponential_smoothing", "ma"],
                    forecast_horizon=forecast_horizon,
                )
                forecast_series = ensemble_result["ensemble_forecast"]
                forecast_model_used = "ensemble (ARIMA + Exponential Smoothing + MA)"
            else:
                # Use a single forecasting method (ARIMA, exponential smoothing, or moving average).
                result = forecast_portfolio_returns(
                    portfolio_returns,
                    method=internal_forecast_method,
                    forecast_horizon=forecast_horizon,
                )
                forecast_series = result["forecast"]
        except Exception as e:
            # If advanced models fail (e.g., statsmodels not installed), fall back to a simple MA forecast.
            st.warning(
                f"Primary forecasting method '{forecast_method_label}' failed ({str(e)}). "
                "Falling back to Moving Average baseline."
            )
            try:
                result = forecast_portfolio_returns(
                    portfolio_returns,
                    method="ma",
                    forecast_horizon=forecast_horizon,
                )
                forecast_series = result["forecast"]
                forecast_model_used = "moving average (fallback)"
                fallback_used = True
            except Exception as e2:
                st.error(f"Forecasting failed even with fallback method: {e2}")
        
        if forecast_series is not None:
            # Build a cumulative performance view combining recent history and the projected path.
            history_window = min(len(portfolio_returns), 252)
            recent_hist = portfolio_returns.iloc[-history_window:]
            cumulative_hist = (1 + recent_hist).cumprod()
            
            cumulative_forecast = (1 + forecast_series).cumprod()
            # Start forecast from the last historical cumulative value to ensure continuity.
            cumulative_forecast = cumulative_forecast * cumulative_hist.iloc[-1]
            
            forecast_fig = go.Figure()
            forecast_fig.add_trace(
                go.Scatter(
                    x=cumulative_hist.index,
                    y=cumulative_hist.values * 100,
                    mode="lines",
                    name="Historical (recent)",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            forecast_fig.add_trace(
                go.Scatter(
                    x=cumulative_forecast.index,
                    y=cumulative_forecast.values * 100,
                    mode="lines",
                    name=f"Forecast (next {forecast_horizon} days)",
                    line=dict(color="#d62728", width=2, dash="dash"),
                )
            )
            forecast_fig.update_layout(
                title="Portfolio cumulative return: history vs forecast",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode="x unified",
                height=420,
            )
            
            col_f1, col_f2 = st.columns([3, 1.2])
            with col_f1:
                st.plotly_chart(forecast_fig, use_container_width=True)
                # Display which model was actually used (important for transparency).
                model_status = "⚠️ Fallback" if fallback_used else "✅ Active"
                st.caption(
                    f"{model_status} **Model used:** {forecast_model_used}. "
                    "The forecast line shows projected cumulative returns based on historical patterns."
                )
            with col_f2:
                # Summarize the forecast in intuitive statistics.
                mean_ret = forecast_series.mean()
                std_ret = forecast_series.std()
                cum_ret = (1 + forecast_series).prod() - 1
                st.markdown("##### Forecast summary")
                st.write(f"- **Model**: {forecast_model_used}")
                st.write(f"- **Mean daily return**: {format_pct(mean_ret, 3)}")
                st.write(f"- **Daily volatility**: {format_pct(std_ret, 3)}")
                st.write(f"- **{forecast_horizon}-day cumulative**: {format_pct(cum_ret, 2)}")
                if fallback_used:
                    st.caption(
                        "💡 Install `statsmodels` (pip install statsmodels) to enable ARIMA and Exponential Smoothing models."
                    )
            
            st.markdown("---")
        
        # ------------------------------------------------------------------
        # Per-asset forecasts: one small chart per ticker, displayed side by side.
        # ------------------------------------------------------------------
        st.subheader(f"Per-asset forecasts for all {len(tickers)} companies")
        st.caption(
            f"Each panel forecasts a single company's returns using the same model as the portfolio forecast above. "
            f"All {len(tickers)} companies are shown below."
        )
        
        # Use the same forecasting method as selected for the portfolio, but fall back gracefully per asset.
        n_cols_assets = min(3, max(1, len(tickers)))
        for i in range(0, len(tickers), n_cols_assets):
            row_tickers = tickers[i : i + n_cols_assets]
            cols = st.columns(len(row_tickers))
            for col_idx, ticker in enumerate(row_tickers):
                with cols[col_idx]:
                    st.markdown(f"**{ticker}**")
                    asset_series = returns[ticker].dropna()
                    if len(asset_series) < 10:
                        st.caption("Not enough data to forecast (need at least 10 days).")
                        continue
                    try:
                        # Try the same method as portfolio, but fall back to MA if it fails for this asset.
                        asset_forecast = None
                        asset_model_used = internal_forecast_method
                        if internal_forecast_method == "ensemble":
                            try:
                                asset_ensemble = ensemble_forecast(
                                    asset_series,
                                    methods=["arima", "exponential_smoothing", "ma"],
                                    forecast_horizon=forecast_horizon,
                                )
                                asset_forecast = asset_ensemble["ensemble_forecast"]
                                asset_model_used = "ensemble"
                            except:
                                asset_result = forecast_portfolio_returns(
                                    asset_series,
                                    method="ma",
                                    forecast_horizon=forecast_horizon,
                                )
                                asset_forecast = asset_result["forecast"]
                                asset_model_used = "ma (fallback)"
                        else:
                            try:
                                asset_result = forecast_portfolio_returns(
                                    asset_series,
                                    method=internal_forecast_method,
                                    forecast_horizon=forecast_horizon,
                                )
                                asset_forecast = asset_result["forecast"]
                                asset_model_used = internal_forecast_method
                            except:
                                # Fallback to MA if the selected method fails for this asset.
                                asset_result = forecast_portfolio_returns(
                                    asset_series,
                                    method="ma",
                                    forecast_horizon=forecast_horizon,
                                )
                                asset_forecast = asset_result["forecast"]
                                asset_model_used = "ma (fallback)"
                    except Exception as e:
                        st.caption(f"Forecast error: {str(e)[:50]}")
                        continue
                    
                    hist_window = min(len(asset_series), 252)
                    recent_asset = asset_series.iloc[-hist_window:]
                    cum_hist_asset = (1 + recent_asset).cumprod()
                    cum_forecast_asset = (1 + asset_forecast).cumprod()
                    cum_forecast_asset = cum_forecast_asset * cum_hist_asset.iloc[-1]
                    
                    fig_asset_f = go.Figure()
                    fig_asset_f.add_trace(
                        go.Scatter(
                            x=cum_hist_asset.index,
                            y=cum_hist_asset.values * 100,
                            mode="lines",
                            name="Historical",
                        )
                    )
                    fig_asset_f.add_trace(
                        go.Scatter(
                            x=cum_forecast_asset.index,
                            y=cum_forecast_asset.values * 100,
                            mode="lines",
                            name="Forecast",
                            line=dict(dash="dash"),
                        )
                    )
                    fig_asset_f.update_layout(
                        margin=dict(l=10, r=10, t=30, b=20),
                        height=260,
                        xaxis_title=None,
                        yaxis_title="Cumulative Return (%)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_asset_f, use_container_width=True)
                    # Show which model was used for this asset's forecast.
                    st.caption(f"Model: {asset_model_used}")
        
        # Keep the original risk-focused forward-looking diagnostics to complement the forecasts.
        st.subheader("Forward-looking diagnostics")
        st.info(
            "These diagnostics translate the chosen risk engine into volatility and tail-risk insights. "
            "Use them to understand how today’s configuration might behave tomorrow."
        )
        
        rolling_vol = rolling_volatility(portfolio_returns, window=252)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,
            mode='lines',
            name='Rolling Volatility (252 days)',
            line=dict(color='#6a1b9a', width=2)
        ))
        vol_fig.update_layout(
            title="Volatility Outlook",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=320
        )
        vol_cols = st.columns([3, 1.2])
        with vol_cols[0]:
            st.plotly_chart(vol_fig, use_container_width=True)
        latest_vol = rolling_vol.dropna().iloc[-1] if not rolling_vol.dropna().empty else 0.0
        mean_vol = rolling_vol.mean()
        vol_default = (
            f"Recent volatility sits near {format_pct(latest_vol, 2)} vs a {format_pct(mean_vol, 2)} average. "
            "Lean in if you can stomach swings at this level; stay cautious if you need smoother rides."
        )
        vol_stats = (
            f"Latest rolling volatility {latest_vol:.4f}; mean {mean_vol:.4f}; "
            f"window length {rolling_vol.count()}."
        )
        with vol_cols[1]:
            render_insight_box("Volatility Outlook", vol_stats, vol_default)
        
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=portfolio_returns.values * 100,
            nbinsx=50,
            name='Portfolio Returns',
            marker_color='#1f77b4'
        ))
        hist_fig.add_vline(
            x=metrics['var_95'] * 100,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="VaR (95%)"
        )
        hist_fig.update_layout(
            title="Distribution Of Returns",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=320
        )
        dist_cols = st.columns([3, 1.2])
        with dist_cols[0]:
            st.plotly_chart(hist_fig, use_container_width=True)
        returns_np = portfolio_returns.dropna()
        skew_val = returns_np.skew()
        kurt_val = returns_np.kurt()
        var_val = metrics.get("var_95", 0)
        cvar_val = metrics.get("cvar_95", 0)
        dist_default = (
            f"Typical daily move is ~{format_pct(returns_np.std(), 2)}; "
            f"VaR 95 is {format_pct(var_val, 2)}, CVaR 95 is {format_pct(cvar_val, 2)}. "
            "Lean in if these tail-loss levels fit your risk budget; stay cautious if the downside feels steep."
        )
        dist_stats = (
            f"Stdev {returns_np.std():.4f}; skew {skew_val:.3f}; kurtosis {kurt_val:.3f}; "
            f"VaR {var_val:.4f}; CVaR {cvar_val:.4f}."
        )
        with dist_cols[1]:
            render_insight_box("Return Distribution", dist_stats, dist_default)
        
        if risk_model == "garch":
            st.success("Because you selected a volatility-forecasting engine, emphasis is placed on recent volatility spikes and decay.")
        elif risk_model == "glasso":
            st.success("Sparse dependency modeling highlights clusters of assets that move together—great for stress testing correlation breakdowns.")
        elif risk_model == "ledoit_wolf":
            st.success("Shrinkage keeps the covariance matrix stable, so the predicted volatility path is smoother than raw history.")
        else:
            st.success("Sample covariance reflects pure historical co-movement. Combine with walk-forward testing to validate robustness.")

else:
    # Initial state - show instructions
    st.info("👈 Please configure your portfolio in the sidebar and click 'Run Analysis' to begin.")
    
    st.markdown("""
    ### Welcome to FinLove Portfolio Construction Dashboard!
    
    This dashboard allows you to:
    
    1. **Input Companies**: Enter ticker symbols (e.g., AAPL, MSFT, GOOGL) or use default sector ETFs
    2. **Select Methods**: Choose from various optimization methods (Markowitz, Black-Litterman, CVaR, etc.)
    3. **Configure Risk**: Select risk models (Ledoit-Wolf, GLASSO, GARCH)
    4. **Run Backtests**: Simple one-time optimization or walk-forward backtesting
    5. **Analyze Results**: View performance metrics, portfolio weights, risk analysis, and more
    
    ### Features:
    - 📊 **Performance Visualization**: Cumulative returns, rolling Sharpe, drawdown charts
    - 💰 **Portfolio Allocation**: Weight distribution and evolution over time
    - 📈 **Risk Analysis**: VaR, CVaR, volatility analysis
    - 🔍 **Company Information**: Detailed company data from Yahoo Finance
    - 📋 **Comprehensive Metrics**: All performance and risk metrics
    
    Start by selecting your assets and configuration in the sidebar!
    """)
