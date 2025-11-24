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

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from src.data import (
    download_data, get_returns, get_company_info,
    prepare_portfolio_data, compute_features,
    get_cache_info, clear_cache
)
from src.risk import get_covariance_matrix
from src.optimize import optimize_portfolio
from src.backtest import walk_forward_backtest, simple_backtest
from src.metrics import (
    calculate_all_metrics, rolling_sharpe, rolling_volatility,
    maximum_drawdown, value_at_risk, conditional_value_at_risk
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

# Additional parameters
st.sidebar.subheader("5. Risk Appetite")
risk_aversion = st.sidebar.slider(
    "Dial down risk-taking",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Higher values make the optimizer far more defensive."
)

# Backtest type
st.sidebar.subheader("6. Testing Style")
backtest_type = st.sidebar.radio(
    "Choose how to test the portfolio",
    ["Simple (One-time optimization)", "Walk-Forward (Rolling window)"]
)

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
    
    analyze_tab, info_tab, predict_tab = st.tabs(["🔍 Analyze", "ℹ️ Information", "🔮 Prediction"])
    
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
        st.plotly_chart(perf_fig, use_container_width=True)
        
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
        st.plotly_chart(sharpe_fig, use_container_width=True)
        
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
        st.plotly_chart(drawdown_fig, use_container_width=True)
        
        st.markdown("#### Allocation & risk snapshot")
        alloc_cols = st.columns((2, 3))
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
    
    with info_tab:
        st.subheader("Model map")
        st.dataframe(MODEL_REFERENCE_DF, use_container_width=True, hide_index=True)
        
        st.subheader("Configuration summary")
        config_df = pd.DataFrame([
            ["Portfolio Objective", opt_label_selection],
            ["Risk Engine", risk_label_selection],
            ["Risk Appetite Dial", f"{risk_aversion:.2f}"],
            ["Transaction Cost", f"{transaction_cost*100:.2f}%"],
            ["Rebalance Band", f"{rebalance_band*100:.2f}%"],
            ["Backtest Type", backtest_type],
        ])
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
    
    with predict_tab:
        st.subheader("Forward-looking diagnostics")
        st.info("These diagnostics translate the chosen risk engine into volatility and tail-risk insights. Use them to understand how today’s configuration might behave tomorrow.")
        
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
        st.plotly_chart(vol_fig, use_container_width=True)
        
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
        st.plotly_chart(hist_fig, use_container_width=True)
        
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

