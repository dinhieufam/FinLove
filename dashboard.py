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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 FinLove Portfolio Construction Dashboard</div>', unsafe_allow_html=True)
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
st.sidebar.subheader("3. Optimization Method")
optimization_method = st.sidebar.selectbox(
    "Method",
    ["markowitz", "min_variance", "sharpe", "black_litterman", "cvar"],
    help="Select portfolio optimization method"
)

# Risk model
st.sidebar.subheader("4. Risk Model")
risk_model = st.sidebar.selectbox(
    "Covariance Estimation",
    ["ledoit_wolf", "sample", "glasso", "garch"],
    help="Select covariance estimation method"
)

# Additional parameters
st.sidebar.subheader("5. Parameters")
risk_aversion = st.sidebar.slider(
    "Risk Aversion",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Higher values = more risk averse (for Markowitz)"
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Proportional transaction cost"
) / 100

rebalance_band = st.sidebar.slider(
    "Rebalance Band (%)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Maximum weight drift before rebalancing"
) / 100

# Backtest type
st.sidebar.subheader("6. Backtest Type")
backtest_type = st.sidebar.radio(
    "Backtest Method",
    ["Simple (One-time optimization)", "Walk-Forward (Rolling window)"]
)

if backtest_type == "Walk-Forward (Rolling window)":
    train_window = st.sidebar.slider(
        "Training Window (months)",
        min_value=12,
        max_value=60,
        value=36,
        step=6
    )
    test_window = st.sidebar.slider(
        "Test Window (months)",
        min_value=1,
        max_value=12,
        value=1,
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
    
    # Header with summary
    st.success(f"✅ Analysis complete for {len(tickers)} assets: {', '.join(tickers)}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annualized Return",
            f"{metrics['annualized_return']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Annualized Volatility",
            f"{metrics['annualized_volatility']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']*100:.2f}%"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance", "💰 Portfolio Weights", "📈 Risk Analysis", "🔍 Company Info", "📋 Detailed Metrics"
    ])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        cumulative_benchmark = (1 + returns.mean(axis=1)).cumprod()  # Equal-weight benchmark
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=cumulative_benchmark.index,
            y=cumulative_benchmark.values * 100,
            mode='lines',
            name='Equal-Weight Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling Sharpe
        rolling_sharpe_series = rolling_sharpe(portfolio_returns, window=252)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rolling_sharpe_series.index,
            y=rolling_sharpe_series.values,
            mode='lines',
            name='Rolling Sharpe (252 days)',
            line=dict(color='green', width=2)
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            title="Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Drawdown chart
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        fig3.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Portfolio Allocation")
        
        # Current weights pie chart
        fig = go.Figure(data=[go.Pie(
            labels=weights.index,
            values=weights.values * 100,
            hole=0.3
        )])
        fig.update_layout(
            title="Current Portfolio Weights",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weights over time
        if len(weights_history) > 1:
            fig2 = go.Figure()
            for asset in weights_history.columns:
                fig2.add_trace(go.Scatter(
                    x=weights_history.index,
                    y=weights_history[asset].values * 100,
                    mode='lines',
                    name=asset,
                    stackgroup='one'
                ))
            fig2.update_layout(
                title="Portfolio Weights Over Time",
                xaxis_title="Date",
                yaxis_title="Weight (%)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Weights table
        st.subheader("Current Weights")
        weights_df = pd.DataFrame({
            'Asset': weights.index,
            'Weight (%)': weights.values * 100
        }).sort_values('Weight (%)', ascending=False)
        st.dataframe(weights_df, use_container_width=True)
    
    with tab3:
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
            st.metric("CVaR (95%)", f"{metrics['cvar_95']*100:.2f}%")
        
        with col2:
            if 'avg_turnover' in metrics:
                st.metric("Avg Turnover", f"{metrics['avg_turnover']*100:.2f}%")
            if 'weight_stability' in metrics:
                st.metric("Weight Stability", f"{metrics['weight_stability']:.3f}")
        
        # Rolling volatility
        rolling_vol = rolling_volatility(portfolio_returns, window=252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values * 100,
            mode='lines',
            name='Rolling Volatility (252 days)',
            line=dict(color='purple', width=2)
        ))
        fig.update_layout(
            title="Rolling Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=portfolio_returns.values * 100,
            nbinsx=50,
            name='Portfolio Returns',
            marker_color='blue'
        ))
        fig2.add_vline(
            x=metrics['var_95'] * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="VaR (95%)"
        )
        fig2.update_layout(
            title="Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.subheader("Company Information")
        
        # Display info for each ticker
        for ticker in tickers[:10]:  # Limit to first 10 to avoid too many API calls
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
    
    with tab5:
        st.subheader("Detailed Performance Metrics")
        
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
        
        # Configuration summary
        st.subheader("Configuration Summary")
        config_df = pd.DataFrame([
            ["Optimization Method", optimization_method],
            ["Risk Model", risk_model],
            ["Risk Aversion", f"{risk_aversion:.2f}"],
            ["Transaction Cost", f"{transaction_cost*100:.2f}%"],
            ["Rebalance Band", f"{rebalance_band*100:.2f}%"],
            ["Backtest Type", backtest_type],
        ])
        config_df.columns = ["Parameter", "Value"]
        st.dataframe(config_df, use_container_width=True, hide_index=True)

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

