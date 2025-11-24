# FinLove Dashboard - User Guide

## Overview

The FinLove Portfolio Construction Dashboard is an interactive web application that allows you to:
- Input company tickers or use default sector ETFs
- Construct optimized portfolios using various methods
- Analyze performance and risk metrics
- Visualize portfolio allocation and returns

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage

### Step 1: Select Assets

**Option A: Company Tickers**
- Select "Company Ticker" in the sidebar
- Enter one or more ticker symbols separated by commas (e.g., `AAPL,MSFT,GOOGL`)
- Examples: `AAPL`, `MSFT,GOOGL,AMZN`, `TSLA`

**Option B: Sector ETFs (Default)**
- Select "Sector ETFs (Default)" to use the default 11 sector ETFs:
  - XLK (Technology), XLF (Financials), XLV (Healthcare), XLY (Consumer Discretionary),
  - XLP (Consumer Staples), XLE (Energy), XLI (Industrials), XLB (Materials),
  - XLU (Utilities), XLRE (Real Estate), XLC (Communication Services)

### Step 2: Configure Date Range

- Select start and end dates for the analysis period
- Recommended: At least 2-3 years of data for reliable results

### Step 3: Choose Optimization Method

- **markowitz**: Mean-variance optimization (maximize return - risk penalty)
- **min_variance**: Minimize portfolio variance
- **sharpe**: Maximize Sharpe ratio
- **black_litterman**: Black-Litterman with market equilibrium and views
- **cvar**: Minimize Conditional Value at Risk

### Step 4: Select Risk Model

- **ledoit_wolf**: Ledoit-Wolf shrinkage covariance (recommended for stability)
- **sample**: Sample covariance matrix
- **glasso**: Graphical LASSO (sparse precision matrix)
- **garch**: GARCH-based time-varying volatility

### Step 5: Set Parameters

- **Risk Aversion**: Higher values = more risk averse (for Markowitz)
- **Transaction Cost**: Proportional cost per rebalancing (e.g., 0.1% = 0.001)
- **Rebalance Band**: Maximum weight drift before rebalancing (e.g., 5% = 0.05)

### Step 6: Choose Backtest Type

- **Simple**: One-time optimization using all historical data
- **Walk-Forward**: Rolling window backtest (more realistic)

### Step 7: Run Analysis

Click the "🚀 Run Analysis" button and wait for results.

## Understanding the Results

### Performance Tab
- **Cumulative Returns**: Portfolio performance over time vs. equal-weight benchmark
- **Rolling Sharpe**: 252-day rolling Sharpe ratio
- **Drawdown Chart**: Portfolio drawdowns over time

### Portfolio Weights Tab
- **Pie Chart**: Current portfolio allocation
- **Weights Over Time**: How allocation changes with rebalancing
- **Weights Table**: Detailed weight breakdown

### Risk Analysis Tab
- **VaR/CVaR**: Value at Risk and Conditional VaR at 95% confidence
- **Rolling Volatility**: 252-day rolling volatility
- **Returns Distribution**: Histogram of daily returns

### Company Info Tab
- Detailed information for each ticker (sector, market cap, P/E ratio, etc.)

### Detailed Metrics Tab
- Comprehensive performance metrics
- Configuration summary

## Tips

1. **Start Simple**: Begin with default ETFs and simple Markowitz optimization
2. **Sufficient Data**: Ensure you have at least 1-2 years of data for each ticker
3. **Risk Models**: Ledoit-Wolf is recommended for most cases (more stable than sample covariance)
4. **Transaction Costs**: Include realistic transaction costs (0.1-0.5% for stocks)
5. **Walk-Forward**: Use walk-forward backtesting for more realistic performance estimates

## Troubleshooting

### "Insufficient data" Error
- Check that tickers are valid and have data for the selected date range
- Try a different date range or different tickers

### "No valid data after cleaning" Error
- Some tickers may have too many missing values
- Try removing problematic tickers or using a shorter date range

### Slow Performance
- Reduce the number of tickers
- Use shorter date ranges
- Use simpler risk models (sample or ledoit_wolf)

## Technical Details

### Portfolio Construction Methods

**Markowitz Mean-Variance**
- Maximizes: μ'w - (λ/2) * w'Σw
- Where μ = expected returns, Σ = covariance, λ = risk aversion

**Minimum Variance**
- Minimizes: w'Σw
- Subject to budget and optional constraints

**Sharpe Maximization**
- Maximizes: (μ'w - rf) / sqrt(w'Σw)
- Where rf = risk-free rate

**Black-Litterman**
- Combines market equilibrium returns with investor views
- More stable than pure Markowitz

**CVaR Optimization**
- Minimizes Conditional Value at Risk
- Focuses on tail risk

### Risk Models

**Ledoit-Wolf Shrinkage**
- Shrinks sample covariance towards target
- Reduces estimation error

**Graphical LASSO**
- Estimates sparse precision matrix
- Useful for high-dimensional portfolios

**GARCH**
- Time-varying volatility per asset
- Captures volatility clustering

## Support

For issues or questions, please refer to the main project README or open an issue on GitHub.

