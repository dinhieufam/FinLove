# FinLove — Risk-Aware Portfolio Construction

**Authors:** Nguyen Van Duy Anh · Pham Dinh Hieu · Cao Pham Minh Dang · Tran Anh Chuong · Ngo Dinh Khanh  
**GitHub:** https://github.com/dinhieufam/FinLove

---

## Overview

FinLove is a Python-based portfolio construction engine that combines advanced risk models, optimization methods, and execution realism to create stable, risk-aware portfolios. The project includes an interactive Streamlit dashboard for portfolio analysis and visualization.

**Key Features:**
- Multiple risk models (Ledoit-Wolf, GLASSO, GARCH, DCC)
- Various optimization methods (Markowitz, Black-Litterman, CVaR, Minimum Variance, Sharpe)
- Realistic backtesting with transaction costs and rebalance bands
- Interactive dashboard with comprehensive visualizations
- Automatic data caching for faster performance

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Pre-download Data

For faster performance, pre-download datasets:

```bash
python download_data.py
```

See [DATA.md](DATA.md) for detailed information about data download and caching.

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

Or use the convenience script:

```bash
./run_dashboard.sh
```

The dashboard will open in your default web browser at `http://localhost:8501`

---

## Quick Start

1. **Start the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

2. **Select assets:**
   - Enter company tickers (e.g., `AAPL,MSFT,GOOGL`)
   - Or use default sector ETFs

3. **Configure settings:**
   - Choose optimization method (Markowitz, Sharpe, CVaR, etc.)
   - Select risk model (Ledoit-Wolf recommended)
   - Set parameters (risk aversion, transaction costs, etc.)

4. **Run analysis:**
   - Click "🚀 Run Analysis"
   - Explore results in the interactive tabs

---

## Project Structure

```
FinLove/
├── src/
│   ├── __init__.py
│   ├── data.py          # Data acquisition & features
│   ├── risk.py          # Risk models (LW, GLASSO, GARCH, DCC)
│   ├── optimize.py      # Optimization methods (MV, BL, CVaR)
│   ├── backtest.py      # Backtesting engine
│   └── metrics.py       # Performance metrics
├── dashboard.py          # Streamlit dashboard
├── download_data.py      # Data pre-download script
├── requirements.txt      # Dependencies
├── run_dashboard.sh     # Run script
├── README.md            # This file
└── DATA.md              # Data download and caching guide
```

---

## Features

### Risk Models

- **Ledoit-Wolf Shrinkage**: Reduces estimation error by shrinking sample covariance
- **Graphical LASSO (GLASSO)**: Estimates sparse precision matrix
- **GARCH(1,1)**: Time-varying volatility per asset
- **DCC**: Dynamic Conditional Correlation approximation

### Optimization Methods

- **Markowitz Mean-Variance**: Maximizes return - risk penalty
- **Minimum Variance**: Minimizes portfolio variance
- **Sharpe Maximization**: Maximizes risk-adjusted returns
- **Black-Litterman**: Combines market equilibrium with investor views
- **CVaR Optimization**: Minimizes Conditional Value at Risk

### Backtesting

- **Simple Backtest**: One-time optimization using all historical data
- **Walk-Forward Backtest**: Rolling window backtest (more realistic)
- **Transaction Costs**: Proportional costs per rebalancing
- **Rebalance Bands**: Drift-based rebalancing to reduce turnover

### Dashboard Features

- **Interactive Interface**: Easy-to-use Streamlit web app
- **Company Input**: Type in company names or tickers
- **Comprehensive Visualizations**:
  - Cumulative returns vs. benchmark
  - Rolling Sharpe ratio
  - Drawdown charts
  - Portfolio weights (pie chart and time series)
  - Risk analysis (VaR, CVaR, volatility)
- **Performance Metrics**: All key metrics in organized tabs
- **Company Information**: Detailed data for each ticker

---

## Usage Guide

### Step 1: Select Assets

**Option A: Company Tickers**
- Select "Company Ticker" in the sidebar
- Enter one or more ticker symbols separated by commas
- Examples: `AAPL`, `MSFT,GOOGL,AMZN`, `TSLA`

**Option B: Sector ETFs (Default)**
- Select "Sector ETFs (Default)" to use 11 default sector ETFs:
  - XLK (Technology), XLF (Financials), XLV (Healthcare), XLY (Consumer Discretionary),
  - XLP (Consumer Staples), XLE (Energy), XLI (Industrials), XLB (Materials),
  - XLU (Utilities), XLRE (Real Estate), XLC (Communication Services)

### Step 2: Configure Date Range

- Select start and end dates for the analysis period
- Recommended: At least 2-3 years of data for reliable results

### Step 3: Choose Optimization Method

- **markowitz**: Mean-variance optimization
- **min_variance**: Minimize portfolio variance
- **sharpe**: Maximize Sharpe ratio
- **black_litterman**: Black-Litterman with market equilibrium
- **cvar**: Minimize Conditional Value at Risk

### Step 4: Select Risk Model

- **ledoit_wolf**: Recommended for stability
- **sample**: Sample covariance matrix
- **glasso**: Graphical LASSO
- **garch**: GARCH-based time-varying volatility

### Step 5: Set Parameters

- **Risk Aversion**: Higher values = more risk averse
- **Transaction Cost**: Proportional cost per rebalancing (e.g., 0.1%)
- **Rebalance Band**: Maximum weight drift before rebalancing (e.g., 5%)

### Step 6: Choose Backtest Type

- **Simple**: One-time optimization using all historical data
- **Walk-Forward**: Rolling window backtest (more realistic)

### Step 7: Run Analysis

Click the "🚀 Run Analysis" button and explore results in the tabs.

---

## Understanding Results

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

---

## Tips

1. **Pre-download Data**: Run `download_data.py` first for faster dashboard performance
2. **Start Simple**: Begin with default ETFs and simple Markowitz optimization
3. **Sufficient Data**: Ensure you have at least 1-2 years of data for each ticker
4. **Risk Models**: Ledoit-Wolf is recommended for most cases (more stable than sample covariance)
5. **Transaction Costs**: Include realistic transaction costs (0.1-0.5% for stocks)
6. **Walk-Forward**: Use walk-forward backtesting for more realistic performance estimates

---

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
- Pre-download data using `download_data.py`

---

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

---

## Dependencies

All dependencies are listed in `requirements.txt`:
- Core: numpy, pandas, scipy
- Optimization: cvxpy, scikit-learn
- Risk models: arch (for GARCH)
- Visualization: matplotlib, seaborn, plotly
- Dashboard: streamlit
- Data: yfinance

---

## Data

- **Source**: Yahoo Finance via `yfinance`
- **Default Universe**: 11 liquid Sector ETFs
- **Data Types**: Historical prices (OHLCV), company information
- **Frequency**: Daily data
- **Caching**: Automatic 24-hour cache for faster performance

See [DATA.md](DATA.md) for detailed information about data download, caching, and management.

---

## Project Status

✅ **Complete** - All features implemented and tested

The project includes:
- ✅ All risk models (Ledoit-Wolf, GLASSO, GARCH, DCC)
- ✅ All optimization methods (Markowitz, Black-Litterman, CVaR, etc.)
- ✅ Backtesting engine with transaction costs and rebalance bands
- ✅ Interactive dashboard with comprehensive visualizations
- ✅ Data caching system for performance
- ✅ Comprehensive documentation

---

## License

See [LICENSE](LICENSE) file for details.

---

## Support

For issues or questions, please refer to:
- [DATA.md](DATA.md) for data-related questions
- GitHub issues for bug reports
- Project repository: https://github.com/dinhieufam/FinLove
