# Quick Start Guide - How to Run FinLove

This guide shows you all the ways to run and use the FinLove portfolio construction system.

## 🚀 Method 1: Run the Interactive Dashboard (Recommended)

The easiest way to use FinLove is through the interactive Streamlit dashboard.

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Dashboard

**Option A: Using the script (macOS/Linux)**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Option B: Direct command**
```bash
streamlit run dashboard.py
```

### Step 3: Use the Dashboard

1. The dashboard will open in your browser at `http://localhost:8501`
2. In the sidebar:
   - Select assets (tickers like `AAPL,MSFT,GOOGL` or use default ETFs)
   - Choose date range
   - Select optimization method (Markowitz, Sharpe, CVaR, etc.)
   - Select risk model (Ledoit-Wolf recommended)
   - Set risk aversion and other parameters
3. Click **"🚀 Run Analysis"**
4. Explore results in the tabs:
   - **Analyze**: Performance charts and metrics
   - **Information**: Model details and company info
   - **Prediction**: Forward-looking diagnostics

---

## 🔮 Method 2: Run Prediction Pipeline

Run the complete prediction pipeline that tests all models and forecasts future returns.

### Basic Usage

```bash
python example_prediction.py
```

This will:
1. Test all 20 model combinations (5 optimization × 4 risk models)
2. Select top 5 models by Sharpe ratio
3. Forecast next 30 days of returns
4. Display results and create visualizations

### Customize the Prediction

Edit `example_prediction.py` or create your own script:

```python
from src.predict import predict_future_performance

results = predict_future_performance(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date="2015-01-01",
    end_date="2024-01-01",
    forecast_horizon=30,  # Forecast next 30 days
    forecast_method='ensemble',  # Use ensemble forecasting
    use_top_models=5  # Use top 5 models
)

# Access results
print(results['aggregated_prediction'])  # Forecasted returns
print(results['top_models'])  # Best models
```

---

## 💻 Method 3: Use Modules Programmatically

### Example 1: Simple Portfolio Optimization

```python
from src.data import prepare_portfolio_data
from src.risk import get_covariance_matrix
from src.optimize import optimize_portfolio
from src.metrics import calculate_all_metrics

# Get data
tickers = ['AAPL', 'MSFT', 'GOOGL']
returns, prices = prepare_portfolio_data(tickers, start_date="2020-01-01")

# Estimate covariance
covariance = get_covariance_matrix(returns, method='ledoit_wolf')

# Optimize portfolio
weights = optimize_portfolio(
    returns,
    covariance,
    method='markowitz',
    constraints={'long_only': True},
    risk_aversion=1.0
)

print("Optimal weights:")
print(weights)
```

### Example 2: Run Backtest

```python
from src.backtest import walk_forward_backtest
from src.data import prepare_portfolio_data

# Get data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
returns, prices = prepare_portfolio_data(tickers, start_date="2015-01-01")

# Run walk-forward backtest
portfolio_returns, weights_history, metrics = walk_forward_backtest(
    returns,
    train_window=36,  # 36 months training
    test_window=1,    # 1 month testing
    optimization_method='markowitz',
    risk_model='ledoit_wolf',
    transaction_cost=0.001,  # 0.1% transaction cost
    rebalance_band=0.05  # 5% rebalance band
)

print("Performance Metrics:")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
print(f"Annualized Volatility: {metrics['annualized_volatility']*100:.2f}%")
```

### Example 3: Collect All Model Results

```python
from src.model_collector import collect_all_model_results, get_best_models

# Collect results from all 20 model combinations
results = collect_all_model_results(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date="2015-01-01",
    end_date="2024-01-01",
    backtest_type='walk_forward',
    train_window=36
)

# Find best models
best_by_sharpe = get_best_models(results, metric='sharpe_ratio', top_n=5)
best_by_return = get_best_models(results, metric='annualized_return', top_n=5)

print("Top 5 by Sharpe Ratio:")
print(best_by_sharpe[['model_id', 'sharpe_ratio', 'annualized_return']])
```

### Example 4: Forecast Future Returns

```python
from src.forecast import forecast_portfolio_returns, ensemble_forecast
from src.data import prepare_portfolio_data
from src.backtest import simple_backtest

# Get historical portfolio returns
tickers = ['AAPL', 'MSFT', 'GOOGL']
returns, prices = prepare_portfolio_data(tickers, start_date="2015-01-01")

# Run backtest to get portfolio returns
portfolio_returns, weights, metrics = simple_backtest(
    returns,
    optimization_method='markowitz',
    risk_model='ledoit_wolf'
)

# Forecast future returns
forecast = forecast_portfolio_returns(
    portfolio_returns,
    method='ensemble',  # Use ensemble of ARIMA, Prophet, MA
    forecast_horizon=30  # Next 30 days
)

print("Forecasted returns:")
print(forecast['forecast'])
```

---

## 📊 Method 4: Pre-download Data (Faster Performance)

For faster dashboard performance, pre-download data:

```bash
python download_data.py
```

This will download data for common tickers and save them in the `Dataset/` directory.

---

## 🐍 Python Script Examples

### Create Your Own Script

Create a file `my_analysis.py`:

```python
#!/usr/bin/env python3
"""
My custom portfolio analysis script.
"""

from src.data import prepare_portfolio_data
from src.backtest import walk_forward_backtest
from src.predict import predict_future_performance

def main():
    # Define your portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Run backtest
    returns, prices = prepare_portfolio_data(tickers, start_date="2020-01-01")
    
    portfolio_returns, weights_history, metrics = walk_forward_backtest(
        returns,
        optimization_method='markowitz',
        risk_model='ledoit_wolf'
    )
    
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Return: {metrics['annualized_return']*100:.2f}%")
    print(f"Volatility: {metrics['annualized_volatility']*100:.2f}%")
    
    # Predict future
    results = predict_future_performance(
        tickers=tickers,
        forecast_horizon=30
    )
    
    print(f"\nForecasted daily return: {results['aggregated_prediction'].mean()*100:.4f}%")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python my_analysis.py
```

---

## 🔧 Installation Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure you're in the project directory
cd /path/to/FinLove

# Install dependencies
pip install -r requirements.txt

# If using virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Streamlit not found"

**Solution:**
```bash
pip install streamlit
```

### Issue: "CVXPY solver errors"

**Solution:**
```bash
# Install additional solvers
pip install ecos scs
```

### Issue: "Statsmodels/Prophet/TensorFlow not available"

These are optional for forecasting. The system will use fallback methods if not installed.

**To install (optional):**
```bash
pip install statsmodels  # For ARIMA
pip install prophet     # For Prophet forecasting
pip install tensorflow  # For LSTM
```

---

## 📝 Common Use Cases

### Use Case 1: Quick Portfolio Analysis

```bash
# Run dashboard
streamlit run dashboard.py

# Select: AAPL, MSFT, GOOGL
# Method: Markowitz
# Risk Model: Ledoit-Wolf
# Click "Run Analysis"
```

### Use Case 2: Compare All Models

```python
from src.model_collector import collect_all_model_results

results = collect_all_model_results(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date="2020-01-01"
)

# View all results
print(results[['model_id', 'sharpe_ratio', 'annualized_return', 'max_drawdown']])
```

### Use Case 3: Predict Next Month

```python
from src.predict import predict_future_performance

results = predict_future_performance(
    tickers=['AAPL', 'MSFT'],
    forecast_horizon=30
)

cumulative_return = (1 + results['aggregated_prediction']).prod() - 1
print(f"Expected return over next 30 days: {cumulative_return*100:.2f}%")
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Run Dashboard | `streamlit run dashboard.py` |
| Run Predictions | `python example_prediction.py` |
| Pre-download Data | `python download_data.py` |
| Install Dependencies | `pip install -r requirements.txt` |

---

## 📚 Next Steps

- Read `README.md` for detailed feature documentation
- Read `PREDICTION_GUIDE.md` for forecasting details
- Read `CODE_STYLE_GUIDE.md` for code standards
- Check `DATA.md` for data management

---

## 💡 Tips

1. **Start with the dashboard** - It's the easiest way to explore features
2. **Use default ETFs first** - They're pre-configured and work well
3. **Pre-download data** - Makes dashboard much faster
4. **Use Ledoit-Wolf** - Most stable risk model for general use
5. **Walk-forward backtest** - More realistic than simple backtest

---

Happy portfolio building! 🚀


