# Time Series Prediction Guide

This guide explains how to use the prediction system to forecast future portfolio performance based on results from multiple optimization models.

## Overview

The prediction system consists of three main modules:

1. **`model_collector.py`** - Collects results from all optimization method and risk model combinations
2. **`forecast.py`** - Implements various time series forecasting models (ARIMA, Prophet, LSTM, etc.)
3. **`predict.py`** - Main interface that combines collection and forecasting

## Quick Start

### Basic Usage

```python
from src.predict import predict_future_performance

# Define your portfolio
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# Run prediction pipeline
results = predict_future_performance(
    tickers=tickers,
    start_date="2015-01-01",
    end_date="2024-01-01",
    forecast_horizon=30,  # Forecast next 30 days
    forecast_method='ensemble',  # Use ensemble of methods
    use_top_models=5,  # Use top 5 models
    backtest_type='walk_forward'
)

# Access results
print(results['aggregated_prediction'])  # Forecasted returns
print(results['top_models'])  # Top performing models
```

### Step-by-Step Approach

#### Step 1: Collect Model Results

```python
from src.model_collector import collect_all_model_results, get_best_models

# Collect results from all combinations
model_results = collect_all_model_results(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date="2015-01-01",
    end_date="2024-01-01",
    backtest_type='walk_forward',
    train_window=36
)

# Find best models
top_models = get_best_models(model_results, metric='sharpe_ratio', top_n=5)
```

#### Step 2: Forecast Future Returns

```python
from src.forecast import forecast_portfolio_returns, ensemble_forecast

# Get portfolio returns from a model
portfolio_returns = top_models.iloc[0]['portfolio_returns']

# Forecast using single method
forecast = forecast_portfolio_returns(
    portfolio_returns,
    method='arima',  # or 'prophet', 'lstm', 'ma'
    forecast_horizon=30
)

# Or use ensemble (recommended)
ensemble = ensemble_forecast(
    portfolio_returns,
    methods=['arima', 'prophet', 'ma'],
    forecast_horizon=30
)
```

## Available Forecasting Methods

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Best for**: Stationary time series with trends
- **Requirements**: `statsmodels`
- **Usage**: `method='arima'`

### 2. Prophet (Facebook's Forecasting Tool)
- **Best for**: Time series with seasonality
- **Requirements**: `prophet` (install: `pip install prophet`)
- **Usage**: `method='prophet'`

### 3. LSTM (Long Short-Term Memory)
- **Best for**: Complex non-linear patterns
- **Requirements**: `tensorflow`
- **Usage**: `method='lstm'`
- **Note**: Requires more data and computation time

### 4. Moving Average (Simple Baseline)
- **Best for**: Baseline comparison
- **Requirements**: None (built-in)
- **Usage**: `method='ma'`

### 5. Exponential Smoothing
- **Best for**: Time series with trends and seasonality
- **Requirements**: `statsmodels`
- **Usage**: `method='exponential_smoothing'`

### 6. Ensemble (Recommended)
- **Best for**: Robust predictions by combining multiple methods
- **Requirements**: Multiple methods available
- **Usage**: `method='ensemble'`

## Model Combinations

The system tests all combinations of:

**Optimization Methods:**
- `markowitz` - Mean-variance optimization
- `min_variance` - Minimum variance portfolio
- `sharpe` - Sharpe ratio maximization
- `black_litterman` - Black-Litterman model
- `cvar` - Conditional Value at Risk optimization

**Risk Models:**
- `sample` - Sample covariance
- `ledoit_wolf` - Ledoit-Wolf shrinkage
- `glasso` - Graphical LASSO
- `garch` - GARCH-based covariance

**Total Combinations:** 5 × 4 = 20 models

## Example Use Cases

### Use Case 1: Predict Next Month's Returns

```python
results = predict_future_performance(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    forecast_horizon=30,
    forecast_method='ensemble',
    use_top_models=5
)

# Expected cumulative return over next 30 days
cumulative_return = (1 + results['aggregated_prediction']).prod() - 1
print(f"Expected return: {cumulative_return*100:.2f}%")
```

### Use Case 2: Compare Forecasting Methods

```python
from src.forecast import forecast_portfolio_returns

methods = ['arima', 'prophet', 'ma', 'exponential_smoothing']
forecasts = {}

for method in methods:
    try:
        result = forecast_portfolio_returns(
            portfolio_returns,
            method=method,
            forecast_horizon=30
        )
        forecasts[method] = result['forecast']
    except Exception as e:
        print(f"{method} failed: {e}")

# Compare forecasts
import pandas as pd
comparison = pd.DataFrame(forecasts)
print(comparison.describe())
```

### Use Case 3: Predict Portfolio Weights

```python
from src.predict import predict_portfolio_weights

# Predict future optimal weights
future_weights = predict_portfolio_weights(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    forecast_horizon=30,
    optimization_method='markowitz',
    risk_model='ledoit_wolf'
)

print(future_weights)
```

## Installation

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Optional Dependencies for Forecasting

**For ARIMA/Exponential Smoothing:**
```bash
pip install statsmodels
```

**For Prophet:**
```bash
pip install prophet
```

**For LSTM:**
```bash
pip install tensorflow
```

**Install All:**
```bash
pip install statsmodels prophet tensorflow
```

## Output Structure

The `predict_future_performance` function returns a dictionary with:

- **`model_results`**: DataFrame with all model combinations and their metrics
- **`top_models`**: DataFrame with top N models selected
- **`predictions`**: Dictionary mapping model_id to forecast Series
- **`aggregated_prediction`**: Combined forecast from all top models
- **`forecast_dates`**: Future dates for predictions
- **`forecast_horizon`**: Number of periods forecasted

## Tips and Best Practices

1. **Use Ensemble Methods**: Ensemble forecasting typically provides more robust predictions than single methods.

2. **Select Top Models**: Use `use_top_models=5` to focus on best-performing models rather than all models.

3. **Adjust Forecast Horizon**: 
   - Short-term (1-30 days): Use ARIMA or MA
   - Medium-term (30-90 days): Use Prophet or Ensemble
   - Long-term (90+ days): Be cautious - forecasts become less reliable

4. **Validate Predictions**: Compare forecasts with actual returns to assess accuracy.

5. **Consider Multiple Metrics**: Don't just use Sharpe ratio - consider return, volatility, and drawdown.

6. **Data Quality**: Ensure sufficient historical data (at least 2-3 years for daily data).

## Troubleshooting

### "statsmodels not available"
```bash
pip install statsmodels
```

### "Prophet not available"
```bash
pip install prophet
```

### "TensorFlow not available"
```bash
pip install tensorflow
```

### "Insufficient data"
- Ensure you have at least 60-100 days of historical data
- Check for missing values in your data

### "ARIMA forecast error"
- Try different ARIMA orders: `order=(1,1,1)` or `order=(2,1,2)`
- Use `method='ma'` as fallback

## Advanced Usage

### Custom Forecasting

```python
from src.forecast import arima_forecast

# Custom ARIMA parameters
forecast, conf_int = arima_forecast(
    portfolio_returns,
    forecast_horizon=30,
    order=(2, 1, 2),  # (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s) for monthly seasonality
)
```

### Weighted Ensemble

```python
from src.model_collector import aggregate_model_predictions

# Weight predictions by Sharpe ratio
predictions = {...}  # Your predictions dict
weights = {...}  # Weights based on Sharpe ratios

# Custom aggregation (you'll need to implement weighted_mean)
aggregated = aggregate_model_predictions(predictions, method='weighted_mean')
```

## Example Script

See `example_prediction.py` for a complete working example.

Run it with:
```bash
python example_prediction.py
```

## Questions?

Check the docstrings in each module for detailed function documentation.


