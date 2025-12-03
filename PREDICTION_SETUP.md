# Prediction Module Setup Guide

## What You Need to Add

The prediction system works out of the box with basic forecasting (moving average), but for advanced forecasting methods, you'll need to install optional dependencies.

## ✅ Required (Already Installed)

These are already in `requirements.txt`:
- `pandas`, `numpy` - Data handling
- `scikit-learn` - Some utilities
- `tqdm` - Progress bars

## 🔧 Optional Dependencies (For Better Forecasting)

### 1. For ARIMA & Exponential Smoothing

```bash
pip install statsmodels
```

**What it enables:**
- `arima` forecasting method
- `exponential_smoothing` forecasting method
- More accurate statistical forecasts

**Without it:**
- System falls back to moving average
- Still works, but less sophisticated

### 2. For Prophet Forecasting

```bash
pip install prophet
```

**What it enables:**
- `prophet` forecasting method
- Better handling of seasonality
- Good for time series with trends

**Without it:**
- Prophet method won't be available
- Other methods still work

### 3. For LSTM Neural Networks

```bash
pip install tensorflow
```

**What it enables:**
- `lstm` forecasting method
- Deep learning predictions
- Best for complex patterns

**Without it:**
- LSTM method won't be available
- Other methods still work

## 🚀 Quick Setup

### Install All Optional Dependencies

```bash
pip install statsmodels prophet tensorflow
```

### Check What's Available

```python
from src.predict import check_forecasting_dependencies, print_dependency_status

# Check dependencies
deps = check_forecasting_dependencies()
print(deps)

# Print status
print_dependency_status()
```

## 📝 Usage Examples

### Example 1: Basic Usage (No Extra Dependencies)

```python
from src.predict import predict_future_performance

# Uses moving average (works without extra dependencies)
results = predict_future_performance(
    tickers=['AAPL', 'MSFT'],
    forecast_method='ma'  # Moving average - always available
)
```

### Example 2: With ARIMA (Requires statsmodels)

```python
# First: pip install statsmodels

from src.predict import predict_future_performance

results = predict_future_performance(
    tickers=['AAPL', 'MSFT'],
    forecast_method='arima'  # Requires statsmodels
)
```

### Example 3: Ensemble (Uses Available Methods)

```python
from src.predict import predict_future_performance

# Automatically uses available methods
# If statsmodels installed: uses arima + ma
# If prophet installed: uses prophet + ma
# If both: uses arima + prophet + ma
results = predict_future_performance(
    tickers=['AAPL', 'MSFT'],
    forecast_method='ensemble'  # Combines available methods
)
```

## 🔍 Check Your Setup

Run this to see what's available:

```python
from src.predict import print_dependency_status

print_dependency_status()
```

Output example:
```
📦 Forecasting Dependencies Status:
==================================================
  arima                : ✅ Available
  prophet              : ❌ Not installed
  lstm                 : ❌ Not installed
  exponential_smoothing: ✅ Available

💡 Tip: Install missing dependencies for better forecasting:
   pip install prophet
   pip install tensorflow
```

## 💡 Recommendations

### Minimum Setup (Works Now)
- ✅ No additional dependencies needed
- ✅ Uses moving average forecasting
- ✅ Good for basic predictions

### Recommended Setup
```bash
pip install statsmodels
```
- ✅ Adds ARIMA forecasting
- ✅ More accurate than moving average
- ✅ Lightweight and fast

### Full Setup (Best Accuracy)
```bash
pip install statsmodels prophet tensorflow
```
- ✅ All forecasting methods available
- ✅ Ensemble uses all methods
- ✅ Best prediction accuracy

## 🐛 Troubleshooting

### "statsmodels not available"
```bash
pip install statsmodels
```

### "Prophet not available"
```bash
pip install prophet
```
Note: Prophet installation can take a few minutes as it compiles C++ code.

### "TensorFlow not available"
```bash
pip install tensorflow
```
Note: TensorFlow is large (~500MB). Only install if you need LSTM.

### Import Errors

If you get import errors, try:
```bash
pip install --upgrade statsmodels prophet tensorflow
```

## 📊 What Each Method Does

| Method | Requires | Best For | Speed |
|--------|----------|----------|-------|
| `ma` | Nothing | Baseline | ⚡ Fastest |
| `arima` | statsmodels | Trends | ⚡ Fast |
| `prophet` | prophet | Seasonality | 🐢 Medium |
| `lstm` | tensorflow | Complex patterns | 🐌 Slowest |
| `ensemble` | Any above | Robust | Depends |

## 🎯 Quick Decision Guide

**Just want it to work?**
- ✅ Nothing to install - use `forecast_method='ma'`

**Want better accuracy?**
- ✅ Install: `pip install statsmodels`
- ✅ Use: `forecast_method='arima'` or `'ensemble'`

**Want best accuracy?**
- ✅ Install: `pip install statsmodels prophet`
- ✅ Use: `forecast_method='ensemble'`

**Want deep learning?**
- ✅ Install: `pip install tensorflow`
- ✅ Use: `forecast_method='lstm'` (slower but powerful)

---

**Note:** The system automatically falls back to available methods, so you can always use `forecast_method='ensemble'` and it will use whatever is installed!


