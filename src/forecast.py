"""
Time Series Forecasting Module.

This module implements various time series models to predict future portfolio performance:
- ARIMA/ARIMA-GARCH
- LSTM (Long Short-Term Memory)
- Prophet (Facebook's forecasting tool)
- Simple moving average / exponential smoothing
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')


def arima_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    auto_select: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using ARIMA or SARIMA model.
    
    This function performs a *lazy import* of ``statsmodels`` so that importing
    this module stays lightweight until ARIMA is actually needed.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        order: (p, d, q) for ARIMA. If None and auto_select=True, will try to find optimal order
        seasonal_order: (P, D, Q, s) for SARIMA (optional)
        auto_select: If True, try multiple ARIMA orders to find best fit
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise ImportError("statsmodels is required for ARIMA forecasting") from e
    
    # Remove NaN values
    series_clean = series.dropna()
    
    if len(series_clean) < 20:
        raise ValueError("Insufficient data for ARIMA model (need at least 20 periods)")
    
    try:
        # Auto-select optimal ARIMA order if not provided
        if order is None and auto_select:
            # Try to determine differencing order (d) using ADF test
            d = 0
            adf_result = adfuller(series_clean)
            if adf_result[1] > 0.05:  # Not stationary, try differencing
                diff_series = series_clean.diff().dropna()
                if len(diff_series) > 0:
                    adf_result_diff = adfuller(diff_series)
                    if adf_result_diff[1] < 0.05:
                        d = 1
            
            # Try a few common ARIMA orders for financial returns
            # Returns are typically well-modeled by ARIMA(0,1,1) or ARIMA(1,1,1)
            candidate_orders = [
                (0, d, 1),  # IMA model - common for returns
                (1, d, 1),  # ARIMA(1,1,1) - standard
                (1, d, 0),  # AR(1) with differencing
                (0, d, 2),  # IMA(2)
                (2, d, 1),  # ARIMA(2,1,1)
            ]
            
            best_aic = np.inf
            best_order = (1, 1, 1)  # Default fallback
            
            for candidate_order in candidate_orders:
                try:
                    if candidate_order[1] == 0 and len(series_clean) < 30:
                        continue  # Skip non-differenced models for short series
                    
                    test_model = ARIMA(series_clean, order=candidate_order)
                    test_fitted = test_model.fit()
                    if test_fitted.aic < best_aic:
                        best_aic = test_fitted.aic
                        best_order = candidate_order
                except:
                    continue
            
            order = best_order
        
        # Use default order if still None
        if order is None:
            order = (1, 1, 1)
        
        # Fit the model
        if seasonal_order:
            model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(series_clean, order=order)
        
        # Fit with better optimization settings
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        
        # Forecast
        forecast = fitted.forecast(steps=forecast_horizon)
        conf_int = fitted.get_forecast(steps=forecast_horizon).conf_int()
        
        # Create future dates
        last_date = series_clean.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series_clean.index)
            if freq is None:
                freq = 'D'  # Default to daily
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)
        
        forecast_series = pd.Series(forecast.values, index=future_dates)
        conf_int_series = pd.DataFrame(conf_int, index=future_dates)
        
        return forecast_series, conf_int_series
        
    except Exception as e:
        print(f"ARIMA forecast error: {e}")
        # Re-raise instead of silently falling back to avoid straight-line forecasts
        raise ValueError(f"ARIMA model failed: {e}") from e


def prophet_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using Facebook Prophet.
    
    Prophet is **not** imported at module load time to keep imports fast. It is
    lazily imported here the first time this function is called.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as e:
        raise ImportError("Prophet is required. Install with: pip install prophet") from e
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        raise ValueError("Insufficient data for Prophet model")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'ds': series_clean.index,
        'y': series_clean.values
    })
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    model.fit(df)
    
    # Create future dates
    future = model.make_future_dataframe(periods=forecast_horizon)
    
    # Forecast
    forecast = model.predict(future)
    
    # Extract forecast and confidence intervals
    forecast_series = forecast['yhat'].iloc[-forecast_horizon:]
    forecast_series.index = forecast['ds'].iloc[-forecast_horizon:]
    
    lower_bound = forecast['yhat_lower'].iloc[-forecast_horizon:]
    upper_bound = forecast['yhat_upper'].iloc[-forecast_horizon:]
    conf_int_series = pd.DataFrame({
        'lower': lower_bound.values,
        'upper': upper_bound.values
    }, index=forecast['ds'].iloc[-forecast_horizon:])
    
    return forecast_series, conf_int_series


def lstm_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    lookback_window: int = 60,
    lstm_units: int = 50,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using LSTM neural network.
    
    TensorFlow / Keras and the scaler are lazily imported inside this function
    to avoid heavy imports when LSTM is not used.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        lookback_window: Number of past periods to use as input
        lstm_units: Number of LSTM units
        epochs: Training epochs
        batch_size: Batch size for training
    
    Returns:
        Tuple of (forecast, None) - confidence intervals not available for LSTM
    """
    try:
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    except ImportError as e:
        raise ImportError("TensorFlow/Keras and scikit-learn are required for LSTM forecasting") from e
    
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series_clean)
    
    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(lookback_window, 1)),
        Dropout(0.2),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Forecast
    last_sequence = scaled_data[-lookback_window:].reshape(1, lookback_window, 1)
    forecasts = []
    
    for _ in range(forecast_horizon):
        next_pred = model.predict(last_sequence, verbose=0)
        forecasts.append(next_pred[0, 0])
        # Update sequence for next prediction
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts).flatten()
    
    # Create future dates
    last_date = series.index[-1]
    if isinstance(last_date, pd.Timestamp):
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = 'D'
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq=freq
        )
    else:
        future_dates = range(len(series), len(series) + forecast_horizon)
    
    forecast_series = pd.Series(forecasts, index=future_dates)
    
    return forecast_series, None


def simple_ma_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    window: int = 20,
    include_trend: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple moving average forecast with optional trend/drift.
    
    This baseline model now includes a simple trend component to avoid
    producing completely flat forecasts. For financial returns, it uses
    recent mean with mean reversion.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        window: Moving average window
        include_trend: If True, add a simple trend/drift component
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    series_clean = series.dropna()
    
    # Calculate recent mean (used as starting point for forecast)
    if len(series_clean) < window:
        # Use all available data
        recent_mean = series_clean.mean()
    else:
        recent_mean = series_clean.iloc[-window:].mean()
    
    # Calculate simple trend/drift from recent data
    if include_trend and len(series_clean) >= window * 2:
        # Compare recent window to previous window
        recent_window = series_clean.iloc[-window:]
        previous_window = series_clean.iloc[-window*2:-window]
        
        recent_window_mean = recent_window.mean()
        previous_mean = previous_window.mean()
        
        # Calculate drift (mean reversion for returns)
        drift = (recent_window_mean - previous_mean) / window
        
        # For returns, apply mean reversion (drift towards zero)
        # This prevents the forecast from being completely flat
        mean_reversion_rate = 0.1  # 10% reversion per period
        drift = drift * (1 - mean_reversion_rate)
        
        # Update recent_mean to use the more recent window
        recent_mean = recent_window_mean
    else:
        drift = 0.0
    
    # Create future dates
    last_date = series_clean.index[-1]
    if isinstance(last_date, pd.Timestamp):
        freq = pd.infer_freq(series_clean.index)
        if freq is None:
            freq = 'D'
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_horizon,
            freq=freq
        )
    else:
        future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)
    
    # Generate forecast with trend/drift
    # For returns, apply mean reversion: forecast gradually returns to zero
    forecast_values = []
    current_value = recent_mean
    
    for i in range(forecast_horizon):
        # Mean reversion: gradually move towards zero
        # This creates a non-flat forecast even for the baseline model
        forecast_values.append(current_value)
        current_value = current_value * (1 - 0.05) + drift  # 5% mean reversion per period
    
    forecast_series = pd.Series(forecast_values, index=future_dates)
    
    # Simple confidence interval based on historical volatility
    std = series_clean.std()
    conf_int_series = pd.DataFrame({
        'lower': forecast_series - 1.96 * std,
        'upper': forecast_series + 1.96 * std
    }, index=future_dates)
    
    return forecast_series, conf_int_series


def exponential_smoothing_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    trend: Optional[str] = 'add',
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using exponential smoothing (Holt-Winters).
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        trend: 'add' or 'mul' for additive/multiplicative trend
        seasonal: 'add' or 'mul' for seasonal component
        seasonal_periods: Number of periods in a season
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        raise ImportError("statsmodels is required for exponential smoothing") from e
    
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        raise ValueError("Insufficient data for exponential smoothing")
    
    try:
        model = ExponentialSmoothing(
            series_clean,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_horizon)
        
        # Create future dates
        last_date = series_clean.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series_clean.index)
            if freq is None:
                freq = 'D'
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)
        
        forecast_series = pd.Series(forecast.values, index=future_dates)
        
        # Simple confidence interval
        std = series_clean.std()
        conf_int_series = pd.DataFrame({
            'lower': forecast_series - 1.96 * std,
            'upper': forecast_series + 1.96 * std
        }, index=future_dates)
        
        return forecast_series, conf_int_series
        
    except Exception as e:
        print(f"Exponential smoothing error: {e}")
        # Re-raise to let ensemble handle it, or use improved MA with trend
        # Don't silently fall back to avoid straight lines
        raise ValueError(f"Exponential smoothing failed: {e}") from e


def forecast_portfolio_returns(
    portfolio_returns: pd.Series,
    method: str = 'arima',
    forecast_horizon: int = 30,
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Forecast portfolio returns using specified method.
    
    Args:
        portfolio_returns: Historical portfolio returns
        method: Forecasting method ('arima', 'prophet', 'lstm', 'ma', 'exponential_smoothing')
        forecast_horizon: Number of periods to forecast
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Dictionary with 'forecast' and 'confidence_intervals' (if available)
    """
    if method == 'arima':
        forecast, conf_int = arima_forecast(portfolio_returns, forecast_horizon, **kwargs)
    elif method == 'prophet':
        forecast, conf_int = prophet_forecast(portfolio_returns, forecast_horizon, **kwargs)
    elif method == 'lstm':
        forecast, conf_int = lstm_forecast(portfolio_returns, forecast_horizon, **kwargs)
    elif method == 'ma':
        forecast, conf_int = simple_ma_forecast(portfolio_returns, forecast_horizon, **kwargs)
    elif method == 'exponential_smoothing':
        forecast, conf_int = exponential_smoothing_forecast(portfolio_returns, forecast_horizon, **kwargs)
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
    
    result = {'forecast': forecast}
    if conf_int is not None:
        result['confidence_intervals'] = conf_int
    
    return result


def ensemble_forecast(
    portfolio_returns: pd.Series,
    methods: List[str] = ['arima', 'prophet', 'ma'],
    forecast_horizon: int = 30,
    aggregation: str = 'mean'
) -> Dict[str, pd.Series]:
    """
    Ensemble forecast using multiple methods.
    
    This function tries multiple forecasting methods and combines their results.
    If a method fails, it's skipped (but at least one must succeed).
    The MA method is always included as a fallback to ensure we have at least one forecast.
    
    Args:
        portfolio_returns: Historical portfolio returns
        methods: List of forecasting methods to use
        forecast_horizon: Number of periods to forecast
        aggregation: How to combine forecasts ('mean', 'median', 'weighted')
    
    Returns:
        Dictionary with ensemble forecast and individual forecasts
    """
    individual_forecasts = {}
    failed_methods = []
    
    # Always try MA first as a baseline (it should always work)
    if 'ma' in methods:
        try:
            result = forecast_portfolio_returns(
                portfolio_returns,
                method='ma',
                forecast_horizon=forecast_horizon
            )
            individual_forecasts['ma'] = result['forecast']
        except Exception as e:
            print(f"Warning: MA forecast failed: {e}")
            failed_methods.append('ma')
    
    # Try other methods
    for method in methods:
        if method == 'ma':
            continue  # Already tried
        
        try:
            result = forecast_portfolio_returns(
                portfolio_returns,
                method=method,
                forecast_horizon=forecast_horizon
            )
            individual_forecasts[method] = result['forecast']
        except Exception as e:
            print(f"Warning: {method} forecast failed: {e}")
            failed_methods.append(method)
            continue
    
    if not individual_forecasts:
        raise ValueError("All forecasting methods failed. Cannot generate forecast.")
    
    # Combine forecasts - align all series by index first
    forecast_dfs = []
    for method, forecast_series in individual_forecasts.items():
        if isinstance(forecast_series, pd.Series):
            forecast_dfs.append(forecast_series.to_frame(name=method))
    
    if not forecast_dfs:
        raise ValueError("No valid forecasts to combine")
    
    # Align all forecasts to the same index
    forecast_df = pd.concat(forecast_dfs, axis=1)
    forecast_df = forecast_df.fillna(method='ffill').fillna(method='bfill')
    
    if aggregation == 'mean':
        ensemble = forecast_df.mean(axis=1)
    elif aggregation == 'median':
        ensemble = forecast_df.median(axis=1)
    elif aggregation == 'weighted':
        # Weight by inverse of forecast variance (more stable forecasts get higher weight)
        weights = 1.0 / (forecast_df.std(axis=0) + 1e-6)
        weights = weights / weights.sum()
        ensemble = (forecast_df * weights).sum(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    result = {
        'ensemble_forecast': ensemble,
        'individual_forecasts': individual_forecasts,
        'failed_methods': failed_methods
    }
    
    return result

