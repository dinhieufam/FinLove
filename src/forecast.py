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

<<<<<<< HEAD
# Local imports
from .model_cache import (
    get_model_cache_key,
    load_model_from_cache,
    save_model_to_cache
)

# Suppress warnings
warnings.filterwarnings('ignore')

=======
# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA models will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. LSTM models will be disabled.")

>>>>>>> origin/trumai

def arima_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
<<<<<<< HEAD
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    auto_select: bool = True
=======
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
>>>>>>> origin/trumai
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using ARIMA or SARIMA model.
    
<<<<<<< HEAD
    This function performs a *lazy import* of ``statsmodels`` so that importing
    this module stays lightweight until ARIMA is actually needed.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        order: (p, d, q) for ARIMA. If None and auto_select=True, will try to find optimal order
        seasonal_order: (P, D, Q, s) for SARIMA (optional)
        auto_select: If True, try multiple ARIMA orders to find best fit
=======
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        order: (p, d, q) for ARIMA
        seasonal_order: (P, D, Q, s) for SARIMA (optional)
>>>>>>> origin/trumai
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
<<<<<<< HEAD
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise ImportError("statsmodels is required for ARIMA forecasting") from e
=======
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ARIMA forecasting")
>>>>>>> origin/trumai
    
    # Remove NaN values
    series_clean = series.dropna()
    
<<<<<<< HEAD
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
=======
    if len(series_clean) < 10:
        raise ValueError("Insufficient data for ARIMA model")
    
    try:
>>>>>>> origin/trumai
        if seasonal_order:
            model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(series_clean, order=order)
        
<<<<<<< HEAD
        # Fit with better optimization settings
        fitted = model.fit(method_kwargs={"warn_convergence": False})
=======
        fitted = model.fit(disp=False)
>>>>>>> origin/trumai
        
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
<<<<<<< HEAD
        # Re-raise instead of silently falling back to avoid straight-line forecasts
        raise ValueError(f"ARIMA model failed: {e}") from e
=======
        # Fallback to simple moving average
        return simple_ma_forecast(series, forecast_horizon)
>>>>>>> origin/trumai


def prophet_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using Facebook Prophet.
    
<<<<<<< HEAD
    Prophet is **not** imported at module load time to keep imports fast. It is
    lazily imported here the first time this function is called.
    
=======
>>>>>>> origin/trumai
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
<<<<<<< HEAD
    try:
        from prophet import Prophet  # type: ignore
    except ImportError as e:
        raise ImportError("Prophet is required. Install with: pip install prophet") from e
=======
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is required. Install with: pip install prophet")
>>>>>>> origin/trumai
    
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
<<<<<<< HEAD
    batch_size: int = 32,
    use_cache: bool = True,
    ticker: str = ""
=======
    batch_size: int = 32
>>>>>>> origin/trumai
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using LSTM neural network.
    
<<<<<<< HEAD
    TensorFlow / Keras and the scaler are lazily imported inside this function
    to avoid heavy imports when LSTM is not used.
    
=======
>>>>>>> origin/trumai
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
<<<<<<< HEAD
    try:
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    except ImportError as e:
        raise ImportError("TensorFlow/Keras and scikit-learn are required for LSTM forecasting") from e
=======
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for LSTM forecasting")
>>>>>>> origin/trumai
    
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
<<<<<<< HEAD
    # Check cache
    cache_key = None
    model = None
    scaler = None
    
    if use_cache and ticker:
        cache_key = get_model_cache_key(
            ticker=ticker,
            model_type='lstm',
            lookback_window=lookback_window,
            epochs=epochs,
            lstm_units=lstm_units
        )
        cached_result = load_model_from_cache(cache_key)
        if cached_result:
            model, scaler, _ = cached_result
            # Model loaded from cache - no training needed
    
    # Normalize data
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series_clean)
    else:
        # Use existing scaler but fit on new data (in case data range changed)
        scaled_data = scaler.fit_transform(series_clean)
=======
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series_clean)
>>>>>>> origin/trumai
    
    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
<<<<<<< HEAD
    # Build and train model if not cached
    if model is None:
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
        
        # Save to cache
        if use_cache and ticker and cache_key:
            save_model_to_cache(
                cache_key=cache_key,
                model=model,
                scaler=scaler,
                ticker=ticker,
                model_type='lstm',
                lookback_window=lookback_window,
                epochs=epochs,
                lstm_units=lstm_units
            )
    
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


def tcn_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    lookback_window: int = 60,
    num_filters: int = 64,
    kernel_size: int = 3,
    num_blocks: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    use_cache: bool = True,
    ticker: str = ""
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using Temporal Convolutional Network (TCN).
    
    TCN uses dilated convolutions to capture long-term dependencies in time series.
    This implementation uses 1D convolutions with causal padding.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        lookback_window: Number of past periods to use as input
        num_filters: Number of filters in each convolutional layer
        kernel_size: Size of the convolutional kernel
        num_blocks: Number of TCN blocks (each block doubles the dilation)
        epochs: Training epochs
        batch_size: Batch size for training
    
    Returns:
        Tuple of (forecast, None) - confidence intervals not available for TCN
    """
    try:
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
        from tensorflow.keras.models import Model  # type: ignore
        from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Dropout, Dense  # type: ignore
        from tensorflow.keras.optimizers import Adam  # type: ignore
    except ImportError as e:
        raise ImportError("TensorFlow/Keras and scikit-learn are required for TCN forecasting") from e
    
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache
    cache_key = None
    model = None
    scaler = None
    
    if use_cache and ticker:
        cache_key = get_model_cache_key(
            ticker=ticker,
            model_type='tcn',
            lookback_window=lookback_window,
            epochs=epochs,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_blocks=num_blocks
        )
        cached_result = load_model_from_cache(cache_key)
        if cached_result:
            model, scaler, _ = cached_result
    
    # Normalize data
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series_clean)
    else:
        scaled_data = scaler.fit_transform(series_clean)
    
    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build TCN model
    # TCN uses dilated causal convolutions
    def tcn_block(x, filters, kernel_size, dilation_rate, block_num):
        """Create a TCN residual block with dilated convolutions."""
        # First convolution with dilation
        conv1 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f'tcn_conv1_block_{block_num}'
        )(x)
        conv1 = Activation('relu')(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        # Second convolution
        conv2 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            name=f'tcn_conv2_block_{block_num}'
        )(conv1)
        conv2 = Activation('relu')(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # Residual connection (if dimensions match)
        if x.shape[-1] == filters:
            res = Add()([x, conv2])
        else:
            # Projection shortcut if dimensions don't match
            shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(x)
            res = Add()([shortcut, conv2])
        
        return res
    
    # Build and train model if not cached
    if model is None:
        # Input layer
        inputs = Input(shape=(lookback_window, 1))
        x = inputs
        
        # Build TCN blocks with increasing dilation rates
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            x = tcn_block(x, num_filters, kernel_size, dilation_rate, i)
        
        # Final layers
        x = Conv1D(filters=num_filters, kernel_size=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        
        # Global average pooling and output
        x = Conv1D(filters=1, kernel_size=1, padding='same')(x)
        x = Dense(1)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Save to cache
        if use_cache and ticker and cache_key:
            save_model_to_cache(
                cache_key=cache_key,
                model=model,
                scaler=scaler,
                ticker=ticker,
                model_type='tcn',
                lookback_window=lookback_window,
                epochs=epochs,
                num_filters=num_filters,
                kernel_size=kernel_size,
                num_blocks=num_blocks
            )
    
    # Forecast
    last_sequence = scaled_data[-lookback_window:].reshape(1, lookback_window, 1)
    forecasts = []
    
    for _ in range(forecast_horizon):
        next_pred = model.predict(last_sequence, verbose=0)
        # Handle different output shapes from TCN model
        if next_pred.ndim == 3:
            # TCN outputs (1, 1, 1) shape
            pred_value = next_pred[0, 0, 0]
        elif next_pred.ndim == 2:
            # Some models output (1, 1) shape
            pred_value = next_pred[0, 0]
        else:
            # Fallback for other shapes
            pred_value = next_pred.flatten()[0]
        
        forecasts.append(pred_value)
        # Update sequence for next prediction - ensure correct shape
        pred_reshaped = np.array([[pred_value]]).reshape(1, 1, 1)
        last_sequence = np.append(last_sequence[:, 1:, :], pred_reshaped, axis=1)
    
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


def xgboost_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    lookback_window: int = 60,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    use_cache: bool = True,
    ticker: str = ""
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using XGBoost gradient boosting.
    
    XGBoost creates features from lagged values and uses gradient boosting
    to predict future returns.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        lookback_window: Number of past periods to use as features
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate for boosting
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        import xgboost as xgb  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ImportError as e:
        raise ImportError("XGBoost and scikit-learn are required for XGBoost forecasting") from e
    
    series_clean = series.dropna()
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache
    cache_key = None
    model = None
    scaler = None
    
    if use_cache and ticker:
        cache_key = get_model_cache_key(
            ticker=ticker,
            model_type='xgboost',
            lookback_window=lookback_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
        cached_result = load_model_from_cache(cache_key)
        if cached_result:
            model, scaler, _ = cached_result
    
    # Create features from lagged values
    def create_features(data, window):
        """Create features from lagged values and rolling statistics."""
        features = []
        targets = []
        
        for i in range(window, len(data)):
            # Lagged values
            lag_features = data.iloc[i-window:i].values
            
            # Rolling statistics
            rolling_mean = data.iloc[i-window:i].mean()
            rolling_std = data.iloc[i-window:i].std()
            rolling_max = data.iloc[i-window:i].max()
            rolling_min = data.iloc[i-window:i].min()
            
            # Combine features
            feature_vector = np.concatenate([
                lag_features,
                [rolling_mean, rolling_std, rolling_max, rolling_min]
            ])
            
            features.append(feature_vector)
            targets.append(data.iloc[i])
        
        return np.array(features), np.array(targets)
    
    # Prepare training data
    X_train, y_train = create_features(series_clean, lookback_window)
    
    # Normalize features
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.fit_transform(X_train)
    
    # Train XGBoost model if not cached
    if model is None:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Save to cache
        if use_cache and ticker and cache_key:
            save_model_to_cache(
                cache_key=cache_key,
                model=model,
                scaler=scaler,
                ticker=ticker,
                model_type='xgboost',
                lookback_window=lookback_window,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate
            )
    
    # Forecast recursively
    forecasts = []
    last_values = series_clean.iloc[-lookback_window:].values
    
    for _ in range(forecast_horizon):
        # Create features from last values
        rolling_mean = np.mean(last_values)
        rolling_std = np.std(last_values)
        rolling_max = np.max(last_values)
        rolling_min = np.min(last_values)
        
        feature_vector = np.concatenate([
            last_values,
            [rolling_mean, rolling_std, rolling_max, rolling_min]
        ]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict next value
        next_pred = model.predict(feature_vector_scaled)[0]
        forecasts.append(next_pred)
        
        # Update last values (shift and append)
        last_values = np.append(last_values[1:], next_pred)
    
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
    
    # Simple confidence interval based on historical residuals
    train_pred = model.predict(X_train_scaled)
    residuals = y_train - train_pred
    std_residual = np.std(residuals)
    
    conf_int_series = pd.DataFrame({
        'lower': forecast_series - 1.96 * std_residual,
        'upper': forecast_series + 1.96 * std_residual
    }, index=future_dates)
    
    return forecast_series, conf_int_series


def transformer_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    lookback_window: int = 60,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    use_cache: bool = True,
    ticker: str = ""
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Forecast using Time Series Transformer.
    
    This implementation uses a simplified transformer architecture with
    multi-head attention for time series forecasting.
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        lookback_window: Number of past periods to use as input
        d_model: Dimension of the model
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        epochs: Training epochs
        batch_size: Batch size for training
    
    Returns:
        Tuple of (forecast, None) - confidence intervals not available for transformer
    """
    try:
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
        from tensorflow.keras.models import Model  # type: ignore
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
            GlobalAveragePooling1D, Add, Conv1D
        )  # type: ignore
        from tensorflow.keras.optimizers import Adam  # type: ignore
    except ImportError as e:
        raise ImportError("TensorFlow/Keras and scikit-learn are required for Transformer forecasting") from e
    
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache
    cache_key = None
    model = None
    scaler = None
    
    if use_cache and ticker:
        cache_key = get_model_cache_key(
            ticker=ticker,
            model_type='transformer',
            lookback_window=lookback_window,
            epochs=epochs,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        cached_result = load_model_from_cache(cache_key)
        if cached_result:
            model, scaler, _ = cached_result
    
    # Normalize data
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series_clean)
    else:
        scaled_data = scaler.fit_transform(series_clean)
    
    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build and train model if not cached
    if model is None:
        # Build Transformer model
        def transformer_block(x, d_model, num_heads, name_prefix):
            """Create a transformer block with self-attention."""
            # Self-attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model,
                name=f'{name_prefix}_attention'
            )(x, x)
            attn_output = Dropout(0.1)(attn_output)
            
            # Add & Norm
            x = Add()([x, attn_output])
            x = LayerNormalization(name=f'{name_prefix}_norm1')(x)
            
            # Feed forward
            ff_output = Dense(d_model * 2, activation='relu')(x)
            ff_output = Dense(d_model)(ff_output)
            ff_output = Dropout(0.1)(ff_output)
            
            # Add & Norm
            x = Add()([x, ff_output])
            x = LayerNormalization(name=f'{name_prefix}_norm2')(x)
            
            return x
        
        # Input layer
        inputs = Input(shape=(lookback_window, 1))
        
        # Project to d_model dimensions
        x = Conv1D(filters=d_model, kernel_size=1, padding='same')(inputs)
        
        # Stack transformer blocks
        for i in range(num_layers):
            x = transformer_block(x, d_model, num_heads, f'block_{i}')
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(d_model, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Save to cache
        if use_cache and ticker and cache_key:
            save_model_to_cache(
                cache_key=cache_key,
                model=model,
                scaler=scaler,
                ticker=ticker,
                model_type='transformer',
                lookback_window=lookback_window,
                epochs=epochs,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers
            )
=======
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
>>>>>>> origin/trumai
    
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
<<<<<<< HEAD
    window: int = 20,
    include_trend: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple moving average forecast with optional trend/drift.
    
    This baseline model now includes a simple trend component to avoid
    producing completely flat forecasts. For financial returns, it uses
    recent mean with mean reversion.
=======
    window: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple moving average forecast (baseline).
>>>>>>> origin/trumai
    
    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        window: Moving average window
<<<<<<< HEAD
        include_trend: If True, add a simple trend/drift component
=======
>>>>>>> origin/trumai
    
    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    series_clean = series.dropna()
    
<<<<<<< HEAD
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
=======
    if len(series_clean) < window:
        # Use all available data
        ma_value = series_clean.mean()
    else:
        ma_value = series_clean.iloc[-window:].mean()
    
    # Forecast is constant (mean)
>>>>>>> origin/trumai
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
    
<<<<<<< HEAD
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
=======
    forecast_series = pd.Series([ma_value] * forecast_horizon, index=future_dates)
>>>>>>> origin/trumai
    
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
<<<<<<< HEAD
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        raise ImportError("statsmodels is required for exponential smoothing") from e
=======
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for exponential smoothing")
>>>>>>> origin/trumai
    
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
<<<<<<< HEAD
        # Re-raise to let ensemble handle it, or use improved MA with trend
        # Don't silently fall back to avoid straight lines
        raise ValueError(f"Exponential smoothing failed: {e}") from e
=======
        return simple_ma_forecast(series, forecast_horizon)
>>>>>>> origin/trumai


def forecast_portfolio_returns(
    portfolio_returns: pd.Series,
    method: str = 'arima',
    forecast_horizon: int = 30,
<<<<<<< HEAD
    use_cache: bool = True,
    ticker: str = "",
=======
>>>>>>> origin/trumai
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Forecast portfolio returns using specified method.
    
    Args:
        portfolio_returns: Historical portfolio returns
<<<<<<< HEAD
        method: Forecasting method ('arima', 'prophet', 'lstm', 'tcn', 'xgboost', 'transformer', 'ma', 'exponential_smoothing')
        forecast_horizon: Number of periods to forecast
        use_cache: Whether to use cached models if available
        ticker: Ticker symbol for caching (optional but recommended)
=======
        method: Forecasting method ('arima', 'prophet', 'lstm', 'ma', 'exponential_smoothing')
        forecast_horizon: Number of periods to forecast
>>>>>>> origin/trumai
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Dictionary with 'forecast' and 'confidence_intervals' (if available)
    """
<<<<<<< HEAD
    # Filter out parameters that are handled explicitly or not applicable to certain methods
    filtered_kwargs = kwargs.copy()
    # Remove use_cache and ticker from kwargs since they're passed explicitly to neural network methods
    filtered_kwargs.pop('use_cache', None)
    filtered_kwargs.pop('ticker', None)
    # Filter out 'epochs' parameter for XGBoost (it doesn't use epochs)
    if method == 'xgboost' and 'epochs' in filtered_kwargs:
        filtered_kwargs.pop('epochs')
    
    if method == 'arima':
        forecast, conf_int = arima_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
    elif method == 'prophet':
        forecast, conf_int = prophet_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
    elif method == 'lstm':
        forecast, conf_int = lstm_forecast(portfolio_returns, forecast_horizon, use_cache=use_cache, ticker=ticker, **filtered_kwargs)
    elif method == 'tcn':
        forecast, conf_int = tcn_forecast(portfolio_returns, forecast_horizon, use_cache=use_cache, ticker=ticker, **filtered_kwargs)
    elif method == 'xgboost':
        forecast, conf_int = xgboost_forecast(portfolio_returns, forecast_horizon, use_cache=use_cache, ticker=ticker, **filtered_kwargs)
    elif method == 'transformer':
        forecast, conf_int = transformer_forecast(portfolio_returns, forecast_horizon, use_cache=use_cache, ticker=ticker, **filtered_kwargs)
    elif method == 'ma':
        forecast, conf_int = simple_ma_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
    elif method == 'exponential_smoothing':
        forecast, conf_int = exponential_smoothing_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
=======
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
>>>>>>> origin/trumai
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
    
<<<<<<< HEAD
    This function tries multiple forecasting methods and combines their results.
    If a method fails, it's skipped (but at least one must succeed).
    The MA method is always included as a fallback to ensure we have at least one forecast.
    
=======
>>>>>>> origin/trumai
    Args:
        portfolio_returns: Historical portfolio returns
        methods: List of forecasting methods to use
        forecast_horizon: Number of periods to forecast
        aggregation: How to combine forecasts ('mean', 'median', 'weighted')
    
    Returns:
        Dictionary with ensemble forecast and individual forecasts
    """
    individual_forecasts = {}
<<<<<<< HEAD
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
        
=======
    
    for method in methods:
>>>>>>> origin/trumai
        try:
            result = forecast_portfolio_returns(
                portfolio_returns,
                method=method,
                forecast_horizon=forecast_horizon
            )
            individual_forecasts[method] = result['forecast']
        except Exception as e:
            print(f"Warning: {method} forecast failed: {e}")
<<<<<<< HEAD
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
=======
            continue
    
    if not individual_forecasts:
        raise ValueError("All forecasting methods failed")
    
    # Combine forecasts
    forecast_df = pd.DataFrame(individual_forecasts)
>>>>>>> origin/trumai
    
    if aggregation == 'mean':
        ensemble = forecast_df.mean(axis=1)
    elif aggregation == 'median':
        ensemble = forecast_df.median(axis=1)
    elif aggregation == 'weighted':
<<<<<<< HEAD
        # Weight by inverse of forecast variance (more stable forecasts get higher weight)
        weights = 1.0 / (forecast_df.std(axis=0) + 1e-6)
        weights = weights / weights.sum()
        ensemble = (forecast_df * weights).sum(axis=1)
=======
        # Equal weights for now
        ensemble = forecast_df.mean(axis=1)
>>>>>>> origin/trumai
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    result = {
        'ensemble_forecast': ensemble,
<<<<<<< HEAD
        'individual_forecasts': individual_forecasts,
        'failed_methods': failed_methods
=======
        'individual_forecasts': individual_forecasts
>>>>>>> origin/trumai
    }
    
    return result

