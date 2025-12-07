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

# Local imports
from .model_cache import (
    get_model_cache_key,
    load_model_from_cache,
    save_model_to_cache
)

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
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError as e:
        raise ImportError("statsmodels is required for ARIMA forecasting") from e

    # Preprocessing
    series_clean = series.dropna()
    if len(series_clean) < 20:
        raise ValueError("Insufficient data for ARIMA model (need at least 20 periods)")

    try:
        if order is None and auto_select:
            # Use ACF/PACF to find likely AR/MA order
            nlags = min(10, max(5, len(series_clean) // 6))
            acf_vals = acf(series_clean, nlags=nlags)
            pacf_vals = pacf(series_clean, nlags=nlags)
            # AR if PACF drops quickly, MA if ACF drops quickly
            ar_lag = np.where(np.abs(pacf_vals) < 0.2)[0]
            ma_lag = np.where(np.abs(acf_vals) < 0.2)[0]
            p = int(ar_lag[0]) if len(ar_lag) > 1 else 1
            q = int(ma_lag[0]) if len(ma_lag) > 1 else 1

            # Use ADF test for differencing - but prefer d=0 or d=1 to avoid over-differencing
            adf_pvalue = adfuller(series_clean)[1]
            # Prefer d=0 for stationary data, d=1 only if clearly non-stationary
            d = 0 if adf_pvalue < 0.05 else 1
            # For financial returns, often d=0 is better (returns are usually stationary)
            if np.abs(series_clean.mean()) < 0.001 and series_clean.std() > 0:
                d = 0  # Returns are typically stationary

            # Generate candidate orders - prefer models with AR/MA terms to avoid flat forecasts
            candidate_orders = []
            # Prioritize orders with at least one AR or MA term to ensure dynamics
            for ap in range(max(1, p-1), min(4, p+2)):  # Start from 1 to ensure dynamics
                for aq in range(max(1, q-1), min(4, q+2)):  # Start from 1 to ensure dynamics
                    for ad in [0, 1]:  # Limit to d=0 or d=1 to avoid over-differencing
                        if ap == 0 and aq == 0:
                            continue  # Skip (0, d, 0) as it produces flat forecasts
                        candidate_orders.append((ap, ad, aq))
            # Add some common good orders
            candidate_orders.extend([(1, 0, 1), (2, 0, 2), (1, 1, 1), (2, 1, 2)])
            candidate_orders = list({tuple(x) for x in candidate_orders})  # Unique

            # Evaluate by BIC, but also check forecast variance
            best_bic = np.inf
            best_order = (1, 0, 1)  # Default to a simple but dynamic model
            best_forecast_std = 0
            
            for candidate in candidate_orders:
                try:
                    m = ARIMA(series_clean, order=candidate)
                    fit = m.fit(method_kwargs={"warn_convergence": False, "maxiter": 200})
                    if fit.bic < best_bic and np.isfinite(fit.bic):
                        # Check if forecast has variance (not flat)
                        try:
                            test_forecast = fit.get_forecast(steps=min(10, forecast_horizon))
                            forecast_std = test_forecast.predicted_mean.std()
                            # Prefer models with some forecast variance
                            if forecast_std > 1e-6 or best_forecast_std == 0:
                                best_bic = fit.bic
                                best_order = candidate
                                best_forecast_std = forecast_std
                        except:
                            # If forecast check fails, still use BIC
                            if best_forecast_std == 0:
                                best_bic = fit.bic
                                best_order = candidate
                except Exception:
                    continue
            order = best_order

        # Ensure order has dynamics (at least one AR or MA term)
        if order[0] == 0 and order[2] == 0:
            # If (0, d, 0), change to (1, d, 1) to add dynamics
            order = (1, order[1], 1)

        # Seasonality inclusion - use trend='c' to add constant term for dynamics
        if seasonal_order:
            model = SARIMAX(series_clean, order=order, seasonal_order=seasonal_order, trend='c')
        else:
            model = ARIMA(series_clean, order=order)

        # Fit with 'lbfgs' then fallback
        try:
            fitted = model.fit(
                method='lbfgs',
                method_kwargs={"warn_convergence": False, "maxiter": 1000}
            )
        except:
            try:
                fitted = model.fit(
                    method_kwargs={"warn_convergence": False, "maxiter": 1000}
                )
            except:
                fitted = model.fit(method_kwargs={"warn_convergence": False})

        # Inverse transform if differencing was used, so forecast is in-sample scale
        forecast_res = fitted.get_forecast(steps=forecast_horizon)
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        # Future dates
        last_date = series_clean.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series_clean.index)
            if freq is None:
                freq = series_clean.index[-1] - series_clean.index[-2]
                if isinstance(freq, pd.Timedelta):
                    freq = pd.tseries.frequencies.to_offset(freq)
                else:
                    freq = 'D'
            future_dates = pd.date_range(
                start=last_date + (freq if isinstance(freq, pd.DateOffset) else pd.Timedelta(days=1)),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)

        forecast_series = pd.Series(forecast.values, index=future_dates)
        conf_int_series = pd.DataFrame(conf_int, index=future_dates)

        # Check if forecast is flat (low variance) - use multiple criteria
        forecast_std = forecast_series.std()
        forecast_range = forecast_series.max() - forecast_series.min()
        is_flat = (forecast_std < 1e-6) or (forecast_range < 1e-6) or np.allclose(forecast_series, forecast_series.iloc[0], atol=1e-6)
        
        # If forecast is flat, try multiple fallback strategies
        if is_flat:
            fallback_orders = [
                (2, 0, 2),  # AR(2) + MA(2) with no differencing
                (1, 0, 2),  # AR(1) + MA(2)
                (2, 0, 1),  # AR(2) + MA(1)
                (1, 1, 1),  # ARIMA(1,1,1) with differencing
            ]
            
            for fallback_order in fallback_orders:
                try:
                    fallback_model = ARIMA(series_clean, order=fallback_order)
                    fallback_fitted = fallback_model.fit(method_kwargs={"warn_convergence": False, "maxiter": 200})
                    fallback_forecast = fallback_fitted.get_forecast(steps=forecast_horizon)
                    fallback_series = pd.Series(fallback_forecast.predicted_mean.values, index=future_dates)
                    
                    # Check if fallback has variance
                    if fallback_series.std() > 1e-6:
                        forecast_series = fallback_series
                        conf_int_series = pd.DataFrame(fallback_forecast.conf_int(), index=future_dates)
                        break
                except Exception:
                    continue
            
            # If still flat, add a small trend/drift based on recent data
            if forecast_series.std() < 1e-6:
                # Calculate recent trend
                recent_mean = series_clean.iloc[-min(20, len(series_clean)):].mean()
                recent_trend = (series_clean.iloc[-1] - series_clean.iloc[-min(10, len(series_clean))]) / min(10, len(series_clean))
                
                # Apply small trend with mean reversion for returns
                base_value = recent_mean
                trend_per_period = recent_trend * 0.1  # Dampen trend
                
                # Generate forecast with small trend
                forecast_values = []
                for i in range(forecast_horizon):
                    # Mean reversion: gradually return to zero for returns
                    value = base_value + trend_per_period * i
                    value = value * (1 - 0.02 * i / forecast_horizon)  # Mean reversion
                    forecast_values.append(value)
                
                forecast_series = pd.Series(forecast_values, index=future_dates)
                # Update confidence intervals
                std = series_clean.std()
                conf_int_series = pd.DataFrame({
                    'lower': forecast_series - 1.96 * std,
                    'upper': forecast_series + 1.96 * std
                }, index=future_dates)

        return forecast_series, conf_int_series

    except Exception as e:
        print(f"ARIMA forecast error: {e}")
        raise ValueError(f"ARIMA model failed: {e}") from e


def sarima_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    auto_select: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using SARIMA (Seasonal ARIMA) model.

    SARIMA extends ARIMA by adding seasonal components, which can be useful
    for time series with seasonal patterns (e.g., weekly patterns in daily returns).

    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        order: (p, d, q) for ARIMA component
        seasonal_order: (P, D, Q, s) for seasonal component. If None, will try common seasonal patterns
        auto_select: If True, try multiple SARIMA orders to find best fit

    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError as e:
        raise ImportError("statsmodels is required for SARIMA forecasting") from e

    series_clean = series.dropna()
    if len(series_clean) < 50:
        raise ValueError("Insufficient data for SARIMA model (need at least 50 periods)")

    try:
        if (order is None or seasonal_order is None) and auto_select:
            # Use ACF/PACF and ADF for nonseasonal order
            nlags = min(10, max(5, len(series_clean) // 8))
            acf_vals = acf(series_clean, nlags=nlags)
            pacf_vals = pacf(series_clean, nlags=nlags)
            ar_lag = np.where(np.abs(pacf_vals) < 0.25)[0]
            ma_lag = np.where(np.abs(acf_vals) < 0.25)[0]
            p = int(ar_lag[0]) if len(ar_lag) > 1 else 1
            q = int(ma_lag[0]) if len(ma_lag) > 1 else 1
            # Prefer d=0 for returns (usually stationary)
            adf_pvalue = adfuller(series_clean)[1]
            d = 0 if adf_pvalue < 0.05 else 1
            if np.abs(series_clean.mean()) < 0.001 and series_clean.std() > 0:
                d = 0  # Returns are typically stationary

            if order is None:
                # Ensure at least one AR or MA term for dynamics
                p = max(1, p)
                q = max(1, q)
                order = (p, d, q)

            # Seasonality search (default: weekly for business days, else monthly)
            if seasonal_order is None:
                s_candidates = [5, 7, 21]  # business/weekly, weekly, monthly
                best_bic = np.inf
                best_seasonal = (1, 0, 1, s_candidates[0])
                best_forecast_std = 0
                
                for s in s_candidates:
                    # Prefer seasonal orders with AR/MA terms
                    for P in [1, 2]:  # Start from 1 to ensure dynamics
                        for Q in [1, 2]:  # Start from 1 to ensure dynamics
                            for D in [0, 1]:  # Limit seasonal differencing
                                candidate_seasonal = (P, D, Q, s)
                                try:
                                    m = SARIMAX(
                                        series_clean,
                                        order=order,
                                        seasonal_order=candidate_seasonal,
                                        trend='c',  # Add constant term for dynamics
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    fit = m.fit(method_kwargs={"warn_convergence": False, "maxiter": 100})
                                    if fit.bic < best_bic and np.isfinite(fit.bic):
                                        # Check forecast variance
                                        try:
                                            test_forecast = fit.get_forecast(steps=min(10, forecast_horizon))
                                            forecast_std = test_forecast.predicted_mean.std()
                                            if forecast_std > 1e-6 or best_forecast_std == 0:
                                                best_bic = fit.bic
                                                best_seasonal = candidate_seasonal
                                                best_forecast_std = forecast_std
                                        except:
                                            if best_forecast_std == 0:
                                                best_bic = fit.bic
                                                best_seasonal = candidate_seasonal
                                except Exception:
                                    continue
                seasonal_order = best_seasonal

        if order is None:
            order = (1, 0, 1)  # Ensure dynamics
        if seasonal_order is None:
            seasonal_order = (1, 0, 1, 5)  # Ensure seasonal dynamics
        
        # Ensure order has dynamics
        if order[0] == 0 and order[2] == 0:
            order = (1, order[1], 1)
        # Ensure seasonal order has dynamics
        if seasonal_order[0] == 0 and seasonal_order[2] == 0:
            seasonal_order = (1, seasonal_order[1], 1, seasonal_order[3])

        # Fit SARIMA with trend='c' to add constant term for dynamics
        model = SARIMAX(
            series_clean,
            order=order,
            seasonal_order=seasonal_order,
            trend='c',  # Add constant term to prevent flat forecasts
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        try:
            fitted = model.fit(
                method='lbfgs',
                method_kwargs={"warn_convergence": False, "maxiter": 1000},
                disp=False
            )
        except:
            fitted = model.fit(
                method_kwargs={"warn_convergence": False, "maxiter": 1000},
                disp=False
            )

        # Out-of-sample forecast
        forecast_res = fitted.get_forecast(steps=forecast_horizon)
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        last_date = series_clean.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series_clean.index)
            if freq is None and len(series_clean.index) > 1:
                freq = series_clean.index[-1] - series_clean.index[-2]
                if isinstance(freq, pd.Timedelta):
                    freq = pd.tseries.frequencies.to_offset(freq)
                else:
                    freq = 'D'
            future_dates = pd.date_range(
                start=last_date + (freq if isinstance(freq, pd.DateOffset) else pd.Timedelta(days=1)),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)
        forecast_series = pd.Series(forecast.values, index=future_dates)
        conf_int_series = pd.DataFrame(conf_int, index=future_dates)

        # Check if forecast is flat (low variance) - use multiple criteria
        forecast_std = forecast_series.std()
        forecast_range = forecast_series.max() - forecast_series.min()
        is_flat = (forecast_std < 1e-6) or (forecast_range < 1e-6) or np.allclose(forecast_series, forecast_series.iloc[0], atol=1e-6)
        
        # If forecast is flat, try multiple fallback strategies
        if is_flat:
            fallback_configs = [
                ((1, 0, 1), (1, 0, 1, 5)),  # Simple seasonal
                ((2, 0, 2), (1, 0, 1, 5)),  # More AR/MA terms
                ((1, 0, 1), (2, 0, 1, 5)),  # More seasonal AR
                ((1, 1, 1), (1, 0, 1, 5)),  # With differencing
            ]
            
            for fallback_order, fallback_seasonal in fallback_configs:
                try:
                    fallback_model = SARIMAX(
                        series_clean,
                        order=fallback_order,
                        seasonal_order=fallback_seasonal,
                        trend='c',  # Add constant term
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fallback_fitted = fallback_model.fit(method_kwargs={"warn_convergence": False, "maxiter": 200}, disp=False)
                    fallback_forecast = fallback_fitted.get_forecast(steps=forecast_horizon)
                    fallback_series = pd.Series(fallback_forecast.predicted_mean.values, index=future_dates)
                    
                    # Check if fallback has variance
                    if fallback_series.std() > 1e-6:
                        forecast_series = fallback_series
                        conf_int_series = pd.DataFrame(fallback_forecast.conf_int(), index=future_dates)
                        break
                except Exception:
                    continue
            
            # If still flat, add a small trend/drift based on recent data
            if forecast_series.std() < 1e-6:
                # Calculate recent trend
                recent_mean = series_clean.iloc[-min(20, len(series_clean)):].mean()
                recent_trend = (series_clean.iloc[-1] - series_clean.iloc[-min(10, len(series_clean))]) / min(10, len(series_clean))
                
                # Apply small trend with mean reversion for returns
                base_value = recent_mean
                trend_per_period = recent_trend * 0.1  # Dampen trend
                
                # Generate forecast with small trend
                forecast_values = []
                for i in range(forecast_horizon):
                    # Mean reversion: gradually return to zero for returns
                    value = base_value + trend_per_period * i
                    value = value * (1 - 0.02 * i / forecast_horizon)  # Mean reversion
                    forecast_values.append(value)
                
                forecast_series = pd.Series(forecast_values, index=future_dates)
                # Update confidence intervals
                std = series_clean.std()
                conf_int_series = pd.DataFrame({
                    'lower': forecast_series - 1.96 * std,
                    'upper': forecast_series + 1.96 * std
                }, index=future_dates)

        return forecast_series, conf_int_series

    except Exception as e:
        print(f"SARIMA forecast error: {e}")
        raise ValueError(f"SARIMA model failed: {e}") from e


def sarimax_forecast(
    series: pd.Series,
    forecast_horizon: int = 30,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    exog: Optional[pd.DataFrame] = None,
    auto_select: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Forecast using SARIMAX (Seasonal ARIMA with eXogenous variables) model.

    SARIMAX extends SARIMA by allowing exogenous variables. For this implementation,
    we create simple exogenous features from the time series itself (lags, rolling stats).

    Args:
        series: Time series to forecast
        forecast_horizon: Number of periods ahead to forecast
        order: (p, d, q) for ARIMA component
        seasonal_order: (P, D, Q, s) for seasonal component
        exog: Exogenous variables (optional). If None, will create simple features
        auto_select: If True, try multiple SARIMAX orders to find best fit

    Returns:
        Tuple of (forecast, confidence_intervals)
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import acf, pacf, adfuller
    except ImportError as e:
        raise ImportError("statsmodels is required for SARIMAX forecasting") from e

    # Remove NaN values
    series_clean = series.dropna()
    if len(series_clean) < 50:
        raise ValueError("Insufficient data for SARIMAX model (need at least 50 periods)")

    try:
        # Exogenous
        if exog is None:
            exog_features = pd.DataFrame(index=series_clean.index)
            exog_features['rolling_mean_5'] = series_clean.rolling(window=5, min_periods=1).mean()
            exog_features['rolling_std_5'] = series_clean.rolling(window=5, min_periods=1).std()
            exog_features['lag_1'] = series_clean.shift(1)
            exog_features = exog_features.bfill().fillna(0)
            exog = exog_features

        exog_aligned = exog.loc[series_clean.index].bfill().fillna(0)

        # Order selection
        if (order is None or seasonal_order is None) and auto_select:
            nlags = min(8, max(3, len(series_clean) // 10))
            acf_vals = acf(series_clean, nlags=nlags)
            pacf_vals = pacf(series_clean, nlags=nlags)
            ar_lag = np.where(np.abs(pacf_vals) < 0.2)[0]
            ma_lag = np.where(np.abs(acf_vals) < 0.2)[0]
            p = int(ar_lag[0]) if len(ar_lag) > 1 else 1
            q = int(ma_lag[0]) if len(ma_lag) > 1 else 1
            # Prefer d=0 for returns (usually stationary)
            adf_pvalue = adfuller(series_clean)[1]
            d = 0 if adf_pvalue < 0.05 else 1
            if np.abs(series_clean.mean()) < 0.001 and series_clean.std() > 0:
                d = 0  # Returns are typically stationary

            # Ensure at least one AR or MA term for dynamics
            p = max(1, p)
            q = max(1, q)
            
            base_orders = [(p, d, q), (p+1, d, q), (p, d, q+1), (2, d, 2)]
            best_bic = np.inf
            best_order = (p, d, q)
            best_forecast_std = 0
            
            for candidate in base_orders:
                # Skip orders without AR/MA terms
                if candidate[0] == 0 and candidate[2] == 0:
                    continue
                try:
                    sm = SARIMAX(
                        series_clean,
                        exog=exog_aligned,
                        order=candidate,
                        trend='c',  # Add constant term for dynamics
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fit = sm.fit(method_kwargs={"warn_convergence": False, "maxiter": 100}, disp=False)
                    if fit.bic < best_bic and np.isfinite(fit.bic):
                        # Check forecast variance
                        try:
                            test_forecast = fit.get_forecast(steps=min(10, forecast_horizon), exog=exog_aligned.iloc[-min(10, len(exog_aligned)):])
                            forecast_std = test_forecast.predicted_mean.std()
                            if forecast_std > 1e-6 or best_forecast_std == 0:
                                best_bic = fit.bic
                                best_order = candidate
                                best_forecast_std = forecast_std
                        except:
                            if best_forecast_std == 0:
                                best_bic = fit.bic
                                best_order = candidate
                except Exception:
                    continue
            order = best_order

            if seasonal_order is None:
                # Try weekly seasonality with dynamics
                seasonal_order = (1, 0, 1, 5)

        if order is None:
            order = (1, 0, 1)  # Ensure dynamics
        if seasonal_order is None:
            seasonal_order = (1, 0, 1, 5)  # Ensure seasonal dynamics
        
        # Ensure order has dynamics
        if order[0] == 0 and order[2] == 0:
            order = (1, order[1], 1)
        # Ensure seasonal order has dynamics
        if seasonal_order[0] == 0 and seasonal_order[2] == 0:
            seasonal_order = (1, seasonal_order[1], 1, seasonal_order[3])

        model = SARIMAX(
            series_clean,
            exog=exog_aligned,
            order=order,
            seasonal_order=seasonal_order,
            trend='c',  # Add constant term to prevent flat forecasts
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        # Fit
        try:
            fitted = model.fit(
                method='lbfgs',
                method_kwargs={"warn_convergence": False, "maxiter": 1000},
                disp=False
            )
        except:
            fitted = model.fit(
                method_kwargs={"warn_convergence": False, "maxiter": 1000},
                disp=False
            )

        # Forecast exog: propagate last value forward, or keep drift
        last_date = series_clean.index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(series_clean.index)
            if freq is None:
                freq = series_clean.index[-1] - series_clean.index[-2]
                if isinstance(freq, pd.Timedelta):
                    freq = pd.tseries.frequencies.to_offset(freq)
                else:
                    freq = 'D'
            future_dates = pd.date_range(
                start=last_date + (freq if isinstance(freq, pd.DateOffset) else pd.Timedelta(days=1)),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            future_dates = range(len(series_clean), len(series_clean) + forecast_horizon)

        exog_forecast = pd.DataFrame(index=future_dates, columns=exog_aligned.columns)
        for col in exog_aligned.columns:
            exog_forecast[col] = exog_aligned[col].iloc[-1]  # propagate last value

        forecast_res = fitted.get_forecast(steps=forecast_horizon, exog=exog_forecast)
        forecast = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        forecast_series = pd.Series(forecast.values, index=future_dates)
        conf_int_series = pd.DataFrame(conf_int, index=future_dates)

        # Check if forecast is flat (low variance) - use multiple criteria
        forecast_std = forecast_series.std()
        forecast_range = forecast_series.max() - forecast_series.min()
        is_flat = (forecast_std < 1e-6) or (forecast_range < 1e-6) or np.allclose(forecast_series, forecast_series.iloc[0], atol=1e-6)
        
        # If forecast is flat, try multiple fallback strategies
        if is_flat:
            fallback_configs = [
                ((1, 0, 1), (1, 0, 1, 5)),  # Simple seasonal
                ((2, 0, 2), (1, 0, 1, 5)),  # More AR/MA terms
                ((1, 0, 1), (2, 0, 1, 5)),  # More seasonal AR
                ((1, 1, 1), (1, 0, 1, 5)),  # With differencing
            ]
            
            for fallback_order, fallback_seasonal in fallback_configs:
                try:
                    fallback_model = SARIMAX(
                        series_clean,
                        exog=exog_aligned,
                        order=fallback_order,
                        seasonal_order=fallback_seasonal,
                        trend='c',  # Add constant term
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fallback_fitted = fallback_model.fit(method_kwargs={"warn_convergence": False, "maxiter": 200}, disp=False)
                    fallback_forecast = fallback_fitted.get_forecast(steps=forecast_horizon, exog=exog_forecast)
                    fallback_series = pd.Series(fallback_forecast.predicted_mean.values, index=future_dates)
                    
                    # Check if fallback has variance
                    if fallback_series.std() > 1e-6:
                        forecast_series = fallback_series
                        conf_int_series = pd.DataFrame(fallback_forecast.conf_int(), index=future_dates)
                        break
                except Exception:
                    continue
            
            # If still flat, add a small trend/drift based on recent data
            if forecast_series.std() < 1e-6:
                # Calculate recent trend
                recent_mean = series_clean.iloc[-min(20, len(series_clean)):].mean()
                recent_trend = (series_clean.iloc[-1] - series_clean.iloc[-min(10, len(series_clean))]) / min(10, len(series_clean))
                
                # Apply small trend with mean reversion for returns
                base_value = recent_mean
                trend_per_period = recent_trend * 0.1  # Dampen trend
                
                # Generate forecast with small trend
                forecast_values = []
                for i in range(forecast_horizon):
                    # Mean reversion: gradually return to zero for returns
                    value = base_value + trend_per_period * i
                    value = value * (1 - 0.02 * i / forecast_horizon)  # Mean reversion
                    forecast_values.append(value)
                
                forecast_series = pd.Series(forecast_values, index=future_dates)
                # Update confidence intervals
                std = series_clean.std()
                conf_int_series = pd.DataFrame({
                    'lower': forecast_series - 1.96 * std,
                    'upper': forecast_series + 1.96 * std
                }, index=future_dates)

        return forecast_series, conf_int_series

    except Exception as e:
        print(f"SARIMAX forecast error: {e}")
        raise ValueError(f"SARIMAX model failed: {e}") from e


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
    batch_size: int = 32,
    use_cache: bool = True,
    ticker: str = ""
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

    # Prepare training data
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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
    
    # Clean and validate input data
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache for existing model
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
    
    # Normalize data using MinMaxScaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series_clean)
    else:
        scaled_data = scaler.fit_transform(series_clean)
    
    # Prepare training data: create sequences of lookback_window length
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build TCN model with dilated causal convolutions
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
        
        # Build TCN blocks with increasing dilation rates (exponential: 1, 2, 4, 8, ...)
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            x = tcn_block(x, num_filters, kernel_size, dilation_rate, i)
        
        # Final layers for output
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
        
        # Save to cache if enabled
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
    
    # Forecast: use recursive prediction
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
    
    # Inverse transform to get original scale
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts).flatten()
    
    # Create future dates for the forecast
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
    
    # Clean and validate input data
    series_clean = series.dropna()
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache for existing model
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
    
    # Create features from lagged values and rolling statistics
    def create_features(data, window):
        """Create features from lagged values and rolling statistics."""
        features = []
        targets = []
        
        for i in range(window, len(data)):
            # Lagged values (past window periods)
            lag_features = data.iloc[i-window:i].values
            
            # Rolling statistics for additional context
            rolling_mean = data.iloc[i-window:i].mean()
            rolling_std = data.iloc[i-window:i].std()
            rolling_max = data.iloc[i-window:i].max()
            rolling_min = data.iloc[i-window:i].min()
            
            # Combine all features
            feature_vector = np.concatenate([
                lag_features,
                [rolling_mean, rolling_std, rolling_max, rolling_min]
            ])
            
            features.append(feature_vector)
            targets.append(data.iloc[i])
        
        return np.array(features), np.array(targets)
    
    # Prepare training data
    X_train, y_train = create_features(series_clean, lookback_window)
    
    # Normalize features using StandardScaler
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
        
        # Save to cache if enabled
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
    
    # Forecast recursively: predict one step ahead, then use that prediction for the next
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
        
        # Update last values (shift and append new prediction)
        last_values = np.append(last_values[1:], next_pred)
    
    # Create future dates for the forecast
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
    
    # Calculate confidence intervals based on historical residuals
    train_pred = model.predict(X_train_scaled)
    residuals = y_train - train_pred
    std_residual = np.std(residuals)
    
    # 95% confidence interval (1.96 standard deviations)
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
    
    # Clean and validate input data
    series_clean = series.dropna().values.reshape(-1, 1)
    
    if len(series_clean) < lookback_window + forecast_horizon:
        raise ValueError(f"Insufficient data. Need at least {lookback_window + forecast_horizon} periods")
    
    # Check cache for existing model
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
    
    # Normalize data using MinMaxScaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series_clean)
    else:
        scaled_data = scaler.fit_transform(series_clean)
    
    # Prepare training data: create sequences of lookback_window length
    X_train, y_train = [], []
    for i in range(lookback_window, len(scaled_data)):
        X_train.append(scaled_data[i-lookback_window:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build and train model if not cached
    if model is None:
        # Build Transformer model with self-attention mechanism
        def transformer_block(x, d_model, num_heads, name_prefix):
            """Create a transformer block with self-attention."""
            # Self-attention layer
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model,
                name=f'{name_prefix}_attention'
            )(x, x)
            attn_output = Dropout(0.1)(attn_output)
            
            # Add & Norm (residual connection and layer normalization)
            x = Add()([x, attn_output])
            x = LayerNormalization(name=f'{name_prefix}_norm1')(x)
            
            # Feed forward network
            ff_output = Dense(d_model * 2, activation='relu')(x)
            ff_output = Dense(d_model)(ff_output)
            ff_output = Dropout(0.1)(ff_output)
            
            # Add & Norm (second residual connection)
            x = Add()([x, ff_output])
            x = LayerNormalization(name=f'{name_prefix}_norm2')(x)
            
            return x
        
        # Input layer
        inputs = Input(shape=(lookback_window, 1))
        
        # Project to d_model dimensions using 1D convolution
        x = Conv1D(filters=d_model, kernel_size=1, padding='same')(inputs)
        
        # Stack transformer blocks
        for i in range(num_layers):
            x = transformer_block(x, d_model, num_heads, f'block_{i}')
        
        # Global pooling and output layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(d_model, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Save to cache if enabled
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
    
    # Forecast: use recursive prediction
    last_sequence = scaled_data[-lookback_window:].reshape(1, lookback_window, 1)
    forecasts = []
    
    for _ in range(forecast_horizon):
        next_pred = model.predict(last_sequence, verbose=0)
        forecasts.append(next_pred[0, 0])
        # Update sequence for next prediction
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform to get original scale
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts).flatten()
    
    # Create future dates for the forecast
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
    # Clean input data
    series_clean = series.dropna()
    
    # Calculate recent mean (used as starting point for forecast)
    if len(series_clean) < window:
        # Use all available data if window is larger than data
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
    
    # Create future dates for the forecast
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
    # 95% confidence interval (1.96 standard deviations)
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
    
    # Clean and validate input data
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        raise ValueError("Insufficient data for exponential smoothing")
    
    try:
        # Fit exponential smoothing model
        model = ExponentialSmoothing(
            series_clean,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_horizon)
        
        # Create future dates for the forecast
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
        
        # Simple confidence interval based on historical volatility
        std = series_clean.std()
        # 95% confidence interval (1.96 standard deviations)
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
    use_cache: bool = True,
    ticker: str = "",
    **kwargs
) -> Dict[str, pd.Series]:
    """
    Forecast portfolio returns using specified method.

    Args:
        portfolio_returns: Historical portfolio returns
        method: Forecasting method ('arima', 'sarima', 'sarimax', 'prophet', 'lstm', 'tcn', 'xgboost', 'transformer', 'ma', 'exponential_smoothing')
        forecast_horizon: Number of periods to forecast
        use_cache: Whether to use cached models if available
        ticker: Ticker symbol for caching (optional but recommended)
        **kwargs: Additional arguments for specific methods

    Returns:
        Dictionary with 'forecast' and 'confidence_intervals' (if available)
    """
    filtered_kwargs = kwargs.copy()
    filtered_kwargs.pop('use_cache', None)
    filtered_kwargs.pop('ticker', None)
    if method == 'xgboost' and 'epochs' in filtered_kwargs:
        filtered_kwargs.pop('epochs')

    if method == 'arima':
        forecast, conf_int = arima_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
    elif method == 'sarima':
        forecast, conf_int = sarima_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
    elif method == 'sarimax':
        forecast, conf_int = sarimax_forecast(portfolio_returns, forecast_horizon, **filtered_kwargs)
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

    for method in methods:
        if method == 'ma':
            continue

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

    forecast_df = pd.concat(forecast_dfs, axis=1)
    forecast_df = forecast_df.fillna(method='ffill').fillna(method='bfill')

    if aggregation == 'mean':
        ensemble = forecast_df.mean(axis=1)
    elif aggregation == 'median':
        ensemble = forecast_df.median(axis=1)
    elif aggregation == 'weighted':
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

