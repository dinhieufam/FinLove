import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field

from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Ensure we load the bundled engine code inside this web/backend folder.
# Add the backend root to sys.path so `src` is importable as a package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.data import prepare_portfolio_data, get_available_tickers  # type: ignore
from src.forecast import forecast_portfolio_returns, ensemble_forecast  # type: ignore
from src.risk import get_covariance_matrix  # type: ignore
from src.optimize import optimize_portfolio  # type: ignore

router = APIRouter()

class PredictionRequest(BaseModel):
    """
    Request payload for portfolio prediction.
    """
    tickers: List[str] = Field(
        ...,
        description="List of ticker symbols to predict.",
        min_items=1
    )
    start_date: str = Field(
        ...,
        description="Start date for historical data (YYYY-MM-DD).",
        examples=["2015-01-01"]
    )
    end_date: str = Field(
        ...,
        description="End date for historical data (YYYY-MM-DD).",
        examples=["2024-12-31"]
    )
    forecast_horizon: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to forecast."
    )
    optimization_method: Literal["markowitz", "min_variance", "sharpe", "black_litterman", "cvar"] = Field(
        default="markowitz",
        description="Portfolio optimization method."
    )
    risk_model: Literal["ledoit_wolf", "sample", "glasso", "garch"] = Field(
        default="ledoit_wolf",
        description="Risk model for covariance estimation."
    )
    risk_aversion: float = Field(
        default=1.0,
        ge=0.01,
        le=20.0,
        description="Risk aversion parameter for Markowitz-style optimizations."
    )
    # forecast_method is ignored - always uses SARIMAX with fixed parameters (p=1,d=1,q=1,P=1,D=1,Q=1,seasonal_period=5)
    forecast_method: Optional[str] = Field(
        default=None,
        description="Deprecated: Always uses SARIMAX. This field is ignored."
    )

def _series_to_payload(series) -> Dict[str, Any]:
    """Convert a pandas Series to a JSON-serializable payload."""
    if series is None or getattr(series, "empty", False):
        return {"dates": [], "values": []}
    
    # Handle numpy types
    values = [float(v) if v is not None else None for v in series.values]
    dates = [str(idx) for idx in series.index]
    
    return {
        "dates": dates,
        "values": values,
    }

@router.post("/predict", summary="Run portfolio prediction")
async def predict_portfolio(payload: PredictionRequest) -> dict:
    """
    Generate portfolio predictions using the configured portfolio strategies.
    """
    try:
        # Get available tickers for validation
        available_tickers = set(get_available_tickers())
        
        # Validate that all requested tickers are available
        invalid_tickers = [t for t in payload.tickers if t.upper() not in available_tickers]
        if invalid_tickers:
            return {
                "ok": False,
                "error": f"Invalid tickers: {', '.join(invalid_tickers)}. Available tickers: {', '.join(sorted(available_tickers))}",
                "available_tickers": sorted(available_tickers)
            }
        
        # Normalize tickers to uppercase
        normalized_tickers = [t.upper() for t in payload.tickers]
        
        # Get historical data
        returns, _ = prepare_portfolio_data(
            normalized_tickers,
            start_date=payload.start_date,
            end_date=payload.end_date
        )
        
        if returns.empty or len(returns) < 60:
            return {
                "ok": False,
                "error": "Insufficient data for prediction."
            }
        
        # Clean returns
        returns = returns.dropna(axis=1, thresh=len(returns) * 0.8)
        if returns.empty:
            return {
                "ok": False,
                "error": "No valid assets after cleaning."
            }
        
        # Get covariance matrix using the configured risk model
        covariance = get_covariance_matrix(returns, method=payload.risk_model)
        
        # Optimize portfolio using the configured optimization method
        weights = optimize_portfolio(
            returns,
            covariance,
            method=payload.optimization_method,
            risk_aversion=payload.risk_aversion,
            constraints={"long_only": True}
        )
        
        # Calculate portfolio returns using optimized weights
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Filter portfolio returns to match the exact date range (start_date to end_date)
        # This ensures we only use data from the specified period
        start_dt = pd.to_datetime(payload.start_date)
        end_dt = pd.to_datetime(payload.end_date)
        portfolio_returns = portfolio_returns[(portfolio_returns.index >= start_dt) & (portfolio_returns.index <= end_dt)]
        
        if portfolio_returns.empty:
            return {
                "ok": False,
                "error": "No data available for the specified date range."
            }
        
        # Get full historical cumulative returns for chart (from start_date to end_date)
        historical_cumulative = (1 + portfolio_returns).cumprod()
        
        # Forecast future portfolio returns using SARIMAX with fixed parameters
        # Order: (p=1, d=1, q=1), Seasonal: (P=1, D=1, Q=1, seasonal_period=5)
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        try:
            # Fit SARIMAX model with specified parameters
            model = SARIMAX(
                portfolio_returns,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 5),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            
            # Forecast
            forecast = fitted.forecast(steps=payload.forecast_horizon)
            
            # Create forecast dates starting from day after end_date
            forecast_start = end_dt + pd.Timedelta(days=1)
            forecast_dates = pd.date_range(
                start=forecast_start,
                periods=payload.forecast_horizon,
                freq='D'
            )
            
            forecast_returns = pd.Series(forecast.values, index=forecast_dates)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "ok": False,
                "error": f"SARIMAX forecast failed: {str(e)}"
            }
        
        # Ensure forecast dates start from the day after end_date
        # The forecast functions should already do this, but let's verify
        if forecast_returns.index[0] <= end_dt:
            # If forecast starts before or on end_date, adjust it
            forecast_start = end_dt + pd.Timedelta(days=1)
            forecast_dates = pd.date_range(
                start=forecast_start,
                periods=payload.forecast_horizon,
                freq='D'
            )
            forecast_returns.index = forecast_dates[:len(forecast_returns)]
        
        # Calculate cumulative forecast starting from the last historical value
        last_hist_val = historical_cumulative.iloc[-1] if not historical_cumulative.empty else 1.0
        forecast_cumulative = (1 + forecast_returns).cumprod() * last_hist_val
        
        return {
            "ok": True,
            "mode": "portfolio",
            "series": {
                "historical": _series_to_payload(historical_cumulative),
                "forecast": _series_to_payload(forecast_cumulative)
            },
            "metrics": {
                "expected_daily_return": float(forecast_returns.mean()),
                "forecast_volatility": float(forecast_returns.std() * np.sqrt(252))
            },
            "forecast_horizon": payload.forecast_horizon,
            "optimization_method": payload.optimization_method,
            "risk_model": payload.risk_model,
            "start_date": payload.start_date,
            "end_date": payload.end_date
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e)
        }
