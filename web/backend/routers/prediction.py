import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Ensure we load the bundled engine code inside this web/backend folder.
# Add the backend root to sys.path so `src` is importable as a package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.predict import predict_future_performance  # type: ignore
from src.data import prepare_portfolio_data  # type: ignore

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
    model: Literal["ensemble", "arima", "prophet", "lstm", "tcn", "xgboost", "transformer", "ma", "exponential_smoothing"] = Field(
        default="ensemble",
        description="Forecasting method to use. Available: ensemble (combines multiple), arima, prophet, lstm, tcn, xgboost, transformer, ma (moving average), exponential_smoothing."
    )
    use_top_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top performing models to use for ensemble."
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
    Generate portfolio predictions using the selected model.
    """
    try:
        # 1. Run prediction pipeline
        # Note: This is a synchronous call and might take time.
        # In a production app, this should be offloaded to a background task (Celery/Redis).
        results = predict_future_performance(
            tickers=payload.tickers,
            start_date=payload.start_date,
            end_date=payload.end_date,
            forecast_horizon=payload.forecast_horizon,
            forecast_method=payload.model,
            use_top_models=payload.use_top_models,
            backtest_type='walk_forward',
            train_window=36 # Default training window
        )
        
        # 2. Extract results
        aggregated_prediction = results['aggregated_prediction']
        top_models_df = results['top_models']
        
        # 3. Get historical data for context (last 90 days)
        # We use the first top model's portfolio returns as the "historical" reference
        if not top_models_df.empty:
            best_model = top_models_df.iloc[0]
            historical_returns = best_model['portfolio_returns']
            # Calculate cumulative returns for history
            historical_cumulative = (1 + historical_returns).cumprod()
            # Normalize to start at 0% change or 1.0? Let's send raw cumulative values
            # But we want to stitch them.
            
            # Let's send the last 90 days of cumulative history
            recent_history = historical_cumulative.iloc[-90:]
        else:
            recent_history = pd.Series()

        # 4. Prepare forecast series
        # Forecast is daily returns, we need cumulative path starting from last historical point
        last_hist_val = recent_history.iloc[-1] if not recent_history.empty else 1.0
        forecast_cumulative = (1 + aggregated_prediction).cumprod() * last_hist_val
        
        # 5. Format top models list
        top_models_list = []
        for _, row in top_models_df.iterrows():
            top_models_list.append({
                "model_id": row['model_id'],
                "sharpe_ratio": float(row['sharpe_ratio']),
                "annualized_return": float(row['annualized_return']),
                "annualized_volatility": float(row['annualized_volatility'])
            })

        return {
            "ok": True,
            "series": {
                "historical": _series_to_payload(recent_history),
                "forecast": _series_to_payload(forecast_cumulative)
            },
            "metrics": {
                "expected_daily_return": float(aggregated_prediction.mean()),
                "forecast_volatility": float(aggregated_prediction.std())
            },
            "top_models": top_models_list,
            "forecast_horizon": payload.forecast_horizon,
            "forecast_method": payload.model  # Include which forecasting method was used
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e)
        }
