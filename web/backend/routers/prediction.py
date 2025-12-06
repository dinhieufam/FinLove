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
    model: Literal["all", "ensemble", "arima", "prophet", "lstm", "tcn", "xgboost", "transformer", "ma", "exponential_smoothing"] = Field(
        default="all",
        description="Forecasting method. Use 'all' to run LSTM, TCN, XGBoost, and Transformer for comparison."
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
        # If model is 'all', we run a specific set of advanced models for each ticker
        if payload.model == 'all':
            models_to_run = ['lstm', 'tcn', 'xgboost', 'transformer']
            all_predictions = {}
            
            # Get historical data for all tickers first
            returns, _ = prepare_portfolio_data(
                payload.tickers,
                start_date=payload.start_date,
                end_date=payload.end_date
            )

            for ticker in payload.tickers:
                if ticker not in returns.columns:
                    continue
                    
                asset_series = returns[ticker].dropna()
                if len(asset_series) < 60: # Minimum history check
                    continue

                all_predictions[ticker] = {}
                
                # Run each model
                for model_name in models_to_run:
                    try:
                        # We use forecast_portfolio_returns but pass a single asset series
                        # It handles Series input correctly
                        from src.forecast import forecast_portfolio_returns
                        
                        result = forecast_portfolio_returns(
                            asset_series,
                            method=model_name,
                            forecast_horizon=payload.forecast_horizon,
                            lookback_window=60, # Default from Streamlit
                            epochs=50 # Default from Streamlit
                        )
                        
                        forecast_series = result['forecast']
                        
                        # Calculate Volatility Outlook (rolling std of forecast)
                        # For a single path forecast, this is just 0 or undefined if we don't have multiple paths
                        # But typically we want to compare the volatility of the forecast period vs history
                        # Or if the model returns a distribution.
                        # The Streamlit app calculates rolling vol of the *forecasted series* itself
                        forecast_vol = forecast_series.rolling(window=10).std() * np.sqrt(252)
                        
                        all_predictions[ticker][model_name] = {
                            "forecast": _series_to_payload(forecast_series),
                            "volatility": _series_to_payload(forecast_vol),
                            "metrics": {
                                "cumulative_return": float((1 + forecast_series).prod() - 1),
                                "volatility": float(forecast_series.std() * np.sqrt(252)),
                                "mean_return": float(forecast_series.mean() * 252)
                            }
                        }
                    except Exception as e:
                        print(f"Error running {model_name} for {ticker}: {e}")
                        all_predictions[ticker][model_name] = None

            return {
                "ok": True,
                "mode": "all",
                "predictions": all_predictions,
                "forecast_horizon": payload.forecast_horizon
            }

        else:
            # Original single/ensemble logic for PORTFOLIO level
            # ... (keep existing logic if needed, or deprecate)
            # For now, let's keep it but maybe the user wants per-asset for everything?
            # The user request specifically mentioned "future cumulative returns of all models... for each company"
            # So the 'all' mode above is what they want.
            
            # Fallback to original logic if specific model requested
            results = predict_future_performance(
                tickers=payload.tickers,
                start_date=payload.start_date,
                end_date=payload.end_date,
                forecast_horizon=payload.forecast_horizon,
                forecast_method=payload.model,
                use_top_models=payload.use_top_models,
                backtest_type='walk_forward',
                train_window=36
            )
            
            # ... (rest of original logic) ...
            aggregated_prediction = results['aggregated_prediction']
            top_models_df = results['top_models']
            
            if not top_models_df.empty:
                best_model = top_models_df.iloc[0]
                historical_returns = best_model['portfolio_returns']
                historical_cumulative = (1 + historical_returns).cumprod()
                recent_history = historical_cumulative.iloc[-90:]
            else:
                recent_history = pd.Series()

            last_hist_val = recent_history.iloc[-1] if not recent_history.empty else 1.0
            forecast_cumulative = (1 + aggregated_prediction).cumprod() * last_hist_val
            
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
                "mode": "portfolio",
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
                "forecast_method": payload.model
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e)
        }
