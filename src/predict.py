"""
Main Prediction Module.

This module combines model collection and forecasting to provide
end-to-end prediction capabilities.
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .forecast import ensemble_forecast, forecast_portfolio_returns
from .model_collector import (
    aggregate_model_predictions,
    collect_all_model_results,
    get_best_models,
    prepare_forecasting_data
)

# Suppress warnings
warnings.filterwarnings('ignore')


# Optional: Check for forecasting dependencies
def check_forecasting_dependencies() -> Dict[str, bool]:
    """
    Check which forecasting dependencies are available.

    Returns:
        Dictionary with availability status for each forecasting method.
    """
    dependencies = {
        'arima': False,
        'prophet': False,
        'lstm': False,
        'exponential_smoothing': False
    }
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        dependencies['arima'] = True
        dependencies['exponential_smoothing'] = True
    except ImportError:
        pass
    
    try:
        from prophet import Prophet
        dependencies['prophet'] = True
    except ImportError:
        pass
    
    try:
        from tensorflow import keras
        dependencies['lstm'] = True
    except ImportError:
        pass
    
    return dependencies


def print_dependency_status() -> None:
    """Print status of forecasting dependencies."""
    deps = check_forecasting_dependencies()
    print("\n📦 Forecasting Dependencies Status:")
    print("=" * 50)
    for method, available in deps.items():
        status = "✅ Available" if available else "❌ Not installed"
        print(f"  {method:20s}: {status}")
    
    if not all(deps.values()):
        print("\n💡 Tip: Install missing dependencies for better forecasting:")
        if not deps['arima']:
            print("   pip install statsmodels")
        if not deps['prophet']:
            print("   pip install prophet")
        if not deps['lstm']:
            print("   pip install tensorflow")
    print()


def predict_future_performance(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    forecast_horizon: int = 30,
    forecast_method: str = 'ensemble',
    use_top_models: int = 5,
    backtest_type: str = 'walk_forward',
    **backtest_kwargs: Any
) -> Dict[str, Any]:
    """
    Complete pipeline: collect model results and predict future performance.
    
    This function:
    1. Runs all model combinations
    2. Selects top performing models
    3. Forecasts future returns for each top model
    4. Aggregates predictions
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        forecast_horizon: Number of days to forecast
        forecast_method: Forecasting method ('arima', 'prophet', 'lstm', 'ma', 'ensemble')
        use_top_models: Number of top models to use for prediction
        backtest_type: 'walk_forward' or 'simple'
        **backtest_kwargs: Additional arguments for backtesting
    
    Returns:
        Dictionary containing:
        - model_results: DataFrame with all model results
        - top_models: DataFrame with top N models
        - predictions: Dictionary mapping model_id to forecast
        - aggregated_prediction: Combined prediction from all top models
        - forecast_dates: Future dates for predictions
    """
    print("=" * 60)
    print("🚀 PREDICTION PIPELINE")
    print("=" * 60)
    
    # Check dependencies (optional, informative only)
    if forecast_method in ['arima', 'prophet', 'lstm', 'exponential_smoothing']:
        deps = check_forecasting_dependencies()
        if not deps.get(forecast_method, False):
            print(f"\n⚠️  Warning: {forecast_method} requires additional dependencies.")
            print("   Falling back to 'ma' (moving average) method.")
            print("   Install with: pip install statsmodels (for arima) or pip install prophet")
            forecast_method = 'ma'  # Fallback to moving average
    
    # Step 1: Collect results from all models
    print("\n📊 Step 1: Collecting results from all model combinations...")
    model_results = collect_all_model_results(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        backtest_type=backtest_type,
        **backtest_kwargs
    )
    
    # Step 2: Select top models
    print(f"\n🏆 Step 2: Selecting top {use_top_models} models...")
    top_models = get_best_models(model_results, metric='sharpe_ratio', top_n=use_top_models)
    
    print("\nTop models:")
    for idx, row in top_models.iterrows():
        print(f"  {row['model_id']:30s} | Sharpe: {row['sharpe_ratio']:.3f} | "
              f"Return: {row['annualized_return']*100:.2f}% | "
              f"Vol: {row['annualized_volatility']*100:.2f}%")
    
    # Step 3: Forecast for each top model
    print(f"\n🔮 Step 3: Forecasting future returns ({forecast_horizon} days)...")
    predictions = {}
    
    for idx, row in top_models.iterrows():
        model_id = row['model_id']
        portfolio_returns = row['portfolio_returns']
        
        try:
            if forecast_method == 'ensemble':
                result = ensemble_forecast(
                    portfolio_returns,
                    methods=['arima', 'prophet', 'ma'],
                    forecast_horizon=forecast_horizon
                )
                predictions[model_id] = result['ensemble_forecast']
            else:
                result = forecast_portfolio_returns(
                    portfolio_returns,
                    method=forecast_method,
                    forecast_horizon=forecast_horizon
                )
                predictions[model_id] = result['forecast']
            
            print(f"  ✅ {model_id}")
            
        except Exception as e:
            print(f"  ⚠️  {model_id}: {e}")
            continue
    
    if not predictions:
        raise ValueError("No successful forecasts generated")
    
    # Step 4: Aggregate predictions
    print("\n📈 Step 4: Aggregating predictions...")
    aggregated_prediction = aggregate_model_predictions(
        predictions,
        method='mean'
    )
    
    # Extract forecast dates
    forecast_dates = aggregated_prediction.index
    
    print(f"\n✅ Prediction complete!")
    print(f"   Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
    print(f"   Expected daily return: {aggregated_prediction.mean()*100:.4f}%")
    print(f"   Forecast volatility: {aggregated_prediction.std()*100:.4f}%")
    
    return {
        'model_results': model_results,
        'top_models': top_models,
        'predictions': predictions,
        'aggregated_prediction': aggregated_prediction,
        'forecast_dates': forecast_dates,
        'forecast_horizon': forecast_horizon
    }


def predict_portfolio_weights(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    forecast_horizon: int = 30,
    optimization_method: str = 'markowitz',
    risk_model: str = 'ledoit_wolf',
    **optimization_kwargs
) -> pd.DataFrame:
    """
    Predict future portfolio weights by forecasting asset returns and re-optimizing.
    
    This is a different approach: forecast individual asset returns, then optimize
    portfolio weights based on predicted returns.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        forecast_horizon: Number of days to forecast
        optimization_method: Optimization method to use
        risk_model: Risk model to use
        **optimization_kwargs: Additional optimization arguments
    
    Returns:
        DataFrame with predicted weights over time
    """
    from .data import prepare_portfolio_data
    from .risk import get_covariance_matrix
    from .optimize import optimize_portfolio
    
    # Get historical data
    returns, prices = prepare_portfolio_data(
        tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    # Forecast returns for each asset
    asset_forecasts = {}
    for ticker in tickers:
        asset_returns = returns[ticker]
        try:
            result = ensemble_forecast(
                asset_returns,
                methods=['arima', 'ma'],
                forecast_horizon=forecast_horizon
            )
            asset_forecasts[ticker] = result['ensemble_forecast']
        except Exception as e:
            print(f"Warning: Could not forecast {ticker}: {e}")
            # Use historical mean as fallback
            asset_forecasts[ticker] = pd.Series(
                [asset_returns.mean()] * forecast_horizon,
                index=pd.date_range(
                    start=returns.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_horizon,
                    freq='D'
                )
            )
    
    # Combine forecasts
    forecasted_returns = pd.DataFrame(asset_forecasts)
    
    # Estimate covariance from historical data
    covariance = get_covariance_matrix(returns, method=risk_model)
    
    # Optimize portfolio based on forecasted returns
    # Use mean of forecasted returns as expected returns
    expected_returns = forecasted_returns.mean() * 252  # Annualized
    
    weights = optimize_portfolio(
        forecasted_returns,
        covariance,
        method=optimization_method,
        **optimization_kwargs
    )
    
    # Create DataFrame with predicted weights
    weights_df = pd.DataFrame(
        [weights] * forecast_horizon,
        index=forecasted_returns.index,
        columns=tickers
    )
    
    return weights_df

