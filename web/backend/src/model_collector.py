"""
Model Results Collector Module.

This module collects results from all optimization method and risk model combinations,
preparing data for time series forecasting.
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .backtest import simple_backtest, walk_forward_backtest
from .data import prepare_portfolio_data
from .metrics import calculate_all_metrics

# Suppress warnings
warnings.filterwarnings('ignore')


# Available optimization methods
OPTIMIZATION_METHODS = ['markowitz', 'min_variance', 'sharpe', 'black_litterman', 'cvar']

# Available risk models
RISK_MODELS = ['sample', 'ledoit_wolf', 'glasso', 'garch']


def collect_all_model_results(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    backtest_type: str = 'walk_forward',
    train_window: int = 36,
    test_window: int = 1,
    transaction_cost: float = 0.001,
    rebalance_band: float = 0.05,
    risk_aversion: float = 1.0,
    constraints: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Collect results from all combinations of optimization methods and risk models.
    
    This function runs backtests for every combination and collects:
    - Portfolio returns (time series)
    - Weights history (time series)
    - Performance metrics (scalars)
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        backtest_type: 'walk_forward' or 'simple'
        train_window: Training window in months (for walk-forward)
        test_window: Test window in months (for walk-forward)
        transaction_cost: Transaction cost
        rebalance_band: Rebalance band threshold
        risk_aversion: Risk aversion parameter
        constraints: Optimization constraints
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame with columns:
        - model_id: Unique identifier (optimization_method_risk_model)
        - optimization_method: Optimization method used
        - risk_model: Risk model used
        - portfolio_returns: Series of portfolio returns
        - weights_history: DataFrame of weights over time
        - metrics: Dictionary of performance metrics
        - sharpe_ratio: Sharpe ratio (for easy filtering)
        - annualized_return: Annualized return
        - annualized_volatility: Annualized volatility
        - max_drawdown: Maximum drawdown
    """
    # Prepare data once
    print("📊 Preparing portfolio data...")
    returns, prices = prepare_portfolio_data(
        tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    if returns.empty:
        raise ValueError("No data available for the specified tickers and date range")
    
    # Filter out assets with too many missing values
    returns = returns.dropna(axis=1, thresh=len(returns) * 0.8)
    tickers = list(returns.columns)
    
    print(f"✅ Data prepared: {len(returns)} days, {len(tickers)} assets")
    print(f"📈 Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # Collect results
    results = []
    total_combinations = len(OPTIMIZATION_METHODS) * len(RISK_MODELS)
    
    print(f"\n🔄 Running {total_combinations} model combinations...")
    
    for opt_method in tqdm(OPTIMIZATION_METHODS, desc="Optimization methods"):
        for risk_model in RISK_MODELS:
            model_id = f"{opt_method}_{risk_model}"
            
            try:
                if backtest_type == 'walk_forward':
                    portfolio_returns, weights_history, metrics = walk_forward_backtest(
                        returns,
                        train_window=train_window,
                        test_window=test_window,
                        optimization_method=opt_method,
                        risk_model=risk_model,
                        transaction_cost=transaction_cost,
                        rebalance_band=rebalance_band,
                        rebalance_frequency='monthly',
                        constraints=constraints or {'long_only': True},
                        risk_aversion=risk_aversion
                    )
                else:  # simple
                    portfolio_returns, weights, metrics = simple_backtest(
                        returns,
                        optimization_method=opt_method,
                        risk_model=risk_model,
                        transaction_cost=transaction_cost,
                        constraints=constraints or {'long_only': True},
                        risk_aversion=risk_aversion
                    )
                    # Convert weights to DataFrame for consistency
                    weights_history = pd.DataFrame(
                        [weights],
                        index=[returns.index[-1]],
                        columns=tickers
                    )
                
                # Store results
                result = {
                    'model_id': model_id,
                    'optimization_method': opt_method,
                    'risk_model': risk_model,
                    'portfolio_returns': portfolio_returns,
                    'weights_history': weights_history,
                    'metrics': metrics,
                    'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
                    'annualized_return': metrics.get('annualized_return', np.nan),
                    'annualized_volatility': metrics.get('annualized_volatility', np.nan),
                    'max_drawdown': metrics.get('max_drawdown', np.nan),
                    'total_return': metrics.get('total_return', np.nan),
                    'var_95': metrics.get('var_95', np.nan),
                    'cvar_95': metrics.get('cvar_95', np.nan),
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️  Error with {model_id}: {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        raise ValueError("No successful model runs. Check your inputs and data.")
    
    print(f"\n✅ Collected results from {len(results_df)} model combinations")
    
    return results_df


def get_best_models(
    results_df: pd.DataFrame,
    metric: str = 'sharpe_ratio',
    top_n: int = 5
) -> pd.DataFrame:
    """
    Get the top N models based on a performance metric.
    
    Args:
        results_df: DataFrame from collect_all_model_results
        metric: Metric to rank by ('sharpe_ratio', 'annualized_return', etc.)
        top_n: Number of top models to return
    
    Returns:
        DataFrame with top N models
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    # Sort by metric (descending)
    sorted_df = results_df.sort_values(metric, ascending=False)
    
    return sorted_df.head(top_n)


def prepare_forecasting_data(
    results_df: pd.DataFrame,
    target: str = 'portfolio_returns',
    model_ids: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Prepare time series data for forecasting from model results.
    
    Args:
        results_df: DataFrame from collect_all_model_results
        target: What to forecast ('portfolio_returns', 'weights', or 'metrics')
        model_ids: List of model IDs to include (None = all)
    
    Returns:
        Dictionary mapping model_id to DataFrame ready for forecasting
    """
    forecasting_data = {}
    
    if model_ids is None:
        model_ids = results_df['model_id'].tolist()
    
    for model_id in model_ids:
        model_row = results_df[results_df['model_id'] == model_id].iloc[0]
        
        if target == 'portfolio_returns':
            # Extract portfolio returns time series
            returns_series = model_row['portfolio_returns']
            if isinstance(returns_series, pd.Series):
                forecasting_data[model_id] = returns_series.to_frame(name='returns')
            else:
                continue
                
        elif target == 'weights':
            # Extract weights history
            weights_df = model_row['weights_history']
            if isinstance(weights_df, pd.DataFrame):
                forecasting_data[model_id] = weights_df
            else:
                continue
                
        elif target == 'metrics':
            # Extract metrics as time series (would need rolling calculation)
            # For now, return metrics as single values
            metrics = model_row['metrics']
            metrics_series = pd.Series(metrics)
            forecasting_data[model_id] = metrics_series.to_frame(name='value')
        else:
            raise ValueError(f"Unknown target: {target}")
    
    return forecasting_data


def aggregate_model_predictions(
    predictions_dict: Dict[str, pd.Series],
    method: str = 'mean'
) -> pd.Series:
    """
    Aggregate predictions from multiple models.
    
    Args:
        predictions_dict: Dictionary mapping model_id to prediction Series
        method: Aggregation method ('mean', 'median', 'weighted_mean')
    
    Returns:
        Aggregated prediction Series
    """
    if not predictions_dict:
        raise ValueError("No predictions provided")
    
    # Align all series by index
    all_series = []
    for model_id, pred in predictions_dict.items():
        if isinstance(pred, pd.Series):
            all_series.append(pred)
    
    if not all_series:
        raise ValueError("No valid predictions found")
    
    # Combine into DataFrame
    pred_df = pd.concat(all_series, axis=1)
    pred_df.columns = list(predictions_dict.keys())[:len(pred_df.columns)]
    
    # Aggregate
    if method == 'mean':
        aggregated = pred_df.mean(axis=1)
    elif method == 'median':
        aggregated = pred_df.median(axis=1)
    elif method == 'weighted_mean':
        # Could weight by Sharpe ratio or other metric
        # For now, equal weights
        aggregated = pred_df.mean(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated

