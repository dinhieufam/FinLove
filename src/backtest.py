"""
Backtesting engine module.

Implements walk-forward backtesting with:
- Transaction costs
- Rebalance bands
- Multiple optimization methods
- Performance tracking
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .data import prepare_portfolio_data
from .metrics import calculate_all_metrics, calculate_returns
from .optimize import optimize_portfolio
from .risk import get_covariance_matrix

# Suppress warnings
warnings.filterwarnings('ignore')


def apply_transaction_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    transaction_cost: float = 0.001
) -> float:
    """
    Calculate transaction costs based on weight changes.

    Args:
        old_weights: Previous portfolio weights.
        new_weights: New portfolio weights.
        transaction_cost: Proportional transaction cost (e.g., 0.001 = 0.1%).
            Defaults to 0.001.

    Returns:
        Total transaction cost as float.
    """
    # Align weights
    aligned_old = old_weights.reindex(new_weights.index, fill_value=0.0)
    aligned_new = new_weights.reindex(old_weights.index, fill_value=0.0)
    
    # Calculate turnover
    turnover = (aligned_new - aligned_old).abs().sum()
    
    # Transaction cost = turnover * cost rate
    cost = turnover * transaction_cost
    
    return cost


def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    rebalance_band: float = 0.05
) -> bool:
    """
    Determine if rebalancing is needed based on drift bands.

    Rebalances if any weight drifts beyond the band.

    Args:
        current_weights: Current portfolio weights.
        target_weights: Target portfolio weights.
        rebalance_band: Maximum allowed drift (e.g., 0.05 = 5%). Defaults to 0.05.

    Returns:
        True if rebalancing is needed, False otherwise.
    """
    # Align weights
    aligned_current = current_weights.reindex(target_weights.index, fill_value=0.0)
    aligned_target = target_weights.reindex(current_weights.index, fill_value=0.0)
    
    # Calculate drift
    drift = (aligned_current - aligned_target).abs()
    
    # Rebalance if any weight drifts beyond band
    return (drift > rebalance_band).any()


def walk_forward_backtest(
    returns: pd.DataFrame,
    train_window: int = 36,
    test_window: int = 1,
    optimization_method: str = 'markowitz',
    risk_model: str = 'ledoit_wolf',
    transaction_cost: float = 0.001,
    rebalance_band: float = 0.05,
    rebalance_frequency: str = 'monthly',
    constraints: Optional[Dict[str, Any]] = None,
    **optimization_kwargs: Any
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Walk-forward backtesting with rolling window.

    Args:
        returns: DataFrame with asset returns (daily).
        train_window: Training window in months. Defaults to 36.
        test_window: Test window in months. Defaults to 1.
        optimization_method: Optimization method ('markowitz', 'min_variance', 'sharpe',
            'black_litterman', 'cvar'). Defaults to 'markowitz'.
        risk_model: Risk model ('sample', 'ledoit_wolf', 'glasso', 'garch').
            Defaults to 'ledoit_wolf'.
        transaction_cost: Proportional transaction cost. Defaults to 0.001.
        rebalance_band: Rebalance band threshold. Defaults to 0.05.
        rebalance_frequency: Rebalance frequency ('monthly', 'weekly', 'daily').
            Defaults to 'monthly'.
        constraints: Optimization constraints dictionary. Defaults to None.
        **optimization_kwargs: Additional arguments for optimization.

    Returns:
        Tuple of (portfolio_returns, weights_history, metrics_dict).
    """
    # Convert returns to monthly for rebalancing
    if rebalance_frequency == 'monthly':
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        rebalance_dates = monthly_returns.index
    elif rebalance_frequency == 'weekly':
        weekly_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
        rebalance_dates = weekly_returns.index
    else:  # daily
        rebalance_dates = returns.index
    
    # Initialize
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    weights_history = pd.DataFrame(index=rebalance_dates, columns=returns.columns, dtype=float)
    current_weights = None
    target_weights = None
    
    # Walk forward
    for i in range(len(rebalance_dates)):
        rebalance_date = rebalance_dates[i]
        
        # Find training period end (before rebalance date)
        train_end_idx = returns.index.get_indexer([rebalance_date], method='nearest')[0]
        
        if train_end_idx < train_window * 21:  # Approximate months to days
            # Not enough data, use equal weights
            n_assets = len(returns.columns)
            target_weights = pd.Series(1.0 / n_assets, index=returns.columns)
        else:
            # Get training data
            train_start_idx = max(0, train_end_idx - train_window * 21)
            train_returns = returns.iloc[train_start_idx:train_end_idx]
            
            if len(train_returns) < 60:  # Need minimum data
                n_assets = len(returns.columns)
                target_weights = pd.Series(1.0 / n_assets, index=returns.columns)
            else:
                # Estimate covariance
                covariance = get_covariance_matrix(train_returns, method=risk_model)
                
                # Optimize portfolio
                opt_kwargs = {
                    'constraints': constraints or {'long_only': True},
                    **optimization_kwargs
                }
                
                target_weights = optimize_portfolio(
                    train_returns,
                    covariance,
                    method=optimization_method,
                    **opt_kwargs
                )
        
        # Check if rebalancing is needed
        if current_weights is None:
            # First period: always rebalance
            should_rebal = True
        else:
            # Check drift
            should_rebal = should_rebalance(current_weights, target_weights, rebalance_band)
        
        # Apply transaction costs if rebalancing
        if should_rebal and current_weights is not None:
            cost = apply_transaction_costs(current_weights, target_weights, transaction_cost)
            # Adjust returns for cost (approximate)
            portfolio_returns.loc[rebalance_date] = -cost
            current_weights = target_weights.copy()
        elif current_weights is None:
            current_weights = target_weights.copy()
        
        # Store weights
        weights_history.loc[rebalance_date] = current_weights
        
        # Calculate portfolio returns for test period
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            test_returns = returns.loc[rebalance_date:next_rebalance]
        else:
            test_returns = returns.loc[rebalance_date:]
        
        # Portfolio returns = weighted sum of asset returns
        portfolio_ret = (test_returns * current_weights).sum(axis=1)
        portfolio_returns.loc[test_returns.index] = portfolio_ret
    
    # Fill NaN with 0
    portfolio_returns = portfolio_returns.fillna(0.0)
    
    # Calculate metrics
    metrics = calculate_all_metrics(portfolio_returns, weights_history)
    
    return portfolio_returns, weights_history, metrics


def simple_backtest(
    returns: pd.DataFrame,
    optimization_method: str = 'markowitz',
    risk_model: str = 'ledoit_wolf',
    transaction_cost: float = 0.0,
    constraints: Optional[Dict[str, Any]] = None,
    **optimization_kwargs: Any
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Simple backtest: optimize once and hold.

    Args:
        returns: DataFrame with asset returns.
        optimization_method: Optimization method. Defaults to 'markowitz'.
        risk_model: Risk model. Defaults to 'ledoit_wolf'.
        transaction_cost: Transaction cost (not used in simple backtest). Defaults to 0.0.
        constraints: Optimization constraints dictionary. Defaults to None.
        **optimization_kwargs: Additional optimization arguments.

    Returns:
        Tuple of (portfolio_returns, weights, metrics_dict).
    """
    # Estimate covariance
    covariance = get_covariance_matrix(returns, method=risk_model)
    
    # Optimize
    opt_kwargs = {
        'constraints': constraints or {'long_only': True},
        **optimization_kwargs
    }
    
    weights = optimize_portfolio(
        returns,
        covariance,
        method=optimization_method,
        **opt_kwargs
    )
    
    # Calculate portfolio returns
    portfolio_returns = calculate_returns(weights, returns)
    
    # Calculate metrics
    weights_df = pd.DataFrame([weights], index=[returns.index[-1]], columns=returns.columns)
    metrics = calculate_all_metrics(portfolio_returns, weights_df)
    
    return portfolio_returns, weights, metrics

