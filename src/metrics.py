"""
Performance metrics module.

Computes various portfolio performance and risk metrics:
- Returns, volatility, Sharpe ratio
- Maximum drawdown
- VaR and CVaR
- Turnover
- Weight stability
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """
    Calculate portfolio returns from weights and asset returns.
    
    Args:
        weights: Portfolio weights (Series with asset names as index)
        returns: Asset returns (DataFrame with assets as columns)
    
    Returns:
        Portfolio returns as Series
    """
    # Align weights with returns columns
    aligned_weights = weights.reindex(returns.columns, fill_value=0.0)
    portfolio_returns = (returns * aligned_weights).sum(axis=1)
    return portfolio_returns


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    rf_daily = risk_free_rate / periods_per_year
    excess_return = ann_return - risk_free_rate
    
    return excess_return / ann_vol


def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Series of returns
    
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    if len(returns) == 0:
        return 0.0, None, None
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Running maximum
    running_max = cumulative.expanding().max()
    
    # Drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # Find peak before drawdown
    peak_idx = cumulative[:max_dd_idx].idxmax() if max_dd_idx else None
    
    return max_dd, peak_idx, max_dd_idx


def value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
    
    Returns:
        VaR (negative value representing loss)
    """
    if len(returns) == 0:
        return 0.0
    return returns.quantile(confidence_level)


def conditional_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.05
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that loss exceeds VaR.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level
    
    Returns:
        CVaR (negative value representing expected loss)
    """
    if len(returns) == 0:
        return 0.0
    
    var = value_at_risk(returns, confidence_level)
    # Average of returns below VaR
    cvar = returns[returns <= var].mean()
    
    return cvar if not np.isnan(cvar) else var


def turnover(weights_history: pd.DataFrame) -> float:
    """
    Calculate average portfolio turnover.
    
    Turnover measures how much the portfolio changes between rebalancing periods.
    
    Args:
        weights_history: DataFrame with weights over time (dates as index, assets as columns)
    
    Returns:
        Average turnover per period
    """
    if len(weights_history) < 2:
        return 0.0
    
    # Calculate change in weights between periods
    weight_changes = weights_history.diff().abs().sum(axis=1)
    # Average turnover (excluding first period which is NaN)
    avg_turnover = weight_changes.iloc[1:].mean()
    
    return avg_turnover


def weight_stability(weights_history: pd.DataFrame) -> float:
    """
    Calculate weight stability (inverse of volatility of weights).
    
    Args:
        weights_history: DataFrame with weights over time
    
    Returns:
        Stability metric (higher = more stable)
    """
    if len(weights_history) < 2:
        return 1.0
    
    # Calculate standard deviation of weights for each asset
    weight_vol = weights_history.std(axis=0)
    # Average volatility across assets
    avg_vol = weight_vol.mean()
    
    # Stability is inverse of volatility (normalized)
    stability = 1.0 / (1.0 + avg_vol)
    
    return stability


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_returns = returns.rolling(window=window).mean() * 252
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    rf_daily = risk_free_rate / 252
    excess_returns = rolling_returns - risk_free_rate
    
    rolling_sharpe = excess_returns / rolling_vol
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
    
    return rolling_sharpe


def rolling_volatility(
    returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window size
    
    Returns:
        Series of rolling volatilities (annualized)
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_all_metrics(
    portfolio_returns: pd.Series,
    weights_history: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate comprehensive portfolio performance metrics.
    
    Args:
        portfolio_returns: Series of portfolio returns
        weights_history: Optional DataFrame with weights over time
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['annualized_return'] = annualized_return(portfolio_returns)
    metrics['annualized_volatility'] = annualized_volatility(portfolio_returns)
    metrics['sharpe_ratio'] = sharpe_ratio(portfolio_returns, risk_free_rate)
    
    # Drawdown
    max_dd, peak_date, trough_date = maximum_drawdown(portfolio_returns)
    metrics['max_drawdown'] = max_dd
    metrics['peak_date'] = peak_date
    metrics['trough_date'] = trough_date
    
    # Risk metrics
    metrics['var_95'] = value_at_risk(portfolio_returns, 0.05)
    metrics['cvar_95'] = conditional_value_at_risk(portfolio_returns, 0.05)
    
    # Turnover and stability
    if weights_history is not None:
        metrics['avg_turnover'] = turnover(weights_history)
        metrics['weight_stability'] = weight_stability(weights_history)
    
    # Total return
    metrics['total_return'] = (1 + portfolio_returns).prod() - 1
    
    return metrics

