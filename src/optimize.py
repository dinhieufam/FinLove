"""
Portfolio optimization module.

Implements various optimization objectives:
- Markowitz (Mean-Variance)
- Black-Litterman
- CVaR (Conditional Value at Risk)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def markowitz_optimization(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 1.0,
    constraints: Optional[Dict] = None
) -> pd.Series:
    """
    Markowitz mean-variance optimization.
    
    Maximizes: μ'w - (λ/2) * w'Σw
    where μ is expected returns, Σ is covariance, λ is risk aversion.
    
    Args:
        expected_returns: Expected returns for each asset
        covariance: Covariance matrix
        risk_aversion: Risk aversion parameter (higher = more risk averse)
        constraints: Dictionary with constraints (e.g., {'long_only': True, 'max_weight': 0.4})
    
    Returns:
        Optimal weights as Series
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    
    # Objective: maximize expected return - risk penalty
    objective = cp.Maximize(
        expected_returns.values @ w - (risk_aversion / 2) * cp.quad_form(w, covariance.values)
    )
    
    # Default constraints
    constraints_list = [cp.sum(w) == 1]  # Budget constraint
    
    # Add optional constraints
    if constraints:
        if constraints.get('long_only', False):
            constraints_list.append(w >= 0)
        
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
        
        if 'min_weight' in constraints:
            constraints_list.append(w >= constraints['min_weight'])
    
    # Solve
    problem = cp.Problem(objective, constraints_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        # Fallback to equal weights
        return pd.Series(1.0 / n, index=expected_returns.index)
    
    weights = pd.Series(w.value, index=expected_returns.index)
    weights = weights / weights.sum()  # Normalize
    
    return weights


def minimum_variance_optimization(
    covariance: pd.DataFrame,
    constraints: Optional[Dict] = None
) -> pd.Series:
    """
    Minimum variance portfolio optimization.
    
    Minimizes: w'Σw
    subject to constraints.
    
    Args:
        covariance: Covariance matrix
        constraints: Dictionary with constraints
    
    Returns:
        Optimal weights as Series
    """
    n = len(covariance)
    w = cp.Variable(n)
    
    # Objective: minimize variance
    objective = cp.Minimize(cp.quad_form(w, covariance.values))
    
    # Constraints
    constraints_list = [cp.sum(w) == 1]
    
    if constraints:
        if constraints.get('long_only', False):
            constraints_list.append(w >= 0)
        
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
    
    problem = cp.Problem(objective, constraints_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        return pd.Series(1.0 / n, index=covariance.index)
    
    weights = pd.Series(w.value, index=covariance.index)
    weights = weights / weights.sum()
    
    return weights


def sharpe_maximization(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    constraints: Optional[Dict] = None
) -> pd.Series:
    """
    Maximize Sharpe ratio portfolio.
    
    Maximizes: (μ'w - rf) / sqrt(w'Σw)
    
    Args:
        expected_returns: Expected returns
        covariance: Covariance matrix
        risk_free_rate: Risk-free rate (annualized)
        constraints: Constraints dictionary
    
    Returns:
        Optimal weights
    """
    # Convert to daily risk-free rate if needed
    rf_daily = risk_free_rate / 252
    
    n = len(expected_returns)
    w = cp.Variable(n)
    
    # Excess returns
    excess_returns = expected_returns.values - rf_daily
    
    # Sharpe ratio = (excess return) / (volatility)
    # We maximize excess return / volatility, which is equivalent to
    # maximizing excess return subject to volatility = 1
    portfolio_return = excess_returns @ w
    portfolio_vol = cp.sqrt(cp.quad_form(w, covariance.values))
    
    objective = cp.Maximize(portfolio_return / portfolio_vol)
    
    constraints_list = [cp.sum(w) == 1]
    
    if constraints:
        if constraints.get('long_only', False):
            constraints_list.append(w >= 0)
        
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
    
    problem = cp.Problem(objective, constraints_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        return pd.Series(1.0 / n, index=expected_returns.index)
    
    weights = pd.Series(w.value, index=expected_returns.index)
    weights = weights / weights.sum()
    
    return weights


def black_litterman_returns(
    market_caps: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
    views: Optional[Dict] = None,
    view_confidences: Optional[Dict] = None
) -> pd.Series:
    """
    Compute Black-Litterman expected returns.
    
    Combines market equilibrium returns with investor views.
    
    Args:
        market_caps: Market capitalizations (for market portfolio)
        covariance: Covariance matrix
        risk_aversion: Risk aversion parameter
        tau: Uncertainty scaling parameter
        views: Dictionary of views {asset: expected_return}
        view_confidences: Dictionary of view confidences {asset: confidence}
    
    Returns:
        BL-adjusted expected returns
    """
    # Market portfolio weights (proportional to market cap)
    market_weights = market_caps / market_caps.sum()
    
    # Market equilibrium returns: Π = λ * Σ * w_market
    equilibrium_returns = risk_aversion * covariance @ market_weights
    
    # If no views, return equilibrium returns
    if views is None or len(views) == 0:
        return equilibrium_returns
    
    # Build view vector and pick matrix
    n_assets = len(covariance)
    n_views = len(views)
    
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    Omega = np.eye(n_views)  # Uncertainty matrix
    
    asset_list = list(covariance.index)
    for i, (asset, view_return) in enumerate(views.items()):
        if asset in asset_list:
            idx = asset_list.index(asset)
            P[i, idx] = 1.0
            Q[i] = view_return
            
            # Set confidence (lower = more confident)
            if view_confidences and asset in view_confidences:
                Omega[i, i] = view_confidences[asset]
            else:
                # Default: use tau * P * Σ * P'
                Omega[i, i] = tau * (P[i] @ covariance.values @ P[i])
    
    # BL formula: μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)Π + P'Ω^(-1)Q]
    tau_sigma = tau * covariance.values
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)
    
    # Compute BL returns
    A = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ equilibrium_returns.values + P.T @ omega_inv @ Q
    
    bl_returns = np.linalg.solve(A, b)
    
    return pd.Series(bl_returns, index=covariance.index)


def cvar_optimization(
    returns: pd.DataFrame,
    confidence_level: float = 0.05,
    constraints: Optional[Dict] = None
) -> pd.Series:
    """
    CVaR (Conditional Value at Risk) optimization.
    
    Minimizes CVaR, which is the expected loss given that loss exceeds VaR.
    
    Args:
        returns: Historical returns (scenarios as rows, assets as columns)
        confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
        constraints: Constraints dictionary
    
    Returns:
        Optimal weights
    """
    n_scenarios, n_assets = returns.shape
    w = cp.Variable(n_assets)
    VaR = cp.Variable()  # Value at Risk
    u = cp.Variable(n_scenarios)  # Auxiliary variables
    
    # CVaR = VaR + (1/α) * E[max(0, -R'w - VaR)]
    # where α is confidence level
    alpha = confidence_level
    
    # Portfolio returns for each scenario
    portfolio_returns = returns.values @ w
    
    # CVaR objective
    objective = cp.Minimize(VaR + (1 / (alpha * n_scenarios)) * cp.sum(u))
    
    # Constraints
    constraints_list = [
        cp.sum(w) == 1,
        u >= 0,
        u >= -portfolio_returns - VaR
    ]
    
    if constraints:
        if constraints.get('long_only', False):
            constraints_list.append(w >= 0)
        
        if 'max_weight' in constraints:
            constraints_list.append(w <= constraints['max_weight'])
    
    problem = cp.Problem(objective, constraints_list)
    problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        return pd.Series(1.0 / n_assets, index=returns.columns)
    
    weights = pd.Series(w.value, index=returns.columns)
    weights = weights / weights.sum()
    
    return weights


def optimize_portfolio(
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    method: str = 'markowitz',
    **kwargs
) -> pd.Series:
    """
    Main portfolio optimization function.
    
    Args:
        returns: Historical returns
        covariance: Covariance matrix
        method: 'markowitz', 'min_variance', 'sharpe', 'black_litterman', 'cvar'
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Optimal weights
    """
    if method == 'markowitz':
        expected_returns = returns.mean() * 252  # Annualized
        return markowitz_optimization(
            expected_returns,
            covariance,
            risk_aversion=kwargs.get('risk_aversion', 1.0),
            constraints=kwargs.get('constraints', {})
        )
    
    elif method == 'min_variance':
        return minimum_variance_optimization(
            covariance,
            constraints=kwargs.get('constraints', {})
        )
    
    elif method == 'sharpe':
        expected_returns = returns.mean() * 252
        return sharpe_maximization(
            expected_returns,
            covariance,
            risk_free_rate=kwargs.get('risk_free_rate', 0.0),
            constraints=kwargs.get('constraints', {})
        )
    
    elif method == 'black_litterman':
        expected_returns = returns.mean() * 252
        market_caps = kwargs.get('market_caps', pd.Series(1.0, index=returns.columns))
        bl_returns = black_litterman_returns(
            market_caps,
            covariance,
            risk_aversion=kwargs.get('risk_aversion', 3.0),
            tau=kwargs.get('tau', 0.05),
            views=kwargs.get('views', None),
            view_confidences=kwargs.get('view_confidences', None)
        )
        return markowitz_optimization(
            bl_returns,
            covariance,
            risk_aversion=kwargs.get('risk_aversion', 1.0),
            constraints=kwargs.get('constraints', {})
        )
    
    elif method == 'cvar':
        return cvar_optimization(
            returns,
            confidence_level=kwargs.get('confidence_level', 0.05),
            constraints=kwargs.get('constraints', {})
        )
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")

