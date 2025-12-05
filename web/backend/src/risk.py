"""
Risk models module.

Implements various covariance estimation methods:
- Sample covariance
- Ledoit-Wolf shrinkage
- Graphical LASSO (GLASSO)
- GARCH(1,1) per asset
- DCC (Dynamic Conditional Correlation)
"""

# Standard library imports
import warnings
from typing import Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

# Suppress warnings
warnings.filterwarnings('ignore')


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sample covariance matrix.
    
    Args:
        returns: DataFrame with returns (assets as columns, dates as index)
    
    Returns:
        Covariance matrix as DataFrame
    """
    return returns.cov() * 252  # Annualized


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ledoit-Wolf shrinkage covariance matrix.
    
    This method shrinks the sample covariance towards a target (identity or constant correlation)
    to reduce estimation error.
    
    Args:
        returns: DataFrame with returns
    
    Returns:
        Shrinkage covariance matrix as DataFrame
    """
    lw = LedoitWolf()
    cov = lw.fit(returns.values)
    cov_matrix = pd.DataFrame(
        cov.covariance_ * 252,  # Annualized
        index=returns.columns,
        columns=returns.columns
    )
    return cov_matrix


def glasso_covariance(returns: pd.DataFrame, alpha: Optional[float] = None) -> pd.DataFrame:
    """
    Compute Graphical LASSO (GLASSO) covariance matrix.
    
    GLASSO estimates a sparse precision matrix (inverse covariance) using L1 regularization.
    
    Args:
        returns: DataFrame with returns
        alpha: Regularization parameter (if None, uses cross-validation)
    
    Returns:
        GLASSO covariance matrix as DataFrame
    """
    if alpha is None:
        # Use cross-validation to select alpha
        model = GraphicalLassoCV(cv=5, max_iter=500)
    else:
        from sklearn.covariance import GraphicalLasso
        model = GraphicalLasso(alpha=alpha, max_iter=500)
    
    model.fit(returns.values)
    cov_matrix = pd.DataFrame(
        model.covariance_ * 252,  # Annualized
        index=returns.columns,
        columns=returns.columns
    )
    return cov_matrix


def garch_volatility(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal'
) -> Tuple[pd.Series, pd.Series]:
    """
    Fit GARCH(p,q) model to estimate time-varying volatility.
    
    Args:
        returns: Series of returns for a single asset
        p: GARCH lag order (default 1)
        q: ARCH lag order (default 1)
        dist: Distribution assumption ('normal', 't', 'skewt')
    
    Returns:
        Tuple of (fitted volatility series, forecasted volatility)
    """
    try:
        # Fit GARCH model
        model = arch_model(returns * 100, vol='Garch', p=p, q=q, dist=dist)
        fitted = model.fit(disp='off')
        
        # Get conditional volatility (convert back from percentage)
        volatility = fitted.conditional_volatility / 100
        
        # Forecast next period volatility
        forecast = fitted.forecast(horizon=1)
        forecast_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
        
        return volatility, pd.Series([forecast_vol], index=[returns.index[-1]])
    except Exception as e:
        print(f"Error fitting GARCH for {returns.name}: {e}")
        # Fallback to rolling volatility
        vol = returns.rolling(window=60).std() * np.sqrt(252)
        return vol, pd.Series([vol.iloc[-1]], index=[returns.index[-1]])


def garch_covariance_matrix(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute GARCH-based covariance matrix using individual GARCH models per asset.
    
    This uses univariate GARCH for each asset's volatility and combines with
    sample correlation (can be extended to DCC for dynamic correlation).
    
    Args:
        returns: DataFrame with returns
        p: GARCH lag order
        q: ARCH lag order
    
    Returns:
        Tuple of (GARCH covariance matrix, forecasted covariance matrix)
    """
    n_assets = len(returns.columns)
    volatilities = {}
    forecast_vols = {}
    
    # Fit GARCH for each asset
    for asset in returns.columns:
        vol, fvol = garch_volatility(returns[asset], p=p, q=q)
        volatilities[asset] = vol
        forecast_vols[asset] = fvol.iloc[0]
    
    # Create volatility matrix
    vol_df = pd.DataFrame(volatilities, index=returns.index)
    
    # Use sample correlation (can be replaced with DCC)
    correlation = returns.corr()
    
    # Construct covariance: C = D * R * D where D is diagonal of volatilities
    # For forecast, use latest volatilities
    latest_vols = np.array([forecast_vols[asset] for asset in returns.columns])
    D = np.diag(latest_vols * np.sqrt(252))  # Annualized
    forecast_cov = D @ correlation.values @ D
    
    # For historical, use time-varying volatilities with constant correlation
    # (simplified - full DCC would have time-varying correlation too)
    historical_cov = {}
    for date in returns.index:
        vols = vol_df.loc[date].values * np.sqrt(252)
        D_t = np.diag(vols)
        cov_t = D_t @ correlation.values @ D_t
        historical_cov[date] = cov_t
    
    # Return average historical and forecast
    avg_historical_cov = pd.DataFrame(
        np.mean(list(historical_cov.values()), axis=0),
        index=returns.columns,
        columns=returns.columns
    )
    
    forecast_cov_df = pd.DataFrame(
        forecast_cov,
        index=returns.columns,
        columns=returns.columns
    )
    
    return avg_historical_cov, forecast_cov_df


def dcc_correlation(
    returns: pd.DataFrame,
    garch_params: Tuple[int, int] = (1, 1)
) -> pd.DataFrame:
    """
    Estimate Dynamic Conditional Correlation (DCC) model.
    
    Note: Full DCC implementation is complex. This is a simplified version
    using rolling correlation with GARCH-adjusted volatilities.
    
    Args:
        returns: DataFrame with returns
        garch_params: (p, q) for GARCH models
    
    Returns:
        Average DCC correlation matrix
    """
    # For a full implementation, we would use a DCC-GARCH library
    # Here we use a simplified approach: GARCH volatilities + rolling correlation
    p, q = garch_params
    
    # Get GARCH volatilities for each asset
    garch_vols = {}
    for asset in returns.columns:
        vol, _ = garch_volatility(returns[asset], p=p, q=q)
        garch_vols[asset] = vol
    
    vol_df = pd.DataFrame(garch_vols, index=returns.index)
    
    # Standardize returns by GARCH volatilities
    standardized = returns.div(vol_df, axis=0)
    
    # Compute rolling correlation of standardized returns
    # This approximates DCC correlation
    window = min(60, len(returns) // 4)
    dcc_corr = standardized.rolling(window=window).corr().iloc[-len(returns.columns):]
    
    # Return average correlation matrix
    if isinstance(dcc_corr.index, pd.MultiIndex):
        # Reshape if needed
        corr_matrix = dcc_corr.groupby(level=0).last()
    else:
        corr_matrix = dcc_corr
    
    return corr_matrix


def get_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'ledoit_wolf'
) -> pd.DataFrame:
    """
    Get covariance matrix using specified method.
    
    Args:
        returns: DataFrame with returns
        method: 'sample', 'ledoit_wolf', 'glasso', 'garch'
    
    Returns:
        Covariance matrix
    """
    if method == 'sample':
        return sample_covariance(returns)
    elif method == 'ledoit_wolf':
        return ledoit_wolf_covariance(returns)
    elif method == 'glasso':
        return glasso_covariance(returns)
    elif method == 'garch':
        cov, _ = garch_covariance_matrix(returns)
        return cov
    else:
        raise ValueError(f"Unknown method: {method}")

