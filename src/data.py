"""
Data acquisition and feature engineering module.

This module handles downloading financial data from Yahoo Finance,
computing returns, and creating features for portfolio construction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings
import os
import pickle
from datetime import datetime, timedelta
import hashlib
warnings.filterwarnings('ignore')

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_key(tickers: List[str], start_date: str, end_date: Optional[str], period: Optional[str]) -> str:
    """
    Generate a cache key for the data request.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        period: Period string
    
    Returns:
        Cache key string
    """
    key_str = f"{sorted(tickers)}_{start_date}_{end_date}_{period}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_data(cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Load data from cache if it exists and is not too old.
    
    Args:
        cache_key: Cache key for the data
        max_age_hours: Maximum age of cached data in hours (default 24)
    
    Returns:
        Cached DataFrame or None if not found/too old
    """
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    metadata_file = os.path.join(CACHE_DIR, f"{cache_key}_meta.pkl")
    
    if not os.path.exists(cache_file) or not os.path.exists(metadata_file):
        return None
    
    try:
        # Check metadata for age
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        cache_time = metadata.get('timestamp', datetime.min)
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            # Cache too old, remove it
            os.remove(cache_file)
            os.remove(metadata_file)
            return None
        
        # Load cached data
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        return data
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def save_to_cache(cache_key: str, data: pd.DataFrame):
    """
    Save data to cache.
    
    Args:
        cache_key: Cache key for the data
        data: DataFrame to cache
    """
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    metadata_file = os.path.join(CACHE_DIR, f"{cache_key}_meta.pkl")
    
    try:
        # Save data
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now(),
            'tickers': list(data.columns.get_level_values(0).unique()) if isinstance(data.columns, pd.MultiIndex) else []
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print(f"Error saving to cache: {e}")


def download_data(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24
) -> pd.DataFrame:
    """
    Download historical price data for given tickers with optional caching.
    
    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (optional)
        period: Alternative to start/end, e.g., '1y', '5y', 'max'
        use_cache: Whether to use cached data (default True)
        cache_max_age_hours: Maximum age of cached data in hours (default 24)
    
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV) and Date index
    """
    # Try to load from cache first
    if use_cache:
        cache_key = get_cache_key(tickers, start_date, end_date, period)
        cached_data = load_cached_data(cache_key, cache_max_age_hours)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data
    try:
        if period:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        else:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Handle single ticker case (yfinance returns different structure)
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        
        # Save to cache
        if use_cache and not data.empty:
            cache_key = get_cache_key(tickers, start_date, end_date, period)
            save_to_cache(cache_key, data)
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()


def get_returns(
    prices: pd.DataFrame,
    method: str = 'log',
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data (typically 'Close' or 'Adj Close')
        method: 'log' for log returns, 'simple' for simple returns
        frequency: 'daily', 'weekly', 'monthly' for aggregation
    
    Returns:
        DataFrame with returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:  # simple returns
        returns = prices.pct_change()
    
    # Remove first row (NaN)
    returns = returns.dropna()
    
    # Aggregate if needed
    if frequency == 'weekly':
        returns = returns.resample('W').last()
    elif frequency == 'monthly':
        returns = returns.resample('M').last()
    
    return returns


def compute_features(
    data: pd.DataFrame,
    ticker: str
) -> pd.DataFrame:
    """
    Compute technical indicators and features for a single ticker.
    
    Args:
        data: DataFrame with OHLCV data for the ticker
        ticker: Ticker symbol
    
    Returns:
        DataFrame with additional feature columns
    """
    df = data.copy()
    
    # Ensure we have the right column structure
    if isinstance(df.columns, pd.MultiIndex):
        close = df[(ticker, 'Close')] if (ticker, 'Close') in df.columns else df['Close']
        high = df[(ticker, 'High')] if (ticker, 'High') in df.columns else df['High']
        low = df[(ticker, 'Low')] if (ticker, 'Low') in df.columns else df['Low']
        volume = df[(ticker, 'Volume')] if (ticker, 'Volume') in df.columns else df['Volume']
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
    
    # Price-based features
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))
    
    # Moving averages
    df['MA_5'] = close.rolling(window=5).mean()
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    df['MA_200'] = close.rolling(window=200).mean()
    
    # Volatility (rolling standard deviation of returns)
    df['volatility_5'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
    
    # Momentum
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    df['momentum_60'] = close / close.shift(60) - 1
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume indicators
    df['volume_ma'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma']
    
    return df


def get_company_info(ticker: str) -> Dict:
    """
    Get company information from Yahoo Finance.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Dictionary with company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        company_data = {
            'symbol': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', np.nan),
            'pe_ratio': info.get('trailingPE', np.nan),
            'beta': info.get('beta', np.nan),
            'dividend_yield': info.get('dividendYield', np.nan) * 100 if info.get('dividendYield') else np.nan,
            '52_week_high': info.get('fiftyTwoWeekHigh', np.nan),
            '52_week_low': info.get('fiftyTwoWeekLow', np.nan),
        }
        
        return company_data
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return {'symbol': ticker, 'name': ticker}


def prepare_portfolio_data(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True
) -> tuple:
    """
    Prepare data for portfolio construction.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date (optional)
        use_cache: Whether to use cached data (default True)
    
    Returns:
        Tuple of (returns DataFrame, prices DataFrame)
    """
    # Download data (with caching)
    data = download_data(tickers, start_date=start_date, end_date=end_date, use_cache=use_cache)
    
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close'] if 'Close' in data.columns else data
    
    # Calculate returns
    returns = get_returns(prices, method='log', frequency='daily')
    
    return returns, prices


def clear_cache(older_than_hours: Optional[int] = None):
    """
    Clear cached data files.
    
    Args:
        older_than_hours: If specified, only clear files older than this many hours.
                          If None, clears all cache.
    """
    if not os.path.exists(CACHE_DIR):
        return
    
    cleared_count = 0
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.pkl'):
            filepath = os.path.join(CACHE_DIR, filename)
            try:
                if older_than_hours is None:
                    os.remove(filepath)
                    cleared_count += 1
                else:
                    file_age_hours = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))).total_seconds() / 3600
                    if file_age_hours > older_than_hours:
                        os.remove(filepath)
                        cleared_count += 1
            except Exception as e:
                print(f"Error removing {filepath}: {e}")
    
    return cleared_count


def get_cache_info() -> Dict:
    """
    Get information about cached data.
    
    Returns:
        Dictionary with cache statistics
    """
    if not os.path.exists(CACHE_DIR):
        return {'total_files': 0, 'total_size_mb': 0, 'oldest_cache': None, 'newest_cache': None}
    
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_meta.pkl')]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR) if f.endswith('.pkl'))
    
    timestamps = []
    for filename in files:
        try:
            metadata_file = os.path.join(CACHE_DIR, filename)
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                timestamps.append(metadata.get('timestamp', datetime.min))
        except:
            pass
    
    return {
        'total_files': len(files),
        'total_size_mb': total_size / (1024 * 1024),
        'oldest_cache': min(timestamps) if timestamps else None,
        'newest_cache': max(timestamps) if timestamps else None
    }

