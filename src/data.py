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

# Dataset directory (for pre-downloaded CSV files)
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'Dataset')


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


def load_from_dataset_csv(tickers: List[str], start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Load data from pre-downloaded CSV files in the Dataset directory.
    
    This function searches for CSV files matching the pattern: {TICKER}_{start_date}_to_{end_date}.csv
    and loads them into a MultiIndex DataFrame format compatible with the rest of the codebase.
    
    Args:
        tickers: List of ticker symbols to load
        start_date: Start date string (optional, for matching file names)
        end_date: End date string (optional, for matching file names)
    
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV) and Date index, or None if files not found
    """
    if not os.path.exists(DATASET_DIR):
        return None
    
    # Get all CSV files in Dataset directory
    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        return None
    
    loaded_data = {}
    
    # Try to find CSV files for each ticker
    for ticker in tickers:
        # Look for files matching the ticker (case-insensitive)
        matching_files = [f for f in csv_files if f.upper().startswith(ticker.upper() + '_')]
        
        if not matching_files:
            # Try to find any file with the ticker name in it
            matching_files = [f for f in csv_files if ticker.upper() in f.upper()]
        
        if matching_files:
            # Use the first matching file (or most recent if multiple)
            csv_file = os.path.join(DATASET_DIR, matching_files[0])
            
            try:
                # Read CSV file - handle the MultiIndex header structure
                # The CSV has headers at rows 0, 1, 2, with actual data starting at row 3
                df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0, parse_dates=True)
                
                # The CSV structure is: Price/Ticker/Date as MultiIndex columns
                # We need to extract the actual data columns (Close, High, Low, Open, Volume)
                # and the ticker name
                
                # Get the ticker name from the second level of columns
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract ticker from column structure
                    # Columns are like: (Close, AAPL, Unnamed: X)
                    ticker_from_file = None
                    for col in df.columns:
                        if len(col) >= 2 and col[1] and col[1] != 'Unnamed: 1_level_2':
                            ticker_from_file = col[1]
                            break
                    
                    # If we couldn't find ticker in columns, try to extract from filename
                    if ticker_from_file is None:
                        ticker_from_file = ticker
                    
                    # Reconstruct DataFrame with proper MultiIndex
                    # Extract OHLCV columns
                    data_dict = {}
                    for col_name in ['Close', 'High', 'Low', 'Open', 'Volume']:
                        # Find column matching this name
                        matching_cols = [c for c in df.columns if c[0] == col_name]
                        if matching_cols:
                            data_dict[(ticker_from_file, col_name)] = df[matching_cols[0]]
                    
                    if data_dict:
                        ticker_df = pd.DataFrame(data_dict)
                        loaded_data[ticker_from_file] = ticker_df
                
            except Exception as e:
                # If MultiIndex reading fails, try simpler approach
                try:
                    # Try reading with skiprows to skip the header rows
                    df = pd.read_csv(csv_file, skiprows=2, index_col=0, parse_dates=True)
                    
                    # Check if columns are what we expect
                    if all(col in df.columns for col in ['Close', 'High', 'Low', 'Open', 'Volume']):
                        # Create MultiIndex columns
                        ticker_name = ticker
                        # Try to extract from filename
                        if '_' in os.path.basename(csv_file):
                            ticker_name = os.path.basename(csv_file).split('_')[0].upper()
                        
                        multi_cols = pd.MultiIndex.from_product([[ticker_name], df.columns])
                        df.columns = multi_cols
                        loaded_data[ticker_name] = df
                except Exception as e2:
                    print(f"Error loading CSV file {csv_file}: {e2}")
                    continue
    
    if not loaded_data:
        return None
    
    # Combine all tickers into a single DataFrame
    if len(loaded_data) == 1:
        return list(loaded_data.values())[0]
    else:
        # Concatenate along columns, aligning by index (date)
        combined = pd.concat(loaded_data.values(), axis=1)
        return combined


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
    
    This function tries multiple data sources in order:
    1. Pre-downloaded CSV files in Dataset directory
    2. Cached pickle files in data_cache directory
    3. Download from Yahoo Finance via yfinance
    
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
    # First, try to load from pre-downloaded CSV files in Dataset directory
    csv_data = load_from_dataset_csv(tickers, start_date, end_date)
    if csv_data is not None and not csv_data.empty:
        # Filter by date range if specified
        if start_date or end_date:
            try:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    csv_data = csv_data[csv_data.index >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    csv_data = csv_data[csv_data.index <= end_dt]
            except Exception as e:
                print(f"Warning: Could not filter CSV data by date range: {e}")
        
        # Verify we have the requested tickers
        if isinstance(csv_data.columns, pd.MultiIndex):
            available_tickers = csv_data.columns.get_level_values(0).unique()
            missing_tickers = [t for t in tickers if t not in available_tickers]
            if missing_tickers:
                print(f"Warning: Some tickers not found in CSV files: {missing_tickers}")
                # Filter to only available tickers
                csv_data = csv_data[[t for t in available_tickers if t in tickers]]
        
        if not csv_data.empty:
            print(f"✅ Loaded data from CSV files in Dataset directory for: {list(csv_data.columns.get_level_values(0).unique()) if isinstance(csv_data.columns, pd.MultiIndex) else 'single ticker'}")
            return csv_data
    
    # Try to load from cache second
    if use_cache:
        cache_key = get_cache_key(tickers, start_date, end_date, period)
        cached_data = load_cached_data(cache_key, cache_max_age_hours)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data
    try:
        # Download without auto_adjust to have consistent column names
        # We'll use 'Adj Close' if available, otherwise 'Close'
        if period:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
        else:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if data.empty:
            return pd.DataFrame()
        
        # Handle different data structures from yfinance
        # When downloading multiple tickers, structure is MultiIndex (ticker, OHLCV)
        # When downloading single ticker, structure is flat
        
        # Normalize to MultiIndex structure for consistency
        if len(tickers) == 1:
            # Single ticker: convert to MultiIndex
            if not isinstance(data.columns, pd.MultiIndex):
                data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        else:
            # Multiple tickers: should already be MultiIndex, but verify
            if not isinstance(data.columns, pd.MultiIndex):
                # This shouldn't happen with multiple tickers, but handle it
                print(f"Warning: Unexpected data structure for multiple tickers")
                # Try to infer - this is a fallback
                if 'Close' in data.columns:
                    # Flat structure with ticker names as columns
                    data = data.stack().unstack(level=0)
        
        # Filter out any tickers that have no data
        if isinstance(data.columns, pd.MultiIndex):
            valid_tickers = []
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    ticker_data = data[ticker]
                    if not ticker_data.empty and not ticker_data.isna().all().all():
                        valid_tickers.append(ticker)
            
            if valid_tickers:
                data = data[valid_tickers]
            else:
                return pd.DataFrame()
        
        # Save to cache
        if use_cache and not data.empty:
            cache_key = get_cache_key(tickers, start_date, end_date, period)
            save_to_cache(cache_key, data)
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Extract close prices - handle different data structures
    # Prefer 'Adj Close' (adjusted for splits/dividends), fallback to 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex structure: (ticker, OHLCV)
        level_1_values = data.columns.get_level_values(1).unique()
        
        if 'Adj Close' in level_1_values:
            prices = data.xs('Adj Close', level=1, axis=1)
        elif 'Close' in level_1_values:
            prices = data.xs('Close', level=1, axis=1)
        else:
            # Fallback: try to get first numeric column for each ticker
            prices = pd.DataFrame(index=data.index)
            for ticker in data.columns.get_level_values(0).unique():
                ticker_data = data[ticker]
                if 'Adj Close' in ticker_data.columns:
                    prices[ticker] = ticker_data['Adj Close']
                elif 'Close' in ticker_data.columns:
                    prices[ticker] = ticker_data['Close']
                elif len(ticker_data.columns) > 0:
                    # Use first column as fallback
                    prices[ticker] = ticker_data.iloc[:, 0]
    else:
        # Single ticker or flat structure
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        elif len(data.columns) > 0:
            # Use first column as fallback
            prices = data.iloc[:, 0]
        else:
            prices = data
    
    # Ensure prices is a DataFrame with tickers as columns
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        if len(tickers) == 1:
            prices.columns = tickers
    
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

