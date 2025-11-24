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
warnings.filterwarnings('ignore')


def download_data(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> pd.DataFrame:
    """
    Download historical price data for given tickers.
    
    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (optional)
        period: Alternative to start/end, e.g., '1y', '5y', 'max'
    
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV) and Date index
    """
    try:
        if period:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        else:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # Handle single ticker case (yfinance returns different structure)
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        
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
    end_date: Optional[str] = None
) -> tuple:
    """
    Prepare data for portfolio construction.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date (optional)
    
    Returns:
        Tuple of (returns DataFrame, prices DataFrame)
    """
    # Download data
    data = download_data(tickers, start_date=start_date, end_date=end_date)
    
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

