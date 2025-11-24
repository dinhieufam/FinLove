import yfinance as yf
import pandas as pd
import numpy as np

# Expanded list of major companies across different sectors
tickers = [
    # Technology Sector
    "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "CRM", "ADBE",
]

def add_technical_indicators(df):
    """
    Add technical indicators and calculated features to the dataframe
    """
    # Daily price changes
    df['Daily_Change'] = df['Close'] - df['Open']
    df['Daily_Change_Pct'] = df['Close'].pct_change() * 100
    
    # Price ranges
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['Daily_Range'] / df['Open']) * 100
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price relative to moving averages
    df['Price_vs_MA5'] = (df['Close'] / df['MA_5'] - 1) * 100
    df['Price_vs_MA20'] = (df['Close'] / df['MA_20'] - 1) * 100
    df['Price_vs_MA50'] = (df['Close'] / df['MA_50'] - 1) * 100
    
    # Volatility (rolling standard deviation)
    df['Volatility_5'] = df['Daily_Change_Pct'].rolling(window=5).std()
    df['Volatility_20'] = df['Daily_Change_Pct'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Price momentum
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def get_company_info(ticker):
    """
    Get additional company information from yfinance info property
    """
    try:
        dat = yf.Ticker(ticker)
        info = dat.info
        
        # Extract key financial metrics
        company_data = {
            'Market_Cap': info.get('marketCap', np.nan),
            'Enterprise_Value': info.get('enterpriseValue', np.nan),
            'P_E_Ratio': info.get('trailingPE', np.nan),
            'Forward_P_E': info.get('forwardPE', np.nan),
            'PEG_Ratio': info.get('pegRatio', np.nan),
            'Price_to_Book': info.get('priceToBook', np.nan),
            'Price_to_Sales': info.get('priceToSalesTrailing12Months', np.nan),
            'Dividend_Yield': info.get('dividendYield', np.nan),
            'Beta': info.get('beta', np.nan),
            '52_Week_High': info.get('fiftyTwoWeekHigh', np.nan),
            '52_Week_Low': info.get('fiftyTwoWeekLow', np.nan),
            'Avg_Volume': info.get('averageVolume', np.nan),
            'Shares_Outstanding': info.get('sharesOutstanding', np.nan),
            'Float_Shares': info.get('floatShares', np.nan),
            'Revenue_Growth': info.get('revenueGrowth', np.nan),
            'Earnings_Growth': info.get('earningsGrowth', np.nan),
            'ROE': info.get('returnOnEquity', np.nan),
            'ROA': info.get('returnOnAssets', np.nan),
            'Debt_to_Equity': info.get('debtToEquity', np.nan),
            'Current_Ratio': info.get('currentRatio', np.nan),
            'Quick_Ratio': info.get('quickRatio', np.nan),
            'Gross_Margin': info.get('grossMargins', np.nan),
            'Operating_Margin': info.get('operatingMargins', np.nan),
            'Profit_Margin': info.get('profitMargins', np.nan)
        }
        
        return company_data
    except Exception as e:
        print(f"Error getting info for {ticker}: {e}")
        return {}

for ticker in tickers:
    try:
        print(f"Processing {ticker}...")
        
        # Get historical data
        dat = yf.Ticker(ticker)
        history = dat.history(period='1y')
        
        # Add technical indicators
        history = add_technical_indicators(history)
        
        # Get company info
        company_info = get_company_info(ticker)
        
        # Add company info as additional columns (same value for all rows)
        for key, value in company_info.items():
            history[key] = value
        
        # Save enhanced data
        history.to_csv(f'{ticker}_enhanced_history.csv')
        print(f"Downloaded {ticker} with enhanced features")
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue