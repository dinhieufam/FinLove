"""
Data Pre-download Script

This script allows you to pre-download and cache financial data for faster dashboard performance.
You can run this script to download data for popular tickers or custom ticker lists.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import download_data, prepare_portfolio_data, clear_cache, get_cache_info

# Default sector ETFs from the project
DEFAULT_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# Popular tech stocks
POPULAR_TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"]

# Popular stocks across sectors
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "JPM", "BAC", "WMT", "JNJ", "PG", "V", "MA", "DIS", "HD", "NKE"
]


def download_ticker_list(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: str = None,
    description: str = ""
):
    """
    Download and cache data for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data (None = today)
        description: Description of the ticker list
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Downloading data for: {description}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    try:
        # Download data (this will automatically cache it)
        returns, prices = prepare_portfolio_data(
            tickers,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        if returns.empty:
            print(f"⚠️  Warning: No data downloaded for {', '.join(tickers)}")
        else:
            print(f"✅ Successfully downloaded and cached data for {len(returns.columns)} assets")
            print(f"   Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
            print(f"   Total days: {len(returns)}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")


def main():
    """Main function to download datasets."""
    print("="*60)
    print("FinLove Data Pre-download Script")
    print("="*60)
    print("\nThis script will download and cache financial data.")
    print("Cached data will be used automatically by the dashboard for faster performance.\n")
    
    # Show current cache info
    cache_info = get_cache_info()
    print(f"Current cache: {cache_info['total_files']} files, {cache_info['total_size_mb']:.2f} MB")
    if cache_info['oldest_cache']:
        print(f"Oldest cache: {cache_info['oldest_cache']}")
        print(f"Newest cache: {cache_info['newest_cache']}")
    print()
    
    # Get user choice
    print("Select dataset to download:")
    print("1. Default Sector ETFs (11 ETFs)")
    print("2. Popular Tech Stocks (8 stocks)")
    print("3. Popular Stocks Across Sectors (19 stocks)")
    print("4. Custom ticker list")
    print("5. Download all (ETFs + Tech + Popular)")
    print("6. Clear cache")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-6): ").strip()
    
    # Default date range (last 10 years)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")
    
    if choice == "1":
        download_ticker_list(DEFAULT_ETFS, start_date, end_date, "Default Sector ETFs")
    
    elif choice == "2":
        download_ticker_list(POPULAR_TECH, start_date, end_date, "Popular Tech Stocks")
    
    elif choice == "3":
        download_ticker_list(POPULAR_STOCKS, start_date, end_date, "Popular Stocks")
    
    elif choice == "4":
        ticker_input = input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
        if ticker_input:
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            download_ticker_list(tickers, start_date, end_date, "Custom Ticker List")
        else:
            print("No tickers entered.")
    
    elif choice == "5":
        print("\nDownloading all datasets...")
        download_ticker_list(DEFAULT_ETFS, start_date, end_date, "Default Sector ETFs")
        download_ticker_list(POPULAR_TECH, start_date, end_date, "Popular Tech Stocks")
        download_ticker_list(POPULAR_STOCKS, start_date, end_date, "Popular Stocks")
        print("\n✅ All datasets downloaded!")
    
    elif choice == "6":
        confirm = input("Clear all cached data? (yes/no): ").strip().lower()
        if confirm == "yes":
            cleared = clear_cache()
            print(f"✅ Cleared {cleared} cached files")
        else:
            print("Cancelled.")
    
    elif choice == "0":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice.")
        return
    
    # Show updated cache info
    cache_info = get_cache_info()
    print(f"\n{'='*60}")
    print(f"Updated cache: {cache_info['total_files']} files, {cache_info['total_size_mb']:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

