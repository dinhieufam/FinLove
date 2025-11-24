"""
Data Pre-download Script

This script allows you to pre-download financial data and store as CSV files for later use.
You can run this script to download data for popular tickers or custom ticker lists.
"""

import os
from datetime import datetime, timedelta
from typing import List

import yfinance as yf
import pandas as pd

# Default sector ETFs from the project
DEFAULT_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]

# Popular tech stocks
POPULAR_TECH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"]

# Popular stocks across sectors
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "JPM", "BAC", "WMT", "JNJ", "PG", "V", "MA", "DIS", "HD", "NKE"
]

# Destination folder for CSVs
DATASET_DIR = "./Dataset"

def ensure_dir_exists(directory):
    """Ensure data directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_ticker_list(
    tickers: List[str],
    start_date: str = "2010-01-01",
    end_date: str = None,
    description: str = ""
):
    """
    Download and save data for a list of tickers as CSV files using yfinance.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data (None = today)
        description: Description of the ticker list
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    ensure_dir_exists(DATASET_DIR)

    print(f"\n{'='*60}")
    print(f"Downloading data for: {description}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"{'='*60}\n")

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if df.empty:
                print(f"⚠️  Warning: No data downloaded for {ticker}")
                continue

            csv_filename = os.path.join(DATASET_DIR, f"{ticker}_{start_date}_to_{end_date}.csv")
            df.to_csv(csv_filename)
            start_idx = df.index[0].date()
            end_idx = df.index[-1].date()
            print(f"✅ {ticker}: Data downloaded and saved as {csv_filename}")
            print(f"   Date range: {start_idx} to {end_idx} | Total days: {len(df)}")
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {e}")
            import traceback
            traceback.print_exc()

def list_downloaded_files():
    ensure_dir_exists(DATASET_DIR)
    files = os.listdir(DATASET_DIR)
    csv_files = [f for f in files if f.endswith(".csv")]
    print(f"\nFiles in dataset directory ({DATASET_DIR}):")
    for f in csv_files:
        print(f" - {f}")
    print(f"Total CSV files: {len(csv_files)}")

def main():
    """Main function to download datasets using yfinance."""
    print("="*60)
    print("FinLove Data Pre-download Script")
    print("="*60)
    print("\nThis script will download financial data and store it as CSV files.")
    print(f"Data will be stored in: {os.path.abspath(DATASET_DIR)}\n")
    
    # Get user choice
    print("Select dataset to download:")
    print("1. Default Sector ETFs (11 ETFs)")
    print("2. Popular Tech Stocks (8 stocks)")
    print("3. Popular Stocks Across Sectors (19 stocks)")
    print("4. Custom ticker list")
    print("5. Download all (ETFs + Tech + Popular)")
    print("6. List downloaded CSV files")
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
        list_downloaded_files()
    
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice.")
        return

if __name__ == "__main__":
    main()
