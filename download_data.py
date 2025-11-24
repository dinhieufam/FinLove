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
POPULAR_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "ADBE", "ORCL", "CRM"
]

# Popular stocks across sectors
POPULAR_STOCKS = [
    # Tech (also included in POPULAR_TECH)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK",
    # Consumer (Retail, Food, eCommerce)
    "WMT", "PG", "HD", "DIS", "NKE", "COST", "SBUX", "TGT", "MCD", "KO", "PEP", "LOW",
    # Healthcare
    "JNJ", "PFE", "MRK", "LLY", "UNH", "ABT", "TMO", "CVS",
    # Industrials
    "CAT", "GE", "MMM", "HON", "LMT", "BA", "DE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "PSX",
    # Utilities
    "NEE", "SO", "DUK", "AEP",
    # Communication
    "TMUS", "VZ", "T", "CHTR",
    # Real Estate
    "PLD", "AMT", "EQIX",
    # Other
    "V", "MA", "PYPL",
]

# Additional stock lists
DIVIDEND_ARISTOCRATS = [
    "KO", "PG", "PEP", "JNJ", "ABBV", "MMM", "TGT", "LOW", "CL", "ABT",
    "SWK", "GD", "AWR", "HRL", "GPC", "APD", "CINF"
]

GLOBAL_STOCKS = [
    "BABA", "TSM", "NSRGY", "TM", "SNP", "RY", "HSBC", "TOT", "NVO", "SAP",
    "SONY", "UL", "RDS.A", "BUD", "BP", "VWAGY", "SNEJF"
]

ESG_LEADERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "ADBE", "CSCO", "V", "MA", "CRM", "INTU"
]

SMALL_CAPS = [
    "AMD", "FSLR", "FFIV", "RGEN", "DOCN", "RUN", "SMAR", "ASML", "CRWD", "ZS"
]

VALUE_STOCKS = [
    "BRK.B", "CVS", "WBA", "T", "VZ", "INTC", "IBM", "XOM", "CVX", "PFE", "KO"
]

GROWTH_STOCKS = [
    "SHOP", "SQ", "ZM", "SNAP", "ROKU", "TWLO", "DOCU", "TEAM", "TDOC", "NET"
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
    print("2. Popular Tech Stocks")
    print("3. Popular Stocks Across Sectors")
    print("4. Dividend Aristocrats")
    print("5. Global Large-Cap Stocks")
    print("6. ESG Leaders")
    print("7. Small Cap Innovators")
    print("8. Value Stocks")
    print("9. Growth Stocks")
    print("10. Custom ticker list")
    print("11. Download all (ETFs + all stock lists)")
    print("12. List downloaded CSV files")
    print("0. Exit")
    choice = input("\nEnter choice (0-12): ").strip()
    
    # Default date range (last 10 years)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")
    
    if choice == "1":
        download_ticker_list(DEFAULT_ETFS, start_date, end_date, "Default Sector ETFs")

    elif choice == "2":
        download_ticker_list(POPULAR_TECH, start_date, end_date, "Popular Tech Stocks")

    elif choice == "3":
        download_ticker_list(POPULAR_STOCKS, start_date, end_date, "Popular Stocks Across Sectors")

    elif choice == "4":
        download_ticker_list(DIVIDEND_ARISTOCRATS, start_date, end_date, "Dividend Aristocrats")

    elif choice == "5":
        download_ticker_list(GLOBAL_STOCKS, start_date, end_date, "Global Large-Cap Stocks")

    elif choice == "6":
        download_ticker_list(ESG_LEADERS, start_date, end_date, "ESG Leaders")

    elif choice == "7":
        download_ticker_list(SMALL_CAPS, start_date, end_date, "Small Cap Innovators")

    elif choice == "8":
        download_ticker_list(VALUE_STOCKS, start_date, end_date, "Value Stocks")

    elif choice == "9":
        download_ticker_list(GROWTH_STOCKS, start_date, end_date, "Growth Stocks")

    elif choice == "10":
        ticker_input = input("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
        if ticker_input:
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            download_ticker_list(tickers, start_date, end_date, "Custom Ticker List")
        else:
            print("No tickers entered.")

    elif choice == "11":
        print("\nDownloading all datasets...")
        download_ticker_list(DEFAULT_ETFS, start_date, end_date, "Default Sector ETFs")
        download_ticker_list(POPULAR_TECH, start_date, end_date, "Popular Tech Stocks")
        download_ticker_list(POPULAR_STOCKS, start_date, end_date, "Popular Stocks Across Sectors")
        download_ticker_list(DIVIDEND_ARISTOCRATS, start_date, end_date, "Dividend Aristocrats")
        download_ticker_list(GLOBAL_STOCKS, start_date, end_date, "Global Large-Cap Stocks")
        download_ticker_list(ESG_LEADERS, start_date, end_date, "ESG Leaders")
        download_ticker_list(SMALL_CAPS, start_date, end_date, "Small Cap Innovators")
        download_ticker_list(VALUE_STOCKS, start_date, end_date, "Value Stocks")
        download_ticker_list(GROWTH_STOCKS, start_date, end_date, "Growth Stocks")
        print("\n✅ All datasets downloaded!")

    elif choice == "12":
        list_downloaded_files()
    
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice.")
        return

if __name__ == "__main__":
    main()
