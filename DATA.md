# Data Download and Caching Guide

## When is Data Downloaded?

### 1. **On-Demand Download** (Automatic)
   - **When**: Every time you click "🚀 Run Analysis" in the dashboard
   - **How**: Data is automatically fetched from Yahoo Finance via `yfinance`
   - **Caching**: Downloaded data is cached for 24 hours
   - **Location**: Cached in `data_cache/` directory

### 2. **Pre-Download** (Recommended)
   - **When**: Before using the dashboard (optional but recommended)
   - **How**: Run `python download_data.py`
   - **Benefits**: 
     - Faster dashboard performance
     - No waiting for downloads during analysis
     - Can download multiple datasets at once

### 3. **Cached Data** (Automatic)
   - **When**: Data is automatically cached after first download
   - **Expiration**: 24 hours (configurable)
   - **Location**: `FinLove/data_cache/` directory
   - **Format**: Pickle files (.pkl) for fast loading

---

## Quick Start

### Option 1: Use Dashboard (Data Downloads Automatically)

1. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

2. Enter tickers and click "Run Analysis"
   - Data downloads automatically on first use
   - Subsequent uses within 24 hours use cached data

### Option 2: Pre-Download Data (Recommended)

1. Run the download script:
   ```bash
   python download_data.py
   ```

2. Select from menu:
   - **Option 1**: Default Sector ETFs (11 ETFs) - Best for portfolio analysis
   - **Option 2**: Popular Tech Stocks (8 stocks)
   - **Option 3**: Popular Stocks (19 stocks across sectors)
   - **Option 4**: Custom ticker list
   - **Option 5**: Download all datasets

3. Data is cached and ready for dashboard use

---

## Data Download Script Usage

```bash
python download_data.py
```

### Menu Options:

1. **Default Sector ETFs**: Downloads 11 sector ETFs
   - XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC

2. **Popular Tech Stocks**: Downloads 8 major tech stocks
   - AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX

3. **Popular Stocks**: Downloads 19 stocks across sectors
   - Includes tech, finance, consumer, healthcare, etc.

4. **Custom Ticker List**: Enter your own tickers
   - Example: `AAPL,MSFT,GOOGL,AMZN`

5. **Download All**: Downloads all popular datasets

6. **Clear Cache**: Remove all cached data

---

## Cache Management

### View Cache Status

**In Dashboard:**
- Open sidebar → "📦 Data Management" section
- Shows: Number of files, total size, newest/oldest cache dates

**In Script:**
- Run `python download_data.py` → Shows cache info at start

### Clear Cache

**In Dashboard:**
- Sidebar → "📦 Data Management" → "🗑️ Clear Cache" button

**In Script:**
- Run `python download_data.py` → Option 6

**Manually:**
- Delete files in `data_cache/` directory

### Cache Details

- **Location**: `FinLove/data_cache/`
- **Files**: 
  - `{hash}.pkl` - Cached data
  - `{hash}_meta.pkl` - Metadata (timestamp, tickers)
- **Expiration**: 24 hours (configurable in code)
- **Size**: Typically 1-10 MB per dataset

---

## Data Source

- **Provider**: Yahoo Finance (via `yfinance` library)
- **Data Types**: 
  - Historical prices (OHLCV)
  - Company information
  - Dividends and splits (auto-adjusted)
- **Frequency**: Daily data
- **Date Range**: Configurable (default: 2010-01-01 to today)

---

## Troubleshooting

### "Insufficient data" Error
- **Cause**: Ticker doesn't exist or has limited history
- **Solution**: 
  - Check ticker symbol is correct
  - Try different date range
  - Use tickers with longer history

### Slow Downloads
- **Cause**: Downloading many tickers or long date ranges
- **Solution**: 
  - Pre-download data using `download_data.py`
  - Use cached data (faster)
  - Reduce number of tickers or date range

### Cache Issues
- **Cause**: Corrupted cache or old data
- **Solution**: 
  - Clear cache (dashboard or script)
  - Re-download data

### Network Issues
- **Cause**: No internet or Yahoo Finance unavailable
- **Solution**: 
  - Check internet connection
  - Use pre-downloaded cached data
  - Try again later

---

## Best Practices

1. **Pre-download Popular Datasets**: Run `download_data.py` first
2. **Use Cached Data**: Let the system cache data automatically
3. **Clear Cache Periodically**: Remove old data to save space
4. **Check Cache Status**: Monitor cache size and age
5. **Download Custom Lists**: Pre-download your frequently used tickers

---

## Technical Details

### Cache Key Generation
- Based on: tickers, start_date, end_date, period
- Format: MD5 hash of parameters
- Ensures: Unique cache per request

### Cache Expiration
- Default: 24 hours
- Configurable: In `src/data.py` → `cache_max_age_hours` parameter
- Reason: Financial data changes daily, but historical data is stable

### Data Format
- **Prices**: MultiIndex DataFrame (ticker, OHLCV)
- **Returns**: DataFrame with assets as columns, dates as index
- **Storage**: Pickle format for fast Python loading

### Data Processing
- **Returns Calculation**: Log returns by default
- **Frequency**: Daily data, can aggregate to weekly/monthly
- **Missing Data**: Automatically handled (assets with >20% missing data removed)

---

## Summary

- **On-Demand**: Data downloads automatically when you use the dashboard
- **Pre-Download**: Use `download_data.py` for faster performance
- **Caching**: Automatic 24-hour cache for speed
- **Management**: View and clear cache in dashboard or script

**For most users: Run `python download_data.py` once, then use the dashboard normally!**

