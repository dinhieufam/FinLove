import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from scipy.optimize import minimize

# --- FIX: CACHE HANDLING ---
# This helps prevent the 'database is locked' error by using a unique cache folder 
# or you can try to disable it via requests, but a clean restart usually helps.
# For this MVP, we rely on standard download but add error handling.

app = Flask(__name__)

# --- CONFIGURATION FROM PROPOSAL ---
DEFAULT_TICKERS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
RISK_FREE_RATE = 0.04  # Assumed 4%

def get_data(tickers, period="2y"):
    """Fetches Adjusted Close prices from Yahoo Finance."""
    print(f"Downloading data for: {tickers}")
    
    # --- FIX 1: force auto_adjust=False ---
    # This ensures we get the 'Adj Close' column that the rest of the code expects.
    # threads=False helps avoid some database locking issues on local machines.
    data = yf.download(tickers, period=period, auto_adjust=False, threads=False)
    
    # Check if data is empty
    if data.empty:
        raise ValueError("No data downloaded. Check your internet connection or ticker symbols.")

    # --- FIX 2: Handle Column Access Safely ---
    # yfinance structure can vary. We try to get 'Adj Close', fallback to 'Close'.
    if 'Adj Close' in data:
        prices = data['Adj Close']
    elif 'Close' in data:
        prices = data['Close']
    else:
        # Fallback for single ticker download which might not have multi-level columns
        prices = data

    # Clean up: Drop rows with all NaNs (holidays etc)
    prices = prices.dropna(how='all')
    
    # Calculate daily log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns

def standard_metrics(returns):
    """Calculates annualized mean returns and covariance matrix."""
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Annualize (252 trading days)
    expected_returns = mean_daily_returns * 252
    cov_matrix_annual = cov_matrix * 252
    return expected_returns, cov_matrix_annual

def get_portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculates portfolio return and volatility."""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(mean_returns * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Handle division by zero just in case
    if p_vol == 0: return 0
    return - (p_ret - risk_free_rate) / p_vol

def optimize_portfolio(mean_returns, cov_matrix, strategy="mvo"):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, RISK_FREE_RATE)
    
    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: 0 <= weight <= 1 (Long only)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Initial Guess: Equal weights
    init_guess = num_assets * [1. / num_assets,]
    
    # --- BLACK-LITTERMAN SIMPLIFICATION ---
    if strategy == 'bl':
        bl_adjustment = pd.Series(0, index=mean_returns.index)
        # Only apply views if the ticker exists in our downloaded data
        if 'XLK' in mean_returns.index: bl_adjustment['XLK'] += 0.05
        if 'XLU' in mean_returns.index: bl_adjustment['XLU'] -= 0.03
        
        mean_returns = (mean_returns * 0.5) + ((mean_returns + bl_adjustment) * 0.5)

    # Optimization
    result = minimize(neg_sharpe_ratio, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, mean_returns

def generate_plot(mean_returns, cov_matrix, optimal_weights, tickers):
    num_portfolios = 2000 
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        
        p_ret = np.sum(mean_returns * weights)
        p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = (p_ret - RISK_FREE_RATE) / p_std 

    opt_ret = np.sum(mean_returns * optimal_weights)
    opt_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    plt.figure(figsize=(10, 6))
    plt.style.use('bmh') 
    
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_std, opt_ret, marker='*', color='red', s=300, label='Optimal Portfolio')
    
    plt.title('Efficient Frontier & Optimal Portfolio')
    plt.xlabel('Volatility (Std. Dev)')
    plt.ylabel('Expected Returns')
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html', default_tickers=", ".join(DEFAULT_TICKERS))

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        ticker_str = data.get('tickers', '')
        strategy = data.get('strategy', 'mvo')
        
        tickers = [t.strip().upper() for t in ticker_str.split(',') if t.strip()]
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400

        log_returns = get_data(tickers)
        
        if log_returns.empty:
             return jsonify({'error': 'Could not fetch data. Check tickers.'}), 400

        expected_returns, cov_matrix = standard_metrics(log_returns)
        
        # Ensure our tickers match the data we actually got back (in case some failed)
        valid_tickers = list(expected_returns.index)
        
        weights, used_returns = optimize_portfolio(expected_returns, cov_matrix, strategy)
        
        allocation = {k: round(v * 100, 2) for k, v in zip(valid_tickers, weights) if v > 0.0001}
        
        p_ret = np.sum(used_returns * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (p_ret - RISK_FREE_RATE) / p_vol
        
        plot_url = generate_plot(expected_returns, cov_matrix, weights, valid_tickers)

        return jsonify({
            'allocation': allocation,
            'metrics': {
                'return': round(p_ret * 100, 2),
                'volatility': round(p_vol * 100, 2),
                'sharpe': round(sharpe, 2)
            },
            'plot': plot_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc() # This prints the full error to your terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)