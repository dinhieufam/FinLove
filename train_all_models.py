"""
Pre-train all forecasting models for all companies in the Dataset folder.

This script:
1. Discovers all tickers from CSV files in the Dataset directory
2. Loads historical returns data for each ticker
3. Trains all 4 models (LSTM, TCN, XGBoost, Transformer) for each ticker
4. Saves trained models to the models_cache directory for fast inference

Usage:
    python train_all_models.py [--force-retrain] [--epochs EPOCHS] [--lookback LOOKBACK] [--gpu GPU_ID]
"""

# Standard library imports
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict

# --- GPU selection: limit to 1 specific GPU ---
# Set this *before* importing torch or tensorflow!
import os
import warnings

def set_cuda_device(gpu_id: int):
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 Using GPU: {gpu_id}")
    except Exception as e:
        print(f"⚠️  Could not set CUDA_VISIBLE_DEVICES: {e}")

# Parse GPU argument early (before torch/tf is imported)
import sys
gpu_arg = None
for i, arg in enumerate(sys.argv):
    if arg in ("--gpu", "-g") and i + 1 < len(sys.argv):
        try:
            gpu_arg = int(sys.argv[i + 1])
        except Exception:
            gpu_arg = None
if gpu_arg is not None:
    set_cuda_device(gpu_arg)
else:
    # If not specified, default to GPU 0
    set_cuda_device(0)

# Third-party imports
import pandas as pd
import numpy as np

# Optionally, assert proper GPU visibility in TensorFlow and Torch
try:
    import torch
    if not torch.cuda.is_available():
        warnings.warn("Torch: CUDA not available, running on CPU.")
    else:
        current_device = torch.cuda.current_device()
        print(f"✔️ PyTorch sees GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
except ImportError:
    pass

try:
    import tensorflow as tf
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        try:
            tf.config.set_visible_devices(physical_gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_gpus[0], True)
            print(f"✔️ TensorFlow using device: {physical_gpus[0].name}")
        except Exception as e:
            print(f"⚠️  TensorFlow GPU setup failed: {e}")
    else:
        warnings.warn("TensorFlow: No GPU found, running on CPU.")
except ImportError:
    pass

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Local imports
from src.data import prepare_portfolio_data
from src.forecast import forecast_portfolio_returns
from src.model_cache import clear_model_cache, get_cache_info

# Default training parameters (matching dashboard defaults)
DEFAULT_EPOCHS = 50
DEFAULT_LOOKBACK_WINDOW = 60
DEFAULT_FORECAST_HORIZON = 30  # Used only for training trigger, not actual forecast

# Model configurations
MODELS_TO_TRAIN = ['lstm', 'tcn', 'xgboost', 'transformer']
MODEL_NAMES = {
    'lstm': 'LSTM',
    'tcn': 'Temporal Convolutional Network',
    'xgboost': 'XGBoost',
    'transformer': 'Time Series Transformer'
}


def get_all_tickers_from_dataset() -> List[str]:
    """
    Extract all unique ticker symbols from CSV files in the Dataset directory.
    
    Returns:
        List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
    """
    dataset_dir = os.path.join(project_root, 'Dataset')
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return []
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {dataset_dir}")
        return []
    
    # Extract ticker from filename (format: TICKER_YYYY-MM-DD_to_YYYY-MM-DD.csv)
    tickers = []
    for csv_file in csv_files:
        # Extract ticker (everything before the first underscore)
        ticker = csv_file.split('_')[0].upper()
        if ticker not in tickers:
            tickers.append(ticker)
    
    return sorted(tickers)


def load_ticker_returns(ticker: str) -> pd.Series:
    """
    Load returns data for a specific ticker from the Dataset directory.
    
    Uses prepare_portfolio_data which handles CSV loading, caching, and returns calculation.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        Series of daily returns with date index
    """
    try:
        # Use prepare_portfolio_data which handles all the complexity
        # It will load from Dataset CSV files, calculate returns, etc.
        returns_df, prices_df = prepare_portfolio_data(
            tickers=[ticker],
            start_date="2015-01-01",  # Use a reasonable start date
            end_date=None,  # Use all available data
            use_cache=True  # Use cache for faster subsequent runs
        )
        
        if returns_df.empty or ticker not in returns_df.columns:
            print(f"  ⚠️  No returns data found for {ticker}")
            return pd.Series(dtype=float)
        
        # Extract returns series for this ticker
        returns = returns_df[ticker].dropna()
        
        if len(returns) == 0:
            print(f"  ⚠️  Empty returns series for {ticker}")
            return pd.Series(dtype=float)
        
        return returns
            
    except Exception as e:
        print(f"  ❌ Error loading data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(dtype=float)


def train_model_for_ticker(
    ticker: str,
    returns: pd.Series,
    model_type: str,
    epochs: int,
    lookback_window: int,
    force_retrain: bool = False
) -> bool:
    """
    Train a specific model for a ticker and save to cache.
    
    Args:
        ticker: Ticker symbol
        returns: Historical returns series
        model_type: Model type ('lstm', 'tcn', 'xgboost', 'transformer')
        epochs: Number of training epochs (for neural networks)
        lookback_window: Lookback window size
        force_retrain: If True, clear cache first to force retraining
        
    Returns:
        True if training succeeded, False otherwise
    """
    try:
        # Check if we have enough data
        min_required = lookback_window + DEFAULT_FORECAST_HORIZON + 10
        if len(returns) < min_required:
            print(f"    ⚠️  Insufficient data: need {min_required}, have {len(returns)}")
            return False
        
        # Clear cache if forcing retrain
        if force_retrain:
            clear_model_cache(ticker=ticker, model_type=model_type)
        
        # Prepare model-specific parameters
        model_params = {
            'lookback_window': lookback_window,
            'epochs': epochs,
            'use_cache': True,  # Will save to cache after training
            'ticker': ticker
        }
        
        # Add model-specific parameters
        if model_type == 'lstm':
            model_params['lstm_units'] = 50
            model_params['batch_size'] = 32
        elif model_type == 'tcn':
            model_params['num_filters'] = 64
            model_params['kernel_size'] = 3
            model_params['num_blocks'] = 2
            model_params['batch_size'] = 32
        elif model_type == 'xgboost':
            model_params['n_estimators'] = 100
            model_params['max_depth'] = 6
            model_params['learning_rate'] = 0.1
            # XGBoost doesn't use epochs
            model_params.pop('epochs', None)
        elif model_type == 'transformer':
            model_params['d_model'] = 64
            model_params['num_heads'] = 4
            model_params['num_layers'] = 2
            model_params['batch_size'] = 32
        
        # Train model by calling forecast (this will train and cache if not already cached)
        # We use a small forecast_horizon just to trigger training
        result = forecast_portfolio_returns(
            portfolio_returns=returns,
            method=model_type,
            forecast_horizon=DEFAULT_FORECAST_HORIZON,
            **model_params
        )
        
        # Check if forecast was generated (indicates training succeeded)
        if result and 'forecast' in result and len(result['forecast']) > 0:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"    ❌ Error training {model_type} for {ticker}: {str(e)[:200]}")
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Pre-train all forecasting models for all companies in Dataset folder'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if models are already cached'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs for neural networks (default: {DEFAULT_EPOCHS})'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK_WINDOW,
        help=f'Lookback window size in days (default: {DEFAULT_LOOKBACK_WINDOW})'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific tickers to train (default: all tickers in Dataset)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=MODELS_TO_TRAIN,
        help='Specific models to train (default: all models)'
    )
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        default=0,
        help='GPU device index to use (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Confirm user-specified GPU
    set_cuda_device(args.gpu)

    print("=" * 80)
    print("🚀 MODEL TRAINING SCRIPT")
    print("=" * 80)
    print(f"Training Parameters:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Lookback Window: {args.lookback} days")
    print(f"  - Force Retrain: {args.force_retrain}")
    print(f"  - Selected GPU: {args.gpu}")
    print()
    
    # Get tickers to train
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"📋 Training for {len(tickers)} specified tickers: {', '.join(tickers)}")
    else:
        tickers = get_all_tickers_from_dataset()
        print(f"📋 Found {len(tickers)} tickers in Dataset directory")
    
    if not tickers:
        print("❌ No tickers to train. Exiting.")
        return
    
    # Get models to train
    models_to_train = args.models if args.models else MODELS_TO_TRAIN
    print(f"🤖 Training {len(models_to_train)} models: {', '.join([MODEL_NAMES[m] for m in models_to_train])}")
    print()
    
    # Training statistics
    stats = {
        'total_tasks': len(tickers) * len(models_to_train),
        'completed': 0,
        'failed': 0,
        'skipped': 0,
        'start_time': datetime.now()
    }
    
    # Train models for each ticker
    for ticker_idx, ticker in enumerate(tickers, 1):
        print(f"[{ticker_idx}/{len(tickers)}] 📊 Processing {ticker}...")
        
        # Load returns data
        returns = load_ticker_returns(ticker)
        
        if returns.empty or len(returns) < args.lookback + DEFAULT_FORECAST_HORIZON + 10:
            print(f"  ⚠️  Skipping {ticker}: insufficient data ({len(returns)} days)")
            stats['skipped'] += len(models_to_train)
            continue
        
        print(f"  ✅ Loaded {len(returns)} days of returns data")
        
        # Train each model
        for model_idx, model_type in enumerate(models_to_train, 1):
            model_name = MODEL_NAMES[model_type]
            print(f"    [{model_idx}/{len(models_to_train)}] 🤖 Training {model_name}...", end=' ', flush=True)
            
            success = train_model_for_ticker(
                ticker=ticker,
                returns=returns,
                model_type=model_type,
                epochs=args.epochs,
                lookback_window=args.lookback,
                force_retrain=args.force_retrain
            )
            
            if success:
                print("✅")
                stats['completed'] += 1
            else:
                print("❌")
                stats['failed'] += 1
        
        print()
    
    # Print summary
    elapsed_time = (datetime.now() - stats['start_time']).total_seconds()
    print("=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Tasks: {stats['total_tasks']}")
    print(f"✅ Completed: {stats['completed']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"⏱️  Time Elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print()
    
    # Show cache info
    cache_info = get_cache_info()
    print("💾 CACHE STATISTICS")
    print(f"  Total cached models: {cache_info['total_files']}")
    print(f"  Total cache size: {cache_info['total_size_mb']:.2f} MB")
    if cache_info['newest_cache']:
        print(f"  Newest cache: {cache_info['newest_cache'].strftime('%Y-%m-%d %H:%M:%S')}")
    if cache_info['oldest_cache']:
        print(f"  Oldest cache: {cache_info['oldest_cache'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if cache_info['models_by_type']:
        print("  Models by type:")
        for model_type, count in sorted(cache_info['models_by_type'].items()):
            print(f"    {model_type}: {count}")
    
    print("=" * 80)
    print("✅ Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

