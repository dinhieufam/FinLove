"""
Model Cache Module.

This module handles saving and loading trained forecasting models to disk
for faster inference. Models are cached with metadata including ticker,
model type, parameters, and training date.
"""

# Standard library imports
import hashlib
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Module-level constants
MODELS_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models')

# Create models cache directory if it doesn't exist
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)


def get_model_cache_key(
    ticker: str,
    model_type: str,
    lookback_window: int,
    epochs: Optional[int] = None,
    **model_params: Any
) -> str:
    """
    Generate a cache key for a model based on its parameters.
    
    Args:
        ticker: Ticker symbol
        model_type: Type of model ('lstm', 'tcn', 'xgboost', 'transformer')
        lookback_window: Lookback window size
        epochs: Number of training epochs (for neural networks)
        **model_params: Additional model-specific parameters
    
    Returns:
        Cache key string (MD5 hash)
    """
    # Create a sorted dictionary of all parameters
    params_dict = {
        'ticker': ticker.upper(),
        'model_type': model_type,
        'lookback_window': lookback_window,
        'epochs': epochs,
        **model_params
    }
    
    # Convert to string and hash
    key_str = '_'.join([f"{k}:{v}" for k, v in sorted(params_dict.items()) if v is not None])
    return hashlib.md5(key_str.encode()).hexdigest()


def get_model_cache_path(cache_key: str) -> Tuple[str, str]:
    """
    Get file paths for model cache and metadata.
    
    Args:
        cache_key: Cache key for the model
    
    Returns:
        Tuple of (model_file_path, metadata_file_path)
    """
    model_file = os.path.join(MODELS_CACHE_DIR, f"{cache_key}_model.pkl")
    metadata_file = os.path.join(MODELS_CACHE_DIR, f"{cache_key}_meta.pkl")
    return model_file, metadata_file


def save_model_to_cache(
    cache_key: str,
    model: Any,
    scaler: Optional[Any] = None,
    ticker: str = "",
    model_type: str = "",
    lookback_window: int = 60,
    epochs: Optional[int] = None,
    training_date: Optional[datetime] = None,
    **model_params: Any
) -> None:
    """
    Save a trained model to cache.
    
    Args:
        cache_key: Cache key for the model
        model: Trained model object
        scaler: Scaler object used for preprocessing (if any)
        ticker: Ticker symbol
        model_type: Type of model
        lookback_window: Lookback window size
        epochs: Number of training epochs
        training_date: Date when model was trained
        **model_params: Additional model-specific parameters
    """
    model_file, metadata_file = get_model_cache_path(cache_key)
    
    try:
        # Save model and scaler
        model_data = {
            'model': model,
            'scaler': scaler
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'ticker': ticker.upper(),
            'model_type': model_type,
            'lookback_window': lookback_window,
            'epochs': epochs,
            'training_date': training_date or datetime.now(),
            'model_params': model_params
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
            
    except Exception as e:
        print(f"Error saving model to cache: {e}")


def load_model_from_cache(
    cache_key: str,
    max_age_days: int = 30
) -> Optional[Tuple[Any, Optional[Any], Dict[str, Any]]]:
    """
    Load a trained model from cache if it exists and is not too old.
    
    Args:
        cache_key: Cache key for the model
        max_age_days: Maximum age of cached model in days. Defaults to 30.
    
    Returns:
        Tuple of (model, scaler, metadata) or None if not found or too old
    """
    model_file, metadata_file = get_model_cache_path(cache_key)
    
    if not os.path.exists(model_file) or not os.path.exists(metadata_file):
        return None
    
    try:
        # Check metadata for age
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        training_date = metadata.get('training_date', datetime.min)
        age_days = (datetime.now() - training_date).days
        
        if age_days > max_age_days:
            # Cache too old, remove it
            try:
                os.remove(model_file)
                os.remove(metadata_file)
            except:
                pass
            return None
        
        # Load model and scaler
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        
        return model, scaler, metadata
        
    except Exception as e:
        print(f"Error loading model from cache: {e}")
        # Try to clean up corrupted cache files
        try:
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
        except:
            pass
        return None


def clear_model_cache(ticker: Optional[str] = None, model_type: Optional[str] = None) -> int:
    """
    Clear model cache files.
    
    Args:
        ticker: If provided, only clear models for this ticker
        model_type: If provided, only clear models of this type
    
    Returns:
        Number of files cleared
    """
    cleared = 0
    
    try:
        for filename in os.listdir(MODELS_CACHE_DIR):
            if filename.endswith('_meta.pkl'):
                metadata_file = os.path.join(MODELS_CACHE_DIR, filename)
                try:
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    # Check if we should clear this file
                    should_clear = True
                    if ticker and metadata.get('ticker', '').upper() != ticker.upper():
                        should_clear = False
                    if model_type and metadata.get('model_type', '') != model_type:
                        should_clear = False
                    
                    if should_clear:
                        # Remove both model and metadata files
                        model_file = metadata_file.replace('_meta.pkl', '_model.pkl')
                        if os.path.exists(model_file):
                            os.remove(model_file)
                        os.remove(metadata_file)
                        cleared += 1
                except:
                    continue
    except Exception as e:
        print(f"Error clearing model cache: {e}")
    
    return cleared


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about cached models.
    
    Returns:
        Dictionary with cache statistics
    """
    total_files = 0
    total_size_mb = 0.0
    newest_cache = None
    oldest_cache = None
    models_by_type = {}
    models_by_ticker = {}
    
    try:
        for filename in os.listdir(MODELS_CACHE_DIR):
            if filename.endswith('_meta.pkl'):
                metadata_file = os.path.join(MODELS_CACHE_DIR, filename)
                try:
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    # Get file size
                    file_size = os.path.getsize(metadata_file)
                    model_file = metadata_file.replace('_meta.pkl', '_model.pkl')
                    if os.path.exists(model_file):
                        file_size += os.path.getsize(model_file)
                    
                    total_size_mb += file_size / (1024 * 1024)
                    total_files += 1
                    
                    # Track dates
                    training_date = metadata.get('training_date')
                    if training_date:
                        if newest_cache is None or training_date > newest_cache:
                            newest_cache = training_date
                        if oldest_cache is None or training_date < oldest_cache:
                            oldest_cache = training_date
                    
                    # Track by type
                    model_type = metadata.get('model_type', 'unknown')
                    models_by_type[model_type] = models_by_type.get(model_type, 0) + 1
                    
                    # Track by ticker
                    ticker = metadata.get('ticker', 'unknown')
                    models_by_ticker[ticker] = models_by_ticker.get(ticker, 0) + 1
                    
                except:
                    continue
    except Exception as e:
        print(f"Error getting cache info: {e}")
    
    return {
        'total_files': total_files,
        'total_size_mb': total_size_mb,
        'newest_cache': newest_cache,
        'oldest_cache': oldest_cache,
        'models_by_type': models_by_type,
        'models_by_ticker': models_by_ticker
    }

