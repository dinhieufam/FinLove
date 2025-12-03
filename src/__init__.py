"""
FinLove Portfolio Construction Package

A comprehensive portfolio construction and optimization system.
"""

__version__ = "1.0.0"

# Core modules
from . import data
from . import optimize
from . import risk
from . import backtest
from . import metrics

# Prediction modules (optional)
try:
    from . import model_collector
    from . import forecast
    from . import predict
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False

