"""
Example script demonstrating how to collect model results and predict future performance.

This script shows:
1. How to collect results from all model combinations
2. How to select top performing models
3. How to forecast future portfolio returns
4. How to visualize predictions
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predict import predict_future_performance
from src.model_collector import collect_all_model_results, get_best_models


def main():
    """Main example function."""
    
    # Example 1: Complete prediction pipeline
    print("\n" + "="*60)
    print("EXAMPLE 1: Complete Prediction Pipeline")
    print("="*60)
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    results = predict_future_performance(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2024-01-01",
        forecast_horizon=30,  # Forecast next 30 days
        forecast_method='ensemble',  # Use ensemble of methods
        use_top_models=5,  # Use top 5 models
        backtest_type='walk_forward',
        train_window=36,
        risk_aversion=1.0
    )
    
    # Display results
    print("\n📊 Model Results Summary:")
    print(results['model_results'][['model_id', 'sharpe_ratio', 'annualized_return', 
                                    'annualized_volatility']].to_string())
    
    print("\n🏆 Top Models Used:")
    print(results['top_models'][['model_id', 'sharpe_ratio', 'annualized_return']].to_string())
    
    print("\n🔮 Aggregated Prediction Summary:")
    pred = results['aggregated_prediction']
    print(f"   Mean daily return: {pred.mean()*100:.4f}%")
    print(f"   Std daily return: {pred.std()*100:.4f}%")
    print(f"   Cumulative return (30 days): {(1 + pred).prod() - 1:.4f}")
    
    # Visualize predictions
    visualize_predictions(results)
    
    # Example 2: Collect all model results and analyze
    print("\n" + "="*60)
    print("EXAMPLE 2: Analyze All Model Combinations")
    print("="*60)
    
    model_results = collect_all_model_results(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2024-01-01",
        backtest_type='walk_forward',
        train_window=36
    )
    
    # Find best model by different metrics
    print("\n📈 Best by Sharpe Ratio:")
    best_sharpe = get_best_models(model_results, metric='sharpe_ratio', top_n=3)
    print(best_sharpe[['model_id', 'sharpe_ratio', 'annualized_return', 'annualized_volatility']].to_string())
    
    print("\n📈 Best by Return:")
    best_return = get_best_models(model_results, metric='annualized_return', top_n=3)
    print(best_return[['model_id', 'annualized_return', 'sharpe_ratio', 'max_drawdown']].to_string())
    
    print("\n📈 Best by Low Volatility:")
    best_vol = get_best_models(model_results, metric='annualized_volatility', top_n=3)
    print(best_vol[['model_id', 'annualized_volatility', 'sharpe_ratio', 'annualized_return']].to_string())


def visualize_predictions(results: dict):
    """Visualize prediction results."""
    
    # Get historical returns from best model
    top_model = results['top_models'].iloc[0]
    historical_returns = top_model['portfolio_returns']
    
    # Get aggregated prediction
    future_prediction = results['aggregated_prediction']
    
    # Create plot
    fig = go.Figure()
    
    # Historical returns (last 60 days for context)
    hist_recent = historical_returns.iloc[-60:]
    cumulative_hist = (1 + hist_recent).cumprod()
    
    fig.add_trace(go.Scatter(
        x=cumulative_hist.index,
        y=cumulative_hist.values * 100,
        mode='lines',
        name='Historical (Last 60 days)',
        line=dict(color='blue', width=2)
    ))
    
    # Future prediction
    cumulative_pred = (1 + future_prediction).cumprod()
    cumulative_pred = cumulative_pred * cumulative_hist.iloc[-1]  # Start from last historical value
    
    fig.add_trace(go.Scatter(
        x=cumulative_pred.index,
        y=cumulative_pred.values * 100,
        mode='lines',
        name='Forecast (Next 30 days)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add confidence intervals if available (from individual forecasts)
    # Note: ensemble doesn't provide confidence intervals directly
    # You can calculate them from individual forecasts if needed
    # Uncomment below to add confidence intervals when available:
    # if 'confidence_intervals' in results:
    #     conf_int = results['confidence_intervals']
    #     fig.add_trace(go.Scatter(
    #         x=conf_int.index,
    #         y=conf_int['upper'].values * 100,
    #         mode='lines',
    #         name='Upper Bound',
    #         line=dict(color='red', width=1, dash='dot'),
    #         showlegend=False
    #     ))
    #     fig.add_trace(go.Scatter(
    #         x=conf_int.index,
    #         y=conf_int['lower'].values * 100,
    #         mode='lines',
    #         name='Lower Bound',
    #         line=dict(color='red', width=1, dash='dot'),
    #         fill='tonexty',
    #         fillcolor='rgba(255,0,0,0.1)',
    #         showlegend=False
    #     ))
    
    fig.update_layout(
        title='Portfolio Performance: Historical vs Forecast',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        height=500
    )
    
    # Save plot
    fig.write_html('prediction_forecast.html')
    print("\n📊 Visualization saved to 'prediction_forecast.html'")
    
    # Also show in console
    try:
        fig.show()
    except:
        pass


if __name__ == "__main__":
    main()

