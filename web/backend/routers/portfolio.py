import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

# Ensure we load the bundled engine code inside this web/backend folder.
# Add the backend root to sys.path so `src` is importable as a package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.backtest import simple_backtest, walk_forward_backtest  # type: ignore
from src.data import prepare_portfolio_data, get_company_info, get_available_tickers, get_available_tickers_with_names  # type: ignore
from src.metrics import rolling_sharpe, rolling_volatility  # type: ignore
from src.risk import get_covariance_matrix  # type: ignore


router = APIRouter()


DEFAULT_UNIVERSE: List[str] = [
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC",
]


class BacktestRequest(BaseModel):
    """
    Request payload for running a portfolio backtest.

    This closely mirrors the controls in the original Streamlit dashboard but
    is trimmed to the core configuration we need for the web app.
    """

    tickers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional explicit list of tickers. If omitted or empty and "
            "`use_default_universe` is true, the default sector ETF universe is used."
        ),
    )
    use_default_universe: bool = Field(
        default=True,
        description="Whether to fall back to the default sector ETF universe.",
    )
    start_date: str = Field(
        ...,
        description="Backtest start date in YYYY-MM-DD format.",
        examples=["2015-01-01"],
    )
    end_date: str = Field(
        ...,
        description="Backtest end date in YYYY-MM-DD format.",
        examples=["2024-12-31"],
    )

    optimization_method: Literal[
        "markowitz", "min_variance", "sharpe", "black_litterman", "cvar"
    ] = Field(
        default="markowitz",
        description="Portfolio optimization objective.",
    )
    risk_model: Literal["ledoit_wolf", "sample", "glasso", "garch"] = Field(
        default="ledoit_wolf",
        description="Covariance / risk model.",
    )
    risk_aversion: float = Field(
        default=1.0,
        ge=0.01,
        le=20.0,
        description="Risk aversion parameter for Markowitz-style optimizations.",
    )

    backtest_type: Literal["simple", "walk_forward"] = Field(
        default="walk_forward",
        description=(
            "'simple' = optimize once and hold, 'walk_forward' = rolling window."
        ),
    )
    transaction_cost: float = Field(
        default=0.001,
        ge=0.0,
        le=0.05,
        description="Proportional transaction cost (e.g. 0.001 = 0.1%).",
    )
    rebalance_band: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Tolerance band before rebalancing (fractional weights).",
    )
    train_window: int = Field(
        default=36,
        ge=6,
        le=120,
        description="Training window in months for walk-forward backtests.",
    )
    test_window: int = Field(
        default=1,
        ge=1,
        le=24,
        description="Test window in months for walk-forward backtests.",
    )


@router.get("/config", summary="Default configuration for main app")
async def get_default_config() -> dict:
    """
    Default configuration used by the main portfolio page.

    This mirrors the ideas from the Streamlit dashboard and README.
    The frontend can call this on load to pre-fill forms.
    """
    return {
        "default_universe": DEFAULT_UNIVERSE,
        "default_optimization_method": "markowitz",
        "default_risk_model": "ledoit_wolf",
        "default_backtest_type": "walk_forward",
    }


@router.get("/available-tickers", summary="Get list of available tickers")
async def get_available_tickers_endpoint() -> dict:
    """
    Get list of all available tickers from the data folder with company names.
    
    Returns:
        Dictionary with list of ticker objects (ticker and name) and simple ticker list.
    """
    tickers_with_names = get_available_tickers_with_names()
    tickers = [t['ticker'] for t in tickers_with_names]
    return {
        "tickers": tickers,
        "tickers_with_names": tickers_with_names,
        "count": len(tickers)
    }


def _series_to_payload(series) -> Dict[str, Any]:
    """Convert a pandas Series to a JSON-serializable payload."""
    if series is None or getattr(series, "empty", False):
        return {"dates": [], "values": []}
    return {
        "dates": [str(idx) for idx in series.index],
        "values": [float(v) if v is not None else None for v in series.values],
    }


def _weights_history_to_payload(weights_history) -> Dict[str, Any]:
    if weights_history is None or weights_history.empty:
        return {"dates": [], "assets": [], "matrix": []}
    assets = list(weights_history.columns)
    dates = [str(idx) for idx in weights_history.index]
    matrix = [
        [float(v) if v is not None else 0.0 for v in row]
        for row in weights_history.to_numpy()
    ]
    return {"dates": dates, "assets": assets, "matrix": matrix}


@router.post("/analyze", summary="Run portfolio analysis")
async def analyze_portfolio(payload: BacktestRequest) -> dict:
    """
    Main analysis endpoint used by the Next.js dashboard.

    It calls into the shared FinLove engine under `src/` to:
    - download data,
    - construct returns,
    - run the chosen backtest,
    - and return metrics / series for visualization.
    """
    # Get available tickers for validation
    available_tickers = set(get_available_tickers())
    
    # Resolve effective universe
    user_tickers = payload.tickers or []
    if payload.use_default_universe or not user_tickers:
        tickers = DEFAULT_UNIVERSE.copy()
    else:
        tickers = [t.strip().upper() for t in user_tickers if t.strip()]
    
    # Validate that all requested tickers are available
    invalid_tickers = [t for t in tickers if t not in available_tickers]
    if invalid_tickers:
        return {
            "ok": False,
            "error": f"Invalid tickers: {', '.join(invalid_tickers)}. Available tickers: {', '.join(sorted(available_tickers))}",
            "tickers": tickers,
            "available_tickers": sorted(available_tickers)
        }

    # Prepare data
    returns, prices = prepare_portfolio_data(
        tickers,
        start_date=payload.start_date,
        end_date=payload.end_date,
        use_cache=True,
    )

    if returns.empty or len(returns) < 60:
        return {
            "ok": False,
            "error": "Insufficient data for the selected tickers/date range.",
            "tickers": tickers,
        }

    # Clean returns as in the Streamlit app
    returns = returns.dropna(axis=1, thresh=len(returns) * 0.8)
    if returns.empty:
        return {
            "ok": False,
            "error": "No valid assets after cleaning; please adjust the universe.",
            "tickers": tickers,
        }

    tickers = list(returns.columns)

    # Run backtest
    if payload.backtest_type == "simple":
        portfolio_returns, weights, metrics = simple_backtest(
            returns,
            optimization_method=payload.optimization_method,
            risk_model=payload.risk_model,
            transaction_cost=payload.transaction_cost,
            constraints={"long_only": True},
            risk_aversion=payload.risk_aversion,
        )
        import pandas as pd

        weights_history = pd.DataFrame(
            [weights], index=[returns.index[-1]], columns=tickers
        )
    else:
        portfolio_returns, weights_history, metrics = walk_forward_backtest(
            returns,
            train_window=payload.train_window,
            test_window=payload.test_window,
            optimization_method=payload.optimization_method,
            risk_model=payload.risk_model,
            transaction_cost=payload.transaction_cost,
            rebalance_band=payload.rebalance_band,
            rebalance_frequency="monthly",
            constraints={"long_only": True},
            risk_aversion=payload.risk_aversion,
        )
        weights = weights_history.iloc[-1]

    # Build series for charts – mirror key Streamlit visuals
    import pandas as pd

    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    benchmark_returns = returns.mean(axis=1)
    cumulative_benchmark = (1 + benchmark_returns).cumprod()

    rolling_sharpe_series = rolling_sharpe(portfolio_returns, window=252)
    drawdown_series = (
        (1 + portfolio_returns).cumprod()
        / (1 + portfolio_returns).cumprod().cummax()
        - 1
    )
    rolling_vol_series = rolling_volatility(portfolio_returns, window=252)

    # Histogram inputs: just pass the raw daily returns
    returns_for_hist = portfolio_returns.dropna()

    # Weights table
    weights_sorted = (
        pd.Series(weights, index=tickers).sort_values(ascending=False)
        if "weights" in locals()
        else pd.Series(dtype=float)
    )
    weights_payload = [
        {"asset": asset, "weight": float(w)}
        for asset, w in weights_sorted.items()
    ]

    response = {
        "ok": True,
        "universe": tickers,
        "metrics": metrics,
        "series": {
            "cumulative_portfolio": _series_to_payload(cumulative_portfolio),
            "cumulative_benchmark": _series_to_payload(cumulative_benchmark),
            "rolling_sharpe": _series_to_payload(rolling_sharpe_series),
            "drawdown": _series_to_payload(drawdown_series),
            "rolling_volatility": _series_to_payload(rolling_vol_series),
            "returns": _series_to_payload(returns_for_hist),
        },
        "weights": {
            "current": weights_payload,
            "history": _weights_history_to_payload(weights_history),
        },
        "config": {
            "optimization_method": payload.optimization_method,
            "risk_model": payload.risk_model,
            "risk_aversion": payload.risk_aversion,
            "backtest_type": payload.backtest_type,
            "transaction_cost": payload.transaction_cost,
            "rebalance_band": payload.rebalance_band,
            "train_window": payload.train_window,
            "test_window": payload.test_window,
            "start_date": payload.start_date,
            "end_date": payload.end_date,
        },
    }

    # --- Full Feature Migration Additions ---
    
    # 1. Risk Matrices
    import numpy as np
    try:
        cov_matrix = get_covariance_matrix(returns, method=payload.risk_model)
        # Calculate correlation matrix
        d = np.sqrt(np.diag(cov_matrix))
        d = np.clip(d, 1e-8, None) # Avoid division by zero
        corr_matrix = cov_matrix.values / np.outer(d, d)
        
        risk_matrices = {
            "covariance": {
                "labels": list(cov_matrix.columns),
                "matrix": cov_matrix.values.tolist()
            },
            "correlation": {
                "labels": list(cov_matrix.columns),
                "matrix": corr_matrix.tolist()
            }
        }
    except Exception as e:
        print(f"Error computing risk matrices: {e}")
        risk_matrices = None

    # 2. Company Info
    company_info = []
    for ticker in tickers:
        try:
            info = get_company_info(ticker)
            company_info.append(info)
        except:
            company_info.append({"symbol": ticker, "name": ticker})

    # 3. Per-Asset Analysis
    assets_analysis = {}
    for ticker in tickers:
        try:
            asset_ret = returns[ticker].dropna()
            if len(asset_ret) > 10:
                # Cumulative
                asset_cum = (1 + asset_ret).cumprod()
                # Rolling Sharpe
                asset_sharpe = rolling_sharpe(asset_ret, window=252)
                # Drawdown
                asset_running_max = asset_cum.expanding().max()
                asset_dd = (asset_cum - asset_running_max) / asset_running_max
                
                assets_analysis[ticker] = {
                    "cumulative": _series_to_payload(asset_cum),
                    "rolling_sharpe": _series_to_payload(asset_sharpe),
                    "drawdown": _series_to_payload(asset_dd),
                    "metrics": {
                        "total_return": float(asset_cum.iloc[-1] - 1),
                        "volatility": float(asset_ret.std() * np.sqrt(252)),
                        "sharpe": float(asset_ret.mean() * 252 / (asset_ret.std() * np.sqrt(252))) if asset_ret.std() > 0 else 0
                    }
                }
        except Exception as e:
            print(f"Error analyzing asset {ticker}: {e}")

    # Update response with new data
    response["risk_matrices"] = risk_matrices
    response["company_info"] = company_info
    response["assets_analysis"] = assets_analysis
    
    # Add raw returns for histograms
    asset_returns_payload = {}
    for ticker in tickers:
        asset_returns_payload[ticker] = [float(x) for x in returns[ticker].dropna().values]
    response["assets_returns"] = asset_returns_payload

    return response


