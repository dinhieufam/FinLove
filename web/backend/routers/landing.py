from fastapi import APIRouter


router = APIRouter()


@router.get("/", summary="Landing page summary data")
async def get_landing_summary() -> dict:
    """
    Basic data for the public landing page.

    Later, this can surface:
    - high-level description text,
    - example performance metrics,
    - list of supported methods/risk models,
    - and any marketing copy you want to show.
    """
    return {
        "title": "FinLove — Risk-Aware Portfolio Construction",
        "tagline": "Build stable, risk-aware portfolios with advanced risk models and backtesting.",
        "features": [
            "Multiple risk models (Ledoit-Wolf, GLASSO, GARCH, DCC)",
            "Optimization methods (Markowitz, Black-Litterman, CVaR, Minimum Variance, Sharpe)",
            "Walk-forward backtesting with transaction costs and rebalance bands",
            "Interactive portfolio visualization and risk analytics",
        ],
    }


