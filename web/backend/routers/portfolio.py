from fastapi import APIRouter


router = APIRouter()


@router.get("/config", summary="Default configuration for main app")
async def get_default_config() -> dict:
    """
    Default configuration used by the main portfolio page.

    This mirrors the ideas from the Streamlit dashboard and README.
    The frontend can call this on load to pre-fill forms.
    """
    return {
        "default_universe": [
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
        ],
        "default_optimization_method": "markowitz",
        "default_risk_model": "ledoit_wolf",
        "default_backtest_type": "walk_forward",
    }


@router.post("/analyze", summary="Run portfolio analysis (placeholder)")
async def analyze_portfolio(payload: dict) -> dict:
    """
    Placeholder for the main FinLove analysis endpoint.

    Later this can call into the existing Python engine under `src/`
    to:
    - load data,
    - build risk models,
    - run optimization and backtests,
    - return metrics and charts to the frontend.
    """
    # For now, just echo the payload structure back to the client.
    return {
        "message": "Portfolio analysis endpoint is not yet implemented.",
        "received": payload,
    }


