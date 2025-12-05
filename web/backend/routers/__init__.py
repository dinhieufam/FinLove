from fastapi import APIRouter

from . import landing, portfolio

api_router = APIRouter()

# Public-facing landing/marketing-ish data for the home page
api_router.include_router(landing.router, prefix="/landing", tags=["landing"])

# Core portfolio-related endpoints for the main app
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])



