import os
# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import api_router
from routers.prediction import router as prediction_router
from routers.qa import router as qa_router


app = FastAPI(
    title="FinLove API",
    description="Backend API for the FinLove risk-aware portfolio web app.",
    version="0.1.0",
)

# In development we allow all origins; tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


# Mount versioned API under /api
app.include_router(api_router, prefix="/api")
app.include_router(prediction_router, prefix="/api/portfolio", tags=["prediction"])
app.include_router(qa_router, prefix="/api/qa", tags=["qa"])


if __name__ == "__main__":
    # Allow: python app.py
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

