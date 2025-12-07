"""
QA (Question-Answering) Router for Portfolio RAG System.

This router handles:
1. Storing portfolio analysis results
2. Answering questions about portfolios using RAG
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure we load the bundled engine code inside this web/backend folder.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.rag_system import get_rag_system  # type: ignore

router = APIRouter()


class StorePortfolioRequest(BaseModel):
    """Request to store a portfolio analysis result."""
    analysis_result: dict = Field(..., description="Portfolio analysis result from /analyze endpoint")


class QARequest(BaseModel):
    """Request for Q&A about a portfolio."""
    question: str = Field(..., description="User's question about the portfolio", min_length=1)
    portfolio_id: Optional[str] = Field(None, description="Optional portfolio ID to filter context")


class QAResponse(BaseModel):
    """Response from Q&A endpoint."""
    answer: str
    portfolio_id: Optional[str] = None
    chunks_used: int = 0


@router.post("/store", summary="Store portfolio analysis for Q&A")
async def store_portfolio(payload: StorePortfolioRequest) -> dict:
    """
    Store a portfolio analysis result in the RAG system.
    
    This should be called after running portfolio analysis to make it
    available for Q&A queries.
    """
    try:
        rag_system = get_rag_system()
        portfolio_id = rag_system.store_portfolio(payload.analysis_result)
        
        return {
            "ok": True,
            "portfolio_id": portfolio_id,
            "message": "Portfolio stored successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing portfolio: {str(e)}")


@router.post("/qa", summary="Ask a question about portfolio")
async def ask_question(payload: QARequest) -> QAResponse:
    """
    Answer a question about a portfolio using RAG.
    
    Steps:
    1. Embed the question
    2. Retrieve relevant chunks (2-4)
    3. Build prompt with context
    4. Call Gemini LLM
    5. Return grounded answer
    """
    try:
        rag_system = get_rag_system()
        
        # Debug: Check what's stored
        total_chunks = len(rag_system.chunks)
        total_portfolios = len(rag_system.portfolio_id_to_index)
        
        print(f"DEBUG QA: Total chunks stored: {total_chunks}")
        print(f"DEBUG QA: Total portfolios stored: {total_portfolios}")
        print(f"DEBUG QA: Requested portfolio_id: {payload.portfolio_id}")
        if payload.portfolio_id:
            print(f"DEBUG QA: Portfolio IDs available: {list(rag_system.portfolio_id_to_index.keys())}")
        
        # Retrieve relevant chunks
        context_chunks = rag_system.retrieve_chunks(
            question=payload.question,
            portfolio_id=payload.portfolio_id,
            top_k=4
        )
        
        print(f"DEBUG QA: Retrieved {len(context_chunks)} chunks")
        
        if not context_chunks:
            # If no portfolio_id provided, try without filtering
            if payload.portfolio_id:
                # Try without portfolio_id filter
                print(f"DEBUG QA: Trying without portfolio_id filter...")
                context_chunks = rag_system.retrieve_chunks(
                    question=payload.question,
                    portfolio_id=None,
                    top_k=4
                )
                print(f"DEBUG QA: Retrieved {len(context_chunks)} chunks without filter")
            
            if not context_chunks:
                error_msg = f"No portfolio data available. Please run a portfolio analysis first."
                if payload.portfolio_id:
                    error_msg += f" (Requested portfolio_id: {payload.portfolio_id}, Available portfolios: {list(rag_system.portfolio_id_to_index.keys())})"
                return QAResponse(
                    answer=error_msg,
                    portfolio_id=payload.portfolio_id,
                    chunks_used=0
                )
        
        # Query LLM with context
        answer = rag_system.query_llm(
            question=payload.question,
            context_chunks=context_chunks
        )
        
        return QAResponse(
            answer=answer,
            portfolio_id=payload.portfolio_id,
            chunks_used=len(context_chunks)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@router.get("/health", summary="Check RAG system health")
async def health_check() -> dict:
    """Check if RAG system is properly initialized."""
    try:
        rag_system = get_rag_system()
        return {
            "ok": True,
            "chunks_stored": len(rag_system.chunks),
            "portfolios_stored": len(rag_system.portfolio_id_to_index),
            "has_embedding_model": hasattr(rag_system, 'embedding_model_name') and rag_system.embedding_model_name is not None,
            "has_llm_model": len(rag_system.api_keys) > 0
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }

