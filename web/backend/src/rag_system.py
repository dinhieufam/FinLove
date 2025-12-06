"""
RAG (Retrieval-Augmented Generation) System for Portfolio Q&A.

This module handles:
1. Generating structured portfolio outputs
2. Converting outputs to text chunks
3. Embedding chunks using Gemini
4. Vector store management (FAISS)
5. Retrieval and LLM querying
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import random

import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Hugging Face embeddings
try:
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Load environment variables
# Try to load from backend directory first, then current directory
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(backend_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    print(f"✓ Loaded .env from: {env_path}")
else:
    # Also try current directory
    load_dotenv(override=True)
    current_env = os.path.join(os.getcwd(), '.env')
    if os.path.exists(current_env):
        print(f"✓ Loaded .env from: {current_env}")
    else:
        print(f"⚠ Warning: No .env file found in {env_path} or {current_env}")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using in-memory vector store.")

# Load multiple API keys and models from environment
def load_api_keys() -> List[str]:
    """Load API keys from environment variables."""
    keys = []
    # Try GEMINI_API_KEY (single key)
    single_key = os.getenv("GEMINI_API_KEY")
    if single_key:
        keys.append(single_key)
    
    # Try GEMINI_API_KEYS (comma-separated list)
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    if keys_str:
        keys.extend([k.strip() for k in keys_str.split(",") if k.strip()])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    
    return unique_keys

def load_models() -> List[str]:
    """Load model names from environment variables."""
    models = []
    # Try GEMINI_MODEL (single model)
    single_model = os.getenv("GEMINI_MODEL")
    if single_model:
        models.append(single_model)
    
    # Try GEMINI_MODELS (comma-separated list)
    models_str = os.getenv("GEMINI_MODELS", "")
    if models_str:
        models.extend([m.strip() for m in models_str.split(",") if m.strip()])
    
    # Default models if none specified
    if not models:
        models = ["gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for model in models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)
    
    return unique_models

# Load API keys and models
API_KEYS = load_api_keys()
MODELS = load_models()
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
HF_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
USE_HF_EMBEDDINGS = HF_EMBEDDING_MODEL is not None and HF_EMBEDDING_MODEL.strip() != ""

# Debug: Print environment variable status
print(f"DEBUG: HUGGINGFACE_EMBEDDING_MODEL from env: '{HF_EMBEDDING_MODEL}'")
print(f"DEBUG: USE_HF_EMBEDDINGS: {USE_HF_EMBEDDINGS}")
print(f"DEBUG: HF_AVAILABLE: {HF_AVAILABLE}")
print(f"DEBUG: All env vars with HUGGINGFACE: {[k for k in os.environ.keys() if 'HUGGINGFACE' in k.upper()]}")

if API_KEYS:
    # Initialize with first key
    genai.configure(api_key=API_KEYS[0])
    print(f"Loaded {len(API_KEYS)} API key(s) and {len(MODELS)} model(s)")
else:
    print("Warning: No GEMINI_API_KEY or GEMINI_API_KEYS found in environment variables.")

# Initialize Hugging Face embedding model if specified
hf_embedding_model = None
if USE_HF_EMBEDDINGS:
    if HF_AVAILABLE:
        try:
            print(f"Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
            hf_embedding_model = SentenceTransformer(HF_EMBEDDING_MODEL)
            print(f"✓ Hugging Face embedding model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load Hugging Face embedding model: {e}")
            print(f"ERROR: Model name was: {HF_EMBEDDING_MODEL}")
            import traceback
            traceback.print_exc()
            hf_embedding_model = None
    else:
        print(f"ERROR: HUGGINGFACE_EMBEDDING_MODEL specified but sentence-transformers not available.")
        print(f"ERROR: Please install with: pip install sentence-transformers")
else:
    print(f"ERROR: HUGGINGFACE_EMBEDDING_MODEL not found in environment variables.")
    print(f"ERROR: Please set HUGGINGFACE_EMBEDDING_MODEL in your .env file.")


class PortfolioRAGSystem:
    """RAG system for portfolio Q&A with API key and model rotation."""
    
    def __init__(self):
        self.embedding_model_name = EMBEDDING_MODEL
        self.use_hf_embeddings = USE_HF_EMBEDDINGS and hf_embedding_model is not None
        self.hf_embedding_model = hf_embedding_model
        self.api_keys = API_KEYS.copy() if API_KEYS else []
        self.models = MODELS.copy() if MODELS else []
        self.current_key_index = 0
        self.current_model_index = 0
        self.vector_store = None
        self.chunks = []
        self.chunk_metadata = []
        self.portfolio_id_to_index = {}
        
        # Initialize with first available key and model (for LLM only)
        if self.api_keys:
            try:
                genai.configure(api_key=self.api_keys[self.current_key_index])
                print(f"Initialized with API key {self.current_key_index + 1}/{len(self.api_keys)} (for LLM only)")
            except Exception as e:
                print(f"Error initializing Gemini API: {e}")
        
        # Require Hugging Face embeddings
        # Check if env var is set first
        if not USE_HF_EMBEDDINGS:
            error_msg = f"ERROR: HUGGINGFACE_EMBEDDING_MODEL not set in .env file."
            print(error_msg)
            print(f"ERROR: Please set HUGGINGFACE_EMBEDDING_MODEL=all-MiniLM-L6-v2 in your .env file")
            raise ValueError(error_msg)
        
        # Check if model loaded successfully
        if not self.hf_embedding_model:
            if not HF_AVAILABLE:
                error_msg = f"ERROR: sentence-transformers library not installed. Please install with: pip install sentence-transformers"
            else:
                error_msg = f"ERROR: Failed to load Hugging Face embedding model '{HF_EMBEDDING_MODEL}'. Please check that the model name is correct and you have internet connection to download it."
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"✓ Using Hugging Face embeddings: {HF_EMBEDDING_MODEL}")
        print("✓ Gemini API will only be used for LLM response generation.")
    
    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota limit error."""
        error_str = str(error).lower()
        quota_keywords = [
            "quota",
            "rate limit",
            "429",
            "resource exhausted",
            "quota exceeded",
            "too many requests"
        ]
        return any(keyword in error_str for keyword in quota_keywords)
    
    def _rotate_api_key(self) -> bool:
        """Rotate to next API key. Returns True if rotation successful."""
        if len(self.api_keys) <= 1:
            return False
        
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        try:
            genai.configure(api_key=self.api_keys[self.current_key_index])
            print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
            return True
        except Exception as e:
            print(f"Error rotating API key: {e}")
            return False
    
    def _rotate_model(self) -> bool:
        """Rotate to next model. Returns True if rotation successful."""
        if len(self.models) <= 1:
            return False
        
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        print(f"Rotated to model {self.current_model_index + 1}/{len(self.models)}: {self.models[self.current_model_index]}")
        return True
    
    def _get_current_model(self) -> str:
        """Get current model name."""
        if self.models:
            return self.models[self.current_model_index]
        return "gemini-2.0-flash-lite"
    
    def generate_portfolio_output(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured JSON output from portfolio analysis.
        
        Args:
            analysis_result: Result from portfolio analysis endpoint
            
        Returns:
            Structured JSON with all portfolio metrics
        """
        if not analysis_result.get("ok"):
            return {}
        
        output = {
            "portfolio_id": self._generate_portfolio_id(analysis_result),
            "timestamp": datetime.now().isoformat(),
            "tickers": analysis_result.get("universe", []),
            "optimization_method": analysis_result.get("config", {}).get("optimization_method", "unknown"),
            "risk_model": analysis_result.get("config", {}).get("risk_model", "unknown"),
            "weights": {
                "current": analysis_result.get("weights", {}).get("current", []),
                "top_holdings": sorted(
                    analysis_result.get("weights", {}).get("current", []),
                    key=lambda x: x.get("weight", 0),
                    reverse=True
                )[:5]
            },
            "risk": {
                "annualized_volatility": analysis_result.get("metrics", {}).get("annualized_volatility", 0),
                "var_95": analysis_result.get("metrics", {}).get("var_95", 0),
                "cvar_95": analysis_result.get("metrics", {}).get("cvar_95", 0),
                "max_drawdown": analysis_result.get("metrics", {}).get("max_drawdown", 0),
            },
            "performance": {
                "total_return": analysis_result.get("metrics", {}).get("total_return", 0),
                "annualized_return": analysis_result.get("metrics", {}).get("annualized_return", 0),
                "sharpe_ratio": analysis_result.get("metrics", {}).get("sharpe_ratio", 0),
                "avg_turnover": analysis_result.get("metrics", {}).get("avg_turnover", 0),
            },
            "date_range": {
                "start_date": analysis_result.get("config", {}).get("start_date", ""),
                "end_date": analysis_result.get("config", {}).get("end_date", ""),
            }
        }
        
        return output
    
    def _generate_portfolio_id(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a unique portfolio ID based on configuration."""
        config = analysis_result.get("config", {})
        key_str = f"{sorted(analysis_result.get('universe', []))}_{config.get('optimization_method')}_{config.get('risk_model')}_{config.get('start_date')}_{config.get('end_date')}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def convert_to_text_chunks(self, portfolio_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert portfolio output into 3-6 text chunks.
        
        Returns:
            List of chunk dictionaries with 'text' and 'type' fields
        """
        chunks = []
        
        # Chunk 1: Allocation
        weights = portfolio_output.get("weights", {}).get("current", [])
        top_holdings = portfolio_output.get("weights", {}).get("top_holdings", [])
        allocation_text = f"Portfolio Allocation: The portfolio contains {len(weights)} assets. "
        if top_holdings:
            holdings_list = []
            for h in top_holdings[:5]:
                asset = h.get("asset", "")
                weight = h.get("weight", 0) * 100
                holdings_list.append(f"{asset} ({weight:.1f}%)")
            allocation_text += f"Top holdings: {', '.join(holdings_list)}. "
        allocation_text += f"Optimization method: {portfolio_output.get('optimization_method', 'unknown')}."
        
        chunks.append({
            "text": allocation_text,
            "type": "allocation",
            "portfolio_id": portfolio_output.get("portfolio_id")
        })
        
        # Chunk 2: Risk Metrics
        risk = portfolio_output.get("risk", {})
        risk_text = f"Risk Metrics: Annualized volatility is {risk.get('annualized_volatility', 0)*100:.2f}%. "
        risk_text += f"Maximum drawdown is {risk.get('max_drawdown', 0)*100:.2f}%. "
        risk_text += f"Value at Risk (95%) is {risk.get('var_95', 0)*100:.2f}%. "
        risk_text += f"Conditional VaR (95%) is {risk.get('cvar_95', 0)*100:.2f}%. "
        risk_text += f"Risk model used: {portfolio_output.get('risk_model', 'unknown')}."
        
        chunks.append({
            "text": risk_text,
            "type": "risk",
            "portfolio_id": portfolio_output.get("portfolio_id")
        })
        
        # Chunk 3: Performance Metrics
        perf = portfolio_output.get("performance", {})
        perf_text = f"Performance Metrics: Total return is {perf.get('total_return', 0)*100:.2f}%. "
        perf_text += f"Annualized return is {perf.get('annualized_return', 0)*100:.2f}%. "
        perf_text += f"Sharpe ratio is {perf.get('sharpe_ratio', 0):.2f}. "
        if perf.get('avg_turnover'):
            perf_text += f"Average turnover is {perf.get('avg_turnover', 0)*100:.2f}%."
        
        chunks.append({
            "text": perf_text,
            "type": "performance",
            "portfolio_id": portfolio_output.get("portfolio_id")
        })
        
        # Chunk 4: Portfolio Composition
        tickers = portfolio_output.get("tickers", [])
        comp_text = f"Portfolio Composition: The portfolio consists of {len(tickers)} assets: {', '.join(tickers[:10])}"
        if len(tickers) > 10:
            comp_text += f" and {len(tickers) - 10} more."
        else:
            comp_text += "."
        
        chunks.append({
            "text": comp_text,
            "type": "composition",
            "portfolio_id": portfolio_output.get("portfolio_id")
        })
        
        # Chunk 5: Strategy Summary
        strategy_text = f"Strategy Summary: Using {portfolio_output.get('optimization_method', 'unknown')} optimization with {portfolio_output.get('risk_model', 'unknown')} risk model. "
        date_range = portfolio_output.get("date_range", {})
        if date_range.get("start_date") and date_range.get("end_date"):
            strategy_text += f"Analysis period: {date_range.get('start_date')} to {date_range.get('end_date')}."
        
        chunks.append({
            "text": strategy_text,
            "type": "strategy",
            "portfolio_id": portfolio_output.get("portfolio_id")
        })
        
        # Chunk 6: Overall Assessment (if we have enough data)
        if perf.get('sharpe_ratio') and risk.get('annualized_volatility'):
            assessment_text = f"Overall Assessment: Portfolio achieved a Sharpe ratio of {perf.get('sharpe_ratio', 0):.2f} "
            assessment_text += f"with {risk.get('annualized_volatility', 0)*100:.2f}% volatility. "
            if perf.get('sharpe_ratio', 0) > 1.0:
                assessment_text += "The Sharpe ratio above 1.0 indicates good risk-adjusted returns."
            elif perf.get('sharpe_ratio', 0) > 0.5:
                assessment_text += "The Sharpe ratio between 0.5 and 1.0 indicates moderate risk-adjusted returns."
            else:
                assessment_text += "The Sharpe ratio below 0.5 indicates lower risk-adjusted returns."
            
            chunks.append({
                "text": assessment_text,
                "type": "assessment",
                "portfolio_id": portfolio_output.get("portfolio_id")
            })
        
        return chunks
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Embed text chunks using Hugging Face embeddings only.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of embedding vectors
        """
        if not self.use_hf_embeddings or not self.hf_embedding_model:
            error_msg = "Hugging Face embedding model not available. Please set HUGGINGFACE_EMBEDDING_MODEL in .env file."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings_list = self.hf_embedding_model.encode(texts, convert_to_numpy=True)
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]
            return embeddings
        except Exception as e:
            error_msg = f"Error using Hugging Face embeddings: {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def store_portfolio(self, analysis_result: Dict[str, Any]) -> str:
        """
        Process and store a portfolio analysis result.
        
        Args:
            analysis_result: Result from portfolio analysis endpoint
            
        Returns:
            Portfolio ID
        """
        # Generate structured output
        portfolio_output = self.generate_portfolio_output(analysis_result)
        if not portfolio_output:
            raise ValueError("Failed to generate portfolio output")
        
        portfolio_id = portfolio_output.get("portfolio_id")
        
        # Convert to chunks
        chunks = self.convert_to_text_chunks(portfolio_output)
        
        # Embed chunks using Hugging Face only
        try:
            embeddings = self.embed_chunks(chunks)
        except Exception as e:
            error_msg = f"Failed to embed chunks: {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
        
        # Store in vector store
        if FAISS_AVAILABLE and len(embeddings) > 0:
            # Initialize or update FAISS index
            embedding_dim = len(embeddings[0])
            
            # If vector store exists but has different dimension, recreate it
            if self.vector_store is None:
                self.vector_store = faiss.IndexFlatL2(embedding_dim)
            elif hasattr(self.vector_store, 'd') and self.vector_store.d != embedding_dim:
                # Dimension mismatch, recreate index
                print(f"Warning: Embedding dimension mismatch ({self.vector_store.d} vs {embedding_dim}). Recreating index.")
                self.vector_store = faiss.IndexFlatL2(embedding_dim)
            
            # Convert embeddings to numpy array
            embeddings_array = np.vstack(embeddings).astype('float32')
            start_idx = len(self.chunks)
            
            # Add to FAISS index
            self.vector_store.add(embeddings_array)
            
            # Store chunks and metadata
            self.chunks.extend([chunk["text"] for chunk in chunks])
            self.chunk_metadata.extend([
                {
                    "type": chunk["type"],
                    "portfolio_id": portfolio_id,
                    "index": start_idx + i
                }
                for i, chunk in enumerate(chunks)
            ])
        else:
            # In-memory storage (fallback)
            self.chunks.extend([chunk["text"] for chunk in chunks])
            self.chunk_metadata.extend([
                {
                    "type": chunk["type"],
                    "portfolio_id": portfolio_id,
                    "index": len(self.chunks) - len(chunks) + i
                }
                for i, chunk in enumerate(chunks)
            ])
        
        # Store portfolio ID mapping
        self.portfolio_id_to_index[portfolio_id] = {
            "start": len(self.chunks) - len(chunks),
            "end": len(self.chunks)
        }
        
        return portfolio_id
    
    def retrieve_chunks(self, question: str, portfolio_id: Optional[str] = None, top_k: int = 4) -> List[str]:
        """
        Retrieve relevant chunks for a question using Hugging Face embeddings only.
        
        Args:
            question: User's question
            portfolio_id: Optional portfolio ID to filter chunks
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunk texts
        """
        if len(self.chunks) == 0:
            return []
        
        # Use Hugging Face embeddings only
        if not self.use_hf_embeddings or not self.hf_embedding_model:
            error_msg = "Hugging Face embedding model not available. Please set HUGGINGFACE_EMBEDDING_MODEL in .env file."
            print(f"ERROR: {error_msg}")
            return []
        
        try:
            query_embedding = self.hf_embedding_model.encode(question, convert_to_numpy=True)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            error_msg = f"Error using Hugging Face embeddings for question: {e}"
            print(f"ERROR: {error_msg}")
            return []
        
        # If we get here, we successfully got query_embedding
        try:
            # Filter chunks by portfolio_id if provided
            if portfolio_id and portfolio_id in self.portfolio_id_to_index:
                portfolio_range = self.portfolio_id_to_index[portfolio_id]
                relevant_indices = list(range(portfolio_range["start"], portfolio_range["end"]))
            else:
                relevant_indices = list(range(len(self.chunks)))
            
            if not relevant_indices:
                return []
            
            # Search in vector store
            if FAISS_AVAILABLE and self.vector_store is not None:
                # For FAISS, we need to search within the relevant indices
                # Create a subset index with only relevant embeddings
                if hasattr(self.vector_store, 'reconstruct') and len(relevant_indices) > 0:
                    # Try to reconstruct embeddings for relevant indices
                    try:
                        relevant_embeddings = np.vstack([
                            self.vector_store.reconstruct(i) for i in relevant_indices
                        ]).astype('float32')
                        
                        # Create a temporary index for search
                        temp_index = faiss.IndexFlatL2(query_embedding.shape[1])
                        temp_index.add(relevant_embeddings)
                        
                        # Search
                        distances, indices = temp_index.search(query_embedding, min(top_k, len(relevant_indices)))
                        
                        # Map back to original chunk indices
                        retrieved_indices = [relevant_indices[idx] for idx in indices[0]]
                    except Exception as e:
                        print(f"Error with FAISS reconstruction: {e}")
                        # Fallback to simple selection
                        retrieved_indices = relevant_indices[:top_k]
                else:
                    # If we can't reconstruct, search the full index and filter
                    # This is less efficient but works
                    distances, indices = self.vector_store.search(query_embedding, min(top_k * 2, self.vector_store.ntotal))
                    # Filter to only relevant indices
                    retrieved_indices = [idx for idx in indices[0] if idx in relevant_indices][:top_k]
            else:
                # Simple selection fallback (no vector search available)
                retrieved_indices = relevant_indices[:top_k]
            
            # Return chunk texts
            return [self.chunks[idx] for idx in retrieved_indices if idx < len(self.chunks)]
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            # Fallback: return first few chunks
            if portfolio_id and portfolio_id in self.portfolio_id_to_index:
                portfolio_range = self.portfolio_id_to_index[portfolio_id]
                return self.chunks[portfolio_range["start"]:portfolio_range["end"]][:top_k]
            return self.chunks[:top_k]
    
    def query_llm(self, question: str, context_chunks: List[str]) -> str:
        """
        Query Gemini LLM with retrieved context, with quota error handling and rotation.
        
        Args:
            question: User's question
            context_chunks: Retrieved context chunks
            
        Returns:
            LLM response
        """
        if not self.api_keys or not self.models:
            return "LLM not initialized. Please check your API key and model configuration."
        
        # Build prompt with updated instructions
        system_message = """You are a helpful financial assistant specializing in portfolio analysis.

IMPORTANT RULES:
1. When answering numeric, performance, or portfolio-specific factual questions (e.g., "What is the Sharpe ratio?", "What is my portfolio return?", "What risk model did I use?"), use ONLY the provided context. If the information is not in the context, explicitly state: "The provided data does not include that information." Never guess or make up numbers or metrics.

2. When the user asks for interpretation, guidance, or how to use the results (e.g., "What does this Sharpe ratio mean?", "How should I interpret this?", "What should I do with this portfolio?"), you may provide general investment reasoning and guidance as long as you do not invent new numbers or metrics. You can explain concepts, provide context, and give general advice based on investment principles.

3. Always clearly distinguish between information from the provided context and general knowledge/explanations you provide."""
        
        context_text = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""{system_message}

Context Information:
{context_text}

User Question: {question}

Answer:"""
        
        max_retries = len(self.api_keys) * len(self.models) * 2  # Try all combinations
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get current model
                current_model_name = self._get_current_model()
                llm_model = genai.GenerativeModel(current_model_name)
                
                response = llm_model.generate_content(prompt)
                
                if hasattr(response, 'text'):
                    return response.text
                elif isinstance(response, dict) and 'text' in response:
                    return response['text']
                else:
                    return str(response)
                    
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"Quota error querying LLM (attempt {retry_count + 1}): {e}")
                    
                    # Try rotating model first
                    if self._rotate_model():
                        retry_count += 1
                        continue
                    
                    # If model rotation didn't help or we've tried all models, rotate API key
                    if self._rotate_api_key():
                        # Reset model index when rotating key
                        self.current_model_index = 0
                        retry_count += 1
                        continue
                    else:
                        # All keys and models exhausted
                        return "All API keys and models have reached their quota limits. Please try again later."
                else:
                    # Non-quota error
                    return f"Error generating response: {str(e)}"
        
        return "Max retries reached. Please try again later."


# Global RAG system instance
_rag_system = None

def get_rag_system() -> PortfolioRAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        _rag_system = PortfolioRAGSystem()
    return _rag_system

