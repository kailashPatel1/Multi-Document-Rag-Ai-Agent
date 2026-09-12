import time
import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...config.settings import settings, mask_api_key, save_env_variables
from ...models.db_models import get_db, DocumentModel, DocumentChunkModel
from ...retrievers.vector_store import vector_store_manager
from ...retrievers.bm25_retriever import bm25_retriever_manager

router = APIRouter(prefix="/api/system", tags=["System & Settings"])
logger = logging.getLogger(__name__)

SERVER_START_TIME = time.time()

class SettingsUpdateRequest(BaseModel):
    default_llm_provider: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k_retrieval: Optional[int] = None
    hybrid_alpha: Optional[float] = None

@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    """Returns current system health, vector database statistics, LLM provider, and masked API key status."""
    doc_count = db.query(DocumentModel).count()
    chunk_count = db.query(DocumentChunkModel).count()
    vector_count = vector_store_manager.get_total_chunks()

    has_groq = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
    has_gemini = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())

    return {
        "status": "operational",
        "version": settings.VERSION,
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "vector_chunks_indexed": vector_count,
        "vector_db_status": "connected",
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "llm_model": settings.GROQ_MODEL if settings.DEFAULT_LLM_PROVIDER == "groq" else settings.GEMINI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "has_groq_key": has_groq,
        "groq_key_masked": mask_api_key(settings.GROQ_API_KEY),
        "has_gemini_key": has_gemini,
        "gemini_key_masked": mask_api_key(settings.GEMINI_API_KEY),
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "hybrid_alpha": settings.HYBRID_ALPHA,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1)
    }

@router.post("/settings")
def update_settings(req: SettingsUpdateRequest):
    """Updates and securely persists settings to .env file and active runtime."""
    env_updates: Dict[str, Any] = {}

    if req.default_llm_provider:
        env_updates["DEFAULT_LLM_PROVIDER"] = req.default_llm_provider
    if req.groq_api_key is not None and req.groq_api_key.strip():
        env_updates["GROQ_API_KEY"] = req.groq_api_key.strip()
    if req.groq_model:
        env_updates["GROQ_MODEL"] = req.groq_model
    if req.gemini_api_key is not None and req.gemini_api_key.strip():
        env_updates["GEMINI_API_KEY"] = req.gemini_api_key.strip()
    if req.gemini_model:
        env_updates["GEMINI_MODEL"] = req.gemini_model
    if req.chunk_size is not None and req.chunk_size > 0:
        env_updates["CHUNK_SIZE"] = req.chunk_size
    if req.chunk_overlap is not None and req.chunk_overlap >= 0:
        env_updates["CHUNK_OVERLAP"] = req.chunk_overlap
    if req.top_k_retrieval is not None and req.top_k_retrieval > 0:
        env_updates["TOP_K_RETRIEVAL"] = req.top_k_retrieval
    if req.hybrid_alpha is not None:
        env_updates["HYBRID_ALPHA"] = req.hybrid_alpha

    if env_updates:
        save_env_variables(env_updates)
        logger.info(f"Updated configuration variables: {list(env_updates.keys())}")

    has_groq = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
    has_gemini = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())

    return {
        "status": "success",
        "message": "Settings securely saved and persisted.",
        "active_settings": {
            "llm_provider": settings.DEFAULT_LLM_PROVIDER,
            "has_groq_key": has_groq,
            "groq_key_masked": mask_api_key(settings.GROQ_API_KEY),
            "has_gemini_key": has_gemini,
            "gemini_key_masked": mask_api_key(settings.GEMINI_API_KEY),
            "groq_model": settings.GROQ_MODEL,
            "gemini_model": settings.GEMINI_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "hybrid_alpha": settings.HYBRID_ALPHA
        }
    }

@router.post("/reset")
def reset_system(db: Session = Depends(get_db)):
    """Clears all indexed documents, ChromaDB vectors, and chat history."""
    try:
        # 1. Reset Vector Store
        vector_store_manager.reset_collection()
        
        # 2. Reset BM25
        bm25_retriever_manager.rebuild_index([])

        # 3. Clean database tables
        db.query(DocumentChunkModel).delete()
        db.query(DocumentModel).delete()
        db.commit()

        return {"status": "success", "message": "Knowledge base and vector database successfully reset."}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
