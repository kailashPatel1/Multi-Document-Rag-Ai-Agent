import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config.settings import settings
from backend.app.models.db_models import init_db, SessionLocal, DocumentChunkModel
from backend.app.retrievers.vector_store import vector_store_manager
from backend.app.retrievers.bm25_retriever import bm25_retriever_manager
from backend.app.services.sample_generator import generate_sample_documents
from langchain_core.documents import Document
import json

from backend.app.api.routes import (
    auth_router,
    documents_router,
    chat_router,
    search_router,
    tools_router,
    evaluation_router,
    system_router
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MultiDocAIAgent")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup lifecycle
    logger.info("Initializing SQLite database tables...")
    init_db()

    logger.info("Verifying ChromaDB persistent vector collection health...")
    try:
        vector_store_manager._ensure_vector_store()
        vector_store_manager.auto_sync_from_database()
        logger.info(f"ChromaDB ready with {vector_store_manager.get_total_chunks()} indexed vectors.")
    except Exception as e:
        logger.warning(f"ChromaDB startup sync warning: {e}")

    logger.info("Syncing BM25 Index with stored chunks...")
    try:
        db = SessionLocal()
        chunks = db.query(DocumentChunkModel).all()
        doc_chunks = [
            Document(
                page_content=c.content,
                metadata={
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                    **(json.loads(c.metadata_json or "{}"))
                }
            )
            for c in chunks
        ]
        bm25_retriever_manager.rebuild_index(doc_chunks)
        db.close()
        logger.info(f"Loaded {len(doc_chunks)} chunks into BM25 index on startup.")
    except Exception as e:
        logger.warning(f"BM25 startup sync warning: {e}")

    logger.info(f"=== {settings.PROJECT_NAME} v{settings.VERSION} Ready ===")
    yield
    # Shutdown lifecycle
    logger.info("Shutting down Multi-Document AI Knowledge Agent server...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Multi-Document AI Knowledge Agent powered by Hybrid RAG, LangGraph, ChromaDB, and Page Citations.",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(tools_router)
app.include_router(evaluation_router)
app.include_router(system_router)

# Static Frontend mounting
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_SRC = BASE_DIR / "frontend"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    @app.get("/{full_path:path}")
    async def serve_frontend_dist(full_path: str):
        if full_path.startswith("api/"):
            return None
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
elif FRONTEND_SRC.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_SRC)), name="static")
    @app.get("/")
    async def serve_frontend_root():
        return FileResponse(str(FRONTEND_SRC / "index.html"))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "chroma_vectors": vector_store_manager.get_total_chunks()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
