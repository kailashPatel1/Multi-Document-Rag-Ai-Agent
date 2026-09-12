import os
import uuid
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from ...config.settings import settings
from ...models.db_models import get_db, DocumentModel, DocumentChunkModel
from ...models.schemas import DocumentResponse, DocumentChunkResponse
from ...rag.document_loader import DocumentLoader, DocumentLoaderError
from ...rag.chunker import DocumentChunker
from ...retrievers.vector_store import vector_store_manager
from ...retrievers.bm25_retriever import bm25_retriever_manager
from ...services.sample_generator import generate_sample_documents

router = APIRouter(prefix="/api/documents", tags=["Documents"])
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Uploads, validates, parses, chunks, and indexes multiple documents into ChromaDB and BM25."""
    uploaded_docs = []
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for upload_file in files:
        filename = upload_file.filename
        ext = Path(filename).suffix.lower()

        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Save to disk
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        safe_filename = f"{doc_id}_{filename}"
        dest_path = settings.UPLOADS_DIR / safe_filename

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        # Validate file size
        file_size = dest_path.stat().st_size
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB.")

        try:
            # 1. Load and parse document pages
            raw_docs, meta = DocumentLoader.load_file(str(dest_path), original_filename=filename)

            # 2. Chunk documents with metadata
            chunks = chunker.chunk_documents(raw_docs, document_id=doc_id, document_name=filename)

            # 3. Index in ChromaDB Vector Store
            vector_store_manager.add_documents(chunks)

            # 4. Add to BM25 Keyword Retriever
            bm25_retriever_manager.add_chunks(chunks)

            # 5. Persist Document & Chunks in SQLite
            doc_record = DocumentModel(
                id=doc_id,
                filename=safe_filename,
                original_name=filename,
                file_type=ext,
                file_path=str(dest_path),
                file_size=file_size,
                total_pages=meta.get("total_pages", 1),
                total_chunks=len(chunks),
                columns_json=json.dumps(meta.get("columns")) if meta.get("columns") else None,
                status="ready"
            )
            db.add(doc_record)

            for idx, c in enumerate(chunks):
                chunk_model = DocumentChunkModel(
                    id=c.metadata.get("chunk_id", f"{doc_id}_c{idx}"),
                    document_id=doc_id,
                    document_name=filename,
                    page_number=c.metadata.get("page_number", 1),
                    chunk_index=idx,
                    content=c.page_content,
                    char_count=len(c.page_content),
                    metadata_json=json.dumps(c.metadata)
                )
                db.add(chunk_model)

            db.commit()
            db.refresh(doc_record)

            uploaded_docs.append(DocumentResponse(
                id=doc_record.id,
                filename=doc_record.filename,
                original_name=doc_record.original_name,
                file_type=doc_record.file_type,
                file_size=doc_record.file_size,
                total_pages=doc_record.total_pages,
                total_chunks=doc_record.total_chunks,
                created_at=doc_record.created_at,
                status=doc_record.status,
                column_names=meta.get("columns")
            ))

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            db.rollback()
            dest_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to process '{filename}': {str(e)}")

    return uploaded_docs

@router.get("", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """Lists all registered documents in the knowledge base."""
    docs = db.query(DocumentModel).order_by(DocumentModel.created_at.desc()).all()
    results = []
    for d in docs:
        cols = json.loads(d.columns_json) if d.columns_json else None
        results.append(DocumentResponse(
            id=d.id,
            filename=d.filename,
            original_name=d.original_name,
            file_type=d.file_type,
            file_size=d.file_size,
            total_pages=d.total_pages,
            total_chunks=d.total_chunks,
            created_at=d.created_at,
            status=d.status,
            summary=d.summary,
            column_names=cols
        ))
    return results

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    cols = json.loads(doc.columns_json) if doc.columns_json else None
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        original_name=doc.original_name,
        file_type=doc.file_type,
        file_size=doc.file_size,
        total_pages=doc.total_pages,
        total_chunks=doc.total_chunks,
        created_at=doc.created_at,
        status=doc.status,
        summary=doc.summary,
        column_names=cols
    )

@router.get("/{document_id}/chunks", response_model=List[DocumentChunkResponse])
def get_document_chunks(document_id: str, db: Session = Depends(get_db)):
    chunks = db.query(DocumentChunkModel).filter(
        DocumentChunkModel.document_id == document_id
    ).order_by(DocumentChunkModel.chunk_index.asc()).all()

    return [
        DocumentChunkResponse(
            chunk_id=c.id,
            document_id=c.document_id,
            document_name=c.document_name,
            page_number=c.page_number,
            content=c.content,
            char_count=c.char_count,
            metadata=json.loads(c.metadata_json or "{}")
        )
        for c in chunks
    ]

@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # 1. Delete Chroma vectors
    vector_store_manager.delete_document_chunks(document_id)

    # 2. Delete BM25 chunks
    bm25_retriever_manager.remove_document(document_id)

    # 3. Delete file on disk
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass

    # 4. Delete SQLite records
    db.delete(doc)
    db.commit()

    return {"status": "success", "message": f"Document '{doc.original_name}' deleted."}

@router.post("/load-demo-files", response_model=List[DocumentResponse])
def load_demo_files(db: Session = Depends(get_db)):
    """Generates the 5 realistic demo documents and automatically indexes them into the knowledge base."""
    file_map = generate_sample_documents()
    chunker = DocumentChunker()
    results = []

    for key, fpath_str in file_map.items():
        fpath = Path(fpath_str)
        orig_name = fpath.name

        # Check if already indexed
        existing = db.query(DocumentModel).filter(DocumentModel.original_name == orig_name).first()
        if existing:
            cols = json.loads(existing.columns_json) if existing.columns_json else None
            results.append(DocumentResponse(
                id=existing.id,
                filename=existing.filename,
                original_name=existing.original_name,
                file_type=existing.file_type,
                file_size=existing.file_size,
                total_pages=existing.total_pages,
                total_chunks=existing.total_chunks,
                created_at=existing.created_at,
                status=existing.status,
                column_names=cols
            ))
            continue

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        safe_filename = f"{doc_id}_{orig_name}"
        dest = settings.UPLOADS_DIR / safe_filename
        shutil.copyfile(fpath, dest)

        raw_docs, meta = DocumentLoader.load_file(str(dest), original_filename=orig_name)
        chunks = chunker.chunk_documents(raw_docs, document_id=doc_id, document_name=orig_name)

        vector_store_manager.add_documents(chunks)
        bm25_retriever_manager.add_chunks(chunks)

        doc_record = DocumentModel(
            id=doc_id,
            filename=safe_filename,
            original_name=orig_name,
            file_type=fpath.suffix.lower(),
            file_path=str(dest),
            file_size=dest.stat().st_size,
            total_pages=meta.get("total_pages", 1),
            total_chunks=len(chunks),
            columns_json=json.dumps(meta.get("columns")) if meta.get("columns") else None,
            status="ready"
        )
        db.add(doc_record)

        for idx, c in enumerate(chunks):
            db.add(DocumentChunkModel(
                id=c.metadata.get("chunk_id", f"{doc_id}_c{idx}"),
                document_id=doc_id,
                document_name=orig_name,
                page_number=c.metadata.get("page_number", 1),
                chunk_index=idx,
                content=c.page_content,
                char_count=len(c.page_content),
                metadata_json=json.dumps(c.metadata)
            ))

        db.commit()
        db.refresh(doc_record)

        results.append(DocumentResponse(
            id=doc_record.id,
            filename=doc_record.filename,
            original_name=doc_record.original_name,
            file_type=doc_record.file_type,
            file_size=doc_record.file_size,
            total_pages=doc_record.total_pages,
            total_chunks=doc_record.total_chunks,
            created_at=doc_record.created_at,
            status=doc_record.status,
            column_names=meta.get("columns")
        ))

    return results
