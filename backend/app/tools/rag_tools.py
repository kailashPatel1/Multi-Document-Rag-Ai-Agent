import logging
from typing import Dict, Any, List, Optional
from ..retrievers.hybrid_retriever import hybrid_retriever
from ..retrievers.reranker import reranker
from ..rag.citation import citation_engine
from ..models.db_models import SessionLocal, DocumentModel, DocumentChunkModel

logger = logging.getLogger(__name__)

def rag_search_tool(query: str, document_ids: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    RAG Search Tool: Searches across uploaded documents using Hybrid Retrieval (Vector + BM25).
    Returns grounded context passages and exact document page citations.
    """
    raw_chunks = hybrid_retriever.retrieve(query=query, top_k=top_k * 2, document_ids=document_ids)
    ranked = reranker.rerank_and_filter(query=query, retrieved_items=raw_chunks, top_k=top_k)
    citations = citation_engine.build_citations(ranked)

    passages = []
    for doc, score, method, grade in ranked:
        meta = doc.metadata
        passages.append(
            f"[Source: {meta.get('document_name', 'Doc')} | Page {meta.get('page_number', 1)} | Relevance: {int(score*100)}%]\n"
            f"{doc.page_content.strip()}"
        )

    context_text = "\n\n---\n\n".join(passages) if passages else "No matching document passages found."

    return {
        "query": query,
        "total_chunks_found": len(ranked),
        "context": context_text,
        "citations": [c.dict() for c in citations]
    }

def document_summary_tool(document_name_or_id: str, focus: Optional[str] = "general") -> Dict[str, Any]:
    """
    Document Summary Tool: Summarizes the contents, major topics, and structure of a specific uploaded document.
    """
    db = SessionLocal()
    try:
        # Find document by ID or partial filename match
        doc = db.query(DocumentModel).filter(
            (DocumentModel.id == document_name_or_id) |
            (DocumentModel.filename.ilike(f"%{document_name_or_id}%")) |
            (DocumentModel.original_name.ilike(f"%{document_name_or_id}%"))
        ).first()

        if not doc:
            # Fallback list all available documents
            all_docs = db.query(DocumentModel).all()
            doc_names = [d.original_name for d in all_docs]
            return {
                "error": f"Document '{document_name_or_id}' not found.",
                "available_documents": doc_names
            }

        # Gather representative chunks (beginning, middle, end)
        chunks = db.query(DocumentChunkModel).filter(
            DocumentChunkModel.document_id == doc.id
        ).order_by(DocumentChunkModel.chunk_index.asc()).all()

        sample_texts = [c.content for c in chunks[:8]]  # first 8 chunks
        combined = "\n\n".join(sample_texts)

        return {
            "document_id": doc.id,
            "document_name": doc.original_name,
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "total_chunks": doc.total_chunks,
            "document_content_sample": combined[:3500]
        }
    finally:
        db.close()

def document_comparison_tool(document_a_name: str, document_b_name: str, comparison_query: str) -> Dict[str, Any]:
    """
    Multi-Document Comparison Tool: Performs comparative reasoning across two documents
    (e.g., Resume vs Job Description, Paper 1 vs Paper 2).
    Retrieves targeted passages from BOTH documents to analyze overlap, gaps, and distinctions.
    """
    db = SessionLocal()
    try:
        doc_a = db.query(DocumentModel).filter(
            (DocumentModel.original_name.ilike(f"%{document_a_name}%")) | (DocumentModel.id == document_a_name)
        ).first()
        doc_b = db.query(DocumentModel).filter(
            (DocumentModel.original_name.ilike(f"%{document_b_name}%")) | (DocumentModel.id == document_b_name)
        ).first()

        results: Dict[str, Any] = {
            "comparison_query": comparison_query,
            "doc_a_name": doc_a.original_name if doc_a else document_a_name,
            "doc_b_name": doc_b.original_name if doc_b else document_b_name,
            "doc_a_chunks": [],
            "doc_b_chunks": [],
            "citations": []
        }

        all_citations = []

        if doc_a:
            raw_a = hybrid_retriever.retrieve(query=comparison_query, top_k=4, document_ids=[doc_a.id])
            ranked_a = reranker.rerank_and_filter(query=comparison_query, retrieved_items=raw_a, top_k=4)
            cites_a = citation_engine.build_citations(ranked_a)
            all_citations.extend(cites_a)
            results["doc_a_chunks"] = [f"[Page {d.metadata.get('page_number', 1)}]: {d.page_content}" for d, _, _, _ in ranked_a]

        if doc_b:
            raw_b = hybrid_retriever.retrieve(query=comparison_query, top_k=4, document_ids=[doc_b.id])
            ranked_b = reranker.rerank_and_filter(query=comparison_query, retrieved_items=raw_b, top_k=4)
            cites_b = citation_engine.build_citations(ranked_b)
            all_citations.extend(cites_b)
            results["doc_b_chunks"] = [f"[Page {d.metadata.get('page_number', 1)}]: {d.page_content}" for d, _, _, _ in ranked_b]

        results["citations"] = [c.dict() for c in all_citations]
        return results
    finally:
        db.close()

def metadata_search_tool(query: str = "") -> Dict[str, Any]:
    """
    Metadata / Document Directory Tool: Lists all registered documents in the knowledge base,
    their file types, sizes, page counts, and chunk statistics.
    """
    db = SessionLocal()
    try:
        docs = db.query(DocumentModel).all()
        doc_list = []
        for d in docs:
            doc_list.append({
                "id": d.id,
                "name": d.original_name,
                "file_type": d.file_type,
                "total_pages": d.total_pages,
                "total_chunks": d.total_chunks,
                "status": d.status
            })
        return {
            "total_documents": len(doc_list),
            "documents": doc_list
        }
    finally:
        db.close()
