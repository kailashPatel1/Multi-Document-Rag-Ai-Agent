import logging
from typing import List
from fastapi import APIRouter, HTTPException
from ...models.schemas import SearchRequest, SearchResponse, SearchResultChunk
from ...retrievers.hybrid_retriever import hybrid_retriever
from ...retrievers.reranker import reranker

router = APIRouter(prefix="/api/search", tags=["Search & Retrieval"])
logger = logging.getLogger(__name__)

@router.post("", response_model=SearchResponse)
def perform_search(request: SearchRequest):
    """
    Search Sandbox: Performs hybrid retrieval across the knowledge base,
    returning chunk matches, similarity scores, and retrieval methods (vector / bm25 / hybrid).
    """
    try:
        raw_items = hybrid_retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            document_ids=request.document_ids,
            hybrid_alpha=request.hybrid_alpha,
            filter_metadata=request.filter_metadata
        )

        ranked = reranker.rerank_and_filter(
            query=request.query,
            retrieved_items=raw_items,
            top_k=request.top_k
        )

        results = []
        for doc, score, method, grade in ranked:
            meta = doc.metadata
            results.append(SearchResultChunk(
                chunk_id=meta.get("chunk_id", ""),
                document_name=meta.get("document_name", "Unknown"),
                document_id=meta.get("document_id", ""),
                page_number=int(meta.get("page_number", 1)),
                content=doc.page_content,
                score=round(score, 4),
                retrieval_method=method,
                metadata=meta
            ))

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
