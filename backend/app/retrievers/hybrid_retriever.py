import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from .vector_store import vector_store_manager
from .bm25_retriever import bm25_retriever_manager
from ..config.settings import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines Dense Vector Semantic Search with BM25 Lexical Search via Reciprocal Rank Fusion."""

    def __init__(self, default_alpha: Optional[float] = None):
        self.default_alpha = default_alpha if default_alpha is not None else settings.HYBRID_ALPHA

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        hybrid_alpha: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float, str]]:
        """
        Executes Hybrid Retrieval.
        Returns List of (Document, combined_score, retrieval_method).
        """
        alpha = hybrid_alpha if hybrid_alpha is not None else self.default_alpha
        alpha = max(0.0, min(1.0, alpha))
        
        q_lower = query.lower()
        # Only include CSV tabular data if the query explicitly mentions csv, dataset, revenue, sales
        is_tabular_query = any(k in q_lower for k in ["csv", "sales", "revenue", "dataset", "transaction", "unit price"])
        exclude_tabular = not is_tabular_query

        fetch_k = max(top_k * 2, 8)
        
        vector_results = []
        if alpha > 0.0:
            try:
                vector_results = vector_store_manager.similarity_search_with_score(
                    query=query,
                    k=fetch_k,
                    document_ids=document_ids,
                    exclude_tabular=exclude_tabular,
                    filter_metadata=filter_metadata
                )
            except Exception as e:
                logger.error(f"Vector search error: {e}")

        bm25_results = []
        if alpha < 1.0:
            try:
                bm25_results = bm25_retriever_manager.search(
                    query=query,
                    k=fetch_k,
                    document_ids=document_ids,
                    exclude_tabular=exclude_tabular
                )
            except Exception as e:
                logger.error(f"BM25 search error: {e}")

        if not vector_results and not bm25_results:
            return []
        if not bm25_results:
            return [(doc, score, "vector") for doc, score in vector_results[:top_k]]
        if not vector_results:
            return [(doc, score, "bm25") for doc, score in bm25_results[:top_k]]

        # Reciprocal Rank Fusion (RRF)
        RRF_K = 60
        fused_scores: Dict[str, Dict[str, Any]] = {}

        for rank, (doc, v_score) in enumerate(vector_results):
            c_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "document": doc,
                    "vector_rank": rank,
                    "vector_score": v_score,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            else:
                fused_scores[c_id]["vector_rank"] = rank
                fused_scores[c_id]["vector_score"] = v_score

            fused_scores[c_id]["rrf_score"] += alpha * (1.0 / (RRF_K + rank + 1))

        for rank, (doc, b_score) in enumerate(bm25_results):
            c_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "document": doc,
                    "vector_rank": None,
                    "vector_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": b_score,
                    "rrf_score": 0.0,
                }
            else:
                fused_scores[c_id]["bm25_rank"] = rank
                fused_scores[c_id]["bm25_score"] = b_score

            fused_scores[c_id]["rrf_score"] += (1.0 - alpha) * (1.0 / (RRF_K + rank + 1))

        combined_items = []
        for c_id, data in fused_scores.items():
            doc = data["document"]
            v_score = data["vector_score"]
            b_score = data["bm25_score"]
            
            # Weighted raw similarity
            linear_score = (alpha * v_score) + ((1.0 - alpha) * b_score)
            
            if data["vector_rank"] is not None and data["bm25_rank"] is not None:
                method = "hybrid"
                final_score = min(1.0, linear_score * 1.15)
            elif data["vector_rank"] is not None:
                method = "vector"
                final_score = v_score
            else:
                method = "bm25"
                final_score = b_score

            combined_items.append((doc, float(round(final_score, 4)), method, data["rrf_score"]))

        # Sort primarily by RRF, secondary by final score
        combined_items.sort(key=lambda x: (x[3], x[1]), reverse=True)
        
        return [(doc, score, method) for doc, score, method, _ in combined_items[:top_k]]

hybrid_retriever = HybridRetriever()
