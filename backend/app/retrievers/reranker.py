import logging
import re
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ContextReranker:
    """Document-agnostic reranker, deduplicator, and score-gap filter with multi-document support."""

    def __init__(self, min_similarity_threshold: Optional[float] = None):
        self.min_threshold = min_similarity_threshold if min_similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    @staticmethod
    def _text_fingerprint(text: str) -> str:
        """Extracts significant alphanumeric words to detect duplicate chunk content."""
        words = re.findall(r"\b\w{3,}\b", text.lower())
        return " ".join(words[:25])

    def rerank_and_filter(
        self,
        query: str,
        retrieved_items: List[Tuple[Document, float, str]],
        top_k: int = 5,
        target_document_hint: Optional[str] = None
    ) -> List[Tuple[Document, float, str, str]]:
        """
        Reranks and filters candidate chunks:
        1. If user explicitly specified a document hint, boosts chunks from that document.
        2. Deduplicates chunks with identical or near-identical text.
        3. Preserves multi-document diversity while filtering low-quality noise.
        4. Assigns High/Medium/Low relevance grades.
        """
        if not retrieved_items:
            return []

        # 1. Apply Dynamic Document Hint Boost if user explicitly mentioned a document
        adjusted_items: List[Tuple[Document, float, str]] = []
        for doc, score, method in retrieved_items:
            d_name = doc.metadata.get("document_name", "").lower()
            adj_score = score
            if target_document_hint and target_document_hint.lower() in d_name:
                adj_score = min(1.0, adj_score * 1.20)
            adjusted_items.append((doc, adj_score, method))

        adjusted_items.sort(key=lambda x: x[1], reverse=True)

        seen_fingerprints = set()
        filtered: List[Tuple[Document, float, str, str]] = []
        doc_counts = {}

        for doc, score, method in adjusted_items:
            # Absolute cutoff for low-quality noise
            if score < 0.28:
                continue

            # Deduplication: check text fingerprint
            fp = self._text_fingerprint(doc.page_content)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            d_id = doc.metadata.get("document_id", "unknown")
            doc_counts[d_id] = doc_counts.get(d_id, 0) + 1

            # Limit chunks per document to 2 to ensure diversity across multi-doc queries
            if doc_counts[d_id] > 2:
                continue

            # Assign intuitive relevance grades
            if score >= 0.60:
                grade = "High"
            elif score >= 0.38:
                grade = "Medium"
            else:
                grade = "Low"

            filtered.append((doc, float(round(score, 4)), method, grade))

            if len(filtered) >= top_k:
                break

        return filtered

reranker = ContextReranker()
