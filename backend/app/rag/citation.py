import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from ..models.schemas import Citation

logger = logging.getLogger(__name__)

class CitationEngine:
    """Extracts, deduplicates, and formats page-level source citations and verifies response grounding."""

    @staticmethod
    def build_citations(
        ranked_chunks: List[Tuple[Document, float, str, str]]
    ) -> List[Citation]:
        """
        Converts retrieved ranked chunks into clean, deduplicated Citation schemas.
        Deduplicates by (document_id, page_number) so that multiple chunks from the same
        document and page are merged into ONE unified entry with the highest relevance grade.
        Preserves different pages and different documents.
        """
        if not ranked_chunks:
            return []

        # Map (doc_id, page_number) -> merged citation data
        page_entries: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for doc, score, method, grade in ranked_chunks:
            meta = doc.metadata
            doc_id = meta.get("document_id", "doc_unknown")
            doc_name = meta.get("document_name", "Unknown Document")
            page_num = int(meta.get("page_number", 1))
            chunk_id = meta.get("chunk_id", "")
            content = doc.page_content.strip()

            clean_snippet = re.sub(r"\s+", " ", content)

            key = (doc_id, page_num)
            if key not in page_entries:
                page_entries[key] = {
                    "document_name": doc_name,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "snippets": [clean_snippet],
                    "similarity_score": float(round(score, 4)),
                    "relevance_grade": grade
                }
            else:
                existing = page_entries[key]
                # Keep highest similarity score and grade
                if score > existing["similarity_score"]:
                    existing["similarity_score"] = float(round(score, 4))
                    existing["relevance_grade"] = grade
                # Append unique text snippets from the same page
                if clean_snippet not in existing["snippets"]:
                    existing["snippets"].append(clean_snippet)

        citations: List[Citation] = []
        for key, entry in page_entries.items():
            combined_snippet = "\n\n---\n\n".join(entry["snippets"])
            if len(combined_snippet) > 400:
                combined_snippet = combined_snippet[:400] + "..."

            citations.append(Citation(
                document_name=entry["document_name"],
                document_id=entry["document_id"],
                page_number=entry["page_number"],
                chunk_id=entry["chunk_id"],
                snippet=combined_snippet,
                similarity_score=entry["similarity_score"],
                relevance_grade=entry["relevance_grade"]
            ))

        # Sort by similarity score descending
        citations.sort(key=lambda c: c.similarity_score, reverse=True)
        return citations

    @staticmethod
    def evaluate_grounding(
        answer: str,
        citations: List[Citation]
    ) -> Tuple[bool, float]:
        """
        Evaluates whether the answer is grounded in retrieved documents.
        Returns (is_grounded, grounding_confidence_score [0.0 - 1.0]).
        """
        if not citations:
            if "insufficient information" in answer.lower() or "couldn't find" in answer.lower():
                return True, 1.0
            return False, 0.0

        if "i couldn't find sufficient information in the uploaded documents" in answer.lower():
            return True, 1.0

        answer_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower()))
        context_tokens = set()
        for c in citations:
            context_tokens.update(re.findall(r"\b[a-zA-Z]{4,}\b", c.snippet.lower()))

        if not answer_tokens:
            return True, 0.9

        overlap = len(answer_tokens.intersection(context_tokens))
        ratio = overlap / len(answer_tokens)
        
        confidence = min(1.0, max(0.2, ratio * 1.5))
        is_grounded = confidence >= 0.35
        return is_grounded, round(confidence, 2)

    @staticmethod
    def format_citations_markdown(citations: List[Citation]) -> str:
        """Formats citations as rich markdown text with document-grouped pages."""
        if not citations:
            return ""

        # Group pages by document_name
        doc_pages: Dict[str, List[Citation]] = {}
        for c in citations:
            if c.document_name not in doc_pages:
                doc_pages[c.document_name] = []
            doc_pages[c.document_name].append(c)

        lines = ["\n\n### 📄 Referenced Sources:"]
        for dname, pages in doc_pages.items():
            if len(pages) == 1:
                p = pages[0]
                lines.append(f"- **{dname}** — Page {p.page_number} ({p.relevance_grade})")
            else:
                lines.append(f"- **{dname}**:")
                for p in pages:
                    lines.append(f"  - Page {p.page_number} — {p.relevance_grade} ({int(p.similarity_score * 100)}%)")

        return "\n".join(lines)

citation_engine = CitationEngine()
