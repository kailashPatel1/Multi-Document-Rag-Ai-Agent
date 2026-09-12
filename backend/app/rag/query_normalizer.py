import re
from typing import Tuple, Optional, List, Dict, Any

class QueryNormalizer:
    """
    Generic, document-agnostic query preprocessor and multi-question decomposer.
    - Cleans query text
    - Dynamically detects target document names
    - Decomposes multi-intent compound queries into independent sub-questions
    """

    @staticmethod
    def _get_active_document_names() -> List[str]:
        """Dynamically retrieves the list of currently indexed document filenames from database."""
        try:
            from ..models.db_models import SessionLocal, DocumentModel
            db = SessionLocal()
            doc_records = db.query(DocumentModel.original_name, DocumentModel.filename).all()
            db.close()
            names = set()
            for orig, fname in doc_records:
                if orig:
                    names.add(orig)
                if fname:
                    names.add(fname)
            return list(names)
        except Exception:
            return []

    @classmethod
    def normalize(cls, query: str, available_documents: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
        """
        Cleans query text and extracts target document hint dynamically.
        Returns (cleaned_query, target_document_name_hint).
        """
        if not query:
            return "", None

        raw = query.strip()
        cleaned_query = re.sub(r"\s+", " ", raw)
        q_lower = cleaned_query.lower()

        docs_to_check = available_documents if available_documents is not None else cls._get_active_document_names()
        target_doc = None

        if docs_to_check:
            sorted_docs = sorted(docs_to_check, key=lambda x: len(x), reverse=True)
            for doc_name in sorted_docs:
                clean_name = doc_name.lower().strip()
                stem = clean_name.rsplit(".", 1)[0]
                
                if clean_name in q_lower:
                    target_doc = doc_name
                    break
                elif len(stem) >= 4 and (f" {stem} " in f" {q_lower} " or f"'{stem}'" in q_lower or f'"{stem}"' in q_lower):
                    target_doc = doc_name
                    break

        return cleaned_query, target_doc

    @staticmethod
    def decompose_sub_questions(query: str) -> List[str]:
        """
        Decomposes a compound query into distinct sub-questions / search intents.
        Examples:
        - "What are his projects? What is his location? What is his education?" ->
          ["What are his projects?", "What is his location?", "What is his education?"]
        - "What are his skills and what is his internship?" ->
          ["What are his skills", "what is his internship"]
        """
        if not query or not query.strip():
            return []

        raw = query.strip()

        # 1. Check for question mark splits
        if "?" in raw:
            parts = [p.strip() + "?" for p in raw.split("?") if len(p.strip()) > 3]
            if len(parts) > 1:
                return parts

        # 2. Check for numbered lists (e.g. "1. ... 2. ...")
        numbered = re.split(r"(?:\r?\n|\s+)(?:\d+[\.\)]|[a-zA-Z][\.\)])\s+", raw)
        if len(numbered) > 1:
            clean_numbered = [p.strip() for p in numbered if len(p.strip()) > 3]
            if len(clean_numbered) > 1:
                return clean_numbered

        # 3. Check for newline or semicolon separated questions
        delims = re.split(r"[\n;]+", raw)
        if len(delims) > 1:
            clean_delims = [p.strip() for p in delims if len(p.strip()) > 3]
            if len(clean_delims) > 1:
                return clean_delims

        # 4. Check for compound conjunctions (e.g. "What are his skills, and what is his internship, and what is his address?")
        conjunction_split = re.split(r"\s+(?:and\s+also|and\s+what|and\s+where|and\s+how|and\s+tell\s+me|and\s+give\s+me)\s+", raw, flags=re.IGNORECASE)
        if len(conjunction_split) > 1:
            clean_conj = [p.strip() for p in conjunction_split if len(p.strip()) > 3]
            if len(clean_conj) > 1:
                return clean_conj

        return [raw]

query_normalizer = QueryNormalizer()
