import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "for", "to",
    "his", "her", "their", "give", "me", "tell", "about", "from", "and", "or", "by",
    "with", "that", "this", "it", "at", "as", "be", "do", "does", "did", "have", "has"
}

class BM25RetrieverManager:
    """Manages BM25 lexical keyword retrieval across all document chunks with auto-sync."""

    _instance: Optional["BM25RetrieverManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BM25RetrieverManager, cls).__new__(cls)
            cls._instance.chunks: List[Document] = []
            cls._instance.bm25: Optional[BM25Okapi] = None
            cls._instance.tokenized_corpus: List[List[str]] = []
        return cls._instance

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenizes text into lowercase non-stopword terms, preserving numbers and hyphens."""
        words = re.findall(r"\b[a-zA-Z0-9_\+\-]{2,}\b", text.lower())
        filtered = [w for w in words if w not in STOP_WORDS]
        return filtered if filtered else words

    def _ensure_synced(self):
        """Auto-syncs chunks from SQLite database if in-memory index is empty."""
        if not self.chunks:
            try:
                from ..models.db_models import SessionLocal, DocumentChunkModel
                import json
                db = SessionLocal()
                db_chunks = db.query(DocumentChunkModel).all()
                if db_chunks:
                    loaded_docs = []
                    for c in db_chunks:
                        meta = json.loads(c.metadata_json) if c.metadata_json else {}
                        meta["chunk_id"] = c.id
                        meta["document_id"] = c.document_id
                        meta["document_name"] = c.document_name
                        meta["page_number"] = c.page_number
                        loaded_docs.append(Document(page_content=c.content, metadata=meta))
                    self.rebuild_index(loaded_docs)
                    logger.info(f"Auto-synced {len(loaded_docs)} chunks into BM25 index.")
                db.close()
            except Exception as e:
                logger.warning(f"Failed to auto-sync BM25 from database: {e}")

    def rebuild_index(self, all_chunks: List[Document]):
        """Rebuilds BM25 index."""
        self.chunks = all_chunks
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_chunks]
        
        if self.tokenized_corpus and len(self.tokenized_corpus) > 0:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Rebuilt BM25 index with {len(all_chunks)} chunks.")
        else:
            self.bm25 = None

    def add_chunks(self, new_chunks: List[Document]):
        """Appends new chunks to BM25 index."""
        current = list(self.chunks)
        existing_ids = {c.metadata.get("chunk_id") for c in current}
        for chunk in new_chunks:
            if chunk.metadata.get("chunk_id") not in existing_ids:
                current.append(chunk)
        self.rebuild_index(current)

    def remove_document(self, document_id: str):
        """Removes chunks belonging to a document."""
        updated = [c for c in self.chunks if c.metadata.get("document_id") != document_id]
        self.rebuild_index(updated)

    def search(
        self,
        query: str,
        k: int = 5,
        document_ids: Optional[List[str]] = None,
        exclude_tabular: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Executes BM25 keyword search.
        Returns List of (Document, normalized_score).
        """
        self._ensure_synced()

        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0

        if max_score <= 0.0:
            return []

        results: List[Tuple[Document, float]] = []
        doc_id_set = set(document_ids) if document_ids else None

        for idx, score in enumerate(raw_scores):
            if score <= 0.0:
                continue

            chunk = self.chunks[idx]
            
            if doc_id_set and chunk.metadata.get("document_id") not in doc_id_set:
                continue

            if exclude_tabular and chunk.metadata.get("file_type") == ".csv":
                continue

            normalized_score = min(1.0, score / max_score)
            results.append((chunk, float(round(normalized_score, 4))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

bm25_retriever_manager = BM25RetrieverManager()
