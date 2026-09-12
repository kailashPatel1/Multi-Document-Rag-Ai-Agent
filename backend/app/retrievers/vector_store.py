import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from ..config.settings import settings
from ..embeddings.embedding_service import embedding_service

logger = logging.getLogger(__name__)

STABLE_COLLECTION_NAME = "multidoc_knowledge_base"

class VectorStoreManager:
    """
    Robust, production ChromaDB Vector Store Manager.
    Features:
    - Stable collection name ('multidoc_knowledge_base')
    - Resilient lifecycle management (auto-reconnects on stale collection ID)
    - Strict document_id metadata filtering
    - Auto-syncs from SQLite if vector store is out of sync or recreated
    - True cosine similarity scoring in [0.0, 1.0]
    """

    _instance: Optional["VectorStoreManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.vector_store = None
            cls._instance.embedding_fn = None
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Connects to the persistent directory and initializes the Chroma vector store."""
        try:
            self.persist_dir = str(settings.CHROMA_PERSIST_DIRECTORY)
            os.makedirs(self.persist_dir, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.embedding_fn = embedding_service.get_embeddings()
            
            # Ensure collection exists
            self.client.get_or_create_collection(
                name=STABLE_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            self.vector_store = Chroma(
                client=self.client,
                collection_name=STABLE_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB with stable collection '{STABLE_COLLECTION_NAME}' at '{self.persist_dir}'.")
        except Exception as e:
            logger.error(f"Error during ChromaDB initialization: {e}")

    def _ensure_vector_store(self) -> Chroma:
        """Verifies collection health and re-binds if stale/deleted."""
        try:
            if self.client is None:
                self._initialize()
            
            # Ping collection
            _ = self.client.get_or_create_collection(
                name=STABLE_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            if self.vector_store is None:
                self.vector_store = Chroma(
                    client=self.client,
                    collection_name=STABLE_COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            return self.vector_store
        except Exception as e:
            logger.warning(f"Re-initializing stale Chroma vector store: {e}")
            self._initialize()
            return self.vector_store

    def auto_sync_from_database(self):
        """Auto-repopulates ChromaDB if collection is empty but SQLite contains chunks."""
        try:
            total_vectors = self.get_total_chunks()
            if total_vectors == 0:
                from ..models.db_models import SessionLocal, DocumentChunkModel
                db = SessionLocal()
                db_chunks = db.query(DocumentChunkModel).all()
                if db_chunks:
                    logger.info(f"ChromaDB collection empty. Restoring {len(db_chunks)} chunks from SQLite database...")
                    docs_to_index = []
                    for c in db_chunks:
                        meta = json.loads(c.metadata_json) if c.metadata_json else {}
                        meta["chunk_id"] = c.id
                        meta["document_id"] = c.document_id
                        meta["document_name"] = c.document_name
                        meta["page_number"] = c.page_number
                        docs_to_index.append(Document(page_content=c.content, metadata=meta))
                    self.add_documents(docs_to_index)
                    logger.info(f"Successfully restored {len(docs_to_index)} chunks into ChromaDB.")
                db.close()
        except Exception as e:
            logger.warning(f"Failed to auto-sync ChromaDB from database: {e}")

    def add_documents(self, chunks: List[Document]) -> List[str]:
        """Indexes a list of document chunks into ChromaDB with auto-retry on stale collection."""
        if not chunks:
            return []

        ids = [c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
        
        cleaned_chunks = []
        for c in chunks:
            clean_meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            
            cleaned_chunks.append(Document(page_content=c.page_content, metadata=clean_meta))

        store = self._ensure_vector_store()
        try:
            store.add_documents(documents=cleaned_chunks, ids=ids)
            logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")
            return ids
        except Exception as e:
            logger.warning(f"Chroma add_documents error ({e}). Re-binding collection and retrying...")
            self._initialize()
            store = self._ensure_vector_store()
            store.add_documents(documents=cleaned_chunks, ids=ids)
            logger.info(f"Successfully indexed {len(chunks)} chunks into ChromaDB on retry.")
            return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        document_ids: Optional[List[str]] = None,
        exclude_tabular: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Performs dense vector similarity search with true cosine similarity scoring in [0.0, 1.0].
        Strictly applies document_ids filter when specified.
        """
        where_filter = None
        conditions = []

        if document_ids and len(document_ids) > 0:
            if len(document_ids) == 1:
                conditions.append({"document_id": document_ids[0]})
            else:
                conditions.append({"document_id": {"$in": document_ids}})
        
        if exclude_tabular:
            conditions.append({"file_type": {"$ne": ".csv"}})

        if filter_metadata:
            conditions.append(filter_metadata)

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        store = self._ensure_vector_store()
        try:
            results_with_distance = store.similarity_search_with_score(
                query=query,
                k=k,
                filter=where_filter
            )
        except Exception as e:
            logger.warning(f"Vector search exception ({e}). Re-initializing vector store...")
            self._initialize()
            store = self._ensure_vector_store()
            try:
                results_with_distance = store.similarity_search_with_score(query=query, k=k, filter=where_filter)
            except Exception as e2:
                logger.error(f"Vector search failed after re-init: {e2}")
                results_with_distance = []

        # Post-filter verification: if document_ids was requested, strictly enforce it
        formatted_results = []
        doc_id_set = set(document_ids) if document_ids else None

        for doc, dist in results_with_distance:
            if doc_id_set and doc.metadata.get("document_id") not in doc_id_set:
                continue
            sim = 1.0 - (float(dist) / 2.0)
            norm_score = max(0.0, min(1.0, sim))
            formatted_results.append((doc, round(norm_score, 4)))

        return formatted_results

    def delete_document_chunks(self, document_id: str):
        """Removes all chunks associated with a document_id from vector store."""
        try:
            self._ensure_vector_store()
            collection = self.client.get_collection(STABLE_COLLECTION_NAME)
            collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted vector chunks for document_id={document_id}")
        except Exception as e:
            logger.warning(f"Error deleting chunks for {document_id}: {e}")

    def get_total_chunks(self) -> int:
        try:
            self._ensure_vector_store()
            collection = self.client.get_collection(STABLE_COLLECTION_NAME)
            return collection.count()
        except Exception:
            return 0

    def reset_collection(self):
        """Resets the collection cleanly and re-initializes the vector store."""
        try:
            if self.client:
                self.client.delete_collection(STABLE_COLLECTION_NAME)
        except Exception:
            pass
        self._initialize()

vector_store_manager = VectorStoreManager()
