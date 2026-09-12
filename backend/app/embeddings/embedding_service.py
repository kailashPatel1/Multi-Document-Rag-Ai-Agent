import os
import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from ..config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Manages text embedding generation using SentenceTransformers or Cloud providers."""
    
    _instance: Optional["EmbeddingService"] = None
    _embedder: Optional[Embeddings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info(f"Initializing embedding provider: {settings.EMBEDDING_PROVIDER} ({settings.EMBEDDING_MODEL_NAME})")
        
        # Default to HuggingFace SentenceTransformers
        if settings.EMBEDDING_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._embedder = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=settings.GEMINI_API_KEY
                )
                logger.info("Successfully loaded Google Generative AI Embeddings.")
                return
            except Exception as e:
                logger.warning(f"Failed to load Google Embeddings ({e}), falling back to HuggingFace SentenceTransformers.")

        # HuggingFace fallback / primary
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embedder = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Successfully loaded HuggingFace SentenceTransformers embeddings.")
        except Exception as e:
            logger.error(f"Error loading HuggingFace embeddings: {e}")
            # Fallback lightweight fake/deterministic embedder for offline safety if needed
            self._embedder = self._build_deterministic_embedder()

    def _build_deterministic_embedder(self) -> Embeddings:
        """Lightweight deterministic embedder as ultimate fallback if torch/transformers fails."""
        import hashlib
        import numpy as np

        class FallbackDeterministicEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self._embed_text(t) for t in texts]

            def embed_query(self, text: str) -> List[float]:
                return self._embed_text(text)

            def _embed_text(self, text: str) -> List[float]:
                # 384-dimensional normalized deterministic vector from hash
                seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                norm = np.linalg.norm(vec)
                return (vec / (norm if norm > 0 else 1.0)).tolist()

        logger.warning("Using FallbackDeterministicEmbeddings.")
        return FallbackDeterministicEmbeddings()

    def get_embeddings(self) -> Embeddings:
        if self._embedder is None:
            self._initialize()
        return self._embedder

    def embed_query(self, text: str) -> List[float]:
        return self.get_embeddings().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.get_embeddings().embed_documents(texts)

embedding_service = EmbeddingService()
