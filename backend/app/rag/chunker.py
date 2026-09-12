import uuid
import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..config.settings import settings

class DocumentChunker:
    """Production chunker that splits multi-page documents while maintaining metadata and section integrity."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "### ", "## ", "# ", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True
        )

    def _extract_section_header(self, text: str) -> Optional[str]:
        """Tries to find markdown headers or title lines at start of chunk."""
        match = re.search(r"^(?:#{1,4}\s+)?([A-Z0-9][^\n]{3,60})(?:\n|$)", text.strip())
        if match:
            return match.group(1).strip()
        return None

    def chunk_documents(
        self,
        docs: List[Document],
        document_id: str,
        document_name: str
    ) -> List[Document]:
        """
        Splits a list of page documents into fine-grained chunks enriched with unique chunk_ids,
        page numbers, section headers, and document relations.
        """
        all_chunks: List[Document] = []
        global_chunk_idx = 0

        for doc in docs:
            page_num = doc.metadata.get("page_number", 1)
            file_type = doc.metadata.get("file_type", "unknown")
            page_text = doc.page_content.strip()

            if not page_text:
                continue

            split_texts = self.splitter.split_text(page_text)

            for local_idx, chunk_text in enumerate(split_texts):
                if not chunk_text.strip():
                    continue

                chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                section_title = self._extract_section_header(chunk_text)

                chunk_meta: Dict[str, Any] = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": int(page_num),
                    "chunk_index": int(global_chunk_idx),
                    "file_type": file_type,
                    "char_count": len(chunk_text),
                    "section_title": section_title or ""
                }

                # Preserve any original document-level metadata
                for k, v in doc.metadata.items():
                    if k not in chunk_meta and isinstance(v, (str, int, float, bool)):
                        chunk_meta[k] = v

                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_meta
                )
                all_chunks.append(chunk_doc)
                global_chunk_idx += 1

        return all_chunks
