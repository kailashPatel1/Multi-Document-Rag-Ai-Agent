import os
import pytest
from pathlib import Path
from backend.app.rag.document_loader import DocumentLoader, DocumentLoaderError
from backend.app.rag.chunker import DocumentChunker
from backend.app.services.sample_generator import generate_sample_documents

@pytest.fixture(scope="module")
def sample_files():
    return generate_sample_documents()

def test_load_pdf(sample_files):
    pdf_path = sample_files["rag_notes"]
    docs, meta = DocumentLoader.load_file(pdf_path)
    assert len(docs) > 0
    assert meta["total_pages"] >= 2
    assert "Retrieval-Augmented Generation" in docs[0].page_content
    assert docs[0].metadata["page_number"] == 1

def test_load_csv(sample_files):
    csv_path = sample_files["sample_sales"]
    docs, meta = DocumentLoader.load_file(csv_path)
    assert len(docs) == 1
    assert "Revenue" in meta["columns"]
    assert "Enterprise AI Platform" in docs[0].page_content

def test_chunker(sample_files):
    pdf_path = sample_files["rag_notes"]
    docs, meta = DocumentLoader.load_file(pdf_path)
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_documents(docs, document_id="doc_test_1", document_name="RAG_Notes.pdf")
    
    assert len(chunks) > len(docs)
    for c in chunks:
        assert "chunk_id" in c.metadata
        assert "page_number" in c.metadata
        assert c.metadata["document_name"] == "RAG_Notes.pdf"
        assert len(c.page_content) <= 400
