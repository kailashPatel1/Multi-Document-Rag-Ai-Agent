import pytest
import time
from pathlib import Path
from langchain_core.documents import Document
from backend.app.models.db_models import init_db
from backend.app.retrievers.vector_store import vector_store_manager

@pytest.fixture(scope="module", autouse=True)
def init_system():
    init_db()

def test_chroma_add_and_query():
    doc_id = f"test_lifecycle_{int(time.time() * 1000)}"
    chunks = [
        Document(
            page_content="Quantum Annealing is a metaheuristic for finding the global minimum of an objective function.",
            metadata={"document_id": doc_id, "document_name": "quantum.txt", "page_number": 1, "chunk_id": f"{doc_id}_c1"}
        ),
        Document(
            page_content="Classical simulated annealing operates using thermal fluctuations.",
            metadata={"document_id": doc_id, "document_name": "quantum.txt", "page_number": 2, "chunk_id": f"{doc_id}_c2"}
        )
    ]
    
    # 1. Add documents
    added_ids = vector_store_manager.add_documents(chunks)
    assert len(added_ids) == 2
    
    # 2. Similarity search scoped to document
    results = vector_store_manager.similarity_search_with_score("Quantum Annealing objective function", k=5, document_ids=[doc_id])
    assert len(results) >= 1
    top_doc, score = results[0]
    assert top_doc.metadata["document_id"] == doc_id
    assert "Quantum Annealing" in top_doc.page_content
    
    # 3. Isolation check: Search with a different doc_id filter should return 0 results
    isolated_results = vector_store_manager.similarity_search_with_score("Quantum Annealing", k=5, document_ids=["non_existent_doc_id_999"])
    assert len(isolated_results) == 0
    
    # 4. Delete document chunks from Chroma
    vector_store_manager.delete_document_chunks(doc_id)
    post_delete_results = vector_store_manager.similarity_search_with_score("Quantum Annealing", k=5, document_ids=[doc_id])
    assert len(post_delete_results) == 0
