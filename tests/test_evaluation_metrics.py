import pytest
from backend.app.models.db_models import init_db, SessionLocal, DocumentModel
from backend.app.evaluation.evaluation_service import (
    evaluation_service,
    normalize_doc_identifier,
    match_document_identity,
    DEFAULT_BENCHMARK_ITEMS
)
from backend.app.models.schemas import EvaluationItem

@pytest.fixture(scope="module", autouse=True)
def init_system():
    init_db()

def test_normalize_doc_identifier():
    assert normalize_doc_identifier("data/uploads/doc_ab4bdfa3bac0_RAG_Notes.pdf") == "rag_notes"
    assert normalize_doc_identifier("Kailash_Resume..pdf") == "kailash_resume"
    assert normalize_doc_identifier("doc_354711f62dd5_Resume.pdf") == "resume"
    assert normalize_doc_identifier("Sample_Sales.csv") == "sample_sales"
    assert normalize_doc_identifier("") == ""

def test_match_document_identity():
    doc_lookup = {
        "doc_rag_1": {"id": "doc_rag_1", "filename": "doc_rag_1_RAG_Notes.pdf", "original_name": "RAG_Notes.pdf"},
        "doc_ml_2": {"id": "doc_ml_2", "filename": "doc_ml_2_ML_Notes.pdf", "original_name": "ML_Notes.pdf"},
    }

    # Match by canonical name
    assert match_document_identity("RAG_Notes.pdf", "RAG_Notes.pdf", "doc_rag_1", doc_lookup) is True
    assert match_document_identity("RAG_Notes.pdf", "doc_rag_1_RAG_Notes.pdf", "doc_rag_1", doc_lookup) is True
    assert match_document_identity("doc_rag_1", "RAG_Notes.pdf", "doc_rag_1", doc_lookup) is True

    # No false-positive matching across different documents
    assert match_document_identity("ML_Notes.pdf", "RAG_Notes.pdf", "doc_rag_1", doc_lookup) is False
    assert match_document_identity("Resume.pdf", "Job_Description.pdf", "doc_jd_3", doc_lookup) is False

def test_evaluation_run():
    # Test running evaluation on 2 benchmark items
    items = [
        EvaluationItem(
            id="test_1",
            question="What is RAG and why is it used?",
            expected_answer="Retrieval-Augmented Generation combines information retrieval with large language models.",
            relevant_document="RAG_Notes.pdf",
            relevant_page=1
        ),
        EvaluationItem(
            id="test_2",
            question="What are transformer attention mechanisms?",
            expected_answer="Self-attention computes dynamic weights between token pairs.",
            relevant_document="ML_Notes.pdf",
            relevant_page=1
        )
    ]
    res = evaluation_service.run_evaluation(items=items, top_k=3)
    assert res.metrics.total_queries == 2
    assert res.metrics.retrieval_precision > 0.0
    assert res.metrics.retrieval_recall > 0.0
    assert len(res.item_results) == 2
    
    # Check trace structure
    it0 = res.item_results[0]
    assert it0["query"] == "What is RAG and why is it used?"
    assert it0["expected_document"] == "RAG_Notes.pdf"
    assert isinstance(it0["retrieved_documents"], list)
    assert "RAG_Notes.pdf" in it0["retrieved_documents"]
    assert it0["retrieval_precision"] > 0.0
    assert it0["retrieval_recall"] == 1.0
