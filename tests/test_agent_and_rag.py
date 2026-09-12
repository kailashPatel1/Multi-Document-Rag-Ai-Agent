import pytest
from backend.app.models.db_models import init_db
from backend.app.agents.knowledge_agent import knowledge_agent
from backend.app.rag.citation import CitationEngine, citation_engine
from backend.app.models.schemas import Citation
from langchain_core.documents import Document

@pytest.fixture(scope="module", autouse=True)
def init_system():
    init_db()

def test_agent_tool_routing():
    # Test Memory routing
    tname, args = knowledge_agent._determine_tool("What did we discuss earlier about RAG?", "sess_1")
    assert tname == "conversation_memory_tool"

    # Test CSV analytics routing
    tname, args = knowledge_agent._determine_tool("What is the average sales in the dataset?", "sess_1")
    assert tname == "csv_analysis_tool"

    # Test Comparison routing
    tname, args = knowledge_agent._determine_tool("Compare my resume with the job description", "sess_1")
    assert tname == "document_comparison_tool"

    # Test Summary routing
    tname, args = knowledge_agent._determine_tool("Summarize all uploaded documents", "sess_1")
    assert tname == "document_summary_tool"

    # Test default RAG search routing
    tname, args = knowledge_agent._determine_tool("What are the key responsibilities in the job description?", "sess_1")
    assert tname == "rag_search_tool"

def test_citation_grounding_eval():
    cites = [
        Citation(
            document_name="RAG_Notes.pdf",
            document_id="doc_1",
            page_number=1,
            chunk_id="chunk_1",
            snippet="Retrieval-Augmented Generation combines information retrieval with large language models to ground answers.",
            similarity_score=0.92,
            relevance_grade="High"
        )
    ]
    
    answer = "Retrieval-Augmented Generation grounds large language models with retrieved information."
    is_grounded, conf = citation_engine.evaluate_grounding(answer, cites)
    assert is_grounded is True
    assert conf > 0.5

def test_citation_deduplication():
    chunks_same_page = [
        (Document(page_content="Header info with contact details", metadata={"document_id": "doc_1", "document_name": "Kailash_Resume..pdf", "page_number": 1, "chunk_id": "c1"}), 0.95, "hybrid", "High"),
        (Document(page_content="Education section B.Tech", metadata={"document_id": "doc_1", "document_name": "Kailash_Resume..pdf", "page_number": 1, "chunk_id": "c2"}), 0.88, "hybrid", "High"),
        (Document(page_content="Page 2 references", metadata={"document_id": "doc_1", "document_name": "Kailash_Resume..pdf", "page_number": 2, "chunk_id": "c3"}), 0.62, "hybrid", "Medium"),
    ]
    citations = CitationEngine.build_citations(chunks_same_page)
    assert len(citations) == 2
    assert citations[0].page_number == 1
    assert citations[1].page_number == 2
