from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth Schemas ---
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

# --- Document Schemas ---
class DocumentChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    content: str
    char_count: int
    metadata: Dict[str, Any] = {}

class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    total_pages: int
    total_chunks: int
    created_at: datetime
    status: str = "ready"
    summary: Optional[str] = None
    column_names: Optional[List[str]] = None

class DocumentSummaryRequest(BaseModel):
    document_id: str
    focus: Optional[str] = "general"

class DocumentCompareRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=2, max_length=5)
    query: Optional[str] = "Compare key concepts, differences, and complementary aspects across these documents."

# --- Citation & Retrieval Schemas ---
class Citation(BaseModel):
    document_name: str
    document_id: str
    page_number: int
    chunk_id: str
    snippet: str
    similarity_score: float
    relevance_grade: str = "High"  # High, Medium, Low

class SearchResultChunk(BaseModel):
    chunk_id: str
    document_name: str
    document_id: str
    page_number: int
    content: str
    score: float
    retrieval_method: str = "hybrid"  # vector, bm25, hybrid
    metadata: Dict[str, Any] = {}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_ids: Optional[List[str]] = None
    hybrid_alpha: Optional[float] = 0.5
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultChunk]

# --- Agent & Chat Schemas ---
class ToolExecutionTrace(BaseModel):
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    execution_time_ms: float
    status: str = "success"

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "agent"  # "agent" (LangGraph tool-calling) or "rag_direct"
    document_ids: Optional[List[str]] = None
    hybrid_alpha: Optional[float] = 0.5
    top_k: Optional[int] = 5

class ChatMessageSchema(BaseModel):
    id: str
    session_id: str
    role: str  # user, assistant, system, tool
    content: str
    citations: Optional[List[Citation]] = []
    tool_traces: Optional[List[ToolExecutionTrace]] = []
    created_at: datetime

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: List[Citation] = []
    tool_traces: List[ToolExecutionTrace] = []
    grounded: bool = True
    grounding_confidence: float = 1.0

# --- CSV Analytics Schemas ---
class CSVQueryRequest(BaseModel):
    document_id: str
    query: str

class CSVQueryResponse(BaseModel):
    document_id: str
    query: str
    result_type: str  # table, number, text, plot_data
    result: Any
    explanation: str

# --- Evaluation Schemas ---
class EvaluationItem(BaseModel):
    id: Optional[str] = None
    question: str
    expected_answer: str
    relevant_document: str
    relevant_page: int

class EvaluationRunRequest(BaseModel):
    dataset_name: Optional[str] = "default"
    custom_items: Optional[List[EvaluationItem]] = None
    top_k: int = 5

class EvaluationMetricSummary(BaseModel):
    total_queries: int
    retrieval_precision: float
    retrieval_recall: float
    context_relevance: float
    answer_faithfulness: float
    answer_correctness: float
    citation_accuracy: float
    average_latency_ms: float

class EvaluationResultResponse(BaseModel):
    run_id: str
    timestamp: datetime
    metrics: EvaluationMetricSummary
    item_results: List[Dict[str, Any]]

# --- System & Status Schemas ---
class SystemStatusResponse(BaseModel):
    status: str
    version: str
    total_documents: int
    total_chunks: int
    vector_db_status: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    uptime_seconds: float
