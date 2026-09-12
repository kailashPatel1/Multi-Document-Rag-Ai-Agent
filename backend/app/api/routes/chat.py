import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...models.db_models import get_db
from ...models.schemas import ChatRequest, ChatResponse, ChatSessionResponse, ChatMessageSchema
from ...agents.knowledge_agent import knowledge_agent
from ...rag.rag_chain import rag_chain
from ...memory.memory_service import memory_service

router = APIRouter(prefix="/api/chat", tags=["Chat & AI Agent"])
logger = logging.getLogger(__name__)

@router.post("", response_model=ChatResponse)
def handle_chat_message(request: ChatRequest):
    """
    Primary chat endpoint.
    mode='agent': Runs LangGraph AI Knowledge Agent with intent routing and tool execution.
    mode='rag_direct': Directly runs the Core Hybrid RAG Pipeline.
    """
    try:
        if request.mode == "rag_direct":
            resp = rag_chain.answer_query(
                query=request.message,
                document_ids=request.document_ids,
                top_k=request.top_k or 5,
                hybrid_alpha=request.hybrid_alpha,
                session_id=request.session_id
            )
            # Record in memory
            sid = memory_service.get_or_create_session(request.session_id)
            memory_service.add_message(session_id=sid, role="user", content=request.message)
            memory_service.add_message(session_id=sid, role="assistant", content=resp.answer, citations=resp.citations)
            resp.session_id = sid
            return resp

        # Default: Agent mode
        resp = knowledge_agent.run(
            query=request.message,
            session_id=request.session_id,
            document_ids=request.document_ids
        )
        return resp

    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/sessions", response_model=List[ChatSessionResponse])
def list_sessions():
    """Lists all user chat sessions."""
    sessions = memory_service.list_sessions()
    return [ChatSessionResponse(**s) for s in sessions]

@router.get("/history/{session_id}")
def get_session_history(session_id: str):
    """Retrieves conversation history for a session."""
    return memory_service.get_session_history(session_id)

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a chat session."""
    success = memory_service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "success", "message": f"Session '{session_id}' deleted."}
