import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.db_models import SessionLocal, ChatSessionModel, ChatMessageModel
from ..models.schemas import ChatMessageSchema, Citation, ToolExecutionTrace

logger = logging.getLogger(__name__)

class MemoryService:
    """Manages persistent conversational history, session tracking, and contextual retrieval."""

    def get_or_create_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Retrieves an existing session ID or initializes a new one in SQLite."""
        db: Session = SessionLocal()
        try:
            if session_id:
                session_obj = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
                if session_obj:
                    return session_obj.id

            new_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
            new_session = ChatSessionModel(
                id=new_id,
                title=title or f"Conversation {datetime.utcnow().strftime('%b %d, %H:%M')}"
            )
            db.add(new_session)
            db.commit()
            return new_id
        finally:
            db.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Citation]] = None,
        tool_traces: Optional[List[ToolExecutionTrace]] = None
    ) -> str:
        """Stores a chat message with structured citations and tool traces."""
        db: Session = SessionLocal()
        try:
            self.get_or_create_session(session_id)
            
            msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            citations_data = [c.dict() if hasattr(c, "dict") else c for c in (citations or [])]
            tool_data = [t.dict() if hasattr(t, "dict") else t for t in (tool_traces or [])]

            msg = ChatMessageModel(
                id=msg_id,
                session_id=session_id,
                role=role,
                content=content,
                citations_json=json.dumps(citations_data),
                tool_traces_json=json.dumps(tool_data)
            )
            db.add(msg)
            
            # Update session timestamp
            sess = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
            if sess:
                sess.updated_at = datetime.utcnow()
                # Update title on first user message if default
                if role == "user" and (sess.title.startswith("Conversation") or sess.title == "New Conversation"):
                    sess.title = (content[:35] + "...") if len(content) > 35 else content

            db.commit()
            return msg_id
        finally:
            db.close()

    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches ordered conversation history for a given session."""
        db: Session = SessionLocal()
        try:
            messages = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            ).order_by(ChatMessageModel.created_at.asc()).limit(limit).all()

            history = []
            for m in messages:
                try:
                    cites = json.loads(m.citations_json or "[]")
                except Exception:
                    cites = []
                try:
                    tools = json.loads(m.tool_traces_json or "[]")
                except Exception:
                    tools = []

                history.append({
                    "id": m.id,
                    "session_id": m.session_id,
                    "role": m.role,
                    "content": m.content,
                    "citations": cites,
                    "tool_traces": tools,
                    "created_at": m.created_at.isoformat()
                })
            return history
        finally:
            db.close()

    def get_recent_context_for_prompt(self, session_id: str, max_turns: int = 4) -> str:
        """
        Retrieves the last few turns formatted compactly to supply context to the LLM
        without bloating token counts.
        """
        db: Session = SessionLocal()
        try:
            messages = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            ).order_by(ChatMessageModel.created_at.desc()).limit(max_turns * 2).all()

            messages.reverse()
            if not messages:
                return ""

            context_lines = ["PREVIOUS CONVERSATION CONTEXT:"]
            for m in messages:
                role_label = "User" if m.role == "user" else "Assistant"
                context_lines.append(f"{role_label}: {m.content}")

            return "\n".join(context_lines)
        finally:
            db.close()

    def search_conversation_memory(self, session_id: str, query: str) -> str:
        """Finds messages in prior dialogue matching key terms."""
        history = self.get_session_history(session_id, limit=30)
        if not history:
            return "No previous conversation history found in this session."

        query_terms = set(query.lower().split())
        matched = []
        for h in history:
            content_lower = h["content"].lower()
            if any(term in content_lower for term in query_terms if len(term) > 3):
                matched.append(f"{h['role'].capitalize()}: {h['content']}")

        if matched:
            return "Relevant prior conversation turns:\n" + "\n---\n".join(matched[-4:])
        return "No specific past conversation turns matched your query, but here is recent context:\n" + self.get_recent_context_for_prompt(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all chat sessions with summary counts."""
        db: Session = SessionLocal()
        try:
            sessions = db.query(ChatSessionModel).order_by(ChatSessionModel.updated_at.desc()).all()
            results = []
            for s in sessions:
                count = db.query(ChatMessageModel).filter(ChatMessageModel.session_id == s.id).count()
                results.append({
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "message_count": count
                })
            return results
        finally:
            db.close()

    def delete_session(self, session_id: str) -> bool:
        """Deletes a chat session and its messages."""
        db: Session = SessionLocal()
        try:
            sess = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
            if sess:
                db.delete(sess)
                db.commit()
                return True
            return False
        finally:
            db.close()

memory_service = MemoryService()
