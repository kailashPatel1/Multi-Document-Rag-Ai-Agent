from typing import Dict, Any
from ..memory.memory_service import memory_service

def conversation_memory_tool(session_id: str, query: str) -> Dict[str, Any]:
    """
    Conversation Memory Tool: Retrieves prior dialogue history, past questions, and previous answers
    from the current or referenced chat session to maintain conversational continuity.
    """
    context = memory_service.search_conversation_memory(session_id=session_id, query=query)
    return {
        "session_id": session_id,
        "query": query,
        "retrieved_memory": context
    }
