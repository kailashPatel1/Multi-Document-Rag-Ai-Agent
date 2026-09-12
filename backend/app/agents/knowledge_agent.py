import time
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, TypedDict
from ..config.settings import settings
from ..tools.rag_tools import document_summary_tool, document_comparison_tool
from ..tools.csv_tool import csv_analysis_tool
from ..tools.memory_tool import conversation_memory_tool
from ..rag.rag_chain import rag_chain
from ..models.schemas import Citation, ToolExecutionTrace, ChatResponse
from ..memory.memory_service import memory_service

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """Production LangGraph-compatible AI Agent focusing on Multi-Document RAG, Comparison, Summary, and Memory."""

    def _determine_tool(self, query: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
        """Determines which tool to execute based on generic intent analysis."""
        q_lower = query.lower().strip()

        # 1. Conversation Memory Tool
        if any(phrase in q_lower for phrase in ["what did i ask", "what did we discuss", "earlier", "previous question", "what did i say", "my previous message"]):
            return "conversation_memory_tool", {"session_id": session_id, "query": query}

        # 2. Structured Tabular / CSV Analytics Tool
        if any(term in q_lower for term in ["average sales", "highest sales", "total revenue", "highest revenue", "lowest sales", "sales in csv", "sales data", "column average", "row count"]):
            return "csv_analysis_tool", {"query": query}

        # 3. Generic Multi-document Comparison Tool
        if any(w in q_lower for w in ["compare", "difference between", "versus", " vs "]):
            from ..rag.query_normalizer import query_normalizer
            active_docs = query_normalizer._get_active_document_names()
            doc_a = active_docs[0] if len(active_docs) > 0 else "Document 1"
            doc_b = active_docs[1] if len(active_docs) > 1 else "Document 2"

            # Check if specific documents were mentioned in the query
            mentioned = []
            for d in active_docs:
                stem = d.rsplit(".", 1)[0].lower()
                if d.lower() in q_lower or stem in q_lower:
                    mentioned.append(d)
            if len(mentioned) >= 2:
                doc_a, doc_b = mentioned[0], mentioned[1]
            elif len(mentioned) == 1 and len(active_docs) >= 2:
                doc_a = mentioned[0]
                doc_b = [d for d in active_docs if d != doc_a][0]

            return "document_comparison_tool", {
                "document_a_name": doc_a,
                "document_b_name": doc_b,
                "comparison_query": query
            }

        # 4. Document Summary Tool
        if any(phrase in q_lower for phrase in ["summarize all", "summary of all", "overview of all uploaded", "summarize library"]):
            return "document_summary_tool", {"document_name_or_id": "all", "focus": "general"}

        # 5. Default: Core RAG Search Tool (works for ANY uploaded document topic)
        return "rag_search_tool", {"query": query}

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> ChatResponse:
        """Runs the Multi-Document Knowledge Agent workflow."""
        start_time = time.time()
        sid = memory_service.get_or_create_session(session_id)
        recent_memory = memory_service.get_recent_context_for_prompt(sid)

        # 1. Routing Step
        tool_name, tool_args = self._determine_tool(query, sid)

        citations: List[Citation] = []
        tool_output: Dict[str, Any] = {}
        final_answer = ""
        status = "success"

        try:
            # 2. Execute selected tool
            if tool_name == "rag_search_tool":
                rag_resp = rag_chain.answer_query(
                    query=query,
                    document_ids=document_ids,
                    session_id=sid
                )
                final_answer = rag_resp.answer
                citations = rag_resp.citations
                tool_output = {
                    "query": query,
                    "retrieved_sources": len(citations),
                    "grounded": rag_resp.grounded
                }

            elif tool_name == "csv_analysis_tool":
                tool_output = csv_analysis_tool(query=query)
                final_answer = tool_output.get("summary", "Analysis completed.")

            elif tool_name == "document_comparison_tool":
                tool_output = document_comparison_tool(
                    document_a_name=tool_args["document_a_name"],
                    document_b_name=tool_args["document_b_name"],
                    comparison_query=tool_args["comparison_query"]
                )
                citations = [Citation(**c) for c in tool_output.get("citations", [])[:3]]
                doc_a = tool_output.get("doc_a_name", "Doc A")
                doc_b = tool_output.get("doc_b_name", "Doc B")
                chunks_a = tool_output.get("doc_a_chunks", [])
                chunks_b = tool_output.get("doc_b_chunks", [])

                final_answer = (
                    f"### 🔍 Multi-Document Comparison: {doc_a} vs {doc_b}\n\n"
                    f"**Key Findings from {doc_a}:**\n"
                    + ("\n".join([f"- {c[:180]}..." for c in chunks_a[:3]]) if chunks_a else "- No matching sections.")
                    + f"\n\n**Key Findings from {doc_b}:**\n"
                    + ("\n".join([f"- {c[:180]}..." for c in chunks_b[:3]]) if chunks_b else "- No matching sections.")
                )

            elif tool_name == "document_summary_tool":
                tool_output = document_summary_tool(document_name_or_id=tool_args["document_name_or_id"])
                final_answer = f"### 📑 Summary of {tool_output.get('document_name', 'Documents')}\n\n{tool_output.get('document_content_sample', '')[:800]}..."

            elif tool_name == "conversation_memory_tool":
                tool_output = conversation_memory_tool(session_id=sid, query=query)
                final_answer = tool_output.get("retrieved_memory", "No previous conversation context found.")

        except Exception as e:
            logger.error(f"Agent error in {tool_name}: {e}")
            final_answer = f"Error processing query: {str(e)}"
            status = "error"

        duration_ms = round((time.time() - start_time) * 1000, 2)
        trace = ToolExecutionTrace(
            tool_name=tool_name,
            input_params=tool_args,
            output=tool_output,
            execution_time_ms=duration_ms,
            status=status
        )

        # 3. Persist to Memory
        memory_service.add_message(session_id=sid, role="user", content=query)
        memory_service.add_message(
            session_id=sid,
            role="assistant",
            content=final_answer,
            citations=citations,
            tool_traces=[trace]
        )

        return ChatResponse(
            answer=final_answer,
            session_id=sid,
            citations=citations,
            tool_traces=[trace],
            grounded=True if citations else ("insufficient information" in final_answer.lower()),
            grounding_confidence=1.0 if citations else 0.8
        )

knowledge_agent = KnowledgeAgent()


