import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from ..config.settings import settings
from ..retrievers.hybrid_retriever import hybrid_retriever
from ..retrievers.reranker import reranker
from ..retrievers.bm25_retriever import bm25_retriever_manager
from .citation import citation_engine
from .query_normalizer import query_normalizer
from ..models.schemas import Citation, ToolExecutionTrace, ChatResponse

logger = logging.getLogger(__name__)

STRICT_RAG_SYSTEM_PROMPT = """You are the AI Knowledge Assistant for the user's private documents.
Answer the user's question(s) accurately, concisely, and naturally based ONLY on the provided document context.

RULES:
1. When multiple questions are asked, address each question clearly and structurally.
2. Answer directly in natural language based on the context.
3. If the information for any specific question is not contained in the provided context, state clearly for that item:
   "I couldn't find sufficient information in the uploaded documents."
4. Do not invent or assume facts.
5. Do not cite external knowledge outside the provided document text."""

GENERIC_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "for", "to",
    "his", "her", "their", "my", "your", "give", "me", "tell", "about", "from",
    "and", "or", "by", "with", "that", "this", "it", "at", "as", "be", "do",
    "does", "did", "have", "has", "can", "could", "would", "should", "please",
    "show", "explain", "describe", "find", "detail", "details", "information"
}

class RAGChain:
    """Production RAG Chain combining Multi-Question Decomposition, Hybrid Retrieval, Context Fusion, and Citations."""

    def _get_llm(self, provider: Optional[str] = None):
        target_provider = provider or settings.DEFAULT_LLM_PROVIDER
        
        # 1. Groq Provider
        if target_provider == "groq":
            groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
            if not groq_key or not groq_key.strip():
                raise ValueError("Groq API key is not configured. Please add GROQ_API_KEY to your .env file or configure it in Settings.")
            
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_key.strip(),
                    model_name=settings.GROQ_MODEL,
                    temperature=0.1,
                    max_tokens=2048
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ChatGroq: {str(e)}")

        # 2. Gemini Provider
        if target_provider == "gemini":
            gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if not gemini_key or not gemini_key.strip():
                raise ValueError("Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file or configure it in Settings.")
            
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    google_api_key=gemini_key.strip(),
                    temperature=0.1,
                    max_output_tokens=2048
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")

        # 3. Generic Extractive Grounding Engine (Offline / Local)
        return self._build_extractive_grounding_llm()

    def _build_extractive_grounding_llm(self):
        """Generic, document-agnostic extractive answer synthesizer for offline local mode."""
        class ExtractiveGroundingLLM:
            def invoke(self, messages):
                content = messages[-1]["content"] if isinstance(messages[-1], dict) else getattr(messages[-1], "content", "")
                
                if "RETRIEVED DOCUMENT CONTEXT:" in content:
                    context_part = content.split("RETRIEVED DOCUMENT CONTEXT:")[1].split("USER QUESTION:")[0].strip()
                    user_q = content.split("USER QUESTION:")[1].split("Please answer")[0].strip()
                else:
                    context_part = content
                    user_q = ""

                q_lower = user_q.lower()
                c_lower = context_part.lower()

                if not context_part or len(context_part.strip()) < 15:
                    return type("Resp", (), {"content": "I couldn't find sufficient information in the uploaded documents."})()

                all_tokens = re.findall(r"\b[a-zA-Z0-9_\+\-]{2,}\b", q_lower)
                query_tokens = [t for t in all_tokens if t not in GENERIC_STOP_WORDS]

                if query_tokens and not any(t in c_lower for t in query_tokens):
                    return type("Resp", (), {"content": "I couldn't find sufficient information in the uploaded documents."})()

                passages = [p.strip() for p in re.split(r"\n\n|\n", context_part) if len(p.strip()) > 15 and not p.strip().startswith("[Document")]
                
                scored_passages = []
                for p in passages:
                    p_lower = p.lower()
                    overlap = sum(1 for t in query_tokens if t in p_lower)
                    if overlap > 0:
                        scored_passages.append((p, overlap))

                scored_passages.sort(key=lambda x: x[1], reverse=True)

                if scored_passages:
                    top_passages = [p for p, score in scored_passages[:5]]
                    answer_text = "\n\n".join(f"- {p.lstrip('•').lstrip('-').strip()}" for p in top_passages)
                    return type("Resp", (), {"content": answer_text})()

                if passages:
                    return type("Resp", (), {"content": "\n\n".join(passages[:2])})()

                return type("Resp", (), {"content": "I couldn't find sufficient information in the uploaded documents."})()

        return ExtractiveGroundingLLM()

    def answer_query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        hybrid_alpha: Optional[float] = None,
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Executes multi-question RAG pipeline:
        1. Decomposes compound queries into independent sub-questions
        2. Retrieves relevant chunks independently for each sub-question
        3. Fuses and deduplicates retrieved context across sub-questions
        4. Synthesizes structured grounded answer with Groq LLM
        5. Attaches precise citations
        """
        # 1. Multi-question Decomposition
        sub_questions = query_normalizer.decompose_sub_questions(query)
        if not sub_questions:
            sub_questions = [query]

        combined_ranked_chunks: List[Tuple[Document, float, str, str]] = []
        seen_fingerprints = set()
        sub_question_traces = []

        # 2. Retrieve Independently for Each Sub-Question
        for sq in sub_questions:
            normalized_sq, target_doc_hint = query_normalizer.normalize(sq)
            effective_sq = normalized_sq if normalized_sq else sq

            # Retrieve candidates for this specific sub-question
            raw_sq_chunks = hybrid_retriever.retrieve(
                query=effective_sq,
                top_k=max(top_k, 6),
                document_ids=document_ids,
                hybrid_alpha=hybrid_alpha
            )

            # Strict post-filter by document_ids if provided
            if document_ids:
                doc_id_set = set(document_ids)
                raw_sq_chunks = [item for item in raw_sq_chunks if item[0].metadata.get("document_id") in doc_id_set]

            # Rerank for this sub-question
            ranked_sq_chunks = reranker.rerank_and_filter(
                query=effective_sq,
                retrieved_items=raw_sq_chunks,
                top_k=min(top_k, 3),
                target_document_hint=target_doc_hint
            )

            # Trace for developer inspection
            sq_trace = {
                "sub_question": sq,
                "retrieved_count": len(ranked_sq_chunks),
                "chunks": [
                    f"{rc[0].metadata.get('document_name', 'Doc')} (P.{rc[0].metadata.get('page_number', 1)}) [Score: {rc[1]}]"
                    for rc in ranked_sq_chunks
                ]
            }
            sub_question_traces.append(sq_trace)

            # Fuse into combined context with deduplication
            for rc in ranked_sq_chunks:
                doc, score, method, grade = rc
                fp = reranker._text_fingerprint(doc.page_content)
                if fp not in seen_fingerprints:
                    seen_fingerprints.add(fp)
                    combined_ranked_chunks.append(rc)

        # 3. Check for empty retrieval across all sub-questions
        if not combined_ranked_chunks or (len(combined_ranked_chunks) > 0 and combined_ranked_chunks[0][1] < 0.28):
            return ChatResponse(
                answer="I couldn't find sufficient information in the uploaded documents.",
                session_id=session_id or "default",
                citations=[],
                tool_traces=[],
                grounded=True,
                grounding_confidence=1.0
            )

        # 4. Assemble Grounded Context from Fused Chunks
        context_blocks = []
        for doc, score, method, grade in combined_ranked_chunks:
            meta = doc.metadata
            c_block = (
                f"[Document: {meta.get('document_name', 'Unknown')} | Page: {meta.get('page_number', 1)}]\n"
                f"{doc.page_content.strip()}"
            )
            context_blocks.append(c_block)

        context_str = "\n\n--------------------\n\n".join(context_blocks)

        user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:
{context_str}

USER QUESTION(S):
{query}

Please answer each question directly, clearly, and concisely based ONLY on the context above.
If the information for any specific question is not present in the context, state clearly for that question:
"I couldn't find sufficient information in the uploaded documents." """

        # 5. Resolve LLM
        try:
            llm = self._get_llm()
        except Exception as e:
            return ChatResponse(
                answer=f"⚠️ Configuration Error: {str(e)}",
                session_id=session_id or "default",
                citations=[],
                tool_traces=[],
                grounded=False,
                grounding_confidence=0.0
            )

        messages = [
            {"role": "system", "content": STRICT_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = llm.invoke(messages)
            raw_answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            if settings.DEFAULT_LLM_PROVIDER == "groq" and "model_not_found" in str(e).lower():
                try:
                    from langchain_groq import ChatGroq
                    for alt_model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]:
                        try:
                            fallback_llm = ChatGroq(
                                api_key=settings.GROQ_API_KEY.strip(),
                                model_name=alt_model,
                                temperature=0.1
                            )
                            resp = fallback_llm.invoke(messages)
                            raw_answer = resp.content if hasattr(resp, "content") else str(resp)
                            settings.GROQ_MODEL = alt_model
                            break
                        except Exception:
                            continue
                except Exception as inner_e:
                    logger.error(f"Fallback model error: {inner_e}")
                    raw_answer = f"Error during generation: {str(e)}"
            else:
                logger.error(f"LLM generation error: {e}")
                raw_answer = f"Error during generation: {str(e)}"

        # 6. Build Citations for Grounded Chunks
        if "i couldn't find sufficient information in the uploaded documents" in raw_answer.lower() and len(sub_questions) == 1:
            citations = []
            is_grounded = True
            confidence = 1.0
        else:
            valid_chunks = [rc for rc in combined_ranked_chunks if rc[1] >= 0.35]
            if document_ids:
                doc_id_set = set(document_ids)
                valid_chunks = [rc for rc in valid_chunks if rc[0].metadata.get("document_id") in doc_id_set]
            citations = citation_engine.build_citations(valid_chunks[:4])
            is_grounded, confidence = citation_engine.evaluate_grounding(raw_answer, citations)

        trace = ToolExecutionTrace(
            tool_name="multi_question_rag_retriever",
            input_params={"query": query, "sub_questions": sub_questions, "document_ids": document_ids},
            output={"sub_question_traces": sub_question_traces, "total_fused_chunks": len(combined_ranked_chunks)},
            execution_time_ms=0,
            status="success"
        )

        return ChatResponse(
            answer=raw_answer,
            session_id=session_id or "default",
            citations=citations,
            tool_traces=[trace],
            grounded=is_grounded,
            grounding_confidence=confidence
        )

rag_chain = RAGChain()
