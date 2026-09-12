import os
import re
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from ..config.settings import settings
from ..rag.rag_chain import rag_chain
from ..models.db_models import SessionLocal, EvaluationRunModel, DocumentModel
from ..models.schemas import EvaluationItem, EvaluationMetricSummary, EvaluationResultResponse

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_ITEMS: List[Dict[str, Any]] = [
    {
        "id": "eval_1",
        "question": "What is RAG and why is it used?",
        "expected_answer": "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models to ground answers in private documents and eliminate hallucinations.",
        "relevant_document": "RAG_Notes.pdf",
        "relevant_page": 1
    },
    {
        "id": "eval_2",
        "question": "How do text embeddings work in vector search?",
        "expected_answer": "Embeddings transform text into dense numerical vectors in a multi-dimensional semantic space where cosine distance captures semantic similarity.",
        "relevant_document": "RAG_Notes.pdf",
        "relevant_page": 2
    },
    {
        "id": "eval_3",
        "question": "What is hybrid retrieval and Reciprocal Rank Fusion?",
        "expected_answer": "Hybrid retrieval fuses dense vector similarity with BM25 keyword matching using Reciprocal Rank Fusion (RRF) to combine semantic understanding with exact term precision.",
        "relevant_document": "RAG_Notes.pdf",
        "relevant_page": 3
    },
    {
        "id": "eval_4",
        "question": "What are transformer attention mechanisms?",
        "expected_answer": "Self-attention computes dynamic weights between all token pairs in a sequence using Query, Key, and Value projections to capture long-range contextual relationships.",
        "relevant_document": "ML_Notes.pdf",
        "relevant_page": 1
    },
    {
        "id": "eval_5",
        "question": "What candidate skills are listed in the resume?",
        "expected_answer": "The resume lists Python, PyTorch, LangChain, ChromaDB, FastAPI, Vector Databases, Transformers, and GenAI system engineering.",
        "relevant_document": "Resume.pdf",
        "relevant_page": 1
    },
    {
        "id": "eval_6",
        "question": "What are the required qualifications in the job description?",
        "expected_answer": "The job description requires experience with LLM agents, LangGraph, RAG pipelines, Vector databases, production API development, and distributed systems.",
        "relevant_document": "Job_Description.pdf",
        "relevant_page": 1
    }
]

def normalize_doc_identifier(name_or_id: Optional[str]) -> str:
    """
    Normalizes document filenames, paths, and UUIDs to a clean canonical token for safe identity matching.
    Handles:
    - Path prefixes (e.g., 'data/uploads/doc_123_RAG_Notes.pdf' -> 'RAG_Notes.pdf')
    - UUID prefixes (e.g., 'doc_ab4bdfa3bac0_RAG_Notes.pdf' -> 'RAG_Notes.pdf')
    - Duplicate dots and extensions (e.g., 'Kailash_Resume..pdf' -> 'kailash_resume')
    - Case insensitivity and whitespace stripping
    """
    if not name_or_id:
        return ""
    
    # Strip any directory path
    base = Path(str(name_or_id)).name
    
    # Strip UUID prefix like doc_0123456789ab_
    base = re.sub(r"^doc_[a-f0-9]{8,16}_", "", base, flags=re.IGNORECASE)
    
    # Remove file extensions (.pdf, .csv, .txt, .docx, .md)
    base = re.sub(r"\.(pdf|csv|docx|txt|md)$", "", base, flags=re.IGNORECASE)
    
    # Remove any duplicate trailing dots or separators
    base = re.sub(r"[\.\s_-]+$", "", base)
    
    # Convert punctuation separators to single underscore and lowercase
    base = re.sub(r"[\s\.\-]+", "_", base.strip().lower())
    
    return base

def match_document_identity(
    expected_identifier: str,
    retrieved_doc_name: str,
    retrieved_doc_id: str,
    doc_lookup: Dict[str, Any]
) -> bool:
    """
    Determines whether a retrieved document/chunk matches the expected benchmark document.
    Uses exact ID matching, DB lookup mapping, and canonical normalized filename comparison.
    Never causes false-positive matches between different documents (e.g. ML_Notes vs RAG_Notes).
    """
    if not expected_identifier:
        return False
        
    exp_raw = str(expected_identifier).strip()
    ret_name_raw = str(retrieved_doc_name or "").strip()
    ret_id_raw = str(retrieved_doc_id or "").strip()
    
    # 1. Exact ID match
    if ret_id_raw and (exp_raw == ret_id_raw):
        return True
        
    # 2. Check if expected_identifier is a document ID present in DB lookup
    if exp_raw in doc_lookup:
        expected_meta = doc_lookup[exp_raw]
        if expected_meta.get("id") == ret_id_raw:
            return True
        exp_raw = expected_meta.get("original_name") or exp_raw

    # 3. Check if retrieved_doc_id is present in DB lookup
    if ret_id_raw in doc_lookup:
        ret_name_raw = doc_lookup[ret_id_raw].get("original_name") or ret_name_raw

    # 4. Canonical normalized token matching
    norm_exp = normalize_doc_identifier(exp_raw)
    norm_ret = normalize_doc_identifier(ret_name_raw)

    if norm_exp and norm_ret and norm_exp == norm_ret:
        return True

    return False

class EvaluationService:
    """Production RAG Evaluation framework measuring retrieval precision/recall, generation faithfulness, correctness, and citation accuracy."""

    def __init__(self):
        self._ensure_benchmark_dataset_exists()

    def _ensure_benchmark_dataset_exists(self):
        benchmark_file = settings.EVAL_DATASETS_DIR / "benchmark_qa.json"
        if not benchmark_file.exists():
            try:
                with open(benchmark_file, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_BENCHMARK_ITEMS, f, indent=2)
                logger.info(f"Created default benchmark dataset at: {benchmark_file}")
            except Exception as e:
                logger.warning(f"Could not save benchmark dataset: {e}")

    def load_benchmark_dataset(self, name: str = "default") -> List[EvaluationItem]:
        benchmark_file = settings.EVAL_DATASETS_DIR / "benchmark_qa.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                return [EvaluationItem(**item) for item in items]
            except Exception as e:
                logger.error(f"Error loading benchmark file: {e}")
        return [EvaluationItem(**item) for item in DEFAULT_BENCHMARK_ITEMS]

    def _build_doc_lookup_map(self) -> Dict[str, Dict[str, Any]]:
        """Builds a fast lookup dictionary from current database documents."""
        lookup = {}
        db = SessionLocal()
        try:
            docs = db.query(DocumentModel).all()
            for d in docs:
                entry = {
                    "id": d.id,
                    "filename": d.filename,
                    "original_name": d.original_name,
                    "file_type": d.file_type,
                    "total_pages": d.total_pages
                }
                lookup[d.id] = entry
                lookup[d.filename] = entry
                lookup[d.original_name] = entry
                norm_name = normalize_doc_identifier(d.original_name)
                if norm_name:
                    lookup[norm_name] = entry
        except Exception as e:
            logger.warning(f"Failed to build document lookup map: {e}")
        finally:
            db.close()
        return lookup

    def run_evaluation(
        self,
        items: Optional[List[EvaluationItem]] = None,
        top_k: int = 5
    ) -> EvaluationResultResponse:
        """
        Runs comprehensive evaluation against the RAG pipeline.
        Calculates:
        - Retrieval Precision: Fraction of retrieved chunks matching the target document
        - Retrieval Recall: Whether the target document was retrieved (1.0 or 0.0)
        - Context Relevance: Average similarity score of retrieved chunks
        - Answer Faithfulness: Grounding score / lack of hallucination
        - Answer Correctness: Semantic word overlap with expected answer
        - Citation Accuracy: Whether citation references the exact target document and page
        """
        eval_items = items or self.load_benchmark_dataset()
        if not eval_items:
            eval_items = [EvaluationItem(**i) for i in DEFAULT_BENCHMARK_ITEMS]

        # Ensure demo documents are indexed if running default benchmark and DB has missing files
        db = SessionLocal()
        try:
            db_doc_count = db.query(DocumentModel).count()
            if db_doc_count == 0:
                from ..api.routes.documents import load_demo_files
                logger.info("Database empty on evaluation start: auto-loading demo benchmark documents...")
                load_demo_files(db=db)
        except Exception as e:
            logger.warning(f"Error checking demo document availability: {e}")
        finally:
            db.close()

        doc_lookup = self._build_doc_lookup_map()

        run_id = f"eval_run_{uuid.uuid4().hex[:10]}"
        start_overall = time.time()

        item_results = []
        precision_scores = []
        recall_scores = []
        context_relevance_scores = []
        faithfulness_scores = []
        correctness_scores = []
        citation_accuracy_scores = []
        latencies = []

        for item in eval_items:
            t0 = time.time()
            # Execute RAG query through the core pipeline
            resp = rag_chain.answer_query(query=item.question, top_k=top_k)
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)

            # Retrieve citations and document metadata
            citations = resp.citations or []
            
            # List of unique retrieved document names for clean display in UI
            retrieved_doc_names = []
            for c in citations:
                dname = c.document_name or c.document_id or "Unknown"
                if dname not in retrieved_doc_names:
                    retrieved_doc_names.append(dname)

            expected_doc = item.relevant_document or ""
            expected_page = item.relevant_page or 1

            # 1. Retrieval Precision & Recall Calculation
            matching_chunks = 0
            has_matching_doc = False
            target_page_hit = False

            for c in citations:
                is_match = match_document_identity(
                    expected_identifier=expected_doc,
                    retrieved_doc_name=c.document_name,
                    retrieved_doc_id=c.document_id,
                    doc_lookup=doc_lookup
                )
                if is_match:
                    matching_chunks += 1
                    has_matching_doc = True
                    if c.page_number == expected_page:
                        target_page_hit = True

            # Precision: Fraction of retrieved citations that belong to expected document
            if len(citations) > 0:
                precision = matching_chunks / len(citations)
            else:
                precision = 0.0

            # Recall: Did retrieval successfully fetch the expected document?
            recall = 1.0 if has_matching_doc else 0.0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # 2. Context Relevance (Average similarity score of retrieved citations)
            if citations:
                avg_sim = sum(c.similarity_score for c in citations) / len(citations)
            else:
                avg_sim = 0.0
            context_relevance_scores.append(avg_sim)

            # 3. Answer Faithfulness (Grounding Confidence)
            faithfulness = resp.grounding_confidence if resp.grounding_confidence is not None else 1.0
            faithfulness_scores.append(faithfulness)

            # 4. Answer Correctness (Token overlap between generated answer & expected answer)
            answer_tokens = set(re.findall(r"\b\w{3,}\b", resp.answer.lower()))
            expected_tokens = set(re.findall(r"\b\w{3,}\b", (item.expected_answer or "").lower()))
            if expected_tokens:
                overlap = len(answer_tokens.intersection(expected_tokens))
                correctness = min(1.0, (overlap / len(expected_tokens)) * 1.5)
            else:
                correctness = 0.8
            correctness_scores.append(correctness)

            # 5. Citation Accuracy (Validates document and page grounding)
            if target_page_hit:
                citation_acc = 1.0
            elif has_matching_doc:
                citation_acc = 0.8
            else:
                citation_acc = 0.0
            citation_accuracy_scores.append(citation_acc)

            # 6. Build item result populated with exact fields expected by frontend
            item_result_entry = {
                "query": item.question,
                "question": item.question,
                "expected_document": expected_doc,
                "relevant_document": expected_doc,
                "expected_answer": item.expected_answer,
                "generated_answer": resp.answer,
                "target_document": expected_doc,
                "target_page": expected_page,
                "retrieved_documents": retrieved_doc_names,
                "retrieved_citations": [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in citations],
                "retrieval_precision": round(precision, 3),
                "retrieval_recall": round(recall, 3),
                "context_relevance": round(avg_sim, 3),
                "answer_faithfulness": round(faithfulness, 3),
                "answer_correctness": round(correctness, 3),
                "citation_accuracy": round(citation_acc, 3),
                "latency_ms": round(latency_ms, 1)
            }
            item_results.append(item_result_entry)

            logger.info(
                f"[EVAL] Query='{item.question[:40]}...' | Expected='{expected_doc}' | "
                f"Retrieved={retrieved_doc_names} | P={precision:.2f} | R={recall:.2f} | "
                f"CitAcc={citation_acc:.2f} | Latency={latency_ms:.0f}ms"
            )

        # Calculate averages across all evaluated queries
        n = max(1, len(eval_items))
        summary = EvaluationMetricSummary(
            total_queries=len(eval_items),
            retrieval_precision=round(sum(precision_scores) / n, 3),
            retrieval_recall=round(sum(recall_scores) / n, 3),
            context_relevance=round(sum(context_relevance_scores) / n, 3),
            answer_faithfulness=round(sum(faithfulness_scores) / n, 3),
            answer_correctness=round(sum(correctness_scores) / n, 3),
            citation_accuracy=round(sum(citation_accuracy_scores) / n, 3),
            average_latency_ms=round(sum(latencies) / n, 1)
        )

        result_resp = EvaluationResultResponse(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            metrics=summary,
            item_results=item_results
        )

        # Persist run to SQLite database
        db = SessionLocal()
        try:
            run_record = EvaluationRunModel(
                id=run_id,
                total_queries=summary.total_queries,
                retrieval_precision=summary.retrieval_precision,
                retrieval_recall=summary.retrieval_recall,
                context_relevance=summary.context_relevance,
                answer_faithfulness=summary.answer_faithfulness,
                answer_correctness=summary.answer_correctness,
                citation_accuracy=summary.citation_accuracy,
                average_latency_ms=summary.average_latency_ms,
                details_json=json.dumps(item_results)
            )
            db.add(run_record)
            db.commit()
        except Exception as e:
            logger.error(f"Error persisting evaluation run to DB: {e}")
        finally:
            db.close()

        # Save to evaluation/results json artifact
        try:
            out_path = settings.EVAL_RESULTS_DIR / f"{run_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result_resp.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Could not write results file: {e}")

        return result_resp

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent evaluation run from SQLite database."""
        db = SessionLocal()
        try:
            latest = db.query(EvaluationRunModel).order_by(EvaluationRunModel.timestamp.desc()).first()
            if not latest:
                return None
            
            try:
                details = json.loads(latest.details_json or "[]")
            except Exception:
                details = []

            return {
                "run_id": latest.id,
                "timestamp": latest.timestamp.isoformat(),
                "metrics": {
                    "total_queries": latest.total_queries,
                    "retrieval_precision": latest.retrieval_precision,
                    "retrieval_recall": latest.retrieval_recall,
                    "context_relevance": latest.context_relevance,
                    "answer_faithfulness": latest.answer_faithfulness,
                    "answer_correctness": latest.answer_correctness,
                    "citation_accuracy": latest.citation_accuracy,
                    "average_latency_ms": latest.average_latency_ms
                },
                "item_results": details
            }
        finally:
            db.close()

evaluation_service = EvaluationService()
