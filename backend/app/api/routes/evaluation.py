import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from ...models.schemas import EvaluationRunRequest, EvaluationResultResponse, EvaluationItem
from ...evaluation.evaluation_service import evaluation_service

router = APIRouter(prefix="/api/evaluation", tags=["RAG Evaluation"])
logger = logging.getLogger(__name__)

@router.post("/run", response_model=EvaluationResultResponse)
def run_evaluation_suite(req: EvaluationRunRequest):
    """
    Executes the automated RAG evaluation suite.
    Calculates Precision, Recall, Context Relevance, Faithfulness, Correctness, Citation Accuracy, and Latency.
    """
    try:
        results = evaluation_service.run_evaluation(items=req.custom_items, top_k=req.top_k)
        return results
    except Exception as e:
        logger.error(f"Evaluation run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/results")
def get_evaluation_results():
    """Retrieves the most recent evaluation benchmark results."""
    results = evaluation_service.get_latest_results()
    if not results:
        # Run default evaluation if none exists yet
        return evaluation_service.run_evaluation()
    return results

@router.get("/dataset", response_model=List[EvaluationItem])
def get_benchmark_dataset():
    """Retrieves the curated benchmark QA evaluation dataset."""
    return evaluation_service.load_benchmark_dataset()
