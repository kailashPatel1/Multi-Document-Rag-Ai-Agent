import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from ...models.schemas import DocumentSummaryRequest, DocumentCompareRequest, CSVQueryRequest, CSVQueryResponse
from ...tools.rag_tools import document_summary_tool, document_comparison_tool, metadata_search_tool
from ...tools.csv_tool import csv_analysis_tool
from ...tools.calc_tool import calculator_tool

router = APIRouter(prefix="/api/tools", tags=["Tools"])
logger = logging.getLogger(__name__)

@router.post("/summary")
def summarize_document(req: DocumentSummaryRequest):
    """Summarizes a specific uploaded document."""
    res = document_summary_tool(document_name_or_id=req.document_id, focus=req.focus)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

@router.post("/compare")
def compare_documents(req: DocumentCompareRequest):
    """Compares multiple documents with target queries."""
    doc_a = req.document_ids[0]
    doc_b = req.document_ids[1]
    res = document_comparison_tool(
        document_a_name=doc_a,
        document_b_name=doc_b,
        comparison_query=req.query or "Compare the main themes and requirements across both documents."
    )
    return res

@router.post("/csv-query", response_model=CSVQueryResponse)
def query_csv_data(req: CSVQueryRequest):
    """Executes safe Pandas calculations on an uploaded CSV."""
    res = csv_analysis_tool(query=req.query, document_name_or_id=req.document_id)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    
    return CSVQueryResponse(
        document_id=req.document_id,
        query=req.query,
        result_type=res.get("operation", "analysis"),
        result=res.get("computed_value") or res.get("record") or res.get("grouped_data") or res.get("descriptive_statistics"),
        explanation=res.get("summary", "")
    )

@router.post("/calculate")
def calculate_expression(expression: str):
    """Evaluates mathematical expressions safely."""
    res = calculator_tool(expression=expression)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@router.get("/metadata")
def get_metadata_catalog():
    """Lists document catalog metadata."""
    return metadata_search_tool()
