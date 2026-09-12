import os
import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from ..models.db_models import SessionLocal, DocumentModel

logger = logging.getLogger(__name__)

def csv_analysis_tool(query: str, document_name_or_id: Optional[str] = None) -> Dict[str, Any]:
    """
    CSV Analysis Tool: Performs safe, deterministic statistical and analytical calculations on uploaded CSV files
    using Pandas (mean, median, sum, top-N, groupings, filters) without hallucinating numerical values.
    """
    db = SessionLocal()
    try:
        # Find CSV document
        query_builder = db.query(DocumentModel).filter(DocumentModel.file_type == ".csv")
        if document_name_or_id:
            doc = query_builder.filter(
                (DocumentModel.id == document_name_or_id) |
                (DocumentModel.original_name.ilike(f"%{document_name_or_id}%"))
            ).first()
        else:
            doc = query_builder.first()

        if not doc or not os.path.exists(doc.file_path):
            return {
                "error": "No uploaded CSV document found to analyze.",
                "available_csvs": [d.original_name for d in db.query(DocumentModel).filter(DocumentModel.file_type == ".csv").all()]
            }

        # Load DataFrame
        try:
            df = pd.read_csv(doc.file_path)
        except Exception:
            df = pd.read_csv(doc.file_path, encoding="latin-1")

        q_lower = query.lower()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        str_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        result_payload: Dict[str, Any] = {
            "document_name": doc.original_name,
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "query": query
        }

        # 1. Average / Mean query
        if "average" in q_lower or "mean" in q_lower or "avg" in q_lower:
            # Find referenced column
            target_col = None
            for c in num_cols:
                if c.lower() in q_lower:
                    target_col = c
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                avg_val = float(df[target_col].mean())
                result_payload["operation"] = f"Average of {target_col}"
                result_payload["computed_value"] = round(avg_val, 2)
                result_payload["summary"] = f"The average (mean) value of '{target_col}' across all {len(df)} records is {round(avg_val, 2)}."
                return result_payload

        # 2. Total / Sum query
        if "total" in q_lower or "sum" in q_lower:
            target_col = None
            for c in num_cols:
                if c.lower() in q_lower:
                    target_col = c
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                sum_val = float(df[target_col].sum())
                result_payload["operation"] = f"Total sum of {target_col}"
                result_payload["computed_value"] = round(sum_val, 2)
                result_payload["summary"] = f"The total sum of '{target_col}' is {round(sum_val, 2)}."
                return result_payload

        # 3. Highest / Maximum / Top
        if "highest" in q_lower or "maximum" in q_lower or "max" in q_lower or "top" in q_lower or "best" in q_lower:
            target_col = None
            for c in num_cols:
                if c.lower() in q_lower:
                    target_col = c
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                top_row = df.loc[df[target_col].idxmax()].to_dict()
                result_payload["operation"] = f"Maximum of {target_col}"
                result_payload["record"] = top_row
                result_payload["summary"] = f"The highest '{target_col}' is {top_row[target_col]}, recorded for: {top_row}."
                return result_payload

        # 4. Lowest / Minimum
        if "lowest" in q_lower or "minimum" in q_lower or "min" in q_lower or "worst" in q_lower:
            target_col = None
            for c in num_cols:
                if c.lower() in q_lower:
                    target_col = c
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                bottom_row = df.loc[df[target_col].idxmin()].to_dict()
                result_payload["operation"] = f"Minimum of {target_col}"
                result_payload["record"] = bottom_row
                result_payload["summary"] = f"The lowest '{target_col}' is {bottom_row[target_col]}, recorded for: {bottom_row}."
                return result_payload

        # 5. Group by / Category aggregation
        for sc in str_cols:
            if sc.lower() in q_lower and num_cols:
                target_num = num_cols[0]
                grouped = df.groupby(sc)[target_num].agg(["sum", "mean", "count"]).reset_index()
                grouped_records = grouped.to_dict(orient="records")
                result_payload["operation"] = f"Grouped by {sc} with {target_num}"
                result_payload["grouped_data"] = grouped_records
                result_payload["summary"] = f"Grouped aggregations for '{sc}' against '{target_num}':\n" + grouped.to_string()
                return result_payload

        # Default: Full descriptive statistics + sample records
        desc = df.describe(include="all").fillna("").to_dict()
        result_payload["operation"] = "Dataset Summary & Descriptive Statistics"
        result_payload["descriptive_statistics"] = desc
        result_payload["sample_records"] = df.head(5).to_dict(orient="records")
        result_payload["summary"] = f"Dataset Overview for '{doc.original_name}':\n" + df.describe().to_string()
        return result_payload

    finally:
        db.close()
