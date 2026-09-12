import pytest
from backend.app.tools.calc_tool import calculator_tool
from backend.app.tools.csv_tool import csv_analysis_tool
from backend.app.services.sample_generator import generate_sample_documents
from backend.app.models.db_models import init_db, SessionLocal, DocumentModel

@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    init_db()
    files = generate_sample_documents()
    db = SessionLocal()
    existing = db.query(DocumentModel).filter(DocumentModel.file_type == ".csv").first()
    if not existing:
        doc = DocumentModel(
            id="doc_csv_test",
            filename="Sample_Sales.csv",
            original_name="Sample_Sales.csv",
            file_type=".csv",
            file_path=files["sample_sales"],
            file_size=1024,
            total_pages=1,
            total_chunks=1,
            status="ready"
        )
        db.add(doc)
        db.commit()
    db.close()

def test_calculator_tool():
    res1 = calculator_tool("120 * 45")
    assert res1["computed_result"] == 5400.0

    res2 = calculator_tool("sqrt(144) + 10")
    assert res2["computed_result"] == 22.0

def test_csv_analysis_average():
    res = csv_analysis_tool(query="What is the average Revenue?")
    assert "average" in res.get("operation", "").lower()
    assert "computed_value" in res
    assert res["computed_value"] > 0

def test_csv_analysis_highest():
    res = csv_analysis_tool(query="Which product has the highest Revenue?")
    assert "maximum" in res.get("operation", "").lower() or "record" in res
    assert "record" in res
