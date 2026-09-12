import os
import pytest
from backend.app.config.settings import settings, mask_api_key, save_env_variables, ENV_FILE
from backend.app.api.routes.system import update_settings, get_system_status, SettingsUpdateRequest
from backend.app.models.db_models import init_db, SessionLocal

@pytest.fixture(scope="module", autouse=True)
def preserve_env_file():
    init_db()
    original_env_content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else None
    yield
    # Teardown: strictly restore original .env file
    if original_env_content is not None:
        ENV_FILE.write_text(original_env_content, encoding="utf-8")

def test_key_masking():
    dummy_key = "gsk_live_test_secret_key_123456789abc"
    masked = mask_api_key(dummy_key)
    assert masked.endswith("9abc")
    assert "secret" not in masked
    assert mask_api_key("") == ""
    assert mask_api_key(None) == ""

def test_settings_persistence():
    test_key = "gsk_pytest_test_token_8899aabbccdd"
    req = SettingsUpdateRequest(
        default_llm_provider="groq",
        groq_api_key=test_key
    )
    res = update_settings(req)
    assert res["status"] == "success"
    assert res["active_settings"]["has_groq_key"] is True
    assert res["active_settings"]["groq_key_masked"].endswith("ccdd")
    
    # Verify .env file on disk
    assert ENV_FILE.exists()
    content = ENV_FILE.read_text(encoding="utf-8")
    assert f"GROQ_API_KEY={test_key}" in content

def test_system_status_endpoint():
    db = SessionLocal()
    status = get_system_status(db)
    db.close()
    assert "llm_provider" in status
    assert "vector_chunks_indexed" in status
    assert status.get("status") == "operational"
