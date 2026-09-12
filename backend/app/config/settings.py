import os
import re
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
CHROMA_DIR = DATA_DIR / "chroma_db"
EVALUATION_DIR = BASE_DIR / "evaluation"
EVAL_DATASETS_DIR = EVALUATION_DIR / "datasets"
EVAL_RESULTS_DIR = EVALUATION_DIR / "results"
ENV_FILE = BASE_DIR / ".env"

for folder in [UPLOADS_DIR, SAMPLE_DOCS_DIR, CHROMA_DIR, EVAL_DATASETS_DIR, EVAL_RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Document AI Knowledge Agent"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    UPLOADS_DIR: Path = UPLOADS_DIR
    SAMPLE_DOCS_DIR: Path = SAMPLE_DOCS_DIR
    CHROMA_DIR: Path = CHROMA_DIR
    EVALUATION_DIR: Path = EVALUATION_DIR
    EVAL_DATASETS_DIR: Path = EVAL_DATASETS_DIR
    EVAL_RESULTS_DIR: Path = EVAL_RESULTS_DIR
    ENV_FILE_PATH: Path = ENV_FILE

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = "groq"  # "groq", "gemini", or "mock"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    # Embedding Settings
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Settings
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 5
    HYBRID_ALPHA: float = 0.5
    SIMILARITY_THRESHOLD: float = 0.25
    RERANKING_ENABLED: bool = True
    
    # Database & Storage
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'knowledge_agent.db'}"
    CHROMA_PERSIST_DIRECTORY: str = str(CHROMA_DIR)
    
    # Security & Auth
    SECRET_KEY: str = "knowledge-agent-super-secret-key-2026-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    
    # File Limits
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".csv", ".md"]

    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="allow")

settings = Settings()

if not settings.GROQ_API_KEY and os.getenv("GROQ_API_KEY"):
    settings.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not settings.GEMINI_API_KEY and os.getenv("GEMINI_API_KEY"):
    settings.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def mask_api_key(key: Optional[str]) -> str:
    """Returns a masked version of the API key for safe UI status display."""
    if not key or not key.strip():
        return ""
    k = key.strip()
    if len(k) <= 8:
        return "••••••••"
    return f"••••••••{k[-4:]}"

def save_env_variables(updates: Dict[str, Any]):
    """
    Safely writes/updates key-value pairs in the .env file and updates runtime settings and os.environ.
    """
    env_path = ENV_FILE
    existing_lines = []
    if env_path.exists():
        try:
            existing_lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            existing_lines = []

    env_dict: Dict[str, str] = {}
    ordered_keys = []
    for line in existing_lines:
        line_clean = line.strip()
        if line_clean and not line_clean.startswith("#") and "=" in line_clean:
            k, v = line_clean.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            env_dict[k] = v
            if k not in ordered_keys:
                ordered_keys.append(k)

    for k, v in updates.items():
        if v is not None:
            v_str = str(v).strip()
            env_dict[k] = v_str
            os.environ[k] = v_str
            if hasattr(settings, k):
                setattr(settings, k, v)
            if k not in ordered_keys:
                ordered_keys.append(k)

    out_lines = []
    for k in ordered_keys:
        out_lines.append(f"{k}={env_dict[k]}")

    env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
