import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

_BASE_DIR = Path(__file__).resolve().parent.parent.parent  # back_end
_REPO_ROOT = _BASE_DIR.parent if (_BASE_DIR.parent / ".env").exists() else _BASE_DIR

# Load .env from repo root first, then back_end if present
if (_REPO_ROOT / ".env").exists():
    load_dotenv(_REPO_ROOT / ".env")
if (_BASE_DIR / ".env").exists():
    load_dotenv(_BASE_DIR / ".env")


class RAGSettings(BaseSettings):
    """
    Settings for the RAG and Vector Database pipeline.
    Reads values from environment variables with sensible defaults.
    Values can be overridden via .env file or shell environment variables.
    """

    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024

    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    VECTOR_DB_COLLECTION_NAME: str = "zar3a_knowledge_base"
    VECTOR_DB_API_KEY: Optional[str] = None

    RAG_BATCH_SIZE: int = 50
    QDRANT_LOCAL_PATH: Optional[str] = None

    RETRIEVER_TOP_K: int = 5
    RRF_K: int = 60
    CANDIDATE_POOL_MULTIPLIER: int = 4

    model_config = {
        "env_file": str(_REPO_ROOT / ".env") if (_REPO_ROOT / ".env").exists() else ".env",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def resolve_local_path(self) -> "RAGSettings":
        if self.QDRANT_LOCAL_PATH:
            p = Path(self.QDRANT_LOCAL_PATH)
            if not p.is_absolute():
                self.QDRANT_LOCAL_PATH = str((_REPO_ROOT / p).resolve())
        return self


rag_config = RAGSettings()

