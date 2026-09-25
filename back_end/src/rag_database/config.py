import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


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
        "env_file": ".env",
        "extra": "ignore",
    }


rag_config = RAGSettings()
