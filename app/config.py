import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """
    Application configuration settings.
    Uses environment variables with fallback to default values.
    """
    
    # === Application Settings ===
    APP_NAME: str = "RAG Document Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # === API Settings ===
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["*"]
    
    # === Model Configuration ===
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    GENERATION_MODEL: str = "microsoft/DialoGPT-medium"
    USE_GPU: bool = False
    DEVICE: str = "cuda" if USE_GPU else "cpu"
    
    # === Document Processing ===
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt"]
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # === Retrieval Settings ===
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # === File Paths ===
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    
    # === Database Files ===
    FAISS_INDEX_PATH: Path = EMBEDDINGS_DIR / "document_index.faiss"
    METADATA_FILE: Path = METADATA_DIR / "documents.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings() 