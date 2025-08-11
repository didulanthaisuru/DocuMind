from functools import lru_cache
from ..config import settings
from ..services.embedding_service import EmbeddingService
from ..services.vector_service import VectorStoreService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.prompt_service import PromptService, PromptConfig
from ..utils.file_handler import FileHandler
from ..utils.text_processor import TextProcessor

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service instance."""
    return EmbeddingService(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.DEVICE
    )

@lru_cache()
def get_vector_service() -> VectorStoreService:
    """Get singleton vector service instance."""
    embedding_service = get_embedding_service()
    dimension = embedding_service.get_embedding_dimension()
    
    return VectorStoreService(
        index_path=settings.FAISS_INDEX_PATH,
        metadata_path=settings.METADATA_DIR / "vectors.json",
        dimension=dimension
    )

@lru_cache()
def get_file_handler() -> FileHandler:
    """Get singleton file handler instance."""
    return FileHandler(
        upload_dir=settings.UPLOAD_DIR,
        allowed_extensions=settings.ALLOWED_EXTENSIONS,
        max_size=settings.MAX_FILE_SIZE
    )

@lru_cache()
def get_text_processor() -> TextProcessor:
    """Get singleton text processor instance."""
    return TextProcessor(
        chunk_size=settings.MAX_CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP
    )

@lru_cache()
def get_document_service() -> DocumentService:
    """Get singleton document service instance."""
    return DocumentService(
        file_handler=get_file_handler(),
        text_processor=get_text_processor(),
        metadata_file=settings.METADATA_FILE
    )

@lru_cache()
def get_prompt_config() -> PromptConfig:
    """Get prompt configuration from settings."""
    return PromptConfig(
        max_context_length=settings.MAX_CONTEXT_LENGTH,
        max_answer_length=settings.MAX_ANSWER_LENGTH,
        temperature=settings.PROMPT_TEMPERATURE,
        include_sources=settings.ENABLE_CITATIONS,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        enable_citations=settings.ENABLE_CITATIONS
    )

@lru_cache()
def get_rag_service() -> RAGService:
    """Get singleton RAG service instance."""
    return RAGService(
        embedding_service=get_embedding_service(),
        vector_service=get_vector_service(),
        document_service=get_document_service(),
        prompt_config=get_prompt_config(),
        language_model_provider=settings.LANGUAGE_MODEL_PROVIDER
    ) 