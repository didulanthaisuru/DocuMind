class RAGException(Exception):
    """Base exception for RAG application"""
    pass

class DocumentProcessingError(RAGException):
    """Raised when document processing fails"""
    pass

class EmbeddingError(RAGException):
    """Raised when embedding generation fails"""
    pass

class VectorStoreError(RAGException):
    """Raised when vector store operations fail"""
    pass

class QueryError(RAGException):
    """Raised when query processing fails"""
    pass 