from fastapi import APIRouter, Depends
from typing import Dict, Any
from ..services.rag_service import RAGService
from ..core.dependencies import get_rag_service

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "RAG Document Assistant",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Detailed health check with system statistics.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        Detailed health and statistics information
    """
    try:
        stats = rag_service.get_system_stats()
        
        # Calculate total chunks from documents
        documents = rag_service.document_service.list_documents()
        total_chunks = sum(doc.total_chunks for doc in documents)
        
        # Get embedding model info
        embedding_model = rag_service.embedding_service.model_name
        
        # Calculate uptime (placeholder - you might want to track this properly)
        import time
        uptime = int(time.time()) % 86400  # Placeholder uptime in seconds
        
        return {
            "status": "healthy",
            "service": "RAG Document Assistant",
            "version": "1.0.0",
            "system_stats": {
                "total_documents": stats.get('total_documents', 0),
                "total_chunks": total_chunks,
                "total_vectors": stats.get('total_vectors', 0),
                "embedding_model": embedding_model,
                "vector_dimension": stats.get('embedding_dimension', 0),
                "index_type": "FAISS",
                "uptime": uptime
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        } 