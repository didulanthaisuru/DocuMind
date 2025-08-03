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
        return {
            "status": "healthy",
            "service": "RAG Document Assistant",
            "version": "1.0.0",
            "statistics": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        } 