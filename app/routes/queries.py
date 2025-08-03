from fastapi import APIRouter, Depends, HTTPException
from ..models.query import QueryRequest, QueryResponse
from ..services.rag_service import RAGService
from ..core.dependencies import get_rag_service
from ..core.exceptions import QueryError

router = APIRouter(prefix="/query", tags=["Queries"])

@router.post("", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Query documents to get answers based on content.
    
    Args:
        query_request: Query request with question and parameters
        rag_service: RAG service instance
        
    Returns:
        Query response with answer and source information
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        response = rag_service.query_documents(query_request)
        return response
        
    except QueryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/stats")
async def get_query_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get system statistics for query capabilities.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        System statistics
    """
    try:
        stats = rag_service.get_system_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}") 