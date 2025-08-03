from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from ..models.document import DocumentUploadResponse, DocumentListResponse, DocumentMetadata
from ..services.rag_service import RAGService
from ..core.dependencies import get_rag_service
from ..core.exceptions import DocumentProcessingError

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload and process a document.
    
    Args:
        background_tasks: FastAPI background tasks for async processing
        file: Uploaded document file
        rag_service: RAG service instance
        
    Returns:
        Document upload response with basic information
        
    Raises:
        HTTPException: If upload or initial processing fails
    """
    try:
        # Upload and extract text (synchronous)
        document_metadata = rag_service.document_service.upload_document(file)
        
        # Schedule embedding generation in background
        background_tasks.add_task(
            rag_service.process_document, 
            document_metadata.document_id
        )
        
        return DocumentUploadResponse(
            document_id=document_metadata.document_id,
            filename=document_metadata.filename,
            status=document_metadata.status,
            message=f"Document uploaded successfully. Processing {document_metadata.total_chunks} chunks in background."
        )
        
    except DocumentProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    List all uploaded documents.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        List of all document metadata
    """
    try:
        documents = rag_service.document_service.list_documents()
        
        return DocumentListResponse(
            documents=documents,
            total_count=len(documents)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get detailed information about a specific document.
    
    Args:
        document_id: Document identifier
        rag_service: RAG service instance
        
    Returns:
        Detailed document metadata
        
    Raises:
        HTTPException: If document not found
    """
    try:
        document = rag_service.document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Delete a document and all associated data.
    
    Args:
        document_id: Document identifier
        rag_service: RAG service instance
        
    Returns:
        Deletion confirmation message
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    try:
        success = rag_service.remove_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}") 