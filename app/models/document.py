from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    """Document processing status enumeration"""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentChunk(BaseModel):
    """Individual document chunk model"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    page: Optional[int] = Field(None, description="Page number (for PDFs)")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    embedding_index: Optional[int] = Field(None, description="Index in FAISS vector store")

class DocumentMetadata(BaseModel):
    """Complete document metadata model"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Storage file path")
    file_type: str = Field(..., description="File extension")
    file_size: int = Field(..., description="File size in bytes")
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADING)
    pages: Optional[int] = Field(None, description="Number of pages")
    total_chunks: int = Field(default=0, description="Total number of chunks")
    chunks: List[DocumentChunk] = Field(default_factory=list)
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str
    filename: str
    status: DocumentStatus
    message: str

class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    documents: List[DocumentMetadata]
    total_count: int 