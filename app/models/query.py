from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """Request model for document queries"""
    question: str = Field(..., min_length=1, description="User question")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to retrieve")
    include_sources: bool = Field(True, description="Include source information in response")

class SourceInfo(BaseModel):
    """Source information for query results"""
    document_id: str
    filename: str
    page: Optional[int]
    chunk_text: str
    similarity_score: float

class QueryResponse(BaseModel):
    """Response model for document queries"""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[SourceInfo] = Field(default_factory=list)
    processing_time: float = Field(..., description="Query processing time in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used") 