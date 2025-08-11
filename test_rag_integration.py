#!/usr/bin/env python3
"""
Test script to verify the complete RAG integration with source attribution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorStoreService
from app.services.document_service import DocumentService
from app.services.prompt_service import PromptConfig
from app.services.language_model_service import LanguageModelService
from app.utils.file_handler import FileHandler
from app.utils.text_processor import TextProcessor
from app.models.query import QueryRequest
from pathlib import Path

def test_rag_integration():
    """Test the complete RAG integration with source attribution."""
    print("=" * 60)
    print("TESTING RAG INTEGRATION WITH SOURCE ATTRIBUTION")
    print("=" * 60)
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_service = VectorStoreService(
        index_path=Path("test_index.faiss"),
        metadata_path=Path("test_metadata.json")
    )
    file_handler = FileHandler()
    text_processor = TextProcessor()
    document_service = DocumentService(
        file_handler=file_handler,
        text_processor=text_processor,
        metadata_file=Path("test_documents.json")
    )
    
    # Create RAG service with prompt configuration
    prompt_config = PromptConfig(
        max_context_length=2000,
        max_answer_length=500,
        temperature=0.7,
        include_sources=True,
        confidence_threshold=0.5,
        enable_citations=True
    )
    
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_service=vector_service,
        document_service=document_service,
        prompt_config=prompt_config,
        language_model_provider="mock"
    )
    
    # Test the enrichment function directly
    print("\nTesting context enrichment:")
    test_metadata = [
        {
            'text': 'RAG systems combine retrieval and generation.',
            'document_id': 'doc_001',
            'similarity_score': 0.95,
            'page': 1
        },
        {
            'text': 'Machine learning enables computers to learn.',
            'document_id': 'doc_002', 
            'similarity_score': 0.88,
            'page': 2
        }
    ]
    
    # Mock document service responses
    class MockDocument:
        def __init__(self, doc_id, filename):
            self.document_id = doc_id
            self.filename = filename
    
    # Override the get_document method for testing
    original_get_document = document_service.get_document
    document_service.get_document = lambda doc_id: MockDocument(doc_id, f"test_{doc_id}.pdf")
    
    try:
        enriched_context = rag_service._enrich_context_results(test_metadata)
        print("Enriched context results:")
        for i, result in enumerate(enriched_context, 1):
            print(f"  {i}. Filename: {result.get('filename', 'Unknown')}")
            print(f"     Document ID: {result.get('document_id', 'Unknown')}")
            print(f"     Similarity: {result.get('similarity_score', 0):.2f}")
            print()
    finally:
        # Restore original method
        document_service.get_document = original_get_document
    
    print("=" * 60)
    print("RAG INTEGRATION TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_rag_integration()
