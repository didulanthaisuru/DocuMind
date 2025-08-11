#!/usr/bin/env python3
"""
Simple test to verify the source attribution fix works in the RAG flow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enrichment_logic():
    """Test the enrichment logic without requiring full service initialization."""
    print("=" * 60)
    print("TESTING ENRICHMENT LOGIC")
    print("=" * 60)
    
    # Simulate the enrichment logic from RAG service
    def enrich_context_results(results_metadata, document_service):
        """Simulate the enrichment function."""
        enriched_results = []
        
        for result in results_metadata:
            # Get full document information
            document = document_service.get_document(result.get('document_id'))
            
            # Create enriched result
            enriched_result = result.copy()
            if document:
                enriched_result['filename'] = document.filename
                enriched_result['document_title'] = getattr(document, 'title', document.filename)
            else:
                enriched_result['filename'] = f"Document {result.get('document_id', 'Unknown')}"
                enriched_result['document_title'] = enriched_result['filename']
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    # Mock document service
    class MockDocumentService:
        def __init__(self):
            self.documents = {
                'doc_001': MockDocument('doc_001', 'research_paper.pdf'),
                'doc_002': MockDocument('doc_002', 'technical_guide.pdf'),
                'doc_003': MockDocument('doc_003', 'user_manual.pdf')
            }
        
        def get_document(self, doc_id):
            return self.documents.get(doc_id)
    
    class MockDocument:
        def __init__(self, doc_id, filename):
            self.document_id = doc_id
            self.filename = filename
    
    # Test data
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
        },
        {
            'text': 'Document without known ID.',
            'document_id': 'doc_999',
            'similarity_score': 0.82,
            'page': 3
        }
    ]
    
    # Test enrichment
    document_service = MockDocumentService()
    enriched_results = enrich_context_results(test_metadata, document_service)
    
    print("Enriched context results:")
    for i, result in enumerate(enriched_results, 1):
        print(f"  {i}. Filename: {result.get('filename', 'Unknown')}")
        print(f"     Document ID: {result.get('document_id', 'Unknown')}")
        print(f"     Similarity: {result.get('similarity_score', 0):.2f}")
        print(f"     Page: {result.get('page', 'N/A')}")
        print()
    
    # Test source info extraction
    def get_source_info(chunk):
        """Simulate the source info extraction."""
        filename = (chunk.get('filename') or 
                   chunk.get('document_title') or 
                   chunk.get('title') or 
                   f"Document {chunk.get('document_id', 'Unknown')}")
        
        page = chunk.get('page')
        score = chunk.get('similarity_score', 0)
        
        source_parts = [filename]
        if page is not None:
            source_parts.append(f"page {page}")
        if score > 0:
            source_parts.append(f"relevance: {score:.2f}")
        
        return " | ".join(source_parts)
    
    print("Source information extraction:")
    for i, result in enumerate(enriched_results, 1):
        source_info = get_source_info(result)
        print(f"  {i}. {source_info}")
    
    print("\n" + "=" * 60)
    print("ENRICHMENT LOGIC TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_enrichment_logic()
