#!/usr/bin/env python3
"""
Test script to verify source attribution in the RAG system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.prompt_service import PromptService, PromptConfig
from app.services.language_model_service import LanguageModelService

def test_source_attribution():
    """Test that source attribution works correctly."""
    print("=" * 60)
    print("TESTING SOURCE ATTRIBUTION FIX")
    print("=" * 60)
    
    # Initialize services
    prompt_service = PromptService()
    
    # Test context with different metadata structures
    test_contexts = [
        {
            'text': 'This is a test document about RAG systems.',
            'filename': 'test_document.pdf',
            'page': 1,
            'similarity_score': 0.95,
            'document_id': 'doc_001'
        },
        {
            'text': 'Another test document about machine learning.',
            'document_title': 'ML_Guide.pdf',
            'page': 3,
            'similarity_score': 0.88,
            'document_id': 'doc_002'
        },
        {
            'text': 'A document without filename.',
            'page': 2,
            'similarity_score': 0.82,
            'document_id': 'doc_003'
        }
    ]
    
    print("\nTesting source information extraction:")
    for i, context in enumerate(test_contexts, 1):
        source_info = prompt_service._get_source_info(context)
        print(f"Context {i}: {source_info}")
    
    print("\nTesting context formatting:")
    formatted_context = prompt_service._format_context(test_contexts)
    print("Formatted context:")
    print(formatted_context)
    
    print("\nTesting complete prompt generation:")
    question = "What is RAG?"
    prompt = prompt_service.generate_prompt(question, test_contexts)
    print("Generated prompt:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n" + "=" * 60)
    print("SOURCE ATTRIBUTION TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_source_attribution()
