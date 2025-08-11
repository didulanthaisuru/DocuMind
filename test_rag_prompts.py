#!/usr/bin/env python3
"""
Test script to demonstrate the refined RAG prompt system.
This script shows how the new prompt service improves answer quality.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.services.prompt_service import PromptService, PromptConfig, PromptType
from app.services.language_model_service import LanguageModelService
from app.utils.logger import setup_logger


def test_prompt_generation():
    """Test different prompt types and their generation."""
    print("=" * 60)
    print("TESTING REFINED RAG PROMPT SYSTEM")
    print("=" * 60)
    
    # Initialize services
    prompt_config = PromptConfig(
        max_context_length=4000,
        max_answer_length=1000,
        temperature=0.7,
        include_sources=True,
        confidence_threshold=0.5,
        enable_citations=True
    )
    
    prompt_service = PromptService(prompt_config)
    language_model = LanguageModelService(provider="mock")
    
    # Sample context (simulating retrieved document chunks)
    sample_context = [
        {
            'text': 'The RAG (Retrieval-Augmented Generation) system is a powerful approach that combines information retrieval with text generation. It allows AI systems to access external knowledge sources and generate more accurate and up-to-date responses.',
            'filename': 'rag_documentation.pdf',
            'page': 1,
            'similarity_score': 0.95
        },
        {
            'text': 'Key benefits of RAG include improved accuracy, reduced hallucination, and the ability to cite sources. The system works by first retrieving relevant documents, then using them as context for answer generation.',
            'filename': 'rag_documentation.pdf',
            'page': 2,
            'similarity_score': 0.88
        },
        {
            'text': 'Traditional language models often struggle with factual accuracy and may generate incorrect information. RAG addresses this by grounding responses in retrieved documents.',
            'filename': 'ai_comparison.pdf',
            'page': 5,
            'similarity_score': 0.82
        }
    ]
    
    # Test different question types
    test_questions = [
        {
            'question': 'What is RAG and how does it work?',
            'type': 'general_qa',
            'description': 'General Q&A about RAG'
        },
        {
            'question': 'Analyze the benefits and limitations of RAG systems.',
            'type': 'analytical',
            'description': 'Analytical question about RAG'
        },
        {
            'question': 'Summarize the key points about RAG technology.',
            'type': 'summarization',
            'description': 'Summarization request'
        },
        {
            'question': 'Compare RAG with traditional language models.',
            'type': 'comparison',
            'description': 'Comparison question'
        },
        {
            'question': 'What are the main advantages of using RAG?',
            'type': 'extractive',
            'description': 'Extractive question'
        }
    ]
    
    for test in test_questions:
        print(f"\n{'='*50}")
        print(f"TEST: {test['description']}")
        print(f"Question: {test['question']}")
        print(f"{'='*50}")
        
        # Determine prompt type
        detected_type = prompt_service.determine_prompt_type(test['question'])
        print(f"Detected prompt type: {detected_type.value}")
        
        # Generate prompt
        prompt = prompt_service.generate_prompt(
            question=test['question'],
            context_results=sample_context,
            prompt_type=detected_type
        )
        
        print(f"\nGenerated Prompt:")
        print("-" * 30)
        print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
        
        # Generate answer
        answer = language_model.generate(prompt)
        
        print(f"\nGenerated Answer:")
        print("-" * 30)
        print(answer)
        
        # Post-process answer
        processed_answer, confidence = prompt_service.post_process_answer(
            answer, sample_context
        )
        
        print(f"\nPost-processed Answer (confidence: {confidence:.2f}):")
        print("-" * 30)
        print(processed_answer)


def test_prompt_templates():
    """Test all available prompt templates."""
    print("\n" + "=" * 60)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 60)
    
    prompt_service = PromptService()
    
    # Sample context
    context = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
            'filename': 'ml_basics.txt',
            'page': 1,
            'similarity_score': 0.9
        }
    ]
    
    question = "What is machine learning?"
    
    for prompt_type in PromptType:
        print(f"\n{'-'*40}")
        print(f"Template: {prompt_type.value.upper()}")
        print(f"{'-'*40}")
        
        prompt = prompt_service.generate_prompt(
            question=question,
            context_results=context,
            prompt_type=prompt_type
        )
        
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)


def test_context_formatting():
    """Test context formatting and deduplication."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT FORMATTING")
    print("=" * 60)
    
    prompt_service = PromptService()
    
    # Context with duplicates
    context_with_duplicates = [
        {
            'text': 'RAG systems improve answer accuracy by using retrieved documents.',
            'filename': 'doc1.pdf',
            'page': 1,
            'similarity_score': 0.95
        },
        {
            'text': 'RAG systems improve answer accuracy by using retrieved documents.',
            'filename': 'doc2.pdf',
            'page': 3,
            'similarity_score': 0.92
        },
        {
            'text': 'The key advantage of RAG is its ability to cite sources.',
            'filename': 'doc1.pdf',
            'page': 2,
            'similarity_score': 0.88
        }
    ]
    
    print("Original context chunks:")
    for i, chunk in enumerate(context_with_duplicates, 1):
        print(f"{i}. {chunk['text']} (score: {chunk['similarity_score']:.2f})")
    
    # Format context
    formatted = prompt_service._format_context(context_with_duplicates)
    
    print(f"\nFormatted context:")
    print("-" * 30)
    print(formatted)


if __name__ == "__main__":
    try:
        # Test prompt generation with different question types
        test_prompt_generation()
        
        # Test all prompt templates
        test_prompt_templates()
        
        # Test context formatting
        test_context_formatting()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
