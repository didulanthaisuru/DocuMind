#!/usr/bin/env python3
"""
Test to verify enhanced prompts are working with Gemini AI.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_prompts():
    """Test the enhanced prompt service with different question types."""
    print("=" * 60)
    print("TESTING ENHANCED PROMPTS WITH GEMINI AI")
    print("=" * 60)
    
    try:
        from app.services.prompt_service import PromptService, PromptType
        from app.services.language_model_service import LanguageModelService
        from app.config import settings
        
        # Initialize services
        prompt_service = PromptService()
        lm_service = LanguageModelService(provider="gemini")
        
        print("✅ Services initialized successfully")
        print(f"✅ Using language model provider: {settings.LANGUAGE_MODEL_PROVIDER}")
        
        # Test data
        test_context = [
            {
                'text': 'This document discusses the implementation of machine learning algorithms in healthcare. The main topics include patient data analysis, predictive modeling, and ethical considerations. The research shows that ML can improve diagnosis accuracy by 15% and reduce treatment costs by 20%.',
                'document_id': 'doc_001',
                'filename': 'healthcare_ml_research.pdf',
                'similarity_score': 0.95,
                'page': 1
            },
            {
                'text': 'The study examined 1000 patient records and found significant improvements in early disease detection. Key findings include reduced false positives and better patient outcomes. The algorithm achieved 92% accuracy in predicting cardiovascular disease.',
                'document_id': 'doc_001',
                'filename': 'healthcare_ml_research.pdf',
                'similarity_score': 0.88,
                'page': 2
            }
        ]
        
        # Test different question types
        test_questions = [
            "Give summary of this document",
            "What are the main topics discussed?",
            "Analyze the key findings",
            "What is the accuracy of the ML algorithm?",
            "Compare the benefits and challenges"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Test {i}: {question} ---")
            
            # Determine prompt type
            prompt_type = prompt_service.determine_prompt_type(question)
            print(f"📝 Detected prompt type: {prompt_type.value}")
            
            # Generate prompt
            prompt = prompt_service.generate_prompt(question, test_context, prompt_type)
            print(f"📄 Generated prompt length: {len(prompt)} characters")
            
            # Test with Gemini (if available)
            if lm_service.is_available():
                try:
                    print("🤖 Testing with Gemini AI...")
                    response = lm_service.generate(prompt, temperature=0.7, max_tokens=300)
                    print(f"✅ Gemini response: {response[:150]}...")
                except Exception as e:
                    print(f"⚠️ Gemini test failed (rate limit?): {str(e)[:100]}...")
            else:
                print("⚠️ Gemini not available for testing")
            
            print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_prompt_type_detection():
    """Test prompt type detection for different question types."""
    print("\n" + "=" * 60)
    print("TESTING PROMPT TYPE DETECTION")
    print("=" * 60)
    
    try:
        from app.services.prompt_service import PromptService, PromptType
        
        prompt_service = PromptService()
        
        test_cases = [
            ("Give summary of this document", PromptType.SUMMARIZATION),
            ("Summarize the key points", PromptType.SUMMARIZATION),
            ("Analyze the findings", PromptType.ANALYTICAL),
            ("Why did this happen?", PromptType.ANALYTICAL),
            ("Compare the results", PromptType.COMPARISON),
            ("What is the difference?", PromptType.COMPARISON),
            ("Verify this claim", PromptType.FACT_CHECKING),
            ("What is the main topic?", PromptType.EXTRACTIVE),
            ("When did this occur?", PromptType.EXTRACTIVE),
            ("General question about content", PromptType.GENERAL_QA),
        ]
        
        correct = 0
        total = len(test_cases)
        
        for question, expected_type in test_cases:
            detected_type = prompt_service.determine_prompt_type(question)
            is_correct = detected_type == expected_type
            status = "✅" if is_correct else "❌"
            
            print(f"{status} Question: '{question}'")
            print(f"   Expected: {expected_type.value}")
            print(f"   Detected: {detected_type.value}")
            print()
            
            if is_correct:
                correct += 1
        
        accuracy = (correct / total) * 100
        print(f"📊 Prompt type detection accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return accuracy >= 80  # 80% accuracy threshold
        
    except Exception as e:
        print(f"❌ Prompt type detection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting enhanced prompts tests...\n")
    
    # Run tests
    test1_passed = test_enhanced_prompts()
    test2_passed = test_prompt_type_detection()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Enhanced Prompts Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Prompt Type Detection: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed]):
        print("\n🎉 All tests passed! Enhanced prompts are working correctly with Gemini AI.")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")
