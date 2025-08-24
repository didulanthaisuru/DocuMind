#!/usr/bin/env python3
"""
Test to verify Gemini integration is working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemini_service():
    """Test the Gemini service directly."""
    print("=" * 60)
    print("TESTING GEMINI INTEGRATION")
    print("=" * 60)
    
    try:
        from app.services.language_model_service import GeminiService
        from app.config import settings
        
        # Initialize Gemini service
        gemini_service = GeminiService(api_key=settings.GEMINI_API_KEY)
        
        # Test availability
        if not gemini_service.is_available():
            print("❌ Gemini service is not available")
            return False
        
        print("✅ Gemini service is available")
        
        # Test simple generation
        test_prompt = "What is the capital of France? Please provide a brief answer."
        try:
            response = gemini_service.generate(test_prompt, temperature=0.7, max_tokens=100)
            print(f"✅ Gemini response: {response[:100]}...")
            return True
        except Exception as e:
            print(f"❌ Gemini generation failed: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Please install google-generativeai: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_language_model_service():
    """Test the unified language model service with Gemini."""
    print("\n" + "=" * 60)
    print("TESTING UNIFIED LANGUAGE MODEL SERVICE")
    print("=" * 60)
    
    try:
        from app.services.language_model_service import LanguageModelService
        from app.config import settings
        
        # Initialize with Gemini provider
        lm_service = LanguageModelService(provider="gemini")
        
        # Test availability
        if not lm_service.is_available():
            print("❌ Language model service is not available")
            return False
        
        print("✅ Language model service is available")
        print(f"✅ Provider: {lm_service.get_provider_info()}")
        
        # Test generation
        test_prompt = "Explain what RAG (Retrieval-Augmented Generation) is in one sentence."
        try:
            response = lm_service.generate(test_prompt, temperature=0.7, max_tokens=150)
            print(f"✅ Response: {response}")
            return True
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_rag_with_gemini():
    """Test RAG service with Gemini provider."""
    print("\n" + "=" * 60)
    print("TESTING RAG SERVICE WITH GEMINI")
    print("=" * 60)
    
    try:
        from app.core.dependencies import get_rag_service
        from app.config import settings
        
        # Get RAG service (should use Gemini based on config)
        rag_service = get_rag_service()
        
        print(f"✅ RAG service initialized with provider: {settings.LANGUAGE_MODEL_PROVIDER}")
        
        # Test that the language model is available
        if not rag_service.language_model.is_available():
            print("❌ RAG service language model is not available")
            return False
        
        print("✅ RAG service language model is available")
        
        # Test provider info
        provider_info = rag_service.language_model.get_provider_info()
        print(f"✅ Provider info: {provider_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG service test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Gemini integration tests...\n")
    
    # Run tests
    test1_passed = test_gemini_service()
    test2_passed = test_language_model_service()
    test3_passed = test_rag_with_gemini()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Gemini Service Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Language Model Service Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"RAG Service Test: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and dependencies.")
