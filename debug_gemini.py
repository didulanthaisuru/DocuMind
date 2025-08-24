#!/usr/bin/env python3
"""
Debug script to check Gemini service configuration.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_gemini_config():
    """Debug Gemini configuration."""
    print("=" * 60)
    print("DEBUGGING GEMINI CONFIGURATION")
    print("=" * 60)
    
    # Check environment variables
    print(f"Environment GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
    print(f"Environment LANGUAGE_MODEL_PROVIDER: {os.getenv('LANGUAGE_MODEL_PROVIDER')}")
    
    # Check settings
    try:
        from app.config import settings
        print(f"Settings GEMINI_API_KEY: {settings.GEMINI_API_KEY}")
        print(f"Settings LANGUAGE_MODEL_PROVIDER: {settings.LANGUAGE_MODEL_PROVIDER}")
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    # Test GeminiService directly
    try:
        from app.services.language_model_service import GeminiService
        
        # Test with explicit API key
        api_key = "AIzaSyA_Z8gMmZrpZmI81BtTJoDtPbLv88QZsSA"
        service = GeminiService(api_key=api_key)
        print(f"GeminiService with explicit key - Available: {service.is_available()}")
        
        # Test with environment variable
        service2 = GeminiService()
        print(f"GeminiService with env var - Available: {service2.is_available()}")
        print(f"GeminiService API key: {service2.api_key}")
        
        # Test generation
        if service.is_available():
            try:
                response = service.generate("Hello, this is a test.", temperature=0.3, max_tokens=50)
                print(f"✅ Generation test successful: {response[:100]}...")
            except Exception as e:
                print(f"❌ Generation test failed: {e}")
        
    except Exception as e:
        print(f"Error testing GeminiService: {e}")
    
    # Test LanguageModelService
    try:
        from app.services.language_model_service import LanguageModelService
        
        lm_service = LanguageModelService(provider="gemini")
        print(f"LanguageModelService - Available: {lm_service.is_available()}")
        print(f"LanguageModelService provider info: {lm_service.get_provider_info()}")
        
    except Exception as e:
        print(f"Error testing LanguageModelService: {e}")

if __name__ == "__main__":
    debug_gemini_config()
