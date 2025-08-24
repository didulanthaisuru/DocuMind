#!/usr/bin/env python3
"""
Test configuration loading.
"""

import os
from app.config import settings

print("Configuration Test:")
print(f"Settings GEMINI_API_KEY: {settings.GEMINI_API_KEY}")
print(f"Environment GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
print(f"Settings LANGUAGE_MODEL_PROVIDER: {settings.LANGUAGE_MODEL_PROVIDER}")
print(f"Environment LANGUAGE_MODEL_PROVIDER: {os.getenv('LANGUAGE_MODEL_PROVIDER')}")
