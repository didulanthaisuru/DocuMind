#!/usr/bin/env python3
"""
Simple startup script for the RAG application.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the RAG application."""
    print("🚀 Starting RAG Document Assistant...")
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Error: 'app' directory not found. Make sure you're in the project root directory.")
        sys.exit(1)
    
    # Check if .env exists, if not copy from config.env
    if not Path(".env").exists() and Path("config.env").exists():
        print("📝 Creating .env file from config.env...")
        import shutil
        shutil.copy("config.env", ".env")
        print("✅ .env file created successfully")
    
    # Create necessary directories
    directories = ["data/uploads", "data/embeddings", "data/metadata", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Data directories created/verified")
    
    # Start the application
    print("🌐 Starting FastAPI server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🏥 Health check available at: http://localhost:8000/health")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 