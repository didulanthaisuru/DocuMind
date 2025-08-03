#!/usr/bin/env python3
"""
Simple test script to verify the RAG application is working.
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def test_health_check():
    """Test basic health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False

def test_detailed_health():
    """Test detailed health check with system stats."""
    print("\nTesting detailed health check...")
    try:
        response = requests.get(f"{BASE_URL}/health/detailed")
        if response.status_code == 200:
            print("✓ Detailed health check passed")
            data = response.json()
            print(f"  Status: {data.get('status')}")
            if 'statistics' in data:
                stats = data['statistics']
                print(f"  Total documents: {stats.get('total_documents', 0)}")
                print(f"  Total vectors: {stats.get('total_vectors', 0)}")
            return True
        else:
            print(f"✗ Detailed health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Detailed health check failed: {str(e)}")
        return False

def test_list_documents():
    """Test listing documents endpoint."""
    print("\nTesting list documents...")
    try:
        response = requests.get(f"{API_BASE}/documents")
        if response.status_code == 200:
            print("✓ List documents passed")
            data = response.json()
            print(f"  Total documents: {data.get('total_count', 0)}")
            return True
        else:
            print(f"✗ List documents failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ List documents failed: {str(e)}")
        return False

def test_query_without_documents():
    """Test query endpoint without any documents."""
    print("\nTesting query without documents...")
    try:
        query_data = {
            "question": "What is this document about?",
            "top_k": 5,
            "include_sources": True
        }
        
        response = requests.post(f"{API_BASE}/query", json=query_data)
        if response.status_code == 200:
            print("✓ Query endpoint passed")
            data = response.json()
            print(f"  Answer: {data.get('answer', '')[:100]}...")
            print(f"  Confidence: {data.get('confidence', 0):.2f}")
            print(f"  Processing time: {data.get('processing_time', 0):.2f}s")
            return True
        else:
            print(f"✗ Query failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Query failed: {str(e)}")
        return False

def create_test_document():
    """Create a simple test document."""
    print("\nCreating test document...")
    test_content = """
    This is a test document for the RAG application.
    
    The RAG (Retrieval-Augmented Generation) system allows users to:
    1. Upload documents in various formats (PDF, DOCX, TXT)
    2. Process and chunk the documents into smaller pieces
    3. Generate embeddings for each chunk using sentence transformers
    4. Store embeddings in a FAISS vector database
    5. Query the documents using semantic search
    6. Generate answers based on the most relevant document chunks
    
    This system is particularly useful for:
    - Research paper analysis
    - Document Q&A systems
    - Knowledge base search
    - Content recommendation
    
    The system uses the sentence-transformers library with the all-mpnet-base-v2 model
    for generating high-quality embeddings that capture semantic meaning.
    """
    
    # Create test file
    test_file_path = Path("test_document.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    return test_file_path

def test_document_upload():
    """Test document upload functionality."""
    print("\nTesting document upload...")
    try:
        # Create test document
        test_file_path = create_test_document()
        
        # Upload document
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            response = requests.post(f"{API_BASE}/documents/upload", files=files)
        
        # Clean up test file
        test_file_path.unlink()
        
        if response.status_code == 200:
            print("✓ Document upload passed")
            data = response.json()
            print(f"  Document ID: {data.get('document_id')}")
            print(f"  Filename: {data.get('filename')}")
            print(f"  Status: {data.get('status')}")
            print(f"  Message: {data.get('message')}")
            return data.get('document_id')
        else:
            print(f"✗ Document upload failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Document upload failed: {str(e)}")
        return None

def test_query_with_document(document_id):
    """Test query with uploaded document."""
    print(f"\nTesting query with document {document_id}...")
    try:
        # Wait a bit for processing
        print("  Waiting for document processing...")
        time.sleep(3)
        
        query_data = {
            "question": "What is RAG and how does it work?",
            "document_ids": [document_id],
            "top_k": 3,
            "include_sources": True
        }
        
        response = requests.post(f"{API_BASE}/query", json=query_data)
        if response.status_code == 200:
            print("✓ Query with document passed")
            data = response.json()
            print(f"  Answer: {data.get('answer', '')[:200]}...")
            print(f"  Confidence: {data.get('confidence', 0):.2f}")
            print(f"  Processing time: {data.get('processing_time', 0):.2f}s")
            print(f"  Sources found: {len(data.get('sources', []))}")
            
            # Show source details
            for i, source in enumerate(data.get('sources', [])[:2]):
                print(f"    Source {i+1}: {source.get('filename')} (score: {source.get('similarity_score', 0):.2f})")
            
            return True
        else:
            print(f"✗ Query with document failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Query with document failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("RAG Application Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    if not test_health_check():
        print("\n❌ Basic health check failed. Make sure the application is running.")
        return
    
    if not test_detailed_health():
        print("\n❌ Detailed health check failed.")
        return
    
    if not test_list_documents():
        print("\n❌ List documents failed.")
        return
    
    if not test_query_without_documents():
        print("\n❌ Query without documents failed.")
        return
    
    # Test document upload and query
    document_id = test_document_upload()
    if document_id:
        if test_query_with_document(document_id):
            print("\n✅ All tests passed! The RAG application is working correctly.")
        else:
            print("\n❌ Query with document failed.")
    else:
        print("\n❌ Document upload failed.")

if __name__ == "__main__":
    main() 