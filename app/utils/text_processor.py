import re
from typing import List, Tuple
from pathlib import Path
from .logger import setup_logger

class TextProcessor:
    """
    Utility class for text processing operations including chunking,
    cleaning, and preprocessing for embedding generation.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize text processor with chunking parameters.
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = setup_logger(self.__class__.__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        Split text into overlapping chunks suitable for embedding.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            self.logger.warning("Empty text provided for chunking")
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 chars
                sentence_end = text.rfind('.', start + self.chunk_size - 100, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_data = {
                    'chunk_id': f"chunk_{chunk_id:04d}",
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    **(metadata or {})
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.overlap, end)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        self.logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks 