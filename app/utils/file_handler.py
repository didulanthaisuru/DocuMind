import os
import uuid
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List
import PyPDF2
import docx
from fastapi import UploadFile, HTTPException
from .logger import setup_logger

class FileHandler:
    """
    Handles file operations including upload, validation, and text extraction.
    """
    
    def __init__(self, upload_dir: Path, allowed_extensions: List[str], max_size: int):
        """
        Initialize file handler with configuration.
        
        Args:
            upload_dir: Directory to store uploaded files
            allowed_extensions: List of allowed file extensions
            max_size: Maximum file size in bytes
        """
        self.upload_dir = Path(upload_dir)
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self.max_size = max_size
        self.logger = setup_logger(self.__class__.__name__)
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_file(self, file: UploadFile) -> Tuple[bool, str]:
        """
        Validate uploaded file against size and type constraints.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if hasattr(file, 'size') and file.size > self.max_size:
            return False, f"File size ({file.size} bytes) exceeds maximum allowed ({self.max_size} bytes)"
        
        # Check file extension
        if not file.filename:
            return False, "Filename is required"
        
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in self.allowed_extensions:
            return False, f"File type '{file_ext}' not allowed. Supported types: {', '.join(self.allowed_extensions)}"
        
        return True, ""
    
    def save_file(self, file: UploadFile) -> Tuple[str, Path]:
        """
        Save uploaded file to disk with unique identifier.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Tuple of (document_id, file_path)
        """
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create filename with document ID
        file_ext = Path(file.filename).suffix.lower()
        filename = f"{document_id}{file_ext}"
        file_path = self.upload_dir / filename
        
        try:
            # Write file content
            with open(file_path, "wb") as buffer:
                content = file.file.read()
                buffer.write(content)
            
            self.logger.info(f"File saved: {filename} ({len(content)} bytes)")
            return document_id, file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save file {filename}: {str(e)}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    def extract_text(self, file_path: Path) -> Tuple[str, Optional[int]]:
        """
        Extract text content from supported file types.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_ext == '.docx':
                return self._extract_docx_text(file_path)
            elif file_ext == '.txt':
                return self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
                
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: Path) -> Tuple[str, int]:
        """Extract text from PDF file."""
        text_content = []
        page_count = 0
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        text_content.append(f"[Page {page_num + 1}]\n{text}")
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
        
        return '\n\n'.join(text_content), page_count
    
    def _extract_docx_text(self, file_path: Path) -> Tuple[str, None]:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        return '\n\n'.join(text_content), None
    
    def _extract_txt_text(self, file_path: Path) -> Tuple[str, None]:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return content, None
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete file from disk.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False 