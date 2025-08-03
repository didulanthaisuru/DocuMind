import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
from ..models.document import DocumentMetadata, DocumentChunk, DocumentStatus
from ..utils.file_handler import FileHandler
from ..utils.text_processor import TextProcessor
from ..core.exceptions import DocumentProcessingError
from ..utils.logger import setup_logger

class DocumentService:
    """
    Service for managing document operations including upload, processing,
    text extraction, and metadata management.
    """
    
    def __init__(self, file_handler: FileHandler, text_processor: TextProcessor, 
                 metadata_file: Path):
        """
        Initialize document service.
        
        Args:
            file_handler: File handling service
            text_processor: Text processing service
            metadata_file: Path to documents metadata file
        """
        self.file_handler = file_handler
        self.text_processor = text_processor
        self.metadata_file = Path(metadata_file)
        self.logger = setup_logger(self.__class__.__name__)
        
        # Ensure metadata file directory exists
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    def upload_document(self, file: UploadFile) -> DocumentMetadata:
        """
        Upload and process a document.
        
        Args:
            file: Uploaded file object
            
        Returns:
            Document metadata with processing results
        """
        start_time = time.time()
        
        # Validate file
        is_valid, error_message = self.file_handler.validate_file(file)
        if not is_valid:
            raise DocumentProcessingError(error_message)
        
        # Save file and get document ID
        document_id, file_path = self.file_handler.save_file(file)
        
        # Create initial metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file.filename,
            file_path=str(file_path),
            file_type=file_path.suffix.lower().lstrip('.'),
            file_size=file_path.stat().st_size,
            status=DocumentStatus.PROCESSING
        )
        
        try:
            # Extract text content
            self.logger.info(f"Processing document: {file.filename}")
            text_content, page_count = self.file_handler.extract_text(file_path)
            
            if not text_content.strip():
                raise DocumentProcessingError("No text content could be extracted from the document")
            
            # Clean text
            cleaned_text = self.text_processor.clean_text(text_content)
            
            # Create chunks
            chunk_metadata = {
                'document_id': document_id,
                'filename': file.filename
            }
            chunks_data = self.text_processor.chunk_text(cleaned_text, chunk_metadata)
            
            # Convert to DocumentChunk objects
            chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk = DocumentChunk(
                    chunk_id=chunk_data['chunk_id'],
                    text=chunk_data['text'],
                    page=chunk_data.get('page'),
                    start_char=chunk_data['start_char'],
                    end_char=chunk_data['end_char'],
                    embedding_index=None  # Will be set when embeddings are created
                )
                chunks.append(chunk)
            
            # Update metadata
            processing_time = time.time() - start_time
            metadata.pages = page_count
            metadata.total_chunks = len(chunks)
            metadata.chunks = chunks
            metadata.processing_time = processing_time
            metadata.status = DocumentStatus.COMPLETED
            
            # Save metadata
            self._save_document_metadata(metadata)
            
            self.logger.info(f"Successfully processed document {file.filename}: "
                           f"{len(chunks)} chunks in {processing_time:.2f}s")
            
            return metadata
            
        except Exception as e:
            # Update metadata with error
            metadata.status = DocumentStatus.FAILED
            metadata.error_message = str(e)
            metadata.processing_time = time.time() - start_time
            
            # Save error metadata
            self._save_document_metadata(metadata)
            
            self.logger.error(f"Failed to process document {file.filename}: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata or None if not found
        """
        documents = self._load_all_documents()
        return documents.get(document_id)
    
    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all documents.
        
        Returns:
            List of all document metadata
        """
        documents = self._load_all_documents()
        return list(documents.values())
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its associated files.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False if document not found
        """
        # Load current documents
        documents = self._load_all_documents()
        
        if document_id not in documents:
            self.logger.warning(f"Document {document_id} not found for deletion")
            return False
        
        document = documents[document_id]
        
        try:
            # Delete physical file
            file_path = Path(document.file_path)
            self.file_handler.delete_file(file_path)
            
            # Remove from metadata
            del documents[document_id]
            self._save_all_documents(documents)
            
            self.logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
            return False
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document chunks
        """
        document = self.get_document(document_id)
        return document.chunks if document else []
    
    def update_chunk_embedding_indices(self, document_id: str, 
                                     chunk_embedding_map: Dict[str, int]) -> bool:
        """
        Update embedding indices for document chunks.
        
        Args:
            document_id: Document identifier
            chunk_embedding_map: Mapping of chunk_id to embedding index
            
        Returns:
            True if successful
        """
        documents = self._load_all_documents()
        
        if document_id not in documents:
            return False
        
        document = documents[document_id]
        
        # Update embedding indices
        for chunk in document.chunks:
            if chunk.chunk_id in chunk_embedding_map:
                chunk.embedding_index = chunk_embedding_map[chunk.chunk_id]
        
        # Save updated metadata
        documents[document_id] = document
        self._save_all_documents(documents)
        
        return True
    
    def _load_all_documents(self) -> Dict[str, DocumentMetadata]:
        """Load all document metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict data back to DocumentMetadata objects
                    documents = {}
                    for doc_id, doc_data in data.items():
                        documents[doc_id] = DocumentMetadata(**doc_data)
                    return documents
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load documents metadata: {str(e)}")
            return {}
    
    def _save_all_documents(self, documents: Dict[str, DocumentMetadata]) -> None:
        """Save all document metadata to file."""
        try:
            # Convert DocumentMetadata objects to dict for JSON serialization
            data = {}
            for doc_id, document in documents.items():
                data[doc_id] = document.dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save documents metadata: {str(e)}")
            raise
    
    def _save_document_metadata(self, document: DocumentMetadata) -> None:
        """Save single document metadata."""
        documents = self._load_all_documents()
        documents[document.document_id] = document
        self._save_all_documents(documents) 