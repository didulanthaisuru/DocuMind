import faiss
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle
from ..core.exceptions import VectorStoreError
from ..utils.logger import setup_logger

class VectorStoreService:
    """
    Service for managing FAISS vector store operations.
    Handles indexing, searching, and persistence of embeddings.
    """
    
    def __init__(self, index_path: Path, metadata_path: Path, dimension: int = 768):
        """
        Initialize vector store service.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            dimension: Embedding dimension
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.index = None
        self.metadata = {}
        self.logger = setup_logger(self.__class__.__name__)
        
        # Initialize or load existing index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize or load existing FAISS index."""
        try:
            if self.index_path.exists():
                self.logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(str(self.index_path))
                self._load_metadata()
            else:
                self.logger.info("Creating new FAISS index")
                # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
                self._save_index()
                self._save_metadata()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise VectorStoreError(f"Index initialization failed: {str(e)}")
    
    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self.metadata)} vectors")
            else:
                self.metadata = {}
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {str(e)}")
            self.metadata = {}
    
    def _save_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            # Ensure directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.logger.debug("FAISS index saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {str(e)}")
            raise VectorStoreError(f"Index save failed: {str(e)}")
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        try:
            # Ensure directory exists
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            self.logger.debug("Metadata saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise VectorStoreError(f"Metadata save failed: {str(e)}")
    
    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            embeddings: Numpy array of embeddings to add
            metadata_list: List of metadata dictionaries for each embedding
            
        Returns:
            List of vector IDs assigned to the added vectors
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        try:
            # Get starting index for new vectors
            start_id = self.index.ntotal
            
            # Add vectors to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata with vector IDs
            vector_ids = []
            for i, metadata in enumerate(metadata_list):
                vector_id = start_id + i
                self.metadata[str(vector_id)] = metadata
                vector_ids.append(vector_id)
            
            # Save to disk
            self._save_index()
            self._save_metadata()
            
            self.logger.info(f"Added {len(embeddings)} vectors to index (total: {self.index.ntotal})")
            return vector_ids
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors: {str(e)}")
            raise VectorStoreError(f"Vector addition failed: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               document_ids: Optional[List[str]] = None) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            document_ids: Optional list of document IDs to filter by
            
        Returns:
            Tuple of (similarity_scores, metadata_list)
        """
        if self.index.ntotal == 0:
            self.logger.warning("Attempting to search empty index")
            return [], []
        
        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Extract results
            results_metadata = []
            results_scores = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Get metadata for this vector
                vector_metadata = self.metadata.get(str(idx), {})
                
                # Filter by document IDs if specified
                if document_ids and vector_metadata.get('document_id') not in document_ids:
                    continue
                
                # Add similarity score to metadata
                vector_metadata['similarity_score'] = float(score)
                vector_metadata['vector_id'] = int(idx)
                
                results_metadata.append(vector_metadata)
                results_scores.append(float(score))
            
            self.logger.debug(f"Found {len(results_metadata)} similar vectors")
            return results_scores, results_metadata
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Vector search failed: {str(e)}")
    
    def remove_document_vectors(self, document_id: str) -> int:
        """
        Remove all vectors associated with a document.
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        
        Args:
            document_id: Document ID to remove
            
        Returns:
            Number of vectors removed
        """
        try:
            # Find vectors to keep
            vectors_to_keep = []
            metadata_to_keep = {}
            removed_count = 0
            
            for vector_id, metadata in self.metadata.items():
                if metadata.get('document_id') == document_id:
                    removed_count += 1
                else:
                    vectors_to_keep.append(int(vector_id))
                    metadata_to_keep[str(len(vectors_to_keep) - 1)] = metadata
            
            if removed_count == 0:
                self.logger.info(f"No vectors found for document {document_id}")
                return 0
            
            # Rebuild index with remaining vectors
            if vectors_to_keep:
                # Get embeddings for vectors to keep
                embeddings_to_keep = []
                for old_vector_id in vectors_to_keep:
                    # This is inefficient but necessary due to FAISS limitations
                    # In production, consider using a different vector database
                    vector = self.index.reconstruct(old_vector_id)
                    embeddings_to_keep.append(vector)
                
                # Create new index
                new_index = faiss.IndexFlatIP(self.dimension)
                if embeddings_to_keep:
                    new_index.add(np.array(embeddings_to_keep).astype(np.float32))
                
                self.index = new_index
                self.metadata = metadata_to_keep
            else:
                # No vectors left, create empty index
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
            
            # Save updated index and metadata
            self._save_index()
            self._save_metadata()
            
            self.logger.info(f"Removed {removed_count} vectors for document {document_id}")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove document vectors: {str(e)}")
            raise VectorStoreError(f"Vector removal failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_size_mb': self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
            'metadata_count': len(self.metadata)
        } 