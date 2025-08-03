import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import time
from ..core.exceptions import EmbeddingError
from ..utils.logger import setup_logger

class EmbeddingService:
    """
    Service for generating embeddings using sentence transformers.
    Handles model loading, caching, and batch processing.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.logger = setup_logger(self.__class__.__name__)
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")
    
    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing multiple texts
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise EmbeddingError("Model not loaded")
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        try:
            self.logger.debug(f"Generating embeddings for {len(texts)} texts")
            start_time = time.time()
            
            # Generate embeddings with no gradient computation
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=len(texts) > 100,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalize for better similarity
                )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Generated {len(embeddings)} embeddings in {processing_time:.2f} seconds")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            raise EmbeddingError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        Convenience method for query processing.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        return self.embed_texts([query])[0] 