import time
from typing import List, Dict, Any, Optional
from ..models.query import QueryRequest, QueryResponse, SourceInfo
from ..services.embedding_service import EmbeddingService
from ..services.vector_service import VectorStoreService
from ..services.document_service import DocumentService
from ..core.exceptions import QueryError
from ..utils.logger import setup_logger
from ..models.document import DocumentStatus

class RAGService:
    """
    Main RAG orchestration service that coordinates document processing,
    embedding generation, and query answering.
    """
    
    def __init__(self, embedding_service: EmbeddingService, 
                 vector_service: VectorStoreService, 
                 document_service: DocumentService):
        """
        Initialize RAG service with required components.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_service: Service for vector storage and search
            document_service: Service for document management
        """
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.document_service = document_service
        self.logger = setup_logger(self.__class__.__name__)
    
    def process_document(self, document_id: str) -> bool:
        """
        Process a document by generating embeddings for its chunks.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Processing document for embedding: {document_id}")
            start_time = time.time()
            
            # Get document chunks
            chunks = self.document_service.get_document_chunks(document_id)
            if not chunks:
                raise QueryError(f"No chunks found for document {document_id}")
            
            # Extract text from chunks
            chunk_texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_service.embed_texts(chunk_texts)
            
            # Prepare metadata for vector store
            metadata_list = []
            for chunk in chunks:
                metadata = {
                    'document_id': document_id,
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text,
                    'page': chunk.page,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char
                }
                metadata_list.append(metadata)
            
            # Add to vector store
            vector_ids = self.vector_service.add_vectors(embeddings, metadata_list)
            
            # Update chunk embedding indices
            chunk_embedding_map = {}
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk_embedding_map[chunk.chunk_id] = vector_id
            
            self.document_service.update_chunk_embedding_indices(document_id, chunk_embedding_map)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed document {document_id} "
                           f"({len(chunks)} chunks) in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document_id}: {str(e)}")
            raise QueryError(f"Document processing failed: {str(e)}")
    
    def query_documents(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a query against the document collection.
        
        Args:
            query_request: Query request object
            
        Returns:
            Query response with answer and sources
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {query_request.question[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query_request.question)
            
            # Search for relevant chunks
            scores, results_metadata = self.vector_service.search(
                query_embedding=query_embedding,
                k=query_request.top_k,
                document_ids=query_request.document_ids
            )
            
            if not results_metadata:
                # No relevant results found
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    confidence=0.0,
                    sources=[],
                    processing_time=time.time() - start_time
                )
            
            # Generate answer from retrieved context
            answer, confidence = self._generate_answer(
                query_request.question, 
                results_metadata
            )
            
            # Prepare source information
            sources = []
            if query_request.include_sources:
                for result in results_metadata:
                    document = self.document_service.get_document(result['document_id'])
                    if document:
                        source = SourceInfo(
                            document_id=result['document_id'],
                            filename=document.filename,
                            page=result.get('page'),
                            chunk_text=result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
                            similarity_score=result['similarity_score']
                        )
                        sources.append(source)
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                processing_time=processing_time
            )
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s "
                           f"(confidence: {confidence:.2f})")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            raise QueryError(f"Failed to process query: {str(e)}")
    
    def _generate_answer(self, question: str, context_results: List[Dict[str, Any]]) -> tuple[str, float]:
        """
        Generate an answer based on the question and retrieved context.
        
        This is a simplified implementation. In production, you might want to use
        a language model like GPT, T5, or other text generation models.
        
        Args:
            question: User question
            context_results: Retrieved context chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not context_results:
            return "No relevant information found.", 0.0
        
        # Simple extractive approach - return most relevant chunk(s)
        # In production, implement proper text generation here
        
        # Sort by similarity score
        sorted_results = sorted(context_results, key=lambda x: x['similarity_score'], reverse=True)
        
        # Take top chunks and combine
        top_chunks = sorted_results[:3]  # Use top 3 chunks
        combined_text = "\n\n".join([chunk['text'] for chunk in top_chunks])
        
        # Calculate average confidence
        avg_confidence = sum(chunk['similarity_score'] for chunk in top_chunks) / len(top_chunks)
        
        # Simple answer generation - in production, use a proper language model
        answer = f"Based on the document content:\n\n{combined_text}"
        
        return answer, min(avg_confidence, 1.0)
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from both document service and vector store.
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful
        """
        try:
            # Remove from vector store
            removed_vectors = self.vector_service.remove_document_vectors(document_id)
            
            # Remove from document service
            document_removed = self.document_service.delete_document(document_id)
            
            if document_removed:
                self.logger.info(f"Removed document {document_id} "
                               f"({removed_vectors} vectors)")
                return True
            else:
                self.logger.warning(f"Document {document_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id}: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        vector_stats = self.vector_service.get_stats()
        documents = self.document_service.list_documents()
        
        return {
            'total_documents': len(documents),
            'total_vectors': vector_stats['total_vectors'],
            'embedding_dimension': vector_stats['dimension'],
            'index_size_mb': vector_stats['index_size_mb'],
            'documents_by_status': {
                status.value: len([d for d in documents if d.status == status])
                for status in DocumentStatus
            }
        } 