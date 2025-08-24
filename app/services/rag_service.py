import time
from typing import List, Dict, Any, Optional
from ..models.query import QueryRequest, QueryResponse, SourceInfo
from ..services.embedding_service import EmbeddingService
from ..services.vector_service import VectorStoreService
from ..services.document_service import DocumentService
from ..services.prompt_service import PromptService, PromptConfig, PromptType
from ..services.language_model_service import LanguageModelService
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
                 document_service: DocumentService,
                 prompt_config: Optional[PromptConfig] = None,
                 language_model_provider: str = "mock"):
        """
        Initialize RAG service with required components.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_service: Service for vector storage and search
            document_service: Service for document management
            prompt_config: Configuration for prompt generation
            language_model_provider: Language model provider to use
        """
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.document_service = document_service
        self.prompt_service = PromptService(prompt_config)
        self.language_model = LanguageModelService(provider=language_model_provider)
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
            
            # Search for relevant chunks (increase k for better context)
            # For abstract/summary requests, get more context
            if any(keyword in query_request.question.lower() for keyword in ['abstract', 'summary', 'summarize', 'overview']):
                search_k = max(query_request.top_k * 4, 15)  # Get even more chunks for comprehensive summaries
            else:
                search_k = max(query_request.top_k * 2, 10)  # Get more chunks for better context
            scores, results_metadata = self.vector_service.search(
                query_embedding=query_embedding,
                k=search_k,
                document_ids=query_request.document_ids
            )
            
            if not results_metadata:
                # No relevant results found
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    confidence=0.0,
                    sources=[],
                    processing_time=time.time() - start_time,
                    total_sources_found=0
                )
            
            # Enrich context results with document information
            enriched_context = self._enrich_context_results(results_metadata)
            
            # Generate answer from retrieved context
            answer, confidence = self._generate_answer(
                query_request.question, 
                enriched_context
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
                            chunk_id=result.get('chunk_id', 'unknown'),
                            text=result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
                            similarity_score=result['similarity_score'],
                            start_index=result.get('start_char', 0),
                            end_index=result.get('end_char', len(result['text']))
                        )
                        sources.append(source)
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                confidence=confidence,
                sources=sources,
                processing_time=processing_time,
                total_sources_found=len(sources)
            )
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s "
                           f"(confidence: {confidence:.2f})")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            raise QueryError(f"Failed to process query: {str(e)}")
    
    def _generate_answer(self, question: str, context_results: List[Dict[str, Any]]) -> tuple[str, float]:
        """
        Generate an answer based on the question and retrieved context using sophisticated prompt engineering.
        
        Args:
            question: User question
            context_results: Retrieved context chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not context_results:
            return "I couldn't find any relevant information to answer your question.", 0.0
        
        try:
            # Determine the best prompt type based on the question
            prompt_type = self.prompt_service.determine_prompt_type(question)
            
            # Generate the prompt
            prompt = self.prompt_service.generate_prompt(
                question=question,
                context_results=context_results,
                prompt_type=prompt_type
            )
            
            # Generate answer using the language model
            # Note: This is where you would integrate with your chosen LLM
            # For now, we'll use a simple approach, but you can replace this with:
            # - OpenAI GPT models
            # - Hugging Face transformers
            # - Local models like Llama, Mistral, etc.
            
            answer = self._call_language_model(prompt, question)
            
            # Post-process the answer
            processed_answer, confidence = self.prompt_service.post_process_answer(
                answer, context_results
            )
            
            self.logger.info(f"Generated answer using {prompt_type.value} prompt type")
            return processed_answer, confidence
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {str(e)}")
            # Fallback to simple approach
            return self._generate_fallback_answer(question, context_results)
    
    def _call_language_model(self, prompt: str, question: str) -> str:
        """
        Call the language model to generate an answer.
        
        Args:
            prompt: Formatted prompt for the language model
            question: Original user question
            
        Returns:
            Generated answer
        """
        try:
            # Use the language model service to generate the answer
            answer = self.language_model.generate(
                prompt,
                temperature=self.prompt_service.config.temperature,
                max_tokens=self.prompt_service.config.max_answer_length
            )
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Language model generation failed: {str(e)}")
            # Fallback to simple answer generation
            return self._generate_simple_answer(prompt, question)
    
    def _enrich_context_results(self, results_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich context results with full document information.
        
        Args:
            results_metadata: Raw metadata from vector search
            
        Returns:
            Enriched context results with document details
        """
        enriched_results = []
        seen_chunks = set()  # Track unique chunks to avoid duplicates
        
        for result in results_metadata:
            # Get full document information
            document = self.document_service.get_document(result.get('document_id'))
            
            # Create enriched result
            enriched_result = result.copy()
            if document:
                enriched_result['filename'] = document.filename
                enriched_result['document_title'] = getattr(document, 'title', document.filename)
            else:
                enriched_result['filename'] = f"Document {result.get('document_id', 'Unknown')}"
                enriched_result['document_title'] = enriched_result['filename']
            
            # Clean and enhance the text content
            text = enriched_result.get('text', '')
            if text:
                # Clean the text
                text = self._clean_text(text)
                enriched_result['text'] = text
                
                # Track unique chunks (based on chunk_id to avoid duplicates)
                chunk_id = enriched_result.get('chunk_id', '')
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    enriched_results.append(enriched_result)
        
        # Limit to top 8 unique chunks for better context management
        enriched_results = enriched_results[:8]
        
        return enriched_results
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and format text content for better processing.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common artifacts
        text = text.replace('...', ' ')
        text = text.replace('..', ' ')
        
        # Clean up punctuation
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        
        return text.strip()

    def _generate_simple_answer(self, prompt: str, question: str) -> str:
        """
        Generate a simple answer when LLM is not available.
        
        Args:
            prompt: The full prompt
            question: User question
            
        Returns:
            Simple answer
        """
        # Extract context from the prompt
        context_start = prompt.find("Context Information:")
        context_end = prompt.find("Question:")
        
        if context_start != -1 and context_end != -1:
            context = prompt[context_start:context_end].replace("Context Information:", "").strip()
            
            # Simple answer based on context
            if "summarize" in question.lower():
                return f"Here's a summary of the key information: {context[:500]}..."
            elif "compare" in question.lower():
                return f"Based on the provided context, here are the key points for comparison: {context[:400]}..."
            elif "analyze" in question.lower():
                return f"Analysis of the information: {context[:400]}..."
            else:
                return f"Based on the document content: {context[:400]}..."
        
        return "I found some relevant information, but I need a language model to provide a proper answer."
    
    def _generate_fallback_answer(self, question: str, context_results: List[Dict[str, Any]]) -> tuple[str, float]:
        """
        Generate a fallback answer when the main generation fails.
        
        Args:
            question: User question
            context_results: Retrieved context chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not context_results:
            return "I couldn't find any relevant information to answer your question.", 0.0
        
        # Sort by similarity score
        sorted_results = sorted(context_results, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        # Take top chunks and combine
        top_chunks = sorted_results[:3]
        combined_text = "\n\n".join([chunk.get('text', '') for chunk in top_chunks])
        
        # Calculate average confidence
        avg_confidence = sum(chunk.get('similarity_score', 0) for chunk in top_chunks) / len(top_chunks)
        
        # Simple answer generation
        answer = f"Based on the document content:\n\n{combined_text[:800]}..."
        
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