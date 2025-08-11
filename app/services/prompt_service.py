import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from ..utils.logger import setup_logger


class PromptType(Enum):
    """Types of prompts for different use cases."""
    GENERAL_QA = "general_qa"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    FACT_CHECKING = "fact_checking"
    EXTRACTIVE = "extractive"


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    max_context_length: int = 4000
    max_answer_length: int = 1000
    temperature: float = 0.7
    include_sources: bool = True
    confidence_threshold: float = 0.5
    enable_citations: bool = True


class PromptService:
    """
    Service for generating sophisticated prompts for RAG systems.
    Handles context formatting, prompt templates, and answer generation.
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize the prompt service.
        
        Args:
            config: Configuration for prompt generation
        """
        self.config = config or PromptConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Define prompt templates
        self.prompt_templates = {
            PromptType.GENERAL_QA: self._get_general_qa_template(),
            PromptType.ANALYTICAL: self._get_analytical_template(),
            PromptType.SUMMARIZATION: self._get_summarization_template(),
            PromptType.COMPARISON: self._get_comparison_template(),
            PromptType.FACT_CHECKING: self._get_fact_checking_template(),
            PromptType.EXTRACTIVE: self._get_extractive_template(),
        }
    
    def generate_prompt(self, 
                       question: str, 
                       context_results: List[Dict[str, Any]], 
                       prompt_type: PromptType = PromptType.GENERAL_QA) -> str:
        """
        Generate a complete prompt for the language model.
        
        Args:
            question: User's question
            context_results: Retrieved context chunks with metadata
            prompt_type: Type of prompt to generate
            
        Returns:
            Formatted prompt string
        """
        try:
            # Format and organize context
            formatted_context = self._format_context(context_results)
            
            # Get the appropriate template
            template = self.prompt_templates[prompt_type]
            
            # Fill the template
            prompt = template.format(
                context=formatted_context,
                question=question,
                max_length=self.config.max_answer_length,
                include_sources="yes" if self.config.include_sources else "no"
            )
            
            self.logger.debug(f"Generated {prompt_type.value} prompt for question: {question[:50]}...")
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to generate prompt: {str(e)}")
            # Fallback to simple prompt
            return self._generate_fallback_prompt(question, context_results)
    
    def _format_context(self, context_results: List[Dict[str, Any]]) -> str:
        """
        Format and organize context chunks for optimal prompt generation.
        
        Args:
            context_results: List of context chunks with metadata
            
        Returns:
            Formatted context string
        """
        if not context_results:
            return "No relevant context found."
        
        # Sort by similarity score
        sorted_results = sorted(context_results, 
                              key=lambda x: x.get('similarity_score', 0), 
                              reverse=True)
        
        # Remove duplicates and organize
        unique_chunks = self._deduplicate_chunks(sorted_results)
        
        # Format each chunk with source information
        formatted_chunks = []
        for i, chunk in enumerate(unique_chunks[:5], 1):  # Limit to top 5 chunks
            text = chunk.get('text', '').strip()
            if not text:
                continue
                
            # Truncate if too long
            if len(text) > 800:
                text = text[:800] + "..."
            
            # Add source information
            source_info = self._get_source_info(chunk)
            formatted_chunk = f"[Source {i}: {source_info}]\n{text}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate or highly similar chunks.
        
        Args:
            chunks: List of context chunks
            
        Returns:
            Deduplicated list of chunks
        """
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            if not text:
                continue
            
            # Create a normalized version for comparison
            normalized = re.sub(r'\s+', ' ', text.lower())[:100]
            
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _get_source_info(self, chunk: Dict[str, Any]) -> str:
        """
        Extract source information from a chunk.
        
        Args:
            chunk: Context chunk with metadata
            
        Returns:
            Formatted source information
        """
        # Try multiple possible filename fields
        filename = (chunk.get('filename') or 
                   chunk.get('document_title') or 
                   chunk.get('title') or 
                   f"Document {chunk.get('document_id', 'Unknown')}")
        
        page = chunk.get('page')
        score = chunk.get('similarity_score', 0)
        
        source_parts = [filename]
        if page is not None:
            source_parts.append(f"page {page}")
        if score > 0:
            source_parts.append(f"relevance: {score:.2f}")
        
        return " | ".join(source_parts)
    
    def _get_general_qa_template(self) -> str:
        """Get template for general Q&A prompts."""
        return """You are a helpful AI assistant that answers questions based on the provided context. 

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
3. Be concise but comprehensive in your response.
4. Provide a clear, structured answer that directly addresses the question.
5. If you reference specific information, mention the source.
6. Keep your answer under {max_length} characters.
7. Be accurate and avoid making up information not present in the context.
8. For questions about topics or content, identify and list the main themes or subjects discussed.

Answer:"""
    
    def _get_analytical_template(self) -> str:
        """Get template for analytical questions."""
        return """You are an analytical AI assistant that provides detailed analysis based on the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Provide a thorough analysis of the question using the context provided.
2. Identify key points, patterns, and relationships in the information.
3. Consider multiple perspectives if the context allows.
4. Support your analysis with specific examples from the context.
5. If the context is insufficient for analysis, clearly state what additional information would be needed.
6. Keep your analysis under {max_length} characters.
7. Cite sources when referencing specific information.

Analysis:"""
    
    def _get_summarization_template(self) -> str:
        """Get template for summarization tasks."""
        return """You are a summarization AI that creates concise summaries of the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Create a comprehensive summary that addresses the question.
2. Focus on the most important and relevant information.
3. Maintain the key facts and relationships from the original context.
4. Use clear, concise language.
5. Organize the summary logically.
6. Keep the summary under {max_length} characters.
7. Include source information for key points.

Summary:"""
    
    def _get_comparison_template(self) -> str:
        """Get template for comparison questions."""
        return """You are a comparison AI that analyzes similarities and differences based on the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Identify the items or concepts to be compared from the question.
2. Analyze similarities and differences based on the context.
3. Use a structured approach (e.g., point-by-point comparison).
4. Be objective and balanced in your comparison.
5. Support your comparison with specific evidence from the context.
6. Keep your comparison under {max_length} characters.
7. Cite sources for your comparisons.

Comparison:"""
    
    def _get_fact_checking_template(self) -> str:
        """Get template for fact-checking questions."""
        return """You are a fact-checking AI that verifies information against the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Carefully examine the question against the provided context.
2. Determine if the information in the question is supported by the context.
3. Provide specific evidence from the context to support or refute claims.
4. Be precise about what can and cannot be verified.
5. If the context is insufficient for verification, clearly state this.
6. Keep your fact-check under {max_length} characters.
7. Cite specific sources for your verification.

Fact Check:"""
    
    def _get_extractive_template(self) -> str:
        """Get template for extractive Q&A (finding specific information)."""
        return """You are an extractive AI that finds specific information from the provided context.

Context Information:
{context}

Question: {question}

Instructions:
1. Extract the specific information requested from the context.
2. Provide direct quotes or precise information when possible.
3. If the exact information isn't found, provide the closest available information.
4. Be precise and avoid paraphrasing unless necessary.
5. Include source information for extracted content.
6. Keep your response under {max_length} characters.
7. If the information isn't in the context, clearly state this.

Extracted Information:"""
    
    def _generate_fallback_prompt(self, question: str, context_results: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback prompt if the main prompt generation fails.
        
        Args:
            question: User's question
            context_results: Retrieved context chunks
            
        Returns:
            Simple fallback prompt
        """
        context_text = "\n\n".join([chunk.get('text', '') for chunk in context_results[:3]])
        
        return f"""Based on the following context, please answer this question: {question}

Context:
{context_text}

Answer:"""
    
    def determine_prompt_type(self, question: str) -> PromptType:
        """
        Automatically determine the best prompt type based on the question.
        
        Args:
            question: User's question
            
        Returns:
            Appropriate prompt type
        """
        question_lower = question.lower()
        
        # Analytical keywords
        analytical_keywords = ['analyze', 'analysis', 'why', 'how', 'explain', 'interpret', 'evaluate']
        if any(keyword in question_lower for keyword in analytical_keywords):
            return PromptType.ANALYTICAL
        
        # Summarization keywords
        summary_keywords = ['summarize', 'summary', 'overview', 'brief', 'sum up']
        if any(keyword in question_lower for keyword in summary_keywords):
            return PromptType.SUMMARIZATION
        
        # Comparison keywords
        comparison_keywords = ['compare', 'difference', 'similar', 'versus', 'vs', 'contrast']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return PromptType.COMPARISON
        
        # Fact-checking keywords
        fact_keywords = ['true', 'false', 'verify', 'check', 'fact', 'accurate', 'correct']
        if any(keyword in question_lower for keyword in fact_keywords):
            return PromptType.FACT_CHECKING
        
        # Extractive keywords (specific information)
        extractive_keywords = ['what is', 'when', 'where', 'who', 'which', 'find', 'locate']
        if any(keyword in question_lower for keyword in extractive_keywords):
            return PromptType.EXTRACTIVE
        
        # Default to general Q&A
        return PromptType.GENERAL_QA
    
    def post_process_answer(self, answer: str, context_results: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Post-process the generated answer to improve quality and add citations.
        
        Args:
            answer: Raw answer from the language model
            context_results: Original context chunks
            
        Returns:
            Tuple of (processed_answer, confidence_score)
        """
        try:
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(context_results)
            
            # Add citations if enabled
            if self.config.enable_citations and context_results:
                answer = self._add_citations(answer, context_results)
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            return answer, confidence
            
        except Exception as e:
            self.logger.error(f"Failed to post-process answer: {str(e)}")
            return answer, 0.5  # Default confidence
    
    def _calculate_confidence(self, context_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on context quality.
        
        Args:
            context_results: Context chunks used for answer generation
            
        Returns:
            Confidence score between 0 and 1
        """
        if not context_results:
            return 0.0
        
        # Calculate average similarity score
        scores = [chunk.get('similarity_score', 0) for chunk in context_results]
        avg_score = sum(scores) / len(scores)
        
        # Consider context length and diversity
        total_length = sum(len(chunk.get('text', '')) for chunk in context_results)
        context_factor = min(total_length / 1000, 1.0)  # Normalize to 0-1
        
        # Combine factors
        confidence = (avg_score * 0.7) + (context_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def _add_citations(self, answer: str, context_results: List[Dict[str, Any]]) -> str:
        """
        Add source citations to the answer.
        
        Args:
            answer: Original answer
            context_results: Context chunks with source information
            
        Returns:
            Answer with citations
        """
        # Simple citation addition - in a more sophisticated system,
        # you might use regex to identify specific claims and add citations
        if not context_results:
            return answer
        
        # Add a sources section
        sources = []
        for i, chunk in enumerate(context_results[:3], 1):
            source_info = self._get_source_info(chunk)
            sources.append(f"{i}. {source_info}")
        
        if sources:
            answer += f"\n\nSources:\n" + "\n".join(sources)
        
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean and format the answer.
        
        Args:
            answer: Raw answer
            
        Returns:
            Cleaned answer
        """
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common LLM artifacts
        answer = re.sub(r'^(Answer:|Response:|Based on the context,?|According to the context,?)', '', answer, flags=re.IGNORECASE)
        
        # Fix incomplete sentences that end with ellipsis or are cut off
        if answer.endswith('...'):
            # Try to complete the sentence or remove the ellipsis
            answer = answer[:-3].strip()
        
        # If the answer is very short or seems incomplete, try to improve it
        if len(answer) < 50 and not answer.endswith(('.', '!', '?')):
            # Add a more complete ending
            answer += " based on the available context."
        
        # Ensure proper sentence ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer.strip()
