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
        for i, chunk in enumerate(unique_chunks[:8], 1):  # Increased to top 8 chunks for better context
            text = chunk.get('text', '').strip()
            if not text:
                continue
                
            # Clean and format the text
            text = self._clean_text(text)
            
            # Truncate if too long (increased limit for better context)
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            # Add source information
            source_info = self._get_source_info(chunk)
            formatted_chunk = f"[Source {i}: {source_info}]\n{text}\n"
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and format text for better readability.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common artifacts
        text = re.sub(r'^\s*[•\-]\s*', '', text)  # Remove bullet points at start
        text = re.sub(r'\s*[•\-]\s*$', '', text)  # Remove bullet points at end
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
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
        """Get template for general Q&A prompts optimized for Gemini AI."""
        return """You are an expert AI assistant powered by Google's Gemini AI. Your task is to provide accurate, well-reasoned answers based on the provided document context.

Context Information:
{context}

Question: {question}

Instructions:
1. **Base your answer ONLY on the information provided in the context above.**
2. **Be precise and factual** - avoid speculation or information not present in the context.
3. **Structure your response clearly** with logical flow and organization.
4. **Cite specific sources** when referencing information (e.g., "According to Source 1...").
5. **If the context is insufficient**, clearly state: "I don't have enough information in the provided context to answer this question completely."
6. **For complex questions**, break down your response into clear sections.
7. **Maintain academic/professional tone** appropriate for document analysis.
8. **Keep your answer under {max_length} characters** while being comprehensive.
9. **Highlight key insights** and important findings from the context.
10. **If asked for summaries**, focus on main themes, key points, and conclusions.

Answer:"""
    
    def _get_analytical_template(self) -> str:
        """Get template for analytical questions optimized for Gemini AI."""
        return """You are an expert analytical AI powered by Google's Gemini AI. Provide deep, insightful analysis based on the provided document context.

Context Information:
{context}

Question: {question}

Instructions:
1. **Conduct thorough analysis** using critical thinking and logical reasoning.
2. **Identify patterns, relationships, and underlying themes** in the information.
3. **Consider multiple perspectives and interpretations** where the context allows.
4. **Support your analysis with specific evidence** from the context.
5. **Apply relevant analytical frameworks** (e.g., SWOT, cause-effect, comparative analysis).
6. **Highlight implications and consequences** of the findings.
7. **Address potential limitations** or gaps in the available information.
8. **Structure your analysis clearly** with logical flow and organization.
9. **Use professional, academic language** appropriate for document analysis.
10. **Keep your analysis under {max_length} characters** while being comprehensive.
11. **Cite specific sources** when referencing information from the context.
12. **If the context is insufficient**, clearly state what additional information would be needed for a complete analysis.

Analysis:"""
    
    def _get_summarization_template(self) -> str:
        """Get template for summarization tasks optimized for Gemini AI."""
        return """You are an expert document summarizer powered by Google's Gemini AI. Create comprehensive, well-structured summaries of the provided document context.

Context Information:
{context}

Question: {question}

Instructions:
1. **Create a comprehensive summary** that directly addresses the question and covers all key content.
2. **For abstract requests**: Focus on the main research objectives, methodology, key findings, and conclusions. Structure as a proper academic abstract.
3. **For general summaries**: Include all major themes, topics, and key insights from the document.
4. **Structure your response logically** with clear sections and flow.
5. **Include all major themes and topics** discussed in the document.
6. **Highlight key findings, conclusions, and important insights**.
7. **Maintain factual accuracy** - only include information present in the context.
8. **Use professional, academic language** appropriate for document analysis.
9. **Include specific examples and details** that support the main points.
10. **Organize information hierarchically** - main points first, then supporting details.
11. **Keep the summary under {max_length} characters** while being thorough and comprehensive.
12. **If summarizing a specific section**, focus on that content while providing context.
13. **Include source citations** for key information when relevant.
14. **Focus on the most relevant information** from the context provided.
15. **Provide a clear, coherent narrative** that flows logically.
16. **Highlight any important data, statistics, or key points** mentioned in the document.
17. **For abstract requests**: Include research background, objectives, methods, results, and implications.
18. **Be comprehensive but concise** - cover all important aspects without being overly verbose.

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
        
        # Summarization keywords (check first for better priority)
        summary_keywords = ['summarize', 'summary', 'overview', 'brief', 'sum up', 'give summary', 'provide summary', 'document summary', 'abstract', 'give abstract', 'provide abstract', 'what is the abstract', 'extract abstract']
        if any(keyword in question_lower for keyword in summary_keywords):
            return PromptType.SUMMARIZATION
        
        # Analytical keywords
        analytical_keywords = ['analyze', 'analysis', 'why', 'how', 'explain', 'interpret', 'evaluate', 'discuss', 'examine']
        if any(keyword in question_lower for keyword in analytical_keywords):
            return PromptType.ANALYTICAL
        
        # Comparison keywords
        comparison_keywords = ['compare', 'difference', 'similar', 'versus', 'vs', 'contrast', 'different', 'similarities']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return PromptType.COMPARISON
        
        # Fact-checking keywords
        fact_keywords = ['true', 'false', 'verify', 'check', 'fact', 'accurate', 'correct', 'validate']
        if any(keyword in question_lower for keyword in fact_keywords):
            return PromptType.FACT_CHECKING
        
        # Extractive keywords (specific information)
        extractive_keywords = ['what is', 'when', 'where', 'who', 'which', 'find', 'locate', 'extract', 'specific']
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
        context_factor = min(total_length / 2000, 1.0)  # Normalize to 0-1 (increased threshold)
        
        # Consider number of sources (more sources = higher confidence)
        source_factor = min(len(context_results) / 5, 1.0)
        
        # Consider score distribution (if all scores are similar, higher confidence)
        if len(scores) > 1:
            score_variance = 1 - (max(scores) - min(scores))  # Lower variance = higher confidence
        else:
            score_variance = 1.0
        
        # Combine factors with better weighting
        confidence = (avg_score * 0.5) + (context_factor * 0.2) + (source_factor * 0.2) + (score_variance * 0.1)
        
        # Boost confidence for Gemini AI (since it's more capable)
        confidence = min(confidence * 1.2, 1.0)
        
        return confidence
    
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
        
        # Remove source citations that might be embedded in the answer
        answer = re.sub(r'\[Source \d+:.*?\]', '', answer)
        answer = re.sub(r'Sources?:.*$', '', answer, flags=re.MULTILINE | re.DOTALL)
        
        # Fix incomplete sentences that end with ellipsis or are cut off
        if answer.endswith('...'):
            # Try to complete the sentence or remove the ellipsis
            answer = answer[:-3].strip()
        
        # Remove any remaining artifacts from the context
        answer = re.sub(r'\d+ Time Series Splitting.*?This ensures.*?appointments\.', '', answer, flags=re.DOTALL)
        answer = re.sub(r'Page \d+.*?Feature Set Finalization.*?Final Features.*?The f\.\.\.', '', answer, flags=re.DOTALL)
        
        # If the answer is very short or seems incomplete, try to improve it
        if len(answer) < 100:
            # Add a more complete ending
            if not answer.endswith(('.', '!', '?')):
                answer += " based on the available context."
        
        # Ensure proper sentence ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Final cleanup
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
