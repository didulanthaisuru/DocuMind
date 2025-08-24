import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from ..utils.logger import setup_logger


class LanguageModelInterface(ABC):
    """Abstract interface for language model providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and configured."""
        pass


class OpenAIService(LanguageModelInterface):
    """OpenAI GPT model service."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI service.
        
        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.logger = setup_logger(self.__class__.__name__)
    
    def is_available(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text
        """
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        try:
            import openai
            openai.api_key = self.api_key
            
            # Set default parameters
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            response = openai.ChatCompletion.create(**params)
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise


class HuggingFaceService(LanguageModelInterface):
    """Hugging Face transformers service."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize Hugging Face service.
        
        Args:
            model_name: Model name from Hugging Face Hub
        """
        self.model_name = model_name
        self.logger = setup_logger(self.__class__.__name__)
        self._pipeline = None
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def _get_pipeline(self):
        """Get or create the text generation pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-generation", model=self.model_name)
            except Exception as e:
                self.logger.error(f"Failed to load Hugging Face model: {str(e)}")
                raise
        
        return self._pipeline
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Hugging Face transformers.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (max_length, temperature, etc.)
            
        Returns:
            Generated text
        """
        if not self.is_available():
            raise ImportError("Transformers package not installed. Run: pip install transformers torch")
        
        try:
            pipeline = self._get_pipeline()
            
            # Set default parameters
            params = {
                "max_length": kwargs.get("max_length", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": True,
                "pad_token_id": pipeline.tokenizer.eos_token_id,
            }
            
            response = pipeline(prompt, **params)
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Hugging Face generation error: {str(e)}")
            raise


class MockLanguageModelService(LanguageModelInterface):
    """Mock language model service for testing and development."""
    
    def __init__(self):
        """Initialize mock service."""
        self.logger = setup_logger(self.__class__.__name__)
    
    def is_available(self) -> bool:
        """Mock service is always available."""
        return True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a mock response based on the prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock generated text
        """
        # Extract context and question from the prompt
        context_start = prompt.find("Context Information:")
        context_end = prompt.find("Question:")
        
        if context_start != -1 and context_end != -1:
            context = prompt[context_start:context_end].replace("Context Information:", "").strip()
            question = prompt[context_end:].replace("Question:", "").strip()
            
            # Generate different types of responses based on question keywords
            question_lower = question.lower()
            
            # Handle "main topics" or "topics" questions
            if "main topic" in question_lower or "topics" in question_lower:
                # Extract key topics from context
                topics = self._extract_topics_from_context(context)
                return f"Based on the document content, the main topics include: {topics}"
            
            elif "summarize" in question_lower or "summary" in question_lower:
                return f"Here's a comprehensive summary of the key information: {context[:300]}..."
            
            elif "compare" in question_lower or "difference" in question_lower:
                return f"Based on the provided context, here are the key points for comparison: {context[:250]}..."
            
            elif "analyze" in question_lower or "analysis" in question_lower:
                return f"Analysis of the information reveals several important points: {context[:250]}..."
            
            elif "what is" in question_lower or "define" in question_lower:
                return f"Based on the context, this refers to: {context[:200]}..."
            
            elif "how" in question_lower or "why" in question_lower:
                return f"The context explains that: {context[:250]}..."
            
            else:
                return f"Based on the document content, I can provide the following information: {context[:300]}..."
        
        return "I found relevant information but need more context to provide a specific answer."
    
    def _extract_topics_from_context(self, context: str) -> str:
        """
        Extract main topics from the context text.
        
        Args:
            context: Context text to analyze
            
        Returns:
            Extracted topics as a string
        """
        # Simple topic extraction based on common patterns
        topics = []
        
        # Look for common topic indicators
        if "abstract" in context.lower():
            topics.append("Abstract guidelines and formatting")
        if "table of contents" in context.lower():
            topics.append("Document structure and organization")
        if "bsc" in context.lower() or "degree" in context.lower():
            topics.append("BSc degree program requirements")
        if "ai" in context.lower() or "artificial intelligence" in context.lower():
            topics.append("Artificial Intelligence curriculum")
        if "guidelines" in context.lower():
            topics.append("Academic guidelines and standards")
        if "report" in context.lower():
            topics.append("Report writing and submission")
        
        # If no specific topics found, extract key phrases
        if not topics:
            # Look for capitalized phrases that might be topics
            import re
            potential_topics = re.findall(r'\b[A-Z][a-zA-Z\s]{3,}\b', context)
            if potential_topics:
                topics = potential_topics[:3]  # Take first 3 potential topics
        
        if topics:
            return ", ".join(topics)
        else:
            return "Document structure, academic guidelines, and program requirements"


class GeminiService(LanguageModelInterface):
    """Google Gemini model service."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Gemini API key (will use environment variable if not provided)
            model: Model to use (gemini-pro, gemini-pro-vision, etc.)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.logger = setup_logger(self.__class__.__name__)
        self._client = None
    
    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        return bool(self.api_key)
    
    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {str(e)}")
                raise
        
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text
        """
        if not self.is_available():
            raise ValueError("Gemini API key not configured")
        
        try:
            client = self._get_client()
            
            # Set default parameters optimized for document analysis
            generation_config = {
                "temperature": kwargs.get("temperature", 0.3),  # Lower temperature for more focused responses
                "max_output_tokens": kwargs.get("max_tokens", 1500),  # Increased for better summaries
                "top_p": kwargs.get("top_p", 0.8),
                "top_k": kwargs.get("top_k", 40),
            }
            
            # Add safety settings for better content generation
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            response = client.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.text:
                return response.text
            else:
                raise ValueError("Empty response from Gemini API")
                
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            raise


class LanguageModelService:
    """
    Unified language model service that can work with different providers.
    """
    
    def __init__(self, provider: str = "mock", **kwargs):
        """
        Initialize the language model service.
        
        Args:
            provider: Provider to use ("openai", "huggingface", "mock")
            **kwargs: Provider-specific configuration
        """
        self.provider = provider.lower()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Initialize the appropriate provider
        if self.provider == "openai":
            self.model = OpenAIService(**kwargs)
        elif self.provider == "huggingface":
            self.model = HuggingFaceService(**kwargs)
        elif self.provider == "gemini":
            self.model = GeminiService(**kwargs)
        elif self.provider == "mock":
            self.model = MockLanguageModelService()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured provider.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if not self.model.is_available():
            raise ValueError(f"Language model provider '{self.provider}' is not available")
        
        try:
            return self.model.generate(prompt, **kwargs)
        except Exception as e:
            self.logger.error(f"Language model generation failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if the language model is available."""
        return self.model.is_available()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.provider,
            "available": self.is_available(),
            "model_type": type(self.model).__name__
        }
