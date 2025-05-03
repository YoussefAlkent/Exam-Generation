import os
import logging
from typing import Dict, Any, Optional, List
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

class LangchainOllama(BaseModelWrapper):
    """Wrapper for Langchain's model integrations."""
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        """Initialize a language model.
        
        Args:
            model_name: The name of the model to use (default: llama3)
            base_url: The URL of the Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        # Check environment variable to determine which model to use
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        logger.info(f"Using LLM provider: {self.provider} with model: {model_name}")
        
        if self.provider == "google":
            try:
                logger.info(f"Initializing Google Gemini model: {model_name}")
                # Make sure Google API key is set
                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY environment variable is not set")
                self.model = ChatGoogleGenerativeAI(model=model_name)
            except Exception as e:
                logger.error(f"Failed to initialize Google model: {e}")
                logger.warning(f"Falling back to Ollama model: {model_name}")
                self.model = Ollama(model=model_name, base_url=base_url)
                self.provider = "ollama"
        else:
            logger.info(f"Initializing Ollama model: {model_name}")
            self.model = Ollama(model=model_name, base_url=base_url)
    
    def get_model(self):
        """Return the underlying LangChain model."""
        return self.model
    
    def get_name(self) -> str:
        """Return the name of the model."""
        return self.model_name
        
    def get_provider(self) -> str:
        """Return the provider of the model."""
        return self.provider