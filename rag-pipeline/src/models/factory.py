import os
from typing import Dict, Any, Optional

from .langchain_ollama import LangchainOllama
from .langchain_groq import LangchainGroq
from .langchain_google import LangchainGoogleGenai
from .base import BaseModelWrapper

class ModelFactory:
    """Factory for creating LLM model instances based on configuration."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModelWrapper:
        """Create a model instance based on the provided configuration.
        
        Args:
            config: Dictionary containing model configuration
                   Must include 'provider' key with value 'ollama', 'groq', or 'google'
                   
        Returns:
            An instance of the appropriate model class
        
        Raises:
            ValueError: If the provider is not supported
        """
        default_provider = os.environ.get("DEFAULT_MODEL_PROVIDER", "ollama").lower()
        default_model_name = os.environ.get("DEFAULT_MODEL_NAME", "llama3")
        default_ollama_url = os.environ.get("DEFAULT_OLLAMA_URL", "http://localhost:11434")
        
        provider = config.get("provider", default_provider).lower()
        
        if provider == "ollama":
            return LangchainOllama(
                model_name=config.get("model_name", default_model_name),
                base_url=config.get("base_url", default_ollama_url)
            )
        elif provider == "groq":
            return LangchainGroq(
                model_name=config.get("model_name", default_model_name),
                api_key=config.get("api_key")
            )
        elif provider == "google":
            return LangchainGoogleGenai(
                model_name=config.get("model_name", default_model_name),
                api_key=config.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")