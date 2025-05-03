from langchain_groq import ChatGroq
from ..models.base import BaseModelWrapper
import os

class LangchainGroq(BaseModelWrapper):
    """Wrapper for Langchain's Groq integration."""
    
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: str = None):
        """Initialize a Groq chat model.
        
        Args:
            model_name: The name of the model to use (default: llama3-70b-8192)
            api_key: Your Groq API key (default: None, will try to load from env)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Either pass it directly or set GROQ_API_KEY environment variable.")
        
        self.model = ChatGroq(
            model_name=model_name,
            groq_api_key=self.api_key
        )
    
    def get_model(self):
        """Return the underlying LangChain model."""
        return self.model
    
    def get_name(self) -> str:
        """Return the name of the model."""
        return self.model_name
        
    def get_provider(self) -> str:
        """Return the provider of the model."""
        return "groq"