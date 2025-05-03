from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.base import BaseModelWrapper
import os

class LangchainGoogleGenai(BaseModelWrapper):
    """Wrapper for Langchain's Google Generative AI integration."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: str = None):
        """Initialize a Google Generative AI chat model.
        
        Args:
            model_name: The name of the model to use (default: gemini-1.5-pro)
            api_key: Your Google API key (default: None, will try to load from env)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required. Either pass it directly or set GOOGLE_API_KEY environment variable.")
        
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key
        )
    
    def get_model(self):
        """Return the underlying LangChain model."""
        return self.model
    
    def get_name(self) -> str:
        """Return the name of the model."""
        return self.model_name
        
    def get_provider(self) -> str:
        """Return the provider of the model."""
        return "google"