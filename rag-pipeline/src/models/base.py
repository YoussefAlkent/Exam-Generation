from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain_core.language_models.base import BaseLanguageModel

class BaseModelWrapper(ABC):
    """Base class for all language model wrappers."""
    
    @abstractmethod
    def get_model(self) -> BaseLanguageModel:
        """Return the underlying LangChain model."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the model."""
        pass
    
    @abstractmethod
    def get_provider(self) -> str:
        """Return the provider of the model."""
        pass