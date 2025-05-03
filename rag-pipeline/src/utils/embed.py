import os
import logging
import google.generativeai as genai
from src.utils.logger import get_logger

# Initiate the logger
logger = get_logger(__name__)

def initialize_google_embeddings():
    """Initialize Google Generative AI for embeddings"""
    # Reading from environment variable instead of hardcoding the model
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    
    # Initialize the Google API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Google API key not found in environment variables")
        return None
    
    genai.configure(api_key=api_key)
    
    try:
        return genai.embed_content(
            model=embedding_model,
            content="Test",
            task_type="retrieval_document"
        )
    except Exception as e:
        logger.error(f"Error initializing Google embeddings: {e}")
        return None