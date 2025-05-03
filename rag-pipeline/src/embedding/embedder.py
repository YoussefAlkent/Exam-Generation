class Embedder:
    def __init__(self, model=None):
        """
        Initialize the embedder with an embedding model.
        
        Args:
            model: The embedding model to use. If None, a default model is used.
        """
        self.model = model
        
        # If no model is provided, we could initialize a default one
        if self.model is None:
            # In a real implementation, a default embedding model would be initialized here
            pass

    def generate_embeddings(self, text):
        """
        Generate embeddings for the given text.
        
        Args:
            text (str): The text to generate embeddings for.
            
        Returns:
            list: The embedding vector for the text.
            
        Raises:
            ValueError: If the input is None.
        """
        if text is None:
            raise ValueError("Input text cannot be None")
            
        if not text:
            return []
            
        # In a real implementation, this would call the model
        if self.model:
            return self.model.embed(text)
        else:
            # Fallback for testing when no model is available
            # Return an empty list or dummy embedding
            return []