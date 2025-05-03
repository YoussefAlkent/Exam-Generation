class Vectorizer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def vectorize_text(self, text):
        """
        Vectorizes the input text using the specified embedding model.
        
        Args:
            text (str): The text to be vectorized.
        
        Returns:
            list: A list of vectors representing the input text.
        """
        return self.embedding_model.embed(text)