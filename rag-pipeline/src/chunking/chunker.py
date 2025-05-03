class Chunker:
    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size

    def chunk_text(self, text, chunk_size=None):
        """Splits the input text into manageable chunks.
        
        Args:
            text (str): The text to be chunked
            chunk_size (int, optional): Override the default chunk size
            
        Returns:
            list: A list of text chunks
        """
        if not text:
            return []
            
        # Use provided chunk_size or default
        chunk_size = chunk_size or self.chunk_size
        
        # Split by words to avoid cutting in the middle of words
        words = text.split()
        
        # Handle empty input
        if not words:
            return []
            
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks