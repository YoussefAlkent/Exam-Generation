import pytest
from src.embedding.embedder import Embedder

class TestEmbedder:
    def setup_method(self):
        self.embedder = Embedder()

    def test_generate_embeddings(self):
        sample_text = "This is a test sentence."
        embeddings = self.embedder.generate_embeddings(sample_text)
        assert embeddings is not None
        assert isinstance(embeddings, list)
        assert len(embeddings) > 0  # Ensure that embeddings are generated

    def test_generate_embeddings_empty(self):
        sample_text = ""
        embeddings = self.embedder.generate_embeddings(sample_text)
        assert embeddings is not None
        assert isinstance(embeddings, list)
        assert len(embeddings) == 0  # Ensure no embeddings for empty input

    def test_generate_embeddings_invalid_input(self):
        with pytest.raises(ValueError):
            self.embedder.generate_embeddings(None)  # Test for handling None input

    # Add more tests as needed for different scenarios and edge cases