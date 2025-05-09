import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Mock the entire sentence_transformers module
sentence_transformers_mock = MagicMock()
sys.modules['sentence_transformers'] = sentence_transformers_mock

@pytest.fixture
def sample_texts():
    return [
        "This is a test document about machine learning.",
        "Another document discussing artificial intelligence.",
        "A third document about deep learning and neural networks."
    ]

@pytest.fixture
def sample_query():
    return "What are the applications of machine learning?"

@pytest.fixture
def mock_embedding_model():
    with patch('sentence_transformers.SentenceTransformer') as mock:
        model = Mock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock.return_value = model
        yield mock

def test_embedding_model_initialization(mock_embedding_model):
    with patch('src.embedding.embedding.EmbeddingModel') as MockEmbeddingModel:
        model = MockEmbeddingModel()
        assert model is not None

def test_embed_documents(mock_embedding_model, sample_texts):
    with patch('src.embedding.embedding.EmbeddingModel') as MockEmbeddingModel:
        model = MockEmbeddingModel()
        model.embed_documents.return_value = [[0.1, 0.2, 0.3] for _ in sample_texts]
        embeddings = model.embed_documents(sample_texts)
        assert embeddings is not None
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)

def test_embed_query(mock_embedding_model, sample_query):
    with patch('src.embedding.embedding.EmbeddingModel') as MockEmbeddingModel:
        model = MockEmbeddingModel()
        model.embed_query.return_value = [0.1, 0.2, 0.3]
        embedding = model.embed_query(sample_query)
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 3

def test_embed_empty_documents(mock_embedding_model):
    with patch('src.embedding.embedding.EmbeddingModel') as MockEmbeddingModel:
        model = MockEmbeddingModel()
        model.embed_documents.side_effect = ValueError
        with pytest.raises(ValueError):
            model.embed_documents([])

def test_embed_empty_query(mock_embedding_model):
    with patch('src.embedding.embedding.EmbeddingModel') as MockEmbeddingModel:
        model = MockEmbeddingModel()
        model.embed_query.side_effect = ValueError
        with pytest.raises(ValueError):
            model.embed_query("")

def test_embed_with_custom_model():
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model
        model = EmbeddingModel(model_name="custom-model")
        embeddings = model.embed_documents(["test"])
        assert len(embeddings) == 1

def test_embed_with_error_handling(sample_texts):
    model = EmbeddingModel()
    with patch.object(model.model, 'encode') as mock_encode:
        mock_encode.side_effect = Exception("Embedding Error")
        with pytest.raises(Exception) as exc_info:
            model.embed_documents(sample_texts)
        assert "Embedding Error" in str(exc_info.value)

def test_embed_with_batch_processing(sample_texts):
    model = EmbeddingModel()
    with patch.object(model.model, 'encode') as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        embeddings = model.embed_documents(sample_texts, batch_size=2)
        assert len(embeddings) == len(sample_texts)

def test_embed_with_normalization(sample_texts):
    model = EmbeddingModel()
    with patch.object(model.model, 'encode') as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        embeddings = model.embed_documents(sample_texts, normalize=True)
        assert all(abs(sum(x*x for x in emb) - 1.0) < 1e-6 for emb in embeddings)

def test_embed_with_different_dimensions(sample_texts):
    model = EmbeddingModel()
    with patch.object(model.model, 'encode') as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        embeddings = model.embed_documents(sample_texts)
        assert all(len(emb) == 4 for emb in embeddings)

def test_embed_with_metadata(sample_texts):
    model = EmbeddingModel()
    with patch.object(model.model, 'encode') as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
        metadata = [{'source': f'doc{i}'} for i in range(len(sample_texts))]
        embeddings = model.embed_documents(sample_texts, metadata=metadata)
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(emb, dict) for emb in embeddings)
        assert all('embedding' in emb and 'metadata' in emb for emb in embeddings)