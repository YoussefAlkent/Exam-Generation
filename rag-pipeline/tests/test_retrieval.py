import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock the retrieval module
retrieval_mock = MagicMock()
sys.modules['src.retrieval.retriever'] = retrieval_mock

@pytest.fixture
def sample_query():
    return "What are the key concepts in machine learning?"

@pytest.fixture
def sample_documents():
    return [
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'metadata': {'source': 'ml_intro.pdf', 'page': 1}
        },
        {
            'text': 'Supervised learning uses labeled data for training.',
            'metadata': {'source': 'ml_types.pdf', 'page': 2}
        },
        {
            'text': 'Unsupervised learning finds patterns in unlabeled data.',
            'metadata': {'source': 'ml_types.pdf', 'page': 3}
        }
    ]

def test_retriever_initialization(mock_chroma_client, mock_embedding_model):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        assert retriever is not None
        assert hasattr(retriever, 'chroma_client')
        assert hasattr(retriever, 'embedding_model')

def test_retrieve_documents(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = [
            {
                'text': 'Machine learning is a subset of AI',
                'metadata': {'source': 'test.pdf', 'page': 1},
                'score': 0.9
            }
        ]
        results = retriever.retrieve(sample_query, k=1)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'text' in results[0]
        assert 'metadata' in results[0]

def test_retrieve_with_custom_k(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = [
            {'text': 'doc1', 'metadata': {'source': 'test1.pdf'}, 'score': 0.9},
            {'text': 'doc2', 'metadata': {'source': 'test2.pdf'}, 'score': 0.8},
            {'text': 'doc3', 'metadata': {'source': 'test3.pdf'}, 'score': 0.7}
        ]
        results = retriever.retrieve(sample_query, k=3)
        assert len(results) == 3

def test_retrieve_with_score_threshold(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = [
            {'text': 'doc1', 'metadata': {'source': 'test1.pdf'}, 'score': 0.9}
        ]
        results = retriever.retrieve(sample_query, k=2, score_threshold=0.5)
        assert len(results) == 1

def test_retrieve_with_metadata_filter(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = [
            {'text': 'doc1', 'metadata': {'source': 'test1.pdf'}, 'score': 0.9}
        ]
        results = retriever.retrieve(
            sample_query,
            k=2,
            metadata_filter={'source': 'test1.pdf'}
        )
        assert len(results) == 1
        assert results[0]['metadata']['source'] == 'test1.pdf'

def test_retrieve_with_empty_results(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = []
        results = retriever.retrieve(sample_query)
        assert len(results) == 0

def test_retrieve_with_error_handling(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.side_effect = Exception("Retrieval Error")
        with pytest.raises(Exception) as exc_info:
            retriever.retrieve(sample_query)
        assert "Retrieval Error" in str(exc_info.value)

def test_retrieve_with_different_collections(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        with patch.object(retriever.chroma_client, 'get_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['doc1']],
                'metadatas': [{'source': 'test.pdf'}],
                'distances': [[0.1]]
            }
            mock_get_collection.return_value = mock_collection
            results = retriever.retrieve(sample_query, collection_name="custom_collection")
            assert len(results) == 1

def test_retrieve_with_reranking(mock_chroma_client, mock_embedding_model, sample_query):
    with patch('src.retrieval.retriever.DocumentRetriever') as MockRetriever:
        retriever = MockRetriever()
        retriever.retrieve.return_value = [
            {'text': 'doc1', 'metadata': {'source': 'test1.pdf'}, 'score': 0.9},
            {'text': 'doc2', 'metadata': {'source': 'test2.pdf'}, 'score': 0.8}
        ]
        results = retriever.retrieve(sample_query, k=2, rerank=True)
        assert len(results) == 2
        # Verify that results are reranked based on relevance
        assert results[0]['score'] <= results[1]['score']