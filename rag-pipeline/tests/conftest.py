import pytest
from unittest.mock import Mock, patch
import os
import sys
import tempfile
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

@pytest.fixture
def mock_model_factory():
    with patch('src.models.factory.ModelFactory') as mock:
        factory = Mock()
        factory.create_model.return_value = Mock()
        mock.return_value = factory
        yield mock

@pytest.fixture
def mock_embedding_model():
    with patch('src.embedding.embedding.EmbeddingModel') as mock:
        model = Mock()
        model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock.return_value = model
        yield mock

@pytest.fixture
def mock_chroma_client():
    with patch('chromadb.Client') as mock:
        client = Mock()
        collection = Mock()
        collection.add.return_value = None
        collection.query.return_value = {
            'documents': [['test document']],
            'metadatas': [{'source': 'test.pdf'}],
            'distances': [[0.1]]
        }
        client.create_collection.return_value = collection
        client.get_collection.return_value = collection
        mock.return_value = client
        yield mock

@pytest.fixture
def temp_pdf_file():
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        # Create a minimal PDF file for testing
        f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF')
        f.flush()
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_streamlit():
    with patch('streamlit') as mock:
        mock.session_state = {}
        mock.sidebar = Mock()
        mock.sidebar.__enter__ = Mock(return_value=mock.sidebar)
        mock.sidebar.__exit__ = Mock(return_value=None)
        yield mock

@pytest.fixture
def sample_rubric():
    return {
        'title': 'Test Rubric',
        'criteria': [
            {
                'name': 'Understanding',
                'description': 'Demonstrates understanding of concepts',
                'weight': 0.4,
                'levels': [
                    {'score': 4, 'description': 'Excellent'},
                    {'score': 3, 'description': 'Good'},
                    {'score': 2, 'description': 'Fair'},
                    {'score': 1, 'description': 'Poor'}
                ]
            }
        ]
    }

@pytest.fixture
def mock_llm_response():
    return {
        'text': 'This is a mock LLM response',
        'usage': {'total_tokens': 100},
        'model': 'test-model'
    }

@pytest.fixture
def temp_chroma_db():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['CHROMA_DB_DIR'] = temp_dir
        yield temp_dir
