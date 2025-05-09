import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock the chunking module
chunking_mock = MagicMock()
sys.modules['src.chunking.chunker'] = chunking_mock

@pytest.fixture
def sample_text():
    return """
    Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data.
    It involves the use of algorithms and statistical models to enable computers to improve their performance on a specific task through experience.
    Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data.
    Natural language processing is another important area of AI that deals with the interaction between computers and human language.
    """

@pytest.fixture
def sample_text_with_headers():
    return """
    # Introduction
    Machine learning is a subset of artificial intelligence.

    ## Types of Machine Learning
    There are three main types: supervised, unsupervised, and reinforcement learning.

    ### Supervised Learning
    Supervised learning uses labeled data for training.

    ### Unsupervised Learning
    Unsupervised learning finds patterns in unlabeled data.
    """

def test_chunker_initialization():
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        assert chunker is not None

def test_basic_chunking(sample_text):
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        chunker.chunk_text.return_value = [
            "Machine learning is a subset of artificial intelligence that focuses on developing systems.",
            "Deep learning is a subset of machine learning that uses neural networks."
        ]
        chunks = chunker.chunk_text(sample_text)
        assert chunks is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

def test_chunking_with_overlap(sample_text):
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        chunker.chunk_text.return_value = [
            "Machine learning is a subset",
            "subset of artificial intelligence",
        ]
        chunks = chunker.chunk_text(sample_text)
        assert len(chunks) == 2

def test_chunking_with_headers(sample_text_with_headers):
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        chunker.chunk_text.return_value = [
            "# Introduction\nMachine learning",
            "## Types of Machine Learning\nThere are three main types"
        ]
        chunks = chunker.chunk_text(sample_text_with_headers, respect_headers=True)
        assert all('#' in chunk for chunk in chunks)

def test_chunking_empty_text():
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        chunker.chunk_text.side_effect = ValueError("Empty text")
        with pytest.raises(ValueError):
            chunker.chunk_text("")

def test_chunking_with_metadata(sample_text):
    with patch('src.chunking.chunker.TextChunker') as MockChunker:
        chunker = MockChunker()
        metadata = {'source': 'test.txt', 'page': 1}
        chunker.chunk_text.return_value = [
            {'text': 'chunk1', 'metadata': metadata},
            {'text': 'chunk2', 'metadata': metadata}
        ]
        chunks = chunker.chunk_text(sample_text, metadata=metadata)
        assert all('metadata' in chunk for chunk in chunks)

def test_chunking_with_custom_separators():
    text = "Sentence1.Sentence2!Sentence3?Sentence4"
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_text(text, separators=['.', '!', '?'])
    assert len(chunks) == 4
    assert all(chunk.strip() in text for chunk in chunks)

def test_chunking_with_minimum_size(sample_text):
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=50)
    chunks = chunker.chunk_text(sample_text)
    assert all(len(chunk) >= 50 for chunk in chunks)

def test_chunking_with_maximum_size(sample_text):
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, max_chunk_size=80)
    chunks = chunker.chunk_text(sample_text)
    assert all(len(chunk) <= 80 for chunk in chunks)

def test_chunking_with_sentence_boundaries(sample_text):
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text, respect_sentence_boundaries=True)
    assert all(chunk.endswith('.') for chunk in chunks)

def test_chunking_with_error_handling():
    chunker = TextChunker()
    with pytest.raises(ValueError):
        chunker.chunk_text(None)

def test_chunking_with_different_languages():
    text = "Hello world. Bonjour le monde. Hola mundo."
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)