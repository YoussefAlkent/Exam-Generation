import pytest
from src.chunking.chunker import Chunker

def test_chunk_text():
    chunker = Chunker()
    text = "This is a sample text that needs to be chunked into smaller pieces."
    expected_chunks = [
        "This is a sample text that needs to be",
        "chunked into smaller pieces."
    ]
    chunks = chunker.chunk_text(text, chunk_size=50)
    assert chunks == expected_chunks

def test_chunk_text_empty():
    chunker = Chunker()
    text = ""
    expected_chunks = []
    chunks = chunker.chunk_text(text, chunk_size=50)
    assert chunks == expected_chunks

def test_chunk_text_long():
    chunker = Chunker()
    text = "A" * 200  # 200 characters of 'A'
    expected_chunks = ["A" * 100, "A" * 100]  # Two chunks of 100 characters each
    chunks = chunker.chunk_text(text, chunk_size=100)
    assert chunks == expected_chunks

def test_chunk_text_no_split():
    chunker = Chunker()
    text = "Short text."
    expected_chunks = ["Short text."]
    chunks = chunker.chunk_text(text, chunk_size=50)
    assert chunks == expected_chunks