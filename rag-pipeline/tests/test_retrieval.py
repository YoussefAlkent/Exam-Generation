import pytest
from src.retrieval.retriever import Retriever
from langchain_community.vectorstores import Chroma
from unittest.mock import MagicMock, patch

@pytest.fixture
def setup_retriever():
    # Mock the vector store
    vector_store = MagicMock(spec=Chroma)
    vector_store.query.return_value = [{"content": "Test document", "metadata": {}}]
    vector_store.embed_query.return_value = [0.1] * 10  # Mock embedding
    
    # Setup code for initializing the Retriever instance
    retriever = Retriever(vector_store)
    return retriever

def test_retrieve_documents(setup_retriever):
    retriever = setup_retriever
    query = "What is the significance of RAG in NLP?"
    results = retriever.retrieve_documents(query)
    
    assert isinstance(results, list)
    assert len(results) > 0  # Ensure that we get some results back

def test_retrieve_documents_empty_query(setup_retriever):
    retriever = setup_retriever
    # Mock query method to return empty list for empty query
    retriever.vector_store.query.return_value = []
    
    query = ""
    results = retriever.retrieve_documents(query)
    
    assert results == []  # Expecting an empty list for an empty query

def test_retrieve_documents_no_results(setup_retriever):
    retriever = setup_retriever
    # Mock query method to return empty list for this test
    retriever.vector_store.query.return_value = []
    
    query = "This query should return no results"
    results = retriever.retrieve_documents(query)
    
    assert results == []  # Expecting an empty list for a query with no results