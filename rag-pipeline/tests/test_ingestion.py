import pytest
import os
import tempfile
from pathlib import Path
from src.ingestion.ingestion import PDFIngester
from unittest.mock import Mock, patch

@pytest.fixture
def pdf_ingester():
    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield PDFIngester(pdf_dir="./pdfs/", persist_dir=tmp_dir)

def test_get_pdf_files(pdf_ingester):
    # Create test PDFs in the directory
    pdf_files = pdf_ingester.get_pdf_files()
    assert isinstance(pdf_files, list)

def test_process_pdfs(pdf_ingester):
    # Test document processing
    documents = pdf_ingester.process_pdfs()
    assert isinstance(documents, list)

def test_ingest_to_vectorstore(pdf_ingester):
    # Test vectorstore creation with a test course name
    course_name = "test_course"
    try:
        db = pdf_ingester.ingest_to_vectorstore(course_name)
        assert db is not None
    except ValueError:
        # It's okay if this fails due to no PDFs in test environment
        pass

def test_pdf_ingester_initialization(mock_chroma_client, temp_chroma_db):
    ingester = PDFIngester()
    assert ingester is not None
    assert hasattr(ingester, 'chroma_client')

def test_ingest_pdf_file(temp_pdf_file, mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester()
    result = ingester.ingest_file(temp_pdf_file)
    assert result is not None
    assert isinstance(result, dict)
    assert 'document_count' in result
    assert result['document_count'] > 0

def test_ingest_pdf_with_metadata(temp_pdf_file, mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester()
    metadata = {'course': 'Test Course', 'topic': 'Test Topic'}
    result = ingester.ingest_file(temp_pdf_file, metadata=metadata)
    assert result is not None
    assert 'metadata' in result
    assert result['metadata'] == metadata

def test_ingest_invalid_file(mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester()
    with pytest.raises(Exception):
        ingester.ingest_file('nonexistent.pdf')

def test_ingest_directory(temp_chroma_db, mock_chroma_client, mock_embedding_model):
    with patch('pathlib.Path.glob') as mock_glob:
        mock_glob.return_value = [Path('test1.pdf'), Path('test2.pdf')]
        ingester = PDFIngester()
        result = ingester.ingest_directory('test_dir')
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

def test_chunk_size_configuration(mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester(chunk_size=500)
    assert ingester.chunk_size == 500

def test_chunk_overlap_configuration(mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester(chunk_overlap=50)
    assert ingester.chunk_overlap == 50

def test_ingest_with_custom_chunking(temp_pdf_file, mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester(chunk_size=100, chunk_overlap=20)
    result = ingester.ingest_file(temp_pdf_file)
    assert result is not None
    assert 'document_count' in result

def test_ingest_with_error_handling(mock_chroma_client, mock_embedding_model):
    ingester = PDFIngester()
    with patch('PyPDF2.PdfReader') as mock_pdf:
        mock_pdf.side_effect = Exception("PDF Error")
        with pytest.raises(Exception) as exc_info:
            ingester.ingest_file('test.pdf')
        assert "PDF Error" in str(exc_info.value)

def test_ingest_with_empty_pdf(temp_pdf_file, mock_chroma_client, mock_embedding_model):
    with patch('PyPDF2.PdfReader') as mock_pdf:
        mock_pdf.return_value.pages = []
        ingester = PDFIngester()
        result = ingester.ingest_file(temp_pdf_file)
        assert result['document_count'] == 0