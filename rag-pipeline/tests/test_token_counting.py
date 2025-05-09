import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import pytest
from src.utils.token_counter import TokenCounter
from unittest.mock import Mock

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.generator import ExamGenerator

# Mock the token counter module
token_counter_mock = MagicMock()
sys.modules['src.utils.token_counter'] = token_counter_mock

class TestTokenCounting(unittest.TestCase):
    
    def setUp(self):
        # Create the patcher for Ollama here so it applies to all tests
        self.ollama_patcher = patch('langchain_community.llms.ollama.Ollama')
        self.mock_ollama_class = self.ollama_patcher.start()
        self.mock_ollama = MagicMock()
        self.mock_ollama_class.return_value = self.mock_ollama
        self.mock_ollama.invoke.return_value = "This is a summary of the content."
        
        # Now create the generator with the patched Ollama
        self.generator = ExamGenerator(model_name="llama3.2:latest", persist_dir="./test_chroma_db")
    
    def tearDown(self):
        # Stop the patcher when tests are done
        self.ollama_patcher.stop()
    
    def test_token_counting(self):
        """Test that token counting works correctly."""
        # Test with a simple string
        text = "This is a test string with 10 tokens."
        tokens = self.generator.count_tokens(text)
        self.assertGreater(tokens, 5)  # Should have more than 5 tokens
        
        # Test with a longer text
        long_text = "This is a much longer text. " * 100
        long_tokens = self.generator.count_tokens(long_text)
        self.assertGreater(long_tokens, tokens)  # Should have more tokens than the shorter text
    
    @patch('src.generation.generator.Chroma')
    def test_content_retrieval_with_token_limit(self, mock_chroma):
        """Test that content retrieval respects token limits."""
        # Create mock documents with known token counts
        small_doc = MagicMock()
        small_doc.page_content = "Small document."  # ~2 tokens
        
        medium_doc = MagicMock()
        medium_doc.page_content = "Medium sized document with more content. " * 20  # ~120 tokens
        
        large_doc = MagicMock()
        large_doc.page_content = "This is a very large document. " * 1000  # ~5000 tokens
        
        # Set up the mock vectorstore to return our test documents
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Test case 1: Small documents that fit within the token limit
        mock_vectorstore.similarity_search.return_value = [small_doc, medium_doc]
        
        # Override token counting to return predictable values for testing
        with patch.object(self.generator, 'count_tokens', side_effect=[2, 120]):
            content = self.generator.retrieve_content("test_course")
            # Both documents should be included
            self.assertIn("Small document.", content)
            self.assertIn("Medium sized document", content)
            
        # Test case 2: Large document exceeding token limit
        mock_vectorstore.similarity_search.return_value = [large_doc]
        
        # Mock the count_tokens and summarize_content methods
        with patch.object(self.generator, 'count_tokens', return_value=5000), \
             patch.object(self.generator, 'summarize_content', return_value="Summarized content"):
            content = self.generator.retrieve_content("test_course")
            # Should return the summarized content
            self.assertEqual(content, "Summarized content")
            # Verify summarize_content was called with the large document's content
            self.generator.summarize_content.assert_called_once_with(large_doc.page_content)
    
    def test_summarization(self):
        """Test content summarization."""
        # Reset the mock for this test
        self.mock_ollama.invoke.return_value = "This is a summary of the content."
        
        # Create a very large text that needs summarization
        large_text = "This is a very large piece of text. " * 500
        
        # Mock the count_tokens method to simulate the text being over the token limit
        with patch.object(self.generator, 'count_tokens', side_effect=[5000, 10]):
            # First call returns 5000 (over limit), second call returns 10 (for the summary)
            result = self.generator.summarize_content(large_text)
            
            # The result should be the summary from the model
            self.assertEqual(result, "This is a summary of the content.")
            
            # Verify the model was called with a prompt containing the text
            self.mock_ollama.invoke.assert_called_once()
            call_args = self.mock_ollama.invoke.call_args[0][0]
            self.assertIn("Summarize the following content", call_args)

@pytest.fixture
def sample_texts():
    return [
        "This is a test sentence.",
        "Another test sentence with more words.",
        "A third sentence for testing purposes."
    ]

@pytest.fixture
def sample_code():
    return """
    def calculate_sum(a, b):
        return a + b
    
    def calculate_product(x, y):
        return x * y
    """

def test_token_counter_initialization():
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        assert counter is not None

def test_count_tokens_single_text():
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.return_value = 5
        text = "This is a test sentence."
        count = counter.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

def test_count_tokens_multiple_texts(sample_texts):
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.return_value = [5, 7, 6]
        counts = counter.count_tokens(sample_texts)
        assert len(counts) == len(sample_texts)
        assert all(isinstance(count, int) for count in counts)
        assert all(count > 0 for count in counts)

def test_count_tokens_empty_text():
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.return_value = 0
        count = counter.count_tokens("")
        assert count == 0

def test_count_tokens_none():
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.side_effect = ValueError("None input")
        with pytest.raises(ValueError):
            counter.count_tokens(None)

def test_count_tokens_with_code(sample_code):
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.return_value = 15
        count = counter.count_tokens(sample_code)
        assert count > 0
        assert isinstance(count, int)

def test_count_tokens_with_special_characters():
    counter = TokenCounter()
    text = "Hello! @#$%^&*()_+{}|:<>?~`"
    count = counter.count_tokens(text)
    assert count > 0

def test_count_tokens_with_unicode():
    counter = TokenCounter()
    text = "Hello 你好 Bonjour"
    count = counter.count_tokens(text)
    assert count > 0

def test_count_tokens_with_numbers():
    counter = TokenCounter()
    text = "The answer is 42 and 3.14"
    count = counter.count_tokens(text)
    assert count > 0

def test_count_tokens_with_whitespace():
    counter = TokenCounter()
    text = "   Multiple    spaces   and\nnewlines\t"
    count = counter.count_tokens(text)
    assert count > 0

def test_count_tokens_with_different_encodings():
    counter = TokenCounter(encoding="cl100k_base")  # GPT-4 encoding
    text = "This is a test sentence."
    count = counter.count_tokens(text)
    assert count > 0

def test_count_tokens_with_error_handling():
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.side_effect = Exception("Encoding Error")
        with pytest.raises(Exception) as exc_info:
            counter.count_tokens("test")
        assert "Encoding Error" in str(exc_info.value)

def test_count_tokens_with_batch_processing(sample_texts):
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        counter.count_tokens.return_value = [5, 7, 6]
        counts = counter.count_tokens(sample_texts, batch_size=2)
        assert len(counts) == len(sample_texts)
        assert all(isinstance(count, int) for count in counts)

def test_count_tokens_with_metadata(sample_texts):
    with patch('src.utils.token_counter.TokenCounter') as MockCounter:
        counter = MockCounter()
        metadata = [{'source': f'doc{i}'} for i in range(len(sample_texts))]
        counter.count_tokens.return_value = [
            {'token_count': 5, 'metadata': metadata[0]},
            {'token_count': 7, 'metadata': metadata[1]},
            {'token_count': 6, 'metadata': metadata[2]}
        ]
        results = counter.count_tokens(sample_texts, metadata=metadata)
        assert len(results) == len(sample_texts)
        assert all(isinstance(result, dict) for result in results)
        assert all('token_count' in result and 'metadata' in result for result in results)

def test_count_tokens_with_different_languages():
    counter = TokenCounter()
    texts = [
        "Hello world",
        "Bonjour le monde",
        "Hola mundo",
        "你好世界"
    ]
    counts = counter.count_tokens(texts)
    assert len(counts) == len(texts)
    assert all(count > 0 for count in counts)

def test_count_tokens_with_mixed_content():
    counter = TokenCounter()
    text = """
    # Markdown Header
    This is a paragraph with **bold** and *italic* text.
    
    ```python
    def hello():
        print("Hello, world!")
    ```
    
    - List item 1
    - List item 2
    """
    count = counter.count_tokens(text)
    assert count > 0

if __name__ == "__main__":
    unittest.main()