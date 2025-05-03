import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.generator import ExamGenerator

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

if __name__ == "__main__":
    unittest.main()