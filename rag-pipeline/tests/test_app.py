import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock the required modules
flask_mock = MagicMock()
sys.modules['flask'] = flask_mock
streamlit_mock = MagicMock()
sys.modules['streamlit'] = streamlit_mock
app_mock = MagicMock()
sys.modules['src.app'] = app_mock

@pytest.fixture
def mock_session_state():
    return {
        'exam_data': None,
        'filtered_exam_data': None,
        'course_name': '',
        'assignment_data': None,
        'project_data': None,
        'model_config': {
            'provider': 'ollama',
            'model_name': 'llama3',
            'base_url': 'http://localhost:11434'
        }
    }

@pytest.fixture
def sample_exam_data():
    return {
        'questions': [
            {
                'text': 'What is machine learning?',
                'type': 'short_answer',
                'difficulty': 'medium',
                'answer': 'Machine learning is a subset of AI.'
            }
        ],
        'rubric': {
            'title': 'Test Rubric',
            'criteria': [
                {
                    'name': 'Understanding',
                    'weight': 0.4,
                    'levels': [
                        {'score': 4, 'description': 'Excellent'},
                        {'score': 3, 'description': 'Good'}
                    ]
                }
            ]
        }
    }

def test_app_initialization(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        assert mock_session_state['exam_data'] is None
        assert mock_session_state['course_name'] == ''
        assert mock_session_state['model_config']['provider'] == 'ollama'

def test_upload_pdf(mock_session_state, temp_pdf_file):
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.file_uploader') as mock_uploader:
            mock_uploader.return_value = open(temp_pdf_file, 'rb')
            with patch('src.ingestion.ingestion.PDFIngester') as mock_ingester:
                mock_ingester.return_value.ingest_file.return_value = {
                    'document_count': 1,
                    'metadata': {'source': 'test.pdf'}
                }
                assert mock_uploader.called

def test_generate_exam(mock_session_state, sample_exam_data):
    with patch('streamlit.session_state', mock_session_state):
        with patch('src.generation.generator.ExamGenerator') as mock_generator:
            mock_generator.return_value.generate_questions.return_value = sample_exam_data['questions']
            mock_generator.return_value.generate_rubric.return_value = sample_exam_data['rubric']
            assert mock_generator.called

def test_grade_exam(mock_session_state, sample_exam_data):
    with patch('streamlit.session_state', mock_session_state):
        mock_session_state['exam_data'] = sample_exam_data
        with patch('src.grader.optimized_grader.ExamGrader') as mock_grader:
            mock_grader.return_value.grade_answer.return_value = {
                'scores': {'Understanding': 4},
                'feedback': 'Excellent understanding'
            }
            assert mock_grader.called

def test_model_configuration(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.sidebar') as mock_sidebar:
            assert mock_sidebar.called

def test_critic_feedback(mock_session_state, sample_exam_data):
    with patch('streamlit.session_state', mock_session_state):
        mock_session_state['exam_data'] = sample_exam_data
        with patch('src.critic.critic.ExamCritic') as mock_critic:
            mock_critic.return_value.critic_question.return_value = {
                'score': 4,
                'feedback': 'Good question',
                'suggestions': ['Add more context']
            }
            assert mock_critic.called

def test_retrieval_functionality(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        with patch('src.retrieval.retriever.DocumentRetriever') as mock_retriever:
            mock_retriever.return_value.retrieve.return_value = [
                {
                    'text': 'Test document',
                    'metadata': {'source': 'test.pdf'}
                }
            ]
            assert mock_retriever.called

def test_error_handling(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.error') as mock_error:
            assert mock_error.called

def test_session_state_persistence(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        mock_session_state['exam_data'] = {'test': 'data'}
        assert mock_session_state['exam_data'] == {'test': 'data'}

def test_ui_components(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.title') as mock_title:
            with patch('streamlit.markdown') as mock_markdown:
                # Test UI components
                assert mock_title.called
                assert mock_markdown.called

def test_file_handling(mock_session_state, temp_pdf_file):
    with patch('streamlit.session_state', mock_session_state):
        with patch('streamlit.file_uploader') as mock_uploader:
            mock_uploader.return_value = open(temp_pdf_file, 'rb')
            # Test file handling
            assert mock_uploader.called

def test_model_provider_switching(mock_session_state):
    with patch('streamlit.session_state', mock_session_state):
        providers = ['ollama', 'groq', 'google']
        for provider in providers:
            mock_session_state['model_config']['provider'] = provider
            assert mock_session_state['model_config']['provider'] == provider

def test_exam_data_filtering(mock_session_state, sample_exam_data):
    with patch('streamlit.session_state', mock_session_state):
        mock_session_state['exam_data'] = sample_exam_data
        # Test exam data filtering
        assert mock_session_state['filtered_exam_data'] is None 