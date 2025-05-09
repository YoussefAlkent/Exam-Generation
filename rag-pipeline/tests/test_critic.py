import pytest
from src.critic.critic import ExamCritic
from src.generation.question_types import QuestionTag
from unittest.mock import Mock, patch

@pytest.fixture
def sample_question():
    return {
        'text': 'What is machine learning?',
        'type': QuestionTag.SHORT_ANSWER,
        'difficulty': 'medium',
        'answer': 'Machine learning is a subset of AI that enables systems to learn from data.'
    }

@pytest.fixture
def sample_rubric():
    return {
        'title': 'Machine Learning Question Rubric',
        'criteria': [
            {
                'name': 'Clarity',
                'description': 'Question is clear and unambiguous',
                'weight': 0.3,
                'levels': [
                    {'score': 4, 'description': 'Very clear'},
                    {'score': 3, 'description': 'Clear'},
                    {'score': 2, 'description': 'Somewhat clear'},
                    {'score': 1, 'description': 'Unclear'}
                ]
            },
            {
                'name': 'Difficulty',
                'description': 'Appropriate difficulty level',
                'weight': 0.3,
                'levels': [
                    {'score': 4, 'description': 'Perfect difficulty'},
                    {'score': 3, 'description': 'Good difficulty'},
                    {'score': 2, 'description': 'Too easy/hard'},
                    {'score': 1, 'description': 'Inappropriate difficulty'}
                ]
            }
        ]
    }

def test_critic_initialization(mock_model_factory):
    critic = ExamCritic()
    assert critic is not None
    assert hasattr(critic, 'model')

def test_critic_question(mock_model_factory, mock_llm_response, sample_question):
    critic = ExamCritic()
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        feedback = critic.critic_question(sample_question)
        assert feedback is not None
        assert isinstance(feedback, dict)
        assert 'score' in feedback
        assert 'feedback' in feedback
        assert 'suggestions' in feedback

def test_critic_rubric(mock_model_factory, mock_llm_response, sample_rubric):
    critic = ExamCritic()
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        feedback = critic.critic_rubric(sample_rubric)
        assert feedback is not None
        assert isinstance(feedback, dict)
        assert 'score' in feedback
        assert 'feedback' in feedback
        assert 'suggestions' in feedback

def test_critic_with_invalid_question(mock_model_factory):
    critic = ExamCritic()
    with pytest.raises(ValueError):
        critic.critic_question(None)

def test_critic_with_invalid_rubric(mock_model_factory):
    critic = ExamCritic()
    with pytest.raises(ValueError):
        critic.critic_rubric(None)

def test_critic_with_custom_criteria(mock_model_factory, mock_llm_response, sample_question):
    critic = ExamCritic()
    custom_criteria = ['clarity', 'relevance', 'difficulty']
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        feedback = critic.critic_question(sample_question, criteria=custom_criteria)
        assert feedback is not None
        assert all(criterion in feedback['feedback'] for criterion in custom_criteria)

def test_critic_with_detailed_feedback(mock_model_factory, mock_llm_response, sample_question):
    critic = ExamCritic()
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        feedback = critic.critic_question(sample_question, detailed=True)
        assert feedback is not None
        assert 'detailed_analysis' in feedback
        assert isinstance(feedback['detailed_analysis'], dict)

def test_critic_with_error_handling(mock_model_factory, sample_question):
    critic = ExamCritic()
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.side_effect = Exception("Critic Error")
        with pytest.raises(Exception) as exc_info:
            critic.critic_question(sample_question)
        assert "Critic Error" in str(exc_info.value)

def test_critic_with_different_question_types(mock_model_factory, mock_llm_response):
    critic = ExamCritic()
    question_types = [
        QuestionTag.MULTIPLE_CHOICE,
        QuestionTag.SHORT_ANSWER,
        QuestionTag.ESSAY
    ]
    for q_type in question_types:
        question = {
            'text': 'Test question',
            'type': q_type,
            'difficulty': 'medium',
            'answer': 'Test answer'
        }
        with patch.object(critic.model, 'generate') as mock_generate:
            mock_generate.return_value = mock_llm_response
            feedback = critic.critic_question(question)
            assert feedback is not None
            assert 'score' in feedback

def test_critic_with_difficulty_levels(mock_model_factory, mock_llm_response):
    critic = ExamCritic()
    difficulties = ['easy', 'medium', 'hard']
    for difficulty in difficulties:
        question = {
            'text': 'Test question',
            'type': QuestionTag.SHORT_ANSWER,
            'difficulty': difficulty,
            'answer': 'Test answer'
        }
        with patch.object(critic.model, 'generate') as mock_generate:
            mock_generate.return_value = mock_llm_response
            feedback = critic.critic_question(question)
            assert feedback is not None
            assert 'difficulty_feedback' in feedback['feedback']

def test_critic_with_rubric_validation(mock_model_factory, mock_llm_response, sample_rubric):
    critic = ExamCritic()
    with patch.object(critic.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        feedback = critic.critic_rubric(sample_rubric, validate=True)
        assert feedback is not None
        assert 'validation' in feedback
        assert 'criteria_completeness' in feedback['validation']
        assert 'weight_distribution' in feedback['validation'] 