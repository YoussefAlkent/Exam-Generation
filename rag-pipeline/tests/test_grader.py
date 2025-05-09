import pytest
from src.grader.optimized_grader import ExamGrader
from src.generation.rubric_types import Rubric, Criterion
from unittest.mock import Mock, patch

@pytest.fixture
def sample_answer():
    return "This is a sample student answer that demonstrates understanding of the concepts."

@pytest.fixture
def sample_rubric():
    return Rubric(
        title="Test Rubric",
        criteria=[
            Criterion(
                name="Understanding",
                description="Demonstrates understanding of concepts",
                weight=0.4,
                levels=[
                    {"score": 4, "description": "Excellent understanding"},
                    {"score": 3, "description": "Good understanding"},
                    {"score": 2, "description": "Fair understanding"},
                    {"score": 1, "description": "Poor understanding"}
                ]
            ),
            Criterion(
                name="Clarity",
                description="Clear and well-structured response",
                weight=0.3,
                levels=[
                    {"score": 4, "description": "Very clear"},
                    {"score": 3, "description": "Clear"},
                    {"score": 2, "description": "Somewhat clear"},
                    {"score": 1, "description": "Unclear"}
                ]
            )
        ]
    )

def test_grader_initialization(mock_model_factory):
    grader = ExamGrader()
    assert grader is not None
    assert hasattr(grader, 'model')

def test_grade_answer(mock_model_factory, mock_llm_response, sample_answer, sample_rubric):
    grader = ExamGrader()
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        result = grader.grade_answer(sample_answer, sample_rubric)
        assert result is not None
        assert isinstance(result, dict)
        assert 'scores' in result
        assert 'feedback' in result
        assert len(result['scores']) == len(sample_rubric.criteria)

def test_grade_with_invalid_rubric(mock_model_factory, sample_answer):
    grader = ExamGrader()
    with pytest.raises(ValueError):
        grader.grade_answer(sample_answer, None)

def test_grade_with_empty_answer(mock_model_factory, sample_rubric):
    grader = ExamGrader()
    with pytest.raises(ValueError):
        grader.grade_answer("", sample_rubric)

def test_grade_with_custom_weights(mock_model_factory, mock_llm_response, sample_answer, sample_rubric):
    grader = ExamGrader()
    custom_weights = {"Understanding": 0.6, "Clarity": 0.4}
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        result = grader.grade_answer(sample_answer, sample_rubric, custom_weights)
        assert result is not None
        assert sum(result['scores'].values()) <= 4.0

def test_grade_with_detailed_feedback(mock_model_factory, mock_llm_response, sample_answer, sample_rubric):
    grader = ExamGrader()
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        result = grader.grade_answer(sample_answer, sample_rubric, detailed_feedback=True)
        assert result is not None
        assert 'detailed_feedback' in result
        assert isinstance(result['detailed_feedback'], dict)

def test_grade_with_error_handling(mock_model_factory, sample_answer, sample_rubric):
    grader = ExamGrader()
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.side_effect = Exception("Grading Error")
        with pytest.raises(Exception) as exc_info:
            grader.grade_answer(sample_answer, sample_rubric)
        assert "Grading Error" in str(exc_info.value)

def test_grade_with_partial_criteria(mock_model_factory, mock_llm_response, sample_answer):
    partial_rubric = Rubric(
        title="Partial Rubric",
        criteria=[
            Criterion(
                name="Understanding",
                description="Demonstrates understanding of concepts",
                weight=1.0,
                levels=[
                    {"score": 4, "description": "Excellent"},
                    {"score": 3, "description": "Good"}
                ]
            )
        ]
    )
    grader = ExamGrader()
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        result = grader.grade_answer(sample_answer, partial_rubric)
        assert result is not None
        assert len(result['scores']) == 1
        assert 'Understanding' in result['scores']

def test_grade_with_numeric_scores(mock_model_factory, mock_llm_response, sample_answer, sample_rubric):
    grader = ExamGrader()
    with patch.object(grader.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        result = grader.grade_answer(sample_answer, sample_rubric)
        assert all(isinstance(score, (int, float)) for score in result['scores'].values())
        assert all(1 <= score <= 4 for score in result['scores'].values()) 