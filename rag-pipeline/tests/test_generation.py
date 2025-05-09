import pytest
from src.generation.generator import ExamGenerator
from src.generation.question_types import QuestionTag
from src.generation.rubric_types import Rubric, RubricType, Criterion
from unittest.mock import Mock, patch

@pytest.fixture
def exam_generator():
    return ExamGenerator()

def test_generate_questions(exam_generator):
    retrieved_content = "This is a sample content for generating questions."
    questions = exam_generator.generate_questions(retrieved_content)
    
    assert isinstance(questions, list)
    assert len(questions) > 0

def test_exam_generator_initialization(mock_model_factory):
    generator = ExamGenerator()
    assert generator is not None
    assert hasattr(generator, 'model')

def test_generate_questions(mock_model_factory, mock_llm_response):
    generator = ExamGenerator()
    with patch.object(generator.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        questions = generator.generate_questions(
            context="Test context",
            num_questions=3,
            difficulty="medium",
            question_types=[QuestionTag.MULTIPLE_CHOICE]
        )
        assert questions is not None
        assert isinstance(questions, list)
        assert len(questions) == 3

def test_generate_rubric(mock_model_factory, mock_llm_response):
    generator = ExamGenerator()
    with patch.object(generator.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        rubric = generator.generate_rubric(
            question="Test question",
            rubric_type=RubricType.ANALYTICAL
        )
        assert rubric is not None
        assert isinstance(rubric, Rubric)
        assert hasattr(rubric, 'criteria')

def test_generate_with_invalid_parameters(mock_model_factory):
    generator = ExamGenerator()
    with pytest.raises(ValueError):
        generator.generate_questions(
            context="Test context",
            num_questions=0,
            difficulty="invalid",
            question_types=[]
        )

def test_generate_with_custom_criteria(mock_model_factory, mock_llm_response):
    generator = ExamGenerator()
    custom_criteria = [
        Criterion(
            name="Understanding",
            description="Demonstrates understanding of concepts",
            weight=0.4,
            levels=[
                {"score": 4, "description": "Excellent"},
                {"score": 3, "description": "Good"}
            ]
        )
    ]
    with patch.object(generator.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        rubric = generator.generate_rubric(
            question="Test question",
            rubric_type=RubricType.ANALYTICAL,
            custom_criteria=custom_criteria
        )
        assert rubric is not None
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "Understanding"

def test_generate_with_different_question_types(mock_model_factory, mock_llm_response):
    generator = ExamGenerator()
    question_types = [
        QuestionTag.MULTIPLE_CHOICE,
        QuestionTag.SHORT_ANSWER,
        QuestionTag.ESSAY
    ]
    with patch.object(generator.model, 'generate') as mock_generate:
        mock_generate.return_value = mock_llm_response
        questions = generator.generate_questions(
            context="Test context",
            num_questions=3,
            difficulty="medium",
            question_types=question_types
        )
        assert len(questions) == 3
        assert all(q['type'] in [t.value for t in question_types] for q in questions)

def test_generate_with_difficulty_levels(mock_model_factory, mock_llm_response):
    generator = ExamGenerator()
    difficulties = ["easy", "medium", "hard"]
    for difficulty in difficulties:
        with patch.object(generator.model, 'generate') as mock_generate:
            mock_generate.return_value = mock_llm_response
            questions = generator.generate_questions(
                context="Test context",
                num_questions=1,
                difficulty=difficulty,
                question_types=[QuestionTag.MULTIPLE_CHOICE]
            )
            assert len(questions) == 1
            assert questions[0]['difficulty'] == difficulty

def test_generate_with_error_handling(mock_model_factory):
    generator = ExamGenerator()
    with patch.object(generator.model, 'generate') as mock_generate:
        mock_generate.side_effect = Exception("Generation Error")
        with pytest.raises(Exception) as exc_info:
            generator.generate_questions(
                context="Test context",
                num_questions=1,
                difficulty="medium",
                question_types=[QuestionTag.MULTIPLE_CHOICE]
            )
        assert "Generation Error" in str(exc_info.value)