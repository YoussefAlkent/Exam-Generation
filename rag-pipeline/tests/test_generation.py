import pytest
from src.generation.generator import ExamGenerator

@pytest.fixture
def exam_generator():
    return ExamGenerator()

def test_generate_questions(exam_generator):
    retrieved_content = "This is a sample content for generating questions."
    questions = exam_generator.generate_questions(retrieved_content)
    
    assert isinstance(questions, list)
    assert len(questions) > 0