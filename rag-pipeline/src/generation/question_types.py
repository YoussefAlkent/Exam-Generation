from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_BLANK = "fill_blank"
    SHORT_ESSAY = "short_essay"
    LONG_ESSAY = "long_essay"
    CODING = "coding"

class QuestionTag(str, Enum):
    CORE_CONCEPT = "Core Concept"
    APPLICATION = "Application"
    ANALYSIS = "Analysis"
    SYNTHESIS = "Synthesis"
    EVALUATION = "Evaluation"
    CRITICAL_THINKING = "Critical Thinking"

class BaseQuestion(BaseModel):
    type: QuestionType
    question: str
    tags: List[QuestionTag] = []
    difficulty: str = "Intermediate"
    points: int = 10

class MultipleChoiceQuestion(BaseQuestion):
    type: QuestionType = QuestionType.MULTIPLE_CHOICE
    options: List[str]
    correct_answer: int
    explanation: Optional[str] = None

class FillBlankQuestion(BaseQuestion):
    type: QuestionType = QuestionType.FILL_BLANK
    answer: str
    hints: Optional[List[str]] = None

class EssayQuestion(BaseQuestion):
    type: QuestionType
    min_words: int = 100
    max_words: int = 500
    rubric: Optional[Dict[str, Any]] = None

class ShortEssayQuestion(EssayQuestion):
    type: QuestionType = QuestionType.SHORT_ESSAY
    min_words: int = 100
    max_words: int = 300

class LongEssayQuestion(EssayQuestion):
    type: QuestionType = QuestionType.LONG_ESSAY
    min_words: int = 300
    max_words: int = 1000

class CodingQuestion(BaseQuestion):
    type: QuestionType = QuestionType.CODING
    language: str
    test_cases: List[Dict[str, str]]
    solution: Optional[str] = None
    starter_code: Optional[str] = None
    hints: Optional[List[str]] = None
    documentation_requirements: Optional[List[str]] = None

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Question:
    def __init__(
        self,
        question_type: QuestionType,
        question_text: str,
        answer: str,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        tags: List[str] = None,
        choices: List[str] = None,
        code_template: Optional[str] = None,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ):
        self.type = question_type
        self.question_text = question_text
        self.answer = answer
        self.difficulty = difficulty
        self.tags = tags or []
        self.choices = choices
        self.code_template = code_template
        self.test_cases = test_cases

    def to_dict(self) -> Dict[str, Any]:
        question_dict = {
            "type": self.type.value,
            "question": self.question_text,
            "answer": self.answer,
            "difficulty": self.difficulty.value,
            "tags": self.tags
        }
        
        if self.choices:
            question_dict["choices"] = self.choices
            
        if self.code_template:
            question_dict["code_template"] = self.code_template
            
        if self.test_cases:
            question_dict["test_cases"] = self.test_cases
            
        return question_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            question_type=QuestionType(data["type"]),
            question_text=data["question"],
            answer=data["answer"],
            difficulty=DifficultyLevel(data.get("difficulty", "medium")),
            tags=data.get("tags", []),
            choices=data.get("choices"),
            code_template=data.get("code_template"),
            test_cases=data.get("test_cases")
        ) 