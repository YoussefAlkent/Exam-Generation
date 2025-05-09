from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal

class RubricType(Enum):
    ASSIGNMENT = "assignment"
    PROJECT = "project"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    PRESENTATION = "presentation"

@dataclass
class Criterion:
    name: str
    description: str
    weight: Decimal
    max_score: Decimal
    levels: List[Dict[str, Any]]  # List of level descriptions and scores

@dataclass
class FeedbackTemplate:
    positive: str
    negative: str
    suggestions: List[str]

class Rubric:
    def __init__(
        self,
        rubric_type: RubricType,
        title: str,
        description: str,
        criteria: List[Criterion],
        total_points: Decimal,
        feedback_template: Optional[FeedbackTemplate] = None,
        tags: List[str] = None
    ):
        self.type = rubric_type
        self.title = title
        self.description = description
        self.criteria = criteria
        self.total_points = total_points
        self.feedback_template = feedback_template
        self.tags = tags or []

    def calculate_score(self, scores: Dict[str, Decimal]) -> Decimal:
        """
        Calculate the total score based on criterion weights and scores.
        
        Args:
            scores: Dictionary mapping criterion names to scores
            
        Returns:
            Total weighted score
        """
        total = Decimal('0')
        for criterion in self.criteria:
            if criterion.name in scores:
                score = scores[criterion.name]
                if score > criterion.max_score:
                    score = criterion.max_score
                weighted_score = (score / criterion.max_score) * criterion.weight
                total += weighted_score
        return total

    def generate_feedback(self, scores: Dict[str, Decimal]) -> Dict[str, Any]:
        """
        Generate feedback based on scores and feedback template.
        
        Args:
            scores: Dictionary mapping criterion names to scores
            
        Returns:
            Dictionary containing feedback for each criterion
        """
        feedback = {}
        for criterion in self.criteria:
            if criterion.name in scores:
                score = scores[criterion.name]
                level_index = int((score / criterion.max_score) * (len(criterion.levels) - 1))
                level = criterion.levels[level_index]
                
                criterion_feedback = {
                    "score": score,
                    "max_score": criterion.max_score,
                    "weight": criterion.weight,
                    "level": level["description"],
                    "feedback": level.get("feedback", "")
                }
                
                if self.feedback_template:
                    if score >= criterion.max_score * Decimal('0.8'):
                        criterion_feedback["template"] = self.feedback_template.positive
                    else:
                        criterion_feedback["template"] = self.feedback_template.negative
                        criterion_feedback["suggestions"] = self.feedback_template.suggestions
                
                feedback[criterion.name] = criterion_feedback
        
        return feedback

    def to_dict(self) -> Dict[str, Any]:
        """Convert rubric to dictionary format."""
        return {
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "criteria": [
                {
                    "name": c.name,
                    "description": c.description,
                    "weight": float(c.weight),
                    "max_score": float(c.max_score),
                    "levels": c.levels
                } for c in self.criteria
            ],
            "total_points": float(self.total_points),
            "feedback_template": {
                "positive": self.feedback_template.positive,
                "negative": self.feedback_template.negative,
                "suggestions": self.feedback_template.suggestions
            } if self.feedback_template else None,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rubric':
        """Create rubric from dictionary format."""
        return cls(
            rubric_type=RubricType(data["type"]),
            title=data["title"],
            description=data["description"],
            criteria=[
                Criterion(
                    name=c["name"],
                    description=c["description"],
                    weight=Decimal(str(c["weight"])),
                    max_score=Decimal(str(c["max_score"])),
                    levels=c["levels"]
                ) for c in data["criteria"]
            ],
            total_points=Decimal(str(data["total_points"])),
            feedback_template=FeedbackTemplate(
                positive=data["feedback_template"]["positive"],
                negative=data["feedback_template"]["negative"],
                suggestions=data["feedback_template"]["suggestions"]
            ) if data.get("feedback_template") else None,
            tags=data.get("tags", [])
        ) 