from typing import List, Dict, Any, Optional
from decimal import Decimal
from src.generation.rubric_types import Rubric, RubricType, Criterion, FeedbackTemplate

class RubricGenerator:
    def __init__(self):
        """Initialize the rubric generator."""
        self._setup_default_templates()
        
    def _setup_default_templates(self):
        """Set up default feedback templates for different rubric types."""
        self.default_templates = {
            RubricType.ASSIGNMENT: FeedbackTemplate(
                positive="Excellent work! The assignment meets or exceeds all requirements.",
                negative="The assignment needs improvement in some areas.",
                suggestions=[
                    "Review the assignment requirements carefully",
                    "Ensure all deliverables are complete",
                    "Check for proper formatting and organization",
                    "Verify that all technical requirements are met"
                ]
            ),
            RubricType.PROJECT: FeedbackTemplate(
                positive="Outstanding project implementation! All aspects are well-executed.",
                negative="The project implementation needs improvement.",
                suggestions=[
                    "Review project requirements and scope",
                    "Ensure all deliverables are complete and properly documented",
                    "Check code quality and organization",
                    "Verify that all technical constraints are met",
                    "Test all functionality thoroughly"
                ]
            ),
            RubricType.CODE_QUALITY: FeedbackTemplate(
                positive="Excellent code quality! The implementation follows best practices.",
                negative="The code quality needs improvement.",
                suggestions=[
                    "Follow consistent coding style",
                    "Add proper documentation and comments",
                    "Implement error handling",
                    "Write unit tests",
                    "Optimize performance where needed"
                ]
            ),
            RubricType.DOCUMENTATION: FeedbackTemplate(
                positive="Excellent documentation! Clear and comprehensive.",
                negative="The documentation needs improvement.",
                suggestions=[
                    "Ensure all sections are complete",
                    "Add more detailed explanations",
                    "Include code examples where appropriate",
                    "Check for proper formatting",
                    "Verify technical accuracy"
                ]
            ),
            RubricType.PRESENTATION: FeedbackTemplate(
                positive="Excellent presentation! Clear and engaging delivery.",
                negative="The presentation needs improvement.",
                suggestions=[
                    "Practice timing and pace",
                    "Improve visual aids",
                    "Add more examples or demonstrations",
                    "Engage with the audience",
                    "Prepare for questions"
                ]
            )
        }

    def generate_assignment_rubric(
        self,
        title: str,
        description: str,
        total_points: Decimal = Decimal('100'),
        custom_criteria: Optional[List[Dict[str, Any]]] = None
    ) -> Rubric:
        """
        Generate a rubric for an assignment.
        
        Args:
            title: Title of the assignment
            description: Description of the assignment
            total_points: Total points possible
            custom_criteria: Optional list of custom criteria
            
        Returns:
            Generated Rubric object
        """
        default_criteria = [
            Criterion(
                name="Completeness",
                description="All required components are present and complete",
                weight=Decimal('0.3'),
                max_score=Decimal('30'),
                levels=[
                    {"description": "Excellent", "feedback": "All components are complete and well-executed"},
                    {"description": "Good", "feedback": "Most components are complete with minor issues"},
                    {"description": "Fair", "feedback": "Some components are missing or incomplete"},
                    {"description": "Poor", "feedback": "Many components are missing or incomplete"}
                ]
            ),
            Criterion(
                name="Quality",
                description="Overall quality of work and attention to detail",
                weight=Decimal('0.3'),
                max_score=Decimal('30'),
                levels=[
                    {"description": "Excellent", "feedback": "Exceptional quality and attention to detail"},
                    {"description": "Good", "feedback": "Good quality with minor issues"},
                    {"description": "Fair", "feedback": "Acceptable quality with some issues"},
                    {"description": "Poor", "feedback": "Poor quality with many issues"}
                ]
            ),
            Criterion(
                name="Technical Accuracy",
                description="Accuracy of technical content and implementation",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Technically accurate with no errors"},
                    {"description": "Good", "feedback": "Mostly accurate with minor errors"},
                    {"description": "Fair", "feedback": "Some technical errors present"},
                    {"description": "Poor", "feedback": "Many technical errors present"}
                ]
            ),
            Criterion(
                name="Documentation",
                description="Quality and completeness of documentation",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Clear, complete, and well-organized documentation"},
                    {"description": "Good", "feedback": "Good documentation with minor issues"},
                    {"description": "Fair", "feedback": "Basic documentation with some issues"},
                    {"description": "Poor", "feedback": "Incomplete or unclear documentation"}
                ]
            )
        ]
        
        criteria = custom_criteria or default_criteria
        return Rubric(
            rubric_type=RubricType.ASSIGNMENT,
            title=title,
            description=description,
            criteria=criteria,
            total_points=total_points,
            feedback_template=self.default_templates[RubricType.ASSIGNMENT]
        )

    def generate_project_rubric(
        self,
        title: str,
        description: str,
        total_points: Decimal = Decimal('100'),
        custom_criteria: Optional[List[Dict[str, Any]]] = None
    ) -> Rubric:
        """
        Generate a rubric for a project.
        
        Args:
            title: Title of the project
            description: Description of the project
            total_points: Total points possible
            custom_criteria: Optional list of custom criteria
            
        Returns:
            Generated Rubric object
        """
        default_criteria = [
            Criterion(
                name="Project Scope",
                description="Completeness of project implementation",
                weight=Decimal('0.25'),
                max_score=Decimal('25'),
                levels=[
                    {"description": "Excellent", "feedback": "Fully implements all requirements"},
                    {"description": "Good", "feedback": "Implements most requirements"},
                    {"description": "Fair", "feedback": "Implements basic requirements"},
                    {"description": "Poor", "feedback": "Fails to implement basic requirements"}
                ]
            ),
            Criterion(
                name="Code Quality",
                description="Quality of code implementation",
                weight=Decimal('0.25'),
                max_score=Decimal('25'),
                levels=[
                    {"description": "Excellent", "feedback": "Clean, efficient, and well-organized code"},
                    {"description": "Good", "feedback": "Good code with minor issues"},
                    {"description": "Fair", "feedback": "Basic code with some issues"},
                    {"description": "Poor", "feedback": "Poor code quality with many issues"}
                ]
            ),
            Criterion(
                name="Documentation",
                description="Quality of project documentation",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Comprehensive and well-organized documentation"},
                    {"description": "Good", "feedback": "Good documentation with minor issues"},
                    {"description": "Fair", "feedback": "Basic documentation with some issues"},
                    {"description": "Poor", "feedback": "Incomplete or unclear documentation"}
                ]
            ),
            Criterion(
                name="Testing",
                description="Quality and coverage of testing",
                weight=Decimal('0.15'),
                max_score=Decimal('15'),
                levels=[
                    {"description": "Excellent", "feedback": "Comprehensive test coverage"},
                    {"description": "Good", "feedback": "Good test coverage with minor gaps"},
                    {"description": "Fair", "feedback": "Basic test coverage"},
                    {"description": "Poor", "feedback": "Minimal or no testing"}
                ]
            ),
            Criterion(
                name="Performance",
                description="Performance and optimization",
                weight=Decimal('0.15'),
                max_score=Decimal('15'),
                levels=[
                    {"description": "Excellent", "feedback": "Optimized and efficient implementation"},
                    {"description": "Good", "feedback": "Good performance with minor issues"},
                    {"description": "Fair", "feedback": "Basic performance"},
                    {"description": "Poor", "feedback": "Poor performance"}
                ]
            )
        ]
        
        criteria = custom_criteria or default_criteria
        return Rubric(
            rubric_type=RubricType.PROJECT,
            title=title,
            description=description,
            criteria=criteria,
            total_points=total_points,
            feedback_template=self.default_templates[RubricType.PROJECT]
        )

    def generate_code_quality_rubric(
        self,
        title: str,
        description: str,
        total_points: Decimal = Decimal('100'),
        custom_criteria: Optional[List[Dict[str, Any]]] = None
    ) -> Rubric:
        """
        Generate a rubric for code quality assessment.
        
        Args:
            title: Title of the code review
            description: Description of the code review
            total_points: Total points possible
            custom_criteria: Optional list of custom criteria
            
        Returns:
            Generated Rubric object
        """
        default_criteria = [
            Criterion(
                name="Code Style",
                description="Adherence to coding standards and style",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Consistent and clean code style"},
                    {"description": "Good", "feedback": "Mostly consistent code style"},
                    {"description": "Fair", "feedback": "Inconsistent code style"},
                    {"description": "Poor", "feedback": "Poor code style"}
                ]
            ),
            Criterion(
                name="Documentation",
                description="Quality of code documentation",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Clear and comprehensive documentation"},
                    {"description": "Good", "feedback": "Good documentation with minor issues"},
                    {"description": "Fair", "feedback": "Basic documentation"},
                    {"description": "Poor", "feedback": "Poor or missing documentation"}
                ]
            ),
            Criterion(
                name="Error Handling",
                description="Quality of error handling and edge cases",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Robust error handling"},
                    {"description": "Good", "feedback": "Good error handling with minor gaps"},
                    {"description": "Fair", "feedback": "Basic error handling"},
                    {"description": "Poor", "feedback": "Poor error handling"}
                ]
            ),
            Criterion(
                name="Testing",
                description="Quality and coverage of unit tests",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Comprehensive test coverage"},
                    {"description": "Good", "feedback": "Good test coverage"},
                    {"description": "Fair", "feedback": "Basic test coverage"},
                    {"description": "Poor", "feedback": "Poor test coverage"}
                ]
            ),
            Criterion(
                name="Performance",
                description="Code efficiency and performance",
                weight=Decimal('0.2'),
                max_score=Decimal('20'),
                levels=[
                    {"description": "Excellent", "feedback": "Optimized and efficient code"},
                    {"description": "Good", "feedback": "Good performance"},
                    {"description": "Fair", "feedback": "Basic performance"},
                    {"description": "Poor", "feedback": "Poor performance"}
                ]
            )
        ]
        
        criteria = custom_criteria or default_criteria
        return Rubric(
            rubric_type=RubricType.CODE_QUALITY,
            title=title,
            description=description,
            criteria=criteria,
            total_points=total_points,
            feedback_template=self.default_templates[RubricType.CODE_QUALITY]
        ) 