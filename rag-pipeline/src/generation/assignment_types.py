from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from datetime import datetime

class ProjectType(str, Enum):
    RESEARCH = "Research"
    IMPLEMENTATION = "Implementation"
    ANALYSIS = "Analysis"
    DESIGN = "Design"
    CODING = "Coding Project"

class ProjectTemplate(str, Enum):
    BASIC = "Basic"
    ADVANCED = "Advanced"
    CUSTOM = "Custom"

class ProjectCategory(str, Enum):
    WEB_DEV = "Web Development"
    MOBILE_APP = "Mobile App"
    DATA_SCIENCE = "Data Science"
    GAME_DEV = "Game Development"
    SYSTEM_DESIGN = "System Design"
    API_DEV = "API Development"
    DEVOPS = "DevOps"
    SECURITY = "Security"

class ProjectComplexity(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    CAPSTONE = "Capstone"

class ProjectTemplateData(BaseModel):
    name: str
    description: str
    structure: Dict[str, Any]
    requirements: List[str]
    deliverables: List[str]
    timeline: List[Dict[str, Any]]
    technical_requirements: Optional[Dict[str, Any]] = None
    documentation_requirements: Optional[List[str]] = None
    test_requirements: Optional[List[str]] = None

class Project(BaseModel):
    title: str
    description: str
    type: ProjectType
    category: Optional[ProjectCategory] = None
    complexity: ProjectComplexity = ProjectComplexity.INTERMEDIATE
    duration_weeks: int = 4
    requirements: List[str]
    deliverables: List[str]
    timeline: List[Dict[str, Any]]
    technical_requirements: Optional[Dict[str, Any]] = None
    documentation_requirements: Optional[List[str]] = None
    test_requirements: Optional[List[str]] = None
    template: Optional[ProjectTemplateData] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

# Predefined project templates
PROJECT_TEMPLATES = {
    ProjectTemplate.BASIC: ProjectTemplateData(
        name="Basic Project Template",
        description="A basic template for simple projects",
        structure={
            "project_root": {
                "src": {},
                "tests": {},
                "docs": {},
                "README.md": "",
                "requirements.txt": ""
            }
        },
        requirements=[
            "Clear project objectives",
            "Basic documentation",
            "Simple test cases"
        ],
        deliverables=[
            "Source code",
            "Documentation",
            "Test results"
        ],
        timeline=[
            {"week": 1, "description": "Project setup and planning"},
            {"week": 2, "description": "Implementation"},
            {"week": 3, "description": "Testing and documentation"},
            {"week": 4, "description": "Final review and submission"}
        ]
    ),
    ProjectTemplate.ADVANCED: ProjectTemplateData(
        name="Advanced Project Template",
        description="A comprehensive template for complex projects",
        structure={
            "project_root": {
                "src": {
                    "core": {},
                    "utils": {},
                    "api": {},
                    "tests": {},
                    "docs": {
                        "api": {},
                        "user": {},
                        "developer": {}
                    },
                    "config": {},
                    "scripts": {}
                },
                "tests": {
                    "unit": {},
                    "integration": {},
                    "e2e": {}
                },
                "docs": {
                    "api": {},
                    "user": {},
                    "developer": {}
                },
                "README.md": "",
                "requirements.txt": "",
                "setup.py": "",
                "Makefile": "",
                "Dockerfile": "",
                "docker-compose.yml": ""
            }
        },
        requirements=[
            "Detailed project specifications",
            "Comprehensive documentation",
            "Unit and integration tests",
            "CI/CD pipeline",
            "Performance benchmarks"
        ],
        deliverables=[
            "Source code with tests",
            "API documentation",
            "User documentation",
            "Developer documentation",
            "Test results and coverage report",
            "Performance test results",
            "Deployment instructions"
        ],
        timeline=[
            {"week": 1, "description": "Project planning and architecture design"},
            {"week": 2, "description": "Core implementation"},
            {"week": 3, "description": "API development and testing"},
            {"week": 4, "description": "Documentation and additional features"},
            {"week": 5, "description": "Integration testing and bug fixes"},
            {"week": 6, "description": "Performance optimization"},
            {"week": 7, "description": "Final testing and documentation review"},
            {"week": 8, "description": "Project submission and presentation"}
        ],
        technical_requirements={
            "code_quality": {
                "test_coverage": ">= 80%",
                "documentation_coverage": ">= 90%",
                "code_style": "PEP 8 compliant"
            },
            "performance": {
                "response_time": "< 200ms",
                "throughput": "> 1000 requests/second"
            }
        },
        documentation_requirements=[
            "API documentation with examples",
            "User guide with screenshots",
            "Developer guide with architecture diagrams",
            "Deployment guide with environment setup"
        ],
        test_requirements=[
            "Unit tests for all components",
            "Integration tests for API endpoints",
            "End-to-end tests for critical flows",
            "Performance tests under load",
            "Security tests for vulnerabilities"
        ]
    )
}

@dataclass
class Deliverable:
    name: str
    description: str
    format: str
    deadline: Optional[str] = None

@dataclass
class TechnicalConstraint:
    name: str
    description: str
    required: bool = True

@dataclass
class PerformanceRequirement:
    metric: str
    target: str
    description: str

class Assignment:
    def __init__(
        self,
        assignment_type: ProjectType,
        title: str,
        description: str,
        deliverables: List[Deliverable],
        complexity: ProjectComplexity = ProjectComplexity.INTERMEDIATE,
        tags: List[str] = None,
        technical_constraints: List[TechnicalConstraint] = None,
        performance_requirements: List[PerformanceRequirement] = None
    ):
        self.type = assignment_type
        self.title = title
        self.description = description
        self.deliverables = deliverables
        self.complexity = complexity
        self.tags = tags or []
        self.technical_constraints = technical_constraints or []
        self.performance_requirements = performance_requirements or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "deliverables": [
                {
                    "name": d.name,
                    "description": d.description,
                    "format": d.format,
                    "deadline": d.deadline
                } for d in self.deliverables
            ],
            "complexity": self.complexity.value,
            "tags": self.tags,
            "technical_constraints": [
                {
                    "name": c.name,
                    "description": c.description,
                    "required": c.required
                } for c in self.technical_constraints
            ],
            "performance_requirements": [
                {
                    "metric": r.metric,
                    "target": r.target,
                    "description": r.description
                } for r in self.performance_requirements
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Assignment':
        return cls(
            assignment_type=ProjectType(data["type"]),
            title=data["title"],
            description=data["description"],
            deliverables=[
                Deliverable(
                    name=d["name"],
                    description=d["description"],
                    format=d["format"],
                    deadline=d.get("deadline")
                ) for d in data["deliverables"]
            ],
            complexity=ProjectComplexity(data.get("complexity", "intermediate")),
            tags=data.get("tags", []),
            technical_constraints=[
                TechnicalConstraint(
                    name=c["name"],
                    description=c["description"],
                    required=c.get("required", True)
                ) for c in data.get("technical_constraints", [])
            ],
            performance_requirements=[
                PerformanceRequirement(
                    metric=r["metric"],
                    target=r["target"],
                    description=r["description"]
                ) for r in data.get("performance_requirements", [])
            ]
        ) 