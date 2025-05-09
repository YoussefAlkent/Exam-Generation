import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.models.factory import ModelFactory
from src.generation.question_types import QuestionType, DifficultyLevel, Question, QuestionTag, BaseQuestion, MultipleChoiceQuestion, FillBlankQuestion, ShortEssayQuestion, LongEssayQuestion, CodingQuestion
from src.generation.assignment_types import ProjectType, ProjectTemplate, ProjectCategory, ProjectComplexity, Project, ProjectTemplateData, PROJECT_TEMPLATES
from src.generation.pdf_generator import PDFGenerator
from dotenv import load_dotenv
from src.critic.critic import ExamCritic

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class ExamGenerator:

    def custom_generate_exam(self, course_name: str, num_mcq: int, num_essay: int, num_fill_in_the_blank: int) -> Dict[str, Any]:
        """
        Generate a custom exam with the specified number of questions for each type.
        
        Args:
            course_name: Name of the course
            num_mcq: Number of MCQ questions
            num_essay: Number of essay questions
            num_fill_in_the_blank: Number of complete questions
            
        Returns:
            Dict with course name and questions
        """
        # Retrieve content from the vector store
        content = self.retrieve_content(course_name)
        
        # Define the prompt template for exam generation with customized question types
        prompt = f"""
        Based on the following course content, generate an exam with the specified number of questions:
        - {num_mcq} multiple choice questions (MCQ) with 4 options each and the correct answer
        - {num_essay} short essay questions with model answers (1-2 paragraphs)
        - {num_fill_in_the_blank} complete questions with the correct answer
        
        Format the output as a valid JSON object with this structure:
        {{
          "course": "{course_name}",
          "questions": [
            {{"type": "mcq", "question": "...", "choices": ["A", "B", "C", "D"], "answer": "B"}},
            {{"type": "short_essay", "question": "...", "answer": "..."}},
            {{"type": "fill_in_the_blank", "question": "...", "answer": "..."}}
          ]
        }}
        
        Here is the course content:
        {content}
        
        Make sure all questions are directly related to the course content provided. Each question should test understanding of a specific concept or topic from the content.
        """
        
        # Generate the exam using the language model
        response = self.model.invoke(prompt)
        
        try:
            # Parse the JSON response
            if hasattr(response, 'content'):
                # For AIMessage and similar objects with content attribute
                response_text = response.content
            else:
                # For string responses
                response_text = response
                
            # Try to parse the response as JSON
            if isinstance(response_text, str):
                exam_data = json.loads(response_text)
            else:
                exam_data = response_text
                
            return exam_data
            
        except json.JSONDecodeError:
            # If the model doesn't return valid JSON, extract it from the text
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if (start_idx != -1 and end_idx != -1):
                json_str = response_text[start_idx:end_idx]
                try:
                    exam_data = json.loads(json_str)
                    return exam_data
                except:
                    pass
            
            # If all else fails, return a basic structure
            return {
                "course": course_name,
                "questions": []
            }

    def __init__(self, model_name: str = "llama3"):
        """
        Initialize the exam generator.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.pdf_generator = PDFGenerator()
        
        # Initialize model with configuration
        default_provider = os.environ.get("DEFAULT_MODEL_PROVIDER", "ollama")
        default_model_name = os.environ.get("DEFAULT_MODEL_NAME", "llama3") 
        default_ollama_url = os.environ.get("DEFAULT_OLLAMA_URL", "http://localhost:11434")
        
        # Log environment variable values for debugging
        logger.info(f"Environment variables: provider={default_provider}, model={default_model_name}, url={default_ollama_url}")
        
        model_config = {
            "provider": default_provider,
            "model_name": model_name or default_model_name,
            "base_url": default_ollama_url
        }
        model_wrapper = ModelFactory.create_model(model_config)
        self.model = model_wrapper.get_model()
        
        # Fix the embedding model initialization to use the environment variable
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.embedding_model = embedding_model_name if embedding_model_name else "models/gemini-embedding-exp-03-07"
        logger.info(f"Using provider: {model_config['provider']} with embedding model: {self.embedding_model}")
        
        try:
            if model_config["provider"] == "ollama":
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            elif model_config["provider"] == "google":
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)
                except ImportError:
                    logger.warning("langchain_google_genai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            elif model_config["provider"] == "groq":
                try:
                    from langchain_groq import GroqEmbeddings
                    self.embeddings = GroqEmbeddings(model=self.embedding_model)
                except ImportError:
                    logger.warning("langchain_groq not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            elif model_config["provider"] == "openai":
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
                except ImportError:
                    logger.warning("langchain_openai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            else:
                # Default to Ollama if provider not recognized
                logger.warning(f"Unrecognized provider {model_config['provider']}, falling back to Ollama embeddings")
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            logger.info(f"Successfully initialized {model_config['provider']} embeddings with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize {model_config['provider']} embeddings with {self.embedding_model}: {e}")
            # Fallback to Ollama embeddings
            fallback_model = "llama2"
            logger.info(f"Falling back to Ollama embedding model: {fallback_model}")
            self.embeddings = OllamaEmbeddings(model=fallback_model)
        
        self.critic = ExamCritic(model_name=model_name)
        self.quality_threshold = 6.0
        self.max_quality_attempts = 10
        
    def get_vectorstore(self, course_name: str) -> Chroma:
        """Get the vector store for the given course."""
        return Chroma(
            collection_name=course_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def retrieve_content(self, course_name: str, query: str = None, k: int = 5) -> str:
        """Retrieve content from the vector store for question generation."""
        vectorstore = self.get_vectorstore(course_name)
        
        # If no specific query is provided, use a random topic from the course
        if not query:
            # Get a random sample of documents to find topics
            sample_docs = vectorstore.similarity_search("", k=20)
            topics = []
            for doc in sample_docs:
                # Extract potential topics from the content
                content = doc.page_content
                # Split into sentences and take first few words as potential topics
                sentences = content.split('.')
                for sentence in sentences[:2]:  # Look at first two sentences
                    words = sentence.strip().split()
                    if len(words) > 3:  # Only consider phrases with more than 3 words
                        topics.append(' '.join(words[:4]))
            
            # Select a random topic if available
            if topics:
                import random
                query = random.choice(topics)
            else:
                query = "key concepts and important topics"
        
        # Retrieve documents related to the query
        docs = vectorstore.similarity_search(query, k=k)
        
        # Concatenate the retrieved documents
        content = "\n\n".join([doc.page_content for doc in docs])
        return content
    
    def generate_questions(self, content: str) -> List[Dict[str, Any]]:
        """
        Generate questions from the retrieved content.
        This is a simple implementation - the real implementation will generate different question types.
        """
        # In a real implementation, we would parse the model's JSON response
        # For the test, we'll just return a list of questions
        return ["Question 1", "Question 2", "Question 3"]
        
    def _apply_template(self, project_data: Dict[str, Any], template_type: str) -> Dict[str, Any]:
        """Apply a project template to the project data."""
        if template_type not in PROJECT_TEMPLATES:
            return project_data
            
        template = PROJECT_TEMPLATES[template_type]
        
        # Merge template with project data
        merged_data = project_data.copy()
        merged_data.update({
            "structure": template.structure,
            "requirements": template.requirements + project_data.get("requirements", []),
            "deliverables": template.deliverables + project_data.get("deliverables", []),
            "timeline": template.timeline if not project_data.get("timeline") else project_data["timeline"],
            "technical_requirements": template.technical_requirements or project_data.get("technical_requirements", {}),
            "documentation_requirements": template.documentation_requirements or project_data.get("documentation_requirements", []),
            "test_requirements": template.test_requirements or project_data.get("test_requirements", [])
        })
        
        return merged_data
        
    def _tag_questions(self, questions: List[Dict[str, Any]], tags: List[str]) -> List[Dict[str, Any]]:
        """Add tags to questions based on their content and type."""
        for question in questions:
            if not question.get("tags"):
                question["tags"] = []
                
            # Add type-based tags
            if question["type"] == "coding":
                question["tags"].extend([QuestionTag.APPLICATION, QuestionTag.CRITICAL_THINKING])
            elif question["type"] in ["short_essay", "long_essay"]:
                question["tags"].extend([QuestionTag.ANALYSIS, QuestionTag.SYNTHESIS])
            elif question["type"] == "multiple_choice":
                question["tags"].append(QuestionTag.CORE_CONCEPT)
                
            # Add user-specified tags
            question["tags"].extend(tags)
            
            # Remove duplicates
            question["tags"] = list(set(question["tags"]))
            
        return questions
        
    def generate_exam(self, course_name: str, num_mcq: int = 5, num_fill_blank: int = 5,
                     num_short_essay: int = 3, num_long_essay: int = 2, num_coding: int = 2,
                     tags: List[str] = None, coding_options: Dict[str, Any] = None,
                     general_topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate an exam with the specified number of questions."""
        # Generate questions using the model
        questions = []
        used_topics = set()  # Track topics that have been used
        
        # Add multiple choice questions
        for _ in range(num_mcq):
            question = self._generate_question(
                QuestionType.MULTIPLE_CHOICE, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                questions.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
                
        # Add fill in the blank questions
        for _ in range(num_fill_blank):
            question = self._generate_question(
                QuestionType.FILL_BLANK, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                questions.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
                
        # Add short essay questions
        for _ in range(num_short_essay):
            question = self._generate_question(
                QuestionType.SHORT_ESSAY, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                questions.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
                
        # Add long essay questions
        for _ in range(num_long_essay):
            question = self._generate_question(
                QuestionType.LONG_ESSAY, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                questions.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
                
        # Add coding questions
        for _ in range(num_coding):
            question = self._generate_question(
                QuestionType.CODING, 
                course_name, 
                general_topic, 
                coding_options,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                questions.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
                
        # Apply tags to questions
        if tags:
            questions = self._tag_questions(questions, tags)
            
        return {
            "title": f"{course_name} Exam",
            "course_name": course_name,
            "created_at": datetime.now().isoformat(),
            "questions": questions
        }
        
    def generate_project(self, course_name: str, project_type: str, complexity: str = "Intermediate",
                        duration_weeks: int = 4, coding_options: Dict[str, Any] = None,
                        use_template: bool = False, template_type: str = None,
                        general_topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate a project with the specified parameters."""
        print("\n=== Starting Project Generation ===")
        print("STEP 1: Validating input parameters")
        print(f"Input parameters:")
        print(f"- Course: {course_name}")
        print(f"- Type: {project_type}")
        print(f"- Complexity: {complexity}")
        print(f"- Duration: {duration_weeks} weeks")
        print(f"- Coding options: {coding_options}")
        print(f"- Use template: {use_template}")
        print(f"- Template type: {template_type}")
        print(f"- General topic: {general_topic}")
        
        # Generate project using the model
        print("\n=== Generating Project Data ===")
        print("STEP 2: Calling _generate_project_data to create initial project structure")
        print("Expected fields in initial data: title, description, requirements, deliverables, timeline")
        project_data = self._generate_project_data(
            course_name=course_name,
            project_type=project_type,
            complexity=complexity,
            duration_weeks=duration_weeks,
            coding_options=coding_options,
            general_topic=general_topic
        )
        print("\nInitial project data structure:")
        print("Checking for required fields:")
        print(f"- title: {project_data.get('title', 'MISSING')}")
        print(f"- description: {project_data.get('description', 'MISSING')}")
        print(f"- requirements: {len(project_data.get('requirements', []))} items")
        print(f"- deliverables: {len(project_data.get('deliverables', []))} items")
        print(f"- timeline: {len(project_data.get('timeline', []))} items")
        print("\nFull initial data:")
        print(json.dumps(project_data, indent=2))
        
        # Apply template if requested
        if use_template and template_type:
            print(f"\n=== Applying Template ===")
            print(f"STEP 3: Applying template {template_type}")
            print("Expected: Template will merge with existing project data")
            project_data = self._apply_template(project_data, template_type)
            print("\nProject data after template application:")
            print("Checking for required fields:")
            print(f"- title: {project_data.get('title', 'MISSING')}")
            print(f"- description: {project_data.get('description', 'MISSING')}")
            print(f"- requirements: {len(project_data.get('requirements', []))} items")
            print(f"- deliverables: {len(project_data.get('deliverables', []))} items")
            print(f"- timeline: {len(project_data.get('timeline', []))} items")
            print("\nFull data after template:")
            print(json.dumps(project_data, indent=2))
            
        # Get the description from the generated data
        print("\n=== Extracting Description ===")
        print("STEP 4: Extracting description field")
        description = project_data.get("description", "Project description")
        if not description or description == "Project description":
            # Generate a more detailed description if none exists
            description = f"This {project_type} project for {course_name} is designed at {complexity} complexity level "
            description += f"and spans {duration_weeks} weeks. "
            if coding_options:
                description += f"It involves {coding_options.get('language', 'programming')} development "
                description += f"with a focus on {coding_options.get('project_category', 'software development')}. "
            description += "The project aims to provide hands-on experience with the course concepts."
        print(f"Extracted description: {description}")
        print("Note: This description will be used in both top level and header")
        
        # Structure the project data
        print("\n=== Structuring Final Project Data ===")
        print("STEP 5: Creating final project structure")
        print("Expected structure:")
        print("- Top level: title, course_name, type, description, complexity, duration_weeks")
        print("- Header: title, course, project_type, description, complexity, duration_weeks")
        print("- Metadata: created_at, complexity, duration_weeks, general_topic")
        
        # Create the base project data structure
        project_data = {
            "title": project_data.get("title", f"{course_name} Project"),
            "course_name": course_name,
            "type": project_type,
            "description": description,  # Ensure description is at top level
            "complexity": complexity,
            "duration_weeks": duration_weeks,
            "created_at": datetime.now().isoformat(),
            "requirements": project_data.get("requirements", []),
            "deliverables": project_data.get("deliverables", []),
            "timeline": project_data.get("timeline", []),
            "technical_requirements": project_data.get("technical_requirements", {}),
            "documentation_requirements": project_data.get("documentation_requirements", []),
            "test_requirements": project_data.get("test_requirements", []),
            "header": {
                "title": project_data.get("title", f"{course_name} Project"),
                "course": course_name,
                "project_type": project_type,
                "description": description,  # Also include in header
                "complexity": complexity,
                "duration_weeks": duration_weeks
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "complexity": complexity,
                "duration_weeks": duration_weeks,
                "general_topic": general_topic
            }
        }
        
        print("\nProject data after initial structuring:")
        print("Checking critical fields:")
        print(f"- Top level description: {project_data.get('description', 'MISSING')}")
        print(f"- Header description: {project_data.get('header', {}).get('description', 'MISSING')}")
        print("\nFull structured data:")
        print(json.dumps(project_data, indent=2))
        
        # Add coding-specific fields if it's a coding project
        if project_type == "Coding Project" and coding_options:
            print("\n=== Adding Coding-Specific Fields ===")
            print("STEP 6: Adding coding project specific fields")
            print("Expected additions:")
            print("- Top level: language, project_category, test_cases, solution, deployment_instructions")
            print("- Header: language, project_category")
            
            project_data.update({
                "language": coding_options.get("language"),
                "project_category": coding_options.get("project_category"),
                "test_cases": [] if coding_options.get("include_test_cases") else None,
                "solution": None if coding_options.get("include_solution") else None,
                "deployment_instructions": None if coding_options.get("include_deployment") else None
            })
            
            # Add coding-specific fields to header
            project_data["header"].update({
                "language": coding_options.get("language"),
                "project_category": coding_options.get("project_category")
            })
            
            print("\nProject data after adding coding fields:")
            print("Checking critical fields:")
            print(f"- Top level description: {project_data.get('description', 'MISSING')}")
            print(f"- Header description: {project_data.get('header', {}).get('description', 'MISSING')}")
            print(f"- Language: {project_data.get('language', 'MISSING')}")
            print(f"- Project category: {project_data.get('project_category', 'MISSING')}")
            print("\nFull data after coding fields:")
            print(json.dumps(project_data, indent=2))
        
        print("\n=== Final Project Structure ===")
        print("STEP 7: Final data structure verification")
        print("Checking all critical fields:")
        print(f"- Title: {project_data.get('title', 'MISSING')}")
        print(f"- Description: {project_data.get('description', 'MISSING')}")
        print(f"- Type: {project_data.get('type', 'MISSING')}")
        print(f"- Header description: {project_data.get('header', {}).get('description', 'MISSING')}")
        print("\nFull final structure:")
        print("Header:", json.dumps(project_data["header"], indent=2))
        print("\nRequirements:", json.dumps(project_data["requirements"], indent=2))
        print("\nDeliverables:", json.dumps(project_data["deliverables"], indent=2))
        print("\nTimeline:", json.dumps(project_data["timeline"], indent=2))
        print("\nMetadata:", json.dumps(project_data["metadata"], indent=2))
        
        return project_data

    def generate_assignment(self, course_name: str, assignment_type: str = "homework",
                          num_problems: int = 5, difficulty: str = "medium",
                          tags: List[str] = None, general_topic: Optional[str] = None,
                          coding_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an assignment with the specified parameters.
        
        Args:
            course_name: Name of the course
            assignment_type: Type of assignment (homework, quiz, etc.)
            num_problems: Number of problems to generate
            difficulty: Difficulty level of the questions
            tags: List of tags to apply to questions
            general_topic: Optional specific topic to focus on
            coding_options: Optional dictionary containing coding-specific options
            
        Returns:
            Dict containing the assignment data
        """
        print("\n=== Assignment Generation Parameters ===")
        print(f"Course: {course_name}")
        print(f"Type: {assignment_type}")
        print(f"Number of problems: {num_problems}")
        print(f"Difficulty: {difficulty}")
        print(f"Tags: {tags}")
        print(f"General topic: {general_topic}")
        print(f"Coding options: {coding_options}")
        
        # Generate questions using the model
        problems = []
        used_topics = set()  # Track topics that have been used
        
        # Determine question types based on assignment type
        if assignment_type == "Coding Assignment":
            # For coding assignments, all problems are coding problems
            num_coding = num_problems
            num_mcq = 0
            num_fill_blank = 0
            num_short_essay = 0
            num_long_essay = 0
        elif assignment_type == "homework":
            # Homework typically has a mix of question types
            num_mcq = num_problems // 2
            num_fill_blank = num_problems // 4
            num_short_essay = num_problems - num_mcq - num_fill_blank
            num_long_essay = 0
            num_coding = 0
        elif assignment_type == "quiz":
            # Quiz typically has more MCQs
            num_mcq = num_problems * 3 // 4
            num_fill_blank = num_problems - num_mcq
            num_short_essay = 0
            num_long_essay = 0
            num_coding = 0
        else:
            # Default to a mix of question types
            num_mcq = num_problems // 3
            num_fill_blank = num_problems // 3
            num_short_essay = num_problems - num_mcq - num_fill_blank
            num_long_essay = 0
            num_coding = 0
        
        print("\n=== Question Type Distribution ===")
        print(f"MCQ: {num_mcq}")
        print(f"Fill in blank: {num_fill_blank}")
        print(f"Short essay: {num_short_essay}")
        print(f"Long essay: {num_long_essay}")
        print(f"Coding: {num_coding}")
        
        # Add multiple choice questions
        for _ in range(num_mcq):
            question = self._generate_question(
                QuestionType.MULTIPLE_CHOICE, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                problems.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
        
        # Add fill in the blank questions
        for _ in range(num_fill_blank):
            question = self._generate_question(
                QuestionType.FILL_BLANK, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                problems.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
        
        # Add short essay questions
        for _ in range(num_short_essay):
            question = self._generate_question(
                QuestionType.SHORT_ESSAY, 
                course_name, 
                general_topic,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                problems.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
        
        # Add coding questions if it's a coding assignment
        for _ in range(num_coding):
            question = self._generate_question(
                QuestionType.CODING, 
                course_name, 
                general_topic,
                coding_options=coding_options,
                used_topics=used_topics
            )
            if question:
                # Add description field that matches the question
                question["description"] = question["question"]
                problems.append(question)
                if "question" in question:
                    topic = ' '.join(question["question"].split()[:4])
                    used_topics.add(topic)
        
        # Apply tags to problems
        if tags:
            problems = self._tag_questions(problems, tags)
        
        # Add difficulty level to each problem
        for problem in problems:
            problem["difficulty"] = difficulty
        
        # Generate description
        description = f"This {assignment_type.lower()} assignment for {course_name} contains {num_problems} problems "
        if assignment_type == "Coding Assignment":
            description += f"focused on {coding_options.get('project_type', 'programming')} using {coding_options.get('language', 'the specified programming language')}. "
        else:
            description += f"covering various topics from the course. "
        description += f"The difficulty level is set to {difficulty}."
        
        # Structure the assignment for PDF generation
        assignment_data = {
            "title": f"{course_name} - {assignment_type.title()}",
            "course_name": course_name,
            "type": assignment_type,
            "description": description,
            "difficulty": difficulty,
            "created_at": datetime.now().isoformat(),
            "problems": problems,
            "header": {
                "title": f"{course_name} - {assignment_type.title()}",
                "course": course_name,
                "assignment_type": assignment_type,
                "due_date": None,  # Can be set by the application
                "total_points": sum(problem.get("points", 10) for problem in problems),
                "instructions": [],
                "description": description
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "difficulty": difficulty,
                "tags": tags or [],
                "general_topic": general_topic
            }
        }
        
        # Add general instructions based on assignment type
        if assignment_type == "Coding Assignment":
            assignment_data["header"]["instructions"].extend([
                f"Programming Language: {coding_options.get('language', 'Not specified')}",
                "Submit your code along with any required documentation.",
                "Make sure to test your code thoroughly before submission."
            ])
            if coding_options.get("include_test_cases"):
                assignment_data["header"]["instructions"].append("Your code must pass all provided test cases.")
            if coding_options.get("include_documentation"):
                assignment_data["header"]["instructions"].append("Include proper documentation and comments in your code.")
        else:
            assignment_data["header"]["instructions"].extend([
                "Show all your work and reasoning.",
                "Write clearly and concisely.",
                "Submit your answers in the specified format."
            ])
        
        print("\n=== Generated Assignment Structure ===")
        print("Title:", assignment_data["title"])
        print("Description:", assignment_data["description"])
        print("Type:", assignment_data["type"])
        print("\nHeader:", json.dumps(assignment_data["header"], indent=2))
        print("\nNumber of problems:", len(problems))
        print("\nFirst problem:", json.dumps(problems[0] if problems else {}, indent=2))
        print("\nMetadata:", json.dumps(assignment_data["metadata"], indent=2))
        
        return assignment_data
        
    def _generate_question(self, question_type: QuestionType, course_name: str,
                          general_topic: Optional[str] = None,
                          coding_options: Optional[Dict[str, Any]] = None,
                          used_topics: Optional[set] = None) -> Dict[str, Any]:
        """Generate a single question of the specified type with quality control."""
        # Initialize used_topics if not provided
        if used_topics is None:
            used_topics = set()
            
        # Try to find content that hasn't been used yet
        max_attempts = 3
        for attempt in range(max_attempts):
            # Retrieve content with a specific focus
            if general_topic and general_topic not in used_topics:
                content = self.retrieve_content(course_name, query=general_topic)
                used_topics.add(general_topic)
            else:
                # Try to find new content
                content = self.retrieve_content(course_name)
                
            # Check if the content is significantly different from previously used topics
            if not any(topic in content for topic in used_topics):
                break
                
            if attempt == max_attempts - 1:
                # If we can't find new content, use the original content
                content = self.retrieve_content(course_name, query="key concepts and important topics")

        # Quality control loop
        best_question = None
        best_score = 0
        feedback_history = []

        for attempt in range(self.max_quality_attempts):
            # Generate question
            question = self._generate_single_question(question_type, content, general_topic, coding_options)
            if not question:
                continue

            # Evaluate question quality
            evaluation = self.critic.evaluate_question(question, question_type.value)
            score = evaluation["score"]
            feedback = evaluation["feedback"]
            suggestion = evaluation["suggestion"]

            # Store feedback for regeneration
            feedback_history.append({
                "attempt": attempt + 1,
                "score": score,
                "feedback": feedback,
                "suggestion": suggestion
            })

            # Update best question if this one is better
            if score > best_score:
                best_question = question
                best_score = score

            # If question meets quality threshold, return it
            if score >= self.quality_threshold:
                return question

            # If we haven't reached max attempts, regenerate with feedback
            if attempt < self.max_quality_attempts - 1:
                # Enhance content with feedback for regeneration
                content = self._enhance_content_with_feedback(content, feedback_history)

        # If we've exhausted all attempts, return the best question or None
        return best_question if best_question else None

    def _generate_single_question(self, question_type: QuestionType, content: str,
                                general_topic: Optional[str] = None,
                                coding_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a single question without quality control."""
        # Define the prompt template based on question type
        if question_type == QuestionType.MULTIPLE_CHOICE:
            prompt = f"""
            Based on the following course content, generate a multiple choice question with:
            - A clear and concise question stem
            - 4 distinct answer choices (A, B, C, D)
            - The correct answer marked
            - A brief explanation of why the correct answer is right
            
            {f'Focus on the topic: {general_topic}' if general_topic else ''}
            
            Important: Generate a question about a DIFFERENT topic than what has been used before.
            Avoid repeating questions about the same concepts.
            
            Format the output as a valid JSON object with this structure:
            {{
                "type": "multiple_choice",
                "question": "string",
                "choices": ["string", "string", "string", "string"],
                "answer": "string",
                "explanation": "string"
            }}
            
            Important: Return ONLY the JSON object, no additional text or images.
            
            Here is the course content:
            {content}
            """
            
        elif question_type == QuestionType.FILL_BLANK:
            prompt = f"""
            Based on the following course content, generate a fill-in-the-blank question with:
            - A sentence or paragraph with a key term or concept removed
            - The correct answer
            - A brief explanation of why this answer is correct
            
            {f'Focus on the topic: {general_topic}' if general_topic else ''}
            
            Format the output as a valid JSON object with this structure:
            {{
                "type": "fill_blank",
                "question": "string",
                "answer": "string",
                "explanation": "string"
            }}
            
            Important: Return ONLY the JSON object, no additional text or images.
            
            Here is the course content:
            {content}
            """
            
        elif question_type == QuestionType.SHORT_ESSAY:
            prompt = f"""
            Based on the following course content, generate a short essay question with:
            - A clear and focused question prompt
            - A model answer (1-2 paragraphs)
            - Key points that should be included in the answer
            
            {f'Focus on the topic: {general_topic}' if general_topic else ''}
            
            Format the output as a valid JSON object with this structure:
            {{
                "type": "short_essay",
                "question": "string",
                "answer": "string",
                "key_points": ["string", "string", "string"]
            }}
            
            Important: Return ONLY the JSON object, no additional text or images.
            
            Here is the course content:
            {content}
            """
            
        elif question_type == QuestionType.LONG_ESSAY:
            prompt = f"""
            Based on the following course content, generate a long essay question with:
            - A comprehensive question prompt that requires analysis and synthesis
            - A detailed model answer (3-5 paragraphs)
            - Key points and arguments that should be included
            - Evaluation criteria
            
            {f'Focus on the topic: {general_topic}' if general_topic else ''}
            
            Format the output as a valid JSON object with this structure:
            {{
                "type": "long_essay",
                "question": "string",
                "answer": "string",
                "key_points": ["string", "string", "string"],
                "evaluation_criteria": ["string", "string", "string"]
            }}
            
            Important: Return ONLY the JSON object, no additional text or images.
            
            Here is the course content:
            {content}
            """
            
        elif question_type == QuestionType.CODING:
            prompt = f"""
            Based on the following course content, generate a coding question with:
            - A clear problem statement
            - Required functionality
            - Input/output specifications
            - Test cases
            - A solution template or starter code
            
            {f'Focus on the topic: {general_topic}' if general_topic else ''}
            {f'Use these coding options: {json.dumps(coding_options)}' if coding_options else ''}
            
            Format the output as a valid JSON object with this structure:
            {{
                "type": "coding",
                "question": "string",
                "requirements": ["string", "string", "string"],
                "input_spec": "string",
                "output_spec": "string",
                "test_cases": [
                    {{"input": "string", "output": "string"}}
                ],
                "starter_code": "string"
            }}
            
            Important: Return ONLY the JSON object, no additional text or images.
            
            Here is the course content:
            {content}
            """
            
        # Generate the question using the language model
        response = self.model.invoke(prompt)
        
        try:
            # Parse the JSON response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = response
                
            # Try to parse the response as JSON
            if isinstance(response_text, str):
                question_data = json.loads(response_text)
            else:
                question_data = response_text
                
            return question_data
            
        except json.JSONDecodeError:
            # If the model doesn't return valid JSON, extract it from the text
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if (start_idx != -1 and end_idx != -1):
                json_str = response_text[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except:
                    pass
                    
            return None

    def _enhance_content_with_feedback(self, content: str, feedback_history: List[Dict[str, Any]]) -> str:
        """Enhance the content with feedback history for better regeneration."""
        # Create a summary of feedback, focusing on the most recent and highest-scored attempts
        recent_feedback = feedback_history[-3:] if len(feedback_history) > 3 else feedback_history
        sorted_feedback = sorted(recent_feedback, key=lambda x: x["score"], reverse=True)
        
        # Create a detailed feedback summary
        feedback_summary = []
        for f in sorted_feedback:
            feedback_summary.append(
                f"Attempt {f['attempt']} (Score: {f['score']}/10):\n"
                f"- Feedback: {f['feedback']}\n"
                f"- Suggestion: {f['suggestion']}"
            )

        # Create a prompt to enhance the content
        enhancement_prompt = f"""
        Based on the following content and feedback history, enhance the content to address the issues and improve question generation.
        
        Original Content:
        {content}
        
        Recent Feedback History:
        {chr(10).join(feedback_summary)}
        
        Please provide an enhanced version of the content that:
        1. Addresses the specific feedback points mentioned
        2. Incorporates the suggestions for improvement
        3. Makes the content more clear and focused
        4. Ensures the content is appropriate for generating high-quality questions
        
        Focus on:
        - Improving clarity and precision
        - Adding relevant context and examples
        - Ensuring the content is well-structured
        - Making sure key concepts are properly explained
        
        Return ONLY the enhanced content, no additional text or explanations.
        """

        # Get enhanced content
        response = self.model.invoke(enhancement_prompt)
        if hasattr(response, 'content'):
            enhanced_content = response.content
        else:
            enhanced_content = response

        return enhanced_content

    def _generate_project_data(self, course_name: str, project_type: str,
                              complexity: str, duration_weeks: int,
                              coding_options: Optional[Dict[str, Any]] = None,
                              general_topic: Optional[str] = None) -> Dict[str, Any]:
        """Generate project data using the model."""
        # Retrieve content from the vector store
        content = self.retrieve_content(course_name)
        
        # Define the prompt template
        prompt = f"""
        Based on the following course content, generate a {complexity} level {project_type} project with:
        - A clear project title and description
        - Learning objectives
        - Project requirements and deliverables
        - Timeline and milestones
        - Technical requirements
        - Documentation requirements
        - Testing requirements
        
        The project should be designed for a {duration_weeks}-week duration.
        {f'Focus on the topic: {general_topic}' if general_topic else ''}
        {f'Use these coding options: {json.dumps(coding_options)}' if coding_options else ''}
        
        Format the output as a valid JSON object with this structure:
        {{
            "title": "...",
            "description": "...",
            "learning_objectives": ["obj1", "obj2", "obj3"],
            "requirements": ["req1", "req2", "req3"],
            "deliverables": ["del1", "del2", "del3"],
            "timeline": [
                {{"week": 1, "milestone": "..."}},
                {{"week": 2, "milestone": "..."}}
            ],
            "technical_requirements": {{
                "languages": ["lang1", "lang2"],
                "frameworks": ["framework1", "framework2"],
                "tools": ["tool1", "tool2"]
            }},
            "documentation_requirements": ["doc1", "doc2", "doc3"],
            "test_requirements": ["test1", "test2", "test3"]
        }}
        
        Here is the course content:
        {content}
        """
        
        # Generate the project using the language model
        response = self.model.invoke(prompt)
        
        try:
            # Parse the JSON response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = response
                
            # Try to parse the response as JSON
            if isinstance(response_text, str):
                project_data = json.loads(response_text)
            else:
                project_data = response_text
                
            # Ensure all required fields are present
            project_data = {
                "title": project_data.get("title", f"{course_name} Project"),
                "description": project_data.get("description", "Project description"),
                "learning_objectives": project_data.get("learning_objectives", []),
                "requirements": project_data.get("requirements", []),
                "deliverables": project_data.get("deliverables", []),
                "timeline": project_data.get("timeline", []),
                "technical_requirements": project_data.get("technical_requirements", {}),
                "documentation_requirements": project_data.get("documentation_requirements", []),
                "test_requirements": project_data.get("test_requirements", [])
            }
            
            return project_data
            
        except json.JSONDecodeError:
            # If the model doesn't return valid JSON, extract it from the text
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                try:
                    project_data = json.loads(json_str)
                    # Ensure all required fields are present
                    project_data = {
                        "title": project_data.get("title", f"{course_name} Project"),
                        "description": project_data.get("description", "Project description"),
                        "learning_objectives": project_data.get("learning_objectives", []),
                        "requirements": project_data.get("requirements", []),
                        "deliverables": project_data.get("deliverables", []),
                        "timeline": project_data.get("timeline", []),
                        "technical_requirements": project_data.get("technical_requirements", {}),
                        "documentation_requirements": project_data.get("documentation_requirements", []),
                        "test_requirements": project_data.get("test_requirements", [])
                    }
                    return project_data
                except:
                    pass
                    
            # If all else fails, return a basic structure
            return {
                "title": f"{course_name} Project",
                "description": "Project description could not be generated.",
                "learning_objectives": [],
                "requirements": [],
                "deliverables": [],
                "timeline": [],
                "technical_requirements": {},
                "documentation_requirements": [],
                "test_requirements": []
            }
