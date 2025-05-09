import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.models.factory import ModelFactory
from src.generation.assignment_types import (
    AssignmentType,
    ProjectComplexity,
    Assignment,
    Deliverable,
    TechnicalConstraint,
    PerformanceRequirement
)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class AssignmentGenerator:
    def __init__(self, model_name: str = None, persist_dir: str = "./chroma_db"):
        """
        Initialize the assignment generator.
        
        Args:
            model_name: Name of the model to use
            persist_dir: Directory where ChromaDB is persisted
        """
        # Initialize model with configuration
        default_provider = os.environ.get("DEFAULT_MODEL_PROVIDER", "ollama")
        default_model_name = os.environ.get("DEFAULT_MODEL_NAME", "llama3") 
        default_ollama_url = os.environ.get("DEFAULT_OLLAMA_URL", "http://localhost:11434")
        
        model_config = {
            "provider": default_provider,
            "model_name": model_name or default_model_name,
            "base_url": default_ollama_url
        }
        model_wrapper = ModelFactory.create_model(model_config)
        self.model = model_wrapper.get_model()
        
        self.persist_dir = persist_dir
        
        # Initialize embeddings
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
        self.embedding_model = embedding_model_name if embedding_model_name else "models/gemini-embedding-exp-03-07"
        
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
                logger.warning(f"Unrecognized provider {model_config['provider']}, falling back to Ollama embeddings")
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = OllamaEmbeddings(model="llama2")

    def get_vectorstore(self, course_name: str) -> Chroma:
        """Get the vector store for the given course."""
        return Chroma(
            collection_name=course_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
    
    def retrieve_content(self, course_name: str, query: str = "Summarize the main topics and concepts", k: int = 10) -> str:
        """Retrieve content from the vector store for assignment generation."""
        vectorstore = self.get_vectorstore(course_name)
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_assignment(
        self,
        course_name: str,
        assignment_type: AssignmentType,
        complexity: ProjectComplexity = ProjectComplexity.INTERMEDIATE,
        num_deliverables: int = 3,
        tags: Optional[List[str]] = None
    ) -> Assignment:
        """
        Generate an assignment based on the course content.
        
        Args:
            course_name: Name of the course
            assignment_type: Type of assignment to generate
            complexity: Complexity level of the assignment
            num_deliverables: Number of deliverables to include
            tags: Optional list of tags to filter content
            
        Returns:
            Assignment object with generated content
        """
        content = self.retrieve_content(course_name)
        
        prompt = f"""
        Based on the following course content, generate a {assignment_type.value} assignment with {complexity.value} complexity level.
        
        The assignment should include:
        - A clear title and description
        - {num_deliverables} specific deliverables with descriptions and formats
        - Technical constraints relevant to the course content
        - Performance requirements if applicable
        {f'- Content should be tagged with: {", ".join(tags)}' if tags else ''}
        
        Format the output as a valid JSON object with this structure:
        {{
          "type": "{assignment_type.value}",
          "title": "...",
          "description": "...",
          "deliverables": [
            {{
              "name": "...",
              "description": "...",
              "format": "...",
              "deadline": "..."
            }}
          ],
          "complexity": "{complexity.value}",
          "tags": ["tag1", "tag2"],
          "technical_constraints": [
            {{
              "name": "...",
              "description": "...",
              "required": true
            }}
          ],
          "performance_requirements": [
            {{
              "metric": "...",
              "target": "...",
              "description": "..."
            }}
          ]
        }}
        
        Here is the course content:
        {content}
        
        Make sure the assignment is directly related to the course content and appropriate for the specified complexity level.
        """
        
        response = self.model.invoke(prompt)
        
        try:
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = response
                
            if isinstance(response_text, str):
                assignment_data = json.loads(response_text)
            else:
                assignment_data = response_text
                
            return Assignment.from_dict(assignment_data)
            
        except json.JSONDecodeError:
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
                    assignment_data = json.loads(json_str)
                    return Assignment.from_dict(assignment_data)
                except:
                    pass
            
            # If all else fails, return a basic assignment
            return Assignment(
                assignment_type=assignment_type,
                title=f"{assignment_type.value.title()} Assignment",
                description="Failed to generate detailed assignment. Please try again.",
                deliverables=[
                    Deliverable(
                        name="Basic Deliverable",
                        description="Please regenerate the assignment for detailed deliverables.",
                        format="Any"
                    )
                ],
                complexity=complexity
            ) 