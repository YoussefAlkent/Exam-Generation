import os
import logging
import json
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models.factory import ModelFactory
from .rubric_types import RubricType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RubricContentAnalyzer")

class RubricContentAnalyzer:
    def __init__(self, model_name: str = None, persist_dir: str = "./chroma_db"):
        """
        Initialize the rubric content analyzer.
        
        Args:
            model_name: Name of the model to use
            persist_dir: Directory where ChromaDB is persisted
        """
        # Initialize model
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
        embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "llama3.2:latest")
        try:
            if model_config["provider"] == "ollama":
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif model_config["provider"] == "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            elif model_config["provider"] == "groq":
                from langchain_groq import GroqEmbeddings
                self.embeddings = GroqEmbeddings(model=embedding_model)
            elif model_config["provider"] == "openai":
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(model=embedding_model)
            else:
                logger.warning(f"Unrecognized provider {model_config['provider']}, falling back to Ollama embeddings")
                self.embeddings = OllamaEmbeddings(model=embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embeddings = OllamaEmbeddings(model="llama2")

    def analyze_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze PDF content to extract rubric-relevant information.
        """
        try:
            logger.info(f"Starting content analysis for PDF: {pdf_path}")
            
            # Load and process the PDF
            with open(pdf_path, 'rb') as file:
                logger.info("PDF file opened successfully")
                
                # Use existing PDFIngester to process the PDF
                from ..ingestion.ingestion import PDFIngester
                ingester = PDFIngester(pdf_dir=os.path.dirname(pdf_path))
                logger.info("PDFIngester initialized")
                
                db = ingester.ingest_to_vectorstore("temp_collection")
                logger.info("Vector store created successfully")
                
                # Extract text chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                logger.info("Text splitter initialized")
                
                # Query the vector store for different aspects
                logger.info("Extracting learning outcomes...")
                learning_outcomes = self._extract_learning_outcomes(db)
                logger.info(f"Found {len(learning_outcomes)} learning outcomes")
                
                logger.info("Extracting assessment criteria...")
                assessment_criteria = self._extract_assessment_criteria(db)
                logger.info(f"Found {len(assessment_criteria)} assessment criteria")
                
                logger.info("Extracting rubric examples...")
                rubric_examples = self._extract_rubric_examples(db)
                logger.info(f"Found {len(rubric_examples)} rubric examples")
                
                logger.info("Extracting project requirements...")
                project_requirements = self._extract_project_requirements(db)
                logger.info(f"Found {len(project_requirements)} project requirements")
                
                # Generate suggested criteria based on the extracted information
                logger.info("Generating suggested criteria...")
                suggested_criteria = self._generate_suggested_criteria(
                    learning_outcomes,
                    assessment_criteria,
                    rubric_examples,
                    project_requirements
                )
                logger.info(f"Generated {len(suggested_criteria)} suggested criteria")
                
                return {
                    "learning_outcomes": learning_outcomes,
                    "assessment_criteria": assessment_criteria,
                    "rubric_examples": rubric_examples,
                    "project_requirements": project_requirements,
                    "suggested_criteria": suggested_criteria
                }
                
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}", exc_info=True)
            return {
                "learning_outcomes": [],
                "assessment_criteria": [],
                "rubric_examples": [],
                "project_requirements": [],
                "suggested_criteria": []
            }

    def _extract_learning_outcomes(self, db: Chroma) -> List[str]:
        """Extract learning outcomes from the content."""
        try:
            logger.info("Starting learning outcomes extraction")
            query = "What are the learning outcomes or objectives of this course/project?"
            results = db.similarity_search(query, k=5)
            logger.info(f"Found {len(results)} relevant chunks")
            
            prompt = f"""
            Based on the following content, extract a list of learning outcomes or objectives.
            Format each outcome as a clear, measurable statement.
            
            Content:
            {results}
            
            Learning Outcomes:
            """
            
            logger.info("Invoking model for learning outcomes")
            response = self.model.invoke(prompt)
            logger.info(f"Model response type: {type(response)}")
            logger.info(f"Model response content: {response}")
            
            parsed_response = self._parse_list_response(response)
            logger.info(f"Parsed {len(parsed_response)} learning outcomes")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error extracting learning outcomes: {str(e)}", exc_info=True)
            return []

    def _extract_assessment_criteria(self, db: Chroma) -> List[str]:
        """Extract assessment criteria from the content."""
        try:
            query = "What are the assessment criteria or evaluation standards?"
            results = db.similarity_search(query, k=5)
            
            prompt = f"""
            Based on the following content, extract a list of assessment criteria or evaluation standards.
            Focus on specific, measurable criteria that can be used for grading.
            
            Content:
            {results}
            
            Assessment Criteria:
            """
            
            response = self.model.invoke(prompt)
            return self._parse_list_response(response)
        except Exception as e:
            logger.error(f"Error extracting assessment criteria: {str(e)}")
            return []

    def _extract_rubric_examples(self, db: Chroma) -> List[Dict[str, Any]]:
        """Extract rubric examples from the content."""
        try:
            query = "Find examples of rubrics, grading criteria, or evaluation matrices"
            results = db.similarity_search(query, k=5)
            
            prompt = f"""
            Based on the following content, extract any examples of rubrics, grading criteria, or evaluation matrices.
            For each example, identify:
            1. The criteria being evaluated
            2. The scoring levels or categories
            3. The descriptions for each level
            
            Content:
            {results}
            
            Rubric Examples:
            """
            
            response = self.model.invoke(prompt)
            return self._parse_rubric_examples(response)
        except Exception as e:
            logger.error(f"Error extracting rubric examples: {str(e)}")
            return []

    def _extract_project_requirements(self, db: Chroma) -> List[str]:
        """Extract project requirements from the content."""
        try:
            query = "What are the project requirements, deliverables, or specifications?"
            results = db.similarity_search(query, k=5)
            
            prompt = f"""
            Based on the following content, extract a list of project requirements, deliverables, or specifications.
            Focus on specific, measurable requirements that can be evaluated.
            
            Content:
            {results}
            
            Project Requirements:
            """
            
            response = self.model.invoke(prompt)
            return self._parse_list_response(response)
        except Exception as e:
            logger.error(f"Error extracting project requirements: {str(e)}")
            return []

    def _generate_suggested_criteria(
        self,
        learning_outcomes: List[str],
        assessment_criteria: List[str],
        rubric_examples: List[Dict[str, Any]],
        project_requirements: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate suggested rubric criteria based on the analyzed content."""
        try:
            print("\n=== Starting _generate_suggested_criteria ===")
            print(f"Input learning outcomes: {learning_outcomes}")
            print(f"Input assessment criteria: {assessment_criteria}")
            print(f"Input project requirements: {project_requirements}")
            
            system_message = """You are a JSON-only response generator. You must ALWAYS return valid JSON.
            Never include any explanatory text, markdown formatting, or other content outside the JSON structure.
            If you need to explain something, include it within the JSON structure as a string value."""
            
            example_json = {
                "criteria": [
                    {
                        "name": "Code Quality",
                        "description": "Evaluation of code style, adherence to best practices, and overall maintainability",
                        "weight": 0.25,
                        "max_score": 10,
                        "levels": [
                            {
                                "description": "Excellent",
                                "feedback": "Code is clean, well-structured, efficient, and adheres strictly to PEP8. Highly maintainable."
                            },
                            {
                                "description": "Good",
                                "feedback": "Code is mostly clean and well-structured, with minor PEP8 violations."
                            },
                            {
                                "description": "Fair",
                                "feedback": "Code is somewhat messy, with noticeable PEP8 violations and maintainability issues."
                            },
                            {
                                "description": "Poor",
                                "feedback": "Code is poorly structured, unreadable, and difficult to maintain; significant PEP8 violations."
                            }
                        ]
                    }
                ]
            }
            
            prompt = f"""
            SYSTEM: {system_message}

            TASK: Generate a rubric in strict JSON format based on the provided learning outcomes, assessment criteria, and project requirements.

            EXAMPLE OUTPUT FORMAT:
            {json.dumps(example_json, indent=2)}

            REQUIRED OUTPUT FORMAT:
            {{
                "criteria": [
                    {{
                        "name": "string (criterion name)",
                        "description": "string (detailed description)",
                        "weight": number (between 0 and 1, representing percentage as decimal)",
                        "max_score": number (maximum points for this criterion)",
                        "levels": [
                            {{
                                "description": "Excellent",
                                "feedback": "string (description of excellent performance)"
                            }},
                            {{
                                "description": "Good",
                                "feedback": "string (description of good performance)"
                            }},
                            {{
                                "description": "Fair",
                                "feedback": "string (description of fair performance)"
                            }},
                            {{
                                "description": "Poor",
                                "feedback": "string (description of poor performance)"
                            }}
                        ]
                    }}
                ]
            }}

            INPUT DATA:
            {{
                "learning_outcomes": {json.dumps(learning_outcomes)},
                "assessment_criteria": {json.dumps(assessment_criteria)},
                "project_requirements": {json.dumps(project_requirements)}
            }}

            REQUIREMENTS:
            1. Generate 3-5 criteria that cover the key aspects of evaluation
            2. Each criterion must have:
               - A clear, specific name
               - A detailed description
               - A weight between 0 and 1 (weights must sum to 1.0)
               - A max_score appropriate for the criterion's importance
               - Four performance levels (Excellent, Good, Fair, Poor) with specific feedback
            3. The output must be valid JSON that can be parsed directly
            4. Do not include any text outside the JSON structure
            5. Do not use markdown formatting or code blocks
            6. Follow the example format exactly

            RESPONSE FORMAT:
            Return ONLY the JSON object, with no additional text, markdown, or formatting.
            The response must be a valid JSON object that can be parsed directly.
            """
            
            print("\n=== Sending Prompt to Model ===")
            print("Prompt:")
            print(prompt)
            
            response = self.model.invoke(prompt)
            print("\n=== Model Response ===")
            print(f"Response type: {type(response)}")
            
            # Handle AIMessage objects
            if hasattr(response, 'content'):
                response_text = response.content
                print("Using response.content")
            elif hasattr(response, 'text'):
                response_text = response.text
                print("Using response.text")
            elif isinstance(response, str):
                response_text = response
                print("Using response as string")
            else:
                response_text = str(response)
                print("Converting response to string")
            
            print("\nRaw response text:")
            print(response_text)
            
            # Try to parse JSON response
            try:
                # Clean the response text to ensure it's valid JSON
                # Remove any markdown code block markers and extra text
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                # Find the first '{' and last '}'
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]
                
                print("\nCleaned response text:")
                print(response_text)
                
                print("\nAttempting to parse JSON...")
                data = json.loads(response_text)
                print(f"Parsed JSON data: {data}")
                
                criteria = data.get('criteria', [])
                print(f"Extracted criteria: {criteria}")
                
                # Validate criteria structure
                validated_criteria = []
                for criterion in criteria:
                    if all(key in criterion for key in ['name', 'description', 'weight', 'max_score', 'levels']):
                        # Ensure weight is between 0 and 1
                        criterion['weight'] = max(0, min(1, float(criterion['weight'])))
                        # Ensure max_score is positive
                        criterion['max_score'] = max(0, float(criterion['max_score']))
                        # Validate levels
                        if len(criterion['levels']) == 4 and all(
                            level.get('description') in ['Excellent', 'Good', 'Fair', 'Poor']
                            for level in criterion['levels']
                        ):
                            validated_criteria.append(criterion)
                
                print(f"\nValidated criteria: {validated_criteria}")
                print("=== Finished _generate_suggested_criteria ===\n")
                return validated_criteria
                
            except json.JSONDecodeError as e:
                print(f"\nJSON parsing error: {str(e)}")
                print(f"Failed to parse text: {response_text}")
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return []
                
        except Exception as e:
            print(f"\nError in _generate_suggested_criteria: {str(e)}")
            logger.error(f"Error generating suggested criteria: {str(e)}")
            return []

    def _parse_list_response(self, response: Any) -> List[str]:
        """Parse a list response from the model."""
        try:
            # Handle AIMessage objects
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Split into lines and clean up
            items = []
            for line in response_text.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Learning Outcomes:', 'Assessment Criteria:', 'Project Requirements:')):
                    # Remove bullet points or numbers
                    line = line.lstrip('•-*1234567890. ')
                    if line:
                        items.append(line)
            return items
        except Exception as e:
            logger.error(f"Error parsing list response: {str(e)}")
            return []

    def _parse_rubric_examples(self, response: str) -> List[Dict[str, Any]]:
        """Parse rubric examples from the model response."""
        # Handle AIMessage objects
        if hasattr(response, 'content'):
            response = response.content
            
        # This is a simplified parser - you might want to make it more robust
        examples = []
        current_example = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Criteria:'):
                if current_example:
                    examples.append(current_example)
                current_example = {'criteria': line.replace('Criteria:', '').strip()}
            elif line.startswith('Levels:'):
                current_example['levels'] = line.replace('Levels:', '').strip()
            elif line.startswith('Description:'):
                current_example['description'] = line.replace('Description:', '').strip()
        
        if current_example:
            examples.append(current_example)
            
        return examples

    def _parse_suggested_criteria(self, response: str) -> List[Dict[str, Any]]:
        """Parse suggested criteria from the model response."""
        print(f"\n=== Starting _parse_suggested_criteria ===")
        print(f"Response type: {type(response)}")
        print(f"Response content: {response}")
        
        # Handle AIMessage objects
        if hasattr(response, 'content'):
            print("Found content attribute, using response.content")
            response = response.content
        elif hasattr(response, 'text'):
            print("Found text attribute, using response.text")
            response = response.text
        elif isinstance(response, str):
            print("Response is already a string")
        else:
            print(f"Converting response to string: {str(response)}")
            response = str(response)
            
        print(f"Processed response type: {type(response)}")
        print(f"Processed response content: {response}")
        
        criteria = []
        current_criterion = {}
        
        print("\nParsing lines...")
        for line in response.split('\n'):
            line = line.strip()
            print(f"Processing line: {line}")
            
            if not line:
                print("Empty line, skipping")
                continue
                
            if line.startswith('Name:'):
                if current_criterion:
                    print(f"Adding criterion: {current_criterion}")
                    criteria.append(current_criterion)
                current_criterion = {'name': line.replace('Name:', '').strip()}
                print(f"New criterion name: {current_criterion['name']}")
            elif line.startswith('Description:'):
                current_criterion['description'] = line.replace('Description:', '').strip()
                print(f"Added description: {current_criterion['description']}")
            elif line.startswith('Weight:'):
                weight_str = line.replace('Weight:', '').strip().rstrip('%')
                print(f"Processing weight: {weight_str}")
                current_criterion['weight'] = float(weight_str) / 100
                print(f"Added weight: {current_criterion['weight']}")
            elif line.startswith('Max Score:'):
                score_str = line.replace('Max Score:', '').strip()
                print(f"Processing max score: {score_str}")
                current_criterion['max_score'] = float(score_str)
                print(f"Added max score: {current_criterion['max_score']}")
            elif line.startswith('Levels:'):
                levels_text = line.replace('Levels:', '').strip()
                print(f"Processing levels: {levels_text}")
                current_criterion['levels'] = self._parse_levels(levels_text)
                print(f"Added levels: {current_criterion['levels']}")
        
        if current_criterion:
            print(f"Adding final criterion: {current_criterion}")
            criteria.append(current_criterion)
            
        print(f"\nFinal criteria list: {criteria}")
        print("=== Finished _parse_suggested_criteria ===\n")
        return criteria

    def _parse_levels(self, levels_text: str) -> List[Dict[str, str]]:
        """Parse performance levels from the model response."""
        print(f"\n=== Starting _parse_levels ===")
        print(f"Levels text type: {type(levels_text)}")
        print(f"Levels text content: {levels_text}")
        
        # Handle AIMessage objects
        if hasattr(levels_text, 'content'):
            print("Found content attribute, using levels_text.content")
            levels_text = levels_text.content
        elif hasattr(levels_text, 'text'):
            print("Found text attribute, using levels_text.text")
            levels_text = levels_text.text
        elif isinstance(levels_text, str):
            print("Levels text is already a string")
        else:
            print(f"Converting levels text to string: {str(levels_text)}")
            levels_text = str(levels_text)
            
        print(f"Processed levels text type: {type(levels_text)}")
        print(f"Processed levels text content: {levels_text}")
        
        levels = []
        current_level = {}
        
        print("\nParsing level lines...")
        for line in levels_text.split('\n'):
            line = line.strip()
            print(f"Processing level line: {line}")
            
            if not line:
                print("Empty line, skipping")
                continue
                
            if line.startswith('Excellent:'):
                if current_level:
                    print(f"Adding level: {current_level}")
                    levels.append(current_level)
                current_level = {'description': 'Excellent', 'feedback': line.replace('Excellent:', '').strip()}
                print(f"New Excellent level: {current_level}")
            elif line.startswith('Good:'):
                if current_level:
                    print(f"Adding level: {current_level}")
                    levels.append(current_level)
                current_level = {'description': 'Good', 'feedback': line.replace('Good:', '').strip()}
                print(f"New Good level: {current_level}")
            elif line.startswith('Fair:'):
                if current_level:
                    print(f"Adding level: {current_level}")
                    levels.append(current_level)
                current_level = {'description': 'Fair', 'feedback': line.replace('Fair:', '').strip()}
                print(f"New Fair level: {current_level}")
            elif line.startswith('Poor:'):
                if current_level:
                    print(f"Adding level: {current_level}")
                    levels.append(current_level)
                current_level = {'description': 'Poor', 'feedback': line.replace('Poor:', '').strip()}
                print(f"New Poor level: {current_level}")
        
        if current_level:
            print(f"Adding final level: {current_level}")
            levels.append(current_level)
            
        print(f"\nFinal levels list: {levels}")
        print("=== Finished _parse_levels ===\n")
        return levels 