import json
import logging
import os
import traceback
from typing import Dict, List, Any
from langchain_community.llms import Ollama

# Import the model factory
from ..models.factory import ModelFactory

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
)
logger = logging.getLogger("ExamGrader")

class ExamGrader:
    def __init__(self, model_name: str = None):
        """
        Initialize the exam grader.
        
        Args:
            model_name: Name of the model to use
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
    
    def grade_mcq(self, question: Dict[str, Any], student_answer: str) -> int:
        """Grade a multiple choice question."""
        correct_answer = question["answer"]
        return 1 if student_answer == correct_answer else 0
    
    def grade_fill_in_blank(self, question: Dict[str, Any], student_answer: str) -> int:
        """Grade a fill-in-the-blank question."""
        correct_answer = question["answer"]
        # Basic string match - could be expanded to include fuzzy matching
        return 1 if student_answer.lower() == correct_answer.lower() else 0
    
    def grade_essay(self, question: Dict[str, Any], student_answer: str, essay_type: str) -> int:
        """
        Grade an essay question using semantic similarity.
        
        Args:
            question: Question object with correct answer
            student_answer: Student's answer
            essay_type: Either "short_essay" or "long_essay"
            
        Returns:
            Score from 0 to 100
        """
        logger.debug(f"Starting to grade {essay_type} question: {question['question'][:50]}...")
        
        correct_answer = question["answer"]
        
        # Create a prompt for the LLM to grade the essay
        prompt = f"""
        You are grading a {essay_type} question. Please evaluate the student's answer compared to the model answer.
        Assign a score from 0 to 100 based on:
        - Content accuracy (60%)
        - Completeness (20%)
        - Clarity and organization (20%)
        
        Question: {question["question"]}
        
        Model Answer: {correct_answer}
        
        Student Answer: {student_answer}
        
        Please provide only the numerical score (0-100) with no explanation.
        """
        
        # Log the prompt
        logger.debug(f"Prompt for essay grading: {prompt[:100]}...")
        
        try:
            # Get the grading response
            logger.debug("Invoking model for essay grading...")
            response = self.model.invoke(prompt)
            
            # Log the response object details
            logger.debug(f"Raw response object: {response}")
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response dir: {dir(response)}")
            
            # Extract the content from AIMessage object
            response_text = ""
            if hasattr(response, 'content'):
                # For LangChain's AIMessage objects
                logger.debug("Response has 'content' attribute, extracting...")
                response_text = response.content
            elif hasattr(response, 'text'):
                # Some models might return with .text attribute
                logger.debug("Response has 'text' attribute, extracting...")
                response_text = response.text
            elif isinstance(response, str):
                # Direct string response
                logger.debug("Response is already a string")
                response_text = response
            else:
                # Try to convert to string if all else fails
                logger.debug("Response doesn't have expected attributes, converting to string...")
                response_text = str(response)
                
            logger.debug(f"Extracted text content: '{response_text}'")
            
            # Extract the score - assuming the model returns just a number
            try:
                logger.debug(f"Attempting to parse score from: '{response_text}'")
                score = int(response_text.strip())
                logger.debug(f"Successfully parsed score: {score}")
                # Ensure the score is in the valid range
                final_score = max(0, min(100, score))
                logger.debug(f"Final score (after clamping): {final_score}")
                return final_score
            except ValueError as e:
                # If we can't parse the score, try to extract it using some basic parsing
                logger.warning(f"Failed to parse score directly: {e}")
                import re
                logger.debug("Attempting to extract score using regex...")
                score_match = re.search(r'\b(\d{1,3})\b', response_text)
                if score_match:
                    score = int(score_match.group(1))
                    logger.debug(f"Extracted score using regex: {score}")
                    return max(0, min(100, score))
                
                # Default to a middle score if parsing fails
                logger.warning(f"Could not extract score from response: '{response_text}'")
                return 50
                
        except Exception as e:
            logger.error(f"Error during essay grading: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 50  # Default score on error
    
    def grade_exam(self, exam_data: Dict[str, Any], student_answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Grade a complete exam.
        
        Args:
            exam_data: The exam data with questions and correct answers
            student_answers: The student's answers keyed by question ID
            
        Returns:
            Dictionary with grading results
        """
        logger.info(f"Starting to grade exam for student ID: {student_answers.get('Student-ID', 'Unknown')}")
        results = []
        
        # Extract student information
        student_id = student_answers.get("Student-ID", "Unknown")
        student_name = student_answers.get("Student-Name", "Anonymous")
        
        # Mapping from question index to student answer
        student_answer_map = {answer["question_index"]: answer["answer"] 
                             for answer in student_answers.get("answers", [])}
        
        logger.debug(f"Student answer map contains {len(student_answer_map)} answers")
        
        for i, question in enumerate(exam_data["questions"]):
            logger.debug(f"Processing question {i}: {question['question'][:50]}...")
            
            # Skip if student didn't answer this question
            if i not in student_answer_map:
                logger.debug(f"No answer provided for question {i}, skipping")
                continue
                
            student_answer = student_answer_map[i]
            question_type = question["type"]
            
            logger.debug(f"Grading {question_type} question...")
            
            try:
                # Grade based on question type
                if question_type == "mcq":
                    score = self.grade_mcq(question, student_answer)
                elif question_type == "fill_in_the_blank":
                    score = self.grade_fill_in_blank(question, student_answer)
                elif question_type in ["short_essay", "long_essay"]:
                    score = self.grade_essay(question, student_answer, question_type)
                else:
                    logger.warning(f"Unknown question type: {question_type}")
                    score = 0
                
                logger.debug(f"Question {i} score: {score}")
                
                # Add result
                results.append({
                    "question": question["question"],
                    "answer": student_answer,
                    "score": score
                })
            except Exception as e:
                logger.error(f"Error grading question {i}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    "question": question["question"],
                    "answer": student_answer,
                    "score": 0,
                    "error": str(e)
                })
        
        logger.info(f"Completed grading exam for student {student_id}")
        return {
            "Student-ID": student_id,
            "Student-Name": student_name,
            "results": results
        }
    
    def grade_multiple_exams(self, exam_data: Dict[str, Any], student_answer_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade multiple student exams against the same exam questions.
        
        Args:
            exam_data: The exam data with questions and correct answers
            student_answer_files: List of student answer dictionaries
            
        Returns:
            List of dictionaries with grading results for each student
        """
        results = []
        
        for student_answers in student_answer_files:
            # Verify student data exists
            if "Student-ID" not in student_answers or "Student-Name" not in student_answers:
                logger.warning(f"Student answer file missing required Student-ID or Student-Name fields. Adding default values.")
                # Add default values instead of raising an error
                if "Student-ID" not in student_answers:
                    student_answers["Student-ID"] = "Unknown"
                if "Student-Name" not in student_answers:
                    student_answers["Student-Name"] = "Anonymous"
            
            # Grade the exam
            student_result = self.grade_exam(exam_data, student_answers)
            results.append(student_result)
        
        return results
    
    def grade_from_json(self, exam_json: Dict[str, Any], answer_json_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade exams directly from uploaded JSON files, independent of exam generation.
        
        Args:
            exam_json: The exam questions and answers as a JSON object
            answer_json_list: List of student answer JSON objects
            
        Returns:
            List of dictionaries with grading results for each student
        """
        if "questions" not in exam_json:
            raise ValueError("Invalid exam JSON: 'questions' field is required")
        
        return self.grade_multiple_exams(exam_json, answer_json_list)
        