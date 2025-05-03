import json
import logging
import os
import traceback
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Import the model factory
from ..models.factory import ModelFactory

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
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
        logger.info(f"Initialized ExamGrader with model: {model_name or default_model_name}")
    
    def grade_mcq(self, question: Dict[str, Any], student_answer: str) -> int:
        """Grade a multiple choice question."""
        correct_answer = question["answer"]
        return 100 if student_answer == correct_answer else 0
    
    def grade_fill_in_blank(self, question: Dict[str, Any], student_answer: str) -> int:
        """Grade a fill-in-the-blank question."""
        correct_answer = question["answer"]
        # Basic string match - could be expanded to include fuzzy matching
        return 100 if student_answer.lower() == correct_answer.lower() else 0
    
    def grade_essay_batch(self, question: Dict[str, Any], student_answers: List[Tuple[str, str]], essay_type: str) -> Dict[str, int]:
        """
        Grade multiple student answers for the same essay question in a single API call.
        
        Args:
            question: Question object with correct answer
            student_answers: List of tuples (student_id, answer)
            essay_type: Either "short_essay" or "long_essay"
            
        Returns:
            Dictionary mapping student_ids to scores
        """
        logger.debug(f"Batch grading {essay_type} question for {len(student_answers)} students")
        
        correct_answer = question["answer"]
        question_text = question["question"]
        
        # Create a prompt for the LLM to grade multiple essays at once
        prompt = f"""
        You are grading a {essay_type} question for multiple students. Please evaluate each student's answer 
        compared to the model answer and assign a score from 0 to 100 based on:
        - Content accuracy (60%)
        - Completeness (20%)
        - Clarity and organization (20%)
        
        Question: {question_text}
        
        Model Answer: {correct_answer}
        
        For each student, provide a score from 0-100. Return the results in the following format:
        STUDENT_1: SCORE_1
        STUDENT_2: SCORE_2
        ...and so on.
        
        Here are the student answers:
        """
        
        # Add each student's answer to the prompt
        for i, (student_id, answer) in enumerate(student_answers):
            prompt += f"\nSTUDENT_{student_id}: {answer}\n"
        
        prompt += "\nProvide only the student IDs and scores in the format requested, with no explanation."
        
        # Log the prompt (truncated for readability)
        logger.debug(f"Batch grading prompt: {prompt[:200]}... (truncated)")
        
        try:
            # Get the grading response
            logger.debug(f"Invoking model for batch essay grading of {len(student_answers)} answers")
            response = self.model.invoke(prompt)
            
            # Extract the content from AIMessage object
            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
                
            logger.debug(f"Batch grading response: {response_text[:200]}... (truncated)")
            
            # Parse the response to extract student scores
            scores = {}
            
            # Use regex to find "STUDENT_ID: SCORE" patterns
            pattern = r"STUDENT_([^:]+):\s*(\d+)"
            matches = re.findall(pattern, response_text)
            
            for student_id, score in matches:
                try:
                    score_value = int(score)
                    # Ensure the score is in valid range
                    scores[student_id] = max(0, min(100, score_value))
                except ValueError:
                    logger.warning(f"Could not parse score for student {student_id}: {score}")
                    scores[student_id] = 50
            
            # Check if we found scores for all students
            missing_students = set(sid for sid, _ in student_answers) - set(scores.keys())
            if missing_students:
                logger.warning(f"Missing scores for students: {missing_students}")
                # Assign default scores for missing students
                for student_id in missing_students:
                    scores[student_id] = 50
            
            return scores
                
        except Exception as e:
            logger.error(f"Error during batch essay grading: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return default scores for all students
            return {student_id: 50 for student_id, _ in student_answers}
    
    def grade_essays_by_question(self, exam_data: Dict[str, Any], student_answers_list: List[Dict[str, Any]]) -> Dict[str, Dict[int, int]]:
        """
        Grade all essay questions across all students, batching by question.
        
        Args:
            exam_data: The exam data with questions and correct answers
            student_answers_list: List of student answer dictionaries
            
        Returns:
            Dictionary mapping student IDs to question indices to scores
        """
        # Map of question_idx -> essay_type
        essay_questions = {}
        for i, question in enumerate(exam_data["questions"]):
            if question["type"] in ["short_essay", "long_essay"]:
                essay_questions[i] = question["type"]
        
        if not essay_questions:
            logger.info("No essay questions found in exam")
            return {}
        
        # Group student answers by question index
        question_to_answers = defaultdict(list)
        
        for student_answers in student_answers_list:
            student_id = student_answers.get("Student-ID", "Unknown")
            
            # Create mapping from question index to answer
            answer_map = {
                answer["question_index"]: answer["answer"]
                for answer in student_answers.get("answers", [])
            }
            
            # Add each essay answer to the appropriate question group
            for q_idx in essay_questions:
                if q_idx in answer_map:
                    question_to_answers[q_idx].append((student_id, answer_map[q_idx]))
        
        # Grade each question for all students at once
        all_scores = defaultdict(dict)
        
        for q_idx, answers in question_to_answers.items():
            if not answers:
                continue
            
            question = exam_data["questions"][q_idx]
            essay_type = essay_questions[q_idx]
            
            logger.info(f"Batch grading question {q_idx} for {len(answers)} students")
            
            # Grade this question for all students
            scores = self.grade_essay_batch(question, answers, essay_type)
            
            # Store scores by student ID
            for student_id, score in scores.items():
                all_scores[student_id][q_idx] = score
        
        return all_scores
    
    def grade_multiple_exams_optimized(self, exam_data: Dict[str, Any], student_answer_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade multiple student exams with optimized batching for essay questions.
        
        Args:
            exam_data: The exam data with questions and correct answers
            student_answer_list: List of student answer dictionaries
            
        Returns:
            List of dictionaries with grading results for each student
        """
        logger.info(f"Starting optimized grading for {len(student_answer_list)} students")
        
        # First, grade all essay questions in batches
        essay_scores = self.grade_essays_by_question(exam_data, student_answer_list)
        
        # Now grade the rest of the questions for each student
        results = []
        
        for student_answers in student_answer_list:
            # Extract student information
            student_id = student_answers.get("Student-ID", "Unknown")
            student_name = student_answers.get("Student-Name", "Anonymous")
            
            logger.info(f"Grading non-essay questions for student: {student_id}")
            
            # Create mapping from question index to answer
            answer_map = {
                answer["question_index"]: answer["answer"]
                for answer in student_answers.get("answers", [])
            }
            
            # Grade each question
            question_results = []
            
            for i, question in enumerate(exam_data["questions"]):
                # Skip if student didn't answer this question
                if i not in answer_map:
                    logger.debug(f"No answer provided for question {i}, skipping")
                    continue
                    
                student_answer = answer_map[i]
                question_type = question["type"]
                
                try:
                    # For essay questions, use the pre-computed scores
                    if question_type in ["short_essay", "long_essay"]:
                        if student_id in essay_scores and i in essay_scores[student_id]:
                            score = essay_scores[student_id][i]
                        else:
                            # If we somehow missed this answer in batch processing
                            logger.warning(f"Missing batch score for student {student_id}, question {i}")
                            score = self.grade_essay(question, student_answer, question_type)
                    # For other question types, grade individually
                    elif question_type == "mcq":
                        score = self.grade_mcq(question, student_answer)
                    elif question_type == "fill_in_the_blank":
                        score = self.grade_fill_in_blank(question, student_answer)
                    else:
                        logger.warning(f"Unknown question type: {question_type}")
                        score = 0
                    
                    question_results.append({
                        "question": question["question"],
                        "answer": student_answer,
                        "score": score
                    })
                except Exception as e:
                    logger.error(f"Error grading question {i} for student {student_id}: {e}")
                    question_results.append({
                        "question": question["question"],
                        "answer": student_answer,
                        "score": 0,
                        "error": str(e)
                    })
            
            # Compile student results
            student_result = {
                "Student-ID": student_id,
                "Student-Name": student_name,
                "results": question_results
            }
            
            results.append(student_result)
        
        return results
    
    def grade_from_json_optimized(self, exam_json: Dict[str, Any], answer_json_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade exams directly from uploaded JSON files, with optimized batching.
        
        Args:
            exam_json: The exam questions and answers as a JSON object
            answer_json_list: List of student answer JSON objects
            
        Returns:
            List of dictionaries with grading results for each student
        """
        if "questions" not in exam_json:
            raise ValueError("Invalid exam JSON: 'questions' field is required")
        
        return self.grade_multiple_exams_optimized(exam_json, answer_json_list)
    
    def grade_essay(self, question: Dict[str, Any], student_answer: str, essay_type: str) -> int:
        """
        Grade a single essay question (fallback method).
        
        Args:
            question: Question object with correct answer
            student_answer: Student's answer
            essay_type: Either "short_essay" or "long_essay"
            
        Returns:
            Score from 0 to 100
        """
        logger.debug(f"Fallback grading for {essay_type} question")
        
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
        
        try:
            # Get the grading response
            response = self.model.invoke(prompt)
            
            # Extract the content from AIMessage object
            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Extract the score - assuming the model returns just a number
            try:
                score = int(response_text.strip())
                # Ensure the score is in the valid range
                return max(0, min(100, score))
            except ValueError:
                # If we can't parse the score, try to extract it using some basic parsing
                import re
                score_match = re.search(r'\b(\d{1,3})\b', response_text)
                if score_match:
                    score = int(score_match.group(1))
                    return max(0, min(100, score))
                
                # Default to a middle score if parsing fails
                logger.warning(f"Failed to parse score from response: {response_text}")
                return 50
                
        except Exception as e:
            logger.error(f"Error during essay grading: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 50  # Default score on error
