import os
import json
import logging
import time
import random
from typing import Dict, Any, List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Import the model factory
from ..models.factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExamCritic")

class ExamCritic:
    def __init__(self, model_name: str = None, persist_dir: str = "./chroma_db"):
        """
        Initialize the exam critic.
        
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
        
        # Initialize embeddings based on the provider
        embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "llama3.2:latest")
        provider = model_config["provider"]
        logger.info(f"Using provider: {provider} with embedding model: {embedding_model}")
        
        try:
            if provider == "ollama":
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif provider == "google":
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_google_genai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif provider == "groq":
                try:
                    from langchain_groq import GroqEmbeddings
                    self.embeddings = GroqEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_groq not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif provider == "openai":
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_openai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            else:
                # Default to Ollama if provider not recognized
                logger.warning(f"Unrecognized provider {provider}, falling back to Ollama embeddings")
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            
            logger.info(f"Successfully initialized {provider} embeddings with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize {provider} embeddings with {embedding_model}: {e}")
            # Fallback to Ollama embeddings
            fallback_model = "llama2"
            logger.info(f"Falling back to Ollama embedding model: {fallback_model}")
            self.embeddings = OllamaEmbeddings(model=fallback_model)

    def get_vectorstore(self, course_name: str) -> Chroma:
        """Get the vector store for the given course."""
        return Chroma(
            collection_name=course_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
    
    def validate_question_relevance(self, course_name: str, question: Dict[str, Any], 
                                    similarity_threshold: float = 0.6, k: int = 3) -> bool:
        """
        Validate the relevance of a question to the course content.
        
        Args:
            course_name: Name of the course
            question: Question object with 'question' key
            similarity_threshold: Minimum similarity threshold
            k: Number of documents to retrieve
            
        Returns:
            Bool indicating if the question is relevant
        """
        # Get the question text
        question_text = question["question"]
        
        # Add retry logic for embedding operations
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vectorstore = self.get_vectorstore(course_name)
                
                # Find similar documents in the vector store
                docs = vectorstore.similarity_search_with_score(question_text, k=k)
                
                # Check if any document has similarity above threshold
                # ChromaDB returns (document, distance) tuples, where lower distance means higher similarity
                # Convert distance to similarity: similarity = 1 - distance
                for doc, distance in docs:
                    similarity = 1 - distance
                    if similarity >= similarity_threshold:
                        return True
                
                return False
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limiting error, apply exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.random()
                    logger.warning(f"Rate limit hit when validating question relevance. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error validating question relevance: {str(e)}")
                    # For rate limiting errors, assume the question is relevant to avoid filtering it out
                    return True

    def filter_questions(self, course_name: str, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out irrelevant questions from the exam.
        
        Args:
            course_name: Name of the course
            exam_data: Exam data with questions
            
        Returns:
            Filtered exam data
        """
        try:
            # Group questions by type
            questions_by_type = {}
            for question in exam_data["questions"]:
                q_type = question["type"]
                if q_type not in questions_by_type:
                    questions_by_type[q_type] = []
                questions_by_type[q_type].append(question)
            
            # Filter questions for each type
            filtered_questions = []
            for q_type, questions in questions_by_type.items():
                # Filter relevant questions
                relevant_questions = [q for q in questions if self.validate_question_relevance(course_name, q)]
                
                # Take top 5 or all if less than 5
                top_questions = relevant_questions[:5]
                filtered_questions.extend(top_questions)
                
                # If we don't have enough questions of this type, use LLM to generate more
                if len(top_questions) < 5:
                    num_additional = 5 - len(top_questions)
                    additional_questions = self.generate_additional_questions(course_name, q_type, num_additional)
                    filtered_questions.extend(additional_questions)
            
            # Return the filtered exam data
            return {
                "course": course_name,
                "questions": filtered_questions
            }
        except Exception as e:
            logger.error(f"Error filtering questions: {str(e)}")
            # Return original exam data in case of error
            return exam_data

    def generate_additional_questions(self, course_name: str, question_type: str, 
                                     num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate additional questions to replace filtered out ones.
        
        Args:
            course_name: Name of the course
            question_type: Type of question to generate
            num_questions: Number of questions to generate
            
        Returns:
            List of additional questions
        """
        # Retrieve content from the vector store
        vectorstore = self.get_vectorstore(course_name)
        docs = vectorstore.similarity_search(f"key concepts about {course_name}", k=5)
        content = "\n\n".join([doc.page_content for doc in docs])
        
        # Define the prompt template
        prompt = f"""
        Based on the following course content, generate {num_questions} {question_type} questions that are highly relevant.
        
        For MCQ questions, include 4 choices and the correct answer.
        For fill-in-the-blank, include the question with a blank and the correct answer.
        For short/long essay questions, include a model answer.
        
        Format the output as a valid JSON array with the appropriate structure for each question type.
        
        Course content: 
        {content}
        """
        
        # Generate the questions
        response = self.model.invoke(prompt)
        
        # Handle different response types (string or AIMessage)
        response_text = response
        if hasattr(response, 'content'):  # Check if it's an AIMessage or similar object
            response_text = response.content
        
        try:
            # Parse the JSON response
            additional_questions = json.loads(response_text)
            return additional_questions[:num_questions]
        except json.JSONDecodeError:
            # If parsing fails, extract JSON from the text
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                try:
                    questions = json.loads(json_str)
                    return questions[:num_questions]
                except:
                    pass
            
            # If all else fails, return empty list
            return []
    
    def evaluate_exam(self, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of an exam.
        
        Args:
            exam_data: The exam data with questions and answers
            
        Returns:
            Dictionary with evaluation results
        """
        # Validate input
        if "questions" not in exam_data or not exam_data["questions"]:
            return {
                "overall_score": 0,
                "feedback": "No questions found in exam data",
                "question_feedback": []
            }
        
        # Group questions by type for evaluation
        questions_by_type = {}
        for question in exam_data["questions"]:
            q_type = question.get("type", "unknown")
            if q_type not in questions_by_type:
                questions_by_type[q_type] = []
            questions_by_type[q_type].append(question)
        
        # Evaluate each question and collect feedback
        question_feedback = []
        
        for q_type, questions in questions_by_type.items():
            for question in questions:
                feedback = self.evaluate_question(question, q_type)
                question_feedback.append(feedback)
        
        # Calculate overall score
        if question_feedback:
            overall_score = sum(feedback["score"] for feedback in question_feedback) / len(question_feedback)
        else:
            overall_score = 0
            
        # Generate overall feedback
        overall_feedback = self.generate_overall_feedback(exam_data, question_feedback, overall_score)
        
        return {
            "overall_score": overall_score,
            "feedback": overall_feedback,
            "question_feedback": question_feedback
        }
    
    def evaluate_question(self, question: Dict[str, Any], question_type: str) -> Dict[str, Any]:
        """
        Evaluate a single question.
        
        Args:
            question: The question object with question text and answer
            question_type: The type of question (mcq, essay, etc.)
            
        Returns:
            Dictionary with score and feedback
        """
        prompt = f"""
        You are evaluating the quality of an exam question. Analyze this {question_type} question and provide feedback.
        
        Question: {question.get("question", "")}
        
        {"Choices: " + str(question.get("choices", [])) if question_type == "mcq" else ""}
        
        Answer: {question.get("answer", "")}
        
        Evaluate the question on:
        1. Clarity (Is the question clearly worded?)
        2. Relevance (Is the question testing important knowledge?)
        3. Difficulty (Is the question appropriate in difficulty?)
        4. Answer quality (Is the answer correct and comprehensive?)
        
        Provide:
        1. A score from 1-10 for the question quality
        2. Brief, specific feedback (max 2 sentences)
        3. One suggestion for improvement
        
        Format your response as a JSON object with keys: "score", "feedback", and "suggestion".
        """
        
        try:
            response = self.model.invoke(prompt)
            
            # Handle different response types (string or AIMessage)
            response_text = response
            if hasattr(response, 'content'):  # Check if it's an AIMessage or similar object
                response_text = response.content
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                # Validate and normalize result
                if "score" not in result:
                    result["score"] = 5  # Default score
                else:
                    result["score"] = min(10, max(1, float(result["score"])))
                
                if "feedback" not in result:
                    result["feedback"] = "No feedback provided"
                
                if "suggestion" not in result:
                    result["suggestion"] = "No suggestion provided"
                
                return {
                    "question": question.get("question", ""),
                    "score": result["score"],
                    "feedback": result["feedback"],
                    "suggestion": result["suggestion"]
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, extract information manually
                lines = response_text.strip().split('\n')
                score_match = None
                feedback = ""
                suggestion = ""
                
                for line in lines:
                    if "score" in line.lower():
                        import re
                        score_match = re.search(r'(\d+(\.\d+)?)', line)
                    elif "feedback" in line.lower():
                        feedback = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    elif "suggestion" in line.lower():
                        suggestion = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                
                score = float(score_match.group(1)) if score_match else 5
                score = min(10, max(1, score))
                
                return {
                    "question": question.get("question", ""),
                    "score": score,
                    "feedback": feedback or "No feedback extracted",
                    "suggestion": suggestion or "No suggestion extracted"
                }
        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            return {
                "question": question.get("question", ""),
                "score": 5,
                "feedback": "Error occurred during evaluation",
                "suggestion": "Review this question manually"
            }
    
    def generate_overall_feedback(self, exam_data: Dict[str, Any], question_feedback: List[Dict[str, Any]], 
                                overall_score: float) -> str:
        """
        Generate overall feedback for the exam.
        
        Args:
            exam_data: The complete exam data
            question_feedback: List of feedback for each question
            overall_score: The calculated overall score
            
        Returns:
            String with overall feedback
        """
        course = exam_data.get("course", "Unknown course")
        question_count = len(exam_data.get("questions", []))
        
        # Count questions by type
        question_types = {}
        for question in exam_data.get("questions", []):
            q_type = question.get("type", "unknown")
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        # Create a summary of question types
        type_summary = ", ".join([f"{count} {q_type}" for q_type, count in question_types.items()])
        
        # Get best and worst questions based on scores
        sorted_feedback = sorted(question_feedback, key=lambda x: x["score"], reverse=True)
        best_questions = sorted_feedback[:3] if len(sorted_feedback) >= 3 else sorted_feedback
        worst_questions = sorted_feedback[-3:] if len(sorted_feedback) >= 3 else sorted_feedback
        
        # Create a prompt for generating feedback
        prompt = f"""
        You are analyzing the quality of an exam for "{course}" with {question_count} questions: {type_summary}.
        
        The overall average score is {overall_score:.1f}/10.
        
        Some strengths (highest-scored questions):
        {json.dumps([q["feedback"] for q in best_questions], indent=2)}
        
        Some areas for improvement (lowest-scored questions):
        {json.dumps([q["feedback"] for q in worst_questions], indent=2)}
        
        Please provide a concise overall assessment of the exam quality (3-5 sentences) with:
        1. General assessment of exam quality
        2. Strengths of the exam
        3. Areas for improvement
        4. Specific recommendations for enhancing the exam
        
        Write in a professional but friendly tone.
        """
        
        try:
            response = self.model.invoke(prompt)
            # Handle different response types (string or AIMessage)
            if hasattr(response, 'content'):  # Check if it's an AIMessage or similar object
                return response.content
            return response
        except Exception as e:
            logger.error(f"Error generating overall feedback: {str(e)}")
            return (f"Exam for '{course}' has an overall quality score of {overall_score:.1f}/10. "
                   f"Review of {question_count} questions suggests varying quality. "
                   "Consider reviewing questions with unclear wording or answers.")
    
    def critique_from_json(self, exam_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critique an exam from directly uploaded JSON without relying on course data.
        
        Args:
            exam_json: The exam data with questions and answers
            
        Returns:
            Dictionary with evaluation results
        """
        if "questions" not in exam_json:
            raise ValueError("Invalid exam JSON: 'questions' field is required")
            
        return self.evaluate_exam(exam_json)