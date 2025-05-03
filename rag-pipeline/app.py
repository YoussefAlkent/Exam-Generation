import os
import json
import streamlit as st
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging before any imports
logging.basicConfig(
    level=logging.ERROR,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('../app.log')  # Also save to a file
    ]
)
logger = logging.getLogger("app")
logger.info("Starting Course Exam Generator application")

# Disable ChromaDB telemetry before any imports that might trigger it
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Import our modules
from src.ingestion.ingestion import PDFIngester
from src.generation.generator import ExamGenerator
from src.critic.critic import ExamCritic
# Import the optimized grader instead of the regular grader
from src.grader.optimized_grader import ExamGrader

# Import the model factory
from src.models.factory import ModelFactory

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Set page configuration
st.set_page_config(
    page_title="Course Exam Generator",
    page_icon="📚",
    layout="wide"
)

# Create necessary directories
os.makedirs("./pdfs", exist_ok=True)
os.makedirs("./chroma_db", exist_ok=True)

# Initialize session state for storing exam data
if "exam_data" not in st.session_state:
    st.session_state.exam_data = None

if "filtered_exam_data" not in st.session_state:
    st.session_state.filtered_exam_data = None

if "course_name" not in st.session_state:
    st.session_state.course_name = ""

# Get default model settings from environment variables
DEFAULT_PROVIDER = os.environ.get("DEFAULT_MODEL_PROVIDER", "ollama")
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "llama3")
DEFAULT_OLLAMA_URL = os.environ.get("DEFAULT_OLLAMA_URL", "http://localhost:11434")

# Get grader-specific settings
GRADER_PLATFORM = os.environ.get("GRADER_PLATFORM", "ollama")
GRADER_MODEL = os.environ.get("GRADER_MODEL", "llama3.2:latest")

if "model_config" not in st.session_state:
    # Default model configuration from environment variables
    if DEFAULT_PROVIDER == "ollama":
        st.session_state.model_config = {
            "provider": DEFAULT_PROVIDER,
            "model_name": DEFAULT_MODEL_NAME,
            "base_url": DEFAULT_OLLAMA_URL
        }
    elif DEFAULT_PROVIDER == "groq":
        st.session_state.model_config = {
            "provider": DEFAULT_PROVIDER,
            "model_name": DEFAULT_MODEL_NAME,
            "api_key": os.environ.get("GROQ_API_KEY", "")
        }
    elif DEFAULT_PROVIDER == "google":
        st.session_state.model_config = {
            "provider": DEFAULT_PROVIDER,
            "model_name": DEFAULT_MODEL_NAME,
            "api_key": os.environ.get("GOOGLE_API_KEY", "")
        }
    else:
        # Fallback to Ollama
        st.session_state.model_config = {
            "provider": "ollama",
            "model_name": "llama3",
            "base_url": "http://localhost:11434"
        }

# Title and description
st.title("📚 Course Exam Generator")
st.markdown("""
This app helps you create exams from your course materials using AI. 
Upload PDF files, generate questions, and grade student responses.
""")

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")
    
    provider = st.selectbox(
        "Select Model Provider",
        options=["ollama", "groq", "google"],
        index=["ollama", "groq", "google"].index(st.session_state.model_config["provider"]),
        help="Choose which LLM provider to use for generation"
    )
    
    # Different model options based on provider
    if provider == "ollama":
        model_name = st.text_input("Model Name", value=st.session_state.model_config.get("model_name", "llama3"))
        base_url = st.text_input("Ollama API URL", value=st.session_state.model_config.get("base_url", "http://localhost:11434"))
        
        # Update model config
        model_config = {
            "provider": provider,
            "model_name": model_name,
            "base_url": base_url
        }
    elif provider == "groq":
        model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
        default_model = st.session_state.model_config.get("model_name", "llama3-70b-8192")
        model_name = st.selectbox(
            "Model Name", 
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0
        )
        api_key = st.text_input("Groq API Key", value=st.session_state.model_config.get("api_key", os.environ.get("GROQ_API_KEY", "")), type="password")
        
        # Update model config
        model_config = {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key
        }
    elif provider == "google":
        model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
        default_model = st.session_state.model_config.get("model_name", "gemini-1.5-pro")
        model_name = st.selectbox(
            "Model Name", 
            options=model_options,
            index=model_options.index(default_model) if default_model in model_options else 0
        )
        api_key = st.text_input("Google API Key", value=st.session_state.model_config.get("api_key", os.environ.get("GOOGLE_API_KEY", "")), type="password")
        
        # Update model config
        model_config = {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key
        }
    
    # Save configuration button
    if st.button("Apply Configuration"):
        st.session_state.model_config = model_config
        st.success(f"Configuration updated for {provider} provider")
        
    # Display current configuration
    st.subheader("Current Configuration")
    st.write(f"Provider: {st.session_state.model_config['provider']}")
    st.write(f"Model: {st.session_state.model_config['model_name']}")

# Create tabs for different functionality
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Upload & Ingest", "🧩 Generate Exam", "🔍 Review Questions", "✅ Grade Exam", "🧐 Critique Exam"])

# Tab 1: Upload and Ingest PDFs
with tab1:
    st.header("Upload and Process PDF Files")
    
    # Option to upload PDFs or use existing folder
    use_existing = st.checkbox("Use existing PDFs in ./pdfs/ folder", value=True)
    
    if not use_existing:
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save uploaded file to pdfs directory
                with open(os.path.join("./pdfs", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"Uploaded {len(uploaded_files)} PDF files to ./pdfs/ folder")
    
    # Input for course name
    course_name = st.text_input("Enter course name", value=st.session_state.course_name)
    
    # Button to start ingestion process
    if st.button("Process PDFs and Create Vector Database"):
        if not course_name:
            st.error("Please enter a course name")
        else:
            st.session_state.course_name = course_name
            
            with st.spinner("Processing PDFs and creating vector database..."):
                try:
                    ingester = PDFIngester(pdf_dir="./pdfs/", persist_dir="./chroma_db")
                    db = ingester.ingest_to_vectorstore(course_name)
                    st.success(f"Successfully processed PDFs and created vector database for course: {course_name}")
                    st.info(f"Collection contains {db._collection.count()} document chunks")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

# Tab 2: Generate Exam
with tab2:
    st.header("Generate Exam Questions")
    
    # Input for course name (consistent with tab 1)
    if not st.session_state.course_name:
        st.session_state.course_name = st.text_input("Enter course name (if not set in Upload tab)", key="tab2_course")
    
    # Add inputs for customizing question counts
    st.subheader("Customize Question Counts")
    col1, col2 = st.columns(2)
    with col1:
        num_mcq = st.number_input("Multiple Choice Questions", min_value=0, max_value=20, value=5, step=1)
        num_fill_blank = st.number_input("Fill in the Blank Questions", min_value=0, max_value=20, value=5, step=1)
    with col2:
        num_short_essay = st.number_input("Short Essay Questions", min_value=0, max_value=20, value=5, step=1)
        num_long_essay = st.number_input("Long Essay Questions", min_value=0, max_value=20, value=5, step=1)
    
    # Button to generate exam
    if st.button("Generate Exam"):
        if not st.session_state.course_name:
            st.error("Please enter a course name")
        else:
            with st.spinner("Generating exam questions..."):
                try:
                    # Use the configured model
                    model_config = st.session_state.model_config
                    generator = ExamGenerator(model_name=model_config.get("model_name", "llama3"))
                    # Pass the question counts to the generator
                    exam_data = generator.generate_exam(
                        st.session_state.course_name,
                        num_mcq=num_mcq,
                        num_fill_blank=num_fill_blank,
                        num_short_essay=num_short_essay,
                        num_long_essay=num_long_essay
                    )
                    
                    if not exam_data or "questions" not in exam_data or not exam_data["questions"]:
                        st.error("Failed to generate questions. Ensure the course material is properly ingested.")
                    else:
                        st.session_state.exam_data = exam_data
                        st.success(f"Successfully generated {len(exam_data['questions'])} questions using {model_config['provider']} model")
                        
                        # Show a sample of questions
                        st.subheader("Sample Questions")
                        for i, q in enumerate(exam_data["questions"][:3]):
                            st.markdown(f"**Question {i+1} ({q['type']})**: {q['question']}")
                        
                        # Option to download the full exam
                        json_str = json.dumps(exam_data, indent=2)
                        st.download_button(
                            label="Download Full Exam JSON",
                            data=json_str,
                            file_name=f"{st.session_state.course_name}_exam.json",
                            mime="application/json",
                        )
                except Exception as e:
                    st.error(f"Error generating exam: {str(e)}")

# Tab 3: Review and Filter Questions
with tab3:
    st.header("Review and Filter Questions")
    
    if st.session_state.exam_data is None:
        st.warning("Please generate an exam first in the 'Generate Exam' tab")
    else:
        # Button to filter questions
        if st.button("Filter Questions for Relevance"):
            with st.spinner("Filtering questions for relevance..."):
                try:
                    # Use the configured model for the critic
                    model_config = st.session_state.model_config
                    critic = ExamCritic(model_name=model_config.get("model_name", "llama3"))
                    st.session_state.filtered_exam_data = critic.filter_questions(
                        st.session_state.course_name, 
                        st.session_state.exam_data
                    )
                    st.success(f"Successfully filtered questions for relevance using {model_config['provider']} model")
                except Exception as e:
                    st.error(f"Error filtering questions: {str(e)}")
        
        # Display questions with expandable details
        exam_data = st.session_state.filtered_exam_data if st.session_state.filtered_exam_data else st.session_state.exam_data
        
        if exam_data:
            # Group questions by type
            question_types = {}
            for q in exam_data["questions"]:
                q_type = q["type"]
                if q_type not in question_types:
                    question_types[q_type] = []
                question_types[q_type].append(q)
            
            # Display questions by type
            for q_type, questions in question_types.items():
                st.subheader(f"{q_type.replace('_', ' ').title()} ({len(questions)})")
                
                for i, q in enumerate(questions):
                    with st.expander(f"Question {i+1}: {q['question'][:100]}..."):
                        st.markdown(f"**Question:** {q['question']}")
                        
                        if q_type == "mcq":
                            st.markdown("**Choices:**")
                            for j, choice in enumerate(q["choices"]):
                                st.markdown(f"- {chr(65+j)}: {choice}")
                            st.markdown(f"**Answer:** {q['answer']}")
                        else:
                            st.markdown(f"**Answer:** {q['answer']}")
            
            # Option to download the exam data
            json_str = json.dumps(exam_data, indent=2)
            st.download_button(
                label="Download Filtered Exam JSON",
                data=json_str,
                file_name=f"{st.session_state.course_name}_filtered_exam.json",
                mime="application/json",
            )

# Tab 4: Grade Exam
with tab4:
    st.header("Grade Student Answers")
    
    # Create subtabs for different grading methods
    grade_tab1, grade_tab2 = st.tabs(["Grade Generated Exam", "Upload & Grade Exams"])
    
    with grade_tab1:
        # Existing functionality for grading against a generated exam
        if st.session_state.exam_data is None and st.session_state.filtered_exam_data is None:
            st.warning("Please generate an exam first in the 'Generate Exam' tab")
        else:
            # Option to upload student answers
            uploaded_answers = st.file_uploader("Upload student answers (JSON format)", 
                                              type="json", key="existing_exam_answers")
            
            if uploaded_answers:
                try:
                    student_answers = json.loads(uploaded_answers.getvalue().decode())
                    st.success("Successfully loaded student answers")
                    
                    # Check for required fields
                    if "Student-ID" not in student_answers:
                        st.warning("Warning: Student-ID not found in the answer file")
                    if "Student-Name" not in student_answers:
                        st.warning("Warning: Student-Name not found in the answer file")
                    if "answers" not in student_answers:
                        st.error("Error: 'answers' field is required in the student answer file")
                    else:
                        # Button to grade exam
                        if st.button("Grade Exam", key="grade_existing_exam"):
                            with st.spinner("Grading exam..."):
                                try:
                                    # Use the configured model for the grader
                                    model_config = st.session_state.model_config
                                    grader = ExamGrader(model_name=model_config.get("model_name", "llama3"))
                                    # Use filtered exam data if available, otherwise use original exam data
                                    exam_data = st.session_state.filtered_exam_data if st.session_state.filtered_exam_data else st.session_state.exam_data
                                    results = grader.grade_exam(exam_data, student_answers)
                                    
                                    # Display results
                                    st.subheader("Grading Results")
                                    
                                    # Display student information
                                    st.markdown(f"**Student ID:** {results.get('Student-ID', 'Unknown')}")
                                    st.markdown(f"**Student Name:** {results.get('Student-Name', 'Anonymous')}")
                                    
                                    # Calculate overall score
                                    total_score = sum(result["score"] for result in results["results"])
                                    max_score = len(results["results"]) * 100  # Assuming max score is 100 for each question
                                    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                    
                                    st.metric("Overall Score", f"{percentage:.1f}%")
                                    
                                    # Display individual question results
                                    for i, result in enumerate(results["results"]):
                                        with st.expander(f"Question {i+1}: {result['question'][:100]}..."):
                                            st.markdown(f"**Question:** {result['question']}")
                                            st.markdown(f"**Student Answer:** {result['answer']}")
                                            st.markdown(f"**Score:** {result['score']}")
                                    
                                    # Option to download results
                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        label="Download Grading Results",
                                        data=json_str,
                                        file_name=f"{results.get('Student-ID', 'unknown')}_grading_results.json",
                                        mime="application/json",
                                    )
                                except Exception as e:
                                    st.error(f"Error grading exam: {str(e)}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in uploaded file")
            else:
                st.info("Please upload a JSON file with student answers")
    
    with grade_tab2:
        # New functionality for independent exam grading
        st.subheader("Independent Exam Grading")
        st.markdown("Upload an exam JSON file and student answer files to grade without generating an exam first.")
        
        # Upload exam file
        uploaded_exam = st.file_uploader("Upload exam JSON file", type="json", key="independent_exam")
        
        # Upload student answer files (multiple)
        uploaded_student_answers = st.file_uploader("Upload student answer JSON files (can select multiple)", 
                                                  type="json", accept_multiple_files=True, key="independent_answers")
        
        if uploaded_exam and uploaded_student_answers:
            try:
                exam_json = json.loads(uploaded_exam.getvalue().decode())
                st.success("Successfully loaded exam file")
                
                if "questions" not in exam_json:
                    st.error("Invalid exam JSON: 'questions' field is required")
                else:
                    # Process each student answer file
                    student_answer_list = []
                    for answer_file in uploaded_student_answers:
                        try:
                            student_answer = json.loads(answer_file.getvalue().decode())
                            
                            if "answers" not in student_answer:
                                st.warning(f"Skipping {answer_file.name} - 'answers' field is required")
                                continue
                            
                            student_answer_list.append(student_answer)
                        except json.JSONDecodeError:
                            st.error(f"Invalid JSON format in answer file {answer_file.name}")
                    
                    if student_answer_list:
                        # Button to grade exams
                        if st.button("Grade Exams", key="grade_independent_exams"):
                            with st.spinner(f"Grading {len(student_answer_list)} student submissions..."):
                                try:
                                    # Use the configured model for the grader
                                    model_config = st.session_state.model_config
                                    grader = ExamGrader(model_name=model_config.get("model_name", "llama3"))
                                    results = grader.grade_from_json_optimized(exam_json, student_answer_list)
                                    
                                    # Display results
                                    st.subheader("Grading Results")
                                    
                                    # Create a summary table
                                    summary_data = []
                                    for result in results:
                                        student_id = result.get("Student-ID", "Unknown")
                                        student_name = result.get("Student-Name", "Anonymous")
                                        
                                        # Calculate score
                                        total_score = sum(question["score"] for question in result["results"])
                                        max_score = len(result["results"]) * 100  # Assuming max score is 100 per question
                                        percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                        
                                        summary_data.append({
                                            "Student ID": student_id,
                                            "Student Name": student_name,
                                            "Score": f"{percentage:.1f}%"
                                        })
                                    
                                    if summary_data:
                                        st.table(summary_data)
                                    
                                    # Show individual results in expandable sections
                                    for i, result in enumerate(results):
                                        student_id = result.get("Student-ID", "Unknown")
                                        student_name = result.get("Student-Name", "Anonymous")
                                        
                                        with st.expander(f"Student: {student_name} ({student_id})"):
                                            # Calculate score again for this display
                                            total_score = sum(question["score"] for question in result["results"])
                                            max_score = len(result["results"]) * 100
                                            percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                            
                                            st.metric("Score", f"{percentage:.1f}%")
                                            
                                            # Show individual questions
                                            for j, question_result in enumerate(result["results"]):
                                                with st.expander(f"Question {j+1}: {question_result['question'][:100]}..."):
                                                    st.markdown(f"**Question:** {question_result['question']}")
                                                    st.markdown(f"**Student Answer:** {question_result['answer']}")
                                                    st.markdown(f"**Score:** {question_result['score']}")
                                    
                                    # Option to download all results
                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        label="Download All Grading Results",
                                        data=json_str,
                                        file_name="all_grading_results.json",
                                        mime="application/json",
                                    )
                                except Exception as e:
                                    st.error(f"Error grading exams: {str(e)}")
                    else:
                        st.warning("No valid student answer files found")
            except json.JSONDecodeError:
                st.error("Invalid JSON format in exam file")
        else:
            st.info("Please upload both an exam JSON file and at least one student answer JSON file")

# Tab 5: Critique Exam
with tab5:
    st.header("Critique Exam")
    st.markdown("Analyze an exam for quality and improvement suggestions.")
    
    # Create subtabs for different critique methods
    critique_tab1, critique_tab2 = st.tabs(["Critique Generated Exam", "Upload & Critique Exam"])
    
    with critique_tab1:
        # Existing functionality for critiquing a generated exam
        if st.session_state.exam_data is None and st.session_state.filtered_exam_data is None:
            st.warning("Please generate an exam first in the 'Generate Exam' tab")
        else:
            # Use the filtered or original exam data
            exam_to_critique = st.session_state.filtered_exam_data if st.session_state.filtered_exam_data else st.session_state.exam_data
            
            # Button to critique exam
            if st.button("Critique Exam", key="critique_existing_exam"):
                with st.spinner("Analyzing exam quality..."):
                    try:
                        # Use the configured model for the critic
                        model_config = st.session_state.model_config
                        critic = ExamCritic(model_name=model_config.get("model_name", None))
                        critique_results = critic.evaluate_exam(exam_to_critique)
                        
                        # Display results
                        st.subheader("Critique Results")
                        
                        # Overall score
                        overall_score = critique_results.get("overall_score", 0)
                        st.metric("Overall Quality Score", f"{overall_score:.1f}/10")
                        
                        # Overall feedback
                        st.markdown("### Overall Feedback")
                        st.markdown(critique_results.get("feedback", "No feedback available"))
                        
                        # Question-by-question feedback
                        st.markdown("### Question Feedback")
                        for i, feedback in enumerate(critique_results.get("question_feedback", [])):
                            with st.expander(f"Question {i+1}: {feedback.get('question', '')[:100]}..."):
                                st.markdown(f"**Question:** {feedback.get('question', '')}")
                                st.markdown(f"**Score:** {feedback.get('score', 0)}/10")
                                st.markdown(f"**Feedback:** {feedback.get('feedback', '')}")
                                st.markdown(f"**Suggestion:** {feedback.get('suggestion', '')}")
                        
                        # Option to download critique results
                        json_str = json.dumps(critique_results, indent=2)
                        st.download_button(
                            label="Download Critique Results",
                            data=json_str,
                            file_name=f"{st.session_state.course_name}_critique_results.json",
                            mime="application/json",
                        )
                    except Exception as e:
                        st.error(f"Error critiquing exam: {str(e)}")
    
    with critique_tab2:
        # New functionality for independent exam critique
        st.subheader("Independent Exam Critique")
        st.markdown("Upload an exam JSON file to critique without generating an exam first.")
        
        # Upload exam file
        uploaded_exam = st.file_uploader("Upload exam JSON file", type="json", key="independent_critique_exam")
        
        if uploaded_exam:
            try:
                exam_json = json.loads(uploaded_exam.getvalue().decode())
                
                if "questions" not in exam_json:
                    st.error("Invalid exam JSON: 'questions' field is required")
                else:
                    # Button to critique exam
                    if st.button("Critique Uploaded Exam", key="critique_independent_exam"):
                        with st.spinner("Analyzing exam quality..."):
                            try:
                                # Use the configured model for the critic
                                model_config = st.session_state.model_config
                                critic = ExamCritic(model_name=model_config.get("model_name", None))
                                critique_results = critic.critique_from_json(exam_json)
                                
                                # Display results
                                st.subheader("Critique Results")
                                
                                # Overall score
                                overall_score = critique_results.get("overall_score", 0)
                                st.metric("Overall Quality Score", f"{overall_score:.1f}/10")
                                
                                # Overall feedback
                                st.markdown("### Overall Feedback")
                                st.markdown(critique_results.get("feedback", "No feedback available"))
                                
                                # Question-by-question feedback
                                st.markdown("### Question Feedback")
                                for i, feedback in enumerate(critique_results.get("question_feedback", [])):
                                    with st.expander(f"Question {i+1}: {feedback.get('question', '')[:100]}..."):
                                        st.markdown(f"**Question:** {feedback.get('question', '')}")
                                        st.markdown(f"**Score:** {feedback.get('score', 0)}/10")
                                        st.markdown(f"**Feedback:** {feedback.get('feedback', '')}")
                                        st.markdown(f"**Suggestion:** {feedback.get('suggestion', '')}")
                                
                                # Option to download critique results
                                json_str = json.dumps(critique_results, indent=2)
                                st.download_button(
                                    label="Download Critique Results",
                                    data=json_str,
                                    file_name="critique_results.json",
                                    mime="application/json",
                                )
                            except Exception as e:
                                st.error(f"Error critiquing exam: {str(e)}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format in uploaded file")
        else:
            st.info("Please upload an exam JSON file to critique")

# Add some styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 Course Exam Generator | Powered by LangChain, Ollama, ChromaDB & Streamlit")