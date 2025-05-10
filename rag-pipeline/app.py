import os
import json
import streamlit as st
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any
from docx import Document

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

from src.generation.rubric_types import Rubric, RubricType, Criterion, FeedbackTemplate
from src.generation.rubric_generator import RubricGenerator
from src.generation.rubric_visualizer import RubricVisualizer
from src.generation.question_types import QuestionTag
from src.generation.assignment_types import ProjectTemplate, ProjectType, ProjectCategory, ProjectComplexity
from src.generation.pdf_generator import PDFGenerator

def display_rubric(rubric: Rubric):
    """Display the current rubric and its components."""
    
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title(rubric.title)
        st.markdown(rubric.description)
    with col2:
        st.metric("Total Points", f"{float(rubric.total_points):.1f}")
        st.metric("Criteria Count", len(rubric.criteria))
    
    # Add download buttons
    st.markdown("### Download Rubric")
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_str = json.dumps(rubric.to_dict(), indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"{rubric.title.lower().replace(' ', '_')}_rubric.json",
            mime="application/json",
            key=f"json_download_{rubric.title.lower().replace(' ', '_')}"
        )
    
    with col2:
        # PDF download
        try:
            # Create a temporary PDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                pdf_path = tmp.name
            
            # Initialize PDF generator
            pdf_generator = PDFGenerator(output_dir="output")
            
            # Prepare rubric data for PDF generation
            rubric_data = {
                "title": rubric.title,
                "description": rubric.description,
                "total_points": float(rubric.total_points),
                "criteria": [
                    {
                        "name": criterion.name,
                        "weight": float(criterion.weight) * 100,
                        "max_score": float(criterion.max_score),
                        "description": criterion.description,
                        "levels": criterion.levels
                    }
                    for criterion in rubric.criteria
                ]
            }
            
            # Generate PDF
            pdf_generator.generate_rubric_pdf(
                rubric_data=rubric_data,
                output_path=pdf_path,
                options={
                    "paper_size": "A4",
                    "include_toc": True,
                    "include_metadata": True
                }
            )
            
            # Create download button
            with open(pdf_path, 'rb') as f:
                st.download_button(
                    label="Download as PDF",
                    data=f.read(),
                    file_name=f"{rubric.title.lower().replace(' ', '_')}_rubric.pdf",
                    mime="application/pdf",
                    key=f"pdf_download_{rubric.title.lower().replace(' ', '_')}"
                )
            
            # Clean up temporary file
            os.unlink(pdf_path)
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            st.error("Failed to generate PDF. Please try again.")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Criteria", "Scoring"])
    
    with tab1:
        display_criteria(rubric)
    
    with tab2:
        display_scoring(rubric)

def display_criteria(rubric: Rubric):
    """Display the rubric criteria in an organized format."""
    st.markdown("### Criteria")
    
    # Add weight adjustment section
    st.markdown("#### Adjust Weights")
    st.markdown("Use the sliders below to adjust the weight of each criterion. The total must equal 100%.")
    
    # Create a container for weight adjustments
    weight_container = st.container()
    
    # Calculate current weights
    current_weights = {criterion.name: float(criterion.weight) * 100 for criterion in rubric.criteria}
    total_weight = sum(current_weights.values())
    
    # Create sliders for each criterion
    new_weights = {}
    with weight_container:
        # Ensure we have at least one criterion
        num_criteria = max(1, len(rubric.criteria))
        # Create columns with equal width
        cols = st.columns([1] * num_criteria)
        
        for i, criterion in enumerate(rubric.criteria):
            with cols[i]:
                new_weight = st.slider(
                    f"{criterion.name} Weight",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_weights[criterion.name],
                    step=1.0,
                    format="%.1f%%"
                )
                new_weights[criterion.name] = new_weight
    
    # Calculate new total
    new_total = sum(new_weights.values())
    
    # Show total and warning if not 100%
    st.metric("Total Weight", f"{new_total:.1f}%")
    if abs(new_total - 100.0) > 0.1:  # Allow for small floating point differences
        st.warning("Total weight must equal 100%. Please adjust the weights.")
    
    # Add apply button if weights have changed
    if new_weights != current_weights:
        if st.button("Apply New Weights", key="apply_weights_button"):
            try:
                # Update weights in the rubric
                for criterion in rubric.criteria:
                    criterion.weight = Decimal(str(new_weights[criterion.name] / 100))
                st.success("Weights updated successfully!")
            except Exception as e:
                logger.error(f"Error updating weights: {str(e)}")
                st.error("Failed to update weights. Please try again.")
    
    st.markdown("---")
    
    # Display criteria as a table
    st.markdown("#### Criteria Table")
    criteria_data = []
    for criterion in rubric.criteria:
        criteria_data.append({
            "Criterion": criterion.name,
            "Weight": f"{float(criterion.weight) * 100:.1f}%",
            "Max Score": f"{float(criterion.max_score):.1f}",
            "Description": criterion.description
        })
    
    st.table(pd.DataFrame(criteria_data))
    
    # Display detailed criteria
    st.markdown("#### Detailed Criteria")
    for i, criterion in enumerate(rubric.criteria, 1):
        with st.expander(f"{i}. {criterion.name}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(criterion.description)
            with col2:
                st.metric("Weight", f"{float(criterion.weight) * 100:.1f}%")
                st.metric("Max Score", f"{float(criterion.max_score):.1f}")
            
            st.markdown("#### Scoring Levels")
            for level in criterion.levels:
                st.markdown(f"**{level['description']}**: {level['feedback']}")

def display_scoring(rubric: Rubric):
    """Display the scoring interface."""
    st.markdown("### Score Entry")
    
    scores: Dict[str, Decimal] = {}
    for criterion in rubric.criteria:
        score = st.slider(
            criterion.name,
            min_value=0.0,
            max_value=float(criterion.max_score),
            value=0.0,
            step=0.5,
            help=criterion.description
        )
        scores[criterion.name] = Decimal(str(score))
    
    if st.button("Calculate Score", key="calculate_score_button"):
        try:
            total_score = rubric.calculate_score(scores)
            feedback = rubric.generate_feedback(scores)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Score", f"{float(total_score):.1f}")
                st.metric("Percentage", f"{(float(total_score) / float(rubric.total_points) * 100):.1f}%")
            
            with col2:
                st.markdown("### Feedback")
                for criterion, feedback_data in feedback.items():
                    with st.expander(criterion):
                        st.markdown(f"**Score**: {float(feedback_data['score']):.1f}")
                        st.markdown(f"**Level**: {feedback_data['level']}")
                        st.markdown(f"**Feedback**: {feedback_data['feedback']}")
                        if 'suggestions' in feedback_data:
                            st.markdown("**Suggestions**:")
                            for suggestion in feedback_data['suggestions']:
                                st.markdown(f"- {suggestion}")
            
            # Save to history
            st.session_state.historical_scores.append(scores)
            
            # Generate visualizations
            generate_visualizations(rubric, scores)
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            st.error("Failed to calculate score. Please try again.")

def generate_visualizations(rubric: Rubric, scores: Dict[str, Decimal]):
    """Generate and display visualizations for the current scores."""
    st.markdown("### Visualizations")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Score chart
            chart_path = st.session_state.rubric_visualizer.generate_score_chart(rubric, scores)
            st.image(chart_path, caption="Score Distribution")
        
        with col2:
            # Performance analytics
            if len(st.session_state.historical_scores) > 1:
                analytics_path, _ = st.session_state.rubric_visualizer.generate_performance_analytics(
                    rubric, st.session_state.historical_scores
                )
                st.image(analytics_path, caption="Performance Analytics")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        st.error("Failed to generate visualizations. Please try again.")

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

# Initialize session state for assignment and project data
if "assignment_data" not in st.session_state:
    st.session_state.assignment_data = None

if "project_data" not in st.session_state:
    st.session_state.project_data = None

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
    if st.button("Apply Configuration", key="apply_config_button"):
        st.session_state.model_config = model_config
        st.success(f"Configuration updated for {provider} provider")
        
    # Display current configuration
    st.subheader("Current Configuration")
    st.write(f"Provider: {st.session_state.model_config['provider']}")
    st.write(f"Model: {st.session_state.model_config['model_name']}")

# Initialize session state for rubric components
if 'rubric_generator' not in st.session_state:
    st.session_state.rubric_generator = RubricGenerator()
if 'rubric_visualizer' not in st.session_state:
    st.session_state.rubric_visualizer = RubricVisualizer()
if 'current_rubric' not in st.session_state:
    st.session_state.current_rubric = None
if 'historical_scores' not in st.session_state:
    st.session_state.historical_scores = []

# Add to session state initialization
if 'question_tags' not in st.session_state:
    st.session_state.question_tags = []
if 'project_template' not in st.session_state:
    st.session_state.project_template = None
if 'pdf_options' not in st.session_state:
    st.session_state.pdf_options = {
        'paper_size': 'A4',
        'include_toc': True,
        'include_metadata': True,
        'compression_level': 0
    }

# Create tabs for different functionality
tab1, tab2, tab3, tab4 = st.tabs(["📝 Upload & Ingest", "🧩 Generate", "✅ Grade & Critique", "📊 Analytics"])

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
    if st.button("Process PDFs and Create Vector Database", key="process_pdfs_button"):
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

# Tab 2: Generate (Exam, Assignment, Project)
with tab2:
    st.header("Generate Content")
    
    # Content type selection
    content_type = st.radio(
        "Select Content Type",
        ["Exam", "Assignment", "Project"],
        horizontal=True
    )
    
    # Input for course name (consistent with tab 1)
    if not st.session_state.course_name:
        st.session_state.course_name = st.text_input("Enter course name (if not set in Upload tab)", key="tab2_course")
    
    # Add general topic field for all content types (optional)
    general_topic = st.text_input(
        "General Topic (Optional)",
        help="Specify the main topic or focus area for the content (e.g., 'Data Structures', 'Web Development', 'Machine Learning'). Leave empty to use all available content from the vector database."
    )
    
    if content_type == "Exam":
        # Exam generation options
        st.subheader("Exam Generation Options")
        col1, col2 = st.columns(2)
        with col1:
            num_mcq = st.number_input("Multiple Choice Questions", min_value=0, max_value=20, value=5, step=1)
            num_fill_blank = st.number_input("Fill in the Blank Questions", min_value=0, max_value=20, value=5, step=1)
            num_coding = st.number_input("Coding Questions", min_value=0, max_value=20, value=3, step=1)
        with col2:
            num_short_essay = st.number_input("Short Essay Questions", min_value=0, max_value=20, value=5, step=1)
            num_long_essay = st.number_input("Long Essay Questions", min_value=0, max_value=20, value=5, step=1)
        
        # Question tagging system
        st.subheader("Question Tagging")
        with st.expander("Question Settings", expanded=False):
            st.multiselect(
                "Question Tags",
                options=[tag.value for tag in QuestionTag],
                default=st.session_state.question_tags,
                key="question_tags"
            )
        
        # Coding question options
        if num_coding > 0:
            st.markdown("### Coding Question Options")
            coding_col1, coding_col2 = st.columns(2)
            with coding_col1:
                programming_language = st.selectbox(
                    "Programming Language",
                    ["Python", "Java", "JavaScript", "C++", "C#"],
                    key="exam_language"
                )
                difficulty_level = st.select_slider(
                    "Difficulty Level",
                    options=["Beginner", "Intermediate", "Advanced"],
                    value="Intermediate"
                )
            with coding_col2:
                include_test_cases = st.checkbox("Include Test Cases", value=True, key="exam_test_cases")
                include_solution = st.checkbox("Include Solution", value=True, key="exam_solution")
        
        if st.button("Generate Exam", key="generate_exam_button"):
            if not st.session_state.course_name:
                st.error("Please enter a course name")
            else:
                with st.spinner("Generating exam questions..."):
                    try:
                        # Add progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        model_config = st.session_state.model_config
                        generator = ExamGenerator(model_name=model_config.get("model_name", "llama3"))
                        
                        # Update progress
                        progress_bar.progress(20)
                        status_text.text("Initializing generator...")
                        
                        # Only include general_topic if it's not empty
                        generation_params = {
                            "course_name": st.session_state.course_name,
                            "num_mcq": num_mcq,
                            "num_fill_blank": num_fill_blank,
                            "num_short_essay": num_short_essay,
                            "num_long_essay": num_long_essay,
                            "num_coding": num_coding,
                            "tags": st.session_state.question_tags,
                            "coding_options": {
                                "language": programming_language if num_coding > 0 else None,
                                "difficulty": difficulty_level if num_coding > 0 else None,
                                "include_test_cases": include_test_cases if num_coding > 0 else False,
                                "include_solution": include_solution if num_coding > 0 else False
                            }
                        }
                        
                        if general_topic.strip():
                            generation_params["general_topic"] = general_topic
                        
                        # Update progress
                        progress_bar.progress(40)
                        status_text.text("Generating questions...")
                        
                        exam_data = generator.generate_exam(**generation_params)
                        
                        # Update progress
                        progress_bar.progress(80)
                        status_text.text("Finalizing exam...")
                        
                        if not exam_data or "questions" not in exam_data or not exam_data["questions"]:
                            st.error("Failed to generate questions. Ensure the course material is properly ingested.")
                        else:
                            st.session_state.exam_data = exam_data
                            st.success(f"Successfully generated {len(exam_data['questions'])} questions")
                            
                            # Show a sample of questions
                            st.subheader("Sample Questions")
                            for i, q in enumerate(exam_data["questions"][:3]):
                                st.markdown(f"**Question {i+1} ({q['type']})**: {q['question']}")
                                if q['type'] == 'coding':
                                    st.markdown("**Programming Language:** " + q.get('language', 'Not specified'))
                                    if 'test_cases' in q:
                                        st.markdown("**Test Cases:**")
                                        for tc in q['test_cases']:
                                            st.code(f"Input: {tc['input']}\nExpected Output: {tc['output']}")
                            
                            # PDF Export Options
                            st.subheader("PDF Export Options")
                            with st.expander("PDF Export Options", expanded=False):
                                st.selectbox(
                                    "Paper Size",
                                    options=["A4", "Letter", "Legal"],
                                    key="pdf_options.paper_size"
                                )
                                
                                st.checkbox(
                                    "Include Table of Contents",
                                    value=st.session_state.pdf_options['include_toc'],
                                    key="pdf_options.include_toc"
                                )
                                
                                st.checkbox(
                                    "Include Metadata",
                                    value=st.session_state.pdf_options['include_metadata'],
                                    key="pdf_options.include_metadata"
                                )
                                
                                st.slider(
                                    "Compression Level",
                                    min_value=0,
                                    max_value=9,
                                    value=st.session_state.pdf_options['compression_level'],
                                    key="pdf_options.compression_level"
                                )
                            
                            # Option to download the full exam
                            json_str = json.dumps(exam_data, indent=2)
                            st.download_button(
                                label="Download Full Exam JSON",
                                data=json_str,
                                file_name=f"{st.session_state.course_name}_exam.json",
                                mime="application/json",
                            )
                            
                            # Generate and handle PDF
                            output_path = os.path.join("output", f"{st.session_state.course_name}_exam.pdf")
                            os.makedirs("output", exist_ok=True)
                            
                            try:
                                # Generate PDF
                                pdf_path = generator.pdf_generator.generate_exam_pdf(
                                    exam_data,
                                    output_path=output_path,
                                    options={
                                        "paper_size": st.session_state.pdf_options['paper_size'],
                                        "include_toc": st.session_state.pdf_options['include_toc'],
                                        "include_metadata": st.session_state.pdf_options['include_metadata'],
                                        "compression_level": st.session_state.pdf_options['compression_level']
                                    }
                                )
                                st.success("PDF generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                            
                            # Show download button if PDF exists
                            if os.path.exists(output_path):
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "Download PDF",
                                        data=f.read(),
                                        file_name=f"{st.session_state.course_name}_exam.pdf",
                                        mime="application/pdf"
                                    )
                            else:
                                st.error("PDF file not found. Please try generating the PDF again.")
                        
                        # Complete progress
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        
                    except Exception as e:
                        st.error(f"Error generating exam: {str(e)}")
    
    elif content_type == "Assignment":
        # Assignment generation options
        st.subheader("Assignment Generation Options")
        assignment_type = st.selectbox(
            "Assignment Type",
            ["Problem Set", "Essay", "Case Study", "Lab Report", "Coding Assignment"]
        )
        
        difficulty = st.slider("Difficulty Level", 1, 5, 3)
        num_problems = st.number_input("Number of Problems", min_value=1, max_value=10, value=3)
        
        # Coding assignment options
        if assignment_type == "Coding Assignment":
            st.markdown("### Coding Assignment Options")
            coding_col1, coding_col2 = st.columns(2)
            with coding_col1:
                programming_language = st.selectbox(
                    "Programming Language",
                    ["Python", "Java", "JavaScript", "C++", "C#"],
                    key="assignment_language"
                )
                project_type = st.selectbox(
                    "Project Type",
                    ["Algorithm", "Data Structure", "Web Application", "API", "Game", "Utility Tool"]
                )
            with coding_col2:
                include_test_cases = st.checkbox("Include Test Cases", value=True, key="assignment_test_cases")
                include_solution = st.checkbox("Include Solution", value=True, key="assignment_solution")
                include_documentation = st.checkbox("Include Documentation Requirements", value=True)
        
        if st.button("Generate Assignment", key="generate_assignment_button"):
            if not st.session_state.course_name:
                st.error("Please enter a course name")
            else:
                with st.spinner("Generating assignment..."):
                    try:
                        model_config = st.session_state.model_config
                        generator = ExamGenerator(model_name=model_config.get("model_name", "llama3"))
                        
                        # Prepare coding options if it's a coding assignment
                        coding_options = None
                        if assignment_type == "Coding Assignment":
                            coding_options = {
                                "language": programming_language,
                                "project_type": project_type,
                                "include_test_cases": include_test_cases,
                                "include_solution": include_solution,
                                "include_documentation": include_documentation
                            }
                        
                        # Only include general_topic if it's not empty
                        generation_params = {
                            "course_name": st.session_state.course_name,
                            "assignment_type": assignment_type,
                            "difficulty": difficulty,
                            "num_problems": num_problems,
                            "coding_options": coding_options
                        }
                        
                        if general_topic.strip():
                            generation_params["general_topic"] = general_topic
                        
                        assignment_data = generator.generate_assignment(**generation_params)
                        
                        st.session_state.assignment_data = assignment_data
                        st.success("Successfully generated assignment")
                        
                        # Display assignment
                        st.subheader("Generated Assignment")
                        st.markdown(assignment_data["description"])
                        
                        for i, problem in enumerate(assignment_data["problems"], 1):
                            with st.expander(f"Problem {i}"):
                                st.markdown(problem["description"])
                                if "hints" in problem:
                                    st.markdown("**Hints:**")
                                    for hint in problem["hints"]:
                                        st.markdown(f"- {hint}")
                                if assignment_type == "Coding Assignment":
                                    if "test_cases" in problem:
                                        st.markdown("**Test Cases:**")
                                        for tc in problem["test_cases"]:
                                            st.code(f"Input: {tc['input']}\nExpected Output: {tc['output']}")
                                    if "documentation_requirements" in problem:
                                        st.markdown("**Documentation Requirements:**")
                                        for req in problem["documentation_requirements"]:
                                            st.markdown(f"- {req}")
                        
                        # PDF Export Options
                        st.subheader("PDF Export Options")
                        with st.expander("PDF Export Options", expanded=False):
                            st.selectbox(
                                "Paper Size",
                                options=["A4", "Letter", "Legal"],
                                key="pdf_options.paper_size"
                            )
                            
                            st.checkbox(
                                "Include Table of Contents",
                                value=st.session_state.pdf_options['include_toc'],
                                key="pdf_options.include_toc"
                            )
                            
                            st.checkbox(
                                "Include Metadata",
                                value=st.session_state.pdf_options['include_metadata'],
                                key="pdf_options.include_metadata"
                            )
                            
                            st.slider(
                                "Compression Level",
                                min_value=0,
                                max_value=9,
                                value=st.session_state.pdf_options['compression_level'],
                                key="pdf_options.compression_level"
                            )
                        
                        # Option to download
                        json_str = json.dumps(assignment_data, indent=2)
                        st.download_button(
                            label="Download Assignment JSON",
                            data=json_str,
                            file_name=f"{st.session_state.course_name}_assignment.json",
                            mime="application/json",
                        )
                        
                        # Generate and handle PDF
                        output_path = os.path.join("output", f"{st.session_state.course_name}_assignment.pdf")
                        os.makedirs("output", exist_ok=True)
                        
                        try:
                            # Generate PDF
                            pdf_path = generator.pdf_generator.generate_assignment_pdf(
                                assignment_data,
                                output_path=output_path,
                                options={
                                    "paper_size": st.session_state.pdf_options['paper_size'],
                                    "include_toc": st.session_state.pdf_options['include_toc'],
                                    "include_metadata": st.session_state.pdf_options['include_metadata'],
                                    "compression_level": st.session_state.pdf_options['compression_level']
                                }
                            )
                            st.success("PDF generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                        
                        # Show download button if PDF exists
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "Download PDF",
                                    data=f.read(),
                                    file_name=f"{st.session_state.course_name}_assignment.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.error("PDF file not found. Please try generating the PDF again.")
                    except Exception as e:
                        st.error(f"Error generating assignment: {str(e)}")
    
    else:  # Project
        # Project generation options
        st.subheader("Project Generation Options")
        project_type = st.selectbox(
            "Project Type",
            ["Research", "Implementation", "Analysis", "Design", "Coding Project"]
        )
        
        complexity = st.slider("Complexity Level", 1, 5, 3)
        duration_weeks = st.number_input("Duration (weeks)", min_value=1, max_value=16, value=4)
        
        # Project Templates
        st.subheader("Project Templates")
        with st.expander("Project Template", expanded=False):
            template_type = st.selectbox(
                "Template Type",
                options=[template.value for template in ProjectTemplate],
                key="project_template"
            )
            
            if template_type == ProjectTemplate.CUSTOM.value:
                st.text_area(
                    "Custom Template Structure",
                    value="",
                    help="Enter your custom project structure in JSON format"
                )
        
        # Coding project options
        if project_type == "Coding Project":
            st.markdown("### Coding Project Options")
            coding_col1, coding_col2 = st.columns(2)
            with coding_col1:
                programming_language = st.selectbox(
                    "Programming Language",
                    ["Python", "Java", "JavaScript", "C++", "C#"],
                    key="project_language"
                )
                project_category = st.selectbox(
                    "Project Category",
                    ["Web Development", "Mobile App", "Data Science", "Game Development", 
                     "System Design", "API Development", "DevOps", "Security"]
                )
            with coding_col2:
                include_test_cases = st.checkbox("Include Test Cases", value=True, key="project_test_cases")
                include_solution = st.checkbox("Include Solution", value=True, key="project_solution")
                include_documentation = st.checkbox("Include Documentation Requirements", value=True)
                include_deployment = st.checkbox("Include Deployment Instructions", value=True)
        
        if st.button("Generate Project", key="generate_project_button"):
            if not st.session_state.course_name:
                st.error("Please enter a course name")
            else:
                with st.spinner("Generating project..."):
                    try:
                        # Add progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        model_config = st.session_state.model_config
                        generator = ExamGenerator(model_name=model_config.get("model_name", "llama3"))
                        
                        # Update progress
                        progress_bar.progress(20)
                        status_text.text("Initializing generator...")
                        
                        # Prepare coding options if it's a coding project
                        coding_options = None
                        if project_type == "Coding Project":
                            coding_options = {
                                "language": programming_language,
                                "project_category": project_category,
                                "include_test_cases": include_test_cases,
                                "include_solution": include_solution,
                                "include_documentation": include_documentation,
                                "include_deployment": include_deployment
                            }
                        
                        # Only include general_topic if it's not empty
                        generation_params = {
                            "course_name": st.session_state.course_name,
                            "project_type": project_type,
                            "complexity": complexity,
                            "duration_weeks": duration_weeks,
                            "coding_options": coding_options,
                            "use_template": bool(st.session_state.project_template),
                            "template_type": st.session_state.project_template,
                            "general_topic": general_topic if general_topic else None
                        }
                        
                        project_data = generator.generate_project(**generation_params)
                        
                        # Update progress
                        progress_bar.progress(80)
                        status_text.text("Finalizing project...")
                        
                        st.session_state.project_data = project_data
                        st.success("Successfully generated project")
                        
                        # Display project
                        st.subheader("Generated Project")
                        st.markdown(project_data["description"])
                        
                        with st.expander("Project Requirements"):
                            for req in project_data["requirements"]:
                                st.markdown(f"• {req}")
                        
                        with st.expander("Deliverables"):
                            for deliverable in project_data["deliverables"]:
                                st.markdown(f"• {deliverable}")
                        
                        with st.expander("Timeline"):
                            for milestone in project_data["timeline"]:
                                st.markdown(f"**Week {milestone['week']}**: {milestone['milestone']}")
                        
                        if project_type == "Coding Project":
                            with st.expander("Technical Details"):
                                if "test_cases" in project_data:
                                    st.markdown("### Test Cases")
                                    for tc in project_data["test_cases"]:
                                        st.code(f"Input: {tc['input']}\nExpected Output: {tc['output']}")
                                
                                if "documentation_requirements" in project_data:
                                    st.markdown("### Documentation Requirements")
                                    for req in project_data["documentation_requirements"]:
                                        st.markdown(f"- {req}")
                                
                                if "deployment_instructions" in project_data:
                                    st.markdown("### Deployment Instructions")
                                    st.markdown(project_data["deployment_instructions"])
                        
                        # PDF Export Options
                        st.subheader("PDF Export Options")
                        with st.expander("PDF Export Options", expanded=False):
                            st.selectbox(
                                "Paper Size",
                                options=["A4", "Letter", "Legal"],
                                key="pdf_options.paper_size"
                            )
                            
                            st.checkbox(
                                "Include Table of Contents",
                                value=st.session_state.pdf_options['include_toc'],
                                key="pdf_options.include_toc"
                            )
                            
                            st.checkbox(
                                "Include Metadata",
                                value=st.session_state.pdf_options['include_metadata'],
                                key="pdf_options.include_metadata"
                            )
                            
                            st.slider(
                                "Compression Level",
                                min_value=0,
                                max_value=9,
                                value=st.session_state.pdf_options['compression_level'],
                                key="pdf_options.compression_level"
                            )
                        
                        # Option to download
                        json_str = json.dumps(project_data, indent=2)
                        st.download_button(
                            label="Download Project JSON",
                            data=json_str,
                            file_name=f"{st.session_state.course_name}_project.json",
                            mime="application/json",
                        )
                        
                        # Generate and handle PDF
                        output_path = os.path.join("output", f"{st.session_state.course_name}_project.pdf")
                        os.makedirs("output", exist_ok=True)
                        
                        try:
                            # Generate PDF
                            pdf_path = generator.pdf_generator.generate_project_pdf(
                                project_data,
                                output_path=output_path,
                                options={
                                    "paper_size": st.session_state.pdf_options['paper_size'],
                                    "include_toc": st.session_state.pdf_options['include_toc'],
                                    "include_metadata": st.session_state.pdf_options['include_metadata'],
                                    "compression_level": st.session_state.pdf_options['compression_level']
                                }
                            )
                            st.success("PDF generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                        
                        # Show download button if PDF exists
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    "Download PDF",
                                    data=f.read(),
                                    file_name=f"{st.session_state.course_name}_project.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.error("PDF file not found. Please try generating the PDF again.")
                        
                        # Complete progress
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        
                    except Exception as e:
                        st.error(f"Error generating project: {str(e)}")

# Tab 3: Grade & Critique
with tab3:
    st.header("Grade & Critique")
    
    # Create subtabs for different grading methods
    grade_tab1, grade_tab2 = st.tabs(["Grade Content", "Upload & Grade"])
    
    with grade_tab1:
        # Grade existing content
        content_to_grade = None
        if st.session_state.exam_data:
            content_to_grade = st.session_state.exam_data
            content_type = "Exam"
        elif st.session_state.assignment_data:
            content_to_grade = st.session_state.assignment_data
            content_type = "Assignment"
        elif st.session_state.project_data:
            content_to_grade = st.session_state.project_data
            content_type = "Project"
        
        if content_to_grade:
            st.subheader(f"Grade {content_type}")
            
            # Upload student answers
            uploaded_answers = st.file_uploader("Upload student answers (JSON format)", 
                                              type="json", key="existing_content_answers")
            
            if uploaded_answers:
                try:
                    student_answers = json.loads(uploaded_answers.getvalue().decode())
                    st.success("Successfully loaded student answers")
                    
                    # Grade button
                    if st.button("Grade Submission", key="grade_submission_button"):
                        with st.spinner("Grading submission..."):
                            try:
                                model_config = st.session_state.model_config
                                grader = ExamGrader(model_name=model_config.get("model_name", "llama3"))
                                
                                if content_type == "Exam":
                                    results = grader.grade_exam(content_to_grade, student_answers)
                                else:
                                    results = grader.grade_content(content_to_grade, student_answers)
                                
                                # Display results
                                st.subheader("Grading Results")
                                
                                # Student info
                                st.markdown(f"**Student ID:** {results.get('Student-ID', 'Unknown')}")
                                st.markdown(f"**Student Name:** {results.get('Student-Name', 'Anonymous')}")
                                
                                # Calculate overall score
                                total_score = sum(result["score"] for result in results["results"])
                                max_score = len(results["results"]) * 100
                                percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                
                                st.metric("Overall Score", f"{percentage:.1f}%")
                                
                                # Display individual results
                                for i, result in enumerate(results["results"]):
                                    with st.expander(f"Item {i+1}: {result['question'][:100]}..."):
                                        st.markdown(f"**Question:** {result['question']}")
                                        st.markdown(f"**Student Answer:** {result['answer']}")
                                        st.markdown(f"**Score:** {result['score']}")
                                        if "feedback" in result:
                                            st.markdown(f"**Feedback:** {result['feedback']}")
                                
                                # Download results
                                json_str = json.dumps(results, indent=2)
                                st.download_button(
                                    label="Download Grading Results",
                                    data=json_str,
                                    file_name=f"{results.get('Student-ID', 'unknown')}_grading_results.json",
                                    mime="application/json",
                                )
                                
                                # Critique button
                                if st.button("Generate Critique", key="generate_critique_button"):
                                    with st.spinner("Generating critique..."):
                                        try:
                                            critic = ExamCritic(model_name=model_config.get("model_name", None))
                                            critique_results = critic.evaluate_content(content_to_grade, results)
                                            
                                            st.subheader("Critique Results")
                                            st.metric("Overall Quality Score", f"{critique_results.get('overall_score', 0):.1f}/10")
                                            
                                            st.markdown("### Overall Feedback")
                                            st.markdown(critique_results.get("feedback", "No feedback available"))
                                            
                                            st.markdown("### Detailed Feedback")
                                            for i, feedback in enumerate(critique_results.get("item_feedback", [])):
                                                with st.expander(f"Item {i+1}"):
                                                    st.markdown(f"**Score:** {feedback.get('score', 0)}/10")
                                                    st.markdown(f"**Feedback:** {feedback.get('feedback', '')}")
                                                    st.markdown(f"**Suggestion:** {feedback.get('suggestion', '')}")
                                            
                                            # Download critique
                                            json_str = json.dumps(critique_results, indent=2)
                                            st.download_button(
                                                label="Download Critique Results",
                                                data=json_str,
                                                file_name=f"{st.session_state.course_name}_critique_results.json",
                                                mime="application/json",
                                            )
                                        except Exception as e:
                                            st.error(f"Error generating critique: {str(e)}")
                            except Exception as e:
                                st.error(f"Error grading submission: {str(e)}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in uploaded file")
        else:
            st.info("Please generate content first in the 'Generate' tab")
    
    with grade_tab2:
        # Upload and grade independent content
        st.subheader("Upload & Grade Content")
        
        # Upload content file
        uploaded_content = st.file_uploader("Upload content JSON file", type="json", key="independent_content")
        
        # Upload student answer files
        uploaded_student_answers = st.file_uploader("Upload student answer JSON files (can select multiple)", 
                                                  type="json", accept_multiple_files=True, key="independent_answers")
        
        if uploaded_content and uploaded_student_answers:
            try:
                content_json = json.loads(uploaded_content.getvalue().decode())
                st.success("Successfully loaded content file")
                
                # Process student answers
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
                    if st.button("Grade Submissions", key="grade_submissions_button"):
                        with st.spinner(f"Grading {len(student_answer_list)} submissions..."):
                            try:
                                model_config = st.session_state.model_config
                                grader = ExamGrader(model_name=model_config.get("model_name", "llama3"))
                                results = grader.grade_from_json_optimized(content_json, student_answer_list)
                                
                                # Display results
                                st.subheader("Grading Results")
                                
                                # Summary table
                                summary_data = []
                                for result in results:
                                    student_id = result.get("Student-ID", "Unknown")
                                    student_name = result.get("Student-Name", "Anonymous")
                                    total_score = sum(question["score"] for question in result["results"])
                                    max_score = len(result["results"]) * 100
                                    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                    
                                    summary_data.append({
                                        "Student ID": student_id,
                                        "Student Name": student_name,
                                        "Score": f"{percentage:.1f}%"
                                    })
                                
                                if summary_data:
                                    st.table(summary_data)
                                
                                # Detailed results
                                for i, result in enumerate(results):
                                    student_id = result.get("Student-ID", "Unknown")
                                    student_name = result.get("Student-Name", "Anonymous")
                                    
                                    with st.expander(f"Student: {student_name} ({student_id})"):
                                        total_score = sum(question["score"] for question in result["results"])
                                        max_score = len(result["results"]) * 100
                                        percentage = (total_score / max_score) * 100 if max_score > 0 else 0
                                        
                                        st.metric("Score", f"{percentage:.1f}%")
                                        
                                        for j, question_result in enumerate(result["results"]):
                                            with st.expander(f"Item {j+1}: {question_result['question'][:100]}..."):
                                                st.markdown(f"**Question:** {question_result['question']}")
                                                st.markdown(f"**Student Answer:** {question_result['answer']}")
                                                st.markdown(f"**Score:** {question_result['score']}")
                                                if "feedback" in question_result:
                                                    st.markdown(f"**Feedback:** {question_result['feedback']}")
                                
                                # Download all results
                                json_str = json.dumps(results, indent=2)
                                st.download_button(
                                    label="Download All Grading Results",
                                    data=json_str,
                                    file_name="all_grading_results.json",
                                    mime="application/json",
                                )
                            except Exception as e:
                                st.error(f"Error grading submissions: {str(e)}")
                else:
                    st.warning("No valid student answer files found")
            except json.JSONDecodeError:
                st.error("Invalid JSON format in content file")
        else:
            st.info("Please upload both a content JSON file and at least one student answer JSON file")

# Tab 4: Analytics
with tab4:
    st.header("Analytics & Rubrics")
    
    # Create subtabs for different analytics views
    analytics_tab1, analytics_tab2 = st.tabs(["Performance Analytics", "Rubric Management"])
    
    with analytics_tab1:
        st.subheader("Performance Analytics")
        
        if not st.session_state.historical_scores:
            st.info("No historical data available. Start grading to see analytics.")
        else:
            try:
                # Generate summary report
                report = st.session_state.rubric_visualizer.generate_summary_report(
                    rubric=st.session_state.current_rubric,
                    scores=st.session_state.historical_scores[-1],
                    historical_scores=st.session_state.historical_scores
                )
                
                # Display visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.image(report['visualizations']['score_chart'], caption="Score Distribution")
                    if 'progress_tracking' in report['visualizations']:
                        st.image(report['visualizations']['progress_tracking'], caption="Progress Tracking")
                
                with col2:
                    st.image(report['visualizations']['performance_analytics'], caption="Performance Analytics")
                    if 'comparison_chart' in report['visualizations']:
                        st.image(report['visualizations']['comparison_chart'], caption="Score Comparison")
                
                # Display statistics
                st.markdown("### Statistics")
                stats_df = pd.DataFrame(report['analytics']['statistics']).T
                st.dataframe(stats_df)
            except Exception as e:
                logger.error(f"Error generating analytics: {str(e)}")
                st.error("Failed to generate analytics. Please try again.")
    
    with analytics_tab2:
        st.subheader("Rubric Management")
        
        # Rubric Type Selection
        rubric_type = st.selectbox(
            "Select Rubric Type",
            [rt.value for rt in RubricType],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Rubric Creation
        st.markdown("### Create New Rubric")
        title = st.text_input("Title")
        description = st.text_area("Description")
        total_points = st.number_input("Total Points", min_value=0, max_value=1000, value=100)
        
        if st.button("Generate Rubric", key="generate_rubric_button"):
            if title and description:
                try:
                    rubric = st.session_state.rubric_generator.generate_assignment_rubric(
                        title=title,
                        description=description,
                        total_points=Decimal(str(total_points))
                    )
                    st.session_state.current_rubric = rubric
                    st.success("Rubric generated successfully!")
                except Exception as e:
                    logger.error(f"Error generating rubric: {str(e)}")
                    st.error("Failed to generate rubric. Please try again.")
            else:
                st.error("Please fill in all required fields.")
        
        # Add new section for PDF-based rubric generation
        st.markdown("### Generate Rubric from PDF")
        st.markdown("Upload a PDF containing project/module information to generate a customized rubric.")
        
        uploaded_pdf = st.file_uploader("Upload PDF", type="pdf", key="rubric_pdf")
        
        if uploaded_pdf:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                pdf_path = tmp.name
                tmp.write(uploaded_pdf.getbuffer())
            
            try:
                # Display PDF content analysis
                with st.spinner("Analyzing PDF content..."):
                    content_analysis = st.session_state.rubric_generator.content_analyzer.analyze_content(pdf_path)
                    
                    # Display extracted information
                    with st.expander("Extracted Information", expanded=True):
                        st.markdown("#### Learning Outcomes")
                        for outcome in content_analysis["learning_outcomes"]:
                            st.markdown(f"- {outcome}")
                        
                        st.markdown("#### Assessment Criteria")
                        for criterion in content_analysis["assessment_criteria"]:
                            st.markdown(f"- {criterion}")
                        
                        st.markdown("#### Project Requirements")
                        for req in content_analysis["project_requirements"]:
                            st.markdown(f"- {req}")
                    
                    # Display suggested criteria
                    st.markdown("#### Suggested Rubric Criteria")
                    for criterion in content_analysis["suggested_criteria"]:
                        with st.expander(f"{criterion['name']} ({criterion['weight']*100}%)"):
                            st.markdown(f"**Description:** {criterion['description']}")
                            st.markdown(f"**Max Score:** {criterion['max_score']}")
                            
                            st.markdown("**Performance Levels:**")
                            for level in criterion['levels']:
                                st.markdown(f"- **{level['description']}**: {level['feedback']}")
                
                # Form for finalizing the rubric
                st.markdown("### Finalize Rubric")
                title = st.text_input("Rubric Title", value=f"Rubric for {uploaded_pdf.name}", key="pdf_rubric_title")
                description = st.text_area("Rubric Description", key="pdf_rubric_description")
                total_points = st.number_input("Total Points", min_value=0, max_value=1000, value=100, key="pdf_rubric_total_points")
                
                # Allow editing of criteria
                st.markdown("#### Edit Criteria")
                edited_criteria = []
                for i, criterion in enumerate(content_analysis["suggested_criteria"]):
                    with st.expander(f"Edit {criterion['name']}", expanded=False):
                        name = st.text_input("Name", value=criterion['name'], key=f"pdf_name_{i}")
                        description = st.text_area("Description", value=criterion['description'], key=f"pdf_desc_{i}")
                        weight = st.slider("Weight (%)", 0, 100, int(criterion['weight']*100), key=f"pdf_weight_{i}")
                        max_score = st.number_input("Max Score", 0, 100, int(criterion['max_score']), key=f"pdf_score_{i}")
                        
                        st.markdown("**Performance Levels**")
                        levels = []
                        for j, level in enumerate(criterion['levels']):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                level_name = st.text_input("Level", value=level['description'], key=f"pdf_level_{i}_{j}")
                            with col2:
                                feedback = st.text_area("Feedback", value=level['feedback'], key=f"pdf_feedback_{i}_{j}")
                            levels.append({"description": level_name, "feedback": feedback})
                        
                        edited_criteria.append({
                            "name": name,
                            "description": description,
                            "weight": weight/100,
                            "max_score": max_score,
                            "levels": levels
                        })
                
                if st.button("Generate Rubric", key="pdf_generate_rubric_button"):
                    try:
                        # Convert edited criteria to Criterion objects
                        criteria = []
                        for criterion_data in edited_criteria:
                            criterion = Criterion(
                                name=criterion_data["name"],
                                description=criterion_data["description"],
                                weight=Decimal(str(criterion_data["weight"])),
                                max_score=Decimal(str(criterion_data["max_score"])),
                                levels=criterion_data["levels"]
                            )
                            criteria.append(criterion)
                        
                        # Create the rubric
                        rubric = Rubric(
                            rubric_type=RubricType.PROJECT,  # Default to PROJECT type
                            title=title,
                            description=description,
                            criteria=criteria,
                            total_points=Decimal(str(total_points)),
                            feedback_template=st.session_state.rubric_generator._generate_feedback_template(
                                content_analysis["learning_outcomes"],
                                content_analysis["assessment_criteria"]
                            )
                        )
                        
                        st.session_state.current_rubric = rubric
                        st.success("Rubric generated successfully!")
                        
                        # Display the generated rubric
                        display_rubric(rubric)
                        
                    except Exception as e:
                        st.error(f"Error generating rubric: {str(e)}")
                
            except Exception as e:
                st.error(f"Error analyzing PDF: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(pdf_path)
        
        # Display current rubric if exists
        if st.session_state.current_rubric:
            st.markdown("### Current Rubric")
            display_rubric(st.session_state.current_rubric)
        else:
            st.info("Create a new rubric using one of the methods above.")

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

def main():
    """Main function to run the Streamlit application."""
    # The main functionality is already implemented in the global scope
    # This function is just a wrapper to satisfy the entry point
    pass

if __name__ == "__main__":
    main()