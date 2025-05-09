import streamlit as st
import pandas as pd
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any
import os
import json
import logging

from src.generation.rubric_types import Rubric, RubricType, Criterion, FeedbackTemplate
from src.generation.rubric_generator import RubricGenerator
from src.generation.rubric_visualizer import RubricVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'rubric_generator' not in st.session_state:
    st.session_state.rubric_generator = RubricGenerator()
if 'rubric_visualizer' not in st.session_state:
    st.session_state.rubric_visualizer = RubricVisualizer()
if 'current_rubric' not in st.session_state:
    st.session_state.current_rubric = None
if 'historical_scores' not in st.session_state:
    st.session_state.historical_scores = []

def main():
    st.set_page_config(
        page_title="Rubric System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Rubric System")
        st.markdown("---")
        
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
        
        if st.button("Generate Rubric"):
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
    
    # Main Content
    if st.session_state.current_rubric:
        display_rubric(st.session_state.current_rubric)
    else:
        st.info("Create a new rubric using the sidebar options.")

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
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Criteria", "Scoring", "Analytics"])
    
    with tab1:
        display_criteria(rubric)
    
    with tab2:
        display_scoring(rubric)
    
    with tab3:
        display_analytics(rubric)

def display_criteria(rubric: Rubric):
    """Display the rubric criteria in an organized format."""
    st.markdown("### Criteria")
    
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
    
    if st.button("Calculate Score"):
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

def display_analytics(rubric: Rubric):
    """Display analytics and visualizations."""
    st.markdown("### Analytics")
    
    if not st.session_state.historical_scores:
        st.info("No historical data available. Start scoring to see analytics.")
        return
    
    try:
        # Generate summary report
        report = st.session_state.rubric_visualizer.generate_summary_report(
            rubric=rubric,
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

if __name__ == "__main__":
    main() 