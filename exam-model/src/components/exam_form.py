from typing import List, Dict, Tuple
import streamlit as st

def create_exam_form(questions: List[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    st.header("Exam Submission Form")
    
    # Student information collection
    student_info = {}
    student_info["id"] = st.text_input("Enter your Student ID:", key="student_id")
    student_info["name"] = st.text_input("Enter your Name:", key="student_name")
    
    answers = {}
    
    for i, question in enumerate(questions):
        question_text = question["question"]
        question_type = question["type"]
        
        if question_type == "mcq":
            # Use choices directly without any update script
            options = question.get("choices", [])
            if options:  # Only use radio button if choices are available
                answers[question_text] = st.radio(
                    question_text, 
                    options,
                    key=f"mcq_{i}"
                )
            else:
                st.warning(f"No choices provided for MCQ: {question_text}")
                answers[question_text] = st.text_input(f"{question_text} (please enter your answer)", key=f"mcq_text_{i}")
        elif question_type == "fill_in_the_blank":
            answers[question_text] = st.text_input(question_text, key=f"fill_{i}")
        elif question_type in ["short_essay", "long_essay"]:
            answers[question_text] = st.text_area(question_text, key=f"essay_{i}")
    
    return student_info, answers