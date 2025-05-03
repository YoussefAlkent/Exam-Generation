import streamlit as st
import os
import json
from utils.file_handlers import read_json_file
from components.exam_form import create_exam_form

def main():
    st.title("Exam Grader")
    
    # Load exam questions from JSON files in the data directory
    data_dir = './data'
    exam_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not exam_files:
        st.error("No exam JSON files found in the data directory.")
        return
    
    # Select an exam file
    selected_exam_file = st.selectbox("Select an exam file", exam_files)
    exam_data = read_json_file(os.path.join(data_dir, selected_exam_file))
    
    # Create the exam form for answers
    if exam_data:
        student_info, answers = create_exam_form(exam_data["questions"])
        
        if st.button("Submit", key="main_submit_button"):
            # Format answers in the structure expected by the grader
            formatted_answers = []
            for i, (question, answer) in enumerate(answers.items()):
                formatted_answers.append({
                    "question_index": i,
                    "answer": answer
                })
            
            # Prepare the answer model
            answer_model = {
                "Student-ID": student_info.get("id", "Not provided"),
                "Student-Name": student_info.get("name", "Not provided"),
                "answers": formatted_answers
            }
            
            # Convert to JSON string
            json_str = json.dumps(answer_model, indent=2)
            
            # Use download_button to provide the JSON as a downloadable file
            filename = f"{student_info.get('id', 'unknown')}_answers.json"
            st.download_button(
                label="Download your answers",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    main()