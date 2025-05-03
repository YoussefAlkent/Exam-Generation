#!/usr/bin/env python3
import json
import os
import random
import argparse
from typing import List, Dict, Any

def generate_test_answers(exam_path: str, num_students: int = 10) -> List[Dict[str, Any]]:
    """
    Generate example student answers for testing the grading system.
    
    Args:
        exam_path: Path to the exam JSON file
        num_students: Number of student answer files to generate
        
    Returns:
        List of student answer dictionaries
    """
    # Load the exam file
    with open(exam_path, 'r') as f:
        exam_data = json.load(f)
    
    if 'questions' not in exam_data:
        raise ValueError("Invalid exam file: missing 'questions' field")
    
    questions = exam_data['questions']
    student_answers = []
    
    # Student names and IDs for the example answers
    example_names = [
        "Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Prince", 
        "Ethan Hunt", "Fiona Gallagher", "George Washington", "Hannah Montana", 
        "Ian Malcolm", "Julia Roberts"
    ]
    
    for i in range(num_students):
        # Create student info
        student_id = f"STU{1000 + i}"
        student_name = example_names[i] if i < len(example_names) else f"Student {i+1}"
        
        # Initialize answers list
        answers = []
        
        # Generate answers for each question
        for q_idx, question in enumerate(questions):
            question_type = question.get('type', '')
            
            # Determine answer quality (70% correct, 15% incorrect, 15% empty)
            answer_quality = random.choices(
                ['correct', 'incorrect', 'empty'], 
                weights=[70, 15, 15], 
                k=1
            )[0]
            
            if answer_quality == 'empty':
                # Leave answer empty
                continue
            
            elif question_type == 'mcq':
                choices = question.get('choices', [])
                if not choices:
                    continue
                
                if answer_quality == 'correct':
                    # Use the correct answer if available, otherwise random
                    answer = question.get('answer', random.choice(choices))
                else:
                    # Choose an incorrect answer
                    correct = question.get('answer', '')
                    incorrect_choices = [c for c in choices if c != correct]
                    answer = random.choice(incorrect_choices) if incorrect_choices else random.choice(choices)
            
            elif question_type == 'fill_in_the_blank':
                if answer_quality == 'correct':
                    answer = question.get('answer', 'Sample answer')
                else:
                    # Generate an incorrect answer by modifying the correct one
                    correct = question.get('answer', 'Sample answer')
                    answer = correct + " (incorrect)" if correct else "Wrong answer"
            
            elif question_type in ['short_essay', 'long_essay']:
                correct_answer = question.get('answer', '')
                
                if answer_quality == 'correct':
                    # Create a high-quality answer (with minor variations)
                    essay_length = len(correct_answer.split())
                    additional_comments = [
                        'this concept is fundamental to understanding the topic.',
                        'this approach is widely used in practice.',
                        'researchers have extensively validated these findings.',
                        'this relates to other key concepts we studied.'
                    ]
                    answer = f"{correct_answer} Additionally, {random.choice(additional_comments)}"
                else:
                    # Create a partially incorrect or incomplete answer
                    # Take just part of the correct answer or add incorrect statements
                    words = correct_answer.split()
                    if len(words) > 10:
                        # Use only first part of the answer
                        partial_length = random.randint(max(5, len(words) // 3), len(words) // 2)
                        answer = ' '.join(words[:partial_length])
                    else:
                        answer = "I think this has something to do with the topic, but I'm not entirely sure of the details."
            
            else:
                # Unknown question type, provide generic answer
                answer = "Cannot determine appropriate answer for this question type"
            
            # Add the answer
            answers.append({
                "question_index": q_idx,
                "answer": answer
            })
        
        # Create the student answer object
        student_answer = {
            "Student-ID": student_id,
            "Student-Name": student_name,
            "answers": answers
        }
        
        student_answers.append(student_answer)
    
    return student_answers

def main():
    parser = argparse.ArgumentParser(description='Generate example student answers for testing')
    parser.add_argument('exam_path', help='Path to the exam JSON file')
    parser.add_argument('--output-dir', '-o', default='./test_answers', help='Directory to save generated answer files')
    parser.add_argument('--num-students', '-n', type=int, default=10, help='Number of student answers to generate')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Generate student answers
        student_answers = generate_test_answers(args.exam_path, args.num_students)
        
        # Save the answers to separate files
        for i, answer in enumerate(student_answers):
            student_id = answer["Student-ID"]
            output_path = os.path.join(args.output_dir, f"{student_id}_answers.json")
            
            with open(output_path, 'w') as f:
                json.dump(answer, f, indent=2)
            
            print(f"Generated answer file: {output_path}")
        
        # Also save all answers to a single file for convenience
        all_answers_path = os.path.join(args.output_dir, "all_answers.json")
        with open(all_answers_path, 'w') as f:
            json.dump(student_answers, f, indent=2)
        
        print(f"Generated combined file with all answers: {all_answers_path}")
        print(f"Successfully generated {len(student_answers)} example student answers")
    
    except Exception as e:
        print(f"Error generating answers: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())