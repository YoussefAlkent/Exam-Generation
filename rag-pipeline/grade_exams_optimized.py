#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.grader.optimized_grader import ExamGrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("grade_exams_cli")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        sys.exit(1)

def load_student_answers(answers_dir: str) -> List[Dict[str, Any]]:
    """
    Load all student answer JSON files in the directory.
    """
    answer_files = []
    for filename in os.listdir(answers_dir):
        if filename.endswith('_answers.json') and filename != 'all_answers.json':
            file_path = os.path.join(answers_dir, filename)
            logger.info(f"Loading student answers from {file_path}")
            answer_files.append(load_json_file(file_path))
    
    if not answer_files:
        logger.error(f"No student answer files found in {answers_dir}")
        sys.exit(1)
    
    return answer_files

def save_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save the grading results to JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual student results
    for result in results:
        student_id = result.get("Student-ID", "Unknown")
        output_path = os.path.join(output_dir, f"{student_id}_graded.json")
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved results for student {student_id} to {output_path}")
    
    # Save all results to a single file
    all_results_path = os.path.join(output_dir, "all_graded.json")
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved combined results to {all_results_path}")

def main():
    parser = argparse.ArgumentParser(description='Grade exams using the optimized batch grader')
    parser.add_argument('exam_file', help='Path to the exam JSON file')
    parser.add_argument('answers_dir', help='Directory containing student answer files')
    parser.add_argument('--output-dir', '-o', default='./graded_results', help='Directory to save graded results')
    parser.add_argument('--model', '-m', default=None, help='LLM model to use for grading')
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Maximum number of students to batch in one API call')
    args = parser.parse_args()
    
    # Load exam data
    logger.info(f"Loading exam from {args.exam_file}")
    exam_data = load_json_file(args.exam_file)
    
    # Load student answers
    logger.info(f"Loading student answers from {args.answers_dir}")
    student_answers = load_student_answers(args.answers_dir)
    
    # Grade answers
    logger.info(f"Grading answers for {len(student_answers)} students using optimized batch processing")
    grader = ExamGrader(model_name=args.model)
    
    # Break up large batches into smaller chunks to avoid exceeding API limits
    all_results = []
    
    for i in range(0, len(student_answers), args.batch_size):
        batch = student_answers[i:i+args.batch_size]
        logger.info(f"Processing batch {i//args.batch_size + 1} with {len(batch)} students")
        
        batch_results = grader.grade_from_json_optimized(exam_data, batch)
        all_results.extend(batch_results)
    
    # Save results
    logger.info(f"Saving results to {args.output_dir}")
    save_results(all_results, args.output_dir)
    
    logger.info("Grading complete!")

if __name__ == "__main__":
    main()