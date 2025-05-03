from flask import Flask, render_template, request, jsonify
import os
import logging
from src.generation.generator import ExamGenerator

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_exam', methods=['POST'])
def generate_exam():
    try:
        # Extract data from request
        data = request.json
        collection_name = data.get('course_name')
        num_questions = int(data.get('num_questions', 5))
        difficulty = data.get('difficulty', 'medium')
        
        # Log request parameters
        logger.info(f"Generating exam with parameters: collection_name={collection_name}, num_questions={num_questions}, difficulty={difficulty}")
        
        if not collection_name:
            return jsonify({"error": "Course name (collection_name) is required"}), 400
        
        # Create the exam generator with the specified collection
        generator = ExamGenerator(
            vector_store_dir=CHROMA_DB_DIR,
            collection_name=collection_name  # Pass the collection_name here
        )
        
        # Generate the exam - also pass collection_name to ensure it's used
        exam_data = generator.generate_exam(
            num_questions=num_questions, 
            difficulty=difficulty,
            collection_name=collection_name  # Pass it here as well to be sure
        )
        
        return jsonify(exam_data)
    except Exception as e:
        logger.error(f"Error generating exam: {str(e)}")
        return jsonify({"error": f"Failed to generate exam: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')