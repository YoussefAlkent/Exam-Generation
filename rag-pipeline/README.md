# Course Exam Generator

A Retrieval-Augmented Generation (RAG) pipeline for automatically creating and grading course exams from PDF materials.

## Features

- **PDF Ingestion & Vectorization**: Extract text from PDFs and create searchable vector embeddings
- **Exam Generation**: Create multiple question types (MCQ, fill-in-the-blank, essays) based on course content
- **Relevance Filtering**: Ensure questions are relevant to the course material
- **Answer Grading**: Automatically grade student answers
- **User-Friendly Interface**: Streamlit web app for easy interaction

## Requirements

- Python 3.8+
- Ollama with the llama3.2:latest model installed
- PDF course materials

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd rag-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with the llama3.2:latest model:
```bash
ollama run llama3.2:latest
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the app in your browser at http://localhost:8501

3. Follow the steps in the app:
   - Upload PDFs or use existing ones in the ./pdfs/ folder
   - Enter a course name and ingest the documents
   - Generate an exam with various question types
   - Filter questions for relevance
   - Upload student answers for grading

## Project Structure

- `src/ingestion/` - PDF parsing and vectorization
- `src/generation/` - Exam question generation
- `src/critic/` - Question filtering and validation
- `src/grader/` - Answer evaluation
- `app.py` - Streamlit frontend
- `pdfs/` - Directory for PDF course materials
- `chroma_db/` - Persistent storage for vector embeddings

## Student Answer Format

Student answers should be uploaded as a JSON file with the following structure:

```json
{
  "answers": [
    {
      "question_index": 0,
      "answer": "B"
    },
    {
      "question_index": 5,
      "answer": "example answer for fill-in-the-blank"
    },
    {
      "question_index": 10,
      "answer": "This is a short essay answer..."
    }
  ]
}
```

## License

MIT