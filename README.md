# Course Exam Generator

A Retrieval-Augmented Generation (RAG) pipeline for automatically creating, evaluating, and grading course exams from PDF materials.

## Overview

The Course Exam Generator is an AI-powered application that helps educators create high-quality exams based on course materials. It uses Retrieval-Augmented Generation (RAG) techniques to extract knowledge from PDF documents and generate relevant exam questions in various formats. The application also includes tools for evaluating exam quality and grading student responses.

## Features

- **PDF Ingestion & Vectorization**: Extract text from PDFs and create searchable vector embeddings
- **Exam Generation**: Create multiple question types (MCQ, fill-in-the-blank, essays) based on course content
- **Relevance Filtering**: Ensure questions are relevant to the course material
- **Exam Quality Assessment**: Evaluate and critique the exam for quality and improvement
- **Answer Grading**: Automatically grade student answers with detailed feedback
- **Multiple LLM Support**: Works with various LLM providers (Ollama, Groq, Google, OpenAI)
- **User-Friendly Interface**: Streamlit web app for easy interaction

## Requirements

- Python 3.8+
- LLM provider (one of the following):
  - Ollama with llama3 model installed (default)
  - Groq API key
  - Google Gemini API key
  - OpenAI API key
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

3. Set up your preferred LLM provider:
   - For Ollama (default): Make sure Ollama is running with the llama3 model
     ```bash
     ollama run llama3
     ```
   - For other providers: Set appropriate API keys in environment variables or through the UI

## Environment Variables

You can customize the application behavior with these environment variables:

- `DEFAULT_MODEL_PROVIDER`: LLM provider to use (ollama, groq, google, openai)
- `DEFAULT_MODEL_NAME`: Default model name for the selected provider
- `DEFAULT_OLLAMA_URL`: URL for Ollama API (default: http://localhost:11434)
- `EMBEDDING_MODEL_NAME`: Model to use for text embeddings
- `GROQ_API_KEY`: API key for Groq
- `GOOGLE_API_KEY`: API key for Google Gemini
- `OPENAI_API_KEY`: API key for OpenAI

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the app in your browser at http://localhost:8501

3. Follow the workflow in the app:
   - Configure your LLM provider in the sidebar
   - Upload PDFs or use existing ones in the ./pdfs/ folder
   - Enter a course name and ingest the documents
   - Generate an exam with various question types
   - Filter questions for relevance
   - Critique the exam for quality assessment
   - Upload student answers for grading

## Project Structure

- `app.py` - Main Streamlit application
- `src/` - Source code directory
  - `ingestion/` - PDF parsing and vectorization
  - `generation/` - Exam question generation
  - `critic/` - Question filtering and exam evaluation
  - `grader/` - Student answer evaluation
  - `models/` - LLM provider implementations
- `pdfs/` - Directory for PDF course materials
- `chroma_db/` - Persistent storage for vector embeddings

## Student Answer Format

Student answers should be uploaded as a JSON file with the following structure:

```json
{
  "Student-ID": "12345",
  "Student-Name": "John Doe",
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

## Exam Format

The system generates exams in a standardized JSON format:

```json
{
  "course_name": "Introduction to Computer Science",
  "questions": [
    {
      "type": "mcq",
      "question": "What is the primary purpose of an algorithm?",
      "choices": ["Store data", "Process information", "Connect to the internet", "Display graphics"],
      "answer": "B"
    },
    {
      "type": "fill_in_blank",
      "question": "A _______ is a sequence of instructions that solves a problem.",
      "answer": "algorithm"
    },
    {
      "type": "essay",
      "question": "Explain the difference between procedural and object-oriented programming.",
      "answer": "Procedural programming organizes code into procedures or functions that operate on data, while object-oriented programming organizes code into objects that contain both data and methods..."
    }
  ]
}
```

## License

MIT