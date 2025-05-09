from setuptools import setup, find_packages

setup(
    name='rag-pipeline',
    version='0.1.0',
    author='Youssef Bedair',
    author_email='youssefalkent@gmail.com',
    description='A Retrieval-Augmented Generation (RAG) pipeline for PDF ingestion and question generation.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pytest>=7.3.1',
        'setuptools',
        'wheel',
        'langchain==0.3.24',
        'langchain-community==0.3.23',
        'langchain-core==0.3.57',
        'chromadb==0.4.17',
        'PyPDF2>=3.0.1',
        'streamlit>=1.28.0',
        'sentence-transformers>=2.2.2',
        'numpy>=1.24.3',
        'pdfplumber==0.10.2',
        'tiktoken==0.4.0',
        'langchain-groq>=0.1.0',
        'langchain-google-genai>=0.0.7',
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
        ],
    },
    python_requires='>=3.7',
)