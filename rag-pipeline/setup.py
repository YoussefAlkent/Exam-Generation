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
        'pytest',
        'setuptools',
        'wheel',
        'langchain',
        'chromadb',
        'PyPDF2',  # or any other PDF parsing library you choose
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