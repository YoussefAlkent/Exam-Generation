import os
import glob
from typing import List, Dict, Any
import PyPDF2
import logging
import time  # Add time import for performance tracking

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure enhanced logging with separate file handler to prevent circular logging with watchdog
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFIngester")

# Remove any existing handlers to avoid duplicate logging
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create a separate log file for the ingestion process
ingestion_log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ingestion.log")
file_handler = logging.FileHandler(ingestion_log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Add console handler for interactive feedback
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class PDFIngester:
    def __init__(self, pdf_dir: str = "./pdfs/", persist_dir: str = "./chroma_db"):
        """
        Initialize the PDF ingester.
        
        Args:
            pdf_dir: Directory containing PDF files
            persist_dir: Directory to persist ChromaDB
        """
        logger.info(f"Initializing PDFIngester with pdf_dir={pdf_dir}, persist_dir={persist_dir}")
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        
        # Get model provider from environment variables
        default_provider = os.environ.get("DEFAULT_MODEL_PROVIDER", "ollama")
        default_model_name = os.environ.get("DEFAULT_MODEL_NAME", "llama3")
        embedding_model = os.environ.get("EMBEDDING_MODEL_NAME", "llama3.2:latest")
        
        logger.info(f"Using provider: {default_provider} with embedding model: {embedding_model}")
        
        try:
            if default_provider == "ollama":
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif default_provider == "google":
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_google_genai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif default_provider == "groq":
                try:
                    from langchain_groq import GroqEmbeddings
                    self.embeddings = GroqEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_groq not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            elif default_provider == "openai":
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embeddings = OpenAIEmbeddings(model=embedding_model)
                except ImportError:
                    logger.warning("langchain_openai not installed. Falling back to Ollama embeddings.")
                    self.embeddings = OllamaEmbeddings(model=embedding_model)
            else:
                # Default to Ollama if provider not recognized
                logger.warning(f"Unrecognized provider {default_provider}, falling back to Ollama embeddings")
                self.embeddings = OllamaEmbeddings(model=embedding_model)
            
            logger.info(f"Successfully initialized {default_provider} embeddings with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize {default_provider} embeddings with {embedding_model}: {e}")
            # Fallback to Ollama embeddings
            fallback_model = "llama2"
            logger.info(f"Falling back to Ollama embedding model: {fallback_model}")
            self.embeddings = OllamaEmbeddings(model=fallback_model)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_pdfs(self) -> List[Dict[str, Any]]:
        """Process PDF files and return a list of documents with metadata."""
        logger.info(f"Processing PDF files in directory: {self.pdf_dir}")
        documents = []
        
        for pdf_file in glob.glob(os.path.join(self.pdf_dir, "*.pdf")):
            logger.info(f"Processing file: {pdf_file}")
            try:
                with open(pdf_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text()
                    
                    # Get the filename for metadata
                    filename = os.path.basename(pdf_file)
                    
                    # Create chunks from the text
                    logger.info(f"Splitting text from {filename} into chunks")
                    chunks = self.text_splitter.split_text(text)
                    logger.info(f"Generated {len(chunks)} chunks from {filename}")
                    
                    # Add metadata to each chunk
                    logger.info(f"Adding metadata to chunks from {filename}")
                    for j, chunk in enumerate(chunks):
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": filename,
                                "chunk_id": j
                            }
                        })
            except Exception as e:
                logger.error(f"Failed to process file {pdf_file}: {e}")
        
        logger.info(f"PDF processing complete. Total documents generated: {len(documents)}")
        return documents
    
    def ingest_to_vectorstore(self, course_name: str) -> Chroma:
        """Ingest documents to ChromaDB."""
        logger.info(f"Beginning ingestion to vector store for course: {course_name}")
        start_time = time.time()
        
        logger.info("Processing PDFs...")
        documents = self.process_pdfs()
        
        if not documents:
            logger.error(f"No documents found in {self.pdf_dir}. Make sure there are PDF files in this directory.")
            raise ValueError(f"No documents found in {self.pdf_dir}")
        
        # Create texts and metadatas for Chroma
        logger.info("Preparing documents for vector store ingestion")
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Sanitize collection name to avoid issues
        sanitized_name = ''.join(c if c.isalnum() else '_' for c in course_name)
        logger.info(f"Using sanitized collection name: {sanitized_name}")
        
        try:
            # Create or get existing collection
            logger.info(f"Creating/accessing ChromaDB collection: {sanitized_name}")
            db = Chroma(
                collection_name=sanitized_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            
            # Add documents to the collection
            if texts:
                logger.info(f"Adding {len(texts)} documents to vector store (this might take a while)...")
                chunk_size = 100
                total_chunks = (len(texts) + chunk_size - 1) // chunk_size
                
                for i in range(0, len(texts), chunk_size):
                    chunk_end = min(i + chunk_size, len(texts))
                    chunk_num = i // chunk_size + 1
                    logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({i}-{chunk_end-1})")
                    db.add_texts(
                        texts=texts[i:chunk_end],
                        metadatas=metadatas[i:chunk_end]
                    )
                    logger.info(f"Completed chunk {chunk_num}/{total_chunks}")
                
                logger.info("Persisting vector store to disk...")
                db.persist()
                logger.info("Vector store persisted successfully")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Ingestion completed successfully in {elapsed_time:.2f} seconds")
            return db
        except Exception as e:
            logger.error(f"Error during vector store ingestion: {str(e)}")
            raise