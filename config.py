"""
Configuration file for Insurance Q&A Chatbot
Contains all configuration parameters and settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insurance-qa-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # Empty string = default namespace

# Ollama Configuration
# Default to Mistral 7B Instruct as the primary generation model
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of similar documents to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score

# Chunking Configuration
CHUNK_SIZE = 512  # Maximum characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Dataset Configuration
DATASET_NAME = "deccan-ai/insuranceQA-v2"
DATASET_SPLIT = "train"  # Can be 'train', 'test', or 'valid'

# Streamlit Configuration
APP_TITLE = "Insurance Q&A Chatbot"
APP_ICON = "🏥"
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}


