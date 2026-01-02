import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B")
    ADAPTER_PATH = os.getenv("ADAPTER_PATH")
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "2"))
    
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "4"))