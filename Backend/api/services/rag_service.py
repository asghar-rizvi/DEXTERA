import chromadb
from sentence_transformers import SentenceTransformer
from api.utils.config import Config
import os

class RAGService:
    def __init__(self):
        if not os.path.exists(Config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector database not found at {Config.VECTOR_DB_PATH}")
        
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        try:
            self.collection = self.client.get_collection("law_contexts")
        except Exception as e:
            raise RuntimeError(f"Error accessing collection: {e}")
        
        self.model = SentenceTransformer(Config.RAG_MODEL_NAME)
    
    def retrieve_context(self, query: str) -> str:
        query_embedding = self.model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=Config.RAG_TOP_K
        )
        
        if not results["documents"]:
            return ""
        
        contexts = [doc for doc in results["documents"][0]]
        return "\n\n".join(contexts)