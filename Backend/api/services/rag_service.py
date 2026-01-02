from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from api.utils.config import Config
import os

class RAGService:
    def __init__(self):
        if not os.path.exists(Config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector database not found at {Config.VECTOR_DB_PATH}")

        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} )
        
        self.vector_store = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=embeddings
        )
   
    def retrieve_context(self, query: str) -> str:
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=Config.RAG_TOP_K)
        
        THRESHOLD = 0.6 
        valid_docs = []
        for doc, score in results_with_scores:
            if score <= THRESHOLD:
                valid_docs.append(doc.page_content)

        if not valid_docs:
            print("No relevant legal context found above threshold. Skipping RAG")
            return ""
        
        return "\n\n".join(valid_docs)