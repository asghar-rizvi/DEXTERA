import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

os.makedirs("./vector_db", exist_ok=True)
client = chromadb.PersistentClient(path="./vector_db")
try:
    client.delete_collection("law_contexts")
    print("Deleted existing collection")
except:
    print("No existing collection to delete")

collection = client.get_or_create_collection("law_contexts")

data_path = "combined_data/RAG Data/contexts_and_sources.json"
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit(1)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

ids, contexts, metadatas = [], [], []
for key, item in data.items():
    ids.append(str(key))
    contexts.append(item["context"])
    metadatas.append({"law_name": item["law_data_set_name"]})

print(f"Processing {len(contexts)} documents...")
embeddings = model.encode(contexts)
collection.add(
    ids=ids,
    documents=contexts,
    metadatas=metadatas,
    embeddings=embeddings
)
print("Vector database created and stored in ./vector_db/")