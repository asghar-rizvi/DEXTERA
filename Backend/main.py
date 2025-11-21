from fastapi import FastAPI
from api.routes import chat
from api.utils.config import Config
import uvicorn

app = FastAPI(
    title="Pakistani Legal AI Assistant API",
    description="API for fine-tuned Llama model with RAG capabilities for Pakistani Criminal Law",
    version="1.0.0"
)

app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pakistani Legal AI Assistant API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        # workers=Config.WORKERS,
        log_level="info"
    )