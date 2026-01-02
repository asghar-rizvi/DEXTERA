from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from api.services.rag_service import RAGService
from api.services.llm_service import LLMService
from typing import Optional

router = APIRouter()

rag_service = RAGService()
llm_service = LLMService()

class ChatRequest(BaseModel):
    message: str
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        user_input = request.message
        context = None
        if request.use_rag:
            context = rag_service.retrieve_context(user_input)
                            
        response = await llm_service.generate_response(user_input, context)
        
        return ChatResponse(response="empty")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))