import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from core.consolidation import memory_ingestor # Import the new ingestor

router = APIRouter()

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    user_id: str
    message: str

class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    observation_id: str

@router.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Handles a chat request, gets a mocked response, and logs the interaction
    as an observation for asynchronous processing.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Mock LLM response generation
    assistant_response = f"Mocked response to '{request.message}'"

    try:
        # The metadata needed by the consolidation task
        observation_meta = {
            "conversation_id": conversation_id,
            "user_id": request.user_id,
            "user_message": request.message,
            "assistant_message": assistant_response,
        }

        # Use the new ingestor to observe the interaction
        observation = memory_ingestor.observe(
            db=db,
            source="user_chat",
            data=f"User: {request.message}\nAssistant: {assistant_response}",
            meta_data=observation_meta
        )

        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_response,
            observation_id=str(observation.id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")