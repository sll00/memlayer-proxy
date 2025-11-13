from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid

from db.database import get_db
from db.models import User
from core.auth import get_current_user
from core.controller import cognitive_controller
from core.cognitive_types import Task

router = APIRouter(prefix="/v1/conversations", tags=["Conversations"])

class ConversationCreateResponse(BaseModel):
    conversation_id: str

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str

@router.post("", response_model=ConversationCreateResponse, summary="Create a new conversation")
def create_conversation(current_user: User = Depends(get_current_user)):
    """
    Creates a new conversation context for the authenticated user.
    """
    # For now, a conversation is just a UUID. We can add more to it later.
    return ConversationCreateResponse(conversation_id=str(uuid.uuid4()))

@router.post("/{conversation_id}/messages", response_model=MessageResponse, summary="Send a message")
async def send_message(
    conversation_id: str,
    request: MessageRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Sends a message within a conversation and gets a memory-augmented response.
    This is the primary interaction endpoint.
    """
    # The context now includes user and conversation info
    task_context = {
        "user_id": str(current_user.id),
        "conversation_id": conversation_id
    }
    task = Task(goal=request.message, context=task_context)
    
    result = await cognitive_controller.execute_task(task)
    
    if result.get("status") == "success":
        return MessageResponse(response=result.get("result", "No response generated."))
    else:
        raise HTTPException(status_code=500, detail=result.get("reason", "Task execution failed"))