from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from db.postgres_repo import postgres_repo
from core.auth import get_current_user

router = APIRouter(prefix="/v1/users", tags=["Users"])

class UserCreateRequest(BaseModel):
    name: str

class UserResponse(BaseModel):
    id: str
    name: str
    api_key: str

@router.post("", response_model=UserResponse, summary="Create a new user")
def create_user(request: UserCreateRequest, db: Session = Depends(get_db)):
    """
    Creates a new user and returns their details, including a newly generated API key.
    This is the only time the API key is returned.
    """
    user = postgres_repo.create_user(db, name=request.name)
    return UserResponse(id=str(user.id), name=user.name, api_key=user.api_key)

@router.get("/me", response_model=UserResponse, summary="Get current user details")
def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the details of the user associated with the provided API key.
    """
    return UserResponse(id=str(current_user.id), name=current_user.name, api_key="********")