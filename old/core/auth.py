from fastapi import Security, Depends, HTTPException
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from starlette import status

from db.database import get_db
from db.models import User
from db.postgres_repo import postgres_repo # We will add the required method to this repo

# Define the API key header
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

async def get_current_user(
    api_key: str = Security(API_KEY_HEADER), db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency to authenticate a user via their API key.
    The key is expected in the 'Authorization' header, e.g., "Authorization: <key>"
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing",
        )
    
    # In a real app, you might strip a "Bearer " prefix, but for simple keys, this is fine.
    
    user = postgres_repo.get_user_by_api_key(db, api_key=api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return user