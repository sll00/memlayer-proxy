from fastapi import APIRouter
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase
import redis

from core.config import settings

router = APIRouter()

@router.get("/status")
def get_status():
    """Checks the connection status of all data stores."""
    status = {
        "postgres": "disconnected",
        "neo4j": "disconnected",
        "redis": "disconnected",
    }

    # Check PostgreSQL
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            if result.scalar() == 1:
                status["postgres"] = "connected"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"

    # Check Neo4j
    try:
        with GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)) as driver:
            driver.verify_connectivity()
            status["neo4j"] = "connected"
    except Exception as e:
        status["neo4j"] = f"error: {str(e)}"

    # Check Redis
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
        if r.ping():
            status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"

    return status