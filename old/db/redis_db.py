import redis
# Corrected imports for redis-py v4+
from redis.commands.search.field import VectorField, TagField, TextField

    # fall back to redis-py 5.x naming
try:
    # redis-py >= 6 (preferred)
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ModuleNotFoundError:
    # fall back to redis-py 5.x naming
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
import uuid
from typing import List
from datetime import datetime, timezone
from core.config import settings

# Constants for Redis index
REDIS_INDEX_NAME = "hot_memories"
VECTOR_DIMENSION = 1536
PREFIX = "mem:hot:"

class RedisRepository:
    def __init__(self):
        """Initializes the Redis connection and ensures the search index exists."""
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False
        )
        self.create_index_if_not_exists()

    def create_index_if_not_exists(self):
        """Creates the RediSearch index for vector search if it doesn't exist."""
        try:
            self.client.ft(REDIS_INDEX_NAME).info()
            print("Redis search index 'hot_memories' already exists.")
        except redis.exceptions.ResponseError:
            print("Creating Redis search index 'hot_memories'...")
            schema = (
                TextField("content"),
                TagField("user_id"),
                TagField("conversation_id"),
                redis.commands.search.field.NumericField("timestamp", as_name="timestamp", sortable=True),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIMENSION,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            # Corrected: IndexType is an enum on the IndexDefinition class itself
            # but is not needed for the constructor in this context.
            # The prefix is passed directly.
            self.client.ft(REDIS_INDEX_NAME).create_index(
                fields=schema, 
                definition=IndexDefinition(prefix=[PREFIX])
            )
            print("Redis index created.")

    def add_hot_memory(self, user_id: str, conversation_id: str, content: str, embedding: List[float], timestamp: datetime) -> str:
        """
        Adds a new memory to the 'hot' cache in Redis.
        Returns the ID of the newly created memory.
        """
        memory_id = str(uuid.uuid4())
        key = f"{PREFIX}{memory_id}"
        
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        memory_data = {
            "content": content,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "timestamp": int(timestamp.timestamp()),
            "embedding": embedding_bytes,
        }
        
        self.client.hset(key, mapping=memory_data)
        self.client.expire(key, 3600) 
        
        return memory_id
    def search_hot_memories(self, query_embedding: np.ndarray, user_id: str, top_k: int = 5) -> List[dict]:
        """
        Performs a vector similarity search on the 'hot' memories in Redis.
        """
        # Prepare the query
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        query_vector = query_embedding_np.tobytes()
        
        # Build the K-Nearest Neighbors (KNN) query
        # This query finds the top_k nearest neighbors for the given vector,
        # filtered by the user_id tag.
        q = (
            Query(f"(@user_id:{{{user_id}}})=>[KNN {top_k} @embedding $query_vec AS vector_score]")
            .sort_by("vector_score")
            .return_fields("content", "conversation_id", "timestamp", "vector_score")
            .dialect(2)
        )
        
        params = {"query_vec": query_vector}
        print(f"--- Executing Redis Search Query: {q.query_string()}") # <-- ADD THIS
        print(f"--- With user_id filter: {user_id}") # <-- ADD THIS
        try:
            results = self.client.ft(REDIS_INDEX_NAME).search(q, query_params=params)
            print(f"--- Redis Search returned {results.total} total results.")
            
            # Format results
            return [
                {
                    "content": doc.content,
                    "score": 1 - float(doc.vector_score),
                    "timestamp": datetime.fromtimestamp(int(doc.timestamp), tz=timezone.utc), # Convert cosine distance to similarity
                    "metadata": {"conversation_id": doc.conversation_id}
                }
                for doc in results.docs
            ]
        except Exception as e:
            print(f"Error searching Redis: {e}")
            return []
# Global instance
redis_repo = RedisRepository()