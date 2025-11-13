from typing import List
from sqlalchemy.orm import Session
from db.models import Interaction, ProceduralMemory, User
from core.llm import llm_interface
from db.models import EpisodicMemory
class PostgresRepository:
    def save_procedural_memory(self, db: Session, proc_mem: ProceduralMemory):
        db.add(proc_mem)
        db.commit()
        db.refresh(proc_mem)
        print(f"Saved new procedural memory: {proc_mem.id}")

    # We will implement the search method later
    def search_procedural_memory(self, db: Session, task_embedding: list[float], similarity_threshold: float = 0.9, top_k: int = 1) -> ProceduralMemory | None:
        """
        Finds the most similar procedural memory using vector search,
        returning it only if it's above the similarity threshold.
        """
        if not task_embedding:
            return None
            
        # Cosine similarity = 1 - cosine_distance.
        # So, we search for distance < (1 - threshold).
        distance_threshold = 1 - similarity_threshold

        # Find the closest procedure
        procedure = db.query(ProceduralMemory).order_by(
            ProceduralMemory.task_embedding.cosine_distance(task_embedding)
        ).first()

        if procedure:
            # Manually calculate the distance to check against the threshold
            # This requires getting the vector back, which can be slow. A better way is to use a WHERE clause.
            # Let's rewrite with a WHERE clause for efficiency.
            
            # Efficient query with a WHERE clause
            result = db.query(ProceduralMemory).filter(
                ProceduralMemory.task_embedding.cosine_distance(task_embedding) < distance_threshold
            ).order_by(
                ProceduralMemory.task_embedding.cosine_distance(task_embedding)
            ).limit(top_k).first()

            return result
        
        return None
    def search_episodic_memories(self, db: Session, user_id: str, query_embedding: list[float], top_k: int = 5) -> List[EpisodicMemory]:
        """
        Searches for long-term episodic memories in PostgreSQL using pgvector.
        """
        if not query_embedding:
            return []
        
        # We need to join through the Interaction table to filter by user_id
        results = (
            db.query(EpisodicMemory)
            .join(Interaction, EpisodicMemory.interaction_id == Interaction.id)
            .filter(Interaction.user_id == user_id)
            .order_by(EpisodicMemory.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )
        return results
    def get_user_by_api_key(self, db: Session, api_key: str) -> User | None:
        return db.query(User).filter(User.api_key == api_key).first()

    def create_user(self, db: Session, name: str) -> User:
        new_user = User(name=name)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
postgres_repo = PostgresRepository()