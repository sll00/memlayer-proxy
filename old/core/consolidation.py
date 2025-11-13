from db.database import SessionLocal
from db.models import RawObservation, Conversation, Interaction, EpisodicMemory
from core.llm import llm_interface
from core.celery_app import celery_app
from db.redis_db import redis_repo
from datetime import datetime, timezone # <-- Add timezone import
from core.prompts import SEMANTIC_EXTRACTION_SYSTEM_PROMPT
from core.llm import slm_planner
from db.graph_db import neo4j_repo # Import the Neo4j repository

# This is now a Celery task
@celery_app.task(name="tasks.process_observation")
@celery_app.task(name="tasks.process_observation")
def process_observation_task(observation_id: str):
    """
    The main background task for consolidating an observation into memory.
    This function performs a multi-stage consolidation:
    1. Creates a durable Episodic Memory in PostgreSQL.
    2. Uses the local SLM to extract semantic entities and relationships.
    3. Populates the Neo4j knowledge graph with the extracted semantics.
    """
    db = SessionLocal()
    try:
        # 1. Fetch the raw observation
        observation = db.query(RawObservation).filter(RawObservation.id == observation_id).first()
        if not observation:
            print(f"Error: Observation with ID {observation_id} not found.")
            return

        observation.status = "processing"
        db.commit()

        episodic_memory = None

        # --- Stage A: Create Episodic Memory (Durable Log) ---
        # This logic is specific to the source of the observation.
        if observation.source == "user_chat":
            data = observation.meta_data
            conversation_id = data.get("conversation_id")
            user_id = data.get("user_id")
            user_message = data.get("user_message")
            assistant_message = data.get("assistant_message")

            # Find or create the parent Conversation
            conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                conversation = Conversation(id=conversation_id, user_id=user_id)
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

            # Create the Interaction record
            interaction = Interaction(
                conversation_id=conversation.id,
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            # Create the core Episodic Memory
            memory_content = f"User: {user_message}\nAssistant: {assistant_message}"
            embedding = llm_interface.get_embedding(memory_content)

            if embedding:
                episodic_memory = EpisodicMemory(
                    interaction_id=interaction.id,
                    content=memory_content,
                    embedding=embedding,
                    timestamp=interaction.timestamp,
                )
                db.add(episodic_memory)
                db.commit()
                db.refresh(episodic_memory)
                print(f"Successfully created episodic memory {episodic_memory.id} in PostgreSQL.")

        # --- Stages B & C require a valid episodic memory to have been created ---
        if episodic_memory:
            # --- Stage B: Semantic Fact Extraction (using local SLM) ---
            print(f"Extracting semantic facts from episodic memory {episodic_memory.id} using local SLM...")
            
            extracted_data = slm_planner.generate_json(
                prompt=episodic_memory.content,
                system_prompt=SEMANTIC_EXTRACTION_SYSTEM_PROMPT
            )

            # --- Stage C: Graph Population ---
            if extracted_data:
                print("Populating knowledge graph...")
                neo4j_repo.add_entities_and_relationships(
                    extracted_data=extracted_data,
                    source_episode_id=str(episodic_memory.id)
                )
            else:
                print(f"Local SLM did not return valid data for semantic extraction from observation {observation_id}.")
        else:
            print(f"Skipping semantic extraction for observation {observation_id} as no episodic memory was created.")

        # Mark the observation as completed
        observation.status = "completed"
        db.commit()
        print(f"Successfully processed observation {observation_id}")

    except Exception as e:
        db.rollback()
        # Mark observation as failed in a new session to ensure the update is not rolled back
        error_db = SessionLocal()
        try:
            failed_observation = error_db.query(RawObservation).filter(RawObservation.id == observation_id).first()
            if failed_observation:
                failed_observation.status = "failed"
                error_db.commit()
        finally:
            error_db.close()
        print(f"Failed to process observation {observation_id}: {e}")
        # Re-raise the exception so Celery can track it as a failure
        raise

    finally:
        db.close()

class MemoryIngestor:
    def observe(self, db: SessionLocal, source: str, data: str, meta_data: dict) -> RawObservation: # type: ignore
        """
        Performs a hybrid consolidation:
        1. Synchronously creates a 'hot' memory in Redis for immediate access.
        2. Asynchronously queues deep consolidation to Postgres/Neo4j.
        """
        # --- Start of Synchronous "Hot" Path ---
        if source == "user_chat":
            # For chat, the 'data' is the full interaction content
            content = data
            embedding = llm_interface.get_embedding(content)
            
            if embedding:
                # Write to Redis immediately
                redis_repo.add_hot_memory(
                    user_id=meta_data.get("user_id"),
                    conversation_id=meta_data.get("conversation_id"),
                    content=content,
                    embedding=embedding,
                    timestamp=datetime.now(timezone.utc)
                )
                print(f"Hot memory for conversation {meta_data.get('conversation_id')} added to Redis.")
        # --- End of Synchronous "Hot" Path ---

        # --- Start of Asynchronous "Cold" Path ---
        # 1. Create the RawObservation record for the background worker
        observation = RawObservation(
            source=source,
            data=data,
            meta_data=meta_data
        )
        db.add(observation)
        db.commit()
        db.refresh(observation)

        # 2. Trigger the background task
        process_observation_task.delay(str(observation.id))
        print(f"Queued observation {observation.id} for deep consolidation.")
        
        return observation

# Global instance
memory_ingestor = MemoryIngestor()