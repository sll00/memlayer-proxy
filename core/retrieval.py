from typing import List
import asyncio

import numpy as np

from core.cognitive_types import MemoryFragment
from core.llm import llm_interface
from db.postgres_repo import postgres_repo
from db.database import SessionLocal
from db.redis_db import redis_repo
from db.graph_db import neo4j_repo 

class RetrievalOrchestrator:
    async def search_episodic(self, query: str, user_id: str) -> List[MemoryFragment]:
        """Searches both hot (Redis) and cold (Postgres) episodic memories."""
        print(f"Searching all episodic memories for: '{query}'")
        query_embedding = llm_interface.get_embedding(query)
        if not query_embedding:
            return []

        # --- Search Hot Cache (Redis) ---
        hot_results = redis_repo.search_hot_memories(np.array(query_embedding), user_id, top_k=3)
        hot_fragments = [
            MemoryFragment(source="episodic_hot", timestamp=res["timestamp"], **res) for res in hot_results
        ]

        # --- Search Cold Storage (Postgres) ---
        db = SessionLocal()
        try:
            # This is a synchronous DB call, but it's fast enough.
            # For extreme performance, this could be run in a thread pool executor.
            cold_results = postgres_repo.search_episodic_memories(db, user_id, query_embedding, top_k=5)
        finally:
            db.close()

        cold_fragments = []
        for mem in cold_results:
            # Calculate similarity score from distance
            # Note: This requires getting the vector back, which is inefficient.
            # A better approach is to calculate distance in the DB and return it.
            # For now, we'll assign a slightly lower base score.
            cold_fragments.append(MemoryFragment(
                source="episodic_cold",
                content=mem.content,
                score=0.8, # Assign a default high score for retrieved cold memories
                timestamp=mem.timestamp,
                metadata={"interaction_id": str(mem.interaction_id)}
            ))

        # --- Merge and Deduplicate Results ---
        all_fragments = hot_fragments + cold_fragments
        # A simple deduplication based on content
        unique_fragments = {frag.content: frag for frag in all_fragments}.values()
        
        return list(unique_fragments)

    async def search_semantic(self, entity: str, user_id: str) -> List[MemoryFragment]:
        """Searches the semantic knowledge graph."""
        print(f"Searching semantic memory for entity: '{entity}'")
        results = neo4j_repo.search_semantic_entities(entity)
        return [
            MemoryFragment(source="semantic_graph", **res) for res in results
        ]

    async def search_procedural(self, task: str, user_id: str) -> List[MemoryFragment]:
        """Searches for procedural memories."""
        print(f"Searching procedural memory for task: '{task}' (mocked)")
        # The procedural search is now real, so we should call it.
        # But if we were mocking it, we would ensure it returns a list.
        
        # Let's update this to use our new REAL procedural search
        db = SessionLocal()
        try:
            task_embedding = llm_interface.get_embedding(task)
            if not task_embedding:
                return [] # Return empty list if embedding fails
            
            procedure = postgres_repo.search_procedural_memory(db, task_embedding)
            if procedure:
                return [
                    MemoryFragment(
                        source="procedural_cold",
                        content=f"Found procedure for task: {procedure.task_description}",
                        score=1.0, # We can assign a perfect score for a direct procedural hit
                        timestamp=procedure.last_used or procedure.created_at,
                        metadata={"procedure_id": str(procedure.id)}
                    )
                ]
        finally:
            db.close()
            
        return []

    async def retrieve(self, recall_requests: List[dict], user_id: str) -> List[MemoryFragment]:
        """
        Takes a list of recall requests from a cognitive plan and executes them in parallel.
        """
        tasks = []
        for request in recall_requests:
            op = request.get("operation")
            params = request.get("parameters", {})
            
            if op == "recall_episodic" and "query" in params:
                tasks.append(self.search_episodic(params["query"], user_id))
            elif op == "recall_semantic" and "entity" in params:
                tasks.append(self.search_semantic(params["entity"], user_id))
            elif op == "recall_procedural" and "task" in params:
                tasks.append(self.search_procedural(params["task"], user_id))

        # Run all search tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single list of fragments
        all_fragments = [fragment for sublist in results_list for fragment in sublist]
        
        # TODO: Implement reranking here
        
        return all_fragments

# Global instance
retrieval_orchestrator = RetrievalOrchestrator()