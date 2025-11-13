from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid

class CognitiveOperation(str, Enum):
    """Enumeration of the fundamental operations the agent can perform."""
    RECALL_EPISODIC = "recall_episodic"      # Remember specific past events/interactions.
    RECALL_SEMANTIC = "recall_semantic"      # Retrieve factual knowledge from the graph.
    RECALL_PROCEDURAL = "recall_procedural"  # Recall how to perform a task.
    REASON = "reason"                        # Step-by-step thinking, analysis, or synthesis.
    GENERATE = "generate"                    # Produce a final response for the user.
    EXECUTE_TOOL = "execute_tool"            # Use an external or internal tool (e.g., search memory).
    REFLECT = "reflect"                      # Analyze the outcome of a plan.

class CognitiveStep(BaseModel):
    """A single step in a cognitive plan."""
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    operation: CognitiveOperation
    parameters: Dict[str, Any]
    rationale: str

class CognitivePlan(BaseModel):
    """The full sequence of cognitive steps to achieve a task."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    task_goal: str
    steps: List[CognitiveStep]

class Task(BaseModel):
    """Represents a high-level goal for the agent to accomplish."""
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    goal: str
    context: Optional[Dict[str, Any]] = None
    status: str = "pending"

class MemoryFragment(BaseModel):
    """A standardized representation of a piece of retrieved memory."""
    source: str # e.g., "episodic_hot", "episodic_cold", "semantic_graph"
    content: str
    score: float # The initial relevance score from the source (e.g., cosine similarity)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkingMemory:
    """A buffer to hold context during a single cognitive task."""
    def __init__(self, task: Task):
        self.task = task
        self.retrieved_fragments: List[MemoryFragment] = []
        self.intermediate_thoughts: List[str] = []
        self.final_result: Any = None

    def add_fragments(self, fragments: List[MemoryFragment]):
        self.retrieved_fragments.extend(fragments)

    def add_thought(self, thought: str):
        self.intermediate_thoughts.append(thought)

    def get_context_for_llm(self) -> str:
        """Formats the entire working memory into a string for an LLM prompt."""
        context = f"Goal: {self.task.goal}\n\n"
        
        if self.retrieved_fragments:
            context += "--- Retrieved Memories ---\n"
            # Sort by score, highest first
            sorted_fragments = sorted(self.retrieved_fragments, key=lambda f: f.score, reverse=True)
            for frag in sorted_fragments[:5]: # Limit to top 5 for context window
                context += f"Source: {frag.source}, Score: {frag.score:.2f}\n"
                context += f"Content: {frag.content}\n---\n"
        
        if self.intermediate_thoughts:
            context += "\n--- Intermediate Thoughts ---\n"
            for thought in self.intermediate_thoughts:
                context += f"- {thought}\n"
        
        return context.strip()