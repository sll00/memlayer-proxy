from typing import List
from fastapi import APIRouter, Body
from pydantic import BaseModel

from core.cognitive_types import Task
from core.controller import cognitive_controller
from core.retrieval import retrieval_orchestrator

router = APIRouter()

class TaskRequest(BaseModel):
    goal: str
from core.cognitive_types import CognitivePlan

@router.post("/execute-manual-plan", summary="Execute a manually provided plan")
async def execute_manual_plan(task_goal: str = Body(...), plan_data: dict = Body(...)):
    """
    Allows manually injecting a 'perfect' plan to teach the agent.
    """
    task = Task(goal=task_goal)
    plan = CognitivePlan(task_goal=task_goal, **plan_data)
    
    result = await cognitive_controller.execute_task_with_plan(task, plan) # We need to create this helper
    return {"task_id": task.task_id, "status": "completed", "result": result}
@router.post("/tasks", summary="Submit a new task to the Cognitive Controller")
async def submit_task(request: TaskRequest): # <-- Make the function async
    """
    Accepts a high-level goal and passes it to the Cognitive Controller
    to be planned and executed.
    """
    task = Task(goal=request.goal)
    result = await cognitive_controller.execute_task(task) # <-- await the call
    
    return {"task_id": task.task_id, "status": "completed", "result": result}

@router.post("/test-retrieval", summary="Test the retrieval orchestrator")
async def test_retrieval(requests: List[dict] = Body(...)):
    """
    A temporary endpoint to test the retrieval orchestrator directly.
    Example body:
    [
        {"operation": "recall_episodic", "parameters": {"query": "project timeline"}},
        {"operation": "recall_semantic", "parameters": {"entity": "Project Phoenix"}}
    ]
    """
    # Assuming a mock user_id for testing
    user_id = "user123"
    fragments = await retrieval_orchestrator.retrieve(requests, user_id)
    return [fragment.model_dump() for fragment in fragments]