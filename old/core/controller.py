from datetime import datetime, timezone
from core.cognitive_types import Task, CognitivePlan, CognitiveStep,WorkingMemory, CognitiveOperation
from core.llm import slm_planner
from core.llm import slm_planner, llm_interface # <-- Add llm_interface
from core.prompts import PLANNER_SYSTEM_PROMPT
from core.retrieval import retrieval_orchestrator # <-- Import the prompt
from pydantic import ValidationError
from db.models import ProceduralMemory
from db.postgres_repo import postgres_repo
from db.database import SessionLocal

class CognitiveController:
    def __init__(self):
        pass
    def reflect_and_learn(self, task: Task, plan: CognitivePlan, result: dict):
        """
        Analyzes a successful task execution and creates a new procedural memory.
        """
        # For now, we'll assume any task that doesn't fail is "successful"
        # In a real system, we'd need more robust success criteria.
        if result.get("status") == "success":
            print(f"Task {task.task_id} successful. Reflecting and creating procedural memory...")
            
            # Generate an embedding for the task goal itself
            task_embedding = llm_interface.get_embedding(task.goal)
            if not task_embedding:
                print("Could not generate embedding for task goal. Skipping procedural memory creation.")
                return

            # Create the ProceduralMemory object
            new_procedure = ProceduralMemory(
                task_description=task.goal,
                task_embedding=task_embedding,
                cognitive_plan=plan.model_dump(), # Store the successful plan as JSON
                success_criteria="Task completed without errors.", # Placeholder
                related_concepts=[] # TODO: Extract concepts from the task goal
            )

            # Save it to the database
            db = SessionLocal()
            try:
                postgres_repo.save_procedural_memory(db, new_procedure)
            finally:
                db.close()     
    def plan(self, task: Task) -> CognitivePlan | None:
        """
        Generates a cognitive plan. First, it tries to recall a known procedure.
        If none is found, it uses the SLM to generate a new plan.
        """
        print(f"Planning for task: {task.goal}")
        
        # --- Step 1: Attempt to recall an existing procedure ---
        task_embedding = llm_interface.get_embedding(task.goal)
        if task_embedding:
            db = SessionLocal()
            try:
                # Search for a procedure with a high similarity (e.g., > 90%)
                found_procedure = postgres_repo.search_procedural_memory(db, task_embedding, similarity_threshold=0.9)
                if found_procedure:
                    print(f"Found highly similar procedural memory: {found_procedure.id}. Reusing successful plan.")
                    # Re-create the CognitivePlan from the stored JSON
                    recalled_plan = CognitivePlan(**found_procedure.cognitive_plan)
                    
                    # Increment usage stats for this procedure
                    found_procedure.usage_count += 1
                    found_procedure.last_used = datetime.now(timezone.utc)
                    db.commit()
                    
                    return recalled_plan
            finally:
                db.close()

        # --- Step 2: If no procedure found, generate a new plan with the SLM ---
        print("No suitable procedure found. Generating new plan with SLM.")
        
        # The user prompt is simply the task's goal.
        user_prompt = task.goal

        try:
            # Call the SLM planner to get the structured JSON plan
            plan_json = slm_planner.generate_json(
                prompt=user_prompt,
                system_prompt=PLANNER_SYSTEM_PROMPT
            )

            if not plan_json:
                print("Error: SLM planner returned an empty response.")
                return None

            # Validate the JSON structure using our Pydantic models.
            # This is a critical step to ensure the SLM is behaving correctly.
            # We expect the JSON to have a "steps" key.
            if "steps" not in plan_json:
                print(f"Error: SLM response is missing 'steps' key. Response: {plan_json}")
                return None

            # Create CognitiveStep objects from the JSON data
            steps = [CognitiveStep(**step_data) for step_data in plan_json["steps"]]
            
            # Assemble the full CognitivePlan
            cognitive_plan = CognitivePlan(task_goal=task.goal, steps=steps)
            
            print(f"Successfully generated plan {cognitive_plan.plan_id} with {len(cognitive_plan.steps)} steps.")
            return cognitive_plan

        except ValidationError as e:
            print(f"Error: SLM output failed Pydantic validation: {e}")
            print(f"Invalid JSON received: {plan_json}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during planning: {e}")
            return None
    async def execute_task_with_plan(self, task: Task, plan: CognitivePlan) -> any:
        """Helper to execute a task with a pre-defined plan."""
        result = await self.execute_plan(plan, task)
        # Wrap the result in a dict format expected by reflect_and_learn
        result_dict = {"status": "success", "result": result}
        self.reflect_and_learn(task, plan, result_dict)
        return result
    async def execute_plan(self, plan: CognitivePlan, task: Task) -> any:
        """
        Executes the steps of a cognitive plan asynchronously, prioritizing an optimized
        path that combines reasoning and generation into a single LLM call.
        """
        print(f"Executing plan {plan.plan_id} for task: {task.goal}")
        working_memory = WorkingMemory(task)
        
        # --- Step 1: Gather all RECALL requests and execute them in parallel ---
        recall_steps = [
            step for step in plan.steps 
            if step.operation in [
                CognitiveOperation.RECALL_EPISODIC,
                CognitiveOperation.RECALL_SEMANTIC,
                CognitiveOperation.RECALL_PROCEDURAL
            ]
        ]
        
        if recall_steps:
            print(f"Found {len(recall_steps)} recall operations. Retrieving in parallel...")
            recall_requests = [step.model_dump() for step in recall_steps]
            user_id = task.context.get("user_id")
            if not user_id:
                raise ValueError("user_id must be provided in the task context for retrieval operations.")
            
            retrieved_fragments = await retrieval_orchestrator.retrieve(recall_requests, user_id)
            working_memory.add_fragments(retrieved_fragments)
            print(f"Retrieved {len(retrieved_fragments)} memory fragments.")

        # --- Step 2: Sequentially execute non-recall steps (REASON, GENERATE, etc.) ---
        final_response = "No result was generated."
        
        for step in plan.steps:
            # Skip recall steps as they have already been processed
            if step.operation in [CognitiveOperation.RECALL_EPISODIC, CognitiveOperation.RECALL_SEMANTIC, CognitiveOperation.RECALL_PROCEDURAL]:
                continue

            # --- Handle discrete REASON steps (the less optimal, but still supported path) ---
            if step.operation == CognitiveOperation.REASON:
                print(f"Executing discrete REASON step with local SLM: {step.parameters.get('thought_process')}")
                
                context_str = working_memory.get_context_for_llm()
                system_prompt = f"You are a reasoning engine. You will be given context and a task. Perform the task and output only the result, with no extra conversational text.\n\n--- CONTEXT ---\n{context_str}"
                prompt = f"Task: {step.parameters.get('thought_process')}"
                
                thought = slm_planner.generate_text(prompt, system_prompt)
                
                if thought:
                    print(f"SLM generated thought: {thought}")
                    working_memory.add_thought(thought)
                else:
                    print("Warning: REASON step failed to produce a thought.")
                    working_memory.add_thought("The reasoning step failed to produce a result.")

            # --- Handle the GENERATE step (the optimized, primary path) ---
            elif step.operation == CognitiveOperation.GENERATE:
                print(f"Executing combined GENERATE step with powerful LLM...")
                context_str = working_memory.get_context_for_llm()
                
                # Extract parameters from the plan step
                reasoning_instructions = step.parameters.get("reasoning_instructions")
                output_format = step.parameters.get("output_format", "a clear and concise answer.") # Default format
                
                # Build the prompt, including reasoning instructions if they exist
                prompt = f"Based on the provided context, generate a final response to the original goal: '{task.goal}'."
                if reasoning_instructions:
                    prompt += f"\nFirst, perform the following reasoning: '{reasoning_instructions}'."
                prompt += f"\nFinally, format your output as: '{output_format}'."
                
                # Call the powerful external LLM for the final, high-quality output
                final_response = llm_interface.generate_response(prompt, context_str)
                working_memory.final_result = final_response
                print("GENERATE step complete.")

            # TODO: Add handlers for other future operations like EXECUTE_TOOL
            # elif step.operation == CognitiveOperation.EXECUTE_TOOL:
            #     ...

        # Return the final result from the working memory
        return working_memory.final_result or final_response

    async def execute_task(self, task: Task) -> any:
        """
        The main async entry point for handling a task from start to finish.
        """
        # 1. Generate a plan
        plan = self.plan(task) # This is synchronous, which is fine.
        if not plan:
            return {"status": "failed", "reason": "Planning failed."}

        # 2. Execute the plan
        result = await self.execute_plan(plan, task)
        
        # 3. Reflect and learn - wrap result in dict format
        result_dict = {"status": "success", "result": result}
        self.reflect_and_learn(task, plan, result_dict)

        # 4. Consolidate this experience (Future milestone, but we can placeholder it)
        # memory_ingestor.observe(...)

        return result_dict

# Global instance of the controller
cognitive_controller = CognitiveController()