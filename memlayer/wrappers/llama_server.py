from typing import Any, Dict, List, Optional, TYPE_CHECKING
import openai
import json
import re
import dateutil.parser
from ..config import is_debug_mode

# Use TYPE_CHECKING to avoid slow imports at module load time
if TYPE_CHECKING:
    from ..ml_gate import SalienceGate
    from ..storage.chroma import ChromaStorage
    from ..storage.networkx import NetworkXStorage
    from ..services import SearchService, ConsolidationService, CurationService
    from ..embedding_models import BaseEmbeddingModel, LocalEmbeddingModel
    from ..observability import Trace
    from .base import BaseLLMWrapper
else:
    SalienceGate = None
    ChromaStorage = None
    NetworkXStorage = None
    SearchService = None
    ConsolidationService = None
    BaseEmbeddingModel = None
    LocalEmbeddingModel = None
    CurationService = None
    Trace = None
    BaseLLMWrapper = object


class LlamaServer(BaseLLMWrapper):
    """
    A memory-enhanced llama-server client that can be used standalone.

    This wrapper connects to llama-server (llama.cpp's OpenAI-compatible server)
    and adds persistent memory capabilities. Designed for 100% offline operation
    using local sentence-transformers for embeddings.

    Usage:
        from memlayer.wrappers.llama_server import LlamaServer

        client = LlamaServer(
            host="http://localhost:8080",
            model="qwen2.5:7b",
            storage_path="./my_memories",
            user_id="user_123"
        )

        response = client.chat(messages=[
            {"role": "user", "content": "My name is Alice"}
        ])
    """

    def __init__(
        self,
        host: str = "http://localhost:8080",
        port: Optional[int] = None,
        model: str = "model",
        temperature: float = 0.7,
        storage_path: str = "./memlayer_data",
        user_id: str = "default_user",
        embedding_model: Optional["BaseEmbeddingModel"] = None,
        salience_threshold: float = 0.0,
        operation_mode: str = "local",  # Force local for offline operation
        scheduler_interval_seconds: int = 60,
        curation_interval_seconds: int = 3600,
        max_concurrent_consolidations: int = 2,
        **kwargs
    ):
        """
        Initialize a memory-enhanced llama-server client.

        Args:
            host: Base URL of llama-server (e.g., "http://localhost:8080")
            port: Port number (if None, extracted from host URL)
            model: Model name/identifier for llama-server
            temperature: Sampling temperature (0.0 to 2.0)
            storage_path: Path where memories will be stored
            user_id: Unique identifier for the user
            embedding_model: Custom embedding model (defaults to LocalEmbeddingModel)
            salience_threshold: Threshold for memory worthiness (-0.1 to 0.2, default 0.0)
            operation_mode: Always "local" for offline operation with sentence-transformers
            scheduler_interval_seconds: Task reminder check interval
            curation_interval_seconds: Memory curation interval
            **kwargs: Additional arguments passed to OpenAI client
        """
        self.model = model
        self.temperature = temperature
        self.user_id = user_id
        self.storage_path = storage_path
        self.salience_threshold = salience_threshold
        # Force local mode for offline operation
        self.operation_mode = "local"
        self._provided_embedding_model = embedding_model
        self.scheduler_interval_seconds = scheduler_interval_seconds
        self.curation_interval_seconds = curation_interval_seconds
        self.max_concurrent_consolidations = max_concurrent_consolidations

        # Parse host and port
        self.host = host
        if port is not None:
            self.base_url = f"{host}:{port}/v1"
        elif "/v1" in host:
            self.base_url = host
        else:
            self.base_url = f"{host}/v1"

        # Lazy-loaded attributes
        self._embedding_model = None
        self._vector_storage = None
        self._graph_storage = None
        self._salience_gate = None
        self._search_service = None
        self._consolidation_service = None
        self._curation_service = None
        self._scheduler_service = None

        # Tool call tracking for proxy mode
        self._pending_tool_calls = None
        self.last_trace = None

        # Initialize OpenAI client pointing to llama-server (lightweight, fast)
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="not-needed",  # llama-server doesn't require authentication
            **kwargs
        )

        # Tool schema for function calling (OpenAI format)
        self.tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Searches the user's long-term memory for information from previous conversations. "
                        "Use this when the user asks about past interactions, personal details, preferences, or tasks they've mentioned before."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A specific and detailed question or search query for the Memlayer."
                            },
                            "search_tier": {
                                "type": "string",
                                "enum": ["fast", "balanced", "deep"],
                                "description": (
                                    "The desired depth of the search. "
                                    "'fast' (<100ms) for simple factual recall, "
                                    "'balanced' (<500ms) for general queries (default), "
                                    "'deep' (<2s) for comprehensive research with knowledge graph traversal."
                                )
                            }
                        },
                        "required": ["query", "search_tier"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_task",
                    "description": (
                        "Schedules a task or reminder for the user at a future date and time. "
                        "The system will automatically surface this reminder when the time comes."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "A detailed, self-contained description of the task or reminder."
                            },
                            "due_date": {
                                "type": "string",
                                "description": (
                                    "The future date and time, preferably in ISO 8601 format "
                                    "(e.g., '2025-12-25T09:00:00') or natural language (e.g., 'tomorrow at 3pm')."
                                )
                            }
                        },
                        "required": ["task_description", "due_date"]
                    }
                }
            }
        ]

        # For observability
        self.last_trace = None

        print(f"LlamaServer client initialized (100% offline mode)")
        print(f"  - llama-server: {self.base_url}")
        print(f"  - Model: {self.model}")
        print(f"  - Storage: {self.storage_path}")
        print(f"  - Embeddings: Local (sentence-transformers)")

    # Lazy-loaded properties
    @property
    def embedding_model(self) -> "BaseEmbeddingModel":
        """Lazy-load the embedding model (always local for offline operation)"""
        if self._embedding_model is None:
            if self._provided_embedding_model is not None:
                self._embedding_model = self._provided_embedding_model
            else:
                # Always use local embeddings for offline operation
                from ..embedding_models import LocalEmbeddingModel
                print("Loading LocalEmbeddingModel (sentence-transformers)...")
                self._embedding_model = LocalEmbeddingModel()
                print(f"Embedding model ready (dimension: {self._embedding_model.dimension})")
        return self._embedding_model

    @property
    def vector_storage(self) -> Optional["ChromaStorage"]:
        """Lazy-load vector storage"""
        if self._vector_storage is None:
            from ..storage.chroma import ChromaStorage
            self._vector_storage = ChromaStorage(
                self.storage_path,
                dimension=self.embedding_model.dimension
            )
        return self._vector_storage

    @property
    def graph_storage(self) -> "NetworkXStorage":
        """Lazy-load graph storage"""
        if self._graph_storage is None:
            from ..storage.networkx import NetworkXStorage
            self._graph_storage = NetworkXStorage(self.storage_path)
        return self._graph_storage

    @property
    def salience_gate(self) -> "SalienceGate":
        """Lazy-load salience gate"""
        if self._salience_gate is None:
            from ..ml_gate import SalienceGate, SalienceMode
            self._salience_gate = SalienceGate(
                threshold=self.salience_threshold,
                embedding_model=self.embedding_model,  # Use local model
                mode=SalienceMode.LOCAL
            )
        return self._salience_gate

    @property
    def search_service(self) -> "SearchService":
        """Lazy-load search service"""
        if self._search_service is None:
            from ..services import SearchService
            self._search_service = SearchService(
                self.vector_storage,
                self.graph_storage,
                self.embedding_model
            )
        return self._search_service

    @property
    def consolidation_service(self) -> "ConsolidationService":
        """Lazy-load consolidation service"""
        if self._consolidation_service is None:
            from ..services import ConsolidationService
            self._consolidation_service = ConsolidationService(
                self.vector_storage,
                self.graph_storage,
                self.embedding_model,
                self.salience_gate,
                llm_client=self,
                max_concurrent_consolidations=self.max_concurrent_consolidations
            )
        return self._consolidation_service

    @property
    def curation_service(self) -> "CurationService":
        """Lazy-load and auto-start curation service"""
        if self._curation_service is None:
            from ..services import CurationService
            self._curation_service = CurationService(
                self.vector_storage,
                self.graph_storage,
                interval_seconds=self.curation_interval_seconds
            )
            self._curation_service.start()
        return self._curation_service

    @property
    def scheduler_service(self) -> "SchedulerService":
        """Lazy-load and auto-start scheduler service"""
        if self._scheduler_service is None:
            from ..services import SchedulerService
            self._scheduler_service = SchedulerService(
                self.graph_storage,
                check_interval_seconds=self.scheduler_interval_seconds
            )
            self._scheduler_service.start()
        return self._scheduler_service

    def chat(self, messages: list, stream: bool = False, **kwargs) -> str:
        """
        Main chat method with memory integration and tool calling support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional arguments for llama-server

        Returns:
            The assistant's response text (or generator if streaming)
        """
        # Ensure curation service is running
        _ = self.curation_service

        # Ensure scheduler service is running
        _ = self.scheduler_service

        # Extract user query from the last message
        user_query = messages[-1]['content'] if messages else ""

        # Check if custom tools are provided (proxy mode with external client)
        client_tools = kwargs.get("tools", [])
        using_custom_tools = bool(client_tools)

        # Handle memory context injection
        triggered_context = self.search_service.get_triggered_tasks_context(self.user_id)

        if triggered_context:
            if using_custom_tools:
                # For custom tools (proxy mode), append memory to existing system message
                # to avoid overriding client's tool descriptions
                system_msg_found = False
                for msg in messages:
                    if msg.get("role") == "system":
                        # Append memory context to existing system message
                        msg["content"] = msg["content"] + f"\n\n## Relevant Context from Memory:\n{triggered_context}"
                        print("[WRAPPER] Memory context appended to existing system message")
                        system_msg_found = True
                        break

                if not system_msg_found:
                    # No system message found, add memory as separate system message
                    messages.insert(0, {"role": "system", "content": f"## Relevant Context from Memory:\n{triggered_context}"})
                    print("[WRAPPER] Memory context added as new system message")
            else:
                # For built-in memlayer usage, inject at the beginning as before
                messages.insert(0, {"role": "system", "content": triggered_context})

        # Consolidate user message BEFORE LLM call (background thread)
        # Keep consolidation even with custom tools - it stores memories without interfering
        if True:  # Always consolidate to maintain memory
            # Convert first-person to third-person for better extraction
            consolidated_text = user_query
            consolidated_text = re.sub(r'\bMy\s+', 'The user\'s ', consolidated_text, flags=re.IGNORECASE)
            consolidated_text = re.sub(r'\bI\s+am\b', 'The user is', consolidated_text, flags=re.IGNORECASE)
            consolidated_text = re.sub(r'\bI\s+', 'The user ', consolidated_text)
            consolidated_text = re.sub(r'\bme\b', 'the user', consolidated_text, flags=re.IGNORECASE)

            self.consolidation_service.consolidate(consolidated_text, self.user_id)

        # Handle streaming
        if stream:
            return self._stream_chat(messages, user_query, kwargs)

        # Prepare kwargs for OpenAI client
        client_tool_choice = kwargs.get("tool_choice", "auto")

        # If client provided tools, use them; otherwise use memlayer's built-in tools
        tools_to_use = client_tools if client_tools else self.tool_schema
        # using_custom_tools already calculated above

        # Always show tool info when tools are present
        if tools_to_use:
            print(f"[WRAPPER] Using {'custom' if using_custom_tools else 'built-in'} tools: {len(tools_to_use)} tools")
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools_to_use]
            print(f"[WRAPPER] Tool names: {tool_names}")

        completion_kwargs = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
            "tools": tools_to_use,
            "tool_choice": client_tool_choice,
            "stream": False,
        }

        # First LLM call with tool availability
        try:
            response = self.client.chat.completions.create(**completion_kwargs)
            response_message = response.choices[0].message

            # Always show response when tools were sent
            if tools_to_use:
                print(f"[WRAPPER] Response from llama-server:")
                print(f"[WRAPPER]   Content: {response_message.content[:100] if response_message.content else None}...")
                print(f"[WRAPPER]   Tool calls: {response_message.tool_calls}")

        except Exception as e:
            print(f"Error calling llama-server: {e}")
            return f"Error: Could not connect to llama-server at {self.base_url}. Please ensure the server is running."

        # Check if LLM called any tools
        if not response_message.tool_calls:
            # No tool call, direct response
            final_response = response_message.content
        else:
            # If using custom tools, return tool calls to client for execution
            if using_custom_tools:
                # Store tool calls in a format the proxy can use
                self._pending_tool_calls = [
                    {
                        "index": idx,
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for idx, tc in enumerate(response_message.tool_calls)
                ]
                # Return indicator that tool calls need client-side execution
                return response_message.content or ""

            # Handle built-in memlayer tool calls - convert to dict for JSON serialization
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]
            })

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "search_memory":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        search_tier = function_args.get("search_tier", "balanced")

                        if is_debug_mode():
                            print(f"[DEBUG] Tool call: search_memory(query='{query}', tier='{search_tier}')")

                        # Execute search
                        search_output = self.search_service.search(
                            query=query,
                            user_id=self.user_id,
                            search_tier=search_tier,
                            llm_client=self
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]

                        # Append tool result
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": search_result_text,
                        })

                    except Exception as e:
                        print(f"Error during search_memory: {e}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: Could not search memory. {str(e)}"
                        })

                elif function_name == "schedule_task":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        description = function_args.get("task_description")
                        due_date_str = function_args.get("due_date")

                        if is_debug_mode():
                            print(f"[DEBUG] Tool call: schedule_task(desc='{description}', due='{due_date_str}')")

                        # Parse due date
                        due_timestamp = dateutil.parser.parse(due_date_str).timestamp()

                        # Store task in graph
                        task_id = self.graph_storage.add_task(description, due_timestamp, self.user_id)

                        tool_response = (
                            f"Task successfully scheduled with ID: {task_id}. "
                            f"Description: {description}. Due: {due_date_str}."
                        )

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_response,
                        })

                    except Exception as e:
                        print(f"Error during schedule_task: {e}")
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: Could not schedule task. {str(e)}"
                        })

                else:
                    # Unknown tool
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error: Unknown tool '{function_name}'"
                    })

            # Second LLM call with tool results (without tools parameter)
            second_kwargs = {
                "model": self.model,
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": messages,
                "stream": False,
            }

            try:
                second_response = self.client.chat.completions.create(**second_kwargs)
                final_response = second_response.choices[0].message.content
            except Exception as e:
                print(f"Error in second LLM call: {e}")
                final_response = "I encountered an error processing the tool results."

        return final_response

    def _stream_chat(self, messages: list, user_query: str, kwargs: dict):
        """Handle streaming responses with tool calling support"""
        # Merge client tools with built-in memlayer tools
        client_tools = kwargs.get("tools", [])
        client_tool_choice = kwargs.get("tool_choice", "auto")

        # If client provided tools, use them; otherwise use memlayer's built-in tools
        tools_to_use = client_tools if client_tools else self.tool_schema
        using_custom_tools = bool(client_tools)  # Track if using custom tools

        # Always show tool info when tools are present (STREAMING)
        if tools_to_use:
            print(f"[WRAPPER-STREAM] Using {'custom' if using_custom_tools else 'built-in'} tools: {len(tools_to_use)} tools")
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools_to_use]
            print(f"[WRAPPER-STREAM] Tool names: {tool_names}")

        completion_kwargs = {
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
            "tools": tools_to_use,
            "tool_choice": client_tool_choice,
            "stream": True,
        }

        try:
            stream_response = self.client.chat.completions.create(**completion_kwargs)
        except Exception as e:
            print(f"Error calling llama-server: {e}")
            yield f"Error: Could not connect to llama-server at {self.base_url}"
            return

        # Buffer for tool calls and content
        tool_calls_buffer = []
        full_response = ""
        finish_reason = None

        # Stream chunks
        print(f"[WRAPPER-STREAM] Starting to iterate over stream_response", flush=True)
        chunk_count = 0
        for chunk in stream_response:
            chunk_count += 1
            if not chunk.choices:
                print(f"[WRAPPER-STREAM] Chunk {chunk_count}: no choices, skipping", flush=True)
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            print(f"[WRAPPER-STREAM] Chunk {chunk_count}: has_content={bool(delta.content)}, has_tool_calls={bool(delta.tool_calls)}, finish_reason={finish_reason}", flush=True)

            # Stream content
            if delta.content:
                full_response += delta.content
                yield delta.content

            # Buffer tool calls
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    idx = tool_call_chunk.index

                    # Extend buffer if needed
                    while len(tool_calls_buffer) <= idx:
                        tool_calls_buffer.append({
                            "index": len(tool_calls_buffer),
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })

                    # Accumulate tool call data
                    if tool_call_chunk.id:
                        tool_calls_buffer[idx]["id"] = tool_call_chunk.id
                    if tool_call_chunk.function:
                        if tool_call_chunk.function.name:
                            tool_calls_buffer[idx]["function"]["name"] += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            tool_calls_buffer[idx]["function"]["arguments"] += tool_call_chunk.function.arguments

        # Handle tool calls if present
        print(f"[WRAPPER-STREAM] Stream loop completed. chunk_count={chunk_count}, tool_calls_count={len(tool_calls_buffer)}, finish_reason={finish_reason}", flush=True)
        if tool_calls_buffer:
            # If using custom tools, don't execute them - let client handle
            if using_custom_tools:
                # Store tool calls for proxy to return to client
                self._pending_tool_calls = tool_calls_buffer
                # Don't execute, just return
                return

            # Create a mock response message object for built-in tool execution
            class ToolCall:
                def __init__(self, data):
                    self.id = data["id"]
                    self.type = data["type"]
                    self.function = type('obj', (object,), {
                        'name': data["function"]["name"],
                        'arguments': data["function"]["arguments"]
                    })

            class ResponseMessage:
                def __init__(self, tool_calls):
                    self.content = None
                    self.tool_calls = [ToolCall(tc) for tc in tool_calls]

            response_message = ResponseMessage(tool_calls_buffer)

            # Convert to dict format for message history (JSON serializable)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_buffer
            })

            # Execute built-in memlayer tool calls
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name

                if function_name == "search_memory":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        query = function_args.get("query")
                        search_tier = function_args.get("search_tier", "balanced")

                        search_output = self.search_service.search(
                            query=query,
                            user_id=self.user_id,
                            search_tier=search_tier,
                            llm_client=self
                        )
                        search_result_text = search_output["result"]
                        self.last_trace = search_output["trace"]

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": search_result_text,
                        })
                    except Exception as e:
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })

                elif function_name == "schedule_task":
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        description = function_args.get("task_description")
                        due_date_str = function_args.get("due_date")

                        due_timestamp = dateutil.parser.parse(due_date_str).timestamp()
                        task_id = self.graph_storage.add_task(description, due_timestamp, self.user_id)

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Task scheduled with ID: {task_id}"
                        })
                    except Exception as e:
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })

            # Second streaming call with tool results
            second_kwargs = {
                "model": self.model,
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": messages,
                "stream": True,
            }

            try:
                second_stream = self.client.chat.completions.create(**second_kwargs)
                for chunk in second_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield f"\nError processing tool results: {str(e)}"

    def analyze_and_extract_knowledge(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract structured knowledge from text using llama-server.

        Args:
            text: The conversation text to analyze

        Returns:
            Dictionary with 'facts', 'entities', and 'relationships' keys
        """
        from datetime import datetime

        current_datetime = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

        system_prompt = f"""You are a Knowledge Graph Engineer AI. Your task is to analyze text and extract structured information.

**Current Date/Time**: {current_datetime}

Extract the following from the text:

1. **facts**: Atomic, declarative statements of information
   - Include "fact" (string), "importance_score" (0.0-1.0), and "expiration_date" (ISO 8601 or null)
   - Example: {{"fact": "The temporary door code is 1234", "importance_score": 0.8, "expiration_date": "2025-11-18T14:30:00Z"}}

2. **entities**: Key nouns (people, places, organizations, projects, concepts)
   - Include "name" (string) and "type" (e.g., "Person", "Organization", "Project", "Concept")
   - Example: {{"name": "Alice", "type": "Person"}}

3. **relationships**: Connections between entities (subject-predicate-object triples)
   - Include "subject", "predicate" (verb/relationship), and "object"
   - Example: {{"subject": "Alice", "predicate": "works_at", "object": "TechCorp"}}

Respond ONLY with valid JSON in this exact format:
{{
  "facts": [...],
  "entities": [...],
  "relationships": [...]
}}

Do not include any explanatory text, only the JSON object."""

        prompt = f"{system_prompt}\n\nInput Text:\n{text}\n\nYour JSON Output:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for structured output
                stream=False
            )

            response_text = response.choices[0].message.content

            # Check if response is empty
            if not response_text or not response_text.strip():
                print(f"Warning: Empty response from LLM during knowledge extraction")
                return {
                    "facts": [{"fact": text, "importance_score": 0.5, "expiration_date": None}],
                    "entities": [],
                    "relationships": []
                }

            # Handle wrapped JSON (e.g., ```json...```)
            if "```json" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            elif "```" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            else:
                # Try to find JSON object in the response
                json_start = response_text.find("{")
                if json_start != -1:
                    # Use JSONDecoder to parse only the first valid JSON object
                    # This handles reasoning models that output JSON followed by additional text
                    try:
                        decoder = json.JSONDecoder()
                        knowledge_graph, end_idx = decoder.raw_decode(response_text[json_start:])
                        # Successfully parsed, skip to the validation below
                        knowledge_graph.setdefault("facts", [])
                        knowledge_graph.setdefault("entities", [])
                        knowledge_graph.setdefault("relationships", [])
                        return knowledge_graph
                    except json.JSONDecodeError:
                        # Fall through to try regular parsing
                        pass

                # Fallback: extract from first { to last }
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end].strip()

            # Try to parse JSON with regular json.loads()
            try:
                knowledge_graph = json.loads(response_text)
            except json.JSONDecodeError as je:
                print(f"JSON decode error: {je}")
                print(f"Response text (first 200 chars): {response_text[:200]}")
                # Fallback: return text as a simple fact
                return {
                    "facts": [{"fact": text, "importance_score": 0.5, "expiration_date": None}],
                    "entities": [],
                    "relationships": []
                }

            # Ensure all required keys exist
            knowledge_graph.setdefault("facts", [])
            knowledge_graph.setdefault("entities", [])
            knowledge_graph.setdefault("relationships", [])

            return knowledge_graph

        except Exception as e:
            print(f"Error during knowledge extraction: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return text as a simple fact
            return {
                "facts": [{"fact": text, "importance_score": 0.5, "expiration_date": None}],
                "entities": [],
                "relationships": []
            }

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Extract key entities from a search query for graph traversal.

        Args:
            query: The search query

        Returns:
            List of entity names
        """
        prompt = f"""Extract the key entities (people, places, projects, concepts, organizations) from this query.
Return ONLY a JSON array of entity names, nothing else.

Query: {query}

JSON array:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=False
            )

            response_text = response.choices[0].message.content

            # Extract JSON array from response
            if "```json" in response_text or "```" in response_text:
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                if json_start != -1 and json_end > json_start:
                    response_text = response_text[json_start:json_end]

            entities = json.loads(response_text)
            return entities if isinstance(entities, list) else []

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def close(self):
        """Gracefully shut down background services"""
        print("Closing LlamaServer client...")
        if self._curation_service:
            self._curation_service.stop()
        if self._scheduler_service:
            self._scheduler_service.stop()
