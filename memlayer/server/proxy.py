"""
Memlayer Proxy - OpenAI-compatible reverse proxy with memory capabilities.

This server acts as a drop-in replacement for the OpenAI API, routing requests
through llama-server while adding persistent memory via Memlayer.
"""

from typing import Dict, Optional, AsyncGenerator
import json
import uuid
import logging
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    Choice,
    Message,
    StreamChoice,
    DeltaMessage,
    Usage,
    ErrorResponse,
    ErrorDetail,
)
from .config import ServerConfig, get_config
from ..wrappers.llama_server import LlamaServer
from ..embedding_models import LocalEmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemlayerProxy:
    """
    OpenAI-compatible reverse proxy that adds memory capabilities to llama-server.

    Features:
    - 100% offline operation (local embeddings)
    - Per-user memory isolation via X-User-ID header
    - Shared embedding model across all users (performance optimization)
    - Streaming support
    - OpenAI-compatible API
    """

    def __init__(
        self,
        llama_server_host: str = "http://localhost:8080",
        llama_server_port: Optional[int] = None,
        storage_path: str = "./memlayer_data",
        config: Optional[ServerConfig] = None,
    ):
        """
        Initialize the Memlayer proxy.

        Args:
            llama_server_host: URL of llama-server instance
            llama_server_port: Port number (optional)
            storage_path: Path for memory storage
            config: ServerConfig instance (optional, uses get_config() if None)
        """
        self.config = config or get_config()
        self.llama_server_host = llama_server_host or self.config.llama_server_host
        self.llama_server_port = llama_server_port or self.config.llama_server_port
        self.storage_path = storage_path or self.config.storage_path

        # Cache of LlamaServer wrappers per user_id
        self._user_clients: Dict[str, LlamaServer] = {}

        # Shared embedding model across all users (singleton pattern)
        self._shared_embedding_model: Optional[LocalEmbeddingModel] = None

        logger.info(f"MemlayerProxy initialized")
        logger.info(f"  llama-server: {self.llama_server_host}")
        logger.info(f"  Storage: {self.storage_path}")
        logger.info(f"  Mode: {self.config.operation_mode} (offline)")

    def _get_shared_embedding_model(self) -> LocalEmbeddingModel:
        """
        Get or create the shared embedding model.

        This model is shared across all users to save memory and initialization time.
        """
        if self._shared_embedding_model is None:
            logger.info("Loading shared LocalEmbeddingModel (sentence-transformers)...")
            self._shared_embedding_model = LocalEmbeddingModel()
            logger.info(f"Embedding model ready (dimension: {self._shared_embedding_model.dimension})")
        return self._shared_embedding_model

    def _get_client(self, user_id: str, model: str) -> LlamaServer:
        """
        Get or create a LlamaServer wrapper for a specific user.

        Wrappers are cached per user_id for performance. The embedding model
        is shared across all users.
        """
        if user_id not in self._user_clients:
            logger.info(f"Creating new LlamaServer wrapper for user: {user_id}")

            # Get shared embedding model first
            embedding_model = self._get_shared_embedding_model()

            # Create wrapper with shared embedding model
            client = LlamaServer(
                host=self.llama_server_host,
                port=self.llama_server_port,
                model=model,
                storage_path=f"{self.storage_path}/{user_id}",  # Per-user storage
                user_id=user_id,
                embedding_model=embedding_model,  # Shared across users
                salience_threshold=self.config.salience_threshold,
                operation_mode="local",  # Always local
                scheduler_interval_seconds=self.config.scheduler_interval,
                curation_interval_seconds=self.config.curation_interval,
                max_concurrent_consolidations=self.config.max_concurrent_consolidations,
            )

            self._user_clients[user_id] = client

        return self._user_clients[user_id]

    def create_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application.

        Returns:
            Configured FastAPI app instance
        """
        app = FastAPI(
            title="Memlayer Server",
            description="OpenAI-compatible API with persistent memory",
            version="0.1.0",
        )

        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            """Health check endpoint"""
            return {
                "service": "memlayer-server",
                "status": "ready",
                "llama_server": self.llama_server_host,
                "mode": "offline (local embeddings)",
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            raw_request: Request,
            x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
        ):
            """
            OpenAI-compatible chat completions endpoint with memory.

            Supports:
            - Standard chat completions
            - Tool calling (function calling)
            - Streaming responses
            - Per-user memory via X-User-ID header
            """
            try:
                # Debug: Log all headers
                logger.info(f"[PROXY] Received headers: {dict(raw_request.headers)}")
                logger.info(f"[PROXY] x_user_id from Header(): {x_user_id}")

                # Try multiple header variations (case-insensitive)
                user_id_from_header = (
                    x_user_id or
                    raw_request.headers.get("x-user-id") or
                    raw_request.headers.get("X-User-ID") or
                    raw_request.headers.get("X-User-Id")
                )

                # Extract user ID (from header or use default)
                user_id = user_id_from_header or self.config.default_user_id

                logger.info(f"[PROXY] Using user_id: {user_id} (from_header={user_id_from_header}, default={self.config.default_user_id})")

                # Get or create client for this user
                client = self._get_client(user_id, request.model)

                # Debug: Show request details
                if request.tools:
                    print(f"\n[PROXY] === Incoming Request with Tools ===")
                    print(f"[PROXY] Model: {request.model}")
                    print(f"[PROXY] Tools: {len(request.tools)}")
                    print(f"[PROXY] Tool choice: {request.tool_choice}")
                    print(f"[PROXY] Streaming: {request.stream}")
                    if request.messages:
                        last_msg = request.messages[-1].content
                        print(f"[PROXY] Last message: {last_msg[:100] if last_msg else 'None'}...")
                    print(f"[PROXY] =====================================\n")
                else:
                    print(f"\n[PROXY] === Incoming Request WITHOUT Tools ===")
                    print(f"[PROXY] Model: {request.model}")
                    print(f"[PROXY] Streaming: {request.stream}")
                    if request.messages:
                        last_msg = request.messages[-1].content
                        print(f"[PROXY] Last message: {last_msg[:100] if last_msg else 'None'}...")
                    print(f"[PROXY] =========================================\n")

                # Convert Pydantic messages to dict format
                messages = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        **({"name": msg.name} if msg.name else {}),
                        **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                        **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
                    }
                    for msg in request.messages
                ]

                # Handle streaming
                if request.stream:
                    print(f"[PROXY] Creating StreamingResponse, about to call _stream_response")
                    print(f"[PROXY] DEBUG: request.tools before creating generator = {bool(request.tools)}, count={len(request.tools) if request.tools else 0}", flush=True)
                    if request.tools:
                        print(f"[PROXY] DEBUG: First tool name = {request.tools[0].function.name if request.tools else 'N/A'}", flush=True)
                    return StreamingResponse(
                        _stream_response(self, client, messages, request),
                        media_type="text/event-stream",
                    )

                # Non-streaming response
                # Convert Pydantic Tool models to dicts for the wrapper
                tools_list = None
                if request.tools:
                    tools_list = [tool.dict() for tool in request.tools]
                    print(f"[PROXY] Forwarding {len(tools_list)} tools to LlamaServer wrapper")
                    for tool in tools_list:
                        print(f"[PROXY]   - {tool.get('function', {}).get('name', 'unknown')}")

                response_text = client.chat(
                    messages=messages,
                    temperature=request.temperature,
                    stream=False,
                    tools=tools_list,
                    tool_choice=request.tool_choice,
                )

                # Build OpenAI-compatible response
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

                # Check if there are pending tool calls (custom tools that need client-side execution)
                tool_calls = None
                finish_reason = "stop"
                if hasattr(client, '_pending_tool_calls') and client._pending_tool_calls:
                    tool_calls = client._pending_tool_calls
                    finish_reason = "tool_calls"
                    print(f"[PROXY] Returning {len(tool_calls)} tool calls to client")
                    for tc in tool_calls:
                        print(f"[PROXY]   - {tc.get('function', {}).get('name', 'unknown')}")
                    # Clear pending tool calls
                    client._pending_tool_calls = None
                else:
                    print(f"[PROXY] No tool calls in response (finish_reason: {finish_reason})")

                response = ChatCompletionResponse(
                    id=completion_id,
                    model=request.model,
                    choices=[
                        Choice(
                            index=0,
                            message=Message(
                                role="assistant",
                                content=response_text,
                                tool_calls=tool_calls,
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=0,  # llama-server doesn't provide this
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                )

                return response

            except Exception as e:
                logger.error(f"Error in chat_completions: {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            message=str(e),
                            type="internal_server_error",
                            code="memlayer_error",
                        )
                    ).dict(),
                )

        async def _stream_response(
            self,
            client: LlamaServer,
            messages: list,
            request: ChatCompletionRequest,
        ) -> AsyncGenerator[str, None]:
            """Generate streaming response in OpenAI SSE format"""
            print(f"[PROXY-STREAM] _stream_response called, request.tools={bool(request.tools)}", flush=True)
            try:
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

                # Convert Pydantic Tool models to dicts for the wrapper
                tools_list = None
                if request.tools:
                    tools_list = [tool.dict() for tool in request.tools]
                    print(f"[PROXY-STREAM] Forwarding {len(tools_list)} tools to LlamaServer wrapper", flush=True)
                    for tool in tools_list:
                        print(f"[PROXY-STREAM]   - {tool.get('function', {}).get('name', 'unknown')}", flush=True)

                # Get streaming generator from client
                logger.info(f"[PROXY-STREAM] About to call client.chat(), tools_list={tools_list is not None}, len={len(tools_list) if tools_list else 0}")
                print(f"[PROXY-STREAM] About to call client.chat(), tools_list={tools_list is not None}, len={len(tools_list) if tools_list else 0}", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                response_generator = client.chat(
                    messages=messages,
                    temperature=request.temperature,
                    stream=True,
                    tools=tools_list,
                    tool_choice=request.tool_choice,
                )
                print(f"[PROXY-STREAM] client.chat() returned generator", flush=True)

                # Stream chunks in OpenAI format
                for chunk_text in response_generator:
                    chunk = ChatCompletionStreamChunk(
                        id=completion_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(content=chunk_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.json()}\n\n"

                # Check if there are pending tool calls (custom tools)
                finish_reason = "stop"
                if hasattr(client, '_pending_tool_calls') and client._pending_tool_calls:
                    print(f"[PROXY-STREAM] Returning {len(client._pending_tool_calls)} tool calls to client")
                    for tc in client._pending_tool_calls:
                        print(f"[PROXY-STREAM]   - {tc.get('function', {}).get('name', 'unknown')}")
                    # Stream tool calls
                    for tool_call in client._pending_tool_calls:
                        tool_chunk = ChatCompletionStreamChunk(
                            id=completion_id,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaMessage(tool_calls=[tool_call]),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {tool_chunk.json()}\n\n"

                    finish_reason = "tool_calls"
                    # Clear pending tool calls
                    client._pending_tool_calls = None

                # Send final chunk with finish_reason
                final_chunk = ChatCompletionStreamChunk(
                    id=completion_id,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield f"data: {final_chunk.json()}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "internal_server_error",
                        "code": "memlayer_stream_error",
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        @app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on server shutdown"""
            logger.info("Shutting down Memlayer proxy...")
            for user_id, client in self._user_clients.items():
                logger.info(f"Closing client for user: {user_id}")
                try:
                    client.close()
                except Exception as e:
                    logger.error(f"Error closing client for {user_id}: {e}")

        return app
