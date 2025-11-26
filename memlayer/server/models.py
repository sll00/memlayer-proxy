"""
Pydantic models for OpenAI API compatibility.

These models ensure the server accepts and returns OpenAI-compatible
request/response formats.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
import time


# ============================================================================
# Request Models
# ============================================================================

class Message(BaseModel):
    """A chat message in OpenAI format"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call definition"""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call in OpenAI format"""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """Function definition for tool schema"""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    """Tool definition in OpenAI format"""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"

    class Config:
        extra = "allow"  # Allow additional fields


# ============================================================================
# Response Models
# ============================================================================

class Choice(BaseModel):
    """A choice in the response"""
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response format"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    class Config:
        extra = "allow"


# ============================================================================
# Streaming Response Models
# ============================================================================

class DeltaMessage(BaseModel):
    """Delta message for streaming"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """A choice in a streaming response"""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionStreamChunk(BaseModel):
    """OpenAI streaming chunk format"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None

    class Config:
        extra = "allow"


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI error response format"""
    error: ErrorDetail
