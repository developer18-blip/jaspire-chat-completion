from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any, Literal
import time


# ---------------------------------------------------------------------------
# OpenAI-compatible request models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """OpenAI-compatible message object"""
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible /v1/chat/completions request.
    Drop-in replacement — same payload as ChatGPT / OpenAI SDK.
    """
    model: Optional[str] = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    # Custom flags
    search_web: Optional[bool] = None  # None = auto-detect, True = force search, False = skip
    conversation_id: Optional[str] = None  # For memory — ties messages together


# ---------------------------------------------------------------------------
# Shared models (used by both streaming and non-streaming)
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Source(BaseModel):
    """A web search source."""
    title: str
    url: str
    snippet: str = ""


# ---------------------------------------------------------------------------
# Streaming response models
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    """SSE chunk — identical shape to OpenAI streaming chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[Usage] = None
    conversation_id: Optional[str] = None
    sources: Optional[List[Source]] = None
    search_performed: Optional[bool] = None


# ---------------------------------------------------------------------------
# Non-streaming response models
# ---------------------------------------------------------------------------

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """Non-streaming response — identical shape to OpenAI + extras"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    # JASPIRE extras (ignored by OpenAI SDKs)
    conversation_id: Optional[str] = None
    sources: Optional[List[Source]] = None
    search_performed: bool = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    vllm_url: str
    model: str
