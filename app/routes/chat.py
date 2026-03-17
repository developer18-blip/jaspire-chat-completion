"""
OpenAI-compatible /v1/chat/completions endpoint.

Features:
  - stream=true  -> Server-Sent Events (SSE)
  - stream=false -> Single JSON response
  - Automatic web search via DuckDuckGo
  - search_web=false to skip search
  - conversation_id for persistent memory across requests
  - Sources/URLs returned in response
"""

import time
import uuid
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ResponseMessage,
    Choice,
    Usage,
    DeltaMessage,
    StreamChoice,
    Source,
    HealthResponse,
)
from app.services.agent import stream_agent_response, get_agent_response
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI-Compatible"])


# ---------------------------------------------------------------------------
# SSE stream generator
# ---------------------------------------------------------------------------

async def _sse_stream(
    request: ChatCompletionRequest,
    completion_id: str,
) -> AsyncGenerator[str, None]:
    """Yield OpenAI-format SSE chunks then [DONE]."""
    created = int(time.time())
    model = request.model or settings.vllm_model_name

    conv_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:12]}"

    # First chunk: announce role + conversation_id
    first = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        conversation_id=conv_id,
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=""),
            finish_reason=None,
        )],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    # Get the streaming generator + metadata
    generator, meta = await stream_agent_response(
        openai_messages=request.messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 1024,
        with_search=request.search_web,  # None = auto-detect
        conversation_id=request.conversation_id,
        user_id=request.user,
    )

    # Stream tokens
    async for token in generator:
        if not token:
            continue
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            conversation_id=meta.conversation_id,
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(content=token),
                finish_reason=None,
            )],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk: mark finish
    final = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        conversation_id=meta.conversation_id,
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop",
        )],
    )
    yield f"data: {final.model_dump_json()}\n\n"

    # Send sources as a custom SSE event (clients that don't need it can ignore)
    if meta.sources:
        import json
        sources_data = {
            "sources": meta.sources,
            "conversation_id": meta.conversation_id,
            "search_performed": meta.search_performed,
        }
        yield f"data: {json.dumps(sources_data)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    **OpenAI-compatible chat completions with memory and web search.**

    Extra fields (ignored by OpenAI SDKs):
    - `conversation_id`: Pass to maintain memory across requests.
      Omit for stateless mode. The response includes the assigned ID.
    - `search_web`: Set false to skip web search (default: true).

    Response extras:
    - `conversation_id`: Use this in your next request to continue the conversation.
    - `sources`: List of web search results used (title, url, snippet).
    - `search_performed`: Whether web search was actually run.
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="messages array cannot be empty",
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        # ── Streaming ────────────────────────────────────────────────────
        if request.stream:
            return StreamingResponse(
                _sse_stream(request, completion_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Non-streaming ────────────────────────────────────────────────
        result = await get_agent_response(
            openai_messages=request.messages,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1024,
            with_search=request.search_web,  # None = auto-detect
            conversation_id=request.conversation_id,
            user_id=request.user,
        )

        # Build sources list
        sources = [
            Source(title=s["title"], url=s["url"], snippet=s.get("snippet", ""))
            for s in result.sources
        ] if result.sources else None

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model or settings.vllm_model_name,
            choices=[Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=result.answer),
                finish_reason="stop",
            )],
            usage=Usage(),
            conversation_id=result.conversation_id,
            sources=sources,
            search_performed=result.search_performed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[/v1/chat/completions] {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Supporting endpoints
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models():
    """List available models — OpenAI-compatible format."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.vllm_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "jaspire",
                "capabilities": {
                    "streaming": True,
                    "web_search": True,
                    "vision": True,
                    "conversation_memory": True,
                },
            }
        ],
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — returns current configuration."""
    return HealthResponse(
        status="healthy",
        vllm_url=settings.crawl4ai_llm_base_url,
        model=settings.vllm_model_name,
    )


@router.get("/")
async def root():
    return {
        "message": "JASPIRE Chat API — OpenAI Compatible",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoint": "POST /v1/chat/completions",
        "features": [
            "streaming",
            "web_search",
            "multi_turn",
            "conversation_memory",
            "sources",
        ],
    }
