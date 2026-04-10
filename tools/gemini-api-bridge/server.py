"""Gemini CLI → OpenAI-compatible API bridge.

Wraps `gemini -p` (Gemini CLI's non-interactive mode) as an OpenAI-compatible
REST API with streaming support. This allows any tool that speaks the OpenAI
API format to use Gemini CLI.

**How it works:**
    1. Receives OpenAI-format /v1/chat/completions requests
    2. Translates messages into a prompt for `gemini -p`
    3. For streaming: uses `--output-format stream-json` and emits SSE chunks
    4. For non-streaming: uses `--output-format json` and returns
       a standard OpenAI-format response

**Usage:**
    uvicorn server:app --host 0.0.0.0 --port 8081

    Then point any OpenAI SDK client at http://localhost:8081/v1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum request body size (10 MB).
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", 10 * 1024 * 1024))


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies larger than MAX_REQUEST_BODY_BYTES."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return Response(
                content=json.dumps({
                    "error": {
                        "message": f"Request body too large (max {MAX_REQUEST_BODY_BYTES} bytes)",
                        "type": "invalid_request_error",
                    }
                }),
                status_code=413,
                media_type="application/json",
            )
        return await call_next(request)


app = FastAPI(
    title="Gemini CLI API Bridge",
    description="OpenAI-compatible API wrapping Gemini CLI",
    version="0.1.0",
)
app.add_middleware(RequestSizeLimitMiddleware)

# Gemini CLI binary — configurable via environment variable
GEMINI_BIN = os.environ.get("GEMINI_BIN", "gemini")

# Default model mapping: OpenAI model names → Gemini CLI --model values.
MODEL_MAP = {
    "gpt-4": "gemini-3.1-pro-preview",
    "gpt-4o": "gemini-3.1-pro-preview",
    "gpt-4-turbo": "gemini-3.1-pro-preview",
    "gpt-3.5-turbo": "gemini-1.5-flash",
    "gemini-pro": "gemini-3.1-pro-preview",
    "gemini-flash": "gemini-1.5-flash",
}

# Available "models" to list
AVAILABLE_MODELS = [
    {"id": "gemini-3.1-pro-preview", "object": "model", "created": 1700000000, "owned_by": "google"},
    {"id": "gemini-1.5-flash", "object": "model", "created": 1700000000, "owned_by": "google"},
    {"id": "gemini-1.5-pro", "object": "model", "created": 1700000000, "owned_by": "google"},
]


def _resolve_model(model: str) -> str:
    """Map OpenAI model name to Gemini CLI --model flag value."""
    return MODEL_MAP.get(model, model)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert OpenAI messages array to a single prompt string for gemini -p."""
    parts: list[str] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle content that's a list of parts (multimodal)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part["text"])
            content = "\n".join(text_parts)

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            parts.append(f"Tool result ({tool_call_id}): {content}")

    prompt = ""
    if system_parts:
        prompt = "System Instructions:\n" + "\n".join(system_parts) + "\n\n"
    prompt += "\n\n".join(parts)
    return prompt


def _build_gemini_cmd(
    prompt: str,
    model: str,
    stream: bool,
    yolo: bool = True,
    mcp_config: str | None = None,
) -> list[str]:
    """Build the gemini CLI command line."""
    # We pass the prompt as an argument to -p. 
    # Gemini CLI will execute this and exit in non-interactive mode.
    cmd = [GEMINI_BIN, "-p", prompt]

    if yolo:
        # --yolo is the shorthand for auto-approving all tools
        cmd.append("--yolo")

    # Model selection
    cmd.extend(["--model", model])

    # Output format
    if stream:
        cmd.extend(["--output-format", "stream-json"])
    else:
        cmd.extend(["--output-format", "json"])

    # Optional MCP config for custom tools (parity with Claude bridge)
    if mcp_config:
        cmd.extend(["--mcp-config", mcp_config])

    return cmd


def _make_completion_id() -> str:
    """Generate an OpenAI-style completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _make_response(
    completion_id: str,
    model: str,
    content: str,
    usage: dict[str, int] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Build an OpenAI-format chat completion response."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _make_chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """Build an OpenAI-format SSE chunk."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """List available models (OpenAI-compatible)."""
    return JSONResponse({
        "object": "list",
        "data": AVAILABLE_MODELS,
    })


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    """Handle chat completion requests (OpenAI-compatible)."""
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "gemini-3.1-pro-preview")
    stream = body.get("stream", False)
    mcp_config = body.get("mcp_config")

    model = _resolve_model(model_name)
    prompt = _messages_to_prompt(messages)
    completion_id = _make_completion_id()

    cmd = _build_gemini_cmd(
        prompt=prompt,
        model=model,
        stream=stream,
        mcp_config=mcp_config,
    )

    if stream:
        return StreamingResponse(
            _stream_gemini(cmd, completion_id, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _run_gemini(cmd, completion_id, model)


async def _run_gemini(
    cmd: list[str],
    completion_id: str,
    model: str,
) -> JSONResponse:
    """Run gemini -p non-streaming and return OpenAI-format response."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.error("Gemini CLI error (exit %d): %s", proc.returncode, error_msg)
            return JSONResponse(
                {"error": {"message": f"Gemini CLI error: {error_msg}", "type": "server_error"}},
                status_code=500,
            )

        output = stdout.decode("utf-8", errors="replace").strip()

        # Parse JSON output from gemini
        try:
            result = json.loads(output)
            # Adjust based on actual Gemini CLI JSON structure
            content = result.get("content", result.get("result", output))
            usage = result.get("usage", {})
            openai_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        except json.JSONDecodeError:
            content = output
            openai_usage = None

        return JSONResponse(_make_response(completion_id, model, content, openai_usage))

    except Exception as e:
        logger.exception("Unexpected error running Gemini CLI")
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


async def _stream_gemini(
    cmd: list[str],
    completion_id: str,
    model: str,
):
    """Stream gemini -p output as OpenAI-compatible SSE chunks."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        yield _make_chunk(completion_id, model, {"role": "assistant"})

        assert proc.stdout is not None
        buffer = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break

            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    # Extract text delta. Based on gemini --help it supports stream-json
                    text = event.get("delta", {}).get("text") or event.get("text")
                    if text:
                        yield _make_chunk(completion_id, model, {"content": text})
                except json.JSONDecodeError:
                    continue

        yield _make_chunk(completion_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

        await proc.wait()

    except Exception as e:
        logger.exception("Error streaming from Gemini CLI")
        error_chunk = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
