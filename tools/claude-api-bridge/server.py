"""Claude Code CLI → OpenAI-compatible API bridge.

Wraps `claude -p` (Claude Code's non-interactive mode) as an OpenAI-compatible
REST API with streaming support. This allows any tool that speaks the OpenAI
API format to use Claude Code with your Max subscription's OAuth credentials.

**How it works:**
    1. Receives OpenAI-format /v1/chat/completions requests
    2. Translates messages into a prompt for `claude -p`
    3. For streaming: uses `--output-format stream-json --verbose
       --include-partial-messages` and emits SSE chunks
    4. For non-streaming: uses `--output-format json` and returns
       a standard OpenAI-format response
    5. Tool/function calling: maps OpenAI tools to `--allowedTools`

**Usage:**
    uvicorn server:app --host 0.0.0.0 --port 8080

    Then point any OpenAI SDK client at http://localhost:8080/v1
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

logger = logging.getLogger(__name__)

# Maximum request body size (10 MB). Prevents memory exhaustion from
# oversized payloads sent by malicious or misconfigured clients.
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
    title="Claude Code API Bridge",
    description="OpenAI-compatible API wrapping Claude Code CLI",
    version="0.1.0",
)
app.add_middleware(RequestSizeLimitMiddleware)

# Claude Code CLI binary — configurable via environment variable
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

# Default model mapping: OpenAI model names → Claude CLI --model values.
# Users can pass any string; these are convenience aliases.
MODEL_MAP = {
    "gpt-4": "sonnet",
    "gpt-4o": "sonnet",
    "gpt-4-turbo": "sonnet",
    "gpt-3.5-turbo": "haiku",
    "claude-sonnet": "sonnet",
    "claude-opus": "opus",
    "claude-haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
    "haiku": "haiku",
}

# Available "models" to list
AVAILABLE_MODELS = [
    {"id": "opus", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "sonnet", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
    {"id": "haiku", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
]


def _resolve_model(model: str) -> str:
    """Map OpenAI model name to Claude CLI --model flag value."""
    return MODEL_MAP.get(model, model)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert OpenAI messages array to a single prompt string for claude -p.

    For multi-turn conversations, we format each message with its role
    so Claude understands the conversation history. The system message
    becomes context at the top.
    """
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
            # Tool results from previous function calls
            tool_call_id = msg.get("tool_call_id", "unknown")
            parts.append(f"Tool result ({tool_call_id}): {content}")

    prompt = ""
    if system_parts:
        prompt = "\n".join(system_parts) + "\n\n"
    prompt += "\n\n".join(parts)
    return prompt


def _tools_to_allowed(tools: list[dict[str, Any]] | None) -> list[str]:
    """Map OpenAI tools/functions to Claude --allowedTools list.

    OpenAI tools have type "function" with a name. We map common names
    to Claude Code tool names. For custom functions, we pass them through
    as-is (they'll be available if matching MCP tools are configured).

    NOTE: claude -p requires --mcp-config for tools to actually work.
    The --allowedTools flag only pre-approves built-in tools. For custom
    tools defined via MCP, pass an mcp_config in the request body or set
    the MCP_CONFIG_PATH environment variable.
    """
    if not tools:
        return []

    # Map of OpenAI function names to Claude built-in tool names
    tool_name_map = {
        "bash": "Bash",
        "read_file": "Read",
        "write_file": "Write",
        "edit_file": "Edit",
        "search": "Grep",
        "glob": "Glob",
        "web_search": "WebSearch",
        "web_fetch": "WebFetch",
    }

    allowed: list[str] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            name = fn.get("name", "")
            mapped = tool_name_map.get(name, name)
            allowed.append(mapped)

    return allowed


def _build_claude_cmd(
    prompt: str,
    model: str,
    stream: bool,
    allowed_tools: list[str],
    system_prompt: str | None = None,
    max_turns: int | None = None,
    session_id: str | None = None,
    mcp_config: str | None = None,
) -> list[str]:
    """Build the claude CLI command line."""
    cmd = [CLAUDE_BIN, "-p"]

    # Model selection
    cmd.extend(["--model", model])

    # Output format
    if stream:
        cmd.extend([
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ])
    else:
        cmd.extend(["--output-format", "json"])

    # Tool permissions
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])

    # System prompt override
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    # Session continuity
    if session_id:
        cmd.extend(["--session-id", session_id])

    # Max agentic turns
    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])

    # MCP configuration — required for tool use in claude -p
    # Priority: explicit mcp_config param > MCP_CONFIG_PATH env var
    if mcp_config:
        cmd.extend(["--mcp-config", mcp_config])
    elif os.environ.get("MCP_CONFIG_PATH"):
        cmd.extend(["--mcp-config", os.environ["MCP_CONFIG_PATH"]])

    # The prompt itself is passed via stdin (safer for long/complex prompts)
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


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> JSONResponse:
    """Get a specific model (OpenAI-compatible)."""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return JSONResponse(model)
    return JSONResponse({"error": {"message": f"Model {model_id} not found"}}, status_code=404)


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    """Handle chat completion requests (OpenAI-compatible).

    Translates the request to a `claude -p` CLI call, runs it, and returns
    the response in OpenAI format. Supports both streaming and non-streaming.
    """
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "sonnet")
    stream = body.get("stream", False)
    tools = body.get("tools")
    max_turns = body.get("max_turns")  # Extension: pass through to claude

    # Extract system prompt from messages if present
    system_prompt = None
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
            break

    # Extension fields (not in OpenAI spec, but useful)
    session_id = body.get("session_id")
    mcp_config = body.get("mcp_config")

    model = _resolve_model(model_name)
    allowed_tools = _tools_to_allowed(tools)
    prompt = _messages_to_prompt(messages)
    completion_id = _make_completion_id()

    cmd = _build_claude_cmd(
        prompt=prompt,
        model=model,
        stream=stream,
        allowed_tools=allowed_tools,
        system_prompt=system_prompt if system_prompt and len(messages) > 1 else None,
        max_turns=max_turns,
        session_id=session_id,
        mcp_config=mcp_config,
    )

    if stream:
        return StreamingResponse(
            _stream_claude(cmd, prompt, completion_id, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _run_claude(cmd, prompt, completion_id, model)


async def _run_claude(
    cmd: list[str],
    prompt: str,
    completion_id: str,
    model: str,
) -> JSONResponse:
    """Run claude -p non-streaming and return OpenAI-format response."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=prompt.encode("utf-8"))

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.error("Claude CLI error (exit %d): %s", proc.returncode, error_msg)
            return JSONResponse(
                {"error": {"message": f"Claude CLI error: {error_msg}", "type": "server_error"}},
                status_code=500,
            )

        output = stdout.decode("utf-8", errors="replace").strip()

        # Parse JSON output from claude
        try:
            result = json.loads(output)
            content = result.get("result", output)
            usage = result.get("usage", {})
            openai_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                ),
            }
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            content = output
            openai_usage = None

        return JSONResponse(_make_response(completion_id, model, content, openai_usage))

    except FileNotFoundError:
        return JSONResponse(
            {"error": {"message": f"Claude CLI not found at '{CLAUDE_BIN}'", "type": "server_error"}},
            status_code=500,
        )
    except Exception as e:
        logger.exception("Unexpected error running Claude CLI")
        return JSONResponse(
            {"error": {"message": str(e), "type": "server_error"}},
            status_code=500,
        )


async def _stream_claude(
    cmd: list[str],
    prompt: str,
    completion_id: str,
    model: str,
):
    """Stream claude -p output as OpenAI-compatible SSE chunks.

    Parses the stream-json output from Claude CLI line by line, extracting
    text deltas and converting them to OpenAI's streaming format.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send prompt to stdin and close it
        assert proc.stdin is not None
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        # Emit initial chunk with role
        yield _make_chunk(completion_id, model, {"role": "assistant"})

        # Read stdout line by line and convert to SSE
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
                except json.JSONDecodeError:
                    continue

                # Extract text content from various event formats
                text = _extract_text_from_event(event)
                if text:
                    yield _make_chunk(completion_id, model, {"content": text})

        # Final chunk with finish_reason
        yield _make_chunk(completion_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

        await proc.wait()

    except FileNotFoundError:
        error_chunk = {
            "error": {"message": f"Claude CLI not found at '{CLAUDE_BIN}'"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Error streaming from Claude CLI")
        error_chunk = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _extract_text_from_event(event: dict[str, Any]) -> str | None:
    """Extract text content from a Claude stream-json event.

    Claude CLI stream-json emits events in this structure:
        {"type": "stream_event", "event": {"type": "content_block_delta",
         "delta": {"type": "text_delta", "text": "..."}}}

    We also handle thinking_delta (skip), tool_use blocks (skip),
    and the final result event.
    """
    event_type = event.get("type", "")

    # stream_event wraps the actual API event
    if event_type == "stream_event":
        inner = event.get("event", {})
        inner_type = inner.get("type", "")

        # Content block delta — the main text streaming path
        if inner_type == "content_block_delta":
            delta = inner.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                return delta.get("text")
            # Skip thinking_delta, tool_use deltas, etc.
            return None

        # Direct delta (older format)
        delta = inner.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")

    # Top-level content_block_delta (some versions emit directly)
    if event_type == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")

    # Skip result events — we already got the text from stream deltas.
    # Skip assistant/message events — these are partial message snapshots
    # that contain the full accumulated text (not deltas), which would
    # cause duplication.

    return None


# Health check endpoint
@app.get("/health")
async def health() -> JSONResponse:
    """Health check."""
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
