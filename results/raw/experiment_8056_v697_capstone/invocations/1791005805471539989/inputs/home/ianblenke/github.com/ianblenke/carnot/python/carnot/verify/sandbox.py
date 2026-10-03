"""Sandboxed code execution via gvisor (runsc) Docker runtime.

Provides a secure alternative to in-process exec() for running untrusted
LLM-generated code. Uses Docker with the gvisor runtime (runsc) for
syscall-level isolation — the generated code runs in a lightweight sandbox
with no access to the host filesystem, network, or other processes.

**How it works:**
    1. Writes the code + test harness to a temporary directory
    2. Runs it inside a Docker container with --runtime=runsc
    3. Captures stdout/stderr and parses the result
    4. Falls back to in-process exec if Docker/gvisor unavailable

**Requirements:**
    - Docker installed and running
    - gvisor runtime registered: docker info | grep runsc
    - Python base image available: python:3.11-slim

Spec: REQ-CODE-001, REQ-SECURITY-001
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Docker image for sandboxed execution — slim Python with no extras
_SANDBOX_IMAGE = "python:3.11-slim"
_GVISOR_RUNTIME = "runsc"
_CONTAINER_TIMEOUT_SECONDS = 10
_CONTAINER_STARTUP_GRACE_SECONDS = 5.0
_MAX_CODE_SIZE_BYTES = 100_000  # 100KB limit on code size


def _gvisor_available() -> bool:
    """Check if Docker with gvisor runtime is available."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return _GVISOR_RUNTIME in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _build_harness(code: str, func_name: str, args: tuple[Any, ...]) -> str:
    """Build a self-contained Python script that executes the function
    and prints the result as JSON to stdout.

    The harness catches all exceptions and prints them as structured
    error output so the caller can distinguish success from failure.
    """
    args_json = json.dumps(args)
    code_json = json.dumps(code)
    return f"""\
import json

USER_CODE = json.loads({code_json!r})

try:
    namespace = {{}}
    exec(USER_CODE, namespace)
    func = namespace.get({func_name!r})
    if func is None:
        raise NameError("Function {func_name!r} not found in code")
    args = json.loads({args_json!r})
    result = func(*args)
    print(json.dumps({{
        "status": "ok",
        "result_repr": repr(result),
        "args_repr": repr(args),
    }}))
except Exception as e:
    print(json.dumps({{"status": "error", "error_type": type(e).__name__, "error_msg": str(e)}}))
"""


def _decode_payload(payload: object) -> Any:
    """Decode a literal payload from sandbox JSON when possible."""
    if not isinstance(payload, str):
        return payload
    try:
        return ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return payload


def _sync_mutable_value(target: Any, source: Any) -> None:
    """Apply sandbox-side argument mutations back to caller-owned containers."""
    if isinstance(target, list) and isinstance(source, list):
        target[:] = source
        return
    if isinstance(target, dict) and isinstance(source, dict):
        target.clear()
        target.update(source)
        return
    if isinstance(target, set) and isinstance(source, (set, list, tuple)):
        target.clear()
        target.update(source)
        return
    if isinstance(target, tuple) and isinstance(source, (list, tuple)):
        for child_target, child_source in zip(target, source, strict=False):
            _sync_mutable_value(child_target, child_source)


def _apply_mutated_args(args: tuple[Any, ...], payload: object) -> None:
    """Mirror sandbox-side in-place mutations onto the original argument tuple."""
    decoded_args = _decode_payload(payload)
    if not isinstance(decoded_args, (list, tuple)):
        return
    for original, mutated in zip(args, decoded_args, strict=False):
        _sync_mutable_value(original, mutated)


def sandboxed_exec_function(
    code: str,
    func_name: str,
    args: tuple[Any, ...],
    timeout: float = _CONTAINER_TIMEOUT_SECONDS,
    *,
    force_sandbox: bool = False,
    allow_fallback: bool = True,
) -> tuple[Any, Exception | None]:
    """Execute a Python function in a gvisor-sandboxed Docker container.

    **Researcher summary:**
        Like safe_exec_function but runs the code in an isolated gvisor
        sandbox. The generated code cannot access the host filesystem,
        network, or other processes.

    **Detailed explanation for engineers:**
        Creates a temporary directory with the code + harness, mounts it
        read-only into a Docker container running with --runtime=runsc,
        captures the JSON output, and parses the result.

        If gvisor/Docker is unavailable:
        - allow_fallback=True (default): falls back to in-process exec
          with a warning. This keeps the pipeline working on dev machines.
        - allow_fallback=False: raises RuntimeError. Use this in production
          or when you need to guarantee sandbox isolation.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to call.
        args: Positional arguments to pass.
        timeout: Container execution timeout in seconds.
        force_sandbox: If True, skip the availability check (for testing).
        allow_fallback: If False, raise instead of falling back to exec.

    Returns:
        Tuple of (result, None) on success, or (None, exception) on failure.

    Spec: REQ-CODE-001, REQ-SECURITY-001
    """
    # Size check — reject absurdly large code
    if len(code.encode("utf-8")) > _MAX_CODE_SIZE_BYTES:
        return None, ValueError(
            f"Code exceeds {_MAX_CODE_SIZE_BYTES} byte limit ({len(code.encode('utf-8'))} bytes)"
        )

    # Check sandbox availability
    if not force_sandbox and not _gvisor_available():
        if allow_fallback:
            logger.warning(
                "SECURITY: gvisor sandbox unavailable — falling back to "
                "in-process exec. Set allow_fallback=False to enforce sandbox."
            )
            from carnot.verify.python_types import safe_exec_function

            return safe_exec_function(code, func_name, args, timeout=timeout)
        raise RuntimeError(
            "gvisor sandbox required but unavailable. "
            "Install Docker + gvisor runtime, or set allow_fallback=True."
        )

    # Build the harness script
    harness = _build_harness(code, func_name, args)

    # Write to temp directory
    tmpdir = tempfile.mkdtemp(prefix="carnot_sandbox_")
    try:
        harness_path = Path(tmpdir) / "harness.py"
        harness_path.write_text(harness, encoding="utf-8")

        # Run in gvisor container
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            f"--runtime={_GVISOR_RUNTIME}",
            "--network=none",  # No network access
            "--read-only",  # Read-only root filesystem
            "--tmpfs",
            "/tmp:size=10m",  # Small writable /tmp
            "--memory=256m",  # Memory limit
            "--cpus=1",  # CPU limit
            "--pids-limit=50",  # Process limit
            "-v",
            f"{tmpdir}:/code:ro",  # Mount code read-only
            _SANDBOX_IMAGE,
            "python",
            "/code/harness.py",
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + _CONTAINER_STARTUP_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None, TimeoutError(f"Sandboxed execution timed out after {timeout}s")

        # Parse output
        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()[:500]
            return None, RuntimeError(
                f"Sandbox produced no output (exit {result.returncode}): {stderr}"
            )

        try:
            output = json.loads(stdout)
        except json.JSONDecodeError:
            return None, RuntimeError(f"Sandbox output not valid JSON: {stdout[:200]}")

        if output.get("status") == "ok":
            _apply_mutated_args(args, output.get("args_repr"))
            result_payload = output.get("result_repr", output.get("result"))
            return _decode_payload(result_payload), None
        elif output.get("status") == "error":
            error_type = output.get("error_type", "Exception")
            error_msg = output.get("error_msg", "unknown error")
            # Reconstruct the exception type if it's a builtin
            exc_class = getattr(builtins, error_type, None)
            if exc_class is None or not isinstance(exc_class, type):
                exc_class = RuntimeError
            return None, exc_class(error_msg)
        else:
            return None, RuntimeError(f"Unexpected sandbox output: {stdout[:200]}")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_sandbox_status() -> dict[str, Any]:
    """Report the current sandbox availability and configuration.

    Returns a dict with:
        available: bool — whether gvisor sandbox is ready
        runtime: str — the container runtime being used
        image: str — the Docker image for sandboxed execution
        fallback: str — what happens if sandbox is unavailable
    """
    available = _gvisor_available()
    return {
        "available": available,
        "runtime": _GVISOR_RUNTIME if available else "none",
        "docker": shutil.which("docker") is not None,
        "image": _SANDBOX_IMAGE,
        "fallback": "in-process exec (UNSAFE for untrusted code)",
        "max_code_size_bytes": _MAX_CODE_SIZE_BYTES,
        "container_timeout_seconds": _CONTAINER_TIMEOUT_SECONDS,
    }


__all__ = [
    "get_sandbox_status",
    "sandboxed_exec_function",
]
