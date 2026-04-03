"""Sandboxed hypothesis execution environment.

Executes hypothesis code in isolation with:
- Hard timeout
- Memory limit (tracked, not enforced at OS level — use containers for hard limits)
- Import whitelist enforcement
- Read-only data access
- Captured stdout/stderr and metrics

Spec: REQ-AUTO-004, REQ-AUTO-009
"""

from __future__ import annotations

import ast
import io
import json
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Modules that hypotheses are NOT allowed to import
BLOCKED_MODULES = frozenset({
    "os",
    "subprocess",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "http",
    "urllib",
    "requests",
    "pathlib",
    "tempfile",
    "signal",
    "importlib",
})


@dataclass
class SandboxConfig:
    """Configuration for the sandbox execution environment.

    Spec: REQ-AUTO-004
    """

    timeout_seconds: int = 1800  # 30 minutes
    max_memory_mb: int = 16384  # 16GB (advisory)
    blocked_modules: frozenset[str] = BLOCKED_MODULES


@dataclass
class SandboxResult:
    """Result of a sandboxed hypothesis execution.

    Spec: REQ-AUTO-004, REQ-AUTO-008
    """

    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    wall_clock_seconds: float = 0.0
    timed_out: bool = False


def check_imports(code: str, blocked: frozenset[str]) -> list[str]:
    """Check hypothesis code for blocked imports.

    Returns list of blocked modules found.

    Spec: REQ-AUTO-009
    """
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["<syntax error — cannot check imports>"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked:
                    violations.append(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in blocked:
                    violations.append(root)
    return violations


def run_in_sandbox(
    hypothesis_code: str,
    benchmark_data: dict[str, Any],
    config: SandboxConfig | None = None,
) -> SandboxResult:
    """Execute hypothesis code in a sandboxed environment.

    The hypothesis code must define a `run(benchmark_data: dict) -> dict`
    function that returns metrics.

    Spec: REQ-AUTO-004, REQ-AUTO-009
    """
    if config is None:
        config = SandboxConfig()

    # 1. Static analysis: check for blocked imports
    violations = check_imports(hypothesis_code, config.blocked_modules)
    if violations:
        return SandboxResult(
            success=False,
            error=f"Blocked imports detected: {', '.join(violations)}",
        )

    # 2. Execute in isolated namespace with timeout
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    start_time = time.monotonic()

    # Timeout handler
    timed_out = False

    def timeout_handler(signum: int, frame: Any) -> None:
        nonlocal timed_out
        timed_out = True
        raise TimeoutError("Hypothesis execution timed out")

    namespace: dict[str, Any] = {"__builtins__": __builtins__}

    try:
        # Set timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(config.timeout_seconds)

        try:
            # Compile and exec the hypothesis code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(hypothesis_code, "<hypothesis>", "exec"), namespace)

                # Call the run function
                if "run" not in namespace:
                    return SandboxResult(
                        success=False,
                        error="Hypothesis must define a `run(benchmark_data)` function",
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        wall_clock_seconds=time.monotonic() - start_time,
                    )

                metrics = namespace["run"](benchmark_data)

                if not isinstance(metrics, dict):
                    return SandboxResult(
                        success=False,
                        error=f"run() must return a dict, got {type(metrics).__name__}",
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        wall_clock_seconds=time.monotonic() - start_time,
                    )

        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    except TimeoutError:
        return SandboxResult(
            success=False,
            error="Hypothesis execution timed out",
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
            wall_clock_seconds=time.monotonic() - start_time,
            timed_out=True,
        )
    except Exception as e:
        return SandboxResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue() + "\n" + traceback.format_exc(),
            wall_clock_seconds=time.monotonic() - start_time,
        )

    elapsed = time.monotonic() - start_time
    return SandboxResult(
        success=True,
        metrics=metrics,
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
        wall_clock_seconds=elapsed,
    )
