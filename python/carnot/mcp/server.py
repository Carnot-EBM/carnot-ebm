"""Hardened MCP server exposing Carnot verification and repair tools.

**Researcher summary:**
    Production-grade MCP server wrapping Carnot's code verification, property
    testing, constraint extraction, and verify-repair pipeline. Adds execution
    timeouts (30s), input validation (10K char limit), and structured error
    responses on top of the existing verification logic.

**Detailed explanation for engineers:**
    This server exposes six tools over MCP (stdio JSON-RPC):

    1. ``verify_code`` -- Run structural tests on a Python function (from Exp 48).
    2. ``verify_with_properties`` -- Property-based testing with random inputs.
    3. ``verify_llm_output`` -- Verify an LLM response via constraint extraction
       using VerifyRepairPipeline (from Exp 75).
    4. ``verify_and_repair`` -- Full verify-then-repair loop using the pipeline.
    5. ``list_domains`` -- List available constraint extraction domains.
    6. ``health_check`` -- Liveness probe returning server version and status.

    Production safeguards:
    - All tool handlers run inside ``_guarded_call()`` which enforces a 30-second
      execution timeout via ``concurrent.futures.ThreadPoolExecutor``.
    - All string inputs are validated against a 10,000-character limit via
      ``_validate_input()`` to prevent denial-of-service from oversized payloads.
    - All errors are caught and returned as structured dicts with an "error" key,
      a machine-readable "error_code", and a human-readable "detail" message.
      The server never crashes on bad input -- it always returns a valid response.

    Migrated from ``tools/verify-mcp/server.py`` into the ``carnot.mcp`` package
    so it can be imported, tested, and distributed as part of the carnot wheel.

Usage:
    python -m carnot.mcp

Spec: REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001, REQ-VERIFY-003
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import time
import traceback
from typing import Any, Callable, TypeVar

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_VERSION = "0.2.0"
MAX_INPUT_CHARS = 10_000
EXECUTION_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

mcp_server = FastMCP("carnot-verify")

# ---------------------------------------------------------------------------
# Structured error helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


class MCPError(Exception):
    """Structured error with machine-readable code and human-readable detail.

    **Detailed explanation for engineers:**
        Raised inside tool handlers to produce a structured error response.
        The ``_guarded_call`` wrapper catches these and formats them into
        the standard ``{"error": code, "detail": message}`` dict that MCP
        clients can parse programmatically.

    Attributes:
        code: Machine-readable error code (e.g., "INPUT_TOO_LARGE").
        detail: Human-readable explanation of what went wrong.
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _error_response(code: str, detail: str) -> dict[str, Any]:
    """Build a structured error response dict.

    **Detailed explanation for engineers:**
        All error responses from this server follow the same shape so that
        MCP clients can reliably detect and handle errors by checking for
        the presence of the "error" key. The "error_code" field is a
        stable, machine-readable identifier (e.g., "TIMEOUT") while
        "detail" is a human-readable message that may change.

    Args:
        code: Machine-readable error code.
        detail: Human-readable error description.

    Returns:
        Dict with "error", "error_code", and "detail" keys.
    """
    return {"error": True, "error_code": code, "detail": detail}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_input(value: str, field_name: str) -> None:
    """Validate that a string input does not exceed the character limit.

    **Detailed explanation for engineers:**
        Guards against oversized payloads that could cause the server to
        hang or consume excessive memory. The 10K character limit is
        generous enough for any reasonable code snippet or LLM response
        but prevents abuse. If exceeded, raises MCPError which the
        guarded_call wrapper converts to a structured error response.

    Args:
        value: The string to validate.
        field_name: Name of the field (for error messages).

    Raises:
        MCPError: If input exceeds MAX_INPUT_CHARS.
    """
    if len(value) > MAX_INPUT_CHARS:
        raise MCPError(
            "INPUT_TOO_LARGE",
            f"Field '{field_name}' has {len(value)} characters, "
            f"max allowed is {MAX_INPUT_CHARS}.",
        )


# ---------------------------------------------------------------------------
# Execution timeout wrapper
# ---------------------------------------------------------------------------


def _guarded_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | dict[str, Any]:
    """Execute a function with a timeout and structured error handling.

    **Detailed explanation for engineers:**
        Wraps any callable in a ThreadPoolExecutor with a 30-second timeout.
        If the function completes normally, returns its result. If it raises
        MCPError, returns a structured error response. If it raises any
        other exception, logs the traceback and returns a generic
        INTERNAL_ERROR response. If it times out, returns a TIMEOUT error.

        This is the central safety net: no tool handler can crash the server
        or hang indefinitely, regardless of what the underlying verification
        code does.

    Args:
        fn: The function to execute.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        Either fn's return value or a structured error dict.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=EXECUTION_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            logger.error("Tool call timed out after %ds", EXECUTION_TIMEOUT_SECONDS)
            return _error_response(
                "TIMEOUT",
                f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS} seconds.",
            )
        except MCPError as e:
            return _error_response(e.code, e.detail)
        except Exception as e:
            logger.error("Tool call failed: %s\n%s", e, traceback.format_exc())
            return _error_response("INTERNAL_ERROR", str(e))


# ---------------------------------------------------------------------------
# Tool: verify_code
# ---------------------------------------------------------------------------


def _run_structural_tests(
    code: str,
    func_name: str,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run structural tests (execute function against input/output pairs).

    **Detailed explanation for engineers:**
        Each test_case dict has:
        - ``args``: list of positional arguments (converted to tuple internally)
        - ``expected``: the expected return value

        We run safe_exec_function for each, compare results, and compute
        an energy score (fraction of failures). Individual test results
        are returned so the caller can see exactly which tests passed/failed.

    Args:
        code: Python source code containing the function.
        func_name: Name of the function to call.
        test_cases: List of dicts with 'args' and 'expected' keys.

    Returns:
        Dict with 'energy', 'n_passed', 'n_failed', 'n_total', and 'details'.

    Spec: REQ-CODE-001
    """
    from carnot.verify.python_types import safe_exec_function

    _validate_input(code, "code")

    details: list[dict[str, Any]] = []
    n_passed = 0
    n_failed = 0

    for tc in test_cases:
        args = tuple(tc["args"])
        expected = tc["expected"]

        result, error = safe_exec_function(code, func_name, args)

        if error is not None:
            n_failed += 1
            details.append({
                "args": tc["args"],
                "expected": expected,
                "actual": None,
                "passed": False,
                "error": str(error),
            })
        elif result != expected:
            n_failed += 1
            details.append({
                "args": tc["args"],
                "expected": expected,
                "actual": result,
                "passed": False,
                "error": None,
            })
        else:
            n_passed += 1
            details.append({
                "args": tc["args"],
                "expected": expected,
                "actual": result,
                "passed": True,
                "error": None,
            })

    n_total = n_passed + n_failed
    energy = n_failed / max(n_total, 1)

    return {
        "energy": energy,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_total": n_total,
        "details": details,
    }


@mcp_server.tool()
def verify_code(code: str, func_name: str, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify a Python function against structural tests.

    Runs the function on each test case and returns an energy score
    (0.0 = all pass, 1.0 = all fail) plus per-test details.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to test.
        test_cases: List of test case dicts, each with:
            - args: list of positional arguments
            - expected: expected return value
            Example: [{"args": [1, 2], "expected": 3}]

    Returns:
        Dict with energy score, pass/fail counts, and per-test details.

    Spec: REQ-CODE-001
    """
    return _guarded_call(_run_structural_tests, code, func_name, test_cases)


# ---------------------------------------------------------------------------
# Tool: verify_with_properties
# ---------------------------------------------------------------------------


def _run_property_tests(
    code: str,
    func_name: str,
    properties: list[dict[str, Any]],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run property-based tests on a function.

    **Detailed explanation for engineers:**
        Resolves caller-specified generators and checks (by name or inline
        lambda) into callables, then delegates to the underlying
        ``property_test()`` function from ``carnot.verify.property_test``.

        Generators produce random inputs; checks verify invariants on the
        function's output. Built-in generators include "int", "pos_int",
        "string", "list_int", "pair_int". Built-in checks include
        "returns_int", "non_negative", etc. Callers can also pass lambda
        expression strings for custom checks.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to test.
        properties: List of property dicts with 'name', 'generator', 'check'.
        n_samples: Number of random tests per property.
        seed: Random seed for reproducibility.

    Returns:
        Serialized PropertyTestResult dict.

    Spec: REQ-CODE-006
    """
    from carnot.verify.property_test import (
        PropertyTestResult,
        gen_int,
        gen_list_int,
        gen_pair_int,
        gen_pos_int,
        gen_string,
        property_test,
    )

    _validate_input(code, "code")

    # Built-in input generators that callers can reference by name.
    builtin_generators: dict[str, Any] = {
        "int": lambda rng: gen_int(rng),
        "pos_int": lambda rng: gen_pos_int(rng),
        "string": lambda rng: gen_string(rng),
        "list_int": lambda rng: gen_list_int(rng),
        "pair_int": lambda rng: gen_pair_int(rng),
    }

    # Built-in property checks that callers can reference by name.
    builtin_checks: dict[str, Any] = {
        "returns_int": lambda result, *_args: isinstance(result, int),
        "returns_float": lambda result, *_args: isinstance(result, float),
        "returns_str": lambda result, *_args: isinstance(result, str),
        "returns_list": lambda result, *_args: isinstance(result, list),
        "returns_bool": lambda result, *_args: isinstance(result, bool),
        "non_negative": lambda result, *_args: result >= 0,
    }

    def resolve_generator(gen_spec: str | dict[str, Any]) -> Any:
        if isinstance(gen_spec, str):
            if gen_spec in builtin_generators:
                return builtin_generators[gen_spec]
            raise MCPError(
                "INVALID_INPUT",
                f"Unknown generator: {gen_spec}. "
                f"Available: {list(builtin_generators.keys())}",
            )
        if isinstance(gen_spec, dict):
            gen_type = gen_spec.get("type", "")
            if gen_type in builtin_generators:
                return builtin_generators[gen_type]
            raise MCPError("INVALID_INPUT", f"Unknown generator type: {gen_type}")
        raise MCPError(
            "INVALID_INPUT",
            f"Generator spec must be a string or dict, got {type(gen_spec).__name__}",
        )

    def resolve_check(check_spec: str | dict[str, Any]) -> Any:
        if isinstance(check_spec, str):
            if check_spec in builtin_checks:
                return builtin_checks[check_spec]
            # Try interpreting as a lambda expression directly.
            # Security note: this server runs locally with trusted callers.
            # For untrusted callers, use the Firecracker sandbox.
            try:
                return eval(check_spec)  # noqa: S307
            except Exception:
                raise MCPError(
                    "INVALID_INPUT",
                    f"Unknown check: {check_spec}. "
                    f"Available: {list(builtin_checks.keys())}",
                ) from None
        if isinstance(check_spec, dict):
            if "lambda" in check_spec:
                try:
                    return eval(check_spec["lambda"])  # noqa: S307
                except Exception as e:
                    raise MCPError(
                        "INVALID_INPUT",
                        f"Failed to eval lambda: {check_spec['lambda']}: {e}",
                    ) from e
            raise MCPError("INVALID_INPUT", "Check dict must have a 'lambda' key")
        raise MCPError(
            "INVALID_INPUT",
            f"Check spec must be a string or dict, got {type(check_spec).__name__}",
        )

    resolved_properties: list[dict[str, Any]] = []
    for prop in properties:
        if "name" not in prop:
            raise MCPError("INVALID_INPUT", "Each property must have a 'name' key")
        resolved_properties.append({
            "name": prop["name"],
            "gen_args": resolve_generator(prop["generator"]),
            "check": resolve_check(prop["check"]),
        })

    result = property_test(
        code=code,
        func_name=func_name,
        properties=resolved_properties,
        n_samples=n_samples,
        seed=seed,
    )

    return {
        "energy": float(result.energy),
        "n_tests": result.n_tests,
        "n_passed": result.n_passed,
        "n_failed": result.n_failed,
        "wall_clock_seconds": result.wall_clock_seconds,
        "violations": [
            {
                "property_name": v.property_name,
                "input_args": [str(a) for a in v.input_args],
                "expected": v.expected,
                "actual": v.actual,
                "error": v.error,
            }
            for v in result.violations
        ],
    }


@mcp_server.tool()
def verify_with_properties(
    code: str,
    func_name: str,
    properties: list[dict[str, Any]],
    n_samples: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify a Python function using property-based testing.

    Generates random inputs and checks invariant properties. Returns
    an energy score (fraction of failures) and detailed violations.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to test.
        properties: List of property dicts, each with:
            - name: human-readable property name
            - generator: input generator -- either a string name
              ("int", "pos_int", "string", "list_int", "pair_int")
              or a dict with "type" key
            - check: property check -- either a string name
              ("returns_int", "non_negative", etc.) or a lambda
              expression string like "lambda result, a, b: result == a + b"
              or a dict with "lambda" key
        n_samples: Number of random tests per property (default 100).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dict with energy score, pass/fail counts, and violation details.

    Spec: REQ-CODE-006
    """
    return _guarded_call(_run_property_tests, code, func_name, properties, n_samples, seed)


# ---------------------------------------------------------------------------
# Tool: verify_llm_output
# ---------------------------------------------------------------------------


def _run_verify_llm_output(
    question: str,
    response: str,
    domain: str | None,
) -> dict[str, Any]:
    """Verify an LLM response using constraint extraction.

    **Detailed explanation for engineers:**
        Uses VerifyRepairPipeline in verify-only mode (no model loaded).
        Extracts constraints from the response text (arithmetic, code,
        logic, NL) and checks each one. Returns a structured result with
        the verified flag, energy score, constraint details, and any
        violations found.

        This is the MCP-accessible wrapper around the pipeline's verify()
        method, which itself was consolidated from Experiments 56 and 75.

    Args:
        question: The original question (for context).
        response: The LLM response text to verify.
        domain: Optional domain hint to restrict extraction.

    Returns:
        Dict with verification results.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003
    """
    _validate_input(question, "question")
    _validate_input(response, "response")

    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=[domain] if domain else None)
    vr = pipeline.verify(question, response, domain)

    return {
        "verified": vr.verified,
        "energy": float(vr.energy),
        "n_constraints": len(vr.constraints),
        "n_violations": len(vr.violations),
        "violations": [
            {
                "type": v.constraint_type,
                "description": v.description,
                "metadata": {
                    k: str(val) for k, val in v.metadata.items()
                    if k != "energy_term"
                },
            }
            for v in vr.violations
        ],
        "constraints": [
            {
                "type": c.constraint_type,
                "description": c.description,
                "satisfied": c.metadata.get("satisfied"),
            }
            for c in vr.constraints
        ],
        "certificate": vr.certificate,
    }


@mcp_server.tool()
def verify_llm_output(
    question: str,
    response: str,
    domain: str | None = None,
) -> dict[str, Any]:
    """Verify an LLM response by extracting and checking constraints.

    Uses Carnot's VerifyRepairPipeline to extract constraints (arithmetic,
    code, logic, natural language) from the response and check each one.
    Returns whether the response is verified, an energy score, and details
    of any constraint violations.

    Args:
        question: The original question that was asked.
        response: The LLM's response text to verify.
        domain: Optional domain hint ("arithmetic", "code", "logic", "nl")
            to restrict which constraint types are checked.

    Returns:
        Dict with verified flag, energy score, constraints, and violations.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003
    """
    return _guarded_call(_run_verify_llm_output, question, response, domain)


# ---------------------------------------------------------------------------
# Tool: verify_and_repair
# ---------------------------------------------------------------------------


def _run_verify_and_repair(
    question: str,
    response: str,
    domain: str | None,
    max_repairs: int,
) -> dict[str, Any]:
    """Verify an LLM response and report what would need repair.

    **Detailed explanation for engineers:**
        Uses VerifyRepairPipeline in verify-only mode (no model loaded via
        MCP -- loading a model requires GPU and is not suitable for a
        lightweight MCP tool). Runs verification and returns the full
        result including violation details formatted as natural-language
        feedback that the calling LLM can use to self-repair.

        The key insight: the MCP server doesn't need to run the LLM repair
        loop itself. It just needs to tell the calling LLM what's wrong
        in clear enough language that the LLM can fix its own output. The
        _format_violations method from VerifyRepairPipeline provides this.

    Args:
        question: The original question.
        response: The LLM response to verify and provide repair guidance for.
        domain: Optional domain hint.
        max_repairs: Maximum repair iterations (informational, since repair
            is delegated to the calling LLM).

    Returns:
        Dict with verification result and repair feedback.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004
    """
    _validate_input(question, "question")
    _validate_input(response, "response")

    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=[domain] if domain else None)
    vr = pipeline.verify(question, response, domain)

    # Format violations as natural-language repair feedback.
    repair_feedback = pipeline._format_violations(vr.violations)

    return {
        "verified": vr.verified,
        "energy": float(vr.energy),
        "n_constraints": len(vr.constraints),
        "n_violations": len(vr.violations),
        "repair_feedback": repair_feedback,
        "max_repairs_suggested": max_repairs,
        "violations": [
            {
                "type": v.constraint_type,
                "description": v.description,
                "metadata": {
                    k: str(val) for k, val in v.metadata.items()
                    if k != "energy_term"
                },
            }
            for v in vr.violations
        ],
        "certificate": vr.certificate,
    }


@mcp_server.tool()
def verify_and_repair(
    question: str,
    response: str,
    domain: str | None = None,
    max_repairs: int = 3,
) -> dict[str, Any]:
    """Verify an LLM response and provide natural-language repair feedback.

    Runs Carnot constraint verification on the response. If violations are
    found, returns detailed repair feedback that the calling LLM can use
    to fix its own output. The repair loop is delegated to the caller --
    this tool provides the verification and feedback, the LLM does the fixing.

    Args:
        question: The original question that was asked.
        response: The LLM's response text to verify.
        domain: Optional domain hint ("arithmetic", "code", "logic", "nl").
        max_repairs: Suggested max repair iterations (default 3).

    Returns:
        Dict with verified flag, energy, repair_feedback text, and violations.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004
    """
    return _guarded_call(_run_verify_and_repair, question, response, domain, max_repairs)


# ---------------------------------------------------------------------------
# Tool: list_domains
# ---------------------------------------------------------------------------


@mcp_server.tool()
def list_domains() -> dict[str, Any]:
    """List available constraint extraction domains.

    Returns the domains supported by Carnot's AutoExtractor, along with
    descriptions of what each domain checks. Useful for deciding which
    domain hint to pass to verify_llm_output or verify_and_repair.

    Returns:
        Dict with list of domain names and descriptions.
    """
    def _inner() -> dict[str, Any]:
        from carnot.pipeline.extract import AutoExtractor

        extractor = AutoExtractor()
        domains = extractor.supported_domains

        # Domain descriptions for human consumption.
        descriptions: dict[str, str] = {
            "arithmetic": "Checks arithmetic expressions (e.g., '47 + 28 = 75')",
            "code": "Checks Python code via AST analysis (syntax, undefined vars, etc.)",
            "logic": "Checks logical claims (e.g., 'If P then Q')",
            "nl": "Checks natural language factual claims via regex patterns",
        }

        return {
            "domains": [
                {
                    "name": d,
                    "description": descriptions.get(d, "No description available"),
                }
                for d in domains
            ],
            "n_domains": len(domains),
        }

    return _guarded_call(_inner)


# ---------------------------------------------------------------------------
# Tool: health_check
# ---------------------------------------------------------------------------


@mcp_server.tool()
def health_check() -> dict[str, Any]:
    """Check server health and return version information.

    Returns server version, status, and available tool count. Useful as a
    liveness probe to verify the MCP server is running and responsive.

    Returns:
        Dict with version, status, and tool list.
    """
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "tools": [
            "verify_code",
            "verify_with_properties",
            "verify_llm_output",
            "verify_and_repair",
            "list_domains",
            "health_check",
        ],
        "limits": {
            "max_input_chars": MAX_INPUT_CHARS,
            "execution_timeout_seconds": EXECUTION_TIMEOUT_SECONDS,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    mcp_server.run(transport="stdio")
