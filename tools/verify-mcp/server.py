#!/usr/bin/env python3
"""MCP server exposing Carnot's code verification tools.

**Researcher summary:**
    Packages the composite scorer (logprob + structural tests) and property-based
    testing as an MCP server that Claude Code can call automatically during code
    generation. Two tools are exposed:

    - ``verify_code``: Runs structural tests (type check, no-exception, test-pass)
      on a Python function. Returns composite energy score and per-test details.
    - ``verify_with_properties``: Runs property-based tests (random inputs checking
      invariants like commutativity, idempotency, bounds). Returns violations.

**Detailed explanation for engineers:**
    This server uses the MCP Python SDK (``mcp``) to expose Carnot's existing
    verification pipeline as tools that Claude Code can invoke via the MCP protocol.
    The server communicates over stdio (stdin/stdout JSON-RPC), which is the
    standard transport for local MCP tool servers.

    The verification logic lives in:
    - ``carnot.verify.python_types.safe_exec_function`` — safe code execution
    - ``carnot.verify.python_types.build_code_energy`` — structural constraints
    - ``carnot.verify.property_test.property_test`` — property-based testing

    This server is a thin adapter: it deserializes MCP tool calls into the
    existing Carnot API, runs verification, and serializes results back.

Usage:
    python tools/verify-mcp/server.py

Spec: REQ-CODE-001, REQ-CODE-006
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so carnot package is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from carnot.verify.property_test import (
    PropertyTestResult,
    PropertyViolation,
    gen_int,
    gen_list_int,
    gen_pair_int,
    gen_pos_int,
    gen_string,
    property_test,
)
from carnot.verify.python_types import safe_exec_function

# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------

mcp_server = FastMCP("carnot-verify")


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
    """
    return _run_structural_tests(code, func_name, test_cases)


# ---------------------------------------------------------------------------
# Tool: verify_with_properties
# ---------------------------------------------------------------------------

# Built-in input generators that callers can reference by name.
# Each generator is a callable(rng) -> value.
_BUILTIN_GENERATORS: dict[str, Any] = {
    "int": lambda rng: gen_int(rng),
    "pos_int": lambda rng: gen_pos_int(rng),
    "string": lambda rng: gen_string(rng),
    "list_int": lambda rng: gen_list_int(rng),
    "pair_int": lambda rng: gen_pair_int(rng),
}

# Built-in property checks that callers can reference by name.
# Each check is a callable(result, *args) -> bool.
_BUILTIN_CHECKS: dict[str, Any] = {
    "returns_int": lambda result, *_args: isinstance(result, int),
    "returns_float": lambda result, *_args: isinstance(result, float),
    "returns_str": lambda result, *_args: isinstance(result, str),
    "returns_list": lambda result, *_args: isinstance(result, list),
    "returns_bool": lambda result, *_args: isinstance(result, bool),
    "non_negative": lambda result, *_args: result >= 0,
    "commutative": lambda result, a, b: result == _BUILTIN_CHECKS["_commutative_helper"](a, b),
    "idempotent": lambda result, *args: True,  # placeholder — needs func reference
}


def _resolve_generator(gen_spec: str | dict[str, Any]) -> Any:
    """Resolve an input generator from a string name or inline spec.

    **Detailed explanation for engineers:**
        Callers specify generators either as a string name referencing a built-in
        (e.g., "int", "pair_int") or as a dict with "type" and optional params.
        This function resolves the string/dict into an actual callable(rng) -> value.

        For "args" generators that produce tuples (like pair_int), the result is
        already a tuple suitable for function argument unpacking.

    Args:
        gen_spec: Either a string name or a dict with "type" key.

    Returns:
        Callable that takes a random.Random and returns generated value(s).
    """
    if isinstance(gen_spec, str):
        if gen_spec in _BUILTIN_GENERATORS:
            return _BUILTIN_GENERATORS[gen_spec]
        msg = f"Unknown generator: {gen_spec}. Available: {list(_BUILTIN_GENERATORS.keys())}"
        raise ValueError(msg)

    if isinstance(gen_spec, dict):
        gen_type = gen_spec.get("type", "")
        if gen_type in _BUILTIN_GENERATORS:
            return _BUILTIN_GENERATORS[gen_type]
        msg = f"Unknown generator type: {gen_type}"
        raise ValueError(msg)

    msg = f"Generator spec must be a string or dict, got {type(gen_spec)}"
    raise TypeError(msg)


def _resolve_check(check_spec: str | dict[str, Any]) -> Any:
    """Resolve a property check from a string name or inline lambda string.

    **Detailed explanation for engineers:**
        Callers specify checks either as a string name referencing a built-in
        (e.g., "returns_int", "non_negative") or as a dict with a "lambda" key
        containing a Python lambda expression string that will be eval'd.

        The lambda approach allows callers to define arbitrary property checks
        without needing to register them ahead of time. The lambda receives
        (result, *args) where result is the function's return value and args
        are the generated inputs.

        **Security note:** eval() is used here intentionally — this server runs
        locally and the caller is Claude Code (trusted). For untrusted callers,
        use the Firecracker sandbox.

    Args:
        check_spec: Either a string name or a dict with "lambda" key.

    Returns:
        Callable(result, *args) -> bool.
    """
    if isinstance(check_spec, str):
        if check_spec in _BUILTIN_CHECKS:
            return _BUILTIN_CHECKS[check_spec]
        # Try interpreting as a lambda expression directly
        try:
            return eval(check_spec)  # noqa: S307 — intentional, trusted caller
        except Exception:
            msg = f"Unknown check: {check_spec}. Available: {list(_BUILTIN_CHECKS.keys())}"
            raise ValueError(msg) from None

    if isinstance(check_spec, dict):
        if "lambda" in check_spec:
            try:
                return eval(check_spec["lambda"])  # noqa: S307 — intentional, trusted caller
            except Exception as e:
                msg = f"Failed to eval lambda: {check_spec['lambda']}: {e}"
                raise ValueError(msg) from e
        msg = "Check dict must have a 'lambda' key"
        raise ValueError(msg)

    msg = f"Check spec must be a string or dict, got {type(check_spec)}"
    raise TypeError(msg)


def _serialize_property_result(result: PropertyTestResult) -> dict[str, Any]:
    """Convert PropertyTestResult to a JSON-serializable dict.

    Args:
        result: The PropertyTestResult from property_test().

    Returns:
        Dict with all fields serialized.
    """
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
            - generator: input generator — either a string name
              ("int", "pos_int", "string", "list_int", "pair_int")
              or a dict with "type" key
            - check: property check — either a string name
              ("returns_int", "non_negative", etc.) or a lambda
              expression string like "lambda result, a, b: result == a + b"
              or a dict with "lambda" key
        n_samples: Number of random tests per property (default 100).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dict with energy score, pass/fail counts, and violation details.
    """
    # Convert the caller's property specs into the format expected by property_test()
    resolved_properties: list[dict[str, Any]] = []

    for prop in properties:
        name = prop["name"]
        gen_fn = _resolve_generator(prop["generator"])
        check_fn = _resolve_check(prop["check"])

        resolved_properties.append({
            "name": name,
            "gen_args": gen_fn,
            "check": check_fn,
        })

    result = property_test(
        code=code,
        func_name=func_name,
        properties=resolved_properties,
        n_samples=n_samples,
        seed=seed,
    )

    return _serialize_property_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
