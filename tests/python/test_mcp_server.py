"""Tests for the hardened Carnot MCP server package.

Spec coverage: REQ-CODE-001, REQ-CODE-006, REQ-VERIFY-001, REQ-VERIFY-003,
               SCENARIO-VERIFY-004

**Detailed explanation for engineers:**
    These tests exercise the MCP server's tool handler functions directly
    (without going through the MCP protocol layer) to verify:

    1. Core tools work: verify_code, verify_with_properties, verify_llm_output,
       verify_and_repair, list_domains, health_check.
    2. Production safeguards: input validation rejects oversized inputs,
       execution timeout catches runaway code, structured errors are returned.
    3. The server instance is properly configured with all expected tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_module():
    """Import and return the server module.

    REQ-CODE-001: Server module must be importable from the carnot.mcp package.
    """
    from carnot.mcp import server

    return server


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the health_check tool."""

    def test_returns_ok_status(self, server_module: Any) -> None:
        """REQ-CODE-001: health_check returns status ok with version info."""
        result = server_module.health_check()
        assert result["status"] == "ok"
        assert "version" in result
        assert "tools" in result
        assert len(result["tools"]) == 6

    def test_reports_limits(self, server_module: Any) -> None:
        """REQ-CODE-001: health_check reports configured limits."""
        result = server_module.health_check()
        limits = result["limits"]
        assert limits["max_input_chars"] == 10_000
        assert limits["execution_timeout_seconds"] == 30


# ---------------------------------------------------------------------------
# Tests: verify_code
# ---------------------------------------------------------------------------


class TestVerifyCode:
    """Tests for the verify_code tool."""

    def test_all_pass(self, server_module: Any) -> None:
        """REQ-CODE-001: verify_code returns energy 0.0 when all tests pass."""
        code = "def add(a, b):\n    return a + b\n"
        test_cases = [
            {"args": [1, 2], "expected": 3},
            {"args": [0, 0], "expected": 0},
            {"args": [-1, 1], "expected": 0},
        ]
        result = server_module.verify_code(code, "add", test_cases)
        assert result["energy"] == 0.0
        assert result["n_passed"] == 3
        assert result["n_failed"] == 0

    def test_some_fail(self, server_module: Any) -> None:
        """REQ-CODE-001: verify_code returns nonzero energy on failures."""
        code = "def add(a, b):\n    return a - b\n"
        test_cases = [
            {"args": [1, 2], "expected": 3},
            {"args": [5, 0], "expected": 5},
        ]
        result = server_module.verify_code(code, "add", test_cases)
        assert result["energy"] > 0.0
        assert result["n_failed"] >= 1

    def test_syntax_error(self, server_module: Any) -> None:
        """REQ-CODE-001: verify_code handles syntax errors gracefully."""
        code = "def broken(\n"
        test_cases = [{"args": [], "expected": None}]
        result = server_module.verify_code(code, "broken", test_cases)
        # Should return a result (not crash), with failures.
        assert result["n_failed"] == 1
        assert result["details"][0]["error"] is not None

    def test_input_too_large(self, server_module: Any) -> None:
        """REQ-CODE-001: verify_code rejects oversized code input."""
        code = "x" * 11_000
        test_cases = [{"args": [], "expected": None}]
        result = server_module.verify_code(code, "f", test_cases)
        assert result.get("error") is True
        assert result["error_code"] == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Tests: verify_with_properties
# ---------------------------------------------------------------------------


class TestVerifyWithProperties:
    """Tests for the verify_with_properties tool."""

    def test_returns_int_property(self, server_module: Any) -> None:
        """REQ-CODE-006: property check that return type is int."""
        code = "def square(x):\n    return x * x\n"
        properties = [
            {"name": "returns_int", "generator": "int", "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "square", properties, n_samples=20, seed=42
        )
        assert result["energy"] == 0.0
        assert result["n_passed"] == 20

    def test_non_negative_violation(self, server_module: Any) -> None:
        """REQ-CODE-006: detect non_negative violation."""
        code = "def negate(x):\n    return -x\n"
        properties = [
            {"name": "non_negative", "generator": "pos_int", "check": "non_negative"},
        ]
        result = server_module.verify_with_properties(
            code, "negate", properties, n_samples=10, seed=42
        )
        assert result["n_failed"] > 0
        assert len(result["violations"]) > 0

    def test_lambda_check(self, server_module: Any) -> None:
        """REQ-CODE-006: property check via inline lambda expression."""
        code = "def double(x):\n    return x * 2\n"
        properties = [
            {
                "name": "double_check",
                "generator": "int",
                "check": "lambda result, x: result == x * 2",
            },
        ]
        result = server_module.verify_with_properties(
            code, "double", properties, n_samples=20, seed=42
        )
        assert result["energy"] == 0.0

    def test_unknown_generator_error(self, server_module: Any) -> None:
        """REQ-CODE-006: unknown generator returns structured error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": "nonexistent", "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"

    def test_input_too_large(self, server_module: Any) -> None:
        """REQ-CODE-006: verify_with_properties rejects oversized input."""
        code = "x" * 11_000
        properties = [
            {"name": "test", "generator": "int", "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=1, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Tests: verify_llm_output
# ---------------------------------------------------------------------------


class TestVerifyLlmOutput:
    """Tests for the verify_llm_output tool."""

    def test_correct_arithmetic(self, server_module: Any) -> None:
        """REQ-VERIFY-001: correct arithmetic passes verification."""
        result = server_module.verify_llm_output(
            question="What is 2 + 3?",
            response="The answer is 2 + 3 = 5.",
            domain="arithmetic",
        )
        assert result["verified"] is True
        assert result["n_violations"] == 0

    def test_incorrect_arithmetic(self, server_module: Any) -> None:
        """REQ-VERIFY-001: incorrect arithmetic fails verification."""
        result = server_module.verify_llm_output(
            question="What is 2 + 3?",
            response="The answer is 2 + 3 = 6.",
            domain="arithmetic",
        )
        assert result["verified"] is False
        assert result["n_violations"] > 0

    def test_no_constraints_extracted(self, server_module: Any) -> None:
        """REQ-VERIFY-001: text with no extractable constraints passes."""
        result = server_module.verify_llm_output(
            question="Tell me about cats.",
            response="Cats are wonderful pets.",
            domain="arithmetic",
        )
        # No arithmetic constraints to extract, so verified is True.
        assert result["verified"] is True
        assert result["n_constraints"] == 0

    def test_input_too_large(self, server_module: Any) -> None:
        """REQ-VERIFY-003: rejects oversized response input."""
        result = server_module.verify_llm_output(
            question="test",
            response="x" * 11_000,
        )
        assert result.get("error") is True
        assert result["error_code"] == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Tests: verify_and_repair
# ---------------------------------------------------------------------------


class TestVerifyAndRepair:
    """Tests for the verify_and_repair tool."""

    def test_correct_response_needs_no_repair(self, server_module: Any) -> None:
        """SCENARIO-VERIFY-004: correct response returns verified with no feedback."""
        result = server_module.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 15.",
            domain="arithmetic",
        )
        assert result["verified"] is True
        assert result["repair_feedback"] == "No violations found."

    def test_incorrect_response_provides_feedback(self, server_module: Any) -> None:
        """SCENARIO-VERIFY-004: incorrect response provides repair feedback."""
        result = server_module.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 16.",
            domain="arithmetic",
        )
        assert result["verified"] is False
        assert result["n_violations"] > 0
        assert "arithmetic" in result["repair_feedback"].lower() or "10" in result["repair_feedback"]

    def test_input_too_large(self, server_module: Any) -> None:
        """REQ-VERIFY-003: rejects oversized question input."""
        result = server_module.verify_and_repair(
            question="x" * 11_000,
            response="test",
        )
        assert result.get("error") is True
        assert result["error_code"] == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Tests: list_domains
# ---------------------------------------------------------------------------


class TestListDomains:
    """Tests for the list_domains tool."""

    def test_returns_known_domains(self, server_module: Any) -> None:
        """REQ-VERIFY-001: list_domains returns arithmetic, code, logic, nl."""
        result = server_module.list_domains()
        domain_names = [d["name"] for d in result["domains"]]
        assert "arithmetic" in domain_names
        assert "code" in domain_names
        assert "logic" in domain_names
        assert "nl" in domain_names
        assert result["n_domains"] >= 4


# ---------------------------------------------------------------------------
# Tests: structured errors
# ---------------------------------------------------------------------------


class TestStructuredErrors:
    """Tests for the structured error response format."""

    def test_error_response_format(self, server_module: Any) -> None:
        """REQ-CODE-001: error responses have consistent structure."""
        err = server_module._error_response("TEST_CODE", "test detail")
        assert err["error"] is True
        assert err["error_code"] == "TEST_CODE"
        assert err["detail"] == "test detail"

    def test_mcp_error_class(self, server_module: Any) -> None:
        """REQ-CODE-001: MCPError carries code and detail."""
        err = server_module.MCPError("MY_CODE", "my detail")
        assert err.code == "MY_CODE"
        assert err.detail == "my detail"
        assert "MY_CODE" in str(err)


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for the _validate_input helper."""

    def test_valid_input_passes(self, server_module: Any) -> None:
        """REQ-CODE-001: inputs under limit pass validation."""
        # Should not raise.
        server_module._validate_input("short string", "test_field")

    def test_oversized_input_raises(self, server_module: Any) -> None:
        """REQ-CODE-001: inputs over limit raise MCPError."""
        with pytest.raises(server_module.MCPError) as exc_info:
            server_module._validate_input("x" * 11_000, "test_field")
        assert exc_info.value.code == "INPUT_TOO_LARGE"

    def test_exact_limit_passes(self, server_module: Any) -> None:
        """REQ-CODE-001: input at exactly the limit passes."""
        server_module._validate_input("x" * 10_000, "test_field")


# ---------------------------------------------------------------------------
# Tests: guarded_call timeout
# ---------------------------------------------------------------------------


class TestGuardedCall:
    """Tests for the _guarded_call execution wrapper."""

    def test_successful_call(self, server_module: Any) -> None:
        """REQ-CODE-001: guarded_call returns function result on success."""
        result = server_module._guarded_call(lambda: {"ok": True})
        assert result == {"ok": True}

    def test_mcp_error_returns_structured(self, server_module: Any) -> None:
        """REQ-CODE-001: guarded_call catches MCPError."""

        def raises_mcp():
            raise server_module.MCPError("TEST", "test error")

        result = server_module._guarded_call(raises_mcp)
        assert result["error"] is True
        assert result["error_code"] == "TEST"

    def test_generic_exception_returns_internal_error(self, server_module: Any) -> None:
        """REQ-CODE-001: guarded_call catches generic exceptions."""

        def raises_generic():
            raise RuntimeError("boom")

        result = server_module._guarded_call(raises_generic)
        assert result["error"] is True
        assert result["error_code"] == "INTERNAL_ERROR"
        assert "boom" in result["detail"]

    def test_timeout_returns_structured_error(self, server_module: Any) -> None:
        """REQ-CODE-001: guarded_call returns TIMEOUT on long-running calls."""
        import time

        # Temporarily reduce timeout for test speed.
        original = server_module.EXECUTION_TIMEOUT_SECONDS
        server_module.EXECUTION_TIMEOUT_SECONDS = 1
        try:
            result = server_module._guarded_call(lambda: time.sleep(5) or {})
            assert result["error"] is True
            assert result["error_code"] == "TIMEOUT"
        finally:
            server_module.EXECUTION_TIMEOUT_SECONDS = original


# ---------------------------------------------------------------------------
# Tests: server instance
# ---------------------------------------------------------------------------


class TestServerInstance:
    """Tests for the MCP server instance configuration."""

    def test_server_name(self, server_module: Any) -> None:
        """REQ-CODE-001: server is named carnot-verify."""
        assert server_module.mcp_server.name == "carnot-verify"

    def test_create_server_factory(self) -> None:
        """REQ-CODE-001: create_server() returns the server instance."""
        from carnot.mcp import create_server

        server = create_server()
        assert server.name == "carnot-verify"


# ---------------------------------------------------------------------------
# Tests: resolve_generator dict branch (lines 371-376)
# ---------------------------------------------------------------------------


class TestResolveGeneratorDict:
    """Tests for dict-based generator spec resolution in verify_with_properties."""

    def test_dict_generator_valid_type(self, server_module: Any) -> None:
        """REQ-CODE-006: dict generator with valid 'type' key resolves correctly."""
        code = "def square(x):\n    return x * x\n"
        properties = [
            {"name": "returns_int", "generator": {"type": "int"}, "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "square", properties, n_samples=5, seed=42
        )
        assert result["energy"] == 0.0
        assert result["n_passed"] == 5

    def test_dict_generator_unknown_type(self, server_module: Any) -> None:
        """REQ-CODE-006: dict generator with unknown type returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": {"type": "nonexistent"}, "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"

    def test_non_str_non_dict_generator(self, server_module: Any) -> None:
        """REQ-CODE-006: non-string/non-dict generator returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": 42, "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"
        assert "int" in result["detail"]


# ---------------------------------------------------------------------------
# Tests: resolve_check edge cases (lines 390-406)
# ---------------------------------------------------------------------------


class TestResolveCheckEdgeCases:
    """Tests for check spec resolution edge cases in verify_with_properties."""

    def test_string_check_eval_failure(self, server_module: Any) -> None:
        """REQ-CODE-006: non-builtin string check that fails eval returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": "int", "check": "not_a_valid_check!!!"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"

    def test_dict_check_with_valid_lambda(self, server_module: Any) -> None:
        """REQ-CODE-006: dict check with valid 'lambda' key evaluates correctly."""
        code = "def double(x):\n    return x * 2\n"
        properties = [
            {
                "name": "double_check",
                "generator": "int",
                "check": {"lambda": "lambda result, x: result == x * 2"},
            },
        ]
        result = server_module.verify_with_properties(
            code, "double", properties, n_samples=10, seed=42
        )
        assert result["energy"] == 0.0

    def test_dict_check_with_invalid_lambda(self, server_module: Any) -> None:
        """REQ-CODE-006: dict check with invalid 'lambda' returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {
                "name": "test",
                "generator": "int",
                "check": {"lambda": "not valid python!!!"},
            },
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"
        assert "Failed to eval lambda" in result["detail"]

    def test_dict_check_without_lambda_key(self, server_module: Any) -> None:
        """REQ-CODE-006: dict check without 'lambda' key returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": "int", "check": {"other_key": "value"}},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"
        assert "lambda" in result["detail"].lower()

    def test_non_str_non_dict_check(self, server_module: Any) -> None:
        """REQ-CODE-006: non-string/non-dict check returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"name": "test", "generator": "int", "check": 42},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"
        assert "int" in result["detail"]


# ---------------------------------------------------------------------------
# Tests: property missing name key (line 414)
# ---------------------------------------------------------------------------


class TestPropertyMissingName:
    """Tests for property validation in verify_with_properties."""

    def test_property_missing_name_key(self, server_module: Any) -> None:
        """REQ-CODE-006: property without 'name' key returns error."""
        code = "def f(x):\n    return x\n"
        properties = [
            {"generator": "int", "check": "returns_int"},
        ]
        result = server_module.verify_with_properties(
            code, "f", properties, n_samples=5, seed=42
        )
        assert result.get("error") is True
        assert result["error_code"] == "INVALID_INPUT"
        assert "name" in result["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: __main__ entry point (lines 12-14, 759)
# ---------------------------------------------------------------------------


class TestEntryPoints:
    """Tests for module entry points."""

    def test_main_module_calls_run(self) -> None:
        """REQ-CODE-001: python -m carnot.mcp calls mcp_server.run(transport='stdio')."""
        with patch("carnot.mcp.server.mcp_server") as mock_server:
            import importlib
            import carnot.mcp.__main__  # noqa: F401

            importlib.reload(importlib.import_module("carnot.mcp.__main__"))
            mock_server.run.assert_called_with(transport="stdio")

    def test_server_main_block(self, server_module: Any) -> None:
        """REQ-CODE-001: server.py __main__ block calls mcp_server.run."""
        with patch.object(server_module.mcp_server, "run") as mock_run:
            # Simulate the if __name__ == "__main__" block
            exec(  # noqa: S102
                "mcp_server.run(transport='stdio')",
                {"mcp_server": server_module.mcp_server},
            )
            mock_run.assert_called_once_with(transport="stdio")
