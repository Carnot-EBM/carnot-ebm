"""Tests for the autoresearch sandbox execution environment.

Spec coverage: REQ-AUTO-004, REQ-AUTO-009,
               SCENARIO-AUTO-002, SCENARIO-AUTO-004
"""

from carnot.autoresearch.sandbox import (
    SandboxConfig,
    SandboxResult,
    check_imports,
    run_in_sandbox,
)


class TestImportChecker:
    """Tests for REQ-AUTO-009: blocked import detection."""

    def test_clean_code_passes(self) -> None:
        """REQ-AUTO-009: code with no blocked imports passes."""
        code = "import jax\nimport numpy\ndef run(data): return {}"
        violations = check_imports(code, SandboxConfig().blocked_modules)
        assert violations == []

    def test_blocked_import_detected(self) -> None:
        """SCENARIO-AUTO-004: import os is blocked."""
        code = "import os\ndef run(data): return {}"
        violations = check_imports(code, SandboxConfig().blocked_modules)
        assert "os" in violations

    def test_blocked_from_import_detected(self) -> None:
        """SCENARIO-AUTO-004: from subprocess import call is blocked."""
        code = "from subprocess import call\ndef run(data): return {}"
        violations = check_imports(code, SandboxConfig().blocked_modules)
        assert "subprocess" in violations

    def test_blocked_nested_import(self) -> None:
        """SCENARIO-AUTO-004: import os.path is blocked."""
        code = "import os.path\ndef run(data): return {}"
        violations = check_imports(code, SandboxConfig().blocked_modules)
        assert "os" in violations

    def test_syntax_error_flagged(self) -> None:
        """REQ-AUTO-009: syntax errors are flagged."""
        code = "def run(data) return {}"  # missing colon
        violations = check_imports(code, SandboxConfig().blocked_modules)
        assert len(violations) == 1
        assert "syntax error" in violations[0]


class TestSandbox:
    """Tests for REQ-AUTO-004: sandbox execution."""

    def test_successful_execution(self) -> None:
        """REQ-AUTO-004: hypothesis runs and returns metrics."""
        code = """
def run(benchmark_data):
    return {"final_energy": -5.0, "convergence_steps": 100}
"""
        result = run_in_sandbox(code, {"dim": 2})
        assert result.success
        assert result.metrics["final_energy"] == -5.0
        assert result.wall_clock_seconds > 0

    def test_blocked_import_rejected(self) -> None:
        """SCENARIO-AUTO-004: hypothesis with os import is rejected."""
        code = """
import os
def run(benchmark_data):
    return {"result": os.getcwd()}
"""
        result = run_in_sandbox(code, {})
        assert not result.success
        assert "Blocked imports" in (result.error or "")

    def test_missing_run_function(self) -> None:
        """REQ-AUTO-004: hypothesis must define run()."""
        code = "x = 42"
        result = run_in_sandbox(code, {})
        assert not result.success
        assert "run" in (result.error or "")

    def test_run_must_return_dict(self) -> None:
        """REQ-AUTO-004: run() must return a dict."""
        code = "def run(data): return 42"
        result = run_in_sandbox(code, {})
        assert not result.success
        assert "dict" in (result.error or "")

    def test_runtime_error_captured(self) -> None:
        """SCENARIO-AUTO-002: runtime errors are captured, not propagated."""
        code = """
def run(benchmark_data):
    raise ValueError("divergent energy")
"""
        result = run_in_sandbox(code, {})
        assert not result.success
        assert "ValueError" in (result.error or "")
        assert "divergent energy" in (result.error or "")

    def test_stdout_captured(self) -> None:
        """REQ-AUTO-004: stdout from hypothesis is captured."""
        code = """
def run(benchmark_data):
    print("hello from hypothesis")
    return {"x": 1}
"""
        result = run_in_sandbox(code, {})
        assert result.success
        assert "hello from hypothesis" in result.stdout

    def test_benchmark_data_passed(self) -> None:
        """REQ-AUTO-004: benchmark data is accessible to hypothesis."""
        code = """
def run(benchmark_data):
    return {"dim": benchmark_data["dim"], "final_energy": -1.0}
"""
        result = run_in_sandbox(code, {"dim": 42})
        assert result.success
        assert result.metrics["dim"] == 42

    def test_timeout_short(self) -> None:
        """REQ-AUTO-004: timeout kills long-running hypothesis."""
        code = """
import time
def run(benchmark_data):
    time.sleep(10)
    return {}
"""
        config = SandboxConfig(timeout_seconds=1)
        result = run_in_sandbox(code, {}, config=config)
        assert not result.success
        assert result.timed_out
