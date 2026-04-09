"""Integration tests for the Carnot CLI via subprocess.

These tests invoke the ``carnot`` CLI as a subprocess, exactly as a user
would from their terminal. They verify exit codes, stdout output, and
error handling for the ``verify`` and ``score`` subcommands.

Spec: REQ-CODE-001, REQ-CODE-006, REQ-INFER-015
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Resolve paths relative to repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
MATH_FUNCS = os.path.join(EXAMPLES_DIR, "math_funcs.py")

# Use the venv python to ensure carnot entrypoint is available.
PYTHON = sys.executable


def run_carnot(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run ``python -m carnot.cli`` with given args as a subprocess."""
    cmd = [PYTHON, "-m", "carnot.cli", *args]
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Help / no-args
# ---------------------------------------------------------------------------


class TestCLIHelp:
    """Test CLI help output and no-args behavior.

    Spec: REQ-CODE-006
    """

    def test_no_args_shows_help(self) -> None:
        """REQ-CODE-006: Running carnot with no args shows help."""
        result = run_carnot()
        assert result.returncode == 1
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_help_flag(self) -> None:
        """REQ-CODE-006: --help shows usage info."""
        result = run_carnot("--help")
        assert result.returncode == 0
        assert "carnot" in result.stdout.lower()
        assert "verify" in result.stdout.lower()

    def test_verify_help(self) -> None:
        """REQ-CODE-006: verify --help shows subcommand usage."""
        result = run_carnot("verify", "--help")
        assert result.returncode == 0
        assert "--func" in result.stdout
        assert "--test" in result.stdout


# ---------------------------------------------------------------------------
# Verify subcommand — passing cases
# ---------------------------------------------------------------------------


class TestCLIVerifyPass:
    """Test CLI verify subcommand with correct functions.

    Spec: REQ-CODE-001, REQ-CODE-006
    """

    @pytest.mark.skipif(
        not os.path.isfile(MATH_FUNCS),
        reason="examples/math_funcs.py not found",
    )
    def test_gcd_correct(self) -> None:
        """REQ-CODE-001: GCD with correct test cases passes."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "gcd",
            "--test", "(12,8):4",
            "--test", "(15,10):5",
        )
        assert "CARNOT VERIFY" in result.stdout
        assert "PASS" in result.stdout
        assert result.returncode == 0

    @pytest.mark.skipif(
        not os.path.isfile(MATH_FUNCS),
        reason="examples/math_funcs.py not found",
    )
    def test_factorial_correct(self) -> None:
        """REQ-CODE-001: Factorial with correct test cases passes."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "factorial",
            "--test", "(5,):120",
            "--test", "(0,):1",
        )
        assert "PASS" in result.stdout
        assert result.returncode == 0

    @pytest.mark.skipif(
        not os.path.isfile(MATH_FUNCS),
        reason="examples/math_funcs.py not found",
    )
    def test_fibonacci_correct(self) -> None:
        """REQ-CODE-001: Fibonacci with correct test cases passes."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "fibonacci",
            "--test", "(10,):55",
        )
        assert "PASS" in result.stdout
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Verify subcommand — failing cases
# ---------------------------------------------------------------------------


class TestCLIVerifyFail:
    """Test CLI verify subcommand with incorrect functions/test cases.

    Spec: REQ-CODE-001, REQ-CODE-006
    """

    @pytest.mark.skipif(
        not os.path.isfile(MATH_FUNCS),
        reason="examples/math_funcs.py not found",
    )
    def test_buggy_add_fails(self) -> None:
        """REQ-CODE-001: Deliberately buggy function fails verification."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "buggy_add",
            "--test", "(2,3):5",
        )
        assert "FAIL" in result.stdout
        assert result.returncode == 1

    @pytest.mark.skipif(
        not os.path.isfile(MATH_FUNCS),
        reason="examples/math_funcs.py not found",
    )
    def test_gcd_wrong_expected(self) -> None:
        """REQ-CODE-001: Correct function with wrong expected value fails."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "gcd",
            "--test", "(12,8):3",
        )
        assert "FAIL" in result.stdout
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Verify subcommand — error handling
# ---------------------------------------------------------------------------


class TestCLIVerifyErrors:
    """Test CLI error handling for malformed inputs.

    Spec: REQ-CODE-006
    """

    def test_missing_file(self) -> None:
        """REQ-CODE-006: Non-existent file produces error."""
        result = run_carnot(
            "verify", "/nonexistent/file.py",
            "--func", "foo",
            "--test", "(1,):1",
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_no_test_cases(self) -> None:
        """REQ-CODE-006: No --test args produces error."""
        result = run_carnot(
            "verify", MATH_FUNCS,
            "--func", "gcd",
        )
        assert result.returncode == 1
        assert "at least one --test" in result.stderr.lower()

    def test_inline_python_file(self, tmp_path) -> None:
        """REQ-CODE-006: Verify works with a temp file."""
        src = tmp_path / "add.py"
        src.write_text(textwrap.dedent("""\
            def add(a: int, b: int) -> int:
                return a + b
        """))
        result = run_carnot(
            "verify", str(src),
            "--func", "add",
            "--test", "(1,2):3",
        )
        assert "PASS" in result.stdout
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Score subcommand — list models
# ---------------------------------------------------------------------------


class TestCLIScore:
    """Test CLI score subcommand.

    Spec: REQ-INFER-015
    """

    def test_list_models(self) -> None:
        """REQ-INFER-015: --list-models shows available EBMs."""
        result = run_carnot("score", "--list-models")
        assert result.returncode == 0
        assert "Available" in result.stdout or "Model ID" in result.stdout
