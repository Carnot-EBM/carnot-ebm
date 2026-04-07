"""Tests for the Carnot CLI.

**Researcher summary:**
    Verifies the CLI command-line interface: test parsing, type resolution,
    and end-to-end verification of correct and buggy functions.

**Detailed explanation for engineers:**
    Tests the CLI functions directly (not via subprocess) to keep test speed
    and coverage simple. Uses the examples/math_funcs.py fixture.

Spec coverage: REQ-CODE-001, REQ-CODE-006, REQ-INFER-015
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from carnot.cli import (
    _find_separator_colon,
    _parse_test_pair,
    _resolve_type,
    cmd_score,
    cmd_verify,
    main,
)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
MATH_FILE = str(EXAMPLES_DIR / "math_funcs.py")


# --- Test parsing ---


def test_parse_test_pair_basic() -> None:
    """Parse simple int test case.

    Spec: REQ-CODE-006
    """
    args, expected = _parse_test_pair("(12,8):4")
    assert args == (12, 8)
    assert expected == 4


def test_parse_test_pair_single_arg() -> None:
    """Single arg wraps in tuple.

    Spec: REQ-CODE-006
    """
    args, expected = _parse_test_pair("5:25")
    assert args == (5,)
    assert expected == 25


def test_parse_test_pair_list() -> None:
    """Parse list test case.

    Spec: REQ-CODE-006
    """
    args, expected = _parse_test_pair("([3,1,2],):[1,2,3]")
    assert args == ([3, 1, 2],)
    assert expected == [1, 2, 3]


def test_parse_test_pair_invalid() -> None:
    """Invalid format raises ValueError.

    Spec: REQ-CODE-006
    """
    with pytest.raises(ValueError, match="Invalid test format"):
        _parse_test_pair("no_colon_here")


def test_find_separator_colon() -> None:
    """Finds rightmost valid colon.

    Spec: REQ-CODE-006
    """
    assert _find_separator_colon("(12,8):4") > 0
    assert _find_separator_colon("nocolon") == -1


# --- Type resolution ---


def test_resolve_type_basic() -> None:
    """Resolves standard type names.

    Spec: REQ-CODE-006
    """
    assert _resolve_type("int") is int
    assert _resolve_type("str") is str
    assert _resolve_type("list") is list


def test_resolve_type_unknown() -> None:
    """Unknown type raises ValueError.

    Spec: REQ-CODE-006
    """
    with pytest.raises(ValueError, match="Unknown type"):
        _resolve_type("frobnicate")


# --- End-to-end verify ---


def test_verify_correct_function(capsys: pytest.CaptureFixture[str]) -> None:
    """Correct function passes verification.

    Spec: REQ-CODE-001, REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="gcd",
        test=["(12,8):4", "(7,13):1"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    exit_code = cmd_verify(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out


def test_verify_buggy_function(capsys: pytest.CaptureFixture[str]) -> None:
    """Buggy function fails verification.

    Spec: REQ-CODE-001, REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="buggy_add",
        test=["(2,3):5"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    exit_code = cmd_verify(args)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out


def test_verify_missing_file() -> None:
    """Missing file returns exit code 1.

    Spec: REQ-CODE-006
    """
    args = argparse.Namespace(
        file="/nonexistent/file.py",
        func="foo",
        test=["1:1"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    assert cmd_verify(args) == 1


def test_verify_no_tests() -> None:
    """No test cases returns exit code 1.

    Spec: REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="gcd",
        test=[],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    assert cmd_verify(args) == 1


def test_verify_bad_test_format() -> None:
    """Bad test format returns exit code 1.

    Spec: REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="gcd",
        test=["not_valid"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    assert cmd_verify(args) == 1


def test_verify_with_properties(capsys: pytest.CaptureFixture[str]) -> None:
    """Property-based testing runs and detects violations.

    Spec: REQ-CODE-001, REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="factorial",
        test=["5:120"],
        type="int",
        properties=True,
        prop_samples=10,
        prop_seed=42,
    )
    cmd_verify(args)
    captured = capsys.readouterr()
    assert "Property-Based Tests" in captured.out


def test_verify_error_detail(capsys: pytest.CaptureFixture[str]) -> None:
    """Failed tests show error detail.

    Spec: REQ-CODE-006
    """
    args = argparse.Namespace(
        file=MATH_FILE,
        func="buggy_add",
        test=["(2,3):5", "(0,0):0"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    cmd_verify(args)
    captured = capsys.readouterr()
    assert "got" in captured.out


def test_find_separator_colon_empty_sides() -> None:
    """Colon at start or end is not a valid separator.

    Spec: REQ-CODE-006
    """
    assert _find_separator_colon(":5") == -1
    assert _find_separator_colon("5:") == -1


def test_parse_test_pair_bad_input() -> None:
    """Unparseable input side raises ValueError.

    Spec: REQ-CODE-006
    """
    # Neither side parses as literal → "Invalid test format"
    with pytest.raises(ValueError, match="Invalid test format"):
        _parse_test_pair("not_python_literal:5")


def test_parse_test_pair_bad_expected() -> None:
    """Unparseable expected side raises ValueError.

    Spec: REQ-CODE-006
    """
    with pytest.raises(ValueError, match="Invalid test format"):
        _parse_test_pair("5:not_python_literal")


def test_verify_exception_detail(
    capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """Execution errors show error detail.

    Spec: REQ-CODE-006
    """
    # Write a function that raises
    code_file = tmp_path / "bad.py"
    code_file.write_text("def boom(x):\n    raise RuntimeError('kaboom')\n")
    args = argparse.Namespace(
        file=str(code_file),
        func="boom",
        test=["5:5"],
        type="int",
        properties=False,
        prop_samples=100,
        prop_seed=42,
    )
    exit_code = cmd_verify(args)
    captured = capsys.readouterr()
    assert "error:" in captured.out
    assert exit_code == 1


def test_main_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """No arguments prints help and returns 1.

    Spec: REQ-CODE-006
    """
    monkeypatch.setattr("sys.argv", ["carnot"])
    assert main() == 1


def test_main_verify_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full CLI invocation via main().

    Spec: REQ-CODE-001, REQ-CODE-006
    """
    monkeypatch.setattr("sys.argv", [
        "carnot", "verify", MATH_FILE,
        "--func", "factorial",
        "--test", "5:120",
        "--test", "0:1",
    ])
    assert main() == 0


# --- Score subcommand tests ---


def test_score_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    """List available EBM models.

    Spec: REQ-INFER-015
    """
    args = argparse.Namespace(
        list_models=True,
        model="per-token-ebm-qwen35-08b-nothink",
        activations_file=None,
    )
    assert cmd_score(args) == 0
    captured = capsys.readouterr()
    assert "qwen3-06b" in captured.out
    assert "qwen35-08b-nothink" in captured.out


def test_score_no_file() -> None:
    """Score without activations file returns 1.

    Spec: REQ-INFER-015
    """
    args = argparse.Namespace(
        list_models=False,
        model="per-token-ebm-qwen35-08b-nothink",
        activations_file=None,
    )
    assert cmd_score(args) == 1


def test_score_with_real_exports(capsys: pytest.CaptureFixture[str]) -> None:
    """Score activations using the real exported models (if available).

    Spec: REQ-INFER-015
    """

    exports_dir = Path(__file__).parent.parent.parent / "exports" / "per-token-ebm-qwen3-06b"
    acts_file = Path(__file__).parent.parent.parent / "data" / "token_activations_large.safetensors"

    if not exports_dir.exists() or not acts_file.exists():
        pytest.skip("No exports or activation data available")

    args = argparse.Namespace(
        list_models=False,
        model="per-token-ebm-qwen3-06b",
        activations_file=str(acts_file),
    )
    exit_code = cmd_score(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Mean energy" in captured.out
    assert "Detection" in captured.out


def test_main_score_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI score --list-models via main().

    Spec: REQ-INFER-015
    """
    monkeypatch.setattr("sys.argv", ["carnot", "score", "--list-models"])
    assert main() == 0
