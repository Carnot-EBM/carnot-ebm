"""Tests for Exp 820: GGUF Import Fix + Live Code Repair v5.

Covers:
    - diagnose_llama_cpp_import returns (True, "") when llama_cpp is importable (REQ-REPAIR-056)
    - diagnose_llama_cpp_import returns (False, msg) on ImportError (REQ-REPAIR-056)
    - attempt_llama_cpp_repair calls pip and returns (True, stdout) on success (SCENARIO-REPAIR-089)
    - attempt_llama_cpp_repair returns (False, stderr) on pip failure (SCENARIO-REPAIR-089)
    - run_problem_baseline returns True for a correct canonical solution (REQ-REPAIR-056)
    - run_problem_baseline returns False for a broken canonical solution (REQ-REPAIR-056)
    - run_problem_with_llm returns True when mock LLM produces correct code (REQ-REPAIR-056)
    - run_problem_with_llm returns False when mock LLM produces incorrect code (REQ-REPAIR-056)
    - build_blocked_artifact emits required schema fields (REQ-REPAIR-056)
    - artifact has import_repair_attempted field when import fails (SCENARIO-REPAIR-089)

Spec: REQ-REPAIR-056, SCENARIO-REPAIR-089
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.experiment_820_gguf_import_fix_code_repair_v5 import (
    attempt_llama_cpp_repair,
    build_blocked_artifact,
    diagnose_llama_cpp_import,
    run_problem_baseline,
    run_problem_with_llm,
)


# ---------------------------------------------------------------------------
# diagnose_llama_cpp_import — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_diagnose_import_succeeds_when_llama_cpp_present() -> None:
    """Returns (True, '') when llama_cpp.Llama is importable.

    Spec: REQ-REPAIR-056
    """
    fake_llama_cpp = MagicMock()
    fake_llama_cpp.Llama = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": fake_llama_cpp}):
        ok, msg = diagnose_llama_cpp_import()
    assert ok is True
    assert msg == ""


def test_diagnose_import_fails_when_llama_cpp_missing() -> None:
    """Returns (False, error_message) when llama_cpp raises ImportError.

    Spec: REQ-REPAIR-056
    """
    with patch.dict(sys.modules, {"llama_cpp": None}):
        ok, msg = diagnose_llama_cpp_import()
    assert ok is False
    assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# attempt_llama_cpp_repair — SCENARIO-REPAIR-089
# ---------------------------------------------------------------------------


def test_repair_returns_true_on_pip_success() -> None:
    """Returns (True, stdout) when pip install exits 0.

    Spec: SCENARIO-REPAIR-089
    """
    fake_result = SimpleNamespace(returncode=0, stdout="Successfully installed", stderr="")
    with patch(
        "scripts.experiment_820_gguf_import_fix_code_repair_v5.subprocess.run",
        return_value=fake_result,
    ) as mock_run:
        ok, output = attempt_llama_cpp_repair()
    assert ok is True
    assert "Successfully installed" in output
    # Verify pip was called with the expected arguments.
    args = mock_run.call_args[0][0]
    assert "pip" in " ".join(args)
    assert "llama-cpp-python" in args


def test_repair_returns_false_on_pip_failure() -> None:
    """Returns (False, stderr) when pip install exits non-zero.

    Spec: SCENARIO-REPAIR-089
    """
    fake_result = SimpleNamespace(returncode=1, stdout="", stderr="ERROR: no matching distribution")
    with patch(
        "scripts.experiment_820_gguf_import_fix_code_repair_v5.subprocess.run",
        return_value=fake_result,
    ):
        ok, output = attempt_llama_cpp_repair()
    assert ok is False
    assert "ERROR" in output


# ---------------------------------------------------------------------------
# run_problem_baseline — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_baseline_returns_true_for_correct_canonical_solution() -> None:
    """Canonical solution that satisfies its own tests returns True.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "canonical_solution": "    return a + b\n",
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is True


def test_baseline_returns_false_for_broken_canonical_solution() -> None:
    """Broken canonical solution that fails its tests returns False.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "canonical_solution": "    return a - b\n",  # deliberately wrong
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is False


# ---------------------------------------------------------------------------
# run_problem_with_llm — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_llm_returns_true_when_generated_code_passes() -> None:
    """Mock LLM that returns correct code yields True.

    The mock Llama call returns the correct function body.  The test asserts
    that run_problem_with_llm correctly exec-s and evaluates the test case.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "    return a + b\n"}]}
    assert run_problem_with_llm(problem, mock_llm) is True


def test_llm_returns_false_when_generated_code_fails() -> None:
    """Mock LLM that returns wrong code yields False.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [{"text": "    return a - b\n"}]  # deliberately wrong
    }
    assert run_problem_with_llm(problem, mock_llm) is False


# ---------------------------------------------------------------------------
# build_blocked_artifact — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_blocked_artifact_has_required_schema_fields() -> None:
    """Blocked artifact contains all REQUIRED_RESULT_FIELDS plus experiment-specific fields.

    Spec: REQ-REPAIR-056
    """
    from scripts.experiment_template import REQUIRED_RESULT_FIELDS, ExperimentTemplate

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpl = ExperimentTemplate(
            820,
            "test title",
            "results/experiment_820_gguf_import_fix_code_repair_v5.json",
            repo_root=Path(tmpdir),
        )
        tmpl.setup()
        artifact = build_blocked_artifact(
            tmpl,
            honest_verdict="still_blocked_import",
            blocked_reason="test reason",
            import_repair_attempted=True,
            import_repair_succeeded=False,
        )

    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["honest_verdict"] == "still_blocked_import"
    assert artifact["import_repair_attempted"] is True
    assert artifact["import_repair_succeeded"] is False
    assert artifact["n_problems"] == 20
    assert artifact["repair_delta"] == 0


def test_blocked_artifact_import_repair_attempted_field() -> None:
    """Blocked artifact correctly reflects import_repair_attempted flag.

    This tests the specific SCENARIO-REPAIR-089 requirement that the artifact
    records whether a repair was attempted, so the retrospective agent can
    distinguish 'import never worked' from 'repair tried and failed'.

    Spec: SCENARIO-REPAIR-089
    """
    from scripts.experiment_template import ExperimentTemplate

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpl = ExperimentTemplate(
            820,
            "test title",
            "results/experiment_820_gguf_import_fix_code_repair_v5.json",
            repo_root=Path(tmpdir),
        )
        tmpl.setup()
        artifact_no_attempt = build_blocked_artifact(
            tmpl,
            honest_verdict="still_blocked_import",
            blocked_reason="never attempted",
            import_repair_attempted=False,
            import_repair_succeeded=False,
        )
        artifact_attempted = build_blocked_artifact(
            tmpl,
            honest_verdict="still_blocked_import",
            blocked_reason="attempted but failed",
            import_repair_attempted=True,
            import_repair_succeeded=False,
        )

    assert artifact_no_attempt["import_repair_attempted"] is False
    assert artifact_attempted["import_repair_attempted"] is True
    assert artifact_attempted["import_repair_succeeded"] is False
