"""Tests for scripts/experiment_680_humaneval_vr.py — HumanEval VR with assertion forcing.

Covers:
- extract_assert_comments parsing (SCENARIO-VERIFY-208)
- compute_honest_verdict_680 logic for all three cases (SCENARIO-VERIFY-209)
- extract_python_code fence stripping
- execute_code pass/fail detection
- _build_blocked_artifact schema completeness
- blocked exit when CARNOT_FORCE_LIVE is absent
- deliverable JSON schema validation when artifact exists on disk

Spec: REQ-VERIFY-157, REQ-VERIFY-158,
      SCENARIO-VERIFY-208, SCENARIO-VERIFY-209
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_680_humaneval_vr as mod
from scripts.experiment_680_humaneval_vr import (
    DELIVERABLE,
    EXP_ID,
    HUMANEVAL_PROBLEMS,
    N_PROBLEMS,
    SCHEMA,
    _build_blocked_artifact,
    compute_honest_verdict_680,
    execute_code,
    extract_assert_comments,
    extract_python_code,
)


# ---------------------------------------------------------------------------
# extract_assert_comments — SCENARIO-VERIFY-208
# ---------------------------------------------------------------------------


def test_extract_assert_comments_standard_case() -> None:
    """Extract (var, value) pairs from standard # ASSERT: lines.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-208
    """
    code = "x = 5\n# ASSERT: x == 5\ny = x + 3\n# ASSERT: y == 8"
    result = extract_assert_comments(code)
    assert result == [("x", "5"), ("y", "8")], f"Got {result}"


def test_extract_assert_comments_empty_code() -> None:
    """Empty code returns empty list.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-208
    """
    assert extract_assert_comments("") == []


def test_extract_assert_comments_no_asserts() -> None:
    """Code with no # ASSERT: lines returns empty list.

    Spec: REQ-VERIFY-158
    """
    code = "def foo(x):\n    return x + 1\n"
    assert extract_assert_comments(code) == []


def test_extract_assert_comments_multiple_per_block() -> None:
    """Multiple ASSERT lines in same function are all captured.

    WHY test multiple: the assert-comment forcing can produce many per function.
    We need to collect them all for the repair feedback summary.

    Spec: REQ-VERIFY-158
    """
    code = (
        "total = 0\n"
        "for i in range(5):\n"
        "    total += i\n"
        "# ASSERT: total == 10\n"
        "result = total * 2\n"
        "# ASSERT: result == 20\n"
    )
    result = extract_assert_comments(code)
    assert len(result) == 2
    assert result[0] == ("total", "10")
    assert result[1] == ("result", "20")


def test_extract_assert_comments_extra_spaces() -> None:
    """ASSERT: lines with extra whitespace around == are still parsed.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-208
    """
    code = "#  ASSERT:  count  ==  42"
    result = extract_assert_comments(code)
    assert len(result) == 1
    assert result[0][0] == "count"
    assert result[0][1] == "42"


# ---------------------------------------------------------------------------
# extract_python_code — REQ-VERIFY-157
# ---------------------------------------------------------------------------


def test_extract_python_code_fenced_block() -> None:
    """Extracts code from ```python ... ``` fenced block.

    Spec: REQ-VERIFY-157
    """
    response = "Here is the solution:\n```python\ndef foo():\n    return 42\n```\n"
    result = extract_python_code(response)
    assert "def foo():" in result
    assert "```" not in result


def test_extract_python_code_unfenced() -> None:
    """When no fences are present, returns stripped response as-is.

    Spec: REQ-VERIFY-157
    """
    code = "def foo():\n    return 42"
    result = extract_python_code(code)
    assert result == code


def test_extract_python_code_plain_fence() -> None:
    """Extracts code from ``` ... ``` (no language tag) fenced block.

    Spec: REQ-VERIFY-157
    """
    response = "```\ndef bar():\n    pass\n```"
    result = extract_python_code(response)
    assert "def bar():" in result
    assert "```" not in result


# ---------------------------------------------------------------------------
# execute_code — REQ-VERIFY-157
# ---------------------------------------------------------------------------


def test_execute_code_correct_function() -> None:
    """execute_code returns True when function is correct and test harness prints PASS.

    Spec: REQ-VERIFY-157, REQ-VERIFY-157-2
    """
    func_code = "def add(a, b):\n    return a + b\n"
    test_code = "result = add(2, 3)\nassert result == 5\nprint('PASS')\n"
    assert execute_code(func_code, test_code) is True


def test_execute_code_wrong_function() -> None:
    """execute_code returns False when function produces wrong output.

    Spec: REQ-VERIFY-157
    """
    func_code = "def add(a, b):\n    return a - b\n"  # wrong: subtraction not addition
    test_code = "result = add(2, 3)\nassert result == 5\nprint('PASS')\n"
    assert execute_code(func_code, test_code) is False


def test_execute_code_syntax_error() -> None:
    """execute_code returns False on Python syntax errors (not raise to caller).

    Spec: REQ-VERIFY-157
    """
    func_code = "def add(a b):\n    return a + b\n"  # missing comma = syntax error
    test_code = "print('PASS')\n"
    assert execute_code(func_code, test_code) is False


def test_execute_code_timeout() -> None:
    """execute_code returns False when code exceeds the timeout.

    WHY test timeout: an infinite loop in generated code must not hang the experiment.

    Spec: REQ-VERIFY-157, REQ-VERIFY-157-3
    """
    func_code = "def loop():\n    while True: pass\n"
    test_code = "loop()\nprint('PASS')\n"
    # Use very short timeout to keep test fast
    result = execute_code(func_code, test_code, timeout=1)
    assert result is False


def test_execute_code_no_pass_in_stdout() -> None:
    """execute_code returns False when exit code is 0 but PASS not in stdout.

    Spec: REQ-VERIFY-157, REQ-VERIFY-157-2
    """
    func_code = "def noop():\n    pass\n"
    test_code = "noop()\n# no print PASS\n"
    assert execute_code(func_code, test_code) is False


# ---------------------------------------------------------------------------
# compute_honest_verdict_680 — SCENARIO-VERIFY-209
# ---------------------------------------------------------------------------


def test_verdict_positive() -> None:
    """signed_improvement > 0 AND live_gpu → 'code_vr_positive'.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-209
    """
    assert compute_honest_verdict_680(0.10, "live_gpu") == "code_vr_positive"


def test_verdict_no_improvement_negative() -> None:
    """Negative signed_improvement AND live_gpu → 'code_vr_no_improvement'.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-209
    """
    assert compute_honest_verdict_680(-0.05, "live_gpu") == "code_vr_no_improvement"


def test_verdict_no_improvement_zero() -> None:
    """signed_improvement == 0.0 → 'code_vr_no_improvement' (not positive).

    Spec: REQ-VERIFY-158
    """
    assert compute_honest_verdict_680(0.0, "live_gpu") == "code_vr_no_improvement"


def test_verdict_blocked() -> None:
    """inference_mode='blocked' always yields 'code_vr_blocked'.

    Spec: REQ-VERIFY-158, SCENARIO-VERIFY-209
    """
    assert compute_honest_verdict_680(0.5, "blocked") == "code_vr_blocked"
    assert compute_honest_verdict_680(0.0, "blocked") == "code_vr_blocked"
    assert compute_honest_verdict_680(-0.1, "blocked") == "code_vr_blocked"


# ---------------------------------------------------------------------------
# _build_blocked_artifact — schema completeness
# ---------------------------------------------------------------------------

REQUIRED_BLOCKED_FIELDS = {
    "experiment", "schema", "run_date", "status", "honest_verdict",
    "blocked_reason", "inference_mode", "n_problems",
    "baseline_pass_at_1", "post_pass_at_1", "signed_improvement",
    "assert_comments_found", "repair_attempts",
}


def test_blocked_artifact_has_all_required_fields() -> None:
    """_build_blocked_artifact emits all required schema fields.

    Spec: REQ-VERIFY-157, REQ-VERIFY-158
    """
    artifact = _build_blocked_artifact("test reason")
    missing = REQUIRED_BLOCKED_FIELDS - set(artifact.keys())
    assert not missing, f"Missing fields: {missing}"


def test_blocked_artifact_values() -> None:
    """_build_blocked_artifact sets correct values for key fields.

    Spec: REQ-VERIFY-158
    """
    artifact = _build_blocked_artifact("CARNOT_FORCE_LIVE=1 not set")
    assert artifact["experiment"] == EXP_ID
    assert artifact["schema"] == SCHEMA
    assert artifact["honest_verdict"] == "code_vr_blocked"
    assert artifact["inference_mode"] == "blocked"
    assert artifact["status"] == "blocked"
    assert artifact["baseline_pass_at_1"] == 0.0
    assert artifact["signed_improvement"] == 0.0


# ---------------------------------------------------------------------------
# Blocked exit when CARNOT_FORCE_LIVE is absent
# ---------------------------------------------------------------------------


def test_blocked_exit_when_no_carnot_force_live(tmp_path: Path) -> None:
    """_run_inner writes a blocked artifact and exits 0 when CARNOT_FORCE_LIVE is unset.

    Spec: REQ-VERIFY-158, REQ-VERIFY-158-4
    """
    written: list[dict] = []

    class _FakeWriter:
        def write(self, data: dict) -> None:
            written.append(data)

    class _FakeTemplate:
        def setup(self) -> None:
            pass

        def assert_deliverable_written(self) -> None:
            pass

        def build_result(self, *a, **kw):
            return {}

        def setup_gpu(self, *a, **kw):
            return {"all_healthy": True, "models": []}

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
    ):
        env = dict(os.environ)
        env.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env, clear=True):
            mock_watchdog = MagicMock()
            with pytest.raises(SystemExit) as exc_info:
                mod._run_inner(mock_watchdog)

    assert exc_info.value.code == 0
    assert len(written) == 1
    result = written[0]
    assert result["honest_verdict"] == "code_vr_blocked"
    assert result["inference_mode"] == "blocked"
    assert result["experiment"] == EXP_ID


def test_blocked_exit_artifact_has_complete_schema(tmp_path: Path) -> None:
    """Blocked artifact from _run_inner has all required schema fields.

    Spec: REQ-VERIFY-158
    """
    written: list[dict] = []

    class _FakeWriter:
        def write(self, data: dict) -> None:
            written.append(data)

    class _FakeTemplate:
        def setup(self) -> None:
            pass

        def assert_deliverable_written(self) -> None:
            pass

        def build_result(self, *a, **kw):
            return {}

        def setup_gpu(self, *a, **kw):
            return {"all_healthy": True, "models": []}

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
    ):
        env = dict(os.environ)
        env.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit):
                mod._run_inner(MagicMock())

    missing = REQUIRED_BLOCKED_FIELDS - set(written[0].keys())
    assert not missing, f"Missing fields in blocked artifact: {missing}"


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 680.  Spec: REQ-VERIFY-157"""
    assert EXP_ID == 680


def test_n_problems() -> None:
    """N_PROBLEMS must be 25.  Spec: REQ-VERIFY-157"""
    assert N_PROBLEMS == 25


def test_schema() -> None:
    """SCHEMA encodes experiment version.  Spec: REQ-VERIFY-157"""
    assert SCHEMA == "carnot.humaneval_vr.v1"


def test_humaneval_problems_count() -> None:
    """HUMANEVAL_PROBLEMS must have exactly 25 entries.  Spec: REQ-VERIFY-157"""
    assert len(HUMANEVAL_PROBLEMS) == N_PROBLEMS


def test_humaneval_problems_have_required_keys() -> None:
    """Each problem dict has prompt, entry_point, and test_code.  Spec: REQ-VERIFY-157"""
    for i, p in enumerate(HUMANEVAL_PROBLEMS):
        assert "prompt" in p, f"Problem {i} missing 'prompt'"
        assert "entry_point" in p, f"Problem {i} missing 'entry_point'"
        assert "test_code" in p, f"Problem {i} missing 'test_code'"


# ---------------------------------------------------------------------------
# Deliverable JSON on disk — validates actual run artifact when present
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """Deliverable JSON must exist and contain all required schema fields.

    Spec: REQ-VERIFY-157, REQ-VERIFY-158
    """
    result_path = _REPO_ROOT / DELIVERABLE
    if not result_path.exists():
        pytest.skip("Deliverable not yet written — run the experiment first")

    data = json.loads(result_path.read_text())

    required = {
        "experiment", "schema", "run_date", "status", "honest_verdict",
        "inference_mode", "n_problems", "baseline_pass_at_1", "post_pass_at_1",
        "signed_improvement", "assert_comments_found", "repair_attempts",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing fields in deliverable: {missing}"

    assert data["experiment"] == EXP_ID
    valid_verdicts = {"code_vr_positive", "code_vr_no_improvement", "code_vr_blocked"}
    assert data["honest_verdict"] in valid_verdicts, (
        f"Unknown honest_verdict: {data['honest_verdict']}"
    )
    assert 0.0 <= data["baseline_pass_at_1"] <= 1.0
    assert 0.0 <= data["post_pass_at_1"] <= 1.0
