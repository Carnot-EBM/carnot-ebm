"""Tests for experiment_1079_live_sota_benchmark_v2.

Covers the helpers added by Exp 1079 (first live SOTA benchmark on GSM8K + HumanEval):
  - _is_gsm8k_correct: last-number extraction and comparison to ground truth
  - _extract_final_answer: numeric extraction edge cases
  - _detect_violations: VeriCoTStepValidator integration (mock mode)
  - _extract_code: code extraction from various model output formats
  - _execute_code: subprocess execution and failure isolation
  - _compute_verdict: all honest_verdict branches
  - _resolve_sota_path: returns a real path or None, never crashes
  - _run_experiment: writes blocked_no_gpu when CUDA is unavailable

These tests are CPU-only.  They cover the utility helpers and the GPU-absent
failure path.  The live GPU path is exercised by the actual experiment run.

Spec: REQ-VERIFY-083 (live_gpu evidence), REQ-EXTRACT-024 (VeriCoT),
      REQ-INFER-SOTA-001 (SOTA-tier model required for headline metric).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_module_importable() -> None:
    """The experiment module imports cleanly without requiring a GPU."""
    import scripts.experiment_1079_live_sota_benchmark_v2 as mod  # noqa: F401

    assert mod.EXP_ID == 1079
    assert mod.GSM8K_N == 100
    assert mod.HUMANEVAL_N == 50
    assert mod.SOTA_NAME == "Qwen3.6-35B-A3B"
    assert "Qwen3.6-35B-A3B-GGUF" in mod.SOTA_HF_ID


# ---------------------------------------------------------------------------
# _extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_final_answer_plain_number() -> None:
    """Returns the last integer from a plain response."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_final_answer

    assert _extract_final_answer("The answer is 42") == pytest.approx(42.0)


def test_extract_final_answer_takes_last() -> None:
    """Takes the LAST number, not the first."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_final_answer

    assert _extract_final_answer("Step 1: 5+3=8. Step 2: 8-1=7.") == pytest.approx(7.0)


def test_extract_final_answer_no_numbers() -> None:
    """Returns None when there are no numbers in the response."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_final_answer

    assert _extract_final_answer("I do not know.") is None


def test_extract_final_answer_decimal() -> None:
    """Handles decimal numbers correctly."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_final_answer

    assert _extract_final_answer("Result: 3.14") == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# _is_gsm8k_correct
# ---------------------------------------------------------------------------


def test_is_gsm8k_correct_exact_match() -> None:
    """Returns True when the last number matches the ground truth exactly."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _is_gsm8k_correct

    assert _is_gsm8k_correct("So the answer is 18", 18) is True


def test_is_gsm8k_correct_wrong_answer() -> None:
    """Returns False when the last number does not match the ground truth."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _is_gsm8k_correct

    assert _is_gsm8k_correct("So the answer is 19", 18) is False


def test_is_gsm8k_correct_no_numbers() -> None:
    """Returns False when no number is found (safe default)."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _is_gsm8k_correct

    assert _is_gsm8k_correct("I cannot compute this.", 18) is False


def test_is_gsm8k_correct_float_gt() -> None:
    """Works when ground truth is a float."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _is_gsm8k_correct

    assert _is_gsm8k_correct("Answer: 3.14", 3.14) is True


# ---------------------------------------------------------------------------
# _detect_violations (mock mode — no model call)
# ---------------------------------------------------------------------------


def test_detect_violations_correct_arithmetic() -> None:
    """Returns False when the CoT has correct arithmetic (Z3 SAT)."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _detect_violations

    cot = "She has 47 plus 28, which gives us 75."
    assert _detect_violations(cot) is False


def test_detect_violations_wrong_arithmetic() -> None:
    """Returns True when the CoT contains a provably wrong arithmetic claim (Z3 UNSAT)."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _detect_violations

    cot = "She has 47 plus 28, which gives 76."
    assert _detect_violations(cot) is True


def test_detect_violations_no_arithmetic() -> None:
    """Returns False for prose with no arithmetic claims — nothing to verify."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _detect_violations

    assert _detect_violations("The sky is blue and the grass is green.") is False


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------


def test_extract_code_from_fence() -> None:
    """Code inside a markdown fence is extracted correctly."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_code

    prompt = "def foo():\n"
    response = "Here is the implementation:\n```python\ndef foo():\n    return 42\n```"
    result = _extract_code(response, prompt)
    assert "def foo" in result
    assert "return 42" in result


def test_extract_code_indented_continuation() -> None:
    """An indented response is treated as a continuation of the function body."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_code

    prompt = "def add(a, b):\n"
    response = "    return a + b"
    result = _extract_code(response, prompt)
    assert "return a + b" in result
    assert "def add" in result


def test_extract_code_full_function() -> None:
    """A response containing 'def' is used directly."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _extract_code

    prompt = "def foo():\n    # implement me\n"
    response = "def foo():\n    return 42\n"
    result = _extract_code(response, prompt)
    assert "def foo" in result


# ---------------------------------------------------------------------------
# _execute_code
# ---------------------------------------------------------------------------


def test_execute_code_passing_test() -> None:
    """Returns True when the generated code passes all test assertions."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _execute_code

    code = "def add(a, b):\n    return a + b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n    assert f(0, 0) == 0\n"
    assert _execute_code(code, test, "add") is True


def test_execute_code_failing_test() -> None:
    """Returns False when the generated code fails a test assertion."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _execute_code

    code = "def add(a, b):\n    return a - b\n"  # wrong: subtract instead of add
    test = "def check(f):\n    assert f(1, 2) == 3\n"
    assert _execute_code(code, test, "add") is False


def test_execute_code_syntax_error() -> None:
    """Returns False when the generated code has a syntax error."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _execute_code

    code = "def add(a, b):\n    return a + b\n    extra bad :"
    test = "def check(f):\n    assert f(1, 2) == 3\n"
    assert _execute_code(code, test, "add") is False


# ---------------------------------------------------------------------------
# _compute_verdict
# ---------------------------------------------------------------------------


def test_verdict_both_improved() -> None:
    """When both tracks improve, verdict is positive_improvement_both."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _compute_verdict

    assert _compute_verdict(0.05, 0.10) == "positive_improvement_both"


def test_verdict_gsm8k_only() -> None:
    """When only GSM8K improves (HumanEval flat), verdict is positive_gsm8k_only."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _compute_verdict

    assert _compute_verdict(0.05, 0.0) == "positive_gsm8k_only"


def test_verdict_humaneval_only() -> None:
    """When only HumanEval improves (GSM8K flat), verdict is positive_humaneval_only."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _compute_verdict

    assert _compute_verdict(0.0, 0.36) == "positive_humaneval_only"


def test_verdict_honest_negative_degradation() -> None:
    """When at least one track degrades, verdict signals honest degradation."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _compute_verdict

    assert _compute_verdict(-0.05, 0.0) == "honest_negative_degradation"


def test_verdict_honest_no_improvement() -> None:
    """When neither track improves nor degrades, verdict is honest_negative_no_improvement."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _compute_verdict

    assert _compute_verdict(0.0, 0.0) == "honest_negative_no_improvement"


# ---------------------------------------------------------------------------
# _resolve_sota_path
# ---------------------------------------------------------------------------


def test_resolve_sota_path_returns_string_or_none() -> None:
    """Returns a string path containing the SOTA token, or None — never crashes."""
    from scripts.experiment_1079_live_sota_benchmark_v2 import _resolve_sota_path, SOTA_TOKEN

    p = _resolve_sota_path()
    assert p is None or (isinstance(p, str) and (SOTA_TOKEN in p or "3.6-35B" in p))


def test_resolve_sota_path_none_when_resolver_missing() -> None:
    """Returns None gracefully when carnot.inference.sota_models is unimportable."""
    import builtins
    import scripts.experiment_1079_live_sota_benchmark_v2 as mod

    real_import = builtins.__import__

    def _fake_import(name: str, *a, **kw):
        if name == "carnot.inference.sota_models":
            raise ImportError("simulated missing module")
        return real_import(name, *a, **kw)

    with patch.object(builtins, "__import__", side_effect=_fake_import):
        assert mod._resolve_sota_path() is None


# ---------------------------------------------------------------------------
# _run_experiment — GPU-absent failure path
# ---------------------------------------------------------------------------


def test_run_experiment_blocks_when_no_gpu(tmp_path: Path, monkeypatch) -> None:
    """When CUDA is unavailable the function returns a blocked_no_gpu artifact."""
    import scripts.experiment_1079_live_sota_benchmark_v2 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    prev = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1079.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev

    assert artifact["honest_verdict"] == "blocked_no_gpu"
    assert artifact["status"] == "blocked"
    assert artifact["inference_mode"] == "blocked_no_gpu"
    assert artifact["gsm8k_n_questions"] == 0
    assert artifact["humaneval_n_problems"] == 0


def test_run_experiment_blocks_when_model_missing(tmp_path: Path, monkeypatch) -> None:
    """When the SOTA GGUF is not cached, returns blocked_no_gpu artifact."""
    import scripts.experiment_1079_live_sota_benchmark_v2 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(mod, "_resolve_sota_path", lambda: None)
    prev = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1079.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_no_gpu"
    assert artifact.get("block_reason") == "model_tier_violation"


def test_blocked_artifact_has_all_required_fields(tmp_path: Path, monkeypatch) -> None:
    """Even the blocked-path artifact carries all schema fields required by the task spec."""
    import scripts.experiment_1079_live_sota_benchmark_v2 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    prev = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1079.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev

    required = {
        "models_used",
        "inference_mode",
        "gsm8k_n_questions",
        "gsm8k_baseline_accuracy",
        "gsm8k_corrected_accuracy",
        "gsm8k_net_improvement",
        "gsm8k_extraction_tp_rate",
        "humaneval_n_problems",
        "humaneval_pass_at_1_before",
        "humaneval_pass_at_1_after",
        "humaneval_net_improvement",
        "honest_verdict",
        # Required by ExperimentTemplate.build_result
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
    }
    missing = required - set(artifact)
    assert not missing, f"missing fields: {missing}"
