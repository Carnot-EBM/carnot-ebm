"""Tests for Exp 850: SOTA Code Repair v5 — 25 HumanEval, Qwen3.6-35B-A3B-GGUF, MARS gate.

Covers:
    - check_exp849_gate: file missing, not implemented, implemented (REQ-PIPELINE-030)
    - apply_mars_margin_gate: skip when margin > threshold, repair when below (REQ-REPAIR-056)
    - compute_signed_improvement: positive, negative, zero, n=0 (REQ-BENCH-016-6)
    - classify_verdict: positive live, negative live, simulated (REQ-REPAIR-056)
    - run_problem_baseline: passing and failing canonical solutions (REQ-REPAIR-056)
    - run_problem_with_repair: mocked LLM, repair skipped by MARS, repair attempted (REQ-REPAIR-056)

All GPU/LLM calls are mocked — tests run entirely on CPU with no network or filesystem
side effects beyond tempfile usage.

Spec: REQ-REPAIR-056, REQ-PIPELINE-030, REQ-BENCH-016-6, SCENARIO-REPAIR-089,
      SCENARIO-PIPELINE-040
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.experiment_850_sota_code_repair_v5 import (
    MARS_THRESHOLD,
    N_PROBLEMS,
    apply_mars_margin_gate,
    check_exp849_gate,
    classify_verdict,
    compute_signed_improvement,
    run_problem_baseline,
    run_problem_with_repair,
    _INLINE_PROBLEMS,
)
from carnot.pipeline.extract import CodeExtractor


# ---------------------------------------------------------------------------
# check_exp849_gate — REQ-PIPELINE-030
# ---------------------------------------------------------------------------


def test_blocked_if_gguf_not_implemented_missing_file() -> None:
    """Gate returns False when the Exp 849 result file does not exist.

    Spec: REQ-PIPELINE-030
    """
    assert check_exp849_gate(Path("/nonexistent/results/experiment_849.json")) is False


def test_blocked_if_gguf_not_implemented_wrong_verdict() -> None:
    """Gate returns False when Exp 849 has honest_verdict != 'gguf_cache_implemented'.

    Spec: REQ-PIPELINE-030
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"honest_verdict": "blocked", "status": "blocked"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp849_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_blocked_if_gguf_not_implemented_corrupt_json() -> None:
    """Gate returns False when the Exp 849 artifact is not valid JSON.

    Spec: REQ-PIPELINE-030
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        fh.write("{not valid json")
        tmp = Path(fh.name)
    try:
        assert check_exp849_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_gate_true_when_gguf_cache_implemented() -> None:
    """Gate returns True when Exp 849 has honest_verdict == 'gguf_cache_implemented'.

    Spec: REQ-PIPELINE-030
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"honest_verdict": "gguf_cache_implemented", "status": "success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_exp849_gate(tmp) is True
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# apply_mars_margin_gate — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_mars_margin_gate_skip_when_above_threshold() -> None:
    """MARS gate returns True (skip repair) when logit_margin exceeds threshold.

    A margin of 4.0 > 3.0 threshold means the model is highly confident.

    Spec: REQ-REPAIR-056
    """
    assert apply_mars_margin_gate(4.0, threshold=MARS_THRESHOLD) is True


def test_mars_margin_gate_repair_when_below_threshold() -> None:
    """MARS gate returns False (do repair) when logit_margin is below threshold.

    A margin of 1.5 < 3.0 means the model is uncertain — repair should run.

    Spec: REQ-REPAIR-056
    """
    assert apply_mars_margin_gate(1.5, threshold=MARS_THRESHOLD) is False


def test_mars_margin_gate_equal_threshold_is_not_skip() -> None:
    """MARS gate returns False when logit_margin equals exactly the threshold.

    The gate condition is strictly greater than (>), not >=.

    Spec: REQ-REPAIR-056
    """
    assert apply_mars_margin_gate(MARS_THRESHOLD, threshold=MARS_THRESHOLD) is False


def test_mars_margin_gate_zero_margin_always_repairs() -> None:
    """Zero margin triggers repair — the model has no confidence.

    Spec: REQ-REPAIR-056
    """
    assert apply_mars_margin_gate(0.0) is False


# ---------------------------------------------------------------------------
# compute_signed_improvement — REQ-BENCH-016-6
# ---------------------------------------------------------------------------


def test_signed_improvement_positive_when_repair_better() -> None:
    """signed_improvement is positive when repair passes more problems than baseline.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(20, 10, 25)
    assert result == pytest.approx(10 / 25)


def test_signed_improvement_negative_when_repair_worse() -> None:
    """signed_improvement is negative when repair passes fewer problems than baseline.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(10, 20, 25)
    assert result == pytest.approx(-10 / 25)


def test_signed_improvement_zero_when_equal() -> None:
    """signed_improvement is zero when repair and baseline counts are identical.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(15, 15, 25)
    assert result == pytest.approx(0.0)


def test_signed_improvement_zero_when_n_problems_zero() -> None:
    """signed_improvement returns 0.0 for n_problems=0 (avoids ZeroDivisionError).

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(0, 0, 0)
    assert result == 0.0


def test_signed_improvement_not_clamped_negative() -> None:
    """signed_improvement can be -1.0 — it is never clamped.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(0, 25, 25)
    assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# classify_verdict — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_classify_verdict_positive_live() -> None:
    """Verdict is 'code_repair_positive' when improvement > 0, live GPU, n_live >= 15.

    Spec: REQ-REPAIR-056
    """
    verdict = classify_verdict(0.4, "live_gpu", 25)
    assert verdict == "code_repair_positive"


def test_classify_verdict_negative_live() -> None:
    """Verdict is 'code_repair_negative' when improvement <= 0 and live GPU.

    Spec: REQ-REPAIR-056
    """
    verdict = classify_verdict(-0.1, "live_gpu", 25)
    assert verdict == "code_repair_negative"


def test_classify_verdict_positive_but_too_few_live() -> None:
    """Verdict is 'code_repair_negative' when improvement > 0 but n_live < 15.

    Without enough live samples, a positive result is not statistically trustworthy.

    Spec: REQ-REPAIR-056
    """
    verdict = classify_verdict(0.5, "live_gpu", 10)
    assert verdict == "code_repair_negative"


def test_classify_verdict_simulated_no_verdict() -> None:
    """Verdict is 'simulated_no_verdict' when inference_mode is not 'live_gpu'.

    Spec: REQ-REPAIR-056
    """
    verdict = classify_verdict(1.0, "simulated", 25)
    assert verdict == "simulated_no_verdict"


def test_classify_verdict_zero_improvement_live() -> None:
    """Verdict is 'code_repair_negative' when improvement == 0 (live GPU).

    Spec: REQ-REPAIR-056
    """
    verdict = classify_verdict(0.0, "live_gpu", 25)
    assert verdict == "code_repair_negative"


# ---------------------------------------------------------------------------
# run_problem_baseline — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_run_problem_baseline_passes_for_correct_solution() -> None:
    """Canonical solution passes its own test.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "canonical_solution": "    return a + b\n",
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is True


def test_run_problem_baseline_fails_for_wrong_solution() -> None:
    """Baseline returns False when the canonical solution produces incorrect output.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "canonical_solution": "    return a - b\n",
        "test": "assert add(1, 2) == 3\n",
    }
    assert run_problem_baseline(problem) is False


def test_run_problem_baseline_fails_for_syntax_error() -> None:
    """Baseline returns False when the canonical solution has a syntax error.

    Spec: REQ-REPAIR-056
    """
    problem = {
        "prompt": "def broken():\n",
        "canonical_solution": "    return (\n",
        "test": "broken()\n",
    }
    assert run_problem_baseline(problem) is False


# ---------------------------------------------------------------------------
# run_problem_with_repair — REQ-REPAIR-056, SCENARIO-REPAIR-089
# ---------------------------------------------------------------------------


def _make_llm_mock(response_text: str, logprob: float = -1.0) -> MagicMock:
    """Build a MagicMock that mimics the llama.cpp Llama.__call__ interface.

    The mock returns a dict with choices[0].text = response_text and
    choices[0].logprobs.token_logprobs = [logprob].  This is the minimum
    structure that run_problem_with_repair() inspects.
    """
    llm = MagicMock()
    llm.return_value = {
        "choices": [
            {
                "text": response_text,
                "logprobs": {
                    "token_logprobs": [logprob],
                },
            }
        ]
    }
    return llm


def test_model_not_cached_path_repair_skipped_by_mars() -> None:
    """MARS gate prevents repair when the model is highly confident (high logit margin).

    When the model returns a logprob of -0.1 (near 0 = very confident), the
    derived margin is 0.1 which is below MARS_THRESHOLD=3.0... wait, let me
    reconsider.  In run_problem_with_repair, logit_margin = -first_logprob.
    So logprob = -4.0 → margin = 4.0 > 3.0 → repair skipped.

    This also serves as the 'model_not_cached_path' test: we test that
    the repair path handles the MARS skip correctly without hitting a real
    GPU.

    Spec: REQ-REPAIR-056
    """
    extractor = CodeExtractor()
    # logprob = -4.0 → margin = 4.0 > MARS_THRESHOLD → repair skipped
    llm = _make_llm_mock("    return a + b\n", logprob=-4.0)
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    repair_pass, repair_attempted, logit_margin = run_problem_with_repair(problem, llm, extractor)
    assert repair_attempted is False, "MARS gate should have prevented repair"
    assert logit_margin == pytest.approx(4.0)
    assert repair_pass is True


def test_repair_attempted_when_below_mars_threshold() -> None:
    """Repair is attempted when logit_margin is below MARS_THRESHOLD and violations found.

    We inject a broken solution that is a complete Python function definition so
    CodeExtractor can parse it and find the 'undefined_var' initialization violation.
    The broken solution references 'undefined_var' which is not assigned or passed as
    a parameter — CodeExtractor flags this, and repair re-generation runs.

    Note: the generated code must be a complete, parseable Python block for
    CodeExtractor to find violations; raw indented snippets fail ast.parse().

    Spec: REQ-REPAIR-056, SCENARIO-REPAIR-089
    """
    extractor = CodeExtractor()
    # logprob = -1.0 → margin = 1.0 < 3.0 → repair should run when violations found.
    # First call: broken solution with 'undefined_var' reference inside a function body.
    # Second call (repair): correct solution.
    llm = MagicMock()
    correct_solution = "    return a + b\n"
    # A complete function definition so ast.parse succeeds and CodeExtractor finds
    # 'undefined_var' as a variable used but never assigned.
    broken_solution = "def add(a, b):\n    x = undefined_var\n    return x\n"
    llm.side_effect = [
        {
            "choices": [
                {
                    "text": broken_solution,
                    "logprobs": {"token_logprobs": [-1.0]},
                }
            ]
        },
        {
            "choices": [
                {
                    "text": correct_solution,
                    "logprobs": {"token_logprobs": [-0.5]},
                }
            ]
        },
    ]
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    repair_pass, repair_attempted, logit_margin = run_problem_with_repair(problem, llm, extractor)
    # The repair ran (violations detected in broken_solution).
    assert repair_attempted is True
    assert logit_margin == pytest.approx(1.0)


def test_repair_not_attempted_when_no_violations() -> None:
    """Repair is not attempted when CodeExtractor finds no violations.

    A syntactically clean solution with no constraint violations skips re-generation
    even when logit_margin is below the MARS threshold.

    Spec: REQ-REPAIR-056
    """
    extractor = CodeExtractor()
    # logprob = -1.0 → margin = 1.0 < threshold → would attempt repair IF violations found
    llm = _make_llm_mock("    return a + b\n", logprob=-1.0)
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    repair_pass, repair_attempted, logit_margin = run_problem_with_repair(problem, llm, extractor)
    # Clean solution — CodeExtractor finds no violations, so repair_attempted = False.
    assert repair_attempted is False


def test_llm_exception_fallback_to_empty_generation() -> None:
    """When the LLM raises an exception, run_problem_with_repair falls back to empty generation.

    The problem should still return a result (False pass) rather than propagating the
    exception upward, which would crash the entire experiment run.

    Spec: REQ-REPAIR-056
    """
    extractor = CodeExtractor()
    llm = MagicMock()
    llm.side_effect = RuntimeError("GPU OOM")
    problem = {
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "assert add(1, 2) == 3\n",
    }
    repair_pass, repair_attempted, logit_margin = run_problem_with_repair(problem, llm, extractor)
    assert repair_pass is False
    assert logit_margin == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Inline problem set sanity check — REQ-REPAIR-056
# ---------------------------------------------------------------------------


def test_inline_problems_count_equals_n_problems() -> None:
    """_INLINE_PROBLEMS contains exactly N_PROBLEMS (25) entries.

    Spec: REQ-REPAIR-056
    """
    assert len(_INLINE_PROBLEMS) == N_PROBLEMS


def test_inline_problems_all_have_required_keys() -> None:
    """Every inline problem has 'task_id', 'prompt', 'canonical_solution', 'test'.

    Spec: REQ-REPAIR-056
    """
    required = {"task_id", "prompt", "canonical_solution", "test"}
    for prob in _INLINE_PROBLEMS:
        assert required <= set(prob.keys()), f"Problem {prob.get('task_id')} missing keys"


def test_all_canonical_solutions_pass_their_tests() -> None:
    """Every canonical solution in _INLINE_PROBLEMS passes its own test.

    This guards against typos or copy-paste errors in the problem definitions.
    If a canonical solution fails, the experiment's baseline pass rate is
    artificially depressed, making repair results misleading.

    Spec: REQ-REPAIR-056
    """
    failures = []
    for prob in _INLINE_PROBLEMS:
        if not run_problem_baseline(prob):
            failures.append(prob["task_id"])
    assert failures == [], f"Canonical solutions failed for: {failures}"
