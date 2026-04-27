"""Tests for Experiment 954: Live GPU Probe Validation.

Covers:
- _correct_prompt: contains question text and 'correctly'
- _error_injected_prompt: contains question text and 'INCORRECT'
- _approx_entropy_from_top_k: returns 0 for empty input
- _approx_entropy_from_top_k: returns positive entropy for non-degenerate top-k
- _approx_entropy_from_top_k: returns 0 for single-token distribution
- compute_spilled_energy_from_llama_result: returns (0.0, []) for empty choices
- compute_spilled_energy_from_llama_result: returns non-zero spill for real logprob data
- _build_virtual_hidden_states: returns N_VIRTUAL_LAYERS arrays for normal input
- _build_virtual_hidden_states: returns stub arrays for empty logprobs
- _build_virtual_hidden_states: each array has shape [>=2, 2]
- _verify_response_thinkprm: returns 1.0 for CORRECT verdict
- _verify_response_thinkprm: returns 0.0 for INCORRECT verdict
- _verify_response_thinkprm: returns 0.5 for uncertain/no-parse
- main: writes blocked artifact when CARNOT_FORCE_LIVE not set
- FACTUAL_QUESTIONS: has exactly 50 entries

Spec: REQ-PROBE-022, REQ-VERIFY-098, REQ-PROBE-010,
      SCENARIO-PROBE-022, SCENARIO-VERIFY-130, SCENARIO-PROBE-015
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiment_954_live_gpu_probe_validation import (  # noqa: E402
    FACTUAL_QUESTIONS,
    N_VIRTUAL_LAYERS,
    _approx_entropy_from_top_k,
    _build_virtual_hidden_states,
    _correct_prompt,
    _error_injected_prompt,
    _verify_response_thinkprm,
    compute_spilled_energy_from_llama_result,
    main,
)


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


def test_correct_prompt_contains_question():
    """_correct_prompt includes the question text and instructs for correct answer."""
    q = "What is the capital of France?"
    prompt = _correct_prompt(q)
    assert q in prompt
    assert "correctly" in prompt.lower() or "correct" in prompt.lower()


def test_error_injected_prompt_contains_question():
    """_error_injected_prompt includes the question and instructs for wrong answer."""
    q = "What is 2 + 2?"
    prompt = _error_injected_prompt(q)
    assert q in prompt
    assert "INCORRECT" in prompt or "wrong" in prompt.lower()


# ---------------------------------------------------------------------------
# _approx_entropy_from_top_k tests
# ---------------------------------------------------------------------------


def test_approx_entropy_empty_returns_zero():
    """Empty top-k dict returns 0.0 entropy."""
    assert _approx_entropy_from_top_k({}) == 0.0


def test_approx_entropy_single_token_near_zero():
    """Single-token distribution (all mass on one token) has near-zero entropy."""
    # One token with logprob=0.0 (probability=1.0).
    result = _approx_entropy_from_top_k({"the": 0.0})
    assert result < 1e-6


def test_approx_entropy_uniform_distribution_positive():
    """Uniform distribution over K tokens has positive entropy."""
    # All tokens equally likely: logprob = log(1/4) = -log(4) for each.
    import math

    k = 4
    lp = -math.log(k)
    top_k = {f"tok{i}": lp for i in range(k)}
    entropy = _approx_entropy_from_top_k(top_k)
    expected = math.log(k)  # H(uniform_K) = log(K)
    assert abs(entropy - expected) < 0.1


# ---------------------------------------------------------------------------
# compute_spilled_energy_from_llama_result tests
# ---------------------------------------------------------------------------


def test_spilled_energy_empty_choices_returns_zero():
    """Empty choices list returns (0.0, [])."""
    spill, lp = compute_spilled_energy_from_llama_result({"choices": []})
    assert spill == 0.0
    assert lp == []


def test_spilled_energy_no_logprobs_returns_zero():
    """Missing logprobs field returns (0.0, [])."""
    result = {"choices": [{"text": "hello", "logprobs": None}]}
    spill, lp = compute_spilled_energy_from_llama_result(result)
    assert spill == 0.0
    assert lp == []


def test_spilled_energy_returns_nonnegative_for_valid_data():
    """Non-empty logprobs yields non-negative spill score."""
    # Simulate a 5-token response with top-5 logprobs at each position.
    import math

    token_lp = [-1.5, -2.0, -0.8, -3.0, -1.2]
    top_k_lp = {"a": -1.5, "b": -2.0, "c": -2.5, "d": -3.0, "e": -3.5}
    result = {
        "choices": [
            {
                "text": "test response",
                "logprobs": {
                    "token_logprobs": token_lp,
                    "top_logprobs": [top_k_lp] * 5,
                },
            }
        ]
    }
    spill, lp = compute_spilled_energy_from_llama_result(result)
    assert spill >= 0.0
    assert lp == token_lp


# ---------------------------------------------------------------------------
# _build_virtual_hidden_states tests
# ---------------------------------------------------------------------------


def test_build_virtual_hidden_states_correct_layer_count():
    """Returns exactly N_VIRTUAL_LAYERS arrays for a normal logprob sequence."""
    lp = list(np.random.uniform(-3.0, -0.5, 50))
    hs = _build_virtual_hidden_states(lp)
    assert len(hs) == N_VIRTUAL_LAYERS


def test_build_virtual_hidden_states_empty_returns_stubs():
    """Empty logprob list returns N_VIRTUAL_LAYERS stub arrays."""
    hs = _build_virtual_hidden_states([])
    assert len(hs) == N_VIRTUAL_LAYERS
    for h in hs:
        assert h.ndim == 2
        assert h.shape[1] == 2


def test_build_virtual_hidden_states_shape_correct():
    """Each virtual layer array has shape [>=2, 2]."""
    lp = list(np.random.uniform(-3.0, -0.5, 60))
    hs = _build_virtual_hidden_states(lp)
    for h in hs:
        assert h.ndim == 2
        assert h.shape[1] == 2
        assert h.shape[0] >= 2


def test_build_virtual_hidden_states_values_normalised():
    """Normalised logprob column values lie in [0, 1]."""
    lp = list(np.random.uniform(-5.0, -0.1, 48))
    hs = _build_virtual_hidden_states(lp)
    for h in hs:
        # First column (logprob_norm) should be in [0, 1].
        assert np.all(h[:, 0] >= -1e-6)
        assert np.all(h[:, 0] <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# _verify_response_thinkprm tests
# ---------------------------------------------------------------------------


def _make_llm_caller(verdict_str: str):
    """Build a stub LLM caller that returns a fixed VERDICT."""

    def caller(prompt: str) -> str:
        return f"Step 1: analysis.\nStep 2: reasoning.\nVERDICT: {verdict_str}"

    return caller


def test_verify_response_thinkprm_correct_returns_1():
    """CORRECT verdict from verifier returns score 1.0."""
    score = _verify_response_thinkprm(
        "What is 2+2?", "The answer is 4.", _make_llm_caller("CORRECT")
    )
    assert score == 1.0


def test_verify_response_thinkprm_incorrect_returns_0():
    """INCORRECT verdict from verifier returns score 0.0."""
    score = _verify_response_thinkprm(
        "What is 2+2?", "The answer is 5.", _make_llm_caller("INCORRECT")
    )
    assert score == 0.0


def test_verify_response_thinkprm_no_verdict_returns_half():
    """No parseable VERDICT line returns 0.5 (uncertain)."""
    score = _verify_response_thinkprm(
        "What is 2+2?", "I don't know.", lambda p: "Some reasoning without a verdict."
    )
    assert score == 0.5


# ---------------------------------------------------------------------------
# FACTUAL_QUESTIONS corpus test
# ---------------------------------------------------------------------------


def test_factual_questions_count():
    """FACTUAL_QUESTIONS must have exactly 50 entries for N_QUESTIONS=50."""
    assert len(FACTUAL_QUESTIONS) == 50


def test_factual_questions_all_have_string_fields():
    """Every question entry is a 2-tuple of non-empty strings."""
    for q, a in FACTUAL_QUESTIONS:
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(a, str) and len(a) > 0


# ---------------------------------------------------------------------------
# main() blocked path tests
# ---------------------------------------------------------------------------


def test_main_blocked_without_live_flag(tmp_path):
    """main() writes blocked artifact when CARNOT_FORCE_LIVE is not set."""
    deliverable = tmp_path / "exp954_out.json"
    with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
        with patch(
            "experiment_954_live_gpu_probe_validation.DELIVERABLE",
            str(deliverable),
        ):
            try:
                main()
            except SystemExit:
                pass

    assert deliverable.exists(), "blocked artifact must be written"
    data = json.loads(deliverable.read_text())
    assert data["honest_verdict"] == "blocked_no_live_gpu"


def test_main_blocked_fields_present(tmp_path):
    """Blocked artifact contains all required schema fields."""
    deliverable = tmp_path / "exp954_out2.json"
    with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
        with patch(
            "experiment_954_live_gpu_probe_validation.DELIVERABLE",
            str(deliverable),
        ):
            try:
                main()
            except SystemExit:
                pass

    data = json.loads(deliverable.read_text())
    for field in [
        "spilled_energy_auroc",
        "thinkprm_auroc",
        "driftprobe_auroc",
        "honest_verdict",
        "n_responses",
        "model_used",
    ]:
        assert field in data, f"required field '{field}' missing from blocked artifact"
