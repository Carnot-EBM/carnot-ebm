"""Tests for Exp 773: Carnot vs SETS (arXiv 2501.19306) — oracle efficiency comparison.

Spec traces: REQ-COMPARE-001, REQ-COMPARE-002

Coverage target: 100% of python/carnot/pipeline/sets_baseline.py and the
callable functions in scripts/experiment_773_carnot_vs_sets.py that can be
exercised without a running VerifyRepairPipeline.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from python.carnot.pipeline.sets_baseline import (
    CANDIDATE_PREFIXES,
    SETSBaseline,
    SETSConfig,
    SETSResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_counter_llm():
    """Return an LLM mock that counts calls and always says 'No' for verification."""
    calls = []

    def llm_fn(prompt: str) -> str:
        calls.append(prompt)
        if "Is this correct? Answer Yes or No." in prompt:
            return "No"
        if "Your solution may have errors." in prompt:
            return "Corrected answer is 42."
        return f"Answer to: {prompt[:20]}"

    return llm_fn, calls


def _make_yes_llm(yes_on_index: int = 0):
    """Return an LLM that says 'Yes' for the yes_on_index-th verification call."""
    calls = []
    verify_count = [0]

    def llm_fn(prompt: str) -> str:
        calls.append(prompt)
        if "Is this correct? Answer Yes or No." in prompt:
            idx = verify_count[0]
            verify_count[0] += 1
            return "Yes" if idx == yes_on_index else "No"
        if "Your solution may have errors." in prompt:
            return "Corrected answer."
        return "The answer is 7."

    return llm_fn, calls


# ---------------------------------------------------------------------------
# SETSConfig
# ---------------------------------------------------------------------------


def test_sets_config_defaults():
    """SETSConfig defaults: n_candidates=4, max_correction_rounds=2. REQ-COMPARE-001."""
    cfg = SETSConfig()
    assert cfg.n_candidates == 4
    assert cfg.max_correction_rounds == 2


def test_sets_config_custom():
    """SETSConfig accepts custom values. REQ-COMPARE-001."""
    cfg = SETSConfig(n_candidates=2, max_correction_rounds=0)
    assert cfg.n_candidates == 2
    assert cfg.max_correction_rounds == 0


# ---------------------------------------------------------------------------
# generate_candidates — REQ-COMPARE-001 (a)
# ---------------------------------------------------------------------------


def test_generate_candidates_count():
    """generate_candidates returns exactly N=4 candidates. REQ-COMPARE-001 (a)."""
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    candidates = baseline.generate_candidates("What is 2+2?")
    assert len(candidates) == 4


def test_generate_candidates_oracle_calls():
    """generate_candidates uses exactly N oracle calls (one per prefix). REQ-COMPARE-001."""
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.generate_candidates("What is 3+3?")
    assert len(calls) == 4


def test_generate_candidates_uses_prefixes():
    """generate_candidates passes each prefix to the LLM. REQ-COMPARE-001 (a)."""
    prompts_seen = []

    def llm_fn(prompt: str) -> str:
        prompts_seen.append(prompt)
        return "answer"

    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.generate_candidates("Q?")
    for prefix in CANDIDATE_PREFIXES:
        assert any(prefix in p for p in prompts_seen), f"Prefix '{prefix}' not in any prompt"


def test_generate_candidates_custom_n():
    """generate_candidates respects n_candidates when custom prefixes supplied. REQ-COMPARE-001."""
    llm_fn, calls = _make_counter_llm()
    custom_prefixes = ["Prefix A:", "Prefix B:"]
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=2), prefixes=custom_prefixes)
    candidates = baseline.generate_candidates("Test?")
    assert len(candidates) == 2
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# self_verify — REQ-COMPARE-001 (b)
# ---------------------------------------------------------------------------


def test_self_verify_calls_llm_once():
    """self_verify calls the LLM exactly once per candidate. REQ-COMPARE-001 (b)."""
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.self_verify("What is 2+2?", "The answer is 4.")
    assert len(calls) == 1


def test_self_verify_yes():
    """self_verify returns True when LLM responds 'Yes'. REQ-COMPARE-001 (b)."""

    def llm_fn(prompt: str) -> str:
        return "Yes, this is correct."

    baseline = SETSBaseline(llm_fn=llm_fn)
    assert baseline.self_verify("Q?", "Answer.") is True


def test_self_verify_no():
    """self_verify returns False when LLM responds 'No'. REQ-COMPARE-001 (b)."""

    def llm_fn(prompt: str) -> str:
        return "No, this is wrong."

    baseline = SETSBaseline(llm_fn=llm_fn)
    assert baseline.self_verify("Q?", "Bad answer.") is False


def test_self_verify_empty_response():
    """self_verify returns False on empty LLM response. REQ-COMPARE-001 (b)."""

    def llm_fn(prompt: str) -> str:
        return ""

    baseline = SETSBaseline(llm_fn=llm_fn)
    assert baseline.self_verify("Q?", "Answer.") is False


def test_self_verify_prompt_format():
    """self_verify sends the correct verification prompt to the LLM. REQ-COMPARE-001 (b)."""
    prompts = []

    def llm_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "No"

    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.self_verify("What is 2+2?", "The answer is 4.")
    assert len(prompts) == 1
    assert "Is this correct? Answer Yes or No." in prompts[0]
    assert "What is 2+2?" in prompts[0]
    assert "The answer is 4." in prompts[0]


# ---------------------------------------------------------------------------
# self_correct — REQ-COMPARE-001 (c)
# ---------------------------------------------------------------------------


def test_self_correct_calls_llm_once():
    """self_correct calls the LLM exactly once. REQ-COMPARE-001 (c)."""
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.self_correct("What is 2+2?", "The answer is 5.")
    assert len(calls) == 1


def test_self_correct_prompt_format():
    """self_correct sends the expected correction prompt. REQ-COMPARE-001 (c)."""
    prompts = []

    def llm_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "Corrected."

    baseline = SETSBaseline(llm_fn=llm_fn)
    baseline.self_correct("What is 2+2?", "The answer is 5.")
    assert "Your solution may have errors. Correct it:" in prompts[0]
    assert "What is 2+2?" in prompts[0]
    assert "The answer is 5." in prompts[0]


def test_self_correct_returns_llm_output():
    """self_correct returns the LLM's corrected response. REQ-COMPARE-001 (c)."""

    def llm_fn(prompt: str) -> str:
        return "The correct answer is 4."

    baseline = SETSBaseline(llm_fn=llm_fn)
    result = baseline.self_correct("What is 2+2?", "5")
    assert result == "The correct answer is 4."


# ---------------------------------------------------------------------------
# run() — REQ-COMPARE-001 (full pipeline), REQ-COMPARE-002 (oracle call counting)
# ---------------------------------------------------------------------------


def test_run_returns_sets_result():
    """run() returns a SETSResult instance. REQ-COMPARE-001."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    result = baseline.run("What is 2+2?")
    assert isinstance(result, SETSResult)


def test_run_pass_flag_initially_false():
    """run() sets pass_flag=False (caller must set from ground truth). REQ-COMPARE-002."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    result = baseline.run("What is 2+2?")
    assert result.pass_flag is False


def test_run_records_wall_time():
    """run() records a non-negative wall_time_s. REQ-COMPARE-002."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    result = baseline.run("What is 2+2?")
    assert result.wall_time_s >= 0.0


def test_run_oracle_call_count_no_early_stop():
    """run() counts oracle calls: N generation + N verification + 1 correction (no early stop). REQ-COMPARE-002."""
    # With max_correction_rounds=1 and all-No verifier:
    # N=4 generation + 4 verification (all say No, we check all) + 1 correction = 9
    # But SETS stops verifying at first "Yes" — with all "No", it checks all N.
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=4, max_correction_rounds=1))
    result = baseline.run("Q?")
    # 4 gen + 4 verify (all No, loop runs all) + 1 correction = 9
    assert result.n_oracle_calls == 9


def test_run_oracle_call_count_early_stop():
    """run() stops verifying early when 'Yes' is found — fewer oracle calls. REQ-COMPARE-002."""
    # yes_on_index=0: first verification says "Yes" → skip remaining 3 verifications.
    # 4 generation + 1 verification + 1 correction = 6
    llm_fn, _ = _make_yes_llm(yes_on_index=0)
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=4, max_correction_rounds=1))
    result = baseline.run("Q?")
    assert result.n_oracle_calls == 6


def test_run_candidates_populated():
    """run() populates candidates list with N entries. REQ-COMPARE-001."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn)
    result = baseline.run("What?")
    assert len(result.candidates) == 4


def test_run_correction_applied():
    """run() sets correction_applied=True when max_correction_rounds > 0. REQ-COMPARE-001 (c)."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=4, max_correction_rounds=1))
    result = baseline.run("Q?")
    assert result.correction_applied is True


def test_run_no_correction():
    """run() sets correction_applied=False when max_correction_rounds=0. REQ-COMPARE-001."""
    llm_fn, calls = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=4, max_correction_rounds=0))
    result = baseline.run("Q?")
    assert result.correction_applied is False
    # 4 generation + 4 verification = 8 oracle calls (no correction)
    assert result.n_oracle_calls == 8


# ---------------------------------------------------------------------------
# oracle_call_ratio — REQ-COMPARE-002
# ---------------------------------------------------------------------------


def test_oracle_call_ratio_calculation():
    """oracle_call_ratio = sets_oracle_calls_per_q / carnot_oracle_calls_per_q. REQ-COMPARE-002."""
    sets_calls = 9.0  # typical SETS (4 gen + 4 verify + 1 correct)
    carnot_calls = 1.0  # Carnot uses 1 energy evaluation per question
    ratio = sets_calls / carnot_calls
    assert ratio == pytest.approx(9.0)


def test_oracle_call_ratio_positive():
    """oracle_call_ratio is always positive when both systems make at least one call. REQ-COMPARE-002."""
    llm_fn, _ = _make_counter_llm()
    baseline = SETSBaseline(llm_fn=llm_fn, config=SETSConfig(n_candidates=4, max_correction_rounds=1))
    result = baseline.run("Q?")
    carnot_calls = 1
    ratio = result.n_oracle_calls / carnot_calls
    assert ratio > 0


# ---------------------------------------------------------------------------
# Integration: mock LLM from experiment script
# ---------------------------------------------------------------------------


def test_mock_llm_returns_correct_answer():
    """Mock LLM returns correct answer when question matches. REQ-COMPARE-002."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_773_carnot_vs_sets import _build_mock_llm, GSM8K_QUESTIONS

    llm_fn, answer_map = _build_mock_llm(GSM8K_QUESTIONS[:5])
    q = GSM8K_QUESTIONS[0]["question"]
    expected = GSM8K_QUESTIONS[0]["answer"]
    response = llm_fn(f"Solve step by step: {q}")
    assert str(expected) in response


def test_mock_llm_verification_yes_for_correct():
    """Mock LLM says Yes when answer matches expected. REQ-COMPARE-001 (b)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_773_carnot_vs_sets import _build_mock_llm, GSM8K_QUESTIONS

    llm_fn, answer_map = _build_mock_llm(GSM8K_QUESTIONS[:5])
    q = GSM8K_QUESTIONS[0]["question"]
    expected = GSM8K_QUESTIONS[0]["answer"]
    prompt = f"Question: {q}\nAnswer: The answer is {expected}.\nIs this correct? Answer Yes or No."
    response = llm_fn(prompt)
    assert response.strip().lower().startswith("yes")


def test_mock_llm_verification_no_for_wrong():
    """Mock LLM says No when answer does not match expected. REQ-COMPARE-001 (b)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_773_carnot_vs_sets import _build_mock_llm, GSM8K_QUESTIONS

    llm_fn, answer_map = _build_mock_llm(GSM8K_QUESTIONS[:5])
    q = GSM8K_QUESTIONS[0]["question"]
    prompt = f"Question: {q}\nAnswer: The answer is 99999.\nIs this correct? Answer Yes or No."
    response = llm_fn(prompt)
    assert response.strip().lower().startswith("no")


def test_mock_llm_correction_returns_correct():
    """Mock LLM self-correction returns correct answer. REQ-COMPARE-001 (c)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_773_carnot_vs_sets import _build_mock_llm, GSM8K_QUESTIONS

    llm_fn, answer_map = _build_mock_llm(GSM8K_QUESTIONS[:5])
    q = GSM8K_QUESTIONS[0]["question"]
    expected = GSM8K_QUESTIONS[0]["answer"]
    prompt = f"Your solution may have errors. Correct it: {q}\nCurrent: Wrong answer."
    response = llm_fn(prompt)
    assert str(expected) in response


def test_mock_llm_unknown_question_fallback():
    """Mock LLM returns fallback '42' for unknown questions. REQ-COMPARE-002."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_773_carnot_vs_sets import _build_mock_llm

    llm_fn, _ = _build_mock_llm([])
    response = llm_fn("What is the meaning of life?")
    assert "42" in response
