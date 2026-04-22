"""Tests for PSVSelfPlayLoop and PSVIteration — 100% coverage of psv_selfplay.py.

What is covered:
  - PSVIteration dataclass: field presence and types.
  - PSVSelfPlayLoop.run_iteration: correct counts (n_correct, n_violations, fp_count),
    constraint memory updated, weight delta computed.
  - PSVSelfPlayLoop._update_from_pairs: correct/violation routing into JitRL memory.
  - PSVSelfPlayLoop._mean_threshold_delta: zero when no domains touched, nonzero
    when at least one domain record is added.
  - Edge cases: all correct, all violations, empty questions list.

Spec: REQ-LEARN-076, REQ-LEARN-077,
      SCENARIO-LEARN-078, SCENARIO-LEARN-079, SCENARIO-LEARN-080
"""

from __future__ import annotations

import pytest

from carnot.pipeline.jitrl_memory import JitRLConstraintMemory
from carnot.training.psv_selfplay import PSVIteration, PSVSelfPlayLoop, _PROXY_VIOLATION_ENERGY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_memory() -> JitRLConstraintMemory:
    """Fresh JitRLConstraintMemory with default parameters."""
    return JitRLConstraintMemory()


@pytest.fixture()
def loop(fresh_memory: JitRLConstraintMemory) -> PSVSelfPlayLoop:
    """PSVSelfPlayLoop with 10 iterations / 5 questions per iteration."""
    return PSVSelfPlayLoop(
        n_iterations=10,
        n_questions_per_iter=5,
        constraint_memory=fresh_memory,
    )


# ---------------------------------------------------------------------------
# PSVIteration dataclass
# ---------------------------------------------------------------------------


def test_psv_iteration_fields() -> None:
    """PSVIteration must have all required fields with correct types."""
    it = PSVIteration(
        iteration=0,
        n_questions=5,
        n_correct=3,
        n_violations=2,
        fp_count=2,
        constraint_weight_delta=0.01,
    )
    assert it.iteration == 0
    assert it.n_questions == 5
    assert it.n_correct == 3
    assert it.n_violations == 2
    assert it.fp_count == 2
    assert isinstance(it.constraint_weight_delta, float)


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-078: correct counts from mixed verify_fn
# ---------------------------------------------------------------------------


def test_run_iteration_mixed_labels(loop: PSVSelfPlayLoop) -> None:
    """SCENARIO-LEARN-078: 5 questions, verify_fn True for even indices, False for odd.

    Even indices: 0, 2, 4 -> 3 correct.
    Odd indices:  1, 3    -> 2 violations.
    """
    questions = [f"Q{i}" for i in range(5)]
    inference_fn = lambda q: f"R_{q}"  # noqa: E731
    verify_fn = lambda r: int(r.split("_Q")[1]) % 2 == 0  # noqa: E731

    result = loop.run_iteration(questions, inference_fn, verify_fn, iteration=0)

    assert result.iteration == 0
    assert result.n_questions == 5
    assert result.n_correct == 3
    assert result.n_violations == 2
    assert result.fp_count == 2  # fp_count == n_violations


def test_run_iteration_all_correct(loop: PSVSelfPlayLoop) -> None:
    """All verify_fn=True: n_violations=0, n_correct=5, fp_count=0."""
    questions = [f"Q{i}" for i in range(5)]
    result = loop.run_iteration(
        questions,
        inference_fn=lambda q: "correct_response",
        verify_fn=lambda r: True,
        iteration=1,
    )
    assert result.n_correct == 5
    assert result.n_violations == 0
    assert result.fp_count == 0


def test_run_iteration_all_violations(loop: PSVSelfPlayLoop) -> None:
    """All verify_fn=False: n_correct=0, n_violations=5, fp_count=5."""
    questions = [f"Q{i}" for i in range(5)]
    result = loop.run_iteration(
        questions,
        inference_fn=lambda q: "wrong_response",
        verify_fn=lambda r: False,
        iteration=2,
    )
    assert result.n_correct == 0
    assert result.n_violations == 5
    assert result.fp_count == 5


def test_run_iteration_empty_questions(loop: PSVSelfPlayLoop) -> None:
    """Empty question list: all counts are zero, delta is 0.0."""
    result = loop.run_iteration(
        [],
        inference_fn=lambda q: "response",
        verify_fn=lambda r: True,
        iteration=0,
    )
    assert result.n_questions == 0
    assert result.n_correct == 0
    assert result.n_violations == 0
    assert result.fp_count == 0
    assert result.constraint_weight_delta == 0.0


# ---------------------------------------------------------------------------
# Memory update routing
# ---------------------------------------------------------------------------


def test_update_from_pairs_violations_recorded(fresh_memory: JitRLConstraintMemory) -> None:
    """Violations call record(was_fp=False); correct pairs call record(was_fp=True)."""
    loop = PSVSelfPlayLoop(1, 4, fresh_memory)
    violations = [("Q1", "wrong1"), ("Q2", "wrong2")]
    correct = [("Q3", "right3")]
    loop._update_from_pairs(violations, correct)

    assert len(fresh_memory.history) == 3

    # Violation records: was_fp=False (real violations we want to catch)
    viol_records = [r for r in fresh_memory.history if not r.was_fp]
    assert len(viol_records) == 2

    # Correct records: was_fp=True (threshold should be raised to reduce over-sensitivity)
    correct_records = [r for r in fresh_memory.history if r.was_fp]
    assert len(correct_records) == 1


def test_update_from_pairs_uses_proxy_energy(fresh_memory: JitRLConstraintMemory) -> None:
    """All records use _PROXY_VIOLATION_ENERGY as the violation_energy value."""
    loop = PSVSelfPlayLoop(1, 2, fresh_memory)
    loop._update_from_pairs([("Q1", "R1")], [("Q2", "R2")])
    for rec in fresh_memory.history:
        assert rec.violation_energy == _PROXY_VIOLATION_ENERGY


def test_update_from_pairs_domain_is_psv_gsm8k(fresh_memory: JitRLConstraintMemory) -> None:
    """All records use domain='psv_gsm8k' so threshold adaptation is unified."""
    loop = PSVSelfPlayLoop(1, 2, fresh_memory)
    loop._update_from_pairs([("Q1", "R1")], [])
    assert fresh_memory.history[0].domain == "psv_gsm8k"


# ---------------------------------------------------------------------------
# Threshold delta computation
# ---------------------------------------------------------------------------


def test_mean_threshold_delta_no_domains(fresh_memory: JitRLConstraintMemory) -> None:
    """Empty before/after dicts -> delta == 0.0."""
    loop = PSVSelfPlayLoop(1, 1, fresh_memory)
    assert loop._mean_threshold_delta({}, {}) == 0.0


def test_mean_threshold_delta_nonzero_after_update(fresh_memory: JitRLConstraintMemory) -> None:
    """After recording a violation, the threshold for psv_gsm8k changes -> delta > 0."""
    loop = PSVSelfPlayLoop(1, 1, fresh_memory)
    before = dict(fresh_memory._thresholds)
    fresh_memory.record("psv_gsm8k", _PROXY_VIOLATION_ENERGY, was_fp=False)
    after = dict(fresh_memory._thresholds)
    delta = loop._mean_threshold_delta(before, after)
    assert delta > 0.0


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-079: FP rate trend slope
# ---------------------------------------------------------------------------


def test_fp_rate_decreasing_trend() -> None:
    """SCENARIO-LEARN-079: decreasing fp_rate list -> slope < 0."""
    from scripts.experiment_688_psv_selfplay import _linear_slope

    fp_rates = [0.5, 0.4, 0.3, 0.2, 0.1]
    slope = _linear_slope(fp_rates)
    assert slope < 0, f"Expected negative slope, got {slope}"


def test_fp_rate_flat_trend() -> None:
    """Flat fp_rate list -> slope == 0."""
    from scripts.experiment_688_psv_selfplay import _linear_slope

    fp_rates = [0.5, 0.5, 0.5, 0.5]
    slope = _linear_slope(fp_rates)
    assert slope == pytest.approx(0.0, abs=1e-10)


def test_fp_rate_single_value() -> None:
    """Single value -> slope is 0.0 (undefined)."""
    from scripts.experiment_688_psv_selfplay import _linear_slope

    assert _linear_slope([0.5]) == 0.0


def test_fp_rate_empty_list() -> None:
    """Empty list -> slope is 0.0."""
    from scripts.experiment_688_psv_selfplay import _linear_slope

    assert _linear_slope([]) == 0.0


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-080: Synthetic mode runs when gate is closed
# ---------------------------------------------------------------------------


def test_synthetic_mode_fns_construction(tmp_path: pytest.TempPathFactory) -> None:
    """_make_synthetic_fns builds callable inference_fn and verify_fn from pairs."""
    from scripts.experiment_688_psv_selfplay import _make_synthetic_fns

    pairs = [
        {"question": "What is 1+1?", "response": "2", "is_correct": True, "model": "Qwen/x"},
        {"question": "What is 2+2?", "response": "5", "is_correct": False, "model": "Qwen/x"},
    ]
    inf_fn, ver_fn, questions = _make_synthetic_fns(pairs)

    assert "What is 1+1?" in questions
    assert inf_fn("What is 1+1?") == "2"
    assert ver_fn("2") is True
    assert ver_fn("5") is False


def test_synthetic_fallback_produces_questions() -> None:
    """_make_synthetic_gsm8k_fallback returns 200 questions with working fns."""
    from scripts.experiment_688_psv_selfplay import (
        N_ITERATIONS,
        N_QUESTIONS_PER_ITER,
        _make_synthetic_gsm8k_fallback,
    )

    inf_fn, ver_fn, questions = _make_synthetic_gsm8k_fallback()
    assert len(questions) == N_ITERATIONS * N_QUESTIONS_PER_ITER
    # inference_fn must return a non-empty string for the first question
    response = inf_fn(questions[0])
    assert isinstance(response, str) and len(response) > 0
    # verify_fn must return a bool
    assert isinstance(ver_fn(response), bool)
