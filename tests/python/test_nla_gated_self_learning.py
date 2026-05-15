"""Tests for NLA-gated FR-11 policy retention gate.

Spec: REQ-LEARN-2151, SCENARIO-LEARN-2151, SCENARIO-LEARN-2152
"""

from __future__ import annotations

import pytest

from carnot.pipeline.nla_gated_self_learning import (
    MAX_ITERATIONS,
    NLA_CONFIDENCE_THRESHOLD,
    NLAGatedLoopResult,
    NLAGatedSelfLearner,
    PolicyCandidate,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _candidate(
    policy_id: str,
    nla_confidence: float,
    is_correct: bool = True,
) -> PolicyCandidate:
    return PolicyCandidate(
        policy_id=policy_id,
        nla_confidence=nla_confidence,
        is_correct=is_correct,
    )


def _learner(**kwargs: float | int) -> NLAGatedSelfLearner:
    return NLAGatedSelfLearner(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# REQ-LEARN-2151-1: default threshold is 0.7
# ---------------------------------------------------------------------------


def test_default_threshold_is_0_7() -> None:
    """REQ-LEARN-2151-1: NLA_CONFIDENCE_THRESHOLD constant equals 0.7."""
    assert NLA_CONFIDENCE_THRESHOLD == 0.7
    learner = NLAGatedSelfLearner()
    assert learner.nla_threshold == 0.7


# ---------------------------------------------------------------------------
# REQ-LEARN-2151-2: single-candidate gating
# ---------------------------------------------------------------------------


def test_high_confidence_candidate_is_retained() -> None:
    """REQ-LEARN-2151-2: candidate with nla_confidence > 0.7 is retained."""
    learner = _learner()
    result = learner.run_iteration(0, [_candidate("p1", nla_confidence=0.8)])
    assert result.n_retained == 1
    assert result.n_rejected_by_nla == 0
    assert result.soundness_mistakes == 0


def test_low_confidence_candidate_is_rejected() -> None:
    """REQ-LEARN-2151-2: candidate with nla_confidence <= 0.7 is rejected."""
    learner = _learner()
    result = learner.run_iteration(0, [_candidate("p1", nla_confidence=0.7)])
    assert result.n_retained == 0
    assert result.n_rejected_by_nla == 1


def test_boundary_exactly_0_7_is_rejected() -> None:
    """REQ-LEARN-2151-2: nla_confidence == 0.7 is NOT above threshold — rejected."""
    learner = _learner()
    result = learner.run_iteration(0, [_candidate("p1", nla_confidence=0.7)])
    assert result.n_retained == 0
    assert result.n_rejected_by_nla == 1


def test_just_above_threshold_is_retained() -> None:
    """REQ-LEARN-2151-2: nla_confidence = 0.701 passes the strict > gate."""
    learner = _learner()
    result = learner.run_iteration(0, [_candidate("p1", nla_confidence=0.701)])
    assert result.n_retained == 1
    assert result.n_rejected_by_nla == 0


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-2151: mixed-confidence batch
# ---------------------------------------------------------------------------


def test_mixed_batch_retains_only_high_confidence() -> None:
    """SCENARIO-LEARN-2151: only candidates with confidence > 0.7 are retained."""
    learner = _learner()
    batch = [
        _candidate("p1", 0.9),   # retained
        _candidate("p2", 0.5),   # rejected
        _candidate("p3", 0.75),  # retained
        _candidate("p4", 0.3),   # rejected
        _candidate("p5", 0.71),  # retained
    ]
    result = learner.run_iteration(0, batch)
    assert result.n_candidates == 5
    assert result.n_retained == 3
    assert result.n_rejected_by_nla == 2
    assert result.soundness_mistakes == 0


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-2152: soundness_mistakes is always 0 for correct candidates
# ---------------------------------------------------------------------------


def test_soundness_mistakes_zero_for_all_correct() -> None:
    """SCENARIO-LEARN-2152: no soundness mistakes when all retained are correct."""
    learner = _learner()
    batch = [_candidate(f"p{i}", 0.9, is_correct=True) for i in range(5)]
    result = learner.run_iteration(0, batch)
    assert result.soundness_mistakes == 0


def test_soundness_mistakes_counted_for_incorrect_retained() -> None:
    """SCENARIO-LEARN-2152: incorrect retained candidate triggers soundness mistake."""
    learner = _learner()
    # High confidence but wrong — would be a soundness failure
    result = learner.run_iteration(
        0, [_candidate("p1", nla_confidence=0.9, is_correct=False)]
    )
    assert result.n_retained == 1
    assert result.soundness_mistakes == 1


def test_rejected_incorrect_candidate_does_not_count_as_soundness_mistake() -> None:
    """SCENARIO-LEARN-2152: incorrect candidate rejected by NLA gate is NOT a mistake."""
    learner = _learner()
    result = learner.run_iteration(
        0, [_candidate("p1", nla_confidence=0.5, is_correct=False)]
    )
    assert result.n_rejected_by_nla == 1
    assert result.soundness_mistakes == 0


# ---------------------------------------------------------------------------
# REQ-LEARN-2151-3: loop runs at most max_iterations
# ---------------------------------------------------------------------------


def test_loop_respects_max_iterations_limit() -> None:
    """REQ-LEARN-2151-3: run_loop caps execution at max_iterations."""
    learner = _learner(max_iterations=3)
    # 10 batches provided but only 3 should run
    batches = [[_candidate(f"p{j}", 0.8) for j in range(2)] for _ in range(10)]
    loop_result = learner.run_loop(batches)
    assert loop_result.iterations_run == 3
    assert len(loop_result.per_iteration) == 3


def test_default_max_iterations_is_10() -> None:
    """REQ-LEARN-2151-3: DEFAULT max_iterations == 10."""
    assert MAX_ITERATIONS == 10


def test_loop_with_fewer_batches_than_max() -> None:
    """REQ-LEARN-2151-3: loop stops when batches are exhausted before max."""
    learner = _learner(max_iterations=10)
    batches = [[_candidate("p0", 0.8)] for _ in range(4)]
    loop_result = learner.run_loop(batches)
    assert loop_result.iterations_run == 4


# ---------------------------------------------------------------------------
# REQ-LEARN-2151-4: aggregate totals are correct
# ---------------------------------------------------------------------------


def test_aggregate_totals_are_correct() -> None:
    """REQ-LEARN-2151-4: retained/rejected/soundness totals aggregate across iterations."""
    learner = _learner(max_iterations=10)
    batches = [
        [_candidate("p0", 0.9), _candidate("p1", 0.5)],  # 1 retained, 1 rejected
        [_candidate("p2", 0.4), _candidate("p3", 0.8)],  # 1 retained, 1 rejected
        [_candidate("p4", 0.8), _candidate("p5", 0.8)],  # 2 retained, 0 rejected
    ]
    loop_result = learner.run_loop(batches)
    assert loop_result.total_retained == 4
    assert loop_result.total_rejected == 2
    assert loop_result.soundness_mistakes == 0


def test_to_dict_contains_required_fields() -> None:
    """REQ-LEARN-2151: NLAGatedLoopResult.to_dict() includes all schema fields."""
    loop_result = NLAGatedLoopResult(
        iterations_run=2,
        total_retained=3,
        total_rejected=1,
        soundness_mistakes=0,
    )
    d = loop_result.to_dict()
    for key in ("iterations_run", "total_retained", "total_rejected", "soundness_mistakes"):
        assert key in d


# ---------------------------------------------------------------------------
# Full 10-iteration loop — the experiment scenario
# ---------------------------------------------------------------------------


def test_ten_iteration_loop_zero_soundness_mistakes() -> None:
    """REQ-LEARN-2151 end-to-end: 10-iteration loop with nla_confidence gate keeps soundness_mistakes=0."""
    learner = NLAGatedSelfLearner(nla_threshold=0.7, max_iterations=10)

    # Simulate 10 iterations; only candidates with confidence > 0.7 are correct
    # (in reality the NLA probe and verifier agree — retained ↔ correct).
    batches = [
        [
            _candidate(f"iter{i}_pass", nla_confidence=0.85, is_correct=True),
            _candidate(f"iter{i}_fail", nla_confidence=0.60, is_correct=False),
        ]
        for i in range(10)
    ]

    result = learner.run_loop(batches)
    assert result.iterations_run == 10
    assert result.soundness_mistakes == 0
    assert result.total_retained == 10   # one high-confidence candidate per iteration
    assert result.total_rejected == 10   # one low-confidence candidate per iteration
