"""Tests for composite energy scorer.

**Researcher summary:**
    Verifies that CompositeEnergyScorer correctly combines logprob confidence
    with structural test failure counts to select the best candidate. Tests
    cover the key invariants from Experiment 14: zero-failure candidates beat
    failing ones, logprob breaks ties among equal-failure candidates, and
    weights are configurable.

**Detailed explanation for engineers:**
    These are pure unit tests — no mocks, no I/O, no model loading. The
    scorer is a simple arithmetic formula, so we test it with known inputs
    and verify exact outputs. The tests are structured around the three
    key selection scenarios identified in Experiment 14.

Spec coverage: REQ-INFER-013
"""

from __future__ import annotations

import pytest
from carnot.inference.composite_scorer import (
    CompositeEnergyConfig,
    CompositeEnergyScorer,
)

# --- score_candidate tests ---


def test_score_candidate_zero_failures_high_logprob() -> None:
    """Candidate with 0 failures and high logprob gets low (good) score.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-01
    """
    scorer = CompositeEnergyScorer()
    # mean_logprob=-0.5 (high confidence), 0 failures
    score = scorer.score_candidate("def f(): return 1", -0.5, 0)
    # score = -1.0 * (-0.5) + 1.0 * 10.0 * 0 = 0.5
    assert score == pytest.approx(0.5)


def test_score_candidate_with_failures() -> None:
    """Candidate with failures gets penalized heavily.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-01
    """
    scorer = CompositeEnergyScorer()
    # mean_logprob=-0.5, 2 failures
    score = scorer.score_candidate("def f(): return 1", -0.5, 2)
    # score = -1.0 * (-0.5) + 1.0 * 10.0 * 2 = 0.5 + 20.0 = 20.5
    assert score == pytest.approx(20.5)


def test_score_candidate_low_logprob_no_failures() -> None:
    """Low-confidence candidate with 0 failures still scores reasonably.

    Spec: REQ-INFER-013
    """
    scorer = CompositeEnergyScorer()
    # mean_logprob=-4.0 (low confidence), 0 failures
    score = scorer.score_candidate("def f(): pass", -4.0, 0)
    # score = -1.0 * (-4.0) + 0 = 4.0
    assert score == pytest.approx(4.0)


# --- select_best: zero failures beats failures ---


def test_zero_failures_beats_failures() -> None:
    """Candidate with 0 failures + high logprob beats candidate with failures.

    This is the core invariant from Experiment 14: structural correctness
    dominates logprob confidence when the default penalty (10.0) is used.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-01
    """
    scorer = CompositeEnergyScorer()
    candidates = [
        # Candidate 0: 2 failures but very high logprob
        ("def f(): return 0", -0.1, 2),
        # Candidate 1: 0 failures but lower logprob
        ("def f(): return x", -3.0, 0),
    ]
    best_idx, best_score = scorer.select_best(candidates)
    assert best_idx == 1
    # Candidate 1 score = 3.0, candidate 0 score = 0.1 + 20 = 20.1
    assert best_score == pytest.approx(3.0)


# --- select_best: logprob breaks ties among equal failures ---


def test_logprob_breaks_ties_equal_failures() -> None:
    """Among candidates with equal test failures, highest logprob wins.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-02
    """
    scorer = CompositeEnergyScorer()
    candidates = [
        # Both have 1 failure, but different logprobs
        ("def f(): return a", -2.0, 1),  # score = 2.0 + 10 = 12.0
        ("def f(): return b", -0.5, 1),  # score = 0.5 + 10 = 10.5
    ]
    best_idx, best_score = scorer.select_best(candidates)
    assert best_idx == 1  # higher logprob wins
    assert best_score == pytest.approx(10.5)


def test_logprob_breaks_ties_zero_failures() -> None:
    """Among all-passing candidates, highest logprob wins.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-02
    """
    scorer = CompositeEnergyScorer()
    candidates = [
        ("def f(): return 1", -3.0, 0),  # score = 3.0
        ("def f(): return 1", -0.2, 0),  # score = 0.2
        ("def f(): return 1", -1.5, 0),  # score = 1.5
    ]
    best_idx, best_score = scorer.select_best(candidates)
    assert best_idx == 1  # -0.2 is the highest (least negative) logprob
    assert best_score == pytest.approx(0.2)


# --- Configurable weights ---


def test_configurable_weights_logprob_dominant() -> None:
    """When logprob_weight is high and penalty is low, logprob dominates.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-03
    """
    config = CompositeEnergyConfig(
        logprob_weight=10.0,
        structural_weight=1.0,
        test_failure_penalty=0.1,
    )
    scorer = CompositeEnergyScorer(config)
    candidates = [
        # High logprob but 1 failure
        ("def f(): return a", -0.1, 1),  # score = -10*(-0.1) + 1*0.1*1 = 1.0 + 0.1 = 1.1
        # Low logprob but 0 failures
        ("def f(): return b", -5.0, 0),  # score = -10*(-5.0) + 0 = 50.0
    ]
    best_idx, _ = scorer.select_best(candidates)
    # With logprob-dominant weights, the high-logprob candidate wins
    # despite having 1 failure
    assert best_idx == 0


def test_configurable_weights_structural_dominant() -> None:
    """When test_failure_penalty is very high, failures always lose.

    Spec: REQ-INFER-013, SCENARIO-INFER-013-03
    """
    config = CompositeEnergyConfig(
        logprob_weight=0.01,
        structural_weight=1.0,
        test_failure_penalty=100.0,
    )
    scorer = CompositeEnergyScorer(config)
    candidates = [
        ("def f(): return a", -0.001, 1),  # score ≈ 0 + 100 = 100
        ("def f(): return b", -10.0, 0),  # score = 0.1 + 0 = 0.1
    ]
    best_idx, _ = scorer.select_best(candidates)
    assert best_idx == 1  # zero-failure candidate wins decisively


# --- Edge cases ---


def test_single_candidate() -> None:
    """Single candidate is always selected.

    Spec: REQ-INFER-013
    """
    scorer = CompositeEnergyScorer()
    candidates = [("def f(): return 1", -1.0, 0)]
    best_idx, best_score = scorer.select_best(candidates)
    assert best_idx == 0
    assert best_score == pytest.approx(1.0)


def test_empty_candidates_raises() -> None:
    """Empty candidates list raises ValueError.

    Spec: REQ-INFER-013
    """
    scorer = CompositeEnergyScorer()
    with pytest.raises(ValueError, match="candidates list must not be empty"):
        scorer.select_best([])


def test_default_config() -> None:
    """Default config has expected values.

    Spec: REQ-INFER-013
    """
    config = CompositeEnergyConfig()
    assert config.logprob_weight == 1.0
    assert config.structural_weight == 1.0
    assert config.test_failure_penalty == 10.0


def test_scorer_default_config() -> None:
    """Scorer creates default config when none provided.

    Spec: REQ-INFER-013
    """
    scorer = CompositeEnergyScorer()
    assert scorer.config.logprob_weight == 1.0
    assert scorer.config.structural_weight == 1.0
    assert scorer.config.test_failure_penalty == 10.0
