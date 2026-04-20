"""Tests for jepa_pure_loss.py — PURE min-form contrastive margin loss.

Spec: REQ-LEARN-061, REQ-LEARN-062,
      SCENARIO-LEARN-095, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.inference.jepa_pure_loss import (
    JEPAChainScore,
    PUREMinFormLoss,
    pairs_to_pure_chains,
)


# ---------------------------------------------------------------------------
# JEPAChainScore construction
# ---------------------------------------------------------------------------


def test_jepa_chain_score_fields() -> None:
    """JEPAChainScore stores all fields correctly."""
    chain = JEPAChainScore(
        chain_id="q1/model_a",
        step_scores=[0.2, 0.5, 0.3],
        min_score=0.2,
        is_correct=True,
    )
    assert chain.chain_id == "q1/model_a"
    assert chain.step_scores == [0.2, 0.5, 0.3]
    assert chain.min_score == 0.2
    assert chain.is_correct is True


def test_jepa_chain_score_incorrect() -> None:
    """JEPAChainScore accepts is_correct=False."""
    chain = JEPAChainScore(
        chain_id="q2/model_b",
        step_scores=[0.9],
        min_score=0.9,
        is_correct=False,
    )
    assert chain.is_correct is False


# ---------------------------------------------------------------------------
# PUREMinFormLoss.compute_loss — SCENARIO-LEARN-095
# ---------------------------------------------------------------------------


def test_compute_loss_positive_when_gap_below_margin() -> None:
    """Loss > 0 when incorrect.min_score - correct.min_score < margin (SCENARIO-LEARN-095)."""
    loss_fn = PUREMinFormLoss(margin=1.0)
    correct = [JEPAChainScore("q/c", [0.3], 0.3, True)]
    incorrect = [JEPAChainScore("q/w", [0.7], 0.7, False)]
    # gap = 0.7 - 0.3 = 0.4; loss = max(0, 1.0 - 0.4) = 0.6
    result = loss_fn.compute_loss(correct, incorrect)
    assert abs(result - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# PUREMinFormLoss.compute_loss — SCENARIO-LEARN-096
# ---------------------------------------------------------------------------


def test_compute_loss_zero_when_gap_exceeds_margin() -> None:
    """Loss = 0 when gap between incorrect and correct min-scores exceeds margin (SCENARIO-LEARN-096)."""
    loss_fn = PUREMinFormLoss(margin=1.0)
    correct = [JEPAChainScore("q/c", [0.0], 0.0, True)]
    incorrect = [JEPAChainScore("q/w", [1.5], 1.5, False)]
    # gap = 1.5 - 0.0 = 1.5; loss = max(0, 1.0 - 1.5) = 0.0
    result = loss_fn.compute_loss(correct, incorrect)
    assert result == 0.0


def test_compute_loss_exact_margin_boundary() -> None:
    """Loss = 0 when gap exactly equals margin."""
    loss_fn = PUREMinFormLoss(margin=1.0)
    correct = [JEPAChainScore("q/c", [0.0], 0.0, True)]
    incorrect = [JEPAChainScore("q/w", [1.0], 1.0, False)]
    result = loss_fn.compute_loss(correct, incorrect)
    assert result == 0.0


def test_compute_loss_cross_product_multiple_pairs() -> None:
    """Loss is mean over all cross-product pairs when multiple chains are passed."""
    loss_fn = PUREMinFormLoss(margin=1.0)
    # Two correct, two incorrect — 4 pairs total.
    correct = [
        JEPAChainScore("q/c1", [0.0], 0.0, True),
        JEPAChainScore("q/c2", [0.1], 0.1, True),
    ]
    incorrect = [
        JEPAChainScore("q/w1", [0.5], 0.5, False),
        JEPAChainScore("q/w2", [0.6], 0.6, False),
    ]
    # pair (c1, w1): max(0, 1-(0.5-0.0))=0.5
    # pair (c1, w2): max(0, 1-(0.6-0.0))=0.4
    # pair (c2, w1): max(0, 1-(0.5-0.1))=0.6
    # pair (c2, w2): max(0, 1-(0.6-0.1))=0.5
    # mean = (0.5+0.4+0.6+0.5)/4 = 2.0/4 = 0.5
    result = loss_fn.compute_loss(correct, incorrect)
    assert abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# PUREMinFormLoss.zero_if_empty — SCENARIO-LEARN-097
# ---------------------------------------------------------------------------


def test_zero_if_empty_returns_zero_for_empty_list() -> None:
    """zero_if_empty returns 0.0 for an empty list (SCENARIO-LEARN-097)."""
    loss_fn = PUREMinFormLoss()
    assert loss_fn.zero_if_empty([]) == 0.0


def test_zero_if_empty_returns_falsy_for_nonempty() -> None:
    """zero_if_empty returns 0.0 (falsy) for non-empty list, enabling or-chaining."""
    loss_fn = PUREMinFormLoss()
    result = loss_fn.zero_if_empty([(1, 2)])
    assert result == 0.0


def test_compute_loss_empty_correct_returns_zero() -> None:
    """compute_loss returns 0.0 when correct_chains is empty."""
    loss_fn = PUREMinFormLoss()
    incorrect = [JEPAChainScore("q/w", [0.8], 0.8, False)]
    assert loss_fn.compute_loss([], incorrect) == 0.0


def test_compute_loss_empty_incorrect_returns_zero() -> None:
    """compute_loss returns 0.0 when incorrect_chains is empty."""
    loss_fn = PUREMinFormLoss()
    correct = [JEPAChainScore("q/c", [0.2], 0.2, True)]
    assert loss_fn.compute_loss(correct, []) == 0.0


# ---------------------------------------------------------------------------
# REQ-LEARN-062: margin parameter is configurable
# ---------------------------------------------------------------------------


def test_margin_configurable_larger_margin_increases_loss() -> None:
    """Larger margin produces larger loss for the same pair (REQ-LEARN-062)."""
    correct = [JEPAChainScore("q/c", [0.3], 0.3, True)]
    incorrect = [JEPAChainScore("q/w", [0.7], 0.7, False)]
    loss_small = PUREMinFormLoss(margin=0.3).compute_loss(correct, incorrect)
    loss_large = PUREMinFormLoss(margin=1.5).compute_loss(correct, incorrect)
    assert loss_large > loss_small


# ---------------------------------------------------------------------------
# compute_chain_scores
# ---------------------------------------------------------------------------


def test_compute_chain_scores_calls_model_per_step() -> None:
    """compute_chain_scores calls model once per embedding and returns a scalar per step."""
    loss_fn = PUREMinFormLoss()
    call_count = [0]

    def mock_model(emb: jnp.ndarray) -> float:
        call_count[0] += 1
        return float(jnp.sum(emb))

    embeddings = [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4]), jnp.array([0.5, 0.6])]
    scores = loss_fn.compute_chain_scores(mock_model, embeddings)
    assert call_count[0] == 3
    assert len(scores) == 3
    assert abs(scores[0] - 0.3) < 1e-5
    assert abs(scores[1] - 0.7) < 1e-5
    assert abs(scores[2] - 1.1) < 1e-5


def test_compute_chain_scores_empty_returns_empty() -> None:
    """compute_chain_scores returns empty list for empty embedding list."""
    loss_fn = PUREMinFormLoss()
    scores = loss_fn.compute_chain_scores(lambda e: 0.0, [])
    assert scores == []


# ---------------------------------------------------------------------------
# pairs_to_pure_chains
# ---------------------------------------------------------------------------


class _MockEntry:
    """Minimal FOVERCorpusEntry-like object for testing pairs_to_pure_chains."""

    def __init__(
        self,
        question: str,
        model_id: str,
        is_correct: bool,
        cot_steps: list,
        response: str = "",
    ) -> None:
        self.question = question
        self.model_id = model_id
        self.is_correct = is_correct
        self.cot_steps = cot_steps
        self.response = response


def test_pairs_to_pure_chains_splits_by_correctness() -> None:
    """pairs_to_pure_chains returns correct and incorrect chains separately."""
    entries = [
        _MockEntry("q1", "m1", True, [{"step_text": "step A"}]),
        _MockEntry("q1", "m2", False, [{"step_text": "step B"}]),
        _MockEntry("q2", "m1", False, [{"step_text": "step C"}]),
    ]
    embed_fn = lambda text: jnp.array([0.5])
    correct, incorrect = pairs_to_pure_chains(entries, embed_fn)
    assert len(correct) == 1
    assert len(incorrect) == 2
    assert correct[0].is_correct is True
    assert all(not c.is_correct for c in incorrect)


def test_pairs_to_pure_chains_min_score_is_minimum_of_step_scores() -> None:
    """pairs_to_pure_chains sets min_score = min(step_scores) across all steps."""
    # Two steps: embed produces [0.8] and [0.2]; mean of each = 0.8 and 0.2
    step_values = [0.8, 0.2]
    call_idx = [0]

    def embed_fn(text: str) -> jnp.ndarray:
        val = step_values[call_idx[0] % len(step_values)]
        call_idx[0] += 1
        return jnp.array([val])

    entries = [
        _MockEntry(
            "q1",
            "m1",
            True,
            [{"step_text": "s1"}, {"step_text": "s2"}],
        )
    ]
    correct, _ = pairs_to_pure_chains(entries, embed_fn)
    assert len(correct) == 1
    assert correct[0].min_score == pytest.approx(0.2, abs=1e-5)


def test_pairs_to_pure_chains_falls_back_to_response_when_no_steps() -> None:
    """pairs_to_pure_chains uses entry.response as single step when cot_steps is empty."""
    entries = [
        _MockEntry("q1", "m1", False, [], response="some response text"),
    ]
    embed_fn = lambda text: jnp.array([1.0])
    _, incorrect = pairs_to_pure_chains(entries, embed_fn)
    assert len(incorrect) == 1
    assert len(incorrect[0].step_scores) == 1


def test_pairs_to_pure_chains_chain_id_format() -> None:
    """pairs_to_pure_chains builds chain_id as question_prefix/model_id."""
    entries = [
        _MockEntry("my question text", "model_x", True, [{"step_text": "s1"}]),
    ]
    embed_fn = lambda text: jnp.array([0.5])
    correct, _ = pairs_to_pure_chains(entries, embed_fn)
    assert "model_x" in correct[0].chain_id
    assert "my question text" in correct[0].chain_id
