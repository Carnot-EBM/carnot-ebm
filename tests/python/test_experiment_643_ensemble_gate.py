"""Tests for Experiment 643 — Ensemble Recall Gate v2.

Why these tests exist:
    compute_ensemble_hits() is the only new logic introduced by Exp 643.
    It OR-combines three detection signals; each branch must be exercised
    independently to guarantee the gate decision is correct.

Spec: REQ-VERIFY-141, REQ-VERIFY-142,
      SCENARIO-VERIFY-186, SCENARIO-VERIFY-187
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.ensemble_gate import compute_ensemble_hits


def _make_monitors(interwhen_returns: list[bool], causal_returns: list[bool]):
    """Build mock InterWhenMonitor and CausalReasoningVerifier.

    Using MagicMock so no real NLP or arithmetic logic runs — we test
    only the OR combination logic, not the detectors themselves.
    """
    interwhen = MagicMock()
    interwhen.any_violation.side_effect = interwhen_returns

    causal = MagicMock()
    causal.any_violation.side_effect = causal_returns

    return interwhen, causal


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-186: OR gate opens when any single signal fires
# ---------------------------------------------------------------------------


class TestEnsembleOrLogic:
    """REQ-VERIFY-141-1: ensemble_hit = interwhen OR hermes OR causal."""

    def test_causal_only_fires(self):
        """If only causal fires, ensemble_hit must be True (SCENARIO-VERIFY-186)."""
        interwhen, causal = _make_monitors([False], [True])
        hits = compute_ensemble_hits(["resp"], [0], interwhen, causal, set())
        assert hits == [True]

    def test_interwhen_only_fires(self):
        """If only interwhen fires, ensemble_hit must be True."""
        interwhen, causal = _make_monitors([True], [False])
        hits = compute_ensemble_hits(["resp"], [0], interwhen, causal, set())
        assert hits == [True]

    def test_hermes_only_fires(self):
        """If only hermes fires (question index in tp set), ensemble_hit must be True."""
        interwhen, causal = _make_monitors([False], [False])
        hits = compute_ensemble_hits(["resp"], [7], interwhen, causal, {7})
        assert hits == [True]

    def test_none_fires(self):
        """When no signal fires, ensemble_hit must be False."""
        interwhen, causal = _make_monitors([False], [False])
        hits = compute_ensemble_hits(["resp"], [0], interwhen, causal, set())
        assert hits == [False]

    def test_all_fire(self):
        """When all three signals fire, ensemble_hit must still be True (not >1)."""
        interwhen, causal = _make_monitors([True], [True])
        hits = compute_ensemble_hits(["resp"], [5], interwhen, causal, {5})
        assert hits == [True]

    def test_multiple_responses(self):
        """Correct per-response tracking across a batch (REQ-VERIFY-141-2)."""
        interwhen, causal = _make_monitors([False, True, False], [False, False, True])
        hits = compute_ensemble_hits(
            ["r0", "r1", "r2"], [0, 1, 2], interwhen, causal, set()
        )
        assert hits == [False, True, True]

    def test_hermes_index_not_in_set(self):
        """Hermes_hit defaults False when question index absent from tp set (REQ-VERIFY-141-4)."""
        interwhen, causal = _make_monitors([False], [False])
        hits = compute_ensemble_hits(["resp"], [99], interwhen, causal, {0, 1, 2})
        assert hits == [False]


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-187: Gate closed when recall below threshold
# ---------------------------------------------------------------------------


class TestGateThreshold:
    """REQ-VERIFY-142-1/2: gate_open = (ensemble_recall >= 0.30)."""

    def test_gate_open_at_threshold(self):
        """ensemble_recall exactly 0.30 on 25 questions means 7.5 -> need 8 TPs for 0.32;
        At exactly 0.30 (7.5 rounds to needing 7.5 -> 8 TPs out of 25? No: 0.30*25=7.5 -> 8 TPs)
        Actually 8/25 = 0.32 >= 0.30. Test with 8 TPs out of 25 responses."""
        n = 25
        # 8 TPs -> recall = 8/25 = 0.32 >= 0.30 -> gate open
        interwhen_hits = [True] * 8 + [False] * 17
        causal_hits = [False] * n
        interwhen, causal = _make_monitors(interwhen_hits, causal_hits)
        responses = [f"r{i}" for i in range(n)]
        indices = list(range(n))
        hits = compute_ensemble_hits(responses, indices, interwhen, causal, set())
        ensemble_recall = sum(hits) / n
        assert ensemble_recall >= 0.30

    def test_gate_closed_below_threshold(self):
        """ensemble_recall < 0.30 with only 3 TPs out of 25 (SCENARIO-VERIFY-187)."""
        n = 25
        interwhen_hits = [True] * 3 + [False] * 22
        causal_hits = [False] * n
        interwhen, causal = _make_monitors(interwhen_hits, causal_hits)
        responses = [f"r{i}" for i in range(n)]
        indices = list(range(n))
        hits = compute_ensemble_hits(responses, indices, interwhen, causal, set())
        ensemble_recall = sum(hits) / n
        assert ensemble_recall < 0.30
