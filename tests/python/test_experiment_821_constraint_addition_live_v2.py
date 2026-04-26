"""Tests for Exp 821: Constraint Addition Live v2.

Covers:
- Gate check reads Exp 819 and blocks when verdict != "injection_field_fixed"  (REQ-LEARN-821-001)
- Session loop accumulates constraints across sessions via mocked pipeline       (REQ-LEARN-821-002)
- delta_overall and delta_s1_to_s3 computed correctly from precision list        (REQ-LEARN-821-002)
- honest_verdict maps correctly to delta value                                   (REQ-LEARN-821-001/002)

Spec: REQ-LEARN-821-001, REQ-LEARN-821-002, SCENARIO-LEARN-821-001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_821_constraint_addition_live_v2 import (
    _check_exp819_gate,
    compute_deltas,
    map_honest_verdict,
    run_session,
    _text_to_spins,
    _violation_spins,
    _correct_spins,
    _extract_violation_constraint,
    GSM8K_TRIPLES,
    N_SPINS,
    EMB_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmpl() -> MagicMock:
    """Minimal ExperimentTemplate mock that returns a blocked artifact dict."""
    tmpl = MagicMock()
    tmpl.build_result.side_effect = lambda *args, **kwargs: {
        "status": "blocked",
        **kwargs,
    }
    return tmpl


# ---------------------------------------------------------------------------
# REQ-LEARN-821-001: Gate check reads Exp 819 and blocks on wrong verdict
# ---------------------------------------------------------------------------


class TestExp819Gate:
    """REQ-LEARN-821-001: Exp 819 gate must block if honest_verdict != 'injection_field_fixed'."""

    def test_gate_blocks_when_file_missing(self, tmp_path: Path) -> None:
        """Gate returns blocked artifact when Exp 819 result file does not exist.

        Spec: REQ-LEARN-821-001
        """
        missing = tmp_path / "no_such_file.json"
        tmpl = _make_tmpl()
        with patch("scripts.experiment_821_constraint_addition_live_v2.EXP_819_PATH", missing):
            result = _check_exp819_gate(tmpl)
        assert result is not None, "Should return blocked artifact when file is missing"
        assert result["honest_verdict"] == "blocked_gate"
        assert result["gate"] == "exp819_injection_not_fixed"
        assert result["status"] == "blocked"

    def test_gate_blocks_when_verdict_is_wrong(self, tmp_path: Path) -> None:
        """Gate returns blocked artifact when honest_verdict is not 'injection_field_fixed'.

        Spec: REQ-LEARN-821-001
        """
        exp819_file = tmp_path / "experiment_819.json"
        exp819_file.write_text(json.dumps({"honest_verdict": "injection_partial"}))
        tmpl = _make_tmpl()
        with patch(
            "scripts.experiment_821_constraint_addition_live_v2.EXP_819_PATH",
            exp819_file,
        ):
            result = _check_exp819_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_gate"
        assert "injection_partial" in result["blocked_reason"]

    def test_gate_passes_when_verdict_is_injection_field_fixed(self, tmp_path: Path) -> None:
        """Gate returns None (proceed) when honest_verdict == 'injection_field_fixed'.

        Spec: REQ-LEARN-821-001
        """
        exp819_file = tmp_path / "experiment_819.json"
        exp819_file.write_text(json.dumps({"honest_verdict": "injection_field_fixed"}))
        tmpl = _make_tmpl()
        with patch(
            "scripts.experiment_821_constraint_addition_live_v2.EXP_819_PATH",
            exp819_file,
        ):
            result = _check_exp819_gate(tmpl)
        assert result is None, "Gate should return None (proceed) when verdict is correct"


# ---------------------------------------------------------------------------
# REQ-LEARN-821-002: Delta computation correctness
# ---------------------------------------------------------------------------


class TestComputeDeltas:
    """REQ-LEARN-821-002: delta_overall and delta_s1_to_s3 computed correctly."""

    def test_delta_zero_when_all_equal(self) -> None:
        """Both deltas are 0.0 when precision is constant across sessions.

        Spec: REQ-LEARN-821-002
        """
        delta_s1_to_s3, delta_overall = compute_deltas([0.5, 0.5, 0.5])
        assert delta_s1_to_s3 == 0.0
        assert delta_overall == 0.0

    def test_delta_s1_to_s3_positive_when_improving(self) -> None:
        """delta_s1_to_s3 equals precision[2] - precision[0].

        Spec: REQ-LEARN-821-002
        """
        delta_s1_to_s3, delta_overall = compute_deltas([0.4, 0.5, 0.7])
        assert abs(delta_s1_to_s3 - 0.3) < 1e-6
        assert abs(delta_overall - 0.3) < 1e-6

    def test_delta_overall_uses_max_not_last(self) -> None:
        """delta_overall uses the maximum precision, not necessarily the last session.

        Spec: REQ-LEARN-821-002, SCENARIO-LEARN-821-001
        """
        # Session 2 peaks then drops in session 3.
        delta_s1_to_s3, delta_overall = compute_deltas([0.4, 0.8, 0.6])
        # delta_s1_to_s3 = 0.6 - 0.4 = 0.2
        assert abs(delta_s1_to_s3 - 0.2) < 1e-6
        # delta_overall = max(0.4, 0.8, 0.6) - 0.4 = 0.4
        assert abs(delta_overall - 0.4) < 1e-6

    def test_empty_precision_list_returns_zeros(self) -> None:
        """compute_deltas returns (0.0, 0.0) when fewer than 3 sessions.

        Spec: REQ-LEARN-821-002
        """
        d1, d2 = compute_deltas([])
        assert d1 == 0.0
        assert d2 == 0.0
        d1, d2 = compute_deltas([0.5, 0.6])
        assert d1 == 0.0
        assert d2 == 0.0


# ---------------------------------------------------------------------------
# REQ-LEARN-821-001 / REQ-LEARN-821-002: honest_verdict mapping
# ---------------------------------------------------------------------------


class TestHonestVerdict:
    """honest_verdict correctly reflects gate and delta outcome."""

    def test_blocked_gate_when_gate_blocked(self) -> None:
        """honest_verdict = 'blocked_gate' when gate_blocked=True.

        Spec: REQ-LEARN-821-001
        """
        assert map_honest_verdict(0.5, gate_blocked=True) == "blocked_gate"

    def test_works_live_when_delta_positive(self) -> None:
        """honest_verdict = 'constraint_addition_works_live' when delta_overall > 0.

        Spec: REQ-LEARN-821-002
        """
        assert map_honest_verdict(0.1) == "constraint_addition_works_live"

    def test_no_delta_when_delta_zero(self) -> None:
        """honest_verdict = 'constraint_addition_no_delta_live' when delta_overall == 0.

        Spec: REQ-LEARN-821-002
        """
        assert map_honest_verdict(0.0) == "constraint_addition_no_delta_live"

    def test_no_delta_when_delta_negative(self) -> None:
        """honest_verdict = 'constraint_addition_no_delta_live' when delta_overall < 0.

        Spec: REQ-LEARN-821-002
        """
        assert map_honest_verdict(-0.1) == "constraint_addition_no_delta_live"


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-821-001: Session loop accumulates constraints across sessions
# ---------------------------------------------------------------------------


class TestSessionLoop:
    """SCENARIO-LEARN-821-001: constraint store grows across sessions."""

    def test_store_accumulates_constraints(self) -> None:
        """Each session adds constraints to the store for future sessions.

        We use 2 simple questions and verify that:
          - store starts empty (session 0)
          - after each session, n_constraints_store increases when failures exist
        This is the core accumulation invariant: more failures → more constraints
        → larger external field → higher precision in subsequent sessions.

        Spec: REQ-LEARN-821-002, SCENARIO-LEARN-821-001
        """
        import numpy as np
        from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
        from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector

        store = EmbeddingConstraintStore()
        injector = IsingConstraintInjector(embedding_dim=EMB_DIM, n_spins=N_SPINS)

        # Start empty.
        assert len(store._store) == 0

        # Use first 5 questions for speed.
        questions = GSM8K_TRIPLES[:5]

        # Session 0: store is empty.
        sr0 = run_session(store, injector, questions, session=0)
        assert sr0["session"] == 1
        assert 0.0 <= sr0["precision"] <= 1.0
        # After session 0, store should have gained constraints for failures.
        n_after_s0 = len(store._store)
        assert n_after_s0 == sr0["n_constraints_added"]

        # Session 1: store may have constraints from session 0.
        sr1 = run_session(store, injector, questions, session=1)
        assert sr1["session"] == 2
        n_after_s1 = len(store._store)
        # Store must only grow (never shrink).
        assert n_after_s1 >= n_after_s0
        assert n_after_s1 == n_after_s0 + sr1["n_constraints_added"]

    def test_run_session_returns_required_fields(self) -> None:
        """run_session result contains all required artifact fields.

        Spec: REQ-LEARN-821-002
        """
        from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
        from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector

        store = EmbeddingConstraintStore()
        injector = IsingConstraintInjector(embedding_dim=EMB_DIM, n_spins=N_SPINS)
        sr = run_session(store, injector, GSM8K_TRIPLES[:3], session=0)

        assert "session" in sr
        assert "precision" in sr
        assert "n_constraints_store" in sr
        assert "n_constraints_added" in sr
        assert 0.0 <= sr["precision"] <= 1.0


# ---------------------------------------------------------------------------
# Utility function tests (for coverage)
# ---------------------------------------------------------------------------


class TestUtilities:
    """Tests for helper functions in Exp 821."""

    def test_text_to_spins_deterministic(self) -> None:
        """_text_to_spins returns same spins for same input text.

        Spec: REQ-LEARN-821-002
        """
        import numpy as np

        spins1 = _text_to_spins("hello world", N_SPINS)
        spins2 = _text_to_spins("hello world", N_SPINS)
        assert np.array_equal(spins1, spins2)
        assert set(spins1).issubset({-1.0, 1.0})
        assert len(spins1) == N_SPINS

    def test_violation_and_correct_spins_differ(self) -> None:
        """violation_spins and correct_spins are not identical.

        Spec: REQ-VERIFY-173
        """
        import numpy as np

        v = _violation_spins(N_SPINS)
        c = _correct_spins(N_SPINS)
        assert not np.array_equal(v, c)
        # Violation: first 4 are +1.
        assert all(v[:4] == 1.0)
        assert all(c == -1.0)

    def test_extract_violation_constraint_returns_spo(self) -> None:
        """_extract_violation_constraint returns a ConstraintSPOTuple with correct fields.

        Spec: REQ-LEARN-821-002
        """
        spo = _extract_violation_constraint("How much is 2+2?", "5", session=0)
        assert spo.predicate == "violates"
        assert spo.object == "arithmetic_precision"
        assert "session_0_arithmetic" in spo.source_violation_type

    def test_gsm8k_triples_count(self) -> None:
        """There are exactly 30 GSM8K triples.

        Spec: SCENARIO-LEARN-821-001
        """
        assert len(GSM8K_TRIPLES) == 30

    def test_each_triple_has_three_elements(self) -> None:
        """Every triple contains (question, correct_answer, wrong_answer).

        Spec: SCENARIO-LEARN-821-001
        """
        for triple in GSM8K_TRIPLES:
            assert len(triple) == 3
            q, correct, wrong = triple
            assert isinstance(q, str) and len(q) > 0
            assert isinstance(correct, str) and len(correct) > 0
            assert isinstance(wrong, str) and len(wrong) > 0
            assert correct != wrong, f"Correct and wrong answer must differ: {q}"
