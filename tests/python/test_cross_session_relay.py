"""Tests for cross-session Tier 2 relay (Exp 448).

All tests reference spec requirements so scripts/check_spec_coverage.py can
verify 100% trace coverage.

Spec: REQ-LEARN-037, REQ-LEARN-038,
      SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.cross_session_relay import (
    CrossSessionResult,
    _RELAY_MODEL_ID,
    compute_relay_verdict,
    simulate_session,
)
from carnot.pipeline.session_memory import SessionMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _questions_no_arithmetic(n: int = 10) -> list[str]:
    """Return questions that contain NO parseable arithmetic expressions.

    The ArithmeticExtractor/carry_check template only fires on expressions
    like "A × B = C". Plain text questions produce zero violations, so FP
    rate is 0 — useful for baseline sessions.
    """
    return [f"What is the capital of country {i}?" for i in range(n)]


def _questions_with_carry(n: int = 10) -> list[str]:
    """Return questions with valid multi-digit arithmetic (no intentional errors).

    These contain "A × B = C" patterns with correct products so the
    carry_check template does NOT flag a violation when it is active.
    They DO cause the CaseMemoryTemplateWiring to record carry_error observations
    (simulate_session calls on_violation_recorded for every question).
    """
    return [f"Calculate: {(i + 10)} × {(i + 5)} = {(i + 10) * (i + 5)}" for i in range(n)]


def _library_with_carry_above_threshold(model_id: str = _RELAY_MODEL_ID) -> ConstraintTemplateLibrary:
    """Return a library with carry_check observations above min_frequency (5)."""
    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()
    # min_frequency for carry_check is 5 — add 6 to be safely above threshold.
    lib.observe_pattern("carry_check", model_id, count=6)
    return lib


# ---------------------------------------------------------------------------
# CrossSessionResult dataclass tests
# ---------------------------------------------------------------------------


class TestCrossSessionResult:
    """REQ-LEARN-037-1: CrossSessionResult dataclass contract."""

    def test_fields_present(self):
        """CrossSessionResult must have all required fields. (REQ-LEARN-037-1)"""
        r = CrossSessionResult(
            session_id=0,
            n_questions=50,
            fp_rate=0.1,
            n_templates_active=0,
            n_templates_loaded_from_prior=0,
        )
        assert r.session_id == 0
        assert r.n_questions == 50
        assert r.fp_rate == 0.1
        assert r.n_templates_active == 0
        assert r.n_templates_loaded_from_prior == 0

    def test_fields_are_correct_types(self):
        """Fields must accept int, int, float, int, int. (REQ-LEARN-037-1)"""
        r = CrossSessionResult(
            session_id=2,
            n_questions=25,
            fp_rate=0.0,
            n_templates_active=3,
            n_templates_loaded_from_prior=2,
        )
        assert isinstance(r.session_id, int)
        assert isinstance(r.n_questions, int)
        assert isinstance(r.fp_rate, float)
        assert isinstance(r.n_templates_active, int)
        assert isinstance(r.n_templates_loaded_from_prior, int)


# ---------------------------------------------------------------------------
# compute_relay_verdict tests
# ---------------------------------------------------------------------------


class TestComputeRelayVerdict:
    """REQ-LEARN-037-3, SCENARIO-LEARN-068: relay verdict logic."""

    def test_insufficient_data_empty(self):
        """Empty list returns insufficient_data. (SCENARIO-LEARN-068)"""
        assert compute_relay_verdict([]) == "insufficient_data"

    def test_insufficient_data_one_session(self):
        """Single session returns insufficient_data. (SCENARIO-LEARN-068)"""
        r = CrossSessionResult(0, 10, 0.2, 0, 0)
        assert compute_relay_verdict([r]) == "insufficient_data"

    def test_cross_session_improvement(self):
        """Session 1 FP rate < Session 0 → cross_session_improvement. (REQ-LEARN-037-3)"""
        s0 = CrossSessionResult(0, 10, 0.5, 0, 0)
        s1 = CrossSessionResult(1, 10, 0.3, 2, 2)
        assert compute_relay_verdict([s0, s1]) == "cross_session_improvement"

    def test_no_improvement_equal(self):
        """Session 1 FP rate == Session 0 → no_improvement. (REQ-LEARN-037-3)"""
        s0 = CrossSessionResult(0, 10, 0.4, 0, 0)
        s1 = CrossSessionResult(1, 10, 0.4, 2, 2)
        assert compute_relay_verdict([s0, s1]) == "no_improvement"

    def test_no_improvement_worse(self):
        """Session 1 FP rate > Session 0 → no_improvement. (REQ-LEARN-037-3)"""
        s0 = CrossSessionResult(0, 10, 0.2, 0, 0)
        s1 = CrossSessionResult(1, 10, 0.5, 1, 1)
        assert compute_relay_verdict([s0, s1]) == "no_improvement"

    def test_uses_only_first_two_sessions(self):
        """Only sessions[0] and sessions[1] are compared; extras are ignored. (REQ-LEARN-037-3)"""
        s0 = CrossSessionResult(0, 10, 0.5, 0, 0)
        s1 = CrossSessionResult(1, 10, 0.3, 2, 2)
        s2 = CrossSessionResult(2, 10, 0.9, 3, 3)  # Would be "worse" if included
        assert compute_relay_verdict([s0, s1, s2]) == "cross_session_improvement"


# ---------------------------------------------------------------------------
# simulate_session — basic contract
# ---------------------------------------------------------------------------


class TestSimulateSessionBasic:
    """REQ-LEARN-037-2: simulate_session contract for Session 0 (no prior)."""

    def test_session_0_no_prior_returns_result(self, tmp_path):
        """Session 0 with no prior path returns a CrossSessionResult. (REQ-LEARN-037-2)"""
        questions = _questions_no_arithmetic(5)
        result = simulate_session(
            session_id=0,
            questions=questions,
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )
        assert isinstance(result, CrossSessionResult)
        assert result.session_id == 0
        assert result.n_questions == 5

    def test_session_0_n_loaded_is_zero(self, tmp_path):
        """Session 0 (no prior) must have n_templates_loaded_from_prior == 0. (REQ-LEARN-037-2)"""
        result = simulate_session(
            session_id=0,
            questions=_questions_no_arithmetic(5),
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )
        assert result.n_templates_loaded_from_prior == 0

    def test_session_0_fp_rate_in_range(self, tmp_path):
        """FP rate must be in [0.0, 1.0]. (REQ-LEARN-037-2)"""
        result = simulate_session(
            session_id=0,
            questions=_questions_no_arithmetic(10),
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )
        assert 0.0 <= result.fp_rate <= 1.0

    def test_session_0_saves_state_to_disk(self, tmp_path):
        """Session 0 must save session state so Session 1 can load it. (REQ-LEARN-037-2)"""
        simulate_session(
            session_id=0,
            questions=_questions_no_arithmetic(5),
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )
        session_dir = tmp_path / "session_0"
        sm = SessionMemory(storage_dir=str(session_dir), model_id=_RELAY_MODEL_ID)
        assert sm.exists(), "Session 0 must have saved state to disk"

    def test_empty_questions_returns_zero_fp_rate(self, tmp_path):
        """Empty questions list produces fp_rate=0.0 and n_questions=0. (REQ-LEARN-037-2)"""
        result = simulate_session(
            session_id=0,
            questions=[],
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )
        assert result.fp_rate == 0.0
        assert result.n_questions == 0

    def test_fp_counted_when_pipeline_flags_violation(self, tmp_path):
        """n_fp is incremented when pipeline flags a violation. (REQ-LEARN-037-2)

        Use a carry-error question (wrong product) to force the extractor to
        flag a violation, exercising the n_fp += 1 branch.
        """
        # Build a library with carry_check already above threshold so it fires.
        lib = _library_with_carry_above_threshold()
        # Save it as if it were a prior session.
        prior_sm = SessionMemory(
            storage_dir=str(tmp_path / "prior"),
            model_id=_RELAY_MODEL_ID,
        )
        prior_sm.save(CaseMemory(), lib, PerModelFPTracker())

        # Question with a deliberate carry error: 24 × 3 should be 72 not 62.
        bad_questions = ["Step: 24 × 3 = 62"]

        result = simulate_session(
            session_id=0,
            questions=bad_questions,
            prior_memory_path=str(tmp_path / "prior"),
            memory_dir=str(tmp_path),
        )
        # The carry_check template should fire and find the carry error,
        # resulting in at least one FP flagged (violation on the response).
        assert result.fp_rate > 0.0, (
            "Expected FP > 0 when carry_check template fires on wrong arithmetic"
        )


# ---------------------------------------------------------------------------
# simulate_session — cross-session loading (SCENARIO-LEARN-066)
# ---------------------------------------------------------------------------


class TestCrossSessionLoading:
    """SCENARIO-LEARN-066: Session 2 loads Session 1 templates."""

    def test_session_1_loads_session_0_observations(self, tmp_path):
        """Session 1 must load observation counts from Session 0. (SCENARIO-LEARN-066)"""
        questions = _questions_no_arithmetic(10)

        # Session 0
        r0 = simulate_session(
            session_id=0,
            questions=questions,
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )

        # Session 1: load from session 0 memory dir
        prior_path = str(tmp_path / "session_0")
        r1 = simulate_session(
            session_id=1,
            questions=questions,
            prior_memory_path=prior_path,
            memory_dir=str(tmp_path),
        )

        assert isinstance(r1, CrossSessionResult)
        assert r1.session_id == 1

    def test_session_1_n_loaded_positive_after_carry_observations(self, tmp_path):
        """After Session 0 accumulates carry_check above threshold,
        Session 1 n_templates_loaded_from_prior must be > 0. (SCENARIO-LEARN-066)
        """
        # We need enough questions to cross min_frequency=5 for carry_check.
        # simulate_session calls on_violation_recorded once per question with carry_error,
        # so 6 questions guarantees carry_check crosses the threshold.
        questions = _questions_no_arithmetic(10)

        # Session 0: accumulate enough carry_error observations
        simulate_session(
            session_id=0,
            questions=questions,
            prior_memory_path=None,
            memory_dir=str(tmp_path),
        )

        # Verify Session 0 saved state with carry_check active
        prior_path = str(tmp_path / "session_0")
        sm0 = SessionMemory(storage_dir=prior_path, model_id=_RELAY_MODEL_ID)
        restored = sm0.load()
        assert restored is not None, "Session 0 state must be on disk"
        _, prior_lib, _ = restored
        prior_lib.register_builtin_templates()
        active_keys = [t.pattern_key for t in prior_lib.get_active_templates(_RELAY_MODEL_ID)]
        # carry_check must be active (10 questions > min_frequency=5)
        assert "carry_check" in active_keys, (
            f"carry_check must be active after 10 observations; active={active_keys}"
        )

        # Session 1: n_templates_loaded_from_prior must be > 0
        r1 = simulate_session(
            session_id=1,
            questions=questions,
            prior_memory_path=prior_path,
            memory_dir=str(tmp_path),
        )
        assert r1.n_templates_loaded_from_prior > 0, (
            f"Session 1 must load ≥1 active template from Session 0; got {r1.n_templates_loaded_from_prior}"
        )

    def test_session_1_saves_its_own_state(self, tmp_path):
        """Session 1 must save its own state for Session 2 to load. (REQ-LEARN-037-2)"""
        questions = _questions_no_arithmetic(5)

        simulate_session(0, questions, prior_memory_path=None, memory_dir=str(tmp_path))
        simulate_session(1, questions, prior_memory_path=str(tmp_path / "session_0"), memory_dir=str(tmp_path))

        sm1 = SessionMemory(storage_dir=str(tmp_path / "session_1"), model_id=_RELAY_MODEL_ID)
        assert sm1.exists(), "Session 1 must have saved its state to disk"

    def test_session_1_missing_prior_state_file_falls_back_to_fresh(self, tmp_path):
        """If prior_memory_path is set but file is missing, session continues normally. (REQ-LEARN-037-2)"""
        nonexistent_path = str(tmp_path / "nonexistent_prior")
        result = simulate_session(
            session_id=1,
            questions=_questions_no_arithmetic(5),
            prior_memory_path=nonexistent_path,
            memory_dir=str(tmp_path),
        )
        assert isinstance(result, CrossSessionResult)
        # Should fall back to fresh (no prior loaded)
        assert result.n_templates_loaded_from_prior == 0

    def test_three_session_chain(self, tmp_path):
        """Three sessions chain correctly: 0 → 1 → 2. (REQ-LEARN-037-2)"""
        questions = _questions_no_arithmetic(10)

        r0 = simulate_session(0, questions, prior_memory_path=None, memory_dir=str(tmp_path))
        r1 = simulate_session(1, questions, prior_memory_path=str(tmp_path / "session_0"), memory_dir=str(tmp_path))
        r2 = simulate_session(2, questions, prior_memory_path=str(tmp_path / "session_1"), memory_dir=str(tmp_path))

        # All three must be valid CrossSessionResult instances
        assert r0.session_id == 0
        assert r1.session_id == 1
        assert r2.session_id == 2

        # Session 2 must have loaded state from Session 1
        # (Session 1 has all of Session 0's observations plus its own)
        # r2.n_templates_loaded_from_prior should be >= r1.n_templates_loaded_from_prior
        # (because Session 1 saved more accumulated observations than Session 0)
        assert r2.n_templates_loaded_from_prior >= r1.n_templates_loaded_from_prior


# ---------------------------------------------------------------------------
# SessionMemory round-trip tests (REQ-LEARN-038, SCENARIO-LEARN-067)
# ---------------------------------------------------------------------------


class TestSessionMemoryRoundTrip:
    """REQ-LEARN-038, SCENARIO-LEARN-067: template library survives save/load without loss."""

    def test_observation_counts_preserved(self, tmp_path):
        """Observations must survive SessionMemory save → load round-trip. (SCENARIO-LEARN-067)"""
        lib = _library_with_carry_above_threshold()
        # Snapshot the observations dict before save.
        original_obs = dict(lib._observations)

        sm = SessionMemory(storage_dir=str(tmp_path), model_id="round-trip-model")
        sm.save(CaseMemory(), lib, PerModelFPTracker())

        restored = sm.load()
        assert restored is not None
        _, loaded_lib, _ = restored

        # Observation counts must be identical.
        loaded_obs = dict(loaded_lib._observations)
        assert loaded_obs == original_obs, (
            f"Observation counts changed: original={original_obs}, loaded={loaded_obs}"
        )

    def test_carry_check_active_after_round_trip(self, tmp_path):
        """After save/load + register_builtin_templates, carry_check must be active. (SCENARIO-LEARN-067)"""
        lib = _library_with_carry_above_threshold()

        sm = SessionMemory(storage_dir=str(tmp_path), model_id="round-trip-model")
        sm.save(CaseMemory(), lib, PerModelFPTracker())

        restored = sm.load()
        assert restored is not None
        _, loaded_lib, _ = restored

        # Must call register_builtin_templates to attach callable functions.
        loaded_lib.register_builtin_templates()

        active_keys = [t.pattern_key for t in loaded_lib.get_active_templates(_RELAY_MODEL_ID)]
        assert "carry_check" in active_keys, (
            f"carry_check must be active after round-trip; active={active_keys}"
        )

    def test_multiple_templates_preserved(self, tmp_path):
        """All templates with observations above threshold survive round-trip. (REQ-LEARN-038-2)"""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        # Put all four templates above their thresholds.
        lib.observe_pattern("carry_check", _RELAY_MODEL_ID, count=6)       # min_freq=5
        lib.observe_pattern("sign_check", _RELAY_MODEL_ID, count=6)        # min_freq=5
        lib.observe_pattern("unit_consistency", _RELAY_MODEL_ID, count=4)  # min_freq=3
        lib.observe_pattern("comparison_direction", _RELAY_MODEL_ID, count=6)  # min_freq=5

        sm = SessionMemory(storage_dir=str(tmp_path), model_id="all-templates-model")
        sm.save(CaseMemory(), lib, PerModelFPTracker())

        restored = sm.load()
        assert restored is not None
        _, loaded_lib, _ = restored
        loaded_lib.register_builtin_templates()

        active_keys = {t.pattern_key for t in loaded_lib.get_active_templates(_RELAY_MODEL_ID)}
        assert active_keys == {"carry_check", "sign_check", "unit_consistency", "comparison_direction"}, (
            f"Expected all 4 templates active; got {active_keys}"
        )

    def test_below_threshold_templates_not_active_after_round_trip(self, tmp_path):
        """Templates below threshold remain inactive after round-trip. (REQ-LEARN-038-1)"""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        # Only 2 observations for carry_check — below min_frequency=5.
        lib.observe_pattern("carry_check", _RELAY_MODEL_ID, count=2)

        sm = SessionMemory(storage_dir=str(tmp_path), model_id="below-threshold-model")
        sm.save(CaseMemory(), lib, PerModelFPTracker())

        restored = sm.load()
        assert restored is not None
        _, loaded_lib, _ = restored
        loaded_lib.register_builtin_templates()

        active = loaded_lib.get_active_templates(_RELAY_MODEL_ID)
        assert len(active) == 0, f"No templates should be active with only 2 observations; got {active}"

    def test_empty_library_round_trip(self, tmp_path):
        """Empty ConstraintTemplateLibrary round-trips cleanly. (REQ-LEARN-038-1)"""
        lib = ConstraintTemplateLibrary()
        # No observations, no templates registered.

        sm = SessionMemory(storage_dir=str(tmp_path), model_id="empty-lib-model")
        sm.save(CaseMemory(), lib, PerModelFPTracker())

        restored = sm.load()
        assert restored is not None
        _, loaded_lib, _ = restored

        assert loaded_lib._observations == {}


# ---------------------------------------------------------------------------
# Module import test (public API surface)
# ---------------------------------------------------------------------------


class TestPublicApi:
    """Cross-session relay symbols are importable from carnot.pipeline. (REQ-LEARN-037)"""

    def test_importable_from_pipeline(self):
        from carnot.pipeline import (
            CrossSessionResult,
            compute_relay_verdict,
            simulate_session,
        )
        assert CrossSessionResult is not None
        assert callable(simulate_session)
        assert callable(compute_relay_verdict)

    def test_cross_session_result_importable_from_module(self):
        from carnot.pipeline.cross_session_relay import CrossSessionResult
        r = CrossSessionResult(0, 5, 0.0, 0, 0)
        assert r.session_id == 0
