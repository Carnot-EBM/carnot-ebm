"""Tests for ppsebm_real_validator (Exp 485, RETRO-043).

Spec: REQ-SELFLEARN-019, REQ-SELFLEARN-020,
      SCENARIO-SELFLEARN-019, SCENARIO-SELFLEARN-020
"""

from __future__ import annotations

import pytest

from carnot.pipeline.ppsebm_real_validator import (
    InterleavedViolationSequence,
    PPSEBMRealValidationResult,
)


# ---------------------------------------------------------------------------
# InterleavedViolationSequence tests
# ---------------------------------------------------------------------------


class TestInterleavedViolationSequence:
    """REQ-SELFLEARN-019, REQ-SELFLEARN-020."""

    def _step(self, domain: str, text: str = "step") -> dict:
        return {"step_text": text, "domain": domain}

    # --- domain_sequence ---

    def test_domain_sequence_preserves_order(self):
        """SCENARIO-SELFLEARN-019: domain_sequence returns labels in natural order."""
        steps = [
            self._step("arithmetic"),
            self._step("code"),
            self._step("arithmetic"),
            self._step("logical"),
        ]
        seq = InterleavedViolationSequence(steps)
        assert seq.domain_sequence == ["arithmetic", "code", "arithmetic", "logical"]

    # --- interleaving_rate ---

    def test_interleaving_rate_all_alternating(self):
        """Alternating arithmetic/code gives interleaving_rate=1.0."""
        steps = [
            self._step("arithmetic"),
            self._step("code"),
            self._step("arithmetic"),
            self._step("code"),
        ]
        seq = InterleavedViolationSequence(steps)
        assert seq.interleaving_rate == 1.0

    def test_interleaving_rate_no_transitions(self):
        """All same domain gives interleaving_rate=0.0."""
        steps = [self._step("arithmetic")] * 5
        seq = InterleavedViolationSequence(steps)
        assert seq.interleaving_rate == 0.0

    def test_interleaving_rate_partial(self):
        """Mixed sequence computes correct fractional rate."""
        # 4 steps: A A B A — 1 transition at index (1,2) and 1 at (2,3) = 2 transitions / 3
        steps = [
            self._step("arithmetic"),
            self._step("arithmetic"),
            self._step("code"),
            self._step("arithmetic"),
        ]
        seq = InterleavedViolationSequence(steps)
        assert abs(seq.interleaving_rate - 2 / 3) < 1e-9

    def test_interleaving_rate_single_step(self):
        """Single step returns 0.0 (no pairs to compare)."""
        seq = InterleavedViolationSequence([self._step("arithmetic")])
        assert seq.interleaving_rate == 0.0

    # --- to_training_batches ---

    def test_to_training_batches_even_split(self):
        """4 steps, batch_size=2 -> 2 batches of 2."""
        steps = [self._step(d) for d in ["arithmetic", "code", "logical", "arithmetic"]]
        seq = InterleavedViolationSequence(steps)
        batches = seq.to_training_batches(batch_size=2)
        assert len(batches) == 2
        assert batches[0] == steps[:2]
        assert batches[1] == steps[2:]

    def test_to_training_batches_last_partial(self):
        """5 steps, batch_size=2 -> [2, 2, 1]."""
        steps = [self._step("arithmetic")] * 5
        seq = InterleavedViolationSequence(steps)
        batches = seq.to_training_batches(batch_size=2)
        assert len(batches) == 3
        assert len(batches[2]) == 1

    def test_to_training_batches_preserves_order(self):
        """Batches preserve natural step order, not sorted by domain."""
        steps = [self._step("code"), self._step("arithmetic"), self._step("logical")]
        seq = InterleavedViolationSequence(steps)
        batches = seq.to_training_batches(batch_size=10)
        # Single batch — all steps in original order
        assert batches[0] == steps

    def test_to_training_batches_empty_steps(self):
        """Empty steps list returns [[]] (one empty batch)."""
        seq = InterleavedViolationSequence.__new__(InterleavedViolationSequence)
        seq.steps = []
        batches = seq.to_training_batches(batch_size=5)
        assert batches == [[]]

    # --- validation: missing domain key ---

    def test_missing_domain_key_raises(self):
        """ValueError raised when any step lacks 'domain' key."""
        steps = [{"step_text": "no domain here"}]
        with pytest.raises(ValueError, match="missing 'domain' key"):
            InterleavedViolationSequence(steps)

    def test_missing_domain_key_reports_index(self):
        """ValueError message includes the failing step index."""
        steps = [self._step("arithmetic"), {"step_text": "bad"}]
        with pytest.raises(ValueError, match="step\\[1\\]"):
            InterleavedViolationSequence(steps)


# ---------------------------------------------------------------------------
# PPSEBMRealValidationResult tests
# ---------------------------------------------------------------------------


class TestPPSEBMRealValidationResult:
    """REQ-SELFLEARN-019, REQ-SELFLEARN-020, SCENARIO-SELFLEARN-019/020."""

    # --- isolation_maintained ---

    def test_isolation_maintained_true_when_score_0_75(self):
        """SCENARIO-SELFLEARN-019: isolation_maintained=True when score=0.75."""
        result = PPSEBMRealValidationResult(
            n_steps=50,
            interleaving_rate=0.6,
            isolation_score_before=1.0,
            isolation_score_after=0.75,
            fp_rate_real=0.05,
        )
        assert result.isolation_maintained is True

    def test_isolation_maintained_false_when_score_0_65(self):
        """isolation_maintained=False when score=0.65 (below 0.7 threshold)."""
        result = PPSEBMRealValidationResult(
            n_steps=50,
            interleaving_rate=0.6,
            isolation_score_before=0.9,
            isolation_score_after=0.65,
            fp_rate_real=0.1,
        )
        assert result.isolation_maintained is False

    def test_isolation_maintained_boundary_exact_0_7(self):
        """isolation_score_after == 0.7 is NOT maintained (threshold is >0.7, not >=)."""
        result = PPSEBMRealValidationResult(
            n_steps=10,
            interleaving_rate=0.5,
            isolation_score_before=0.9,
            isolation_score_after=0.7,
            fp_rate_real=0.0,
        )
        assert result.isolation_maintained is False

    # --- better_than_synthetic ---

    def test_better_than_synthetic_true_equal(self):
        """SCENARIO-SELFLEARN-020: >= synthetic baseline is sufficient."""
        result = PPSEBMRealValidationResult(
            n_steps=57,
            interleaving_rate=0.4,
            isolation_score_before=0.9,
            isolation_score_after=1.0,
            fp_rate_real=0.02,
            synthetic_isolation_score=1.0,
        )
        assert result.better_than_synthetic is True

    def test_better_than_synthetic_false_when_degraded(self):
        """better_than_synthetic=False when real score < synthetic baseline."""
        result = PPSEBMRealValidationResult(
            n_steps=57,
            interleaving_rate=0.5,
            isolation_score_before=0.9,
            isolation_score_after=0.85,
            fp_rate_real=0.05,
            synthetic_isolation_score=1.0,
        )
        assert result.better_than_synthetic is False

    def test_default_synthetic_baseline_is_1_0(self):
        """Default synthetic_isolation_score = 1.0 (Exp 470 result)."""
        result = PPSEBMRealValidationResult(
            n_steps=20,
            interleaving_rate=0.3,
            isolation_score_before=1.0,
            isolation_score_after=0.8,
            fp_rate_real=0.0,
        )
        # 0.8 < 1.0 => not better than synthetic
        assert result.better_than_synthetic is False

    def test_retro_043_scenario_success(self):
        """Full RETRO-043 success path: isolation_maintained AND better_than_synthetic."""
        result = PPSEBMRealValidationResult(
            n_steps=57,
            interleaving_rate=0.4,
            isolation_score_before=1.0,
            isolation_score_after=1.0,
            fp_rate_real=0.0,
            synthetic_isolation_score=1.0,
        )
        assert result.isolation_maintained is True
        assert result.better_than_synthetic is True
