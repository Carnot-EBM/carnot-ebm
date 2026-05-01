"""Tests for python/carnot/eval/phi_test.py — Φ (alpha_t) measurement.

Spec: REQ-PHI-001 (alpha_t measurement),
      REQ-PHI-002 (AND-composition bypass rate),
      REQ-PHI-003 (convergence gate).
"""

from __future__ import annotations

import pytest

from carnot.eval.phi_test import (
    AlphaTResult,
    AndCompositionResult,
    VerdictRecord,
    and_compose_verifiers,
    measure_alpha_t,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_records(
    ids: list[str], verdicts: list[str], scores: list[float] | None = None
) -> list[VerdictRecord]:
    """Build a list of VerdictRecord from parallel lists."""
    if scores is None:
        scores = [0.0] * len(ids)
    return [VerdictRecord(eid, v, s) for eid, v, s in zip(ids, verdicts, scores)]


# ---------------------------------------------------------------------------
# measure_alpha_t tests
# ---------------------------------------------------------------------------


class TestMeasureAlphaT:
    """REQ-PHI-001: alpha_t = 0 when always agree; alpha_t = 1 when always differ."""

    def test_always_agree_gives_alpha_t_zero(self):
        """When energy and temperature verdicts are identical, α_t must be 0.0.

        This is the degenerate case where Carnot adds NO new information —
        every selection it makes is the same as temperature baseline.  Zenil
        Theorem 4 says the loop will NOT converge in this case.
        """
        ids = ["q0_c0", "q0_c1", "q1_c0", "q1_c1"]
        verdicts = ["correct", "incorrect", "correct", "correct"]
        ev = _make_records(ids, verdicts)
        tv = _make_records(ids, verdicts)  # identical

        result = measure_alpha_t(ev, tv)

        assert result.alpha_t == 0.0
        assert result.n_disagreements == 0
        assert result.delta_example_ids == []
        assert result.n_total == 4

    def test_always_differ_gives_alpha_t_one(self):
        """When energy and temperature verdicts always disagree, α_t must be 1.0.

        This is the maximally independent case — Carnot always overrides the
        temperature signal.  α_t = 1.0 does NOT mean Carnot is correct; it
        just means it always differs from the baseline.
        """
        ids = ["q0_c0", "q0_c1", "q1_c0"]
        energy_verdicts = _make_records(ids, ["correct", "incorrect", "correct"])
        temp_verdicts = _make_records(ids, ["incorrect", "correct", "incorrect"])

        result = measure_alpha_t(energy_verdicts, temp_verdicts)

        assert result.alpha_t == 1.0
        assert result.n_disagreements == 3
        assert set(result.delta_example_ids) == set(ids)
        assert result.n_total == 3

    def test_partial_disagreement(self):
        """α_t is the exact fraction of disagreements — tested with a known value."""
        ids = [f"ex_{i}" for i in range(10)]
        energy_vs = ["correct"] * 10
        # Temperature disagrees on first 4 examples
        temp_vs = ["incorrect"] * 4 + ["correct"] * 6

        ev = _make_records(ids, energy_vs)
        tv = _make_records(ids, temp_vs)

        result = measure_alpha_t(ev, tv)

        assert result.alpha_t == pytest.approx(0.4)
        assert result.n_disagreements == 4
        assert result.delta_example_ids == ids[:4]

    def test_length_mismatch_raises(self):
        """Mismatched list lengths must raise ValueError."""
        ev = _make_records(["a", "b"], ["correct", "incorrect"])
        tv = _make_records(["a"], ["correct"])

        with pytest.raises(ValueError, match="same length"):
            measure_alpha_t(ev, tv)

    def test_id_mismatch_raises(self):
        """example_id mismatch at same position must raise ValueError."""
        ev = _make_records(["a", "b"], ["correct", "correct"])
        tv = _make_records(["a", "X"], ["correct", "correct"])

        with pytest.raises(ValueError, match="mismatch"):
            measure_alpha_t(ev, tv)

    def test_empty_lists_give_zero(self):
        """Empty input produces α_t = 0.0 (no information case)."""
        result = measure_alpha_t([], [])
        assert result.alpha_t == 0.0
        assert result.n_total == 0

    def test_return_type(self):
        """Return type must be AlphaTResult (NamedTuple contract)."""
        ev = _make_records(["x"], ["correct"])
        tv = _make_records(["x"], ["correct"])
        result = measure_alpha_t(ev, tv)
        assert isinstance(result, AlphaTResult)


# ---------------------------------------------------------------------------
# and_compose_verifiers tests
# ---------------------------------------------------------------------------


class TestAndComposeVerifiers:
    """REQ-PHI-002: AND-composition bypass_rate and pass/fail logic."""

    def test_all_agree_correct_passes(self):
        """When all k verifiers say 'correct', AND result is 'correct'."""
        ids = ["ex_0", "ex_1"]
        v1 = _make_records(ids, ["correct", "correct"])
        v2 = _make_records(ids, ["correct", "correct"])
        v3 = _make_records(ids, ["correct", "correct"])

        result = and_compose_verifiers([v1, v2, v3])

        assert result.n_passed == 2
        assert all(r.verdict == "correct" for r in result.and_verdicts)

    def test_one_disagrees_blocks_all(self):
        """One 'incorrect' among k verifiers makes the AND verdict 'incorrect'."""
        ids = ["ex_0"]
        v1 = _make_records(ids, ["correct"])
        v2 = _make_records(ids, ["incorrect"])  # one disagrees
        v3 = _make_records(ids, ["correct"])

        result = and_compose_verifiers([v1, v2, v3])

        assert result.n_passed == 0
        assert result.and_verdicts[0].verdict == "incorrect"

    def test_bypass_rate_zero_when_all_agree(self):
        """When all verifiers agree with the AND result, bypass_rate = 0.0."""
        ids = ["e0", "e1", "e2"]
        # All verifiers say "incorrect" -> AND is "incorrect" -> no bypass
        v_lists = [_make_records(ids, ["incorrect", "incorrect", "incorrect"]) for _ in range(3)]

        result = and_compose_verifiers(v_lists)

        assert result.bypass_rate == 0.0
        assert result.n_bypassed == 0

    def test_bypass_rate_positive_when_verifiers_differ(self):
        """Bypass_rate > 0 when AND produces a different outcome than some verifier."""
        ids = ["e0", "e1"]
        # v1 says correct for e0; v2 says incorrect for e0 -> AND says incorrect
        # v1 disagrees with AND on e0 -> bypass
        v1 = _make_records(ids, ["correct", "correct"])
        v2 = _make_records(ids, ["incorrect", "correct"])

        result = and_compose_verifiers([v1, v2])

        # e0: AND="incorrect", v1="correct" -> bypass; e1: AND="correct", both agree -> no bypass
        assert result.n_bypassed == 1
        assert result.bypass_rate == pytest.approx(0.5)

    def test_return_type(self):
        """Return type must be AndCompositionResult."""
        ids = ["x"]
        v = _make_records(ids, ["correct"])
        result = and_compose_verifiers([v])
        assert isinstance(result, AndCompositionResult)

    def test_empty_verifier_list_raises(self):
        """Providing zero verifier lists must raise ValueError."""
        with pytest.raises(ValueError):
            and_compose_verifiers([])

    def test_id_mismatch_across_verifiers_raises(self):
        """Mismatched example_ids across verifier lists must raise ValueError."""
        v1 = _make_records(["a", "b"], ["correct", "correct"])
        v2 = _make_records(["a", "X"], ["correct", "correct"])

        with pytest.raises(ValueError, match="does not match reference"):
            and_compose_verifiers([v1, v2])

    def test_length_mismatch_raises(self):
        """Verifier lists of different lengths must raise ValueError."""
        v1 = _make_records(["a", "b"], ["correct", "correct"])
        v2 = _make_records(["a"], ["correct"])

        with pytest.raises(ValueError, match="length"):
            and_compose_verifiers([v1, v2])
