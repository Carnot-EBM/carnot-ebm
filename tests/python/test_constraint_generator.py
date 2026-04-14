"""Tests for the ConstraintGenerator pipeline (CaseMemory → new constraint types).

Spec: REQ-LEARN-010, REQ-LEARN-011,
SCENARIO-LEARN-015, SCENARIO-LEARN-016, SCENARIO-LEARN-017, SCENARIO-LEARN-018.
"""

from __future__ import annotations

import importlib
from dataclasses import fields

import pytest


def load_module():
    """Import the constraint_generator module under test."""
    return importlib.import_module("carnot.pipeline.constraint_generator")


def load_case_module():
    return importlib.import_module("carnot.pipeline.case_memory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_memory():
    """Return a fresh empty CaseMemory."""
    mod = load_case_module()
    return mod.CaseMemory()


def _record(
    *,
    cm,
    violation_types: tuple[str, ...],
    repair_outcome: str,
    support: int = 1,
):
    """Record `support` identical CaseRecord instances into case_memory.

    `repair_outcome` is forced by choosing baseline_success / repair_success
    combinations that map to the desired outcome string.
    """
    mod = load_case_module()
    baseline, repaired = {
        "improved": (False, True),
        "regressed": (True, False),
        "unchanged_success": (True, True),
        "unchanged_failure": (False, False),
    }[repair_outcome]
    for i in range(support):
        rec = mod.CaseRecord.normalize(
            benchmark="test_bench",
            benchmark_slice="test_bench/arithmetic",
            model_name="test_model",
            case_id=f"case-{violation_types[0]}-{repair_outcome}-{i}",
            violation_types=violation_types,
            baseline_success=baseline,
            repair_success=repaired,
            confidence=0.9,
        )
        cm.record(rec)


# ---------------------------------------------------------------------------
# ConstraintPattern dataclass
# ---------------------------------------------------------------------------


class TestConstraintPattern:
    """REQ-LEARN-010: ConstraintPattern dataclass structure."""

    def test_fields_present(self):
        """ConstraintPattern must expose all required fields."""
        mod = load_module()
        field_names = {f.name for f in fields(mod.ConstraintPattern)}
        assert "pattern_type" in field_names
        assert "violation_family" in field_names
        assert "observed_precision" in field_names
        assert "support_count" in field_names
        assert "example_violations" in field_names
        assert "constraint_template" in field_names
        assert "source_memory_keys" in field_names

    def test_can_instantiate(self):
        """ConstraintPattern can be created with all required fields."""
        mod = load_module()
        pattern = mod.ConstraintPattern(
            pattern_type="carry_check",
            violation_family="carry_error",
            observed_precision=0.92,
            support_count=12,
            example_violations=["7 + 5 = 11"],
            constraint_template="verify carry propagation",
            source_memory_keys=["key_a", "key_b"],
        )
        assert pattern.pattern_type == "carry_check"
        assert pattern.violation_family == "carry_error"
        assert pattern.observed_precision == 0.92
        assert pattern.support_count == 12


# ---------------------------------------------------------------------------
# extract_patterns
# ---------------------------------------------------------------------------


class TestExtractPatterns:
    """REQ-LEARN-010, SCENARIO-LEARN-015: pattern extraction from CaseMemory."""

    def test_empty_memory_returns_empty(self):
        """Empty CaseMemory → empty pattern list (no crash)."""
        mod = load_module()
        cm = _make_case_memory()
        result = mod.extract_patterns(cm)
        assert result == []

    def test_below_min_support_excluded(self):
        """Families with total support < min_support are excluded."""
        mod = load_module()
        cm = _make_case_memory()
        # Only 2 entries for carry_error but min_support=3
        _record(cm=cm, violation_types=("carry_error:check",), repair_outcome="improved", support=2)
        result = mod.extract_patterns(cm, min_support=3)
        assert result == []

    def test_exactly_min_support_included(self):
        """Families with support == min_support are included."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:check",), repair_outcome="improved", support=3)
        result = mod.extract_patterns(cm, min_support=3)
        assert len(result) == 1
        assert result[0].violation_family == "carry_error"

    def test_precision_all_improved(self):
        """Precision = 1.0 when all cases are improved."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert pattern.observed_precision == pytest.approx(1.0)

    def test_precision_none_improved(self):
        """Precision = 0.0 when no cases are improved."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("sign_error:x",), repair_outcome="unchanged_failure", support=4)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert pattern.observed_precision == pytest.approx(0.0)

    def test_precision_mixed(self):
        """SCENARIO-LEARN-015: precision = improved / total (3 improved, 2 failed)."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=3)
        _record(cm=cm, violation_types=("carry_error:y",), repair_outcome="unchanged_failure", support=2)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert pattern.observed_precision == pytest.approx(3 / 5)
        assert pattern.support_count == 5

    def test_multiple_families_separated(self):
        """Entries with different violation families produce separate patterns."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=4)
        _record(cm=cm, violation_types=("sign_error:x",), repair_outcome="improved", support=4)
        result = mod.extract_patterns(cm, min_support=3)
        families = {p.violation_family for p in result}
        assert "carry_error" in families
        assert "sign_error" in families
        assert len(result) == 2

    def test_source_memory_keys_populated(self):
        """source_memory_keys are non-empty fingerprints from contributing entries."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=3)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert len(pattern.source_memory_keys) >= 1
        assert all(isinstance(k, str) for k in pattern.source_memory_keys)

    def test_example_violations_populated(self):
        """example_violations are non-empty strings from violation_types."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:cascade",), repair_outcome="improved", support=3)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert len(pattern.example_violations) >= 1

    def test_default_min_support_is_3(self):
        """extract_patterns uses min_support=3 by default."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=2)
        # 2 < 3 (default) → excluded
        assert mod.extract_patterns(cm) == []

    def test_unknown_family_uses_fallback_template(self):
        """Families not in _FAMILY_TEMPLATES get the generic fallback pattern_type."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("zzz_unknown_family:x",), repair_outcome="improved", support=3)
        [pattern] = mod.extract_patterns(cm, min_support=3)
        assert pattern.violation_family == "zzz_unknown_family"
        # Fallback pattern_type is used for unknown families
        assert "zzz_unknown_family" in pattern.constraint_template


# ---------------------------------------------------------------------------
# soundness_filter
# ---------------------------------------------------------------------------


class TestSoundnessFilter:
    """REQ-LEARN-011, SCENARIO-LEARN-016: soundness filtering."""

    def _make_pattern(self, mod, *, family: str, precision: float, support: int = 5):
        return mod.ConstraintPattern(
            pattern_type=f"type_{family}",
            violation_family=family,
            observed_precision=precision,
            support_count=support,
            example_violations=[],
            constraint_template=f"check {family}",
            source_memory_keys=[],
        )

    def test_high_precision_passes(self):
        """Patterns at or above threshold are returned."""
        mod = load_module()
        p = self._make_pattern(mod, family="carry_error", precision=0.90)
        result = mod.soundness_filter([p])
        assert result == [p]

    def test_exactly_at_threshold_passes(self):
        """Pattern with precision == min_precision (0.85) passes."""
        mod = load_module()
        p = self._make_pattern(mod, family="carry_error", precision=0.85)
        result = mod.soundness_filter([p])
        assert result == [p]

    def test_below_threshold_rejected(self):
        """SCENARIO-LEARN-016: patterns below threshold are NOT included in result."""
        mod = load_module()
        p = self._make_pattern(mod, family="sign_error", precision=0.60)
        result = mod.soundness_filter([p])
        assert result == []

    def test_mixed_keeps_only_sound(self):
        """Only high-precision patterns pass; low-precision ones are excluded."""
        mod = load_module()
        high = self._make_pattern(mod, family="carry_error", precision=0.92)
        low = self._make_pattern(mod, family="sign_error", precision=0.60)
        result = mod.soundness_filter([high, low])
        assert result == [high]

    def test_custom_min_precision(self):
        """Custom min_precision threshold is respected."""
        mod = load_module()
        p70 = self._make_pattern(mod, family="carry_error", precision=0.70)
        p60 = self._make_pattern(mod, family="sign_error", precision=0.60)
        result = mod.soundness_filter([p70, p60], min_precision=0.65)
        assert result == [p70]

    def test_empty_list_returns_empty(self):
        """soundness_filter on empty list returns empty list."""
        mod = load_module()
        assert mod.soundness_filter([]) == []

    def test_all_rejected_returns_empty(self):
        """soundness_filter returns [] when all patterns are below threshold."""
        mod = load_module()
        p = self._make_pattern(mod, family="magnitude_error", precision=0.10)
        result = mod.soundness_filter([p])
        assert result == []


# ---------------------------------------------------------------------------
# generate_arithmetic_constraint
# ---------------------------------------------------------------------------


class TestGenerateArithmeticConstraint:
    """REQ-LEARN-010, SCENARIO-LEARN-017: constraint generation per family."""

    def _make_pattern(self, mod, *, family: str, precision: float = 0.92):
        return mod.ConstraintPattern(
            pattern_type=f"type_{family}",
            violation_family=family,
            observed_precision=precision,
            support_count=5,
            example_violations=["example"],
            constraint_template=f"check {family}",
            source_memory_keys=["k1"],
        )

    def test_carry_error_family(self):
        """SCENARIO-LEARN-017: carry_error family → carry-check constraint."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="carry_error")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.constraint_id == "learned:carry_error"
        assert "carry" in constraint.description.lower()

    def test_sign_error_family(self):
        """SCENARIO-LEARN-017: sign_error family → sign-consistency constraint."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="sign_error")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.constraint_id == "learned:sign_error"
        assert "sign" in constraint.description.lower()

    def test_magnitude_error_family(self):
        """SCENARIO-LEARN-017: magnitude_error family → order-of-magnitude constraint."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="magnitude_error")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.constraint_id == "learned:magnitude_error"
        assert "magnitude" in constraint.description.lower()

    def test_unknown_family_fallback(self):
        """Unknown family produces a generic learned constraint (no crash)."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="unknown_xyz_family")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.constraint_id == "learned:unknown_xyz_family"
        assert isinstance(constraint.description, str)
        assert len(constraint.description) > 0

    def test_constraint_has_pattern_reference(self):
        """Generated LearnedConstraint retains a reference to its source pattern."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="carry_error")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.pattern is pattern

    def test_constraint_has_family(self):
        """Generated LearnedConstraint records the violation family."""
        mod = load_module()
        pattern = self._make_pattern(mod, family="sign_error")
        constraint = mod.generate_arithmetic_constraint(pattern)
        assert constraint.family == "sign_error"


# ---------------------------------------------------------------------------
# constraint_already_exists
# ---------------------------------------------------------------------------


class TestConstraintAlreadyExists:
    """REQ-LEARN-010: duplicate-prevention guard."""

    def test_no_attr_returns_false(self):
        """Extractor without _dynamic_constraints → constraint does not exist."""
        mod = load_module()

        class DummyExtractor:
            pass

        assert not mod.constraint_already_exists(DummyExtractor(), "learned:carry_error")

    def test_empty_list_returns_false(self):
        """Extractor with empty _dynamic_constraints → constraint does not exist."""
        mod = load_module()

        class DummyExtractor:
            _dynamic_constraints: list = []

        assert not mod.constraint_already_exists(DummyExtractor(), "learned:carry_error")

    def test_not_in_list_returns_false(self):
        """Extractor has constraints, but not the queried id → False."""
        mod = load_module()

        class FakeConstraint:
            def __init__(self, cid):
                self.constraint_id = cid

        class DummyExtractor:
            _dynamic_constraints = [FakeConstraint("learned:sign_error")]

        assert not mod.constraint_already_exists(DummyExtractor(), "learned:carry_error")

    def test_in_list_returns_true(self):
        """Constraint id is present in _dynamic_constraints → True."""
        mod = load_module()

        class FakeConstraint:
            def __init__(self, cid):
                self.constraint_id = cid

        class DummyExtractor:
            _dynamic_constraints = [FakeConstraint("learned:carry_error")]

        assert mod.constraint_already_exists(DummyExtractor(), "learned:carry_error")


# ---------------------------------------------------------------------------
# add_to_extractor
# ---------------------------------------------------------------------------


class TestAddToExtractor:
    """REQ-LEARN-010: additive constraint addition."""

    def _make_constraint(self, mod, *, family: str = "carry_error"):
        pattern = mod.ConstraintPattern(
            pattern_type="carry_check",
            violation_family=family,
            observed_precision=0.92,
            support_count=5,
            example_violations=[],
            constraint_template="check carry",
            source_memory_keys=[],
        )
        return mod.generate_arithmetic_constraint(pattern)

    def test_creates_list_when_absent(self):
        """add_to_extractor creates _dynamic_constraints when attr is absent."""
        mod = load_module()

        class DummyExtractor:
            pass

        ex = DummyExtractor()
        constraint = self._make_constraint(mod)
        mod.add_to_extractor(ex, constraint)
        assert hasattr(ex, "_dynamic_constraints")
        assert constraint in ex._dynamic_constraints

    def test_appends_to_existing_list(self):
        """add_to_extractor appends without removing prior constraints."""
        mod = load_module()

        class DummyExtractor:
            _dynamic_constraints: list

        ex = DummyExtractor()
        c1 = self._make_constraint(mod, family="carry_error")
        c2 = self._make_constraint(mod, family="sign_error")
        mod.add_to_extractor(ex, c1)
        mod.add_to_extractor(ex, c2)
        assert c1 in ex._dynamic_constraints
        assert c2 in ex._dynamic_constraints
        assert len(ex._dynamic_constraints) == 2

    def test_does_not_remove_existing(self):
        """Existing constraints are preserved — add_to_extractor is purely additive."""
        mod = load_module()

        class FakeConstraint:
            def __init__(self, cid):
                self.constraint_id = cid

        class DummyExtractor:
            pass

        ex = DummyExtractor()
        existing = FakeConstraint("pre_existing")
        ex._dynamic_constraints = [existing]

        new_c = self._make_constraint(mod, family="carry_error")
        mod.add_to_extractor(ex, new_c)
        assert existing in ex._dynamic_constraints
        assert new_c in ex._dynamic_constraints


# ---------------------------------------------------------------------------
# ConstraintGenerator.generate_from_memory
# ---------------------------------------------------------------------------


class TestConstraintGenerator:
    """REQ-LEARN-010, REQ-LEARN-011, SCENARIO-LEARN-018: full pipeline."""

    def test_empty_memory_returns_empty_list(self):
        """Empty CaseMemory → no constraints added, empty result."""
        mod = load_module()
        cm = _make_case_memory()

        class DummyExtractor:
            pass

        gen = mod.ConstraintGenerator()
        result = gen.generate_from_memory(cm, DummyExtractor())
        assert result == []

    def test_empty_memory_log_is_empty(self):
        """Empty CaseMemory → generation_log is empty dict."""
        mod = load_module()
        cm = _make_case_memory()

        class DummyExtractor:
            pass

        gen = mod.ConstraintGenerator()
        gen.generate_from_memory(cm, DummyExtractor())
        assert gen.generation_log == {}

    def test_low_precision_logged_as_rejected_soundness(self):
        """SCENARIO-LEARN-018: patterns below min_precision → 'rejected_soundness' in log."""
        mod = load_module()
        cm = _make_case_memory()
        # 1 improved out of 5 → precision=0.2, well below 0.85
        _record(cm=cm, violation_types=("sign_error:x",), repair_outcome="improved", support=1)
        _record(cm=cm, violation_types=("sign_error:y",), repair_outcome="unchanged_failure", support=4)

        class DummyExtractor:
            pass

        gen = mod.ConstraintGenerator()
        result = gen.generate_from_memory(cm, DummyExtractor())
        assert result == []
        # At least one entry with "rejected_soundness"
        assert any(v == "rejected_soundness" for v in gen.generation_log.values())

    def test_high_precision_constraint_added(self):
        """SCENARIO-LEARN-018: high-precision pattern → constraint added to extractor."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)

        class DummyExtractor:
            pass

        ex = DummyExtractor()
        gen = mod.ConstraintGenerator()
        result = gen.generate_from_memory(cm, ex)
        assert len(result) == 1
        assert result[0].constraint_id == "learned:carry_error"
        assert hasattr(ex, "_dynamic_constraints")
        assert any(c.constraint_id == "learned:carry_error" for c in ex._dynamic_constraints)

    def test_added_constraint_logged_as_added(self):
        """Newly added constraint has 'added' in generation_log."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)

        class DummyExtractor:
            pass

        gen = mod.ConstraintGenerator()
        gen.generate_from_memory(cm, DummyExtractor())
        assert any(v == "added" for v in gen.generation_log.values())

    def test_already_existing_logged_as_already_exists(self):
        """SCENARIO-LEARN-018: existing constraint → 'already_exists' in log."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)

        class FakeConstraint:
            def __init__(self, cid):
                self.constraint_id = cid

        class DummyExtractor:
            _dynamic_constraints: list

        ex = DummyExtractor()
        ex._dynamic_constraints = [FakeConstraint("learned:carry_error")]

        gen = mod.ConstraintGenerator()
        result = gen.generate_from_memory(cm, ex)
        assert result == []
        assert any(v == "already_exists" for v in gen.generation_log.values())

    def test_all_three_outcomes_in_one_run(self):
        """SCENARIO-LEARN-018: added, rejected_soundness, already_exists in one run."""
        mod = load_module()
        cm = _make_case_memory()
        # High precision carry_error → should be added
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)
        # Low precision sign_error → rejected_soundness
        _record(cm=cm, violation_types=("sign_error:y",), repair_outcome="unchanged_failure", support=5)
        # magnitude_error → already_exists
        _record(cm=cm, violation_types=("magnitude_error:z",), repair_outcome="improved", support=5)

        class FakeConstraint:
            def __init__(self, cid):
                self.constraint_id = cid

        class DummyExtractor:
            pass

        ex = DummyExtractor()
        ex._dynamic_constraints = [FakeConstraint("learned:magnitude_error")]

        gen = mod.ConstraintGenerator()
        result = gen.generate_from_memory(cm, ex)

        outcomes = set(gen.generation_log.values())
        assert "added" in outcomes
        assert "rejected_soundness" in outcomes
        assert "already_exists" in outcomes
        # Only carry_error should be in the returned list
        assert len(result) == 1
        assert result[0].constraint_id == "learned:carry_error"

    def test_generation_log_reset_between_calls(self):
        """generation_log is reset at the start of each generate_from_memory call."""
        mod = load_module()
        cm = _make_case_memory()
        _record(cm=cm, violation_types=("carry_error:x",), repair_outcome="improved", support=5)

        class DummyExtractor:
            pass

        gen = mod.ConstraintGenerator()
        gen.generate_from_memory(cm, DummyExtractor())
        first_log = dict(gen.generation_log)

        # Second call with empty memory — log should be reset to {}
        empty_cm = _make_case_memory()
        gen.generate_from_memory(empty_cm, DummyExtractor())
        assert gen.generation_log == {}
        # First log had content
        assert first_log != {}
