"""Tests for carnot.pipeline.constraint_template_library.

ConstraintTemplateLibrary implements Tier 2 → Tier 1 constraint ADDITION:
when CaseMemory observes a specific error pattern frequently enough, a new
constraint template is activated — adding a new constraint TYPE to the pipeline
rather than just reweighting existing ones (which Exp 134 proved ineffective).

Coverage targets:
- ConstraintTemplate dataclass fields and defaults
- ConstraintTemplateLibrary: add_template, observe_pattern, get_active_templates,
  apply_active_templates, to_dict, from_dict, register_builtin_templates
- Four built-in template functions: carry_check, sign_check, unit_consistency,
  comparison_direction — each with both "found arithmetic" and "no match" paths

Spec: REQ-LEARN-017, REQ-LEARN-018,
      SCENARIO-LEARN-029, SCENARIO-LEARN-030, SCENARIO-LEARN-031, SCENARIO-LEARN-032
"""

from __future__ import annotations

import pytest

from carnot.pipeline.constraint_template_library import (
    CaseMemoryTemplateWiring,
    ConstraintTemplate,
    ConstraintTemplateLibrary,
    carry_check_template,
    comparison_direction_template,
    sign_check_template,
    unit_consistency_template,
)
from carnot.pipeline.extract import ConstraintResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_template_fn(response: str) -> list[ConstraintResult]:
    """Simple template function for testing: returns one result if 'X' is in response."""
    if "X" in response:
        return [
            ConstraintResult(
                constraint_type="dummy",
                description="dummy constraint",
                metadata={"satisfied": True},
            )
        ]
    return []


def _make_template(
    pattern_key: str = "dummy",
    min_frequency: int = 3,
    template_fn=None,
) -> ConstraintTemplate:
    """Build a minimal ConstraintTemplate for testing."""
    return ConstraintTemplate(
        pattern_key=pattern_key,
        description=f"Test template for {pattern_key}",
        min_frequency=min_frequency,
        template_fn=template_fn or _dummy_template_fn,
    )


# ---------------------------------------------------------------------------
# ConstraintTemplate dataclass
# ---------------------------------------------------------------------------


class TestConstraintTemplate:
    """Tests for the ConstraintTemplate dataclass."""

    def test_required_fields(self):
        """ConstraintTemplate stores pattern_key, description, min_frequency, template_fn."""
        t = _make_template(pattern_key="carry_check", min_frequency=5)
        assert t.pattern_key == "carry_check"
        assert t.min_frequency == 5
        assert callable(t.template_fn)

    def test_default_is_active_false(self):
        """is_active defaults to False before any activation."""
        t = _make_template()
        assert t.is_active is False

    def test_default_activation_count_zero(self):
        """activation_count defaults to 0 before any activation."""
        t = _make_template()
        assert t.activation_count == 0

    def test_fields_are_mutable(self):
        """is_active and activation_count can be updated (they are regular fields)."""
        t = _make_template()
        t.is_active = True
        t.activation_count = 7
        assert t.is_active is True
        assert t.activation_count == 7


# ---------------------------------------------------------------------------
# ConstraintTemplateLibrary
# ---------------------------------------------------------------------------


class TestConstraintTemplateLibrary:
    """Tests for ConstraintTemplateLibrary registration, observation, and activation."""

    # --- add_template ---

    def test_add_template_registers_by_pattern_key(self):
        """add_template stores the template keyed by pattern_key."""
        lib = ConstraintTemplateLibrary()
        t = _make_template(pattern_key="carry_check")
        lib.add_template(t)
        assert "carry_check" in lib._templates
        assert lib._templates["carry_check"] is t

    def test_add_template_replaces_existing(self):
        """add_template with the same pattern_key replaces the previous registration."""
        lib = ConstraintTemplateLibrary()
        t1 = _make_template(pattern_key="carry_check", min_frequency=3)
        t2 = _make_template(pattern_key="carry_check", min_frequency=10)
        lib.add_template(t1)
        lib.add_template(t2)
        assert lib._templates["carry_check"] is t2

    # --- observe_pattern ---

    def test_observe_pattern_increments_count(self):
        """observe_pattern increments observation count for (pattern_key, model_id)."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "qwen3.5-0.8b")
        assert lib._observations[("carry_check", "qwen3.5-0.8b")] == 1

    def test_observe_pattern_additive(self):
        """Multiple calls to observe_pattern accumulate the count."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "qwen3.5-0.8b", count=3)
        lib.observe_pattern("carry_check", "qwen3.5-0.8b", count=5)
        assert lib._observations[("carry_check", "qwen3.5-0.8b")] == 8

    def test_observe_pattern_default_count_one(self):
        """observe_pattern defaults to count=1."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("sign_check", "model_a")
        assert lib._observations[("sign_check", "model_a")] == 1

    def test_observe_pattern_independent_per_model(self):
        """Observations for different model_ids are tracked independently."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "model_a", count=2)
        lib.observe_pattern("carry_check", "model_b", count=7)
        assert lib._observations[("carry_check", "model_a")] == 2
        assert lib._observations[("carry_check", "model_b")] == 7

    def test_observe_unregistered_pattern_key_still_counts(self):
        """Observation counting proceeds even for pattern_keys with no registered template."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("unknown_pattern", "model_a", count=100)
        assert lib._observations[("unknown_pattern", "model_a")] == 100

    # --- get_active_templates ---

    def test_get_active_templates_below_threshold_empty(self):
        """Templates are NOT active when observation count < min_frequency."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template(pattern_key="carry_check", min_frequency=5))
        lib.observe_pattern("carry_check", "model_a", count=4)
        assert lib.get_active_templates("model_a") == []

    def test_get_active_templates_at_threshold(self):
        """Templates become active when observation count == min_frequency."""
        lib = ConstraintTemplateLibrary()
        t = _make_template(pattern_key="carry_check", min_frequency=5)
        lib.add_template(t)
        lib.observe_pattern("carry_check", "model_a", count=5)
        active = lib.get_active_templates("model_a")
        assert len(active) == 1
        assert active[0] is t

    def test_get_active_templates_above_threshold(self):
        """Templates remain active when observation count > min_frequency."""
        lib = ConstraintTemplateLibrary()
        t = _make_template(pattern_key="carry_check", min_frequency=5)
        lib.add_template(t)
        lib.observe_pattern("carry_check", "model_a", count=20)
        active = lib.get_active_templates("model_a")
        assert len(active) == 1

    def test_get_active_templates_no_observations_for_model(self):
        """Templates with no observations for a model are not active for that model."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template(pattern_key="carry_check", min_frequency=5))
        lib.observe_pattern("carry_check", "other_model", count=100)
        # model_a has zero observations
        assert lib.get_active_templates("model_a") == []

    def test_get_active_templates_multiple(self):
        """Multiple templates can be active simultaneously for a model."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template("carry_check", min_frequency=2))
        lib.add_template(_make_template("sign_check", min_frequency=3))
        lib.observe_pattern("carry_check", "m1", count=5)
        lib.observe_pattern("sign_check", "m1", count=5)
        active = lib.get_active_templates("m1")
        assert len(active) == 2

    def test_get_active_templates_partial_activation(self):
        """Only templates that crossed their individual thresholds are returned."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template("carry_check", min_frequency=5))
        lib.add_template(_make_template("sign_check", min_frequency=5))
        lib.observe_pattern("carry_check", "m1", count=10)
        lib.observe_pattern("sign_check", "m1", count=2)  # below threshold
        active = lib.get_active_templates("m1")
        assert len(active) == 1
        assert active[0].pattern_key == "carry_check"

    # --- apply_active_templates ---

    def test_apply_active_templates_returns_constraints(self):
        """apply_active_templates calls template_fn for active templates."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template("dummy", min_frequency=3))
        lib.observe_pattern("dummy", "m1", count=5)
        constraints = lib.apply_active_templates("response with X", "m1")
        assert len(constraints) == 1
        assert constraints[0].constraint_type == "dummy"

    def test_apply_active_templates_empty_when_no_match(self):
        """apply_active_templates returns [] when template_fn finds nothing."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template("dummy", min_frequency=3))
        lib.observe_pattern("dummy", "m1", count=5)
        constraints = lib.apply_active_templates("response without match", "m1")
        assert constraints == []

    def test_apply_active_templates_empty_when_not_active(self):
        """apply_active_templates returns [] when no templates are active."""
        lib = ConstraintTemplateLibrary()
        lib.add_template(_make_template("dummy", min_frequency=5))
        lib.observe_pattern("dummy", "m1", count=2)  # below threshold
        constraints = lib.apply_active_templates("response with X", "m1")
        assert constraints == []

    def test_apply_active_templates_sets_is_active(self):
        """apply_active_templates sets template.is_active = True after first call."""
        lib = ConstraintTemplateLibrary()
        t = _make_template("dummy", min_frequency=3)
        lib.add_template(t)
        lib.observe_pattern("dummy", "m1", count=5)
        assert t.is_active is False
        lib.apply_active_templates("response with X", "m1")
        assert t.is_active is True

    def test_apply_active_templates_increments_activation_count(self):
        """apply_active_templates increments template.activation_count on each call."""
        lib = ConstraintTemplateLibrary()
        t = _make_template("dummy", min_frequency=3)
        lib.add_template(t)
        lib.observe_pattern("dummy", "m1", count=5)
        assert t.activation_count == 0
        lib.apply_active_templates("response with X", "m1")
        assert t.activation_count == 1
        lib.apply_active_templates("another response with X", "m1")
        assert t.activation_count == 2

    def test_apply_active_templates_no_templates_registered(self):
        """apply_active_templates returns [] when library has no registered templates."""
        lib = ConstraintTemplateLibrary()
        assert lib.apply_active_templates("5 + 3 = 8", "m1") == []

    # --- to_dict / from_dict ---

    def test_to_dict_contains_observations(self):
        """to_dict includes the observation counts."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "model_a", count=7)
        d = lib.to_dict()
        assert "observations" in d
        assert len(d["observations"]) == 1
        entry = d["observations"][0]
        assert entry["pattern_key"] == "carry_check"
        assert entry["model_id"] == "model_a"
        assert entry["count"] == 7

    def test_to_dict_empty_library(self):
        """to_dict on empty library returns observations: []."""
        lib = ConstraintTemplateLibrary()
        d = lib.to_dict()
        assert d == {"observations": []}

    def test_from_dict_restores_observations(self):
        """from_dict restores observation counts from a previously serialized dict."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "model_a", count=7)
        lib.observe_pattern("sign_check", "model_b", count=3)
        d = lib.to_dict()

        lib2 = ConstraintTemplateLibrary.from_dict(d)
        assert lib2._observations[("carry_check", "model_a")] == 7
        assert lib2._observations[("sign_check", "model_b")] == 3

    def test_from_dict_no_templates_without_register(self):
        """from_dict does not restore templates (only observations)."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "model_a", count=7)
        d = lib.to_dict()
        lib2 = ConstraintTemplateLibrary.from_dict(d)
        assert lib2._templates == {}

    def test_from_dict_empty_payload(self):
        """from_dict handles an empty payload without error."""
        lib = ConstraintTemplateLibrary.from_dict({})
        assert lib._observations == {}
        assert lib._templates == {}

    def test_roundtrip_with_register_builtin(self):
        """Round-trip: serialize observations, restore, register builtins, verify activation."""
        lib = ConstraintTemplateLibrary()
        lib.observe_pattern("carry_check", "qwen3.5-0.8b", count=20)
        d = lib.to_dict()

        lib2 = ConstraintTemplateLibrary.from_dict(d)
        lib2.register_builtin_templates()

        active = lib2.get_active_templates("qwen3.5-0.8b")
        keys = {t.pattern_key for t in active}
        assert "carry_check" in keys

    # --- register_builtin_templates ---

    def test_register_builtin_templates_registers_all_four(self):
        """register_builtin_templates registers carry_check, sign_check, unit_consistency,
        and comparison_direction."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        assert "carry_check" in lib._templates
        assert "sign_check" in lib._templates
        assert "unit_consistency" in lib._templates
        assert "comparison_direction" in lib._templates

    def test_register_builtin_carry_check_min_frequency(self):
        """carry_check builtin has min_frequency=5."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        assert lib._templates["carry_check"].min_frequency == 5

    def test_register_builtin_sign_check_min_frequency(self):
        """sign_check builtin has min_frequency=5."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        assert lib._templates["sign_check"].min_frequency == 5

    def test_register_builtin_unit_consistency_min_frequency(self):
        """unit_consistency builtin has min_frequency=3 (lower because unit errors are rarer)."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        assert lib._templates["unit_consistency"].min_frequency == 3

    def test_register_builtin_comparison_direction_min_frequency(self):
        """comparison_direction builtin has min_frequency=5."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        assert lib._templates["comparison_direction"].min_frequency == 5

    def test_register_builtin_templates_callable(self):
        """All registered builtin templates have callable template_fn."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        for t in lib._templates.values():
            assert callable(t.template_fn)


# ---------------------------------------------------------------------------
# carry_check_template
# ---------------------------------------------------------------------------


class TestCarryCheckTemplate:
    """Tests for the carry_check_template function.

    Spec: SCENARIO-LEARN-029
    """

    def test_no_arithmetic_returns_empty(self):
        """Returns [] when no multiplication patterns found."""
        assert carry_check_template("The answer is forty-two.") == []

    def test_single_digit_multiplication_skipped(self):
        """Single-digit × single-digit requires no carry — not checked."""
        # 3 × 7 = 21 uses no multi-digit carry, so no constraint generated.
        result = carry_check_template("We have 3 × 7 = 21.")
        assert result == []

    def test_multi_digit_correct(self):
        """Multi-digit multiplication with correct product: satisfied=True."""
        results = carry_check_template("24 × 3 = 72")
        assert len(results) == 1
        r = results[0]
        assert r.constraint_type == "carry_check"
        assert r.metadata["a"] == 24
        assert r.metadata["b"] == 3
        assert r.metadata["claimed"] == 72
        assert r.metadata["correct"] == 72
        assert r.metadata["satisfied"] is True

    def test_multi_digit_incorrect(self):
        """Multi-digit multiplication with wrong product: satisfied=False."""
        # 24 × 3 should be 72, not 62 (carry drop error)
        results = carry_check_template("24 × 3 = 62")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False
        assert r.metadata["correct"] == 72
        assert r.metadata["claimed"] == 62

    def test_star_multiplication_symbol(self):
        """Accepts * as multiplication symbol as well as ×."""
        results = carry_check_template("12 * 5 = 60")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_multiple_multiplications(self):
        """Finds all multi-digit multiplication expressions in the response."""
        response = "Step 1: 12 × 3 = 36. Step 2: 25 × 4 = 100."
        results = carry_check_template(response)
        assert len(results) == 2

    def test_multi_digit_second_factor(self):
        """Activates when the second factor (not the first) is multi-digit."""
        results = carry_check_template("3 × 15 = 45")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True


# ---------------------------------------------------------------------------
# sign_check_template
# ---------------------------------------------------------------------------


class TestSignCheckTemplate:
    """Tests for the sign_check_template function.

    Spec: SCENARIO-LEARN-030
    """

    def test_no_negative_pattern_returns_empty(self):
        """Returns [] when no (-A) × (-B) pattern found."""
        assert sign_check_template("The answer is 12.") == []

    def test_positive_multiplication_not_matched(self):
        """Positive × positive is not a sign check pattern."""
        assert sign_check_template("3 × 4 = 12") == []

    def test_neg_neg_positive_result_satisfied(self):
        """(-3) × (-4) = 12 is satisfied (positive result)."""
        results = sign_check_template("(-3) × (-4) = 12")
        assert len(results) == 1
        r = results[0]
        assert r.constraint_type == "sign_check"
        assert r.metadata["satisfied"] is True
        assert r.metadata["claimed"] == 12.0

    def test_neg_neg_negative_result_violated(self):
        """(-3) × (-4) = -12 is a violation (wrong sign)."""
        results = sign_check_template("(-3) × (-4) = -12")
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False
        assert r.metadata["claimed"] == -12.0

    def test_star_multiplication_symbol(self):
        """Accepts * as multiplication symbol."""
        results = sign_check_template("(-5) * (-2) = 10")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_decimal_factors(self):
        """Handles decimal factors like (-1.5) × (-2.0) = 3.0."""
        results = sign_check_template("(-1.5) × (-2.0) = 3.0")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_metadata_stores_negative_a_b(self):
        """Metadata stores a and b as negative (the actual factor values)."""
        results = sign_check_template("(-3) × (-4) = 12")
        r = results[0]
        assert r.metadata["a"] == -3.0
        assert r.metadata["b"] == -4.0


# ---------------------------------------------------------------------------
# unit_consistency_template
# ---------------------------------------------------------------------------


class TestUnitConsistencyTemplate:
    """Tests for the unit_consistency_template function.

    Spec: SCENARIO-LEARN-031
    """

    def test_no_units_returns_empty(self):
        """Returns [] when no unit annotations found."""
        assert unit_consistency_template("The sum is 42.") == []

    def test_consistent_units_kg_only(self):
        """Single unit type (kg) is consistent — returns satisfied=True."""
        results = unit_consistency_template("5 kg + 3 kg = 8 kg")
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_inconsistent_kg_g_flagged(self):
        """Mixing kg and g without conversion: satisfied=False."""
        results = unit_consistency_template("5 kg and 200 g are added together")
        violated = [r for r in results if r.metadata["satisfied"] is False]
        assert len(violated) == 1
        pair = set(violated[0].metadata["inconsistent_pair"])
        assert pair == {"kg", "g"}

    def test_inconsistent_km_m_flagged(self):
        """Mixing km and m without conversion: satisfied=False."""
        results = unit_consistency_template("Distance is 2 km + 500 m")
        violated = [r for r in results if r.metadata["satisfied"] is False]
        assert len(violated) >= 1
        pairs = [set(r.metadata["inconsistent_pair"]) for r in violated]
        assert {"km", "m"} in pairs

    def test_inconsistent_L_ml_flagged(self):
        """Mixing L and ml without conversion: satisfied=False."""
        results = unit_consistency_template("Volume: 2 L and 300 ml")
        violated = [r for r in results if r.metadata["satisfied"] is False]
        assert len(violated) >= 1
        pairs = [set(r.metadata["inconsistent_pair"]) for r in violated]
        assert {"L", "ml"} in pairs

    def test_constraint_type_is_unit_consistency(self):
        """All results have constraint_type='unit_consistency'."""
        results = unit_consistency_template("5 kg and 200 g")
        for r in results:
            assert r.constraint_type == "unit_consistency"

    def test_units_found_in_metadata(self):
        """metadata['units_found'] lists all units discovered in the response."""
        results = unit_consistency_template("5 kg box weighs 5 kg")
        assert "kg" in results[0].metadata["units_found"]


# ---------------------------------------------------------------------------
# comparison_direction_template
# ---------------------------------------------------------------------------


class TestComparisonDirectionTemplate:
    """Tests for the comparison_direction_template function.

    Spec: SCENARIO-LEARN-032
    """

    def test_no_comparison_returns_empty(self):
        """Returns [] when no X > Y pattern found."""
        assert comparison_direction_template("The sum is 10.") == []

    def test_comparison_without_matching_subtraction_returns_empty(self):
        """Returns [] when X > Y is found but no matching X - Y = Z subtraction exists."""
        result = comparison_direction_template("We know that 50 > 30.")
        assert result == []

    def test_consistent_comparison_satisfied(self):
        """50 > 30 with 50 - 30 = 20: satisfied=True (20 > 0)."""
        response = "Since 50 > 30, we compute 50 - 30 = 20."
        results = comparison_direction_template(response)
        assert len(results) == 1
        r = results[0]
        assert r.constraint_type == "comparison_direction"
        assert r.metadata["satisfied"] is True
        assert r.metadata["z"] == 20.0

    def test_inconsistent_comparison_violated(self):
        """50 > 30 with 50 - 30 = -20: satisfied=False (direction wrong)."""
        response = "Since 50 > 30, we compute 50 - 30 = -20."
        results = comparison_direction_template(response)
        assert len(results) == 1
        r = results[0]
        assert r.metadata["satisfied"] is False

    def test_subtraction_without_gt_pair_not_matched(self):
        """Subtraction X - Y = Z is only checked when the exact (X, Y) > pair exists."""
        # 40 > 20 but subtraction uses 50 - 30 — no match
        response = "40 > 20 but 50 - 30 = 20"
        results = comparison_direction_template(response)
        assert results == []

    def test_metadata_fields(self):
        """comparison_direction result includes x, y, z in metadata."""
        response = "70 > 40 and 70 - 40 = 30"
        results = comparison_direction_template(response)
        assert len(results) == 1
        r = results[0]
        assert r.metadata["x"] == 70.0
        assert r.metadata["y"] == 40.0
        assert r.metadata["z"] == 30.0

    def test_decimal_comparison(self):
        """Handles decimal comparisons like 3.5 > 2.1 and 3.5 - 2.1 = 1.4."""
        response = "3.5 > 2.1 so 3.5 - 2.1 = 1.4"
        results = comparison_direction_template(response)
        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True


# ---------------------------------------------------------------------------
# VerifyRepairPipeline integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CaseMemoryTemplateWiring
# ---------------------------------------------------------------------------


class TestCaseMemoryTemplateWiring:
    """Tests for CaseMemoryTemplateWiring — the Tier 2 → Tier 1 feedback bridge.

    Spec: REQ-LEARN-019, SCENARIO-LEARN-033, SCENARIO-LEARN-034
    """

    # --- violation_type_to_pattern_key ---

    def test_carry_error_maps_to_carry_check(self):
        """'carry_error' maps to 'carry_check'.

        Spec: SCENARIO-LEARN-033
        """
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("carry_error") == "carry_check"

    def test_sign_error_maps_to_sign_check(self):
        """'sign_error' maps to 'sign_check'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("sign_error") == "sign_check"

    def test_unit_error_maps_to_unit_consistency(self):
        """'unit_error' maps to 'unit_consistency'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("unit_error") == "unit_consistency"

    def test_comparison_error_maps_to_comparison_direction(self):
        """'comparison_error' maps to 'comparison_direction'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("comparison_error") == "comparison_direction"

    def test_substring_carry_in_type_maps_to_carry_check(self):
        """Any type containing 'carry' maps to 'carry_check' (substring match)."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("multi_carry_propagation") == "carry_check"

    def test_substring_sign_in_type_maps_to_sign_check(self):
        """Any type containing 'sign' maps to 'sign_check'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("wrong_sign_result") == "sign_check"

    def test_substring_unit_in_type_maps_to_unit_consistency(self):
        """Any type containing 'unit' maps to 'unit_consistency'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("unit_mismatch") == "unit_consistency"

    def test_substring_comparison_in_type_maps_to_comparison_direction(self):
        """Any type containing 'comparison' maps to 'comparison_direction'."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("numeric_comparison_error") == "comparison_direction"

    def test_unknown_type_passes_through(self):
        """An unrecognized violation_type is returned unchanged (pass-through).

        Spec: SCENARIO-LEARN-034
        """
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("range_check") == "range_check"

    def test_unknown_type_with_no_keyword_passes_through(self):
        """A type with none of the keyword substrings passes through unchanged."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("semantic_grounding") == "semantic_grounding"

    def test_case_insensitive_carry(self):
        """Mapping is case-insensitive: 'CARRY_ERROR' maps to 'carry_check'.

        Spec: REQ-LEARN-019-4
        """
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("CARRY_ERROR") == "carry_check"

    def test_case_insensitive_sign(self):
        """'SIGN_ERROR' maps to 'sign_check' (case-insensitive)."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("Sign_Error") == "sign_check"

    def test_case_insensitive_unit(self):
        """'UNIT_MISMATCH' maps to 'unit_consistency' (case-insensitive)."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("UNIT_MISMATCH") == "unit_consistency"

    def test_case_insensitive_comparison(self):
        """'COMPARISON_FAILURE' maps to 'comparison_direction' (case-insensitive)."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring.violation_type_to_pattern_key("COMPARISON_FAILURE") == "comparison_direction"

    # --- on_violation_recorded ---

    def test_on_violation_recorded_increments_carry_check(self):
        """on_violation_recorded('carry_error', model) increments carry_check count.

        Spec: SCENARIO-LEARN-033
        """
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("carry_error", "qwen3.5-0.8b")
        assert lib._observations[("carry_check", "qwen3.5-0.8b")] == 1

    def test_on_violation_recorded_increments_sign_check(self):
        """on_violation_recorded('sign_error', model) increments sign_check count."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("sign_error", "model_a")
        assert lib._observations[("sign_check", "model_a")] == 1

    def test_on_violation_recorded_increments_unit_consistency(self):
        """on_violation_recorded('unit_error', model) increments unit_consistency count."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("unit_error", "model_a")
        assert lib._observations[("unit_consistency", "model_a")] == 1

    def test_on_violation_recorded_increments_comparison_direction(self):
        """on_violation_recorded('comparison_error', model) increments comparison_direction."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("comparison_error", "model_a")
        assert lib._observations[("comparison_direction", "model_a")] == 1

    def test_on_violation_recorded_unknown_passes_through(self):
        """on_violation_recorded with unknown type increments that type's count directly.

        Spec: SCENARIO-LEARN-034
        """
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("range_check", "model_a")
        assert lib._observations[("range_check", "model_a")] == 1

    def test_on_violation_recorded_accumulates(self):
        """Repeated calls to on_violation_recorded accumulate the count."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        for _ in range(7):
            wiring.on_violation_recorded("carry_error", "qwen3.5-0.8b")
        assert lib._observations[("carry_check", "qwen3.5-0.8b")] == 7

    def test_on_violation_recorded_independent_per_model(self):
        """Violations for different models are tracked independently."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        wiring.on_violation_recorded("carry_error", "model_a")
        wiring.on_violation_recorded("carry_error", "model_b")
        wiring.on_violation_recorded("carry_error", "model_b")
        assert lib._observations[("carry_check", "model_a")] == 1
        assert lib._observations[("carry_check", "model_b")] == 2

    def test_wiring_activates_template_after_threshold(self):
        """After enough on_violation_recorded calls, the corresponding template activates."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        wiring = CaseMemoryTemplateWiring(lib)
        # carry_check needs min_frequency=5
        for _ in range(5):
            wiring.on_violation_recorded("carry_error", "test-model")
        active = lib.get_active_templates("test-model")
        keys = {t.pattern_key for t in active}
        assert "carry_check" in keys

    def test_wiring_does_not_activate_below_threshold(self):
        """Template stays inactive if on_violation_recorded has not crossed min_frequency."""
        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        wiring = CaseMemoryTemplateWiring(lib)
        # carry_check needs 5; only 4 calls
        for _ in range(4):
            wiring.on_violation_recorded("carry_error", "test-model")
        active = lib.get_active_templates("test-model")
        assert active == []

    def test_wiring_stores_library_reference(self):
        """CaseMemoryTemplateWiring stores the library it was given."""
        lib = ConstraintTemplateLibrary()
        wiring = CaseMemoryTemplateWiring(lib)
        assert wiring._library is lib


class TestVerifyRepairPipelineIntegration:
    """Tests that VerifyRepairPipeline accepts and uses a ConstraintTemplateLibrary.

    Spec: REQ-LEARN-017-3
    """

    def test_pipeline_accepts_template_library_param(self):
        """VerifyRepairPipeline can be constructed with template_library=..."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        lib = ConstraintTemplateLibrary()
        lib.register_builtin_templates()
        # Should not raise
        pipeline = VerifyRepairPipeline(
            model=None,
            template_library=lib,
        )
        assert pipeline._template_library is lib

    def test_pipeline_none_template_library_by_default(self):
        """VerifyRepairPipeline._template_library is None when not provided."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(model=None)
        assert pipeline._template_library is None

    def test_pipeline_verify_merges_template_constraints(self):
        """verify() merges constraints from active templates into the result."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        lib = ConstraintTemplateLibrary()
        # Register a dummy template that always returns one violated constraint
        def always_violated(response: str) -> list[ConstraintResult]:
            return [ConstraintResult(
                constraint_type="carry_check",
                description="carry error injected by template",
                metadata={"satisfied": False},
            )]

        lib.add_template(ConstraintTemplate(
            pattern_key="carry_check",
            description="Carry check",
            min_frequency=1,
            template_fn=always_violated,
        ))
        # Observe the pattern 5 times to trigger activation
        lib.observe_pattern("carry_check", "test-model", count=5)

        # Use model=None (verify-only mode) then set _model_name for template lookup.
        # We cannot pass model="test-model" as that triggers a real HuggingFace load.
        pipeline = VerifyRepairPipeline(
            model=None,
            template_library=lib,
        )
        # Manually set the model name so apply_active_templates looks up "test-model"
        pipeline._model_name = "test-model"

        result = pipeline.verify(
            question="What is 24 × 3?",
            response="24 × 3 = 62",
        )
        # The template constraint (carry error) should appear in the result
        template_constraints = [
            c for c in result.constraints if c.constraint_type == "carry_check"
        ]
        assert len(template_constraints) >= 1


# ---------------------------------------------------------------------------
# ViolationPatternLibrary tests — REQ-SELF-020
# ---------------------------------------------------------------------------


class TestViolationPatternLibrary:
    """Tests for ViolationPatternLibrary — the FR-11 cross-session relay store.

    Covers: __init__, _load, _save, add_template (new + duplicate), get_fp_rate.

    Spec: REQ-SELF-020, SCENARIO-SELF-025, SCENARIO-SELF-026
    """

    def test_init_empty_no_file(self, tmp_path):
        """Library starts with zero templates when backing file does not exist.

        Spec: REQ-SELF-020-1
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        assert lib.templates == []

    def test_add_template_creates_entry(self, tmp_path):
        """add_template creates a ViolationPatternEntry and persists it.

        Spec: REQ-SELF-020-2, SCENARIO-SELF-025
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        entry = lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        assert entry.template_id == "0"
        assert entry.violation_pattern == "COMPUTE: 47 + 28 = 76"
        assert entry.violation_type == "arithmetic"
        assert entry.source_experiment == 656
        assert len(lib.templates) == 1

    def test_add_template_deduplicates(self, tmp_path):
        """Adding the same pattern twice returns the existing entry without duplication.

        Spec: REQ-SELF-020-2
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        e1 = lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        e2 = lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        assert e1.template_id == e2.template_id
        assert len(lib.templates) == 1

    def test_add_template_strips_whitespace(self, tmp_path):
        """add_template normalises leading/trailing whitespace before deduplication.

        Spec: REQ-SELF-020-2
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        lib.add_template("  COMPUTE: 47 + 28 = 76  ", "arithmetic", 656)
        assert lib.templates[0].violation_pattern == "COMPUTE: 47 + 28 = 76"

    def test_persistence_round_trip(self, tmp_path):
        """Templates written by add_template are reloaded on next instantiation.

        Spec: REQ-SELF-020-1, REQ-SELF-020-2
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        path = str(tmp_path / "templates.json")
        lib1 = ViolationPatternLibrary(path)
        lib1.add_template("total is 80", "arithmetic", 656)
        lib1.add_template("therefore 15", "arithmetic", 656)

        lib2 = ViolationPatternLibrary(path)
        assert len(lib2.templates) == 2
        assert lib2.templates[0].violation_pattern == "total is 80"
        assert lib2.templates[1].violation_pattern == "therefore 15"

    def test_get_fp_rate_no_match(self, tmp_path):
        """FP rate is 0.0 when no correct response contains any stored pattern.

        Spec: REQ-SELF-020-3, SCENARIO-SELF-026
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        lib.add_template("total is 80", "arithmetic", 656)

        correct = ["The answer is 18.", "Janet makes $18 per day."]
        assert lib.get_fp_rate(correct) == 0.0

    def test_get_fp_rate_partial_match(self, tmp_path):
        """FP rate counts responses that contain any stored pattern.

        Spec: REQ-SELF-020-3
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        lib.add_template("total is 80", "arithmetic", 656)

        responses = [
            "The total is 80 items.",  # matches
            "The answer is 18.",       # no match
        ]
        rate = lib.get_fp_rate(responses)
        assert rate == pytest.approx(0.5)

    def test_get_fp_rate_empty_templates(self, tmp_path):
        """FP rate is 0.0 when no templates are stored.

        Spec: REQ-SELF-020-3
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        assert lib.get_fp_rate(["any response"]) == 0.0

    def test_get_fp_rate_empty_responses(self, tmp_path):
        """FP rate is 0.0 when the responses list is empty.

        Spec: REQ-SELF-020-3
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        assert lib.get_fp_rate([]) == 0.0

    def test_load_handles_corrupt_file(self, tmp_path):
        """Library starts fresh when the backing file is corrupt JSON.

        Spec: REQ-SELF-020-1
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        path = tmp_path / "templates.json"
        path.write_text("{not valid json!!!}")
        lib = ViolationPatternLibrary(str(path))
        assert lib.templates == []

    def test_n_triggered_default_zero(self, tmp_path):
        """New ViolationPatternEntry starts with n_triggered == 0.

        Spec: REQ-SELF-020
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        entry = lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        assert entry.n_triggered == 0

    def test_added_date_is_set(self, tmp_path):
        """New ViolationPatternEntry has a non-empty added_date timestamp.

        Spec: REQ-SELF-020
        """
        from carnot.pipeline.constraint_template_library import ViolationPatternLibrary

        lib = ViolationPatternLibrary(str(tmp_path / "templates.json"))
        entry = lib.add_template("COMPUTE: 47 + 28 = 76", "arithmetic", 656)
        assert isinstance(entry.added_date, str) and len(entry.added_date) > 0

    def test_violation_pattern_entry_exported(self):
        """ViolationPatternEntry and ViolationPatternLibrary are in __all__.

        Spec: REQ-SELF-020
        """
        import carnot.pipeline.constraint_template_library as mod

        assert "ViolationPatternEntry" in mod.__all__
        assert "ViolationPatternLibrary" in mod.__all__
