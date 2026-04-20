"""Tests for HISR wire-in into ConstraintAdditionFromMemory.

Covers the new hisr_weighted_add() and add_from_memory() methods added in Exp 610.

Spec: REQ-LEARN-075, SCENARIO-LEARN-110, SCENARIO-LEARN-111
"""

from __future__ import annotations

import pytest

from carnot.pipeline.constraint_addition import (
    ConstraintAdditionFromMemory,
    ViolationPattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _five_violations() -> list[ViolationPattern]:
    """Return 5 violation patterns for a single chain.

    For final_correct=False, HISR scores are:
        index 0 (carry):    1/(1+4) = 0.20  → filtered
        index 1 (sign):     1/(1+3) = 0.25  → filtered
        index 2 (unit):     1/(1+2) = 0.33  → filtered
        index 3 (carry):    1/(1+1) = 0.50  → retained
        index 4 (sign):     1/(1+0) = 1.00  → retained
    """
    return [
        ViolationPattern(type="carry", count=1, example_steps=["early_step_carry"]),
        ViolationPattern(type="sign", count=1, example_steps=["early_step_sign"]),
        ViolationPattern(type="unit", count=1, example_steps=["mid_step_unit"]),
        ViolationPattern(type="carry", count=1, example_steps=["late_step_carry"]),
        ViolationPattern(type="sign", count=1, example_steps=["late_step_sign"]),
    ]


# ---------------------------------------------------------------------------
# REQ-LEARN-075-1 / SCENARIO-LEARN-110: incorrect chain filters low-signal
# ---------------------------------------------------------------------------


def test_hisr_weighted_add_incorrect_chain_filters_low_signal() -> None:
    """Only violations at or above the 0.5 HISR threshold are observed.

    For a 5-violation chain ending incorrectly:
    - Indices 0-2 get scores < 0.5 and must NOT be observed.
    - Index 3 gets score 0.5 and MUST be observed.
    - Index 4 gets score 1.0 and MUST be observed.

    Spec: REQ-LEARN-075-1, SCENARIO-LEARN-110
    """
    monitor = ConstraintAdditionFromMemory(threshold=100)  # high threshold — no additions
    monitor.hisr_weighted_add(_five_violations(), final_correct=False)

    counts = monitor.get_pattern_counts()

    # 'unit' (index 2, score=0.33) must NOT appear.
    assert "unit" not in counts, (
        "unit (score=0.33) must be filtered out by HISR threshold"
    )

    # 'carry' and 'sign' must appear (observed from indices 3 and 4).
    assert "carry" in counts, "carry (score >= 0.5) must be observed"
    assert "sign" in counts, "sign (score >= 0.5) must be observed"

    # Total observations: 1 carry + 1 sign = 2.
    total = sum(counts.values())
    assert total == 2, f"Expected 2 high-signal observations, got {total}"


# ---------------------------------------------------------------------------
# REQ-LEARN-075-2 / SCENARIO-LEARN-111: correct chain adds nothing
# ---------------------------------------------------------------------------


def test_hisr_weighted_add_correct_chain_observes_nothing() -> None:
    """When final_correct=True, no violations are observed.

    All violations in a correct chain are false positives (score=0.0) and
    must not be added to the pattern counts.

    Spec: REQ-LEARN-075-2, SCENARIO-LEARN-111
    """
    monitor = ConstraintAdditionFromMemory(threshold=1)
    added = monitor.hisr_weighted_add(_five_violations(), final_correct=True)

    assert monitor.get_pattern_counts() == {}, (
        "No violations should be observed when final_correct=True"
    )
    assert added == [], "No constraints should be added from a correct chain"


# ---------------------------------------------------------------------------
# REQ-LEARN-075-3: add_from_memory delegates to hisr when use_hisr=True
# ---------------------------------------------------------------------------


def test_add_from_memory_use_hisr_delegates_correctly() -> None:
    """add_from_memory(use_hisr=True) delegates to hisr_weighted_add.

    Verifies same filtering behaviour as test_hisr_weighted_add_incorrect_chain.

    Spec: REQ-LEARN-075-3
    """
    monitor = ConstraintAdditionFromMemory(threshold=100)
    monitor.add_from_memory(_five_violations(), final_correct=False, use_hisr=True)

    counts = monitor.get_pattern_counts()
    assert "unit" not in counts, "unit must be filtered by HISR via add_from_memory"
    assert "carry" in counts
    assert "sign" in counts


# ---------------------------------------------------------------------------
# REQ-LEARN-075-4: add_from_memory uniform path observes all violations
# ---------------------------------------------------------------------------


def test_add_from_memory_uniform_observes_all() -> None:
    """add_from_memory(use_hisr=False) observes all violations uniformly.

    Each ViolationPattern.count must be respected — the same type is observed
    ``count`` times.

    Spec: REQ-LEARN-075-4
    """
    violations = [
        ViolationPattern(type="carry", count=3, example_steps=["s1"]),
        ViolationPattern(type="unit", count=2, example_steps=["s2"]),
    ]
    monitor = ConstraintAdditionFromMemory(threshold=100)
    monitor.add_from_memory(violations, use_hisr=False)

    counts = monitor.get_pattern_counts()
    assert counts.get("carry") == 3, f"Expected carry=3, got {counts.get('carry')}"
    assert counts.get("unit") == 2, f"Expected unit=2, got {counts.get('unit')}"


# ---------------------------------------------------------------------------
# add_from_memory triggers constraint addition when threshold crossed
# ---------------------------------------------------------------------------


def test_add_from_memory_uniform_triggers_addition_at_threshold() -> None:
    """add_from_memory (uniform) adds a constraint once count >= threshold.

    Spec: REQ-LEARN-075-4
    """
    violations = [
        ViolationPattern(type="carry", count=5, example_steps=["s"]),
    ]
    monitor = ConstraintAdditionFromMemory(threshold=5)
    added = monitor.add_from_memory(violations, use_hisr=False)

    assert "carry_check_constraint" in added, (
        "carry_check_constraint must be added when count >= threshold"
    )


def test_add_from_memory_hisr_triggers_addition_when_high_signal_crosses_threshold() -> None:
    """HISR path adds constraint when high-signal violation count crosses threshold.

    After 2 calls with final_correct=False, the high-signal violation types
    (carry and sign) each accumulate 2 observations, crossing threshold=2.

    Spec: REQ-LEARN-075-3
    """
    violations = [
        ViolationPattern(type="carry", count=1, example_steps=["e0"]),
        ViolationPattern(type="sign", count=1, example_steps=["e1"]),
        ViolationPattern(type="unit", count=1, example_steps=["e2"]),
        ViolationPattern(type="carry", count=1, example_steps=["e3"]),
        ViolationPattern(type="sign", count=1, example_steps=["e4"]),
    ]
    monitor = ConstraintAdditionFromMemory(threshold=2)

    # First call — carry and sign each observed once; threshold not crossed yet.
    added_first = monitor.add_from_memory(violations, final_correct=False, use_hisr=True)
    assert "carry_check_constraint" not in added_first

    # Second call — now carry and sign each observed twice → threshold crossed.
    added_second = monitor.add_from_memory(violations, final_correct=False, use_hisr=True)
    assert "carry_check_constraint" in added_second or "sign_check_constraint" in added_second, (
        "At least one constraint must be added after threshold is crossed"
    )
