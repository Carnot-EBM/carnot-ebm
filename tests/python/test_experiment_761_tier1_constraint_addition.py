"""Tests for Experiment 761 — Tier 1 Constraint Addition Engine.

Coverage targets (REQ-LEARN-040, REQ-LEARN-041):

- test_scan_for_patterns_groups_by_violation_type:
    scan_for_patterns() reads _violations_by_type and returns ConstraintPattern
    objects only for types where count >= min_count.  (REQ-LEARN-040-2, SCENARIO-LEARN-080)

- test_scan_for_patterns_excludes_below_threshold:
    Violation types with count < min_count are NOT returned.  (REQ-LEARN-040-2)

- test_scan_for_patterns_sorted_by_count_descending:
    Patterns are returned in descending order of count.  (REQ-LEARN-040-2)

- test_generate_constraint_carry_error:
    generate_constraint returns CarryCheckConstraint for "carry_error".  (REQ-LEARN-040-3)

- test_generate_constraint_sign_error:
    generate_constraint returns SignCheckConstraint for "sign_error".  (REQ-LEARN-040-3)

- test_generate_constraint_unit_error:
    generate_constraint returns UnitCheckConstraint for "unit_error".  (REQ-LEARN-040-3)

- test_generate_constraint_comparison_error:
    generate_constraint returns ComparisonDirectionConstraint for "comparison_error".
    (REQ-LEARN-040-3)

- test_generate_constraint_unknown_type_returns_none:
    generate_constraint returns None for an unrecognised violation type.  (REQ-LEARN-040-3)

- test_inject_into_pipeline_adds_new_constraints:
    inject_into_pipeline appends constraints not already present and returns the count.
    (REQ-LEARN-040-4)

- test_inject_into_pipeline_avoids_duplicates:
    inject_into_pipeline does NOT add a constraint whose name is already in the list,
    and returns 0 for a fully-duplicate call.  (REQ-LEARN-040-4, SCENARIO-LEARN-081)

- test_inject_into_pipeline_no_active_constraints_attribute:
    inject_into_pipeline returns 0 gracefully when pipeline lacks active_constraints.
    (REQ-LEARN-040-4)

- test_constraint_addition_engine_empty_memory:
    When session_memory has no violations, scan returns [] and inject returns 0.
    (REQ-LEARN-040-2)

- test_ten_session_precision_non_decreasing:
    Running the full synthetic 10-session experiment produces monotonic_non_decreasing=True
    and constraints_added_total > 0.  (REQ-LEARN-041)
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

# Ensure repo root is on path for imports.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.constraint_addition_engine import (
    CarryCheckConstraint,
    ComparisonDirectionConstraint,
    ConstraintAdditionEngine,
    ConstraintPattern,
    SignCheckConstraint,
    UnitCheckConstraint,
)
from python.carnot.pipeline.session_memory import SessionMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_memory(violations: dict[str, int]) -> SessionMemory:
    """Create a SessionMemory stub with pre-populated _violations_by_type."""
    sm = SessionMemory(storage_dir="/tmp/exp761_test", model_id="test_model")
    sm._violations_by_type = dict(violations)
    return sm


class _StubPipeline:
    """Minimal stub pipeline with active_constraints list."""

    def __init__(self, existing: list | None = None) -> None:
        self.active_constraints: list = list(existing) if existing else []


class _NoPipeline:
    """Stub pipeline WITHOUT active_constraints attribute."""

    pass


# ---------------------------------------------------------------------------
# scan_for_patterns tests
# ---------------------------------------------------------------------------


def test_scan_for_patterns_groups_by_violation_type() -> None:
    """scan_for_patterns groups by violation_type and returns ConstraintPattern objects.

    Spec: REQ-LEARN-040-2, SCENARIO-LEARN-080
    """
    sm = _make_session_memory({"carry_error": 5, "sign_error": 3, "unit_error": 1})
    engine = ConstraintAdditionEngine(sm, min_count=3)

    patterns = engine.scan_for_patterns()

    vtypes = {p.violation_type for p in patterns}
    assert "carry_error" in vtypes
    assert "sign_error" in vtypes
    assert "unit_error" not in vtypes  # count=1 < min_count=3


def test_scan_for_patterns_excludes_below_threshold() -> None:
    """Violation types with count < min_count are excluded.

    Spec: REQ-LEARN-040-2, SCENARIO-LEARN-080
    """
    sm = _make_session_memory({"carry_error": 5, "sign_error": 2})
    engine = ConstraintAdditionEngine(sm, min_count=3)

    patterns = engine.scan_for_patterns()

    assert len(patterns) == 1
    assert patterns[0].violation_type == "carry_error"
    assert patterns[0].count == 5


def test_scan_for_patterns_sorted_by_count_descending() -> None:
    """Patterns are sorted highest count first.

    Spec: REQ-LEARN-040-2
    """
    sm = _make_session_memory({"sign_error": 4, "carry_error": 9, "comparison_error": 6})
    engine = ConstraintAdditionEngine(sm, min_count=3)

    patterns = engine.scan_for_patterns()

    counts = [p.count for p in patterns]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# generate_constraint tests
# ---------------------------------------------------------------------------


def test_generate_constraint_carry_error() -> None:
    """generate_constraint returns CarryCheckConstraint for carry_error.

    Spec: REQ-LEARN-040-3
    """
    sm = _make_session_memory({})
    engine = ConstraintAdditionEngine(sm)
    pattern = ConstraintPattern(violation_type="carry_error", count=5, example_text="test")

    result = engine.generate_constraint(pattern)

    assert isinstance(result, CarryCheckConstraint)
    assert result.name == "carry_check_constraint"


def test_generate_constraint_sign_error() -> None:
    """generate_constraint returns SignCheckConstraint for sign_error.

    Spec: REQ-LEARN-040-3
    """
    sm = _make_session_memory({})
    engine = ConstraintAdditionEngine(sm)
    pattern = ConstraintPattern(violation_type="sign_error", count=4, example_text="test")

    result = engine.generate_constraint(pattern)

    assert isinstance(result, SignCheckConstraint)
    assert result.name == "sign_check_constraint"


def test_generate_constraint_unit_error() -> None:
    """generate_constraint returns UnitCheckConstraint for unit_error.

    Spec: REQ-LEARN-040-3
    """
    sm = _make_session_memory({})
    engine = ConstraintAdditionEngine(sm)
    pattern = ConstraintPattern(violation_type="unit_error", count=3, example_text="test")

    result = engine.generate_constraint(pattern)

    assert isinstance(result, UnitCheckConstraint)
    assert result.name == "unit_check_constraint"


def test_generate_constraint_comparison_error() -> None:
    """generate_constraint returns ComparisonDirectionConstraint for comparison_error.

    Spec: REQ-LEARN-040-3
    """
    sm = _make_session_memory({})
    engine = ConstraintAdditionEngine(sm)
    pattern = ConstraintPattern(
        violation_type="comparison_error", count=7, example_text="test"
    )

    result = engine.generate_constraint(pattern)

    assert isinstance(result, ComparisonDirectionConstraint)
    assert result.name == "comparison_direction_constraint"


def test_generate_constraint_unknown_type_returns_none() -> None:
    """generate_constraint returns None for unrecognised violation types.

    Spec: REQ-LEARN-040-3
    """
    sm = _make_session_memory({})
    engine = ConstraintAdditionEngine(sm)
    pattern = ConstraintPattern(
        violation_type="totally_unknown_error", count=10, example_text="test"
    )

    result = engine.generate_constraint(pattern)

    assert result is None


# ---------------------------------------------------------------------------
# inject_into_pipeline tests
# ---------------------------------------------------------------------------


def test_inject_into_pipeline_adds_new_constraints() -> None:
    """inject_into_pipeline appends new constraints and returns the injected count.

    Spec: REQ-LEARN-040-4
    """
    sm = _make_session_memory({"carry_error": 5, "sign_error": 4})
    engine = ConstraintAdditionEngine(sm, min_count=3)
    pipeline = _StubPipeline()

    n = engine.inject_into_pipeline(pipeline)

    assert n == 2
    names = {c.name for c in pipeline.active_constraints}
    assert "carry_check_constraint" in names
    assert "sign_check_constraint" in names


def test_inject_into_pipeline_avoids_duplicates() -> None:
    """inject_into_pipeline does not add a constraint already present by name.

    Spec: REQ-LEARN-040-4, SCENARIO-LEARN-081
    """
    sm = _make_session_memory({"carry_error": 5})
    engine = ConstraintAdditionEngine(sm, min_count=3)

    # Pre-populate pipeline with the constraint that would be injected.
    existing = CarryCheckConstraint()
    pipeline = _StubPipeline(existing=[existing])
    assert len(pipeline.active_constraints) == 1

    n = engine.inject_into_pipeline(pipeline)

    assert n == 0  # nothing new was injected
    assert len(pipeline.active_constraints) == 1  # still only one


def test_inject_into_pipeline_no_active_constraints_attribute() -> None:
    """inject_into_pipeline returns 0 gracefully when pipeline lacks active_constraints.

    Spec: REQ-LEARN-040-4
    """
    sm = _make_session_memory({"carry_error": 5})
    engine = ConstraintAdditionEngine(sm, min_count=3)
    pipeline = _NoPipeline()

    n = engine.inject_into_pipeline(pipeline)

    assert n == 0


def test_constraint_addition_engine_empty_memory() -> None:
    """scan_for_patterns returns [] and inject returns 0 when memory is empty.

    Spec: REQ-LEARN-040-2
    """
    sm = SessionMemory(storage_dir="/tmp/exp761_test_empty", model_id="empty")
    # _violations_by_type is not set — simulates a fresh SessionMemory.
    engine = ConstraintAdditionEngine(sm, min_count=3)
    pipeline = _StubPipeline()

    patterns = engine.scan_for_patterns()
    n = engine.inject_into_pipeline(pipeline)

    assert patterns == []
    assert n == 0


# ---------------------------------------------------------------------------
# Full 10-session integration test
# ---------------------------------------------------------------------------


def test_ten_session_precision_non_decreasing() -> None:
    """10-session synthetic run produces monotonic_non_decreasing=True and constraints > 0.

    This exercises the full experiment logic end-to-end on CPU in a few seconds.
    Spec: REQ-LEARN-041
    """
    # Import the experiment helpers directly.
    from scripts.experiment_761_tier1_constraint_addition import (
        SyntheticPipeline,
        make_questions,
        run_session,
    )

    N_SESSIONS = 10
    N_QUESTIONS = 50

    import tempfile

    storage_dir = tempfile.mkdtemp(prefix="exp761_integration_")
    session_memory = SessionMemory(storage_dir=storage_dir, model_id="test")
    pipeline = SyntheticPipeline()
    engine = ConstraintAdditionEngine(session_memory, min_count=3)

    precision_per_session: list[float] = []
    total_injected = 0

    for sid in range(N_SESSIONS):
        questions = make_questions(sid, N_QUESTIONS)
        metrics = run_session(sid, pipeline, session_memory, questions)
        precision_per_session.append(metrics["precision"])
        total_injected += engine.inject_into_pipeline(pipeline)

    precision_s1 = precision_per_session[0]
    precision_s10 = precision_per_session[-1]
    monotonic = all(
        precision_per_session[i] <= precision_per_session[i + 1]
        for i in range(len(precision_per_session) - 1)
    )

    assert total_injected > 0, "Expected at least one constraint to be injected"
    assert precision_s10 >= precision_s1, (
        f"Precision regressed: s1={precision_s1:.4f} s10={precision_s10:.4f}"
    )
    assert monotonic, f"Precision not monotonic: {precision_per_session}"
