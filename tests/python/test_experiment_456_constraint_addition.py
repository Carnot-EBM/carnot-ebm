"""Tests for ViolationPattern and ConstraintAdditionFromMemory.

100% coverage of the two new classes added in Exp 456.

Spec: REQ-SELFLEARN-010, REQ-SELFLEARN-011, REQ-SELFLEARN-012,
SCENARIO-SELFLEARN-010, SCENARIO-SELFLEARN-011, SCENARIO-SELFLEARN-012
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock


def _mod():
    return importlib.import_module("carnot.pipeline.constraint_addition")


# ---------------------------------------------------------------------------
# ViolationPattern
# ---------------------------------------------------------------------------


def test_violation_pattern_fields():
    """SCENARIO-SELFLEARN-010: ViolationPattern holds type, count, example_steps."""
    mod = _mod()
    vp = mod.ViolationPattern(type="carry", count=7, example_steps=["step a", "step b"])
    assert vp.type == "carry"
    assert vp.count == 7
    assert vp.example_steps == ["step a", "step b"]


def test_violation_pattern_empty_examples():
    mod = _mod()
    vp = mod.ViolationPattern(type="sign", count=1, example_steps=[])
    assert vp.example_steps == []


# ---------------------------------------------------------------------------
# ConstraintAdditionFromMemory — construction and threshold
# ---------------------------------------------------------------------------


def test_default_threshold_is_five():
    """REQ-SELFLEARN-011: default threshold is 5 when env var is absent."""
    env_backup = os.environ.pop("CARNOT_ADDITION_THRESHOLD", None)
    try:
        mod = _mod()
        cam = mod.ConstraintAdditionFromMemory()
        assert cam._threshold == 5
    finally:
        if env_backup is not None:
            os.environ["CARNOT_ADDITION_THRESHOLD"] = env_backup


def test_env_var_overrides_threshold(monkeypatch):
    """REQ-SELFLEARN-011: CARNOT_ADDITION_THRESHOLD env var sets threshold."""
    monkeypatch.setenv("CARNOT_ADDITION_THRESHOLD", "3")
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    assert cam._threshold == 3


def test_explicit_threshold_overrides_env(monkeypatch):
    """Explicit threshold parameter wins over env var."""
    monkeypatch.setenv("CARNOT_ADDITION_THRESHOLD", "3")
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=10)
    assert cam._threshold == 10


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def test_observe_increments_count():
    """SCENARIO-SELFLEARN-010: observe() increments count for violation_type."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    cam.observe("carry", "step text 1")
    cam.observe("carry", "step text 2")
    assert cam.get_pattern_counts()["carry"] == 2


def test_observe_stores_examples():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    cam.observe("carry", "example step alpha")
    patterns = cam.get_patterns()
    assert len(patterns) == 1
    assert patterns[0].example_steps == ["example step alpha"]


def test_observe_caps_examples_at_five():
    """Examples are capped at 5 regardless of observation count."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    for i in range(10):
        cam.observe("carry", f"step {i}")
    patterns = cam.get_patterns()
    assert len(patterns[0].example_steps) == 5


def test_observe_multiple_types():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    cam.observe("carry", "a")
    cam.observe("sign", "b")
    counts = cam.get_pattern_counts()
    assert counts["carry"] == 1
    assert counts["sign"] == 1


# ---------------------------------------------------------------------------
# get_pattern_counts() returns a copy
# ---------------------------------------------------------------------------


def test_get_pattern_counts_returns_copy():
    """Mutating the returned dict does not affect internal state."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    cam.observe("carry", "step")
    counts = cam.get_pattern_counts()
    counts["carry"] = 999
    assert cam.get_pattern_counts()["carry"] == 1


# ---------------------------------------------------------------------------
# check_and_add() — threshold not met → returns []
# ---------------------------------------------------------------------------


def test_check_and_add_below_threshold_returns_empty():
    """SCENARIO-SELFLEARN-011: 4 observations < threshold=5 → no constraint added."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(4):
        cam.observe("carry", f"step {i}")
    result = cam.check_and_add()
    assert result == []


# ---------------------------------------------------------------------------
# check_and_add() — threshold met → returns constraint name
# ---------------------------------------------------------------------------


def test_check_and_add_at_threshold_returns_carry_constraint():
    """SCENARIO-SELFLEARN-010: 5 carry observations → returns ['carry_check_constraint']."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("carry", f"carry step {i}")
    result = cam.check_and_add()
    assert result == ["carry_check_constraint"]


def test_check_and_add_above_threshold_returns_carry_constraint():
    """50 observations also triggers the constraint (any count >= threshold qualifies)."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(50):
        cam.observe("carry", f"s{i}")
    result = cam.check_and_add()
    assert "carry_check_constraint" in result


def test_check_and_add_sign_constraint():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("sign", f"s{i}")
    result = cam.check_and_add()
    assert result == ["sign_check_constraint"]


def test_check_and_add_unit_constraint():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("unit", f"s{i}")
    result = cam.check_and_add()
    assert result == ["unit_check_constraint"]


def test_check_and_add_comparison_constraint():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("comparison", f"s{i}")
    result = cam.check_and_add()
    assert result == ["comparison_direction_constraint"]


def test_check_and_add_unknown_type_is_ignored():
    """Unknown violation types produce no constraint name."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=2)
    cam.observe("unknown_error_type", "step")
    cam.observe("unknown_error_type", "step2")
    result = cam.check_and_add()
    assert result == []


# ---------------------------------------------------------------------------
# check_and_add() — idempotency
# ---------------------------------------------------------------------------


def test_check_and_add_is_idempotent():
    """SCENARIO-SELFLEARN-011: duplicate addition is not made on second call."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    first = cam.check_and_add()
    second = cam.check_and_add()
    assert first == ["carry_check_constraint"]
    assert second == []


def test_check_and_add_idempotent_after_more_observations():
    """Even after more observations, already-added constraints are not re-added."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    cam.check_and_add()
    for i in range(10):
        cam.observe("carry", f"extra{i}")
    second = cam.check_and_add()
    assert "carry_check_constraint" not in second


# ---------------------------------------------------------------------------
# check_and_add() — pipeline integration
# ---------------------------------------------------------------------------


def test_check_and_add_with_pipeline_calls_observe_pattern():
    """When a pipeline with template_library is provided, observe_pattern is called."""
    mod = _mod()
    mock_lib = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.template_library = mock_lib

    cam = mod.ConstraintAdditionFromMemory(threshold=5, pipeline=mock_pipeline)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    result = cam.check_and_add()

    assert "carry_check_constraint" in result
    mock_lib.observe_pattern.assert_called_once_with("carry", "exp456_session2", count=5)


def test_check_and_add_pipeline_override_takes_precedence():
    """Pipeline passed to check_and_add() overrides the constructor pipeline."""
    mod = _mod()
    constructor_lib = MagicMock()
    constructor_pipeline = MagicMock()
    constructor_pipeline.template_library = constructor_lib

    override_lib = MagicMock()
    override_pipeline = MagicMock()
    override_pipeline.template_library = override_lib

    cam = mod.ConstraintAdditionFromMemory(threshold=5, pipeline=constructor_pipeline)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    cam.check_and_add(pipeline=override_pipeline)

    override_lib.observe_pattern.assert_called_once()
    constructor_lib.observe_pattern.assert_not_called()


def test_check_and_add_pipeline_without_template_library():
    """Pipeline without template_library does not raise."""
    mod = _mod()
    mock_pipeline = MagicMock(spec=[])  # no attributes
    cam = mod.ConstraintAdditionFromMemory(threshold=5, pipeline=mock_pipeline)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    result = cam.check_and_add()
    assert "carry_check_constraint" in result


def test_check_and_add_none_pipeline():
    """check_and_add(pipeline=None) with None constructor pipeline is safe."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5, pipeline=None)
    for i in range(5):
        cam.observe("carry", f"s{i}")
    result = cam.check_and_add(pipeline=None)
    assert "carry_check_constraint" in result


# ---------------------------------------------------------------------------
# get_patterns()
# ---------------------------------------------------------------------------


def test_get_patterns_returns_sorted_list():
    """get_patterns() returns ViolationPattern objects sorted by type."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    cam.observe("unit", "u1")
    cam.observe("carry", "c1")
    cam.observe("sign", "s1")
    patterns = cam.get_patterns()
    types = [p.type for p in patterns]
    assert types == sorted(types)


def test_get_patterns_count_matches_observations():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    for _ in range(7):
        cam.observe("carry", "step")
    patterns = cam.get_patterns()
    assert patterns[0].count == 7


def test_get_patterns_empty_when_no_observations():
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory()
    assert cam.get_patterns() == []


# ---------------------------------------------------------------------------
# Multiple violation types simultaneously qualifying
# ---------------------------------------------------------------------------


def test_multiple_types_qualifying_together():
    """When carry and sign both hit threshold, both constraints are returned."""
    mod = _mod()
    cam = mod.ConstraintAdditionFromMemory(threshold=5)
    for i in range(5):
        cam.observe("carry", f"c{i}")
        cam.observe("sign", f"s{i}")
    result = cam.check_and_add()
    assert "carry_check_constraint" in result
    assert "sign_check_constraint" in result
    assert result == sorted(result)  # result is sorted
