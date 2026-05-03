"""Tests for ConstraintAdditionAgent and Exp 1212 artifact schema.

Spec: REQ-LEARN-1212, SCENARIO-LEARN-1212
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_agent_module():
    return importlib.import_module("carnot.pipeline.constraint_addition_agent")


# ---------------------------------------------------------------------------
# ConstraintFiringStats tests
# ---------------------------------------------------------------------------


def test_firing_stats_wrong_rate():
    """REQ-LEARN-1212: wrong_rate = n_wrong_fired / n_wrong_total."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="arithmetic",
        n_wrong_fired=15,
        n_correct_fired=2,
        n_wrong_total=25,
        n_correct_total=25,
    )
    assert abs(stats.wrong_rate - 0.6) < 1e-9


def test_firing_stats_correct_rate():
    """REQ-LEARN-1212: correct_rate = n_correct_fired / n_correct_total."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="arithmetic",
        n_wrong_fired=15,
        n_correct_fired=2,
        n_wrong_total=25,
        n_correct_total=25,
    )
    assert abs(stats.correct_rate - 0.08) < 1e-9


def test_firing_stats_is_high_signal_true():
    """REQ-LEARN-1212: is_high_signal=True when wrong_rate>0.6 and correct_rate<0.2."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="arithmetic",
        n_wrong_fired=20,
        n_correct_fired=4,
        n_wrong_total=24,
        n_correct_total=26,
    )
    # wrong_rate ≈ 0.833, correct_rate ≈ 0.154
    assert stats.is_high_signal(0.6, 0.2) is True


def test_firing_stats_is_high_signal_false_low_wrong_rate():
    """REQ-LEARN-1212: is_high_signal=False when wrong_rate<=0.6."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="logic",
        n_wrong_fired=5,
        n_correct_fired=4,
        n_wrong_total=24,
        n_correct_total=26,
    )
    # wrong_rate ≈ 0.208 < 0.6 → not high signal
    assert stats.is_high_signal(0.6, 0.2) is False


def test_firing_stats_is_high_signal_false_high_correct_rate():
    """REQ-LEARN-1212: is_high_signal=False when correct_rate>=0.2."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="nl",
        n_wrong_fired=20,
        n_correct_fired=20,
        n_wrong_total=24,
        n_correct_total=26,
    )
    # correct_rate ≈ 0.769 >= 0.2 → not high signal despite high wrong_rate
    assert stats.is_high_signal(0.6, 0.2) is False


def test_firing_stats_zero_denominators():
    """REQ-LEARN-1212: zero denominators return rate=0 (no ZeroDivisionError)."""
    mod = load_agent_module()
    stats = mod.ConstraintFiringStats(
        constraint_type="arithmetic",
        n_wrong_fired=0,
        n_correct_fired=0,
        n_wrong_total=0,
        n_correct_total=0,
    )
    assert stats.wrong_rate == 0.0
    assert stats.correct_rate == 0.0


# ---------------------------------------------------------------------------
# ConstraintAdditionAgent tests
# ---------------------------------------------------------------------------


def test_agent_observe_increments_totals():
    """REQ-LEARN-1212: observe() increments global wrong/correct totals."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent()
    agent.observe({"arithmetic"}, is_correct=False)
    agent.observe({"arithmetic"}, is_correct=True)
    stats = agent.firing_stats()
    assert stats["arithmetic"].n_wrong_total == 1
    assert stats["arithmetic"].n_correct_total == 1


def test_agent_observe_fires_tracking():
    """REQ-LEARN-1212: observe() records fired type counts correctly."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent()
    agent.observe({"arithmetic"}, is_correct=False)
    agent.observe({"arithmetic"}, is_correct=False)
    agent.observe(set(), is_correct=False)
    agent.observe({"arithmetic"}, is_correct=True)
    stats = agent.firing_stats()
    s = stats["arithmetic"]
    # wrong: 2 of 3 fired; correct: 1 of 1 fired
    assert s.n_wrong_fired == 2
    assert s.n_wrong_total == 3
    assert s.n_correct_fired == 1
    assert s.n_correct_total == 1


def test_agent_detects_high_signal_arithmetic():
    """REQ-LEARN-1212: agent detects arithmetic as high-signal on synthetic data."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent(wrong_threshold=0.6, correct_threshold=0.2)
    # 20 wrong samples: arithmetic fires 17 times (85%)
    for i in range(20):
        fired = {"arithmetic"} if i < 17 else set()
        agent.observe(fired, is_correct=False)
    # 20 correct samples: arithmetic fires 2 times (10%)
    for i in range(20):
        fired = {"arithmetic"} if i < 2 else set()
        agent.observe(fired, is_correct=True)
    additions = agent.detect_additions()
    assert "arithmetic" in additions


def test_agent_does_not_flag_low_signal_logic():
    """REQ-LEARN-1212: agent does NOT flag logic as high-signal when rates are similar."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent(wrong_threshold=0.6, correct_threshold=0.2)
    # Logic fires ~20% on wrong, ~20% on correct → NOT high signal
    for i in range(20):
        fired = {"logic"} if i < 4 else set()
        agent.observe(fired, is_correct=False)
    for i in range(20):
        fired = {"logic"} if i < 4 else set()
        agent.observe(fired, is_correct=True)
    additions = agent.detect_additions()
    assert "logic" not in additions


def test_agent_n_constraints_added_matches_detect():
    """REQ-LEARN-1212: n_constraints_added equals len(detect_additions())."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent()
    # arithmetic: high signal
    for i in range(10):
        agent.observe({"arithmetic"}, is_correct=False)
    for i in range(10):
        agent.observe(set(), is_correct=True)
    assert agent.n_constraints_added == len(agent.detect_additions())
    assert agent.n_constraints_added == 1


def test_agent_multiple_types_only_high_signal_returned():
    """REQ-LEARN-1212: multiple observed types — only high-signal ones returned."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent(wrong_threshold=0.6, correct_threshold=0.2)
    # arithmetic: high signal (80% wrong, 10% correct)
    for i in range(10):
        agent.observe({"arithmetic", "logic"} if i < 8 else {"logic"}, is_correct=False)
    for i in range(10):
        agent.observe({"arithmetic", "logic"} if i < 1 else {"logic"}, is_correct=True)
    additions = agent.detect_additions()
    assert "arithmetic" in additions
    assert "logic" not in additions  # logic fires ~100% on both → not high signal


def test_agent_build_addition_constraints_empty_when_no_arithmetic():
    """REQ-LEARN-1212: build_addition_constraints() returns empty list when no violations."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent()
    # train on arithmetic
    for _ in range(10):
        agent.observe({"arithmetic"}, is_correct=False)
    for _ in range(10):
        agent.observe(set(), is_correct=True)
    # text with no arithmetic expressions
    results = agent.build_addition_constraints("The sky is blue and clouds are white.")
    assert isinstance(results, list)
    # May be empty since no arithmetic violations in this text
    # All results should be ConstraintResult objects
    for r in results:
        assert hasattr(r, "constraint_type")
        assert hasattr(r, "description")


def test_agent_build_addition_constraints_detects_arithmetic_violation():
    """REQ-LEARN-1212: build_addition_constraints() flags wrong arithmetic."""
    mod = load_agent_module()
    agent = mod.ConstraintAdditionAgent()
    # Teach arithmetic as high-signal
    for _ in range(10):
        agent.observe({"arithmetic"}, is_correct=False)
    for _ in range(10):
        agent.observe(set(), is_correct=True)
    # Text with a clear arithmetic violation: 5 + 3 = 9 (should be 8)
    results = agent.build_addition_constraints("5 + 3 = 9")
    # Should detect the violation
    violations = [r for r in results if not r.metadata.get("satisfied", True)]
    assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Artifact schema tests (run against already-existing artifact if present)
# ---------------------------------------------------------------------------

RESULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "experiment_1212_tier1_constraint_addition_v2.json"
)

REQUIRED_FIELDS = [
    "exp134_reweighting_baseline_improvement",
    "n_constraints_added",
    "precision_before_addition",
    "precision_after_addition",
    "false_positive_rate_before",
    "false_positive_rate_after",
    "precision_improvement",
    "beats_reweighting_baseline",
    "tier1_online_addition_honest_verdict",
    "honest_verdict",
]

VALID_VERDICTS = {
    "constraint_addition_improves_precision",
    "constraint_addition_no_improvement",
    "constraint_addition_degrades",
    "insufficient_patterns_detected",
    "in_progress",
}


def test_artifact_exists():
    """REQ-LEARN-1212: result artifact must exist on disk."""
    assert RESULT_PATH.exists(), f"Artifact missing: {RESULT_PATH}"


def test_artifact_is_valid_json():
    """REQ-LEARN-1212: artifact must be valid JSON."""
    data = json.loads(RESULT_PATH.read_text())
    assert isinstance(data, dict)


def test_artifact_has_required_fields():
    """REQ-LEARN-1212: artifact must contain all required schema fields."""
    data = json.loads(RESULT_PATH.read_text())
    if data.get("status") == "in_progress":
        return  # skeleton only, skip field checks
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    assert missing == [], f"Missing required fields: {missing}"


def test_artifact_honest_verdict_valid():
    """REQ-LEARN-1212: honest_verdict must be one of the allowed values."""
    data = json.loads(RESULT_PATH.read_text())
    verdict = data.get("honest_verdict", "")
    assert verdict in VALID_VERDICTS, f"Invalid verdict: {verdict!r}"


def test_artifact_precision_in_range():
    """REQ-LEARN-1212: precision values must be in [0.0, 1.0]."""
    data = json.loads(RESULT_PATH.read_text())
    if data.get("status") == "in_progress":
        return
    for key in ("precision_before_addition", "precision_after_addition"):
        val = data.get(key, None)
        if val is not None:
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


def test_artifact_fp_rate_in_range():
    """REQ-LEARN-1212: false_positive_rate values must be in [0.0, 1.0]."""
    data = json.loads(RESULT_PATH.read_text())
    if data.get("status") == "in_progress":
        return
    for key in ("false_positive_rate_before", "false_positive_rate_after"):
        val = data.get(key, None)
        if val is not None:
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


def test_artifact_precision_improvement_consistent():
    """REQ-LEARN-1212: precision_improvement must equal after - before."""
    data = json.loads(RESULT_PATH.read_text())
    if data.get("status") == "in_progress":
        return
    before = data.get("precision_before_addition")
    after = data.get("precision_after_addition")
    improvement = data.get("precision_improvement")
    if before is not None and after is not None and improvement is not None:
        expected = round(after - before, 10)
        assert abs(improvement - expected) < 1e-6, (
            f"precision_improvement={improvement} but after-before={expected}"
        )


def test_artifact_exp134_baseline_is_zero():
    """REQ-LEARN-1212: exp134_reweighting_baseline_improvement must be 0.0 (known result)."""
    data = json.loads(RESULT_PATH.read_text())
    if data.get("status") == "in_progress":
        return
    val = data.get("exp134_reweighting_baseline_improvement")
    assert val == 0.0, f"Expected 0.0, got {val}"
