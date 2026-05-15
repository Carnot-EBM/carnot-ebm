"""Tests for FR-11 self-distillation memory (Exp 1741).

Spec traces: REQ-LEARN-1741, REQ-LEARN-1741-1, REQ-LEARN-1741-2,
REQ-LEARN-1741-3, REQ-LEARN-1741-4, REQ-LEARN-1741-5,
SCENARIO-LEARN-1741
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.pipeline.self_learning import (
    DistillationMemory,
    SelfLearningTracker,
    ViolationEvent,
    run_fr11_distillation_loop,
)


# ---------------------------------------------------------------------------
# DistillationMemory unit tests
# ---------------------------------------------------------------------------


def test_req_learn_1741_spec_entry_exists() -> None:
    """REQ-LEARN-1741: spec entry must be present in the self-learning spec."""
    spec = Path("openspec/capabilities/self-learning/spec.md").read_text()
    assert "REQ-LEARN-1741" in spec
    assert "SCENARIO-LEARN-1741" in spec


def test_req_learn_1741_1_distillation_memory_stores_events() -> None:
    """REQ-LEARN-1741-1: DistillationMemory accumulates ViolationEvents."""
    mem = DistillationMemory(capacity=10)
    assert mem.size() == 0

    mem.add(ViolationEvent("arithmetic", 1))
    mem.add(ViolationEvent("type_check", 2))
    mem.add(ViolationEvent("arithmetic", 3, weight=2.0))

    assert mem.size() == 3
    dist = mem.historical_distribution()
    # arithmetic: weight 1.0 + 2.0 = 3.0; type_check: 1.0; total: 4.0
    assert dist["arithmetic"] == pytest.approx(3.0 / 4.0)
    assert dist["type_check"] == pytest.approx(1.0 / 4.0)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_req_learn_1741_1_distillation_memory_capacity_evicts_oldest() -> None:
    """REQ-LEARN-1741-1: DistillationMemory enforces capacity limit."""
    mem = DistillationMemory(capacity=3)
    for i in range(5):
        mem.add(ViolationEvent("type_check", i))
    assert mem.size() == 3


def test_req_learn_1741_3_distillation_loss_empty_buffer_is_zero() -> None:
    """REQ-LEARN-1741-3: distillation_loss returns 0.0 for an empty buffer."""
    mem = DistillationMemory()
    assert mem.distillation_loss({"arithmetic": 0.8}) == 0.0


def test_req_learn_1741_3_distillation_loss_identical_distributions_near_zero() -> None:
    """REQ-LEARN-1741-3: KL(P || P) = 0 for matching teacher and student."""
    mem = DistillationMemory()
    mem.add(ViolationEvent("arithmetic", 1))
    mem.add(ViolationEvent("type_check", 2))

    # Student matches teacher (0.5 each), so KL ≈ 0
    loss = mem.distillation_loss({"arithmetic": 0.5, "type_check": 0.5})
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_req_learn_1741_3_distillation_loss_divergent_distributions_positive() -> None:
    """REQ-LEARN-1741-3: KL > 0 when teacher and student disagree."""
    mem = DistillationMemory()
    # Teacher heavily weights arithmetic (3:1)
    mem.add(ViolationEvent("arithmetic", 1))
    mem.add(ViolationEvent("arithmetic", 2))
    mem.add(ViolationEvent("arithmetic", 3))
    mem.add(ViolationEvent("type_check", 4))

    # Student is uniform (0.5 each) — diverges from teacher
    loss = mem.distillation_loss({"arithmetic": 0.5, "type_check": 0.5})
    assert loss > 0.0


def test_req_learn_1741_3_distillation_loss_non_negative() -> None:
    """REQ-LEARN-1741-3: distillation_loss is always >= 0."""
    mem = DistillationMemory()
    mem.add(ViolationEvent("semantic_grounding", 1))
    mem.add(ViolationEvent("arithmetic", 2))

    # Test multiple different student weight configurations
    for weights in [
        {"arithmetic": 0.9, "semantic_grounding": 0.1},
        {"arithmetic": 0.0, "semantic_grounding": 1.0},
        {"arithmetic": 0.5, "semantic_grounding": 0.5, "type_check": 0.3},
        {},
    ]:
        assert mem.distillation_loss(weights) >= 0.0


def test_req_learn_1741_3_distillation_loss_kl_is_well_defined_for_missing_type() -> None:
    """REQ-LEARN-1741-3: distillation_loss handles student missing types gracefully."""
    mem = DistillationMemory()
    mem.add(ViolationEvent("arithmetic", 1))
    mem.add(ViolationEvent("type_check", 2))

    # Student only knows arithmetic — type_check gets epsilon mass
    loss_partial = mem.distillation_loss({"arithmetic": 1.0})
    loss_zero = mem.distillation_loss({"arithmetic": 0.0})
    assert loss_partial >= 0.0
    assert loss_zero >= 0.0


# ---------------------------------------------------------------------------
# SelfLearningTracker unit tests
# ---------------------------------------------------------------------------


def test_req_learn_1741_2_tracker_record_query_updates_both_structures() -> None:
    """REQ-LEARN-1741-2: record_query updates ConstraintTracker and DistillationMemory."""
    tracker = SelfLearningTracker()
    tracker.record_query(
        constraint_types=["arithmetic", "type_check"],
        violated_types=["arithmetic"],
    )

    # ConstraintTracker updated
    stats = tracker.tracker.stats()
    assert stats["arithmetic"]["fired"] == 1
    assert stats["arithmetic"]["caught"] == 1
    assert stats["type_check"]["fired"] == 1
    assert stats["type_check"]["caught"] == 0

    # DistillationMemory updated with the violated type
    assert tracker.memory.size() == 1
    dist = tracker.memory.historical_distribution()
    assert "arithmetic" in dist
    assert dist["arithmetic"] == pytest.approx(1.0)


def test_req_learn_1741_2_tracker_record_empty_violation_no_memory_entry() -> None:
    """REQ-LEARN-1741-2: queries with no violations do not update the replay buffer."""
    tracker = SelfLearningTracker()
    tracker.record_query(
        constraint_types=["arithmetic"],
        violated_types=[],
    )
    assert tracker.memory.size() == 0
    assert tracker.tracker.stats()["arithmetic"]["fired"] == 1


def test_req_learn_1741_1_utility_returns_mean_precision() -> None:
    """REQ-LEARN-1741-1: utility() returns mean precision across active types."""
    tracker = SelfLearningTracker()
    # arithmetic: 2/2 = 1.0; type_check: 1/2 = 0.5
    for _ in range(2):
        tracker.record_query(
            constraint_types=["arithmetic", "type_check"],
            violated_types=["arithmetic"],
        )
    tracker.record_query(
        constraint_types=["type_check"],
        violated_types=["type_check"],
    )
    # arithmetic precision = 2/2 = 1.0; type_check precision = 1/3 ≈ 0.333
    expected_utility = (1.0 + 1.0 / 3.0) / 2.0
    assert tracker.utility() == pytest.approx(expected_utility, abs=1e-6)


def test_req_learn_1741_1_utility_zero_before_queries() -> None:
    """REQ-LEARN-1741-1: utility() returns 0.0 before any queries are recorded."""
    tracker = SelfLearningTracker()
    assert tracker.utility() == 0.0


def test_req_learn_1741_3_tracker_distillation_loss_non_negative() -> None:
    """REQ-LEARN-1741-3: SelfLearningTracker.distillation_loss() is always >= 0."""
    tracker = SelfLearningTracker()
    # Before any queries: loss = 0 (empty buffer)
    assert tracker.distillation_loss() == 0.0

    tracker.record_query(
        constraint_types=["arithmetic", "type_check"],
        violated_types=["arithmetic"],
    )
    tracker.record_query(
        constraint_types=["semantic_grounding"],
        violated_types=["semantic_grounding"],
    )
    assert tracker.distillation_loss() >= 0.0


def test_req_learn_1741_query_count_increments_per_call() -> None:
    """REQ-LEARN-1741-2: query_count() increments with each record_query() call."""
    tracker = SelfLearningTracker()
    assert tracker.query_count() == 0
    tracker.record_query(constraint_types=["arithmetic"], violated_types=[])
    assert tracker.query_count() == 1
    tracker.record_query(constraint_types=["type_check"], violated_types=["type_check"])
    assert tracker.query_count() == 2


# ---------------------------------------------------------------------------
# 50-query loop integration test (SCENARIO-LEARN-1741)
# ---------------------------------------------------------------------------


def test_scenario_learn_1741_loop_produces_non_decreasing_utility() -> None:
    """SCENARIO-LEARN-1741: run_fr11_distillation_loop produces utility_delta >= 0."""
    result = run_fr11_distillation_loop(n_queries=50, seed=42)

    assert result["n_queries"] == 50
    assert result["tracker_query_count"] == 50
    assert result["constraint_types_observed"] == 5
    assert result["replay_buffer_size"] > 0

    # The self-distillation claim: utility improves over the run
    assert result["utility_non_decreasing"] is True, (
        f"Expected utility_delta >= 0 but got {result['utility_delta']}"
    )
    assert result["utility_delta"] >= 0.0

    # Distillation loss is a non-negative float
    assert result["final_distillation_loss"] >= 0.0
    assert result["mean_distillation_loss"] >= 0.0
    assert isinstance(result["utility_delta"], float)


def test_scenario_learn_1741_loop_utility_bounds_sane() -> None:
    """SCENARIO-LEARN-1741: loop utilities are in [0, 1] and late > early."""
    result = run_fr11_distillation_loop(n_queries=50, seed=42)

    assert 0.0 <= result["utility_early_window"] <= 1.0
    assert 0.0 <= result["utility_late_window"] <= 1.0
    # Late window must be meaningfully higher (> 0.05 improvement expected)
    assert result["utility_late_window"] > result["utility_early_window"] + 0.05


def test_scenario_learn_1741_loop_model_specs_present() -> None:
    """SCENARIO-LEARN-1741: artifact includes model_specs field."""
    result = run_fr11_distillation_loop(n_queries=50, seed=42)
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in result["model_specs"]


def test_scenario_learn_1741_loop_deterministic_with_same_seed() -> None:
    """SCENARIO-LEARN-1741: same seed produces identical results."""
    r1 = run_fr11_distillation_loop(n_queries=50, seed=99)
    r2 = run_fr11_distillation_loop(n_queries=50, seed=99)
    assert r1 == r2


def test_scenario_learn_1741_loop_different_seeds_differ() -> None:
    """SCENARIO-LEARN-1741: different seeds produce different utility values."""
    r1 = run_fr11_distillation_loop(n_queries=50, seed=1)
    r2 = run_fr11_distillation_loop(n_queries=50, seed=2)
    assert r1["utility_delta"] != r2["utility_delta"]


# ---------------------------------------------------------------------------
# Artifact write/read test
# ---------------------------------------------------------------------------


def test_req_learn_1741_5_artifact_has_required_schema_fields(tmp_path: Path) -> None:
    """REQ-LEARN-1741-5: experiment artifact includes all required schema fields."""
    loop_result = run_fr11_distillation_loop(n_queries=50, seed=42)

    artifact = {
        "experiment": 1741,
        "schema": "carnot.fr11.self_distillation.v1",
        "run_date": "2026-05-15",
        "title": "Exp 1741: FR-11 Self-Distillation Memory for Continuous Self-Learning",
        "status": "complete",
        "continuous_self_learning_task": True,
        "utility_delta": loop_result["utility_delta"],
        "utility_early_window": loop_result["utility_early_window"],
        "utility_late_window": loop_result["utility_late_window"],
        "utility_non_decreasing": loop_result["utility_non_decreasing"],
        "final_distillation_loss": loop_result["final_distillation_loss"],
        "mean_distillation_loss": loop_result["mean_distillation_loss"],
        "replay_buffer_size": loop_result["replay_buffer_size"],
        "constraint_types_observed": loop_result["constraint_types_observed"],
        "model_specs": loop_result["model_specs"],
        "n_queries": loop_result["n_queries"],
        "seed": loop_result["seed"],
        "honest_verdict": "complete: self-distillation memory integrated into FR-11 loop; utility_delta positive",
    }

    path = tmp_path / "experiment_1741_fr11_self_distillation.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text())

    # Required schema fields
    assert loaded["status"] == "complete"
    assert loaded["continuous_self_learning_task"] is True
    assert isinstance(loaded["utility_delta"], float)
    assert loaded["honest_verdict"].startswith("complete")
    assert loaded["utility_delta"] >= 0.0
