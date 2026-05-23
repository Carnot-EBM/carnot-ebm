"""Tests for Exp 2933 KAN/KAC per-knot continuous self-learning probe.

Spec: REQ-LEARN-2933,
      SCENARIO-LEARN-2933,
      SCENARIO-LEARN-2933-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.eval import fr11_kan_cl_per_knot_self_learning_v1 as exp


def test_req_learn_2933_dataset_stream_is_seeded_and_split() -> None:
    """REQ-LEARN-2933-1: seed 2933 produces deterministic train/holdout splits."""

    first = exp.build_constraint_stream(random_seed=2933)
    second = exp.build_constraint_stream(random_seed=2933)

    assert first.manifest() == second.manifest()
    assert first.manifest()["random_seed"] == 2933
    assert first.manifest()["train_example_count"] == 72
    assert first.manifest()["holdout_example_count"] == 48
    assert first.manifest()["rbf_center_count"] == 12
    assert [rule.constraint_id for rule in first.rules] == [
        "arithmetic_bounds",
        "code_shape",
        "logic_coherence",
    ]
    assert len(first.train_by_constraint["arithmetic_bounds"]) == 24
    assert len(first.holdout_by_constraint["logic_coherence"]) == 16
    assert [
        exp.exact_verifier(row, first.rule_by_id[row.constraint_id]) for row in first.all_rows()
    ][:6] == [row.label for row in first.all_rows()[:6]]


def test_req_learn_2933_rbf_importance_updates_local_centers() -> None:
    """REQ-LEARN-2933-3: exact verifier labels update local RBF importance only."""

    stream = exp.build_constraint_stream(random_seed=2933)
    memory = exp.RBFImportanceMemory(
        centers=stream.centers,
        sigma=0.09,
        learning_rate=1.0,
        active_threshold=0.35,
    )
    rows = stream.train_by_constraint["arithmetic_bounds"][:4]
    before = [memory.predict_proba(row.features) for row in rows]

    memory.update(rows, stream.rule_by_id)
    after = [memory.predict_proba(row.features) for row in rows]

    assert memory.updated_count() > 0
    assert memory.updated_count() < stream.manifest()["rbf_center_count"]
    assert max(after) > max(before)
    assert min(after) < min(before)
    np.testing.assert_allclose(
        memory.importance[memory.importance == 0.0],
        np.zeros_like(memory.importance[memory.importance == 0.0]),
    )


def test_scenario_learn_2933_rbf_importance_beats_replay_without_forgetting(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2933: structural memory improves utility and passes forgetting."""

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 14.25,
        ),
        tests_run=["unit-placeholder"],
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kan_cl_self_learning_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["random_seed"] == 2933
    assert artifact["inference_substrate"] == "local_training_simulation"
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["tests_run"] == ["unit-placeholder"]

    assert artifact["dataset_manifest"]["train_example_count"] == 72
    assert artifact["dataset_manifest"]["holdout_example_count"] == 48
    assert artifact["kan_update_config"]["memory_type"] == "rbf_per_center_importance"
    assert artifact["updated_knot_or_rbf_count"] == 12
    assert artifact["utility_delta_vs_replay_only"] > 0.0
    assert artifact["energy_proxy_delta"] > 0.0
    assert artifact["forgetting_rate"] <= artifact["forgetting_threshold"]
    assert artifact["non_forgetting_passed"] is True

    baselines = artifact["baselines"]
    assert baselines["no_update"]["final_holdout_utility"] == pytest.approx(0.5)
    assert baselines["replay_scheduler_only"]["replay_scheduler_updated"] is True
    assert (
        baselines["kan_rbf_importance_update"]["final_holdout_utility"]
        > baselines["replay_scheduler_only"]["final_holdout_utility"]
    )
    assert (
        baselines["kan_rbf_importance_update"]["mean_post_update_utility"]
        > baselines["kan_rbf_importance_update"]["mean_pre_update_utility"]
    )

    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_learn_2933_guard_blocks_high_forgetting() -> None:
    """SCENARIO-LEARN-2933-GUARD: forgetting above threshold blocks the headline."""

    payload = {
        "honest_verdict": "complete: provisional",
        "kan_cl_self_learning_ready": True,
        "utility_delta_vs_replay_only": 0.2,
        "forgetting_rate": 0.2,
        "forgetting_threshold": 0.05,
    }

    gated = exp.apply_headline_gate(payload)

    assert gated["honest_verdict"] == "complete: kan_rbf_importance_probe_forgetting_guard_failed"
    assert gated["kan_cl_self_learning_ready"] is False
    assert gated["non_forgetting_passed"] is False

    no_utility = exp.apply_headline_gate(
        dict(
            payload,
            utility_delta_vs_replay_only=0.0,
            forgetting_rate=0.0,
            updated_knot_or_rbf_count=1,
        )
    )
    assert no_utility["honest_verdict"] == "complete: kan_rbf_importance_probe_not_ready"
    assert no_utility["kan_cl_self_learning_ready"] is False


def test_req_learn_2933_schema_validation_rejects_incomplete_payload() -> None:
    """REQ-LEARN-2933-4: required artifact fields are checked before delivery."""

    artifact = exp.run_experiment(write=False, tests_run=["unit-placeholder"])
    exp.validate_artifact(artifact)

    incomplete = dict(artifact)
    incomplete.pop("energy_proxy_delta")
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(incomplete)

    regressed = dict(artifact, forgetting_rate=artifact["forgetting_threshold"] + 0.01)
    with pytest.raises(AssertionError, match="headline readiness disagrees"):
        exp.validate_artifact(regressed)
