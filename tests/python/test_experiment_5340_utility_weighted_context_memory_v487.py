"""Tests for Exp5340 utility-weighted context memory.

Spec refs: REQ-LEARN-5340, SCENARIO-LEARN-5340-UTILITY,
SCENARIO-LEARN-5340-POLICY, SCENARIO-LEARN-5340-NOOP.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5340_utility_weighted_context_memory_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5340_spec_declares_utility_contract() -> None:
    """REQ-LEARN-5340: OpenSpec anchors utility fields, controls, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5340") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5340",
        "SCENARIO-LEARN-5340-UTILITY",
        "SCENARIO-LEARN-5340-POLICY",
        "SCENARIO-LEARN-5340-NOOP",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.Q_VALUE_TABLE_RELATIVE_PATH),
        "positive, stale, poisoned, irrelevant, and shuffled/no-op",
        "always-full",
        "transition-only",
        "utility-weighted retrieval",
        "shuffled-utility/no-op",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_learn_5340_utility_updates_from_deterministic_feedback() -> None:
    """SCENARIO-LEARN-5340-UTILITY: feedback rows update operation Q-values."""

    feedback = exp.build_utility_feedback_panel()
    table = exp.learn_utility_values(feedback)

    assert {row.feedback_label for row in feedback} == set(exp.FEEDBACK_LABELS)
    assert {row.action for row in feedback}.issuperset(exp.UTILITY_OPERATIONS)
    assert table["utility_update_count"] == len(feedback)
    assert set(table["operation_q_values"]) == set(exp.UTILITY_OPERATIONS)
    assert table["no_weight_mutation"] is True
    assert all(
        update["feedback_label"] in exp.FEEDBACK_LABELS
        and update["source_case_id"]
        and update["model_weights_mutated"] is False
        for update in table["utility_updates"]
    )
    assert table["operation_q_values"]["retrieve"]["feedback_counts"]["stale"] == 2
    assert table["operation_q_values"]["rollback"]["q_value"] > 0.0


def test_scenario_learn_5340_policy_comparison_preserves_quality() -> None:
    """SCENARIO-LEARN-5340-POLICY: utility retrieval matches full quality."""

    q_table = exp.learn_utility_values(exp.build_utility_feedback_panel())
    evaluation = exp.evaluate_utility_memory(q_table)
    metrics = evaluation["policy_metrics"]

    assert evaluation["all_policies_run"] is True
    assert set(metrics) == set(exp.POLICY_ARMS)
    assert exp._same_case_ids(evaluation["policy_rows"]) is True
    assert metrics[exp.UTILITY_WEIGHTED_POLICY]["final_quality"] == metrics[exp.ALWAYS_FULL_POLICY]["final_quality"]
    assert metrics[exp.UTILITY_WEIGHTED_POLICY]["final_quality"] > metrics[exp.TRANSITION_ONLY_POLICY]["final_quality"]
    assert evaluation["quality_delta_vs_always_full"] == 0.0
    assert evaluation["verifier_calls_avoided"] > 0
    assert evaluation["unsafe_false_accepts"] == 0
    assert evaluation["rollback_events"] > 0
    assert evaluation["no_op_control_delta"] <= 0.0
    assert evaluation["utility_memory_ready"] is True


def test_scenario_learn_5340_noop_control_blocks_spurious_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5340-NOOP: positive no-op deltas block readiness."""

    original = exp.evaluate_utility_memory

    def noisy_noop(q_table: exp.JsonDict) -> exp.JsonDict:
        evaluation = original(q_table)
        evaluation["no_op_control_delta"] = 0.125
        evaluation["utility_memory_ready"] = False
        return evaluation

    monkeypatch.setattr(exp, "evaluate_utility_memory", noisy_noop)
    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit utility memory", "outcome": "passed"}],
    )

    assert artifact["utility_memory_ready"] is False
    assert artifact["status"]["value"] == "blocked_fixture_certificate_or_tests"
    assert artifact["no_op_control_delta"] == 0.125
    assert artifact["honest_verdict"]["value"].startswith("blocked_utility_memory_not_ready:")
    exp.validate_artifact(artifact)


def test_req_learn_5340_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5340-5: run() writes result and Q-value table artifacts."""

    tests_run = [{"command": "unit utility weighted context memory", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    q_table_path = tmp_path / exp.Q_VALUE_TABLE_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        q_table_path=q_table_path,
        tests_run=tests_run,
    )
    written_q_table = json.loads(q_table_path.read_text(encoding="utf-8"))

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert written_q_table == artifact["q_value_table"]
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "utility_memory_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["utility_update_count"] == len(exp.build_utility_feedback_panel())
    assert artifact["q_value_table_path"]["value"] == str(exp.Q_VALUE_TABLE_RELATIVE_PATH)
    assert artifact["quality_delta_vs_always_full"] == 0.0
    assert artifact["verifier_calls_avoided"] > 0
    assert artifact["no_op_control_delta"] <= 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["rollback_events"] > 0
    assert artifact["utility_memory_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5340_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5340: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["utility_memory_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_learn_5340_blocked_artifact_when_certificate_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5340-1/5: failed certificate gates produce blocked artifacts."""

    def blocked_certificate(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "anytime_certificate_gate_ready": False,
            "no_weight_mutation": True,
            "failed_gates": ["anytime_certificate_gate_ready"],
            "all_passed": False,
            "source_honest_verdict": "blocked_certificate",
        }

    monkeypatch.setattr(exp, "confirm_certificate_gate", blocked_certificate)

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "blocked certificate unit", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_fixture_certificate_or_tests"
    assert artifact["honest_verdict"]["value"].startswith("blocked_utility_memory_not_ready:")
    assert artifact["utility_memory_ready"] is False
    assert artifact["utility_update_count"] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5340_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5340-5: artifact validation rejects gate and scalar drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit utility memory", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_policy_rollout"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_learning = deepcopy(artifact)
    bad_learning["continuous_self_learning_target"] = {"value": True}
    with pytest.raises(ValueError, match="continuous_self_learning_target"):
        exp.validate_artifact(bad_learning)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_count = deepcopy(artifact)
    bad_count["utility_update_count"] = True
    with pytest.raises(ValueError, match="utility_update_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["no_op_control_delta"] = {"value": 0.0}
    with pytest.raises(ValueError, match="no_op_control_delta"):
        exp.validate_artifact(bad_numeric)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["utility_memory_ready"] = "yes"
    with pytest.raises(ValueError, match="utility_memory_ready"):
        exp.validate_artifact(bad_ready)
