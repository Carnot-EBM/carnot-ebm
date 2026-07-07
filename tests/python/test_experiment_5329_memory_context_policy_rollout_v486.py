"""Tests for Exp5329 memory/context policy rollout.

Spec refs: REQ-LEARN-5329, SCENARIO-LEARN-5329-POLICY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5329_memory_context_policy_rollout_v486 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5329_spec_declares_policy_rollout_contract() -> None:
    """REQ-LEARN-5329: OpenSpec anchors policy arms, metrics, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5329") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5329",
        "SCENARIO-LEARN-5329-POLICY",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5328 context-object lifecycle fixture",
        "bank-maintenance failure metrics",
        "retrieval failure metrics",
        "answer-time failure metrics",
        "rollback metrics",
        "recoverability metrics",
        "always-full verification",
        "transition-only verifier",
        "context-lifecycle policy with rollback",
        "final quality",
        "verifier calls",
        "unsafe false accepts",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5329_fixture_gate_confirms_lifecycle_metrics() -> None:
    """REQ-LEARN-5329-1: Exp5328 exposes the required fixture metrics."""

    gate = exp.confirm_fixture_gate(root=REPO)

    assert gate["all_passed"] is True
    assert gate["context_lifecycle_fixture_ready"] is True
    assert gate["bank_failure_detection_rate_present"] is True
    assert gate["retrieval_failure_detection_rate_present"] is True
    assert gate["answer_failure_detection_rate_present"] is True
    assert gate["rollback_success_rate_present"] is True
    assert gate["recoverability_metrics_present"] is True
    assert gate["no_weight_mutation"] is True
    assert gate["failed_gates"] == []

    blocked = exp.confirm_fixture_gate(
        artifact={
            "context_lifecycle_fixture_ready": True,
            "bank_failure_detection_rate": 1.0,
            "retrieval_failure_detection_rate": 1.0,
            "answer_failure_detection_rate": 1.0,
            "rollback_success_rate": 1.0,
            "no_weight_mutation": False,
            "lifecycle_rows": [{"recovered_from_sidecar": True, "sidecar_preserved": True}],
        }
    )

    assert blocked["all_passed"] is False
    assert blocked["failed_gates"] == ["no_weight_mutation"]


def test_scenario_learn_5329_policy_rollout_runs_same_cases() -> None:
    """SCENARIO-LEARN-5329-POLICY: lifecycle policy matches quality and saves calls."""

    evaluation = exp.evaluate_policy_rollout(exp.build_policy_panel())

    assert evaluation["all_variants_ran"] is True
    assert evaluation["policy_rollout_ready"] is True
    assert evaluation["quality_delta_vs_always_full"] == 0.0
    assert evaluation["verifier_calls_avoided"] == 3
    assert evaluation["bank_failure_delta"] == 0.0
    assert evaluation["retrieval_failure_delta"] == 0.0
    assert evaluation["answer_failure_delta"] == 0.0
    assert evaluation["unsafe_false_accepts"] == 0
    assert evaluation["rollback_events"] == 2
    assert evaluation["recoveries"] == 2

    metrics = evaluation["policy_metrics"]
    assert set(metrics) == set(exp.POLICY_ARMS)
    case_ids = {
        tuple(row["case_id"] for row in evaluation["policy_rows"][policy])
        for policy in exp.POLICY_ARMS
    }
    assert len(case_ids) == 1

    always = metrics["always_full_verification"]
    transition = metrics["transition_only_verifier"]
    lifecycle = metrics["context_lifecycle_policy_with_rollback"]

    assert always["final_quality"] == 1.0
    assert lifecycle["final_quality"] == 1.0
    assert transition["final_quality"] == 13 / 16
    assert always["verifier_calls"] == 16
    assert lifecycle["verifier_calls"] == 13
    assert transition["verifier_calls"] == 13
    assert transition["retrieval_failure_rate"] == 1.0
    assert transition["answer_failure_rate"] == 0.5
    assert lifecycle["retrieval_failure_rate"] == 0.0
    assert lifecycle["answer_failure_rate"] == 0.0
    assert lifecycle["unsafe_false_accepts"] == 0
    assert lifecycle["rollback_events"] == 2
    assert lifecycle["recoveries"] == 2


def test_req_learn_5329_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5329-5: run() writes the required terminal artifact."""

    tests_run = [{"command": "unit memory context policy rollout", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "policy_rollout_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["quality_delta_vs_always_full"] == 0.0
    assert artifact["verifier_calls_avoided"] == 3
    assert artifact["bank_failure_delta"] == 0.0
    assert artifact["retrieval_failure_delta"] == 0.0
    assert artifact["answer_failure_delta"] == 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["rollback_events"] == 2
    assert artifact["recoveries"] == 2
    assert artifact["policy_rollout_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5329_blocked_artifact_when_fixture_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5329-1/5: failed fixture gates produce blocked artifacts."""

    def blocked_gate(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "context_lifecycle_fixture_ready": False,
            "bank_failure_detection_rate_present": True,
            "retrieval_failure_detection_rate_present": True,
            "answer_failure_detection_rate_present": True,
            "rollback_success_rate_present": True,
            "recoverability_metrics_present": True,
            "no_weight_mutation": True,
            "failed_gates": ["context_lifecycle_fixture_ready"],
            "all_passed": False,
            "source_honest_verdict": "blocked_fixture",
        }

    monkeypatch.setattr(exp, "confirm_fixture_gate", blocked_gate)

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "blocked fixture unit", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_fixture_gate_or_tests"
    assert artifact["honest_verdict"]["value"].startswith("blocked_fixture_gate_not_ready:")
    assert artifact["policy_rollout_ready"] is False
    assert artifact["policy_metrics"]["always_full_verification"]["n"] == 0
    assert artifact["verifier_calls_avoided"] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5329_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5329: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["policy_rollout_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    exp.validate_artifact(result)


def test_req_learn_5329_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5329-5: artifact validation rejects gate and schema drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit memory context policy rollout", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_lifecycle_fixture"
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

    bad_numeric = deepcopy(artifact)
    bad_numeric["quality_delta_vs_always_full"] = "0.0"
    with pytest.raises(ValueError, match="quality_delta_vs_always_full"):
        exp.validate_artifact(bad_numeric)

    bad_integer = deepcopy(artifact)
    bad_integer["verifier_calls_avoided"] = 3.0
    with pytest.raises(ValueError, match="verifier_calls_avoided"):
        exp.validate_artifact(bad_integer)

    bad_ready = deepcopy(artifact)
    bad_ready["policy_rollout_ready"] = {"value": True}
    with pytest.raises(ValueError, match="policy_rollout_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5329_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5329: helper edge paths stay deterministic and explicit."""

    panel = exp.build_policy_panel()
    empty = exp._blocked_evaluation()
    missing_recover_id = exp.exp5328.LifecycleCase(
        case_id="edge-no-recover-id",
        action="retrieve",
        prior_state={},
        proposed_state={},
        expected_state={},
    )
    missing_recover_object = exp.exp5328.LifecycleCase(
        case_id="edge-missing-recover-object",
        action="retrieve",
        prior_state={},
        proposed_state={},
        expected_state={},
        recover_object_id="ctx.missing",
    )

    assert exp._same_case_ids({"a": [], "b": []}) is True
    assert exp._same_case_ids({"a": [{"case_id": "x"}], "b": [{"case_id": "y"}]}) is False
    assert exp._rate(1, 0) == 0.0
    assert exp._delta(0.3333333333333, 0.1111111111111) == 0.222222222222
    assert exp._json_ready((Path("x"), {"y": Path("z")})) == ["x", {"y": "z"}]
    assert exp._recoverable_from_sidecar(missing_recover_id) is False
    assert exp._recoverable_from_sidecar(missing_recover_object) is False
    assert exp._recoverability_metrics_present({"rows": []}) is False
    assert empty["policy_rollout_ready"] is False
    assert [case.case_id for case in panel] == [case.case_id for case in exp.build_policy_panel()]
