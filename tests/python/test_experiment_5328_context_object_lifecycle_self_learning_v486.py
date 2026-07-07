"""Tests for Exp5328 context-object lifecycle fixture.

Spec refs: REQ-LEARN-5328, SCENARIO-LEARN-5328, SCENARIO-LEARN-5329.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5328_context_object_lifecycle_self_learning_v486 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5328_spec_declares_lifecycle_contract() -> None:
    """REQ-LEARN-5328: OpenSpec anchors fields, cases, and action vocabulary."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5328") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5328",
        "SCENARIO-LEARN-5328",
        "SCENARIO-LEARN-5329",
        str(exp.RESULT_RELATIVE_PATH),
        "stable context object IDs",
        "recoverable sidecars",
        "ghost memory",
        "stale retrieval",
        "unsafe prune",
        "bank-maintenance",
        "retrieval",
        "answer-time",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for action in exp.LIFECYCLE_ACTION_SET:
        assert action in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5328_fixture_declares_stable_objects_and_actions() -> None:
    """REQ-LEARN-5328-1/2/3: fixture covers object schema and action set."""

    cases = exp.build_lifecycle_fixture()
    actions = [case.action for case in cases]
    object_ids = exp.stable_context_object_ids(cases)
    action_counts = exp.lifecycle_action_counts(cases)

    assert set(exp.LIFECYCLE_ACTION_SET).issubset(action_counts)
    assert action_counts["prune"] == 1
    assert all(action in actions for action in exp.LIFECYCLE_ACTION_SET)
    assert len(object_ids) == exp.context_object_count(cases)
    assert len(object_ids) >= 8
    assert len(object_ids) == len(set(object_ids))

    failure_modes = {case.failure_mode for case in cases if case.failure_mode}
    assert failure_modes == {
        "answer_corruption",
        "answer_stale_context",
        "corruption",
        "ghost_memory",
        "mask_leakage",
        "omission",
        "stale_retrieval",
        "unsafe_prune",
    }

    for case in cases:
        for obj in exp.objects_in_case(case):
            assert obj["object_id"] in object_ids
            assert obj["object_type"] in exp.OBJECT_TYPES
            assert obj["current_label"] in exp.CURRENT_LABELS
            assert isinstance(obj["historical_labels"], tuple)
            assert obj["transition_label"] in exp.TRANSITION_LABELS
            assert obj["recoverable_sidecar"]["recoverable"] is True


def test_scenario_learn_5328_safe_commits_recover_sidecars() -> None:
    """SCENARIO-LEARN-5328: safe commits, archive, retrieve, and rollback work."""

    evaluation = exp.evaluate_lifecycle_fixture(exp.build_lifecycle_fixture())
    rows = {row["case_id"]: row for row in evaluation["lifecycle_rows"]}

    for case_id in (
        "life-create-runtime",
        "life-revise-runtime",
        "life-fold-runtime-rubric",
        "life-mask-secret",
        "life-archive-policy",
        "life-commit-patch",
        "life-rollback-corrupt-patch",
    ):
        assert rows[case_id]["accepted"] is True
        assert rows[case_id]["detected_failure"] is False
        assert rows[case_id]["transition_verifier_reused"] is True

    assert rows["life-retrieve-archive-recover"]["accepted"] is True
    assert rows["life-retrieve-archive-recover"]["recovered_from_sidecar"] is True
    assert rows["life-rollback-corrupt-patch"]["rollback_success"] is True
    assert rows["life-mask-secret"]["sidecar_preserved"] is True
    assert rows["life-archive-policy"]["sidecar_preserved"] is True
    assert evaluation["rollback_success_rate"] == 1.0
    assert evaluation["no_weight_mutation"] is True


def test_scenario_learn_5329_unsafe_bank_actions_reject_before_commit() -> None:
    """SCENARIO-LEARN-5329: unsafe bank actions reject with unchanged state."""

    evaluation = exp.evaluate_lifecycle_fixture(exp.build_lifecycle_fixture())
    rows = {row["case_id"]: row for row in evaluation["lifecycle_rows"]}

    for case_id, family in (
        ("life-ghost-memory", "bank"),
        ("life-omission-sensor-rule", "bank"),
        ("life-corrupt-rubric", "bank"),
        ("life-unsafe-prune-runtime", "bank"),
    ):
        assert rows[case_id]["failure_family"] == family
        assert rows[case_id]["accepted"] is False
        assert rows[case_id]["detected_failure"] is True
        assert rows[case_id]["committed_state_changed"] is False
        assert rows[case_id]["committed_state"] == rows[case_id]["prior_state"]
        assert rows[case_id]["rejection_reasons"]

    assert evaluation["bank_failure_detection_rate"] == 1.0
    assert evaluation["failure_counts"]["bank"] == {"detected": 4, "total": 4}


def test_req_learn_5328_scores_failure_families_separately() -> None:
    """REQ-LEARN-5328-4: bank, retrieval, and answer failures score apart."""

    evaluation = exp.evaluate_lifecycle_fixture(exp.build_lifecycle_fixture())
    rows = {row["case_id"]: row for row in evaluation["lifecycle_rows"]}

    for case_id in ("life-stale-retrieval", "life-mask-retrieval-leak"):
        assert rows[case_id]["failure_family"] == "retrieval"
        assert rows[case_id]["accepted"] is False
        assert rows[case_id]["detected_failure"] is True
        assert rows[case_id]["answer_context_allowed"] is False

    for case_id in ("life-answer-stale-context", "life-answer-corrupt-context"):
        assert rows[case_id]["failure_family"] == "answer"
        assert rows[case_id]["accepted"] is False
        assert rows[case_id]["detected_failure"] is True
        assert rows[case_id]["answer_context_allowed"] is False

    assert evaluation["retrieval_failure_detection_rate"] == 1.0
    assert evaluation["answer_failure_detection_rate"] == 1.0
    assert evaluation["failure_counts"]["retrieval"] == {"detected": 2, "total": 2}
    assert evaluation["failure_counts"]["answer"] == {"detected": 2, "total": 2}
    assert evaluation["context_lifecycle_fixture_ready"] is True


def test_req_learn_5328_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5328-6: run() writes the required Exp5328 artifact."""

    tests_run = [{"command": "unit context lifecycle fixture", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "ready_for_exp5329_exp5330"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert isinstance(artifact["context_object_count"], int)
    assert artifact["context_object_count"] == exp.context_object_count(
        exp.build_lifecycle_fixture()
    )
    assert artifact["lifecycle_action_set"]["value"] == list(exp.LIFECYCLE_ACTION_SET)
    assert artifact["bank_failure_detection_rate"] == 1.0
    assert artifact["retrieval_failure_detection_rate"] == 1.0
    assert artifact["answer_failure_detection_rate"] == 1.0
    assert artifact["rollback_success_rate"] == 1.0
    assert artifact["context_lifecycle_fixture_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5328_blocked_artifact_when_tests_not_recorded() -> None:
    """REQ-LEARN-5328-6: downstream ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_tests_or_fixture_not_ready"
    assert artifact["honest_verdict"]["value"].startswith("blocked_tests_or_fixture_not_ready:")
    assert artifact["context_lifecycle_fixture_ready"] is False
    assert artifact["tests_run"]["value"] == []
    exp.validate_artifact(artifact)


def test_req_learn_5328_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5328: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["context_lifecycle_fixture_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    exp.validate_artifact(result)


def test_req_learn_5328_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5328-6: artifact validation rejects downstream gate drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit context lifecycle fixture", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_memory_policy_rollout_no_llm"
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
    bad_count["context_object_count"] = 3.0
    with pytest.raises(ValueError, match="context_object_count"):
        exp.validate_artifact(bad_count)

    bad_actions = deepcopy(artifact)
    bad_actions["lifecycle_action_set"]["value"] = ["create"]
    with pytest.raises(ValueError, match="lifecycle_action_set"):
        exp.validate_artifact(bad_actions)

    bad_rate = deepcopy(artifact)
    bad_rate["bank_failure_detection_rate"] = 0.75
    with pytest.raises(ValueError, match="bank_failure_detection_rate"):
        exp.validate_artifact(bad_rate)

    bad_rate_type = deepcopy(artifact)
    bad_rate_type["retrieval_failure_detection_rate"] = "1.0"
    with pytest.raises(ValueError, match="retrieval_failure_detection_rate"):
        exp.validate_artifact(bad_rate_type)

    bad_ready = deepcopy(artifact)
    bad_ready["context_lifecycle_fixture_ready"] = {"value": True}
    with pytest.raises(ValueError, match="context_lifecycle_fixture_ready"):
        exp.validate_artifact(bad_ready)

    bad_ready_tests = deepcopy(artifact)
    bad_ready_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_ready_tests)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5328_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5328: helper edge paths stay deterministic and explicit."""

    missing_retrieval = exp.LifecycleCase(
        case_id="edge-missing-retrieval",
        action="retrieve",
        prior_state={},
        proposed_state={},
        expected_state={},
        retrieval_object_ids=("ctx.missing",),
    )
    missing_answer = exp.LifecycleCase(
        case_id="edge-missing-answer",
        action="retrieve",
        prior_state={},
        proposed_state={},
        expected_state={},
        answer_context_object_ids=("ctx.missing",),
    )
    missing_recover = exp.LifecycleCase(
        case_id="edge-missing-recover",
        action="retrieve",
        prior_state={},
        proposed_state={},
        expected_state={},
        recover_object_id="ctx.missing",
    )

    assert exp._retrieval_context_safe(missing_retrieval) is False
    assert exp._answer_context_safe(missing_answer) is False
    assert exp._recovered_from_sidecar(missing_recover) is False
    assert exp._rate(1, 0) == 0.0
    assert exp._json_ready(Path("local/context.json")) == "local/context.json"
