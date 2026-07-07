"""Tests for Exp5341 bounded context compressor drift monitor.

Spec refs: REQ-LEARN-5341, SCENARIO-LEARN-5341-RECALL,
SCENARIO-LEARN-5341-DRIFT, SCENARIO-LEARN-5341-RECOVERY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5341_bounded_compressor_drift_monitor_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _rows_by_id(evaluation: exp.JsonDict) -> dict[str, exp.JsonDict]:
    return {row["case_id"]: row for row in evaluation["compressor_rows"]}


def test_req_learn_5341_spec_declares_compressor_contract() -> None:
    """REQ-LEARN-5341: OpenSpec anchors fields, budgets, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5341") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5341",
        "SCENARIO-LEARN-5341-RECALL",
        "SCENARIO-LEARN-5341-DRIFT",
        "SCENARIO-LEARN-5341-RECOVERY",
        str(exp.RESULT_RELATIVE_PATH),
        "recalled_not_committed",
        "benign recall",
        "stale recall",
        "poisoned candidate memory",
        "compression omission",
        "over-compression",
        "safe recovery",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5341_cases_reuse_exp5328_context_objects() -> None:
    """REQ-LEARN-5341-1: cases reuse stable object IDs and lifecycle evidence."""

    cases = exp.build_compressor_cases()
    source_ids = set(
        exp.exp5328.stable_context_object_ids(exp.exp5328.build_lifecycle_fixture())
    )

    assert [case.case_id for case in cases] == [
        "compress-benign-recall",
        "compress-stale-recall",
        "compress-poisoned-candidate",
        "compress-omission-drift",
        "compress-over-compression-drift",
        "compress-safe-recovery",
    ]
    assert {case.expected_anomaly for case in cases} == {
        None,
        "stale_recall",
        "poisoned_candidate_memory",
        "compression_omission",
        "over_compression",
    }
    assert all(set(case.source_object_ids).issubset(source_ids) for case in cases)
    assert any(case.rollback_case_id == "life-rollback-corrupt-patch" for case in cases)
    assert {case.lifecycle_action for case in cases}.issuperset(
        {"retrieve", "create", "fold", "rollback"}
    )


def test_scenario_learn_5341_recall_is_not_commitment() -> None:
    """SCENARIO-LEARN-5341-RECALL: recall rows do not mutate persistent state."""

    evaluation = exp.evaluate_compressor_cases(exp.build_compressor_cases())
    rows = _rows_by_id(evaluation)

    for case_id in ("compress-benign-recall", "compress-stale-recall"):
        row = rows[case_id]
        assert row["commit_decision"] == exp.RECALLED_NOT_COMMITTED
        assert row["persistent_state_changed"] is False
        assert row["accepted_commit"] is False

    assert rows["compress-benign-recall"]["detected_anomaly"] is False
    assert rows["compress-stale-recall"]["detected_anomaly"] is True
    assert evaluation["recalled_not_committed_count"] == 2
    assert evaluation["recall_commit_separation_rate"] == 1.0
    assert evaluation["unsafe_commits"] == 0


def test_scenario_learn_5341_drift_stale_and_poison_reject_before_commit() -> None:
    """SCENARIO-LEARN-5341-DRIFT: unsafe candidate rows reject before commit."""

    evaluation = exp.evaluate_compressor_cases(exp.build_compressor_cases())
    rows = _rows_by_id(evaluation)

    for case_id in (
        "compress-stale-recall",
        "compress-poisoned-candidate",
        "compress-omission-drift",
        "compress-over-compression-drift",
    ):
        row = rows[case_id]
        assert row["detected_anomaly"] is True
        assert row["accepted_commit"] is False
        assert row["persistent_state_changed"] is False
        assert row["unsafe_commit"] is False
        assert row["rejection_reasons"]

    assert evaluation["drift_detection_rate"] == 1.0
    assert evaluation["stale_recall_detection_rate"] == 1.0
    assert evaluation["poison_rejection_rate"] == 1.0
    assert evaluation["unsafe_commits"] == 0
    assert evaluation["verifier_call_cost"]["total_verifier_calls"] == len(
        exp.build_compressor_cases()
    )


def test_scenario_learn_5341_safe_recovery_preserves_bounded_state() -> None:
    """SCENARIO-LEARN-5341-RECOVERY: accepted recovery stays within budgets."""

    evaluation = exp.evaluate_compressor_cases(exp.build_compressor_cases())
    recovery = _rows_by_id(evaluation)["compress-safe-recovery"]

    assert recovery["accepted_commit"] is True
    assert recovery["recoverable_from_sidecar"] is True
    assert recovery["rollback_success"] is True
    assert recovery["persistent_state_changed"] is True
    assert recovery["summary_object_count"] <= exp.BOUNDED_STATE_OBJECT_LIMIT
    assert recovery["summary_token_count"] <= exp.BOUNDED_STATE_TOKEN_LIMIT
    assert evaluation["recoverability_rate"] == 1.0
    assert evaluation["compression_budget"]["within_budget"] is True
    assert evaluation["compressor_drift_fixture_ready"] is True

    for summary in evaluation["bounded_state_summaries"]:
        assert len(summary["source_object_ids"]) <= exp.BOUNDED_STATE_OBJECT_LIMIT
        assert summary["token_count"] <= exp.BOUNDED_STATE_TOKEN_LIMIT


def test_req_learn_5341_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5341-6: run() writes the requested terminal artifact."""

    tests_run = [{"command": "unit bounded compressor drift monitor", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "compressor_drift_fixture_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["bounded_state_object_limit"] == exp.BOUNDED_STATE_OBJECT_LIMIT
    assert artifact["recalled_not_committed_count"] == 2
    assert artifact["drift_detection_rate"] == 1.0
    assert artifact["stale_recall_detection_rate"] == 1.0
    assert artifact["poison_rejection_rate"] == 1.0
    assert artifact["recoverability_rate"] == 1.0
    assert artifact["unsafe_commits"] == 0
    assert artifact["compressor_drift_fixture_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5341_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5341: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["compressor_drift_fixture_ready"] is True
    assert result["unsafe_commits"] == 0
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    exp.validate_artifact(result)


def test_req_learn_5341_blocked_artifact_when_certificate_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5341-1/6: failed upstream gates produce blocked artifacts."""

    def blocked_certificate(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "anytime_certificate_gate_ready": False,
            "no_weight_mutation": True,
            "unsafe_promotions_zero": True,
            "failed_gates": ["anytime_certificate_gate_ready"],
            "all_passed": False,
            "source_honest_verdict": "blocked_certificate",
        }

    monkeypatch.setattr(exp, "confirm_certificate_gate", blocked_certificate)

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "blocked compressor unit", "outcome": "passed"}],
    )

    assert artifact["status"]["value"] == "blocked_fixture_certificate_or_tests"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_compressor_drift_not_ready:"
    )
    assert artifact["compressor_drift_fixture_ready"] is False
    assert artifact["unsafe_commits"] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5341_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5341-6: artifact validation rejects gate and scalar drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit bounded compressor", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_utility_learning"
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

    bad_limit = deepcopy(artifact)
    bad_limit["bounded_state_object_limit"] = True
    with pytest.raises(ValueError, match="bounded_state_object_limit"):
        exp.validate_artifact(bad_limit)

    bad_numeric = deepcopy(artifact)
    bad_numeric["drift_detection_rate"] = "1.0"
    with pytest.raises(ValueError, match="drift_detection_rate"):
        exp.validate_artifact(bad_numeric)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_commits"] = 1
    with pytest.raises(ValueError, match="unsafe_commits"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["compressor_drift_fixture_ready"] = "yes"
    with pytest.raises(ValueError, match="compressor_drift_fixture_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5341_blocked_artifact_when_tests_not_recorded() -> None:
    """REQ-LEARN-5341-6: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_fixture_certificate_or_tests"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_compressor_drift_not_ready:"
    )
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["compressor_drift_fixture_ready"] is False
    assert artifact["unsafe_commits"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5341_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5341: helper paths stay deterministic and explicit."""

    blocked = exp._blocked_evaluation()

    assert exp._rate(1, 0) == 0.0
    assert exp._delta(0.3333333333333, 0.1111111111111) == 0.222222
    assert exp._json_ready((Path("x"), {"y": Path("z")})) == ["x", {"y": "z"}]
    assert exp._is_numeric(0.0) is True
    assert exp._is_numeric(True) is False
    assert exp._wrapped_value({"value": "x"}) == "x"
    assert exp._wrapped_value("x") == "x"
    assert exp._token_count("alpha_beta gamma") == 3
    assert blocked["compressor_drift_fixture_ready"] is False
    assert blocked["compressor_rows"] == []

    too_many_objects = exp.CompressorCase(
        case_id="bad-budget",
        lifecycle_action="fold",
        source_case_id="life-fold-runtime-rubric",
        source_object_ids=(
            "ctx.runtime.receipt",
            "ctx.arc.rubric",
            "ctx.folded.runtime_rubric",
            "ctx.archive.policy",
            "ctx.patch.autofix",
        ),
        recalled_object_ids=(),
        candidate_summary="too many objects",
        expected_anomaly=None,
        expected_commit=True,
    )
    row = exp._evaluate_case(too_many_objects, {})
    assert row["accepted_commit"] is False
    assert "object_budget_exceeded" in row["rejection_reasons"]

    too_many_tokens = exp.CompressorCase(
        case_id="bad-token-budget",
        lifecycle_action="commit",
        source_case_id="life-commit-patch",
        source_object_ids=("ctx.patch.autofix",),
        recalled_object_ids=(),
        candidate_summary="one two three four five six seven eight nine ten eleven twelve thirteen",
        expected_anomaly=None,
        expected_commit=True,
        model_weights_mutated=True,
    )
    token_row = exp._evaluate_case(too_many_tokens, {})
    assert token_row["accepted_commit"] is False
    assert "token_budget_exceeded" in token_row["rejection_reasons"]
    assert "model_weight_mutation_forbidden" in token_row["rejection_reasons"]

    missing_sidecar = exp._source_sidecar({}, "ctx.missing")
    assert missing_sidecar["recoverable"] is False
    assert missing_sidecar["sidecar_id"] == "sidecar:ctx.missing:unavailable"
