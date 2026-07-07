"""Tests for Exp5330 SEA anytime certificate gate.

Spec refs: REQ-LEARN-5330, SCENARIO-LEARN-5330-PROMOTE,
SCENARIO-LEARN-5330-REJECT, SCENARIO-LEARN-5330-DEFER,
SCENARIO-LEARN-5330-NOOP, SCENARIO-LEARN-5330-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5330_sea_anytime_certificate_gate_v486 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _decisions_by_id(evaluation: exp.JsonDict) -> dict[str, exp.JsonDict]:
    return {row["policy_id"]: row for row in evaluation["certificate_rows"]}


def test_req_learn_5330_spec_declares_certificate_gate_contract() -> None:
    """REQ-LEARN-5330: OpenSpec anchors fields, policies, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5330") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5330",
        "SCENARIO-LEARN-5330-PROMOTE",
        "SCENARIO-LEARN-5330-REJECT",
        "SCENARIO-LEARN-5330-DEFER",
        "SCENARIO-LEARN-5330-NOOP",
        "SCENARIO-LEARN-5330-ROLLBACK",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5328 context-object lifecycle fixture",
        "no-op/shuffled-control policy",
        "promote",
        "reject",
        "defer",
        "rollback behavior",
        "model weight updates",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5330_loads_fixture_and_defines_candidate_policies() -> None:
    """REQ-LEARN-5330-1/2: fixture and candidate policy vocabulary are explicit."""

    gate = exp.confirm_fixture_gate(root=REPO)
    policies = exp.build_candidate_policies()

    assert gate["all_passed"] is True
    assert gate["context_lifecycle_fixture_ready"] is True
    assert gate["no_weight_mutation"] is True
    assert len(policies) == 5
    assert {policy.decision_mode for policy in policies} == {
        "context_lifecycle",
        "unsafe_pass_through",
        "thin_context_lifecycle",
        "no_op_shuffled_control",
        "later_invalidated_fast_path",
    }
    assert sum(1 for policy in policies if policy.is_control) == 1
    assert all(set(policy.lifecycle_actions).issubset(exp.exp5328.LIFECYCLE_ACTION_SET) for policy in policies)


def test_scenario_learn_5330_promotes_safe_policy() -> None:
    """SCENARIO-LEARN-5330-PROMOTE: safe lifecycle policy clears the certificate."""

    evaluation = exp.evaluate_certificate_gate(
        exp.build_candidate_policies(),
        exp.load_fixture_artifact(root=REPO),
    )
    rows = _decisions_by_id(evaluation)
    promoted = rows["context_lifecycle_certificate_update"]

    assert promoted["decision"] == exp.DECISION_PROMOTE
    assert promoted["certified_delta"] > evaluation["no_op_control_delta"]
    assert promoted["unsafe_accepts"] == 0
    assert promoted["rollback_event"] is False
    assert evaluation["policy_promotions"] == 1
    assert evaluation["unsafe_promotions"] == 0
    assert evaluation["decisions_reproducible"] is True


def test_scenario_learn_5330_rejects_unsafe_and_noop_control() -> None:
    """SCENARIO-LEARN-5330-REJECT/NOOP: unsafe and control policies are not promoted."""

    evaluation = exp.evaluate_certificate_gate(
        exp.build_candidate_policies(),
        exp.load_fixture_artifact(root=REPO),
    )
    rows = _decisions_by_id(evaluation)
    unsafe = rows["unsafe_accept_all_lifecycle_actions"]
    control = rows["no_op_shuffled_control"]

    assert unsafe["decision"] == exp.DECISION_REJECT
    assert unsafe["unsafe_accepts"] > 0
    assert unsafe["rollback_event"] is True
    assert "unsafe_accepts" in unsafe["reasons"]
    assert control["decision"] == exp.DECISION_REJECT
    assert control["is_control"] is True
    assert control["certified_delta"] == evaluation["no_op_control_delta"]
    assert "control_policy_not_promotable" in control["reasons"]
    assert evaluation["policy_rejections"] == 3


def test_scenario_learn_5330_defers_insufficient_evidence() -> None:
    """SCENARIO-LEARN-5330-DEFER: bounded anytime evidence can defer promotion."""

    evaluation = exp.evaluate_certificate_gate(
        exp.build_candidate_policies(),
        exp.load_fixture_artifact(root=REPO),
    )
    deferred = _decisions_by_id(evaluation)["thin_evidence_context_lifecycle"]

    assert deferred["decision"] == exp.DECISION_DEFER
    assert deferred["evidence_count"] < exp.ANYTIME_MIN_EVIDENCE
    assert "insufficient_evidence" in deferred["reasons"]
    assert evaluation["policy_deferrals"] == 1


def test_scenario_learn_5330_records_later_invalidated_rollback() -> None:
    """SCENARIO-LEARN-5330-ROLLBACK: invalidated promotion gets rolled back."""

    evaluation = exp.evaluate_certificate_gate(
        exp.build_candidate_policies(),
        exp.load_fixture_artifact(root=REPO),
    )
    invalidated = _decisions_by_id(evaluation)["retrieval_fast_path_later_invalidated"]

    assert invalidated["preliminary_decision"] == exp.DECISION_PROMOTE
    assert invalidated["decision"] == exp.DECISION_REJECT
    assert invalidated["rollback_event"] is True
    assert "later_invalidated_promotion" in invalidated["reasons"]
    assert evaluation["rollback_events"] == 2


def test_req_learn_5330_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5330-6: run() writes the required terminal artifact."""

    tests_run = [{"command": "unit sea anytime certificate gate", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "anytime_certificate_gate_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["candidate_policy_count"] == 5
    assert artifact["policy_promotions"] == 1
    assert artifact["policy_rejections"] == 3
    assert artifact["policy_deferrals"] == 1
    assert artifact["no_op_control_delta"] == 0.0
    assert artifact["unsafe_promotions"] == 0
    assert artifact["rollback_events"] == 2
    assert artifact["anytime_certificate_gate_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5330_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5330: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["anytime_certificate_gate_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_promotions"] == 0
    exp.validate_artifact(result)


def test_req_learn_5330_blocked_artifact_when_fixture_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5330-1/6: failed fixture gates produce blocked artifacts."""

    def blocked_gate(*, root: Path | str = exp.REPO_ROOT) -> dict[str, object]:
        assert root == REPO
        return {
            "context_lifecycle_fixture_ready": False,
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
    assert artifact["anytime_certificate_gate_ready"] is False
    assert artifact["candidate_policy_count"] == 0
    assert artifact["no_weight_mutation"] is True
    exp.validate_artifact(artifact)


def test_req_learn_5330_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5330-6: artifact validation rejects gate and scalar drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit sea anytime certificate gate", "outcome": "passed"}],
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
    bad_count["candidate_policy_count"] = True
    with pytest.raises(ValueError, match="candidate_policy_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["no_op_control_delta"] = "0.0"
    with pytest.raises(ValueError, match="no_op_control_delta"):
        exp.validate_artifact(bad_numeric)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_promotions"] = 1
    with pytest.raises(ValueError, match="unsafe_promotions"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["anytime_certificate_gate_ready"] = {"value": True}
    with pytest.raises(ValueError, match="anytime_certificate_gate_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5330_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5330: helper paths stay deterministic and explicit."""

    empty = exp._blocked_evaluation()

    assert exp._decision_counts([]) == {
        exp.DECISION_PROMOTE: 0,
        exp.DECISION_REJECT: 0,
        exp.DECISION_DEFER: 0,
    }
    assert exp._rate(1, 0) == 0.0
    assert exp._delta(0.3333333333333, 0.1111111111111) == 0.222222222222
    assert exp._json_ready((Path("x"), {"y": Path("z")})) == ["x", {"y": "z"}]
    assert exp._is_numeric(0.0) is True
    assert exp._is_numeric(True) is False
    assert exp._wrapped_value({"value": "x"}) == "x"
    assert exp._wrapped_value("x") == "x"
    assert exp._anytime_bound(1, 0) == 1.0
    assert empty["anytime_certificate_gate_ready"] is False
    assert empty["certificate_rows"] == []

    unknown = exp.CandidatePolicy(
        policy_id="unknown",
        decision_mode="unknown_mode",
        lifecycle_actions=(),
    )
    with pytest.raises(ValueError, match="unknown policy decision_mode"):
        exp._policy_evidence(unknown, [], [], 0.0)

    tied = exp.CandidatePolicy(
        policy_id="tied",
        decision_mode="context_lifecycle",
        lifecycle_actions=(),
    )
    decision = exp._certify_policy(
        tied,
        {
            "evidence_count": exp.ANYTIME_MIN_EVIDENCE,
            "final_quality": 1.0,
            "baseline_quality": 1.0,
            "observed_delta": 0.0,
            "anytime_bound": 0.0,
            "certified_delta": 0.0,
            "unsafe_accepts": 0,
        },
        control_delta=0.0,
    )
    assert decision["decision"] == exp.DECISION_REJECT
    assert "no_better_than_no_op_control" in decision["reasons"]
