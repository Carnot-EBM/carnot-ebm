"""Tests for Exp5460 frozen-model governed CSL policy bandit.

Spec refs: REQ-LEARN-5460,
SCENARIO-LEARN-5460-GATES,
SCENARIO-LEARN-5460-ROLLBACK,
SCENARIO-LEARN-5460-CONTROLS,
SCENARIO-LEARN-5460-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5460_csl_policy_bandit_v496 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
RECEIPTS_PATH = REPO / exp.CONFIDENCE_RECEIPTS_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5460_csl_policy_bandit_v496.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5460_csl_policy_bandit_v496.py "
    "-m pytest tests/python/test_experiment_5460_csl_policy_bandit_v496.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
)


def test_req_learn_5460_spec_declares_frozen_policy_contract() -> None:
    """REQ-LEARN-5460: OpenSpec anchors the V496 policy-bandit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5460") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5460",
        "SCENARIO-LEARN-5460-GATES",
        "SCENARIO-LEARN-5460-ROLLBACK",
        "SCENARIO-LEARN-5460-CONTROLS",
        "SCENARIO-LEARN-5460-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.CONFIDENCE_RECEIPTS_RELATIVE_PATH),
        "no-memory, naive-ICL, always-full-context, and governed-memory choices",
        "provenance, replay, verifier, access, and no-weight-mutation gates",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5460_stream_covers_policy_and_baseline_cases() -> None:
    """REQ-LEARN-5460-1/2: stream covers shifts, poison, replay, and cheap controls."""

    evaluation = exp.evaluate_policy_bandit(root=REPO)
    rows = evaluation["trace_rows"]
    selected_by_family = {
        row["case_family"]: row["policy_decision"]["selected_arm"] for row in rows
    }

    assert evaluation["multi_session_trace_count"] == len(rows)
    assert evaluation["multi_session_trace_count"] >= 9
    assert set(evaluation["baseline_metrics"]) == set(exp.BASELINE_NAMES)
    assert set(evaluation["policy_action_space"]) == set(exp.POLICY_ARMS)
    assert {row["case_family"] for row in rows} >= exp.REQUIRED_CASE_FAMILIES
    assert {row["session_id"] for row in rows} >= {"session-a", "session-b", "session-c"}
    assert selected_by_family["no_memory_competitive"] == "no_memory"
    assert selected_by_family["naive_icl_competitive"] == "naive_icl"
    assert all(row["raw_trace_receipt"]["checksum"].startswith("sha256:") for row in rows)


def test_scenario_learn_5460_policy_updates_cannot_bypass_gates() -> None:
    """SCENARIO-LEARN-5460-GATES: failed provenance or replay cannot update stats."""

    policy = exp.FrozenPolicyBandit()
    row = exp.build_policy_stream()[0]
    outcome = deepcopy(row["arm_outcomes"]["governed_memory"])
    good = policy.record_policy_evidence(
        trace_id=row["trace_id"],
        context_key=row["context_key"],
        arm="governed_memory",
        outcome=outcome,
        evidence_id="evidence-good",
    )
    before = policy.snapshot()

    no_provenance = deepcopy(outcome)
    no_provenance["provenance_pass"] = False
    rejected_provenance = policy.record_policy_evidence(
        trace_id=row["trace_id"],
        context_key=row["context_key"],
        arm="governed_memory",
        outcome=no_provenance,
        evidence_id="evidence-no-provenance",
    )
    no_replay = deepcopy(outcome)
    no_replay["replay_pass"] = False
    rejected_replay = policy.record_policy_evidence(
        trace_id=row["trace_id"],
        context_key=row["context_key"],
        arm="governed_memory",
        outcome=no_replay,
        evidence_id="evidence-no-replay",
    )

    assert good["update_status"] == "accepted"
    assert rejected_provenance["update_status"] == "rejected_by_governance"
    assert "provenance_failed" in rejected_provenance["gate_receipt"]["reasons"]
    assert rejected_replay["update_status"] == "rejected_by_governance"
    assert "replay_failed" in rejected_replay["gate_receipt"]["reasons"]
    assert policy.snapshot() == before
    assert "evidence-no-provenance" not in before["arm_stats"]["cad|governed_memory"][
        "accepted_evidence_ids"
    ]


def test_scenario_learn_5460_confidence_receipts_and_controls() -> None:
    """SCENARIO-LEARN-5460-CONTROLS: policy receipts explain safe decisions."""

    evaluation = exp.evaluate_policy_bandit(root=REPO)
    policy = evaluation["policy_metrics"]
    baselines = evaluation["baseline_metrics"]
    receipts = evaluation["confidence_receipts"]
    negative_rows = [
        row for row in evaluation["trace_rows"] if row["negative_transfer_candidate"] is True
    ]

    assert len(receipts) == evaluation["multi_session_trace_count"]
    assert all(set(receipt["arm_scores"]) == set(exp.POLICY_ARMS) for receipt in receipts)
    assert all(
        {"score", "uncertainty", "expected_reward", "context_cost", "verifier_cost"}
        <= set(score)
        for receipt in receipts
        for score in receipt["arm_scores"].values()
    )
    assert policy["constraint_violations"] == 0
    assert baselines["ungated_memory"]["constraint_violations"] > 0
    assert evaluation["negative_transfer_deflection_rate"] == 1.0
    assert all(row["policy_decision"]["selected_arm"] != "governed_memory" for row in negative_rows)
    assert evaluation["regret_proxy_delta_vs_no_memory"] > 0.0
    assert evaluation["quality_delta_vs_naive_icl"] >= 0.0
    assert evaluation["context_efficiency_delta"] > 0.0
    assert evaluation["verifier_cost_delta"] > 0.0


def test_scenario_learn_5460_rollback_removes_bad_policy_evidence() -> None:
    """SCENARIO-LEARN-5460-ROLLBACK: bad policy evidence is removed from stats."""

    policy = exp.FrozenPolicyBandit()
    row = exp.build_policy_stream()[0]
    outcome = deepcopy(row["arm_outcomes"]["governed_memory"])
    accepted = policy.record_policy_evidence(
        trace_id=row["trace_id"],
        context_key=row["context_key"],
        arm="governed_memory",
        outcome=outcome,
        evidence_id="evidence-to-rollback",
    )
    policy.record_policy_evidence(
        trace_id=row["trace_id"],
        context_key=row["context_key"],
        arm="no_memory",
        outcome=row["arm_outcomes"]["no_memory"],
        evidence_id="evidence-to-keep",
    )
    rollback = policy.rollback_evidence("evidence-to-rollback")
    missing = policy.rollback_evidence("missing-evidence")
    snapshot = policy.snapshot()
    post_decision = policy.decide(exp.build_policy_stream()[-1])

    assert accepted["update_status"] == "accepted"
    assert rollback["rollback_success"] is True
    assert rollback["removed_evidence_id"] == "evidence-to-rollback"
    assert missing["rollback_success"] is False
    assert "evidence-to-rollback" not in snapshot["arm_stats"]["cad|governed_memory"][
        "accepted_evidence_ids"
    ]
    assert "evidence-to-keep" in snapshot["arm_stats"]["cad|no_memory"][
        "accepted_evidence_ids"
    ]
    assert "evidence-to-rollback" not in post_decision["cited_evidence_ids"]


def test_scenario_learn_5460_no_weight_mutation_boundary() -> None:
    """SCENARIO-LEARN-5460-NO-WEIGHT-MUTATION: learning stays in policy stats."""

    evaluation = exp.evaluate_policy_bandit(root=REPO)

    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "contextual_bandit_policy_statistics_only",
    }


def test_req_learn_5460_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5460-6: run() writes the terminal policy artifact and receipts."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    receipts_path = tmp_path / exp.CONFIDENCE_RECEIPTS_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        receipts_path=receipts_path,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
    )
    receipts = [
        json.loads(line)
        for line in receipts_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["policy_update_count"] > 0
    assert artifact["multi_session_trace_count"] == len(artifact["trace_rows"])
    assert artifact["baseline_names"] == list(exp.BASELINE_NAMES)
    assert artifact["policy_confidence_receipts_path"] == str(
        exp.CONFIDENCE_RECEIPTS_RELATIVE_PATH
    )
    assert len(receipts) == artifact["multi_session_trace_count"]
    assert artifact["regret_proxy_delta_vs_no_memory"] > 0.0
    assert artifact["quality_delta_vs_naive_icl"] >= 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["cumulative_constraint_violations"] == 0
    assert artifact["negative_transfer_deflection_rate"] == 1.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_policy_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["research_conductor_modified"] is False
    exp.validate_artifact(artifact)


def test_req_learn_5460_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5460-6: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    receipts = [
        json.loads(line)
        for line in RECEIPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert len(receipts) == result["multi_session_trace_count"]
    assert result["csl_policy_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5460_blocked_artifact_reports_missing_tests() -> None:
    """REQ-LEARN-5460-6: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["csl_policy_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5460_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5460-6: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND, COVERAGE_COMMAND])

    bad_missing = deepcopy(artifact)
    bad_missing.pop("policy_update_count")
    with pytest.raises(ValueError, match="policy_update_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["policy_update_count"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_baselines = deepcopy(artifact)
    bad_baselines["baseline_names"] = ["no_memory"]
    with pytest.raises(ValueError, match="baseline_names"):
        exp.validate_artifact(bad_baselines)

    bad_receipts_path = deepcopy(artifact)
    bad_receipts_path["policy_confidence_receipts_path"] = "results/elsewhere.jsonl"
    with pytest.raises(ValueError, match="policy_confidence_receipts_path"):
        exp.validate_artifact(bad_receipts_path)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_trace_count = deepcopy(artifact)
    bad_trace_count["multi_session_trace_count"] += 1
    with pytest.raises(ValueError, match="multi_session_trace_count"):
        exp.validate_artifact(bad_trace_count)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor_modified"):
        exp.validate_artifact(bad_conductor)

    bad_complete_status = deepcopy(artifact)
    bad_complete_status["csl_policy_ready"] = False
    bad_complete_status["status"] = "complete"
    with pytest.raises(ValueError, match="csl_policy_ready"):
        exp.validate_artifact(bad_complete_status)

    bad_blocked_status = deepcopy(artifact)
    bad_blocked_status["status"] = "blocked"
    with pytest.raises(ValueError, match="csl_policy_ready"):
        exp.validate_artifact(bad_blocked_status)

    bad_ready = deepcopy(artifact)
    bad_ready["readiness_checks"]["all_passed"] = False
    bad_ready["tests_run"] = []
    bad_ready["policy_update_count"] = 0
    bad_ready["regret_proxy_delta_vs_no_memory"] = 0.0
    bad_ready["quality_delta_vs_naive_icl"] = -0.1
    bad_ready["context_efficiency_delta"] = 0.0
    bad_ready["verifier_cost_delta"] = 0.0
    bad_ready["cumulative_constraint_violations"] = 1
    bad_ready["negative_transfer_deflection_rate"] = 0.0
    bad_ready["rollback_recovery_rate"] = 0.0
    bad_ready["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_ready)
