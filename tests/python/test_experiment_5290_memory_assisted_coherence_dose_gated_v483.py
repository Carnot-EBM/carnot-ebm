"""Tests for Exp 5290 memory-assisted coherence dosing.

Spec refs: REQ-VERIFY-5290, SCENARIO-VERIFY-5290.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from carnot import experiment_5290_memory_assisted_coherence_dose_gated_v483 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5290_spec_declares_coherence_dose_contract() -> None:
    """REQ-VERIFY-5290: OpenSpec anchors the no-LLM coherence-dose replay."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5290") :]

    for marker in (
        "REQ-VERIFY-5290",
        "SCENARIO-VERIFY-5290",
        str(mod.RESULT_RELATIVE_PATH),
        "coherence_fixture_ready=true",
        "memory_attribution_ready=true",
        "always-full coherence verification",
        "no-memory dosing",
        "governed-memory dosing",
        "stale memory",
        "shuffled scope/routing",
        "harmful memory",
        "safety-negative",
        "aggregation_from_upstream_artifacts",
        "offline_deterministic_fixture_no_llm",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    normalized_section = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5290_step0_preconditions_block_without_upstream_gates(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5290-1: upstream readiness gates are checked before metrics."""

    artifacts = mod.load_upstream_artifacts(REPO)
    preconditions = mod.check_preconditions(root=REPO, upstream_artifacts=artifacts)

    assert preconditions["exp5285.coherence_fixture_ready"] is True
    assert preconditions["exp5289.memory_attribution_ready"] is True
    assert preconditions["all_gates_ready"] is True
    assert preconditions["blockers"] == []

    blocked_artifacts = deepcopy(artifacts)
    blocked_artifacts["exp5285"] = {
        **blocked_artifacts["exp5285"],
        "coherence_fixture_ready": False,
    }
    artifact = mod.build_result_artifact(
        root=tmp_path,
        upstream_artifacts=blocked_artifacts,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["coherence_dose_positive"] is False
    assert artifact["decision_quality_delta"]["value"]["governed_minus_always_full"] == 0.0
    assert artifact["full_verifier_calls_avoided"]["value"]["vs_always_full"] == 0
    assert artifact["unsafe_false_accepts"]["value"]["count"] == 0
    assert "exp5285.coherence_fixture_ready" in artifact["preconditions_checked"]["blockers"]
    mod.validate_artifact(artifact)

    attribution_blocked = deepcopy(artifacts)
    attribution_blocked["exp5289"] = {
        **attribution_blocked["exp5289"],
        "memory_attribution_ready": False,
    }
    attribution_preconditions = mod.check_preconditions(
        root=REPO,
        upstream_artifacts=attribution_blocked,
    )
    assert "exp5289.memory_attribution_ready" in attribution_preconditions["blockers"]


def test_scenario_verify_5290_compares_three_policies_on_same_fixture() -> None:
    """SCENARIO-VERIFY-5290: governed memory avoids checks without quality loss."""

    artifacts = mod.load_upstream_artifacts(REPO)
    rows = mod.build_coherence_rows(root=REPO, upstream_artifacts=artifacts)
    malformed_artifacts = deepcopy(artifacts)
    malformed_artifacts["exp5285"]["case_results"] = [None] + malformed_artifacts["exp5285"][
        "case_results"
    ]
    assert len(mod.build_coherence_rows(root=REPO, upstream_artifacts=malformed_artifacts)) == 7
    evaluation = mod.evaluate_policies(rows, attribution_artifact=artifacts["exp5289"])

    assert len(rows) == 7
    assert evaluation["policy_metrics"]["always_full"]["full_verifier_calls"] == 7
    assert evaluation["policy_metrics"]["no_memory"]["full_verifier_calls"] == 5
    assert evaluation["policy_metrics"]["governed_memory"]["full_verifier_calls"] == 3
    assert evaluation["policy_metrics"]["always_full"]["quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["no_memory"]["quality_rate"] == 1.0
    assert evaluation["policy_metrics"]["governed_memory"]["quality_rate"] == 1.0
    assert evaluation["decision_quality_delta"]["governed_minus_always_full"] == 0.0
    assert evaluation["decision_quality_delta"]["governed_minus_no_memory"] == 0.0
    assert evaluation["full_verifier_calls_avoided"]["vs_always_full"] == 4
    assert evaluation["full_verifier_calls_avoided"]["additional_vs_no_memory"] == 2
    assert evaluation["unsafe_false_accepts"]["count"] == 0
    assert evaluation["coherence_dose_positive"] is True
    assert evaluation["route_counts"]["governed_memory"] == {
        "cheap_deterministic": 1,
        "full_verifier": 3,
        "memory_guided_coherence_check": 3,
    }


def test_req_verify_5290_operation_stage_controls_drive_reduction_or_escalation() -> None:
    """REQ-VERIFY-5290-3/4: stage attribution gates every memory reduction."""

    artifacts = mod.load_upstream_artifacts(REPO)
    rows = mod.build_coherence_rows(root=REPO, upstream_artifacts=artifacts)
    evaluation = mod.evaluate_policies(rows, attribution_artifact=artifacts["exp5289"])
    by_id = {row["case_id"]: row for row in evaluation["policy_rows"]["governed_memory"]}

    assert by_id["ktc-001-supported-runtime"]["route"] == "memory_guided_coherence_check"
    assert by_id["ktc-002-unsupported-sensor"]["route"] == "memory_guided_coherence_check"
    assert by_id["ktc-003-partial-trial"]["route"] == "memory_guided_coherence_check"
    assert by_id["ktc-004-stale-route"]["route"] == "full_verifier"
    assert by_id["ktc-004-stale-route"]["escalation_reason"] == "stale_or_conflicting_memory"
    assert by_id["ktc-005-contradictory-lab"]["route"] == "full_verifier"
    assert by_id["ktc-005-contradictory-lab"]["escalation_reason"] == "shuffled_scope_or_routing"
    assert by_id["ktc-006-safety-negative-dose"]["route"] == "full_verifier"
    assert by_id["ktc-006-safety-negative-dose"]["escalation_reason"] == (
        "safety_negative_or_harmful_memory"
    )
    assert by_id["ktc-007-supported-format-invalid"]["route"] == "cheap_deterministic"

    stage = evaluation["attribution_stage_contributions"]
    assert stage["reductions_by_stage"] == {"use": 3}
    assert stage["escalations_by_stage"] == {"maintenance": 1, "rollback": 1, "routing": 1}
    assert stage["upstream_operation_stage_error_counts"]["use"] == 1
    assert evaluation["stale_conflict_handling"]["stale_or_conflict_escalations"] == 1
    assert evaluation["stale_conflict_handling"]["shuffled_scope_escalations"] == 1
    assert evaluation["rollback_triggers"]["trigger_count"] == 1
    assert evaluation["rollback_triggers"]["case_ids"] == ["ktc-006-safety-negative-dose"]


def test_req_verify_5290_nonpositive_branches_are_explicit() -> None:
    """REQ-VERIFY-5290-5: null, harmful, and blocked verdicts are distinguishable."""

    artifacts = mod.load_upstream_artifacts(REPO)
    rows = mod.build_coherence_rows(root=REPO, upstream_artifacts=artifacts)
    evaluation = mod.evaluate_policies(rows, attribution_artifact=artifacts["exp5289"])

    fallback = replace(
        rows[0],
        memory_control_kind="none",
        attribution_stage=None,
        operation_stage_label=None,
    )
    assert mod.choose_governed_memory_route(fallback) == mod.choose_no_memory_route(fallback)

    non_supported_cheap = replace(
        rows[1],
        memory_control_kind="none",
        attribution_stage=None,
        operation_stage_label=None,
        lexical_baseline_accept=False,
    )
    assert mod.decision_for_route(non_supported_cheap, mod.ROUTE_CHEAP) == "reject"

    untrusted = replace(
        rows[0],
        memory_control_kind="missing_provenance",
        attribution_stage="extraction",
        operation_stage_label="extraction",
    )
    untrusted_row = mod._decision_row(untrusted, mod.choose_governed_memory_route(untrusted))
    assert untrusted_row["escalation_reason"] == "untrusted_memory_control"

    unsafe = deepcopy(evaluation)
    unsafe["unsafe_false_accepts"]["count"] = 1
    unsafe["coherence_dose_positive"] = False
    assert mod._honest_verdict({"all_gates_ready": True, "blockers": []}, unsafe).startswith(
        "harmful_"
    )

    quality_drop = deepcopy(evaluation)
    quality_drop["decision_quality_delta"]["governed_minus_always_full"] = -0.1
    quality_drop["coherence_dose_positive"] = False
    assert "reduced always-full quality" in mod._honest_verdict(
        {"all_gates_ready": True, "blockers": []},
        quality_drop,
    )

    no_gain = deepcopy(evaluation)
    no_gain["full_verifier_calls_avoided"]["additional_vs_no_memory"] = 0
    no_gain["coherence_dose_positive"] = False
    assert mod._honest_verdict({"all_gates_ready": True, "blockers": []}, no_gain).startswith(
        "null:"
    )

    incomplete_positive_gate = deepcopy(evaluation)
    incomplete_positive_gate["coherence_dose_positive"] = False
    assert (
        mod._honest_verdict(
            {"all_gates_ready": True, "blockers": []},
            incomplete_positive_gate,
        )
        == "null: memory-assisted coherence dosing did not satisfy every positive gate"
    )
    assert mod._sha256_file(REPO / "missing-5290.json") is None


def test_req_verify_5290_artifact_schema_and_run_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-5290: run() writes the required principle-wrapped artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit coherence dose", "outcome": "passed"}]
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "memory-assisted coherence dosing helped" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["coherence_dose_positive"] is True
    assert (
        artifact["coherence_dose_positive_principle"]
        == (mod.FIELD_PRINCIPLES["coherence_dose_positive"])
    )
    assert artifact["decision_quality_delta"]["value"]["governed_minus_always_full"] == 0.0
    assert artifact["full_verifier_calls_avoided"]["value"]["additional_vs_no_memory"] == 2
    assert artifact["unsafe_false_accepts"]["value"]["count"] == 0
    assert artifact["stale_conflict_handling"]["value"]["all_escalated"] is True
    assert artifact["rollback_triggers"]["value"]["trigger_count"] == 1
    assert artifact["attribution_stage_contributions"]["value"]["reductions_by_stage"] == {"use": 3}
    assert artifact["continuous_self_learning_loop"]["value"]["memory_affects"] == (
        "claim_coherence_check_allocation_only"
    )
    assert artifact["tests_run"] == tests_run
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["source_artifact_checksums"]["exp5285"].startswith("sha256:")
    assert artifact["source_artifact_checksums"]["exp5289"].startswith("sha256:")

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_verify_5290_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5290: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["coherence_dose_positive"] is True
    assert result["full_verifier_calls_avoided"]["value"]["vs_always_full"] == 4
    assert result["full_verifier_calls_avoided"]["value"]["additional_vs_no_memory"] == 2
    assert result["unsafe_false_accepts"]["value"]["count"] == 0
    assert result["inference_substrate"]["value"] == "aggregation_from_upstream_artifacts"
    mod.validate_artifact(result)
