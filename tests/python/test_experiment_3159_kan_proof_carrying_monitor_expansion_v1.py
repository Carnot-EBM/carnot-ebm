"""Tests for Exp 3159 KAN proof-carrying monitor expansion.

Spec refs: REQ-KAN-3159, SCENARIO-KAN-3159.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import kan_proof_carrying_monitor_expansion_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "kan" / "spec.md"


def test_req_kan_3159_spec_anchor_exists() -> None:
    """REQ-KAN-3159: the expansion schema is declared before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-3159" in spec
    assert "SCENARIO-KAN-3159" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_kan_3159_writes_expanded_exact_row_records(tmp_path: Path) -> None:
    """SCENARIO-KAN-3159: clean exact rows extend prior proof records."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["focused-REQ-KAN-3159"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["kan_proof_carrying_monitor_expansion_v1_ready"] is True
    assert artifact["monitor_record_count"] == 4
    assert artifact["new_monitor_record_count"] == 2
    assert artifact["exact_row_coverage_count"] == 4
    assert artifact["deployed_verifier_claim_allowed"] is False
    assert artifact["tests_run"] == ["focused-REQ-KAN-3159"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete_")

    records = artifact["pwa_milp_bound_records"]
    assert [record["fixture_id"] for record in records] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
        "resyn-3084-arith-000",
        "resyn-3084-arith-001",
    ]
    assert [record["record_origin"] for record in records] == [
        "carried_forward_exp3145",
        "carried_forward_exp3145",
        "new_exp3159_exact_clean_row",
        "new_exp3159_exact_clean_row",
    ]
    assert [record["exact_row_set"] for record in records] == [
        "false_accept",
        "false_accept",
        "clean",
        "clean",
    ]

    clean_accept = records[2]
    assert clean_accept["exact_label_link"]["exact_label"] == "VALID"
    assert clean_accept["exact_label_link"]["expected_action"] == "accept"
    assert clean_accept["exact_label_link"]["live_decision"] == "accept"
    assert clean_accept["exact_label_link"]["final_answer_consistent_with_exact"] is True
    assert clean_accept["exact_label_link"]["is_monitor_violation"] is False

    clean_reject = records[3]
    assert clean_reject["exact_label_link"]["exact_label"] == "INVALID"
    assert clean_reject["exact_label_link"]["expected_action"] == "reject"
    assert clean_reject["exact_label_link"]["live_decision"] == "reject"
    assert clean_reject["exact_label_link"]["final_answer_consistent_with_ledger"] is True

    for record in records:
        assert record["domain_bounds"]["input_domain"] == [-0.5, 0.5]
        assert record["domain_bounds"]["property_threshold"] == pytest.approx(0.625)
        assert record["domain_bounds"]["certified_upper_bound"] == pytest.approx(0.53125)
        assert record["domain_bounds"]["max_local_error_bound"] == pytest.approx(0.0625)
        assert record["domain_bounds"]["global_error_bound"] == pytest.approx(0.09375)
        assert record["pwa_milp_status"]["property_verified"] is True
        assert record["pwa_milp_status"]["solver_status"] == "optimal"
        assert record["pwa_milp_status"]["milp_backend_name"] == "z3"
        assert record["deployed_verifier_claim_allowed"] is False
        assert record["residual_risk"]
        assert record["record_checksum"] == mod.record_checksum(record)
        mod.validate_bound_record(record)

    assert artifact["exact_row_sets"]["false_accept_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    assert artifact["exact_row_sets"]["selected_new_clean_row_ids"] == [
        "resyn-3084-arith-000",
        "resyn-3084-arith-001",
    ]
    assert "No deployed accept/reject gate consumes these proof records." in artifact[
        "residual_blockers"
    ]

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "checked_in_artifact_kan_monitor_expansion"
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["hardware_execution"] is False
    assert substrate["deployed_verifier_claim_allowed"] is False
    mod.validate_artifact(artifact)


def test_req_kan_3159_exact_row_sets_are_loaded_from_autopsy() -> None:
    """REQ-KAN-3159: false-accept and clean rows come from exact autopsy rows."""

    exp3136 = mod.read_json_object(REPO_ROOT / mod.EXP3136_REL_PATH)
    exp3126 = mod.read_json_object(REPO_ROOT / mod.EXP3126_REL_PATH)
    groups = mod.monitor_event_groups_by_fixture(exp3126.get("monitor_events"))
    row_sets = mod.exact_row_sets(exp3136)

    assert row_sets["false_accept_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    assert row_sets["clean_exact_row_ids"] == [
        "resyn-3084-arith-000",
        "resyn-3084-arith-001",
        "resyn-3084-arith-002",
        "resyn-3084-repair-json-000",
    ]
    assert mod.selected_new_exact_row_ids(
        row_sets["clean_exact_row_ids"],
        prior_ids={"resyn-3084-arith-003", "resyn-3084-smt-000"},
        available_ids=groups,
        limit=2,
    ) == ["resyn-3084-arith-000", "resyn-3084-arith-001"]
    assert mod.is_clean_exact_row(
        {"row_id": "synthetic-drift", "failure_mechanism_from_exp3124": "contradiction"},
        false_accept_ids=[],
    ) is False


def test_req_kan_3159_validation_blocks_overclaims(tmp_path: Path) -> None:
    """REQ-KAN-3159: validation rejects missing fields and deployment claims."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=2.0,
        tests_run=["validation"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    record = artifact["pwa_milp_bound_records"][0]
    broken_record = dict(record)
    broken_record["record_checksum"] = "bad"
    carried_only_records = [
        candidate
        for candidate in artifact["pwa_milp_bound_records"]
        if candidate["record_origin"] == "carried_forward_exp3145"
    ]

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"honest_verdict": "ready"}, "honest_verdict"),
        (artifact | {"deployed_verifier_claim_allowed": True}, "deployed verifier"),
        (artifact | {"monitor_record_count": 3}, "monitor_record_count"),
        (artifact | {"new_monitor_record_count": 1}, "new_monitor_record_count"),
        (artifact | {"exact_row_coverage_count": 3}, "exact_row_coverage_count"),
        (artifact | {"pwa_milp_bound_records": [broken_record]}, "record checksum"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            },
            "live LLM inference",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_model_inference": True}
            },
            "live model inference",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            },
            "model weights",
        ),
        (
            artifact
            | {"inference_substrate": artifact["inference_substrate"] | {"hardware_execution": True}},
            "hardware execution",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"deployed_verifier_claim_allowed": True}
            },
            "deployed verifier",
        ),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
        (artifact | {"residual_blockers": []}, "residual_blockers"),
        (
            artifact
            | {
                "kan_proof_carrying_monitor_expansion_v1_ready": True,
                "monitor_record_count": len(carried_only_records),
                "new_monitor_record_count": 0,
                "exact_row_coverage_count": len(carried_only_records),
                "pwa_milp_bound_records": carried_only_records,
            },
            "ready expansion requires new monitor records",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    def checked_record(**updates: Any) -> dict[str, Any]:
        candidate = dict(record)
        candidate.update(updates)
        candidate["record_checksum"] = mod.record_checksum(candidate)
        return candidate

    missing_record_field = dict(record)
    missing_record_field.pop("schema")
    with pytest.raises(ValueError, match="missing bound record fields"):
        mod.validate_bound_record(missing_record_field)

    record_cases = [
        (checked_record(schema="bad"), "schema mismatch"),
        (checked_record(exact_label_link={}), "exact_label_link"),
        (checked_record(domain_bounds={}), "domain_bounds"),
        (
            checked_record(pwa_milp_status={"property_verified": False, "solver_status": "optimal"}),
            "verified PWA/MILP",
        ),
        (
            checked_record(pwa_milp_status={"property_verified": True, "solver_status": "sat"}),
            "optimal",
        ),
        (checked_record(deployed_verifier_claim_allowed=True), "deployed verifier"),
        (checked_record(residual_risk=[]), "residual_risk"),
    ]
    for bad_record, message in record_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_bound_record(bad_record)


def test_req_kan_3159_fails_closed_when_sources_are_missing(tmp_path: Path) -> None:
    """REQ-KAN-3159: absent source artifacts produce a non-deployment boundary."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        tests_run=["missing-source"],
    )

    assert artifact["kan_proof_carrying_monitor_expansion_v1_ready"] is False
    assert artifact["monitor_record_count"] == 0
    assert artifact["new_monitor_record_count"] == 0
    assert artifact["exact_row_coverage_count"] == 0
    assert artifact["deployed_verifier_claim_allowed"] is False
    assert artifact["implementation_blockers"]
    assert mod.EXP3145_REL_PATH.as_posix() in artifact["implementation_blockers"]
    assert "No deployed accept/reject gate consumes these proof records." in artifact[
        "residual_blockers"
    ]
    assert artifact["honest_verdict"].startswith("complete_")
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=Path("relative-exp3159.json"),
        started_s=6.0,
        now_s=6.25,
        tests_run=["relative-missing-source"],
    )
    assert relative_output == tmp_path / "relative-exp3159.json"
    assert json.loads(relative_output.read_text("utf-8"))["monitor_record_count"] == 0
