"""Tests for Exp 3145 KAN proof-carrying monitor boundary v2.

Spec refs: REQ-KAN-3145, SCENARIO-KAN-3145.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import kan_proof_carrying_monitor_boundary_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "kan" / "spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_kan_3145_spec_anchor_exists() -> None:
    """REQ-KAN-3145: OpenSpec declares the proof-carrying boundary first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-3145" in spec
    assert "SCENARIO-KAN-3145" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_kan_3145_writes_tiny_proof_carrying_records(tmp_path: Path) -> None:
    """SCENARIO-KAN-3145: known .291 false accepts get replayable KAN records."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.25,
        tests_run=["focused-REQ-KAN-3145"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["kan_proof_carrying_monitor_v2_ready"] is True
    assert artifact["kan_code_present"] is True
    assert artifact["attached_monitor_record_count"] == 2
    assert artifact["milp_property_check_count"] == 1
    assert artifact["deployed_verifier_claim"] is False
    assert artifact["implementation_blockers"] == []
    assert artifact["tests_run"] == ["focused-REQ-KAN-3145"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete_")

    schema = artifact["monitor_record_schema"]
    assert schema["schema"] == mod.MONITOR_RECORD_SCHEMA
    assert "exact_fixture_link" in schema["required_record_fields"]
    assert "milp_property_result" in schema["required_record_fields"]

    local_summary = artifact["local_error_bound_summary"]
    assert local_summary["procedure"] == "max_per_segment_midpoint_residual"
    assert local_summary["unit_count"] == 2
    assert local_summary["max_local_error_bound"] == pytest.approx(0.0625)

    global_summary = artifact["global_error_bound_summary"]
    assert global_summary["procedure"] == "weighted_output_error_propagation"
    assert global_summary["global_error_bound"] == pytest.approx(0.09375)
    assert global_summary["bounds_distinct_by_construction"] is True

    records = artifact["monitor_records"]
    assert [record["fixture_id"] for record in records] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    for record in records:
        assert record["schema"] == mod.MONITOR_RECORD_SCHEMA
        assert record["pwa_abstraction_parameters"]["unit_count"] == 2
        assert record["milp_property_result"]["property_verified"] is True
        assert record["milp_property_result"]["solver_status"] == "optimal"
        assert record["exact_fixture_link"]["ledger_action"] == "reject"
        assert record["exact_fixture_link"]["live_decision"] == "accept"
        assert record["record_checksum"] == mod.record_checksum(record)
        mod.validate_monitor_record(record)

    relevance = artifact["false_accept_relevance"]
    assert relevance["known_false_accept_row_ids"] == [
        "resyn-3084-arith-003",
        "resyn-3084-smt-000",
    ]
    assert relevance["attached_false_accept_record_count"] == 2
    assert relevance["would_help_replay_audit"] is True
    assert relevance["would_prevent_live_false_accept"] is False
    assert relevance["deployed_gate_missing"] is True
    assert "contradiction" in relevance["covered_false_accept_families"]

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "checked_in_artifact_kan_monitor_boundary"
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["hardware_execution"] is False
    assert substrate["deployed_verifier_claim"] is False
    mod.validate_artifact(artifact)


def test_req_kan_3145_record_link_is_derived_from_monitor_events() -> None:
    """REQ-KAN-3145: exact fixture links are replayed from Exp 3126 events."""

    exp3126 = mod.read_json_object(REPO_ROOT / mod.EXP3126_REL_PATH)
    exp3131 = mod.read_json_object(REPO_ROOT / mod.EXP3131_REL_PATH)
    groups = mod.monitor_event_groups_by_fixture(exp3126.get("monitor_events"))
    proof = mod.kan_proof_payload(exp3131)
    record = mod.build_monitor_record(
        "resyn-3084-arith-003",
        groups["resyn-3084-arith-003"],
        proof,
    )

    assert record["exact_fixture_link"]["fixture_id"] == "resyn-3084-arith-003"
    assert record["exact_fixture_link"]["exact_label"] == "INVALID"
    assert record["exact_fixture_link"]["expected_action"] == "reject"
    assert record["exact_fixture_link"]["monitor_failure_mechanism"] == "contradiction"
    assert record["exact_fixture_link"]["final_answer_consistent_with_ledger"] is False
    assert record["exact_fixture_link"]["monitor_event_indices"] == [15, 16, 17, 18, 19]
    assert record["pwa_abstraction_parameters"]["property_domain"] == [-0.5, 0.5]
    assert record["milp_property_result"]["certified_upper_bound"] == pytest.approx(0.53125)
    assert record["record_checksum"] == mod.record_checksum(record)


def test_req_kan_3145_validation_blocks_overclaims(tmp_path: Path) -> None:
    """REQ-KAN-3145: validation rejects missing fields and deployment claims."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=2.0,
        tests_run=["validation"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    broken_record = artifact["monitor_records"][0] | {"record_checksum": "bad"}

    def checked_record(**updates: Any) -> dict[str, Any]:
        record = dict(artifact["monitor_records"][0])
        record.update(updates)
        record["record_checksum"] = mod.record_checksum(record)
        return record

    invalid_cases = [
        ({}, "missing required fields"),
        (artifact | {"honest_verdict": "ready"}, "honest_verdict"),
        (artifact | {"deployed_verifier_claim": True}, "deployed_verifier_claim"),
        (artifact | {"attached_monitor_record_count": 0}, "attached_monitor_record_count"),
        (artifact | {"milp_property_check_count": 2}, "milp_property_check_count"),
        (artifact | {"monitor_records": [broken_record]}, "record checksum"),
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
                | {"model_weight_mutation": True}
            },
            "model weights",
        ),
        (
            artifact
            | {"inference_substrate": artifact["inference_substrate"] | {"hardware_execution": True}},
            "hardware execution",
        ),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
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
                | {"deployed_verifier_claim": True}
            },
            "deployed verifier claim",
        ),
        (artifact | {"kan_code_present": False}, "ready boundary requires KAN code"),
        (
            artifact | {"implementation_blockers": ["missing live integration"]},
            "ready boundary cannot have implementation blockers",
        ),
        (
            artifact
            | {
                "monitor_records": [],
                "attached_monitor_record_count": 0,
                "milp_property_check_count": 0,
            },
            "ready boundary requires attached monitor records",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)

    record_cases = [
        ({field: None for field in ()}, ""),
        ({"schema": "bad"}, "schema mismatch"),
        ({"exact_fixture_link": {}}, "exact_fixture_link"),
        ({"pwa_abstraction_parameters": {}}, "pwa_abstraction_parameters"),
        ({"milp_property_result": {"property_verified": False, "solver_status": "optimal"}}, "verified property"),
        ({"milp_property_result": {"property_verified": True, "solver_status": "sat"}}, "optimal"),
    ]
    missing_record = dict(artifact["monitor_records"][0])
    missing_record.pop("schema")
    with pytest.raises(ValueError, match="missing monitor record fields"):
        mod.validate_monitor_record(missing_record)
    for updates, message in record_cases[1:]:
        with pytest.raises(ValueError, match=message):
            mod.validate_monitor_record(checked_record(**updates))


def test_req_kan_3145_fails_closed_when_sources_are_missing(tmp_path: Path) -> None:
    """REQ-KAN-3145: missing KAN/monitor inputs produce an honest boundary."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        tests_run=["missing-source"],
    )

    assert artifact["kan_proof_carrying_monitor_v2_ready"] is False
    assert artifact["kan_code_present"] is False
    assert artifact["attached_monitor_record_count"] == 0
    assert artifact["milp_property_check_count"] == 0
    assert artifact["deployed_verifier_claim"] is False
    assert artifact["implementation_blockers"]
    assert "python/carnot/verify/kan_pwa_milp_corrigendum.py" in artifact["implementation_blockers"]
    assert mod.EXP3126_REL_PATH.as_posix() in artifact["implementation_blockers"]
    assert artifact["honest_verdict"].startswith("complete_")
    mod.validate_artifact(artifact)

    relative_output = mod.write_artifact(
        tmp_path,
        output_path=Path("relative-boundary.json"),
        started_s=6.0,
        now_s=6.5,
        tests_run=["relative-missing-source"],
    )
    assert relative_output == tmp_path / "relative-boundary.json"
    assert json.loads(relative_output.read_text("utf-8"))["attached_monitor_record_count"] == 0


def test_req_kan_3145_defensive_helpers_cover_bad_inputs(tmp_path: Path) -> None:
    """REQ-KAN-3145: parsers and selectors are deterministic on malformed input."""

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_json) == {}

    assert mod.mapping_rows("bad") == []
    assert mod.monitor_event_groups_by_fixture("bad") == {}
    assert mod.selected_fixture_ids([], ["z"], limit=2) == []
    assert mod.selected_fixture_ids(["b", "a", "c"], ["c", "a"], limit=1) == ["a"]
    assert mod.unique_milp_property_check_count([]) == 0
    assert any(
        "proof-carrying monitor records" in blocker
        for blocker in mod.implementation_blockers(
            REPO_ROOT,
            True,
            mod.source_artifacts(REPO_ROOT),
            [],
        )
    )
    assert mod.duration(10.0, 9.0) == 0.0

    _write_json(tmp_path, "ok.json", {"ok": True})
    assert mod.read_json_object(tmp_path / "ok.json") == {"ok": True}
