"""Tests for Exp6449 generation-to-verdict path receipt contract.

Spec refs: REQ-VERIFY-6449, SCENARIO-VERIFY-6449-CHAIN,
SCENARIO-VERIFY-6449-CONTROLS, SCENARIO-VERIFY-6449-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6449_generation_to_verdict_path_receipt_contract as mod
from carnot import path_receipts


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _test_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6449-data",
        fixture_limit=24,
        test_exit_codes=_test_exit_codes(),
        duration_s=1.25,
        write=write,
    )


def test_req_verify_6449_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6449: OpenSpec owns the V555 path-receipt contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6449") :]
    for marker in (
        "SCENARIO-VERIFY-6449-CHAIN",
        "SCENARIO-VERIFY-6449-CONTROLS",
        "SCENARIO-VERIFY-6449-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "deterministic_fixture_path_receipt_replay_no_llm",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_verify_6449_controls_localize_declared_boundary(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6449-CONTROLS: matched controls localize wrapper effects."""

    artifact = _artifact(tmp_path, write=True)
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert result_path.is_file()
    assert artifact["status"] == "success"
    assert artifact["path_receipt_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("success:")

    rows = artifact["per_unit_rows"]["rows"]
    assert len(rows) == 24 * 3
    assert artifact["per_unit_rows"]["unit_count"] == 24
    assert artifact["per_unit_rows"]["control_count"] == 3
    assert {row["control_id"] for row in rows} == set(mod.CONTROL_IDS)
    for row in rows:
        assert [stage["stage_name"] for stage in row["stages"]] == list(
            path_receipts.REQUIRED_STAGE_NAMES
        )
        assert all(stage["stage_hash"].startswith("sha256:") for stage in row["stages"])
        assert row["expected_verdict"] == row["observed_verdict"]

    assert artifact["identity_replay_results"]["all_replayed"] is True
    assert artifact["identity_replay_results"]["replayed_count"] == 24
    assert artifact["injected_boundary_results"]["all_localized"] is True
    assert artifact["injected_boundary_results"]["changed_boundaries"] == {
        "checker_transport": 24
    }
    assert artifact["restored_boundary_results"]["all_restored"] is True
    assert artifact["restored_boundary_results"]["matched_identity_terminal_hash_count"] == 24
    assert artifact["terminal_verdict_recomputation"]["all_recomputed"] is True
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert mod.validate_artifact(result_path)["valid"] is True

    bad = deepcopy(artifact)
    del bad["status"]
    bad["aggregate_row_recomputation"]["reported_row_count"] += 1
    bad_path = tmp_path / "bad-exp6449.json"
    bad_path.write_text(json.dumps(bad, indent=2, sort_keys=True), encoding="utf-8")
    bad_report = mod.validate_artifact(bad_path)
    assert bad_report["valid"] is False
    assert any(error.startswith("missing_fields:status") for error in bad_report["errors"])
    assert "aggregate_mismatch" in bad_report["errors"]
    assert "ready_score_claim_with_errors" in bad_report["errors"]

    bad_chain = deepcopy(artifact)
    bad_chain["per_unit_rows"]["rows"][0]["stages"][1]["parent_hash"] = (
        path_receipts.sha256_text("broken parent")
    )
    bad_chain_path = tmp_path / "bad-chain-exp6449.json"
    bad_chain_path.write_text(json.dumps(bad_chain, indent=2, sort_keys=True), encoding="utf-8")
    bad_chain_report = mod.validate_artifact(bad_chain_path)
    assert bad_chain_report["valid"] is False
    assert "stage_chain_invalid" in bad_chain_report["errors"]


def test_scenario_verify_6449_chain_rejects_stage_tampering(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6449-CHAIN: stage-chain validator rejects tampering."""

    artifact = _artifact(tmp_path)
    row = next(row for row in artifact["per_unit_rows"]["rows"] if row["control_id"] == "identity")
    allowed = set(artifact["code_and_configuration_hashes"]["allowed_code_hashes"].values())
    clean = path_receipts.validate_stage_chain(row["stages"], allowed_code_hashes=allowed)
    assert clean["accepted"] is True

    duplicate = deepcopy(row["stages"])
    duplicate[1]["stage_id"] = duplicate[0]["stage_id"]
    report = path_receipts.validate_stage_chain(duplicate, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "duplicate_stage_id" in report["reasons"]

    duplicate_name = deepcopy(row["stages"])
    duplicate_name[1]["stage_name"] = duplicate_name[0]["stage_name"]
    report = path_receipts.validate_stage_chain(duplicate_name, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "duplicate_stage_name" in report["reasons"]

    missing_field = deepcopy(row["stages"])
    del missing_field[2]["output_hash"]
    report = path_receipts.validate_stage_chain(missing_field, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "missing_stage_fields:typed_facts" in report["reasons"]

    negative_interval = deepcopy(row["stages"])
    negative_interval[3]["monotonic_end_ns"] = negative_interval[3]["monotonic_start_ns"] - 1
    negative_interval[3] = path_receipts.refresh_stage_hash(negative_interval[3])
    report = path_receipts.validate_stage_chain(negative_interval, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "negative_stage_interval:energy_input" in report["reasons"]

    mutated_input = deepcopy(row["stages"])
    mutated_input[2]["input_hash"] = path_receipts.sha256_text("silent mutation")
    report = path_receipts.validate_stage_chain(mutated_input, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "stage_hash_mismatch:typed_facts" in report["reasons"]
    assert "silent_input_mutation:typed_facts" in report["reasons"]

    bad_verdict = deepcopy(row["stages"])
    bad_verdict[-1]["output_payload"]["observed_verdict"] = "forged"
    bad_verdict[-1] = path_receipts.refresh_stage_hash(bad_verdict[-1])
    report = path_receipts.validate_stage_chain(bad_verdict, allowed_code_hashes=allowed)
    assert report["accepted"] is False
    assert "final_verdict_recompute_mismatch" in report["reasons"]

    assert mod._fixture_raw_event_id(b"{") == ""
    with pytest.raises(ValueError, match="unknown stage attack"):
        mod.mutate_row_for_attack("unknown_attack", row)


def test_scenario_verify_6449_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6449-ATTACKS: every named attack fails closed."""

    artifact = _artifact(tmp_path)
    matrix = artifact["attack_matrix"]
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    assert {row["attack_id"] for row in matrix["rows"]} == set(mod.ATTACK_IDS)
    assert all(row["detected"] for row in matrix["rows"])
    assert all(row["fail_closed"] for row in matrix["rows"])

    tampered = deepcopy(artifact)
    tampered["aggregate_row_recomputation"]["reported_row_count"] += 1
    recomputed = mod.recompute_aggregate_rows(tampered["per_unit_rows"]["rows"], tampered)
    assert recomputed["matches_reported"] is False
    assert "reported_row_count_mismatch" in recomputed["reasons"]

    tampered["aggregate_row_recomputation"]["reported_unit_count"] += 1
    tampered["aggregate_row_recomputation"]["reported_control_counts"] = {"identity": 1}
    recomputed = mod.recompute_aggregate_rows(tampered["per_unit_rows"]["rows"], tampered)
    assert "reported_unit_count_mismatch" in recomputed["reasons"]
    assert "reported_control_counts_mismatch" in recomputed["reasons"]

    partial_rows = [
        row
        for row in artifact["per_unit_rows"]["rows"]
        if row["control_id"] != "restored_wrapper"
    ]
    partial = {"aggregate_row_recomputation": {}}
    recomputed = mod.recompute_aggregate_rows(partial_rows, partial)
    assert "expected_row_count_mismatch" in recomputed["reasons"]
    assert "controls_missing" in recomputed["reasons"]

    findings = mod.current_findings(
        chain={"all_valid": False},
        terminal={"all_recomputed": True},
        attacks={"all_critical_fail_closed": True},
        aggregate={"matches_reported": True},
        protected={"unchanged": True},
    )
    assert findings == [
        {"severity": "critical", "kind": "stage_chain_validation", "detail": "gate failed"}
    ]


def test_req_verify_6449_preconditions_block_terminal_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-6449: failed preconditions write an honest blocked artifact."""

    artifact = mod.run(
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6449-data",
        exp6427_result_path=tmp_path / "missing-exp6427.json",
        fixture_limit=24,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=True,
    )
    assert (tmp_path / mod.RESULT_RELATIVE_PATH.name).is_file()
    assert artifact["status"] == "blocked"
    assert artifact["path_receipt_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["blocked_reason"]
    assert "failed" in artifact["gate_check_summary"]
    assert any(
        check["resource"] == "exp6427_fixture_artifact" and check["available"] is False
        for check in artifact["preconditions_checked"]
    )


def test_req_verify_6449_fixture_precondition_failures_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-6449: fixture precondition failures name the bad boundary."""

    source_payload = mod.read_json_object(REPO / mod.EXP6427_RESULT_RELATIVE_PATH)
    source_rows = source_payload["per_unit_rows"]["rows"]
    source_row = deepcopy(source_rows[0])
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "exp6427.json"

    hash_bad = deepcopy(source_payload)
    hash_bad["per_unit_rows"]["rows"] = [deepcopy(source_row)]
    hash_bad["per_unit_rows"]["rows"][0]["raw_output_sha256"] = "sha256:" + "1" * 64
    hash_bad["manifest_path_hash_counts_balance_and_partition_seals"]["path"] = str(
        REPO / mod.EXP6427_DATA_RELATIVE_PATH / "manifest/fresh_constraint_saturation_events.json"
    )
    artifact_path.write_text(json.dumps(hash_bad), encoding="utf-8")
    with pytest.raises(ValueError, match="fixture hash mismatch"):
        mod.load_fixture_units(artifact_path, fixture_limit=1)

    missing_event = deepcopy(source_payload)
    missing_event["per_unit_rows"]["rows"] = [deepcopy(source_row)]
    missing_event["manifest_path_hash_counts_balance_and_partition_seals"]["path"] = str(
        manifest_path
    )
    manifest_path.write_text(json.dumps({"events": []}), encoding="utf-8")
    artifact_path.write_text(json.dumps(missing_event), encoding="utf-8")
    with pytest.raises(ValueError, match="fixture event missing"):
        mod.load_fixture_units(artifact_path, fixture_limit=1)

    raw_path = tmp_path / "raw.json"
    raw_bytes = b'{"event_id":"other-unit","proposal":{"factor_proposal":{"effects":[]}}}'
    raw_path.write_bytes(raw_bytes)
    raw_id_bad = deepcopy(source_payload)
    raw_id_bad["per_unit_rows"]["rows"] = [deepcopy(source_row)]
    raw_id_bad["per_unit_rows"]["rows"][0]["raw_output_path"] = str(raw_path)
    raw_id_bad["per_unit_rows"]["rows"][0]["raw_output_sha256"] = path_receipts.sha256_bytes(
        raw_bytes
    )
    raw_id_bad["manifest_path_hash_counts_balance_and_partition_seals"]["path"] = str(
        manifest_path
    )
    manifest_path.write_text(json.dumps({"events": [{"event_id": source_row["row_id"]}]}))
    artifact_path.write_text(json.dumps(raw_id_bad), encoding="utf-8")
    with pytest.raises(ValueError, match="raw event id mismatch"):
        mod.load_fixture_units(artifact_path, fixture_limit=1)

    missing_raw = deepcopy(raw_id_bad)
    missing_raw["per_unit_rows"]["rows"][0]["raw_output_path"] = str(tmp_path / "absent.json")
    artifact_path.write_text(json.dumps(missing_raw), encoding="utf-8")
    checks, _units, _manifest = mod.check_preconditions(
        result_path=tmp_path / "out.json",
        data_dir=tmp_path / "data",
        exp6427_result_path=artifact_path,
        fixture_limit=1,
    )
    immutable = next(check for check in checks if check["resource"] == "immutable_fixture_bytes")
    assert immutable["available"] is False
    assert "FileNotFoundError" in immutable["detail"]
