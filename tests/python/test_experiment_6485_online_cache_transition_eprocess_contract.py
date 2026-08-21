"""Tests for Exp6485 online cache transition e-process contracts.

Spec refs: REQ-INFRA-6485, SCENARIO-INFRA-6485-EVENTS,
SCENARIO-INFRA-6485-ACTIONS, SCENARIO-INFRA-6485-EPROCESS,
SCENARIO-INFRA-6485-LIFECYCLE, SCENARIO-INFRA-6485-ATTACKS,
SCENARIO-INFRA-6485-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6485_online_cache_transition_eprocess_contract as mod
import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _contract() -> dict:
    return mod.build_contract_rows(root=REPO)


def _validate(contract: dict) -> dict:
    return mod.validate_contract_rows(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )


def _with_checksum(artifact: dict) -> dict:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_infra_6485_spec_declares_contract_fields_and_scenarios() -> None:
    """REQ-INFRA-6485: OpenSpec owns the transition contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6485") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6485-EVENTS",
        "SCENARIO-INFRA-6485-ACTIONS",
        "SCENARIO-INFRA-6485-EPROCESS",
        "SCENARIO-INFRA-6485-LIFECYCLE",
        "SCENARIO-INFRA-6485-ATTACKS",
        "SCENARIO-INFRA-6485-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "Exp5895 exact-slot",
        "Exp6479 readiness",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_infra_6485_events_actions_and_eprocess_validate() -> None:
    """SCENARIO-INFRA-6485-EVENTS/ACTIONS/EPROCESS: rows validate."""

    contract = _contract()
    report = _validate(contract)
    event_rows = contract["event_rows"]
    action_rows = contract["action_rows"]
    evidence_rows = contract["evidence_process_rows"]

    assert report["accepted"] is True
    assert report["reasons"] == []
    assert [row["event_type"] for row in event_rows] == list(mod.EVENT_TYPES)
    assert [row["chronology_index"] for row in event_rows] == list(range(len(mod.EVENT_TYPES)))
    assert [row["action_event_id"] for row in action_rows] == [
        row["event_id"] for row in event_rows
    ]
    assert {row["event_type"] for row in evidence_rows if row["adaptive_peek_charged"]} == {
        "verify",
        "propose",
        "admit",
        "promote",
        "restart",
    }
    assert all(row["cumulative_e_value"] >= 1.0 for row in evidence_rows)
    assert any(row["decision_kind"] == "adaptive_decision" for row in evidence_rows)
    assert any(row["decision_kind"] == "fixed_horizon_comparison" for row in evidence_rows)
    assert contract["frozen_null_receipt"]["frozen_at_monotonic_ns"] < min(
        row["monotonic_receipt_ns"] for row in event_rows
    )
    assert contract["evidence_process_spec"]["mixture"]["family"] == "geometric"
    assert contract["evidence_process_spec"]["thresholds_frozen_before_events"] is True


def test_scenario_infra_6485_lifecycle_and_attack_matrix_fail_closed() -> None:
    """SCENARIO-INFRA-6485-LIFECYCLE/ATTACKS: attacks are rejected."""

    contract = _contract()
    matrix = mod.mutation_attack_matrix(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    expected_reasons = {
        "duplicate_events": "duplicate_event_id",
        "backdated_writes": "action_backdated",
        "stated_write_without_action": "stated_write_without_action",
        "action_without_exact_admission": "action_without_exact_admission",
        "threshold_editing": "threshold_edited",
        "repeated_peeking": "peek_charge_mismatch",
        "missing_null": "missing_null",
        "rollback_omission": "rollback_omission",
        "tombstone_resurrection": "tombstone_resurrection",
        "restart_drift": "restart_drift",
    }
    for attack_id, reason in expected_reasons.items():
        assert by_id[attack_id]["fail_closed"] is True
        assert reason in by_id[attack_id]["reasons"]

    lifecycle_by_type = {row["lifecycle_type"]: row for row in contract["lifecycle_rows"]}
    assert lifecycle_by_type["tombstone"]["tombstone_persisted"] is True
    assert lifecycle_by_type["rollback"]["rollback_restored_prior_state"] is True
    assert lifecycle_by_type["restart"]["restart_replay_state_hash"] == lifecycle_by_type[
        "restart"
    ]["expected_state_hash"]

    with pytest.raises(ValueError, match="unknown attack_id"):
        mod.mutate_rows_for_attack("unknown", contract["rows"])


def test_req_infra_6485_validator_defensive_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6485: malformed receipt edges fail closed."""

    contract = _contract()

    bad_event_schema = deepcopy(contract["event_schema"])
    bad_event_schema["schema_hash"] = "sha256:" + "4" * 64
    assert "event_schema_mismatch" in mod.validate_contract_rows(
        contract["rows"],
        event_schema=bad_event_schema,
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad_action_schema = deepcopy(contract["action_receipt_schema"])
    bad_action_schema["schema_hash"] = "sha256:" + "5" * 64
    assert "action_schema_mismatch" in mod.validate_contract_rows(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=bad_action_schema,
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad_spec = deepcopy(contract["evidence_process_spec"])
    bad_spec["spec_hash"] = "sha256:" + "6" * 64
    assert "eprocess_spec_mismatch" in mod.validate_contract_rows(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=bad_spec,
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad_null = deepcopy(contract["frozen_null_receipt"])
    bad_null["null_id"] = "missing"
    bad_null["frozen_at_monotonic_ns"] = min(
        row["monotonic_receipt_ns"] for row in contract["event_rows"]
    )
    assert "missing_null" in mod.validate_contract_rows(
        contract["rows"],
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=bad_null,
    )["reasons"]

    bad = deepcopy(contract["rows"])
    bad[0]["fixture_label"] = "tampered"
    assert "row_hash_mismatch" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    event = next(row for row in bad if row["row_type"] == "event")
    event["event_id"] = "evt:bad"
    mod._refresh_row(event)
    assert "event_id_mismatch" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    event = next(row for row in bad if row["row_type"] == "event")
    event["event_payload_hash"] = "sha256:" + "7" * 64
    mod._refresh_row(event)
    assert "event_payload_hash_mismatch" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    bad = [row for row in bad if not (row["row_type"] == "action" and row["event_type"] == "evict")]
    assert "event_action_count_mismatch" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    action = next(row for row in bad if row["row_type"] == "action" and row["event_type"] == "admit")
    action["durable"] = False
    action["no_action_reason"] = "claimed_write_without_disk_receipt"
    mod._refresh_row(action)
    assert "stated_write_without_action" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    action = next(row for row in bad if row["row_type"] == "action" and row["event_type"] == "observe")
    action["no_action_reason"] = ""
    action["action_hash"] = mod._action_hash(action)
    mod._refresh_row(action)
    assert "explicit_no_action_missing" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    evidence = next(row for row in bad if row["row_type"] == "evidence_process")
    evidence["null_receipt_hash"] = "sha256:" + "9" * 64
    mod._refresh_row(evidence)
    assert "missing_null" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    evidence = next(row for row in bad if row["row_type"] == "evidence_process")
    evidence["threshold_hash"] = "sha256:" + "a" * 64
    evidence["stopping_boundary"] = 3.0
    evidence["adaptive_peek_charged"] = False
    evidence["null_frozen_before_event"] = False
    mod._refresh_row(evidence)
    reasons = mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]
    assert {"threshold_edited", "peek_charge_mismatch", "missing_null"} <= set(reasons)

    bad = [row for row in deepcopy(contract["rows"]) if row.get("row_type") != "lifecycle"]
    assert "lifecycle_count_mismatch" in mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    lifecycle = next(row for row in bad if row["row_type"] == "lifecycle" and row["lifecycle_type"] == "restart")
    lifecycle["restart_replay_state_hash"] = "sha256:" + "8" * 64
    lifecycle["active_tombstoned_event_ids"] = ["event-was-tombstoned"]
    mod._refresh_row(lifecycle)
    reasons = mod.validate_contract_rows(
        bad,
        event_schema=contract["event_schema"],
        action_receipt_schema=contract["action_receipt_schema"],
        evidence_process_spec=contract["evidence_process_spec"],
        frozen_null_receipt=contract["frozen_null_receipt"],
    )["reasons"]
    assert {"restart_drift", "tombstone_resurrection"} <= set(reasons)

    written = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "build-write.json",
        write=True,
        duration_s=1.0,
        tests_run=[],
    )
    assert json.loads((tmp_path / "build-write.json").read_text(encoding="utf-8")) == written


def test_scenario_infra_6485_artifact_recomputes_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6485-ARTIFACT: terminal artifact is row-recomputed."""

    artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=False,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_online_cache_transition_eprocess_contract"
    assert artifact["online_transition_contract_ready_score"] == 1.0
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"],
        event_schema=artifact["event_schema"],
        action_receipt_schema=artifact["action_receipt_schema"],
        evidence_process_spec=artifact["evidence_process_spec"],
        frozen_null_receipt=artifact["frozen_null_receipt"],
    )
    assert artifact["protected_files_unchanged"]["protected_files_unchanged"] is True
    assert artifact["preconditions_checked"]["exp6479_ready"] is True
    assert artifact["preconditions_checked"]["exp5895_exact_slot_reused"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete:")

    bad = _with_checksum({**artifact, "online_transition_contract_ready_score": 0.0})
    assert "online_transition_contract_ready_score mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    bad = _with_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "inference_substrate": "live_llm_inference"})
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "verifier_is_oracle": False})
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["protected_files_unchanged"] = False
    bad = _with_checksum(bad)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_provenance": {}})
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "field_principles": {}})
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = _with_checksum({**artifact, "honest_verdict": "done"})
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = {**artifact, "reproducibility_checksum": "sha256:bad"}
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_protected_unchanged",
            lambda root, before: {"protected_files_unchanged": False, "files": {}},
        )
        blocked = mod.build_artifact(
            root=REPO,
            result_path=tmp_path / "blocked.json",
            write=False,
            duration_s=1.0,
            tests_run=[],
        )
    assert blocked["status"] == "blocked_online_cache_transition_eprocess_contract"
    assert blocked["online_transition_contract_ready_score"] == 0.0
    assert "protected_files_unchanged" in blocked["gate_check_summary"]["failed_gates"]
    assert blocked["honest_verdict"].startswith("complete_blocked:")


def test_req_infra_6485_run_write_cli_and_substrate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-INFRA-6485: run writes, validates, and declares no-LLM work."""

    result = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        date="20260821",
        result_path=result,
        write=True,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["online_transition_contract_ready_score"] == 1.0

    result_cli = tmp_path / "cli.json"
    assert mod.main(["--date", "20260821", "--result-path", str(result_cli)]) == 0
    written = json.loads(result_cli.read_text(encoding="utf-8"))
    assert written["status"] == "complete_online_cache_transition_eprocess_contract"

    assert mod.main(["--validate", "--result-path", str(result_cli)]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out == {"errors": ["artifact missing"], "ok": False}

    classification = av._classify_inference_substrate(artifact)
    floor = av.duration_floor_for_artifact(artifact)
    report = av.verify_artifact(result)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert report["flags"] == []
