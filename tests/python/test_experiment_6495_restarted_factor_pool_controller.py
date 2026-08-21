"""Tests for Exp6495 restarted factor-pool controller.

Spec refs: REQ-INFRA-6495, SCENARIO-INFRA-6495-PAIRED-EVIDENCE,
SCENARIO-INFRA-6495-DECISIONS, SCENARIO-INFRA-6495-CAPACITY,
SCENARIO-INFRA-6495-ROLLBACK-RESTART, SCENARIO-INFRA-6495-ADMISSION,
SCENARIO-INFRA-6495-ATTACKS, SCENARIO-INFRA-6495-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6495_restarted_factor_pool_controller as mod
import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _contract() -> dict:
    return mod.build_controller_rows(root=REPO)


def _validate(contract: dict) -> dict:
    return mod.validate_controller_rows(
        contract["rows"],
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )


def _with_checksum(artifact: dict) -> dict:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_infra_6495_spec_declares_controller_contract() -> None:
    """REQ-INFRA-6495: OpenSpec owns the controller contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6495") : text.index("REQ-INFRA-6351")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-INFRA-6495-PAIRED-EVIDENCE",
        "SCENARIO-INFRA-6495-DECISIONS",
        "SCENARIO-INFRA-6495-CAPACITY",
        "SCENARIO-INFRA-6495-ROLLBACK-RESTART",
        "SCENARIO-INFRA-6495-ADMISSION",
        "SCENARIO-INFRA-6495-ATTACKS",
        "SCENARIO-INFRA-6495-ARTIFACT",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_infra_6495_paired_evidence_and_decisions_validate() -> None:
    """SCENARIO-INFRA-6495-PAIRED-EVIDENCE/DECISIONS: rows validate."""

    contract = _contract()
    report = _validate(contract)
    event_rows = contract["event_rows"]
    evidence_rows = contract["evidence_update_rows"]
    decision_rows = contract["decision_action_rows"]

    assert report["accepted"] is True
    assert report["reasons"] == []
    assert [row["chronology_index"] for row in event_rows] == list(range(len(event_rows)))
    assert len(evidence_rows) == len(event_rows) * 2
    assert {row["process_kind"] for row in evidence_rows} == {"reuse", "spawn"}
    assert len({row["spend_token"] for row in evidence_rows}) == len(evidence_rows)
    assert all(row["multiplicity_corrected"] is True for row in evidence_rows)
    assert all(row["threshold_hash"] == contract["evidence_process_spec"]["threshold_hash"] for row in evidence_rows)
    assert all(row["null_frozen_before_event"] is True for row in evidence_rows)

    decisions = {row["fixture_id"]: row for row in decision_rows}
    assert decisions["positive_spawn_alpha"]["decision"] == "spawn"
    assert decisions["recurrent_reuse_alpha"]["decision"] == "reuse"
    assert decisions["null_defer"]["decision"] == "defer"
    assert decisions["contradictory_defer"]["decision"] == "defer"
    assert decisions["corrupted_quarantine"]["decision"] == "quarantine"
    assert decisions["capacity_overflow_spawn_gamma"]["action_type"] == "evict_then_spawn_write"
    assert decisions["outside_authority_no_write"]["decision"] == "no_write"
    assert decisions["outside_authority_no_write"]["durable"] is False
    assert decisions["outside_authority_no_write"]["no_write_reason"] == "outside_exact_authority"


def test_scenario_infra_6495_capacity_rollback_restart_and_admission() -> None:
    """SCENARIO-INFRA-6495-CAPACITY/ROLLBACK-RESTART/ADMISSION: lifecycle holds."""

    contract = _contract()
    state_by_fixture = {row["fixture_id"]: row for row in contract["pool_state_rows"]}
    decisions = {row["fixture_id"]: row for row in contract["decision_action_rows"]}
    admissions = {row["action_id"]: row for row in contract["exact_admission_receipts"]}

    assert all(row["active_factor_count"] <= mod.POOL_CAPACITY for row in contract["pool_state_rows"])
    assert state_by_fixture["capacity_overflow_spawn_gamma"]["active_factor_ids"] == [
        "factor_alpha",
        "factor_gamma",
    ]
    assert state_by_fixture["tombstone_beta"]["tombstoned_factor_ids"] == ["factor_beta"]
    assert state_by_fixture["rollback_to_pre_overflow"]["active_factor_ids"] == ["factor_alpha"]
    assert state_by_fixture["rollback_to_pre_overflow"]["rollback_suppressed_tombstones"] == [
        "factor_beta"
    ]
    assert state_by_fixture["restart_replay"]["active_factor_ids"] == ["factor_alpha"]
    assert state_by_fixture["restart_replay"]["restart_replay_state_hash"] == state_by_fixture[
        "restart_replay"
    ]["state_hash"]

    durable_actions = [row for row in contract["decision_action_rows"] if row["durable"]]
    assert durable_actions
    for action in durable_actions:
        receipt = admissions[action["action_id"]]
        assert receipt["exact_admission_passed"] is True
        assert receipt["event_id"] == action["event_id"]
        assert receipt["exact_admission_hash"] == action["exact_admission_hash"]
        assert receipt["authority"] == "exact_fixture_verifier"

    no_write_actions = [row for row in contract["decision_action_rows"] if not row["durable"]]
    assert no_write_actions
    assert all(row["no_write_reason"] for row in no_write_actions)
    assert decisions["restart_replay"]["restart_epoch_after"] == 1


def test_scenario_infra_6495_attack_matrix_fails_closed() -> None:
    """SCENARIO-INFRA-6495-ATTACKS: controller attacks are rejected."""

    contract = _contract()
    matrix = mod.mutation_attack_matrix(
        contract["rows"],
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )
    by_id = {row["attack_id"]: row for row in matrix["rows"]}

    assert set(by_id) == set(mod.ATTACK_IDS)
    assert matrix["all_critical_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    expected_reasons = {
        "duplicate_event_id": "duplicate_event_id",
        "backdated_event": "event_chronology_not_monotonic",
        "adaptive_peek_reuse": "evidence_spend_token_reused",
        "threshold_edit": "threshold_edited",
        "outside_authority_write": "durable_write_without_exact_admission",
        "capacity_overflow": "capacity_exceeded",
        "rollback_target_corruption": "rollback_state_mismatch",
        "tombstone_resurrection": "tombstone_resurrection",
    }
    for attack_id, reason in expected_reasons.items():
        assert by_id[attack_id]["fail_closed"] is True
        assert reason in by_id[attack_id]["reasons"]

    with pytest.raises(ValueError, match="unknown attack_id"):
        mod.mutate_rows_for_attack("unknown", contract["rows"])


def test_req_infra_6495_validator_defensive_edges(tmp_path: Path) -> None:
    """REQ-INFRA-6495: malformed rows, specs, and artifacts fail closed."""

    contract = _contract()
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)

    deferred = mod._decision_for_event(
        {
            "event_type": "spawn",
            "authority": "shadow_scorer",
            "exact_admission_passed": False,
        },
        [
            {"process_kind": "reuse", "e_value_after_spend": 1.0},
            {"process_kind": "spawn", "e_value_after_spend": 2.5},
        ],
        {},
    )
    assert deferred == ("defer", "defer_no_write", "missing_exact_authority", False)
    boundary_miss = mod._decision_for_event(
        {
            "event_type": "defer",
            "authority": "exact_fixture_verifier",
            "exact_admission_passed": True,
        },
        [
            {"process_kind": "reuse", "e_value_after_spend": 2.5},
            {"process_kind": "spawn", "e_value_after_spend": 1.0},
        ],
        {},
    )
    assert boundary_miss == ("defer", "defer_no_write", "decision_boundary_not_met", False)

    bad_controller = deepcopy(contract["controller_spec"])
    bad_controller["spec_hash"] = "sha256:" + "1" * 64
    assert "controller_spec_mismatch" in mod.validate_controller_rows(
        contract["rows"],
        controller_spec=bad_controller,
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad_evidence = deepcopy(contract["evidence_process_spec"])
    bad_evidence["threshold_hash"] = "sha256:" + "2" * 64
    assert "evidence_process_spec_mismatch" in mod.validate_controller_rows(
        contract["rows"],
        controller_spec=contract["controller_spec"],
        evidence_process_spec=bad_evidence,
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad_multiplicity = deepcopy(contract["multiplicity_spec"])
    bad_multiplicity["spec_hash"] = "sha256:" + "3" * 64
    assert "multiplicity_spec_mismatch" in mod.validate_controller_rows(
        contract["rows"],
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=bad_multiplicity,
    )["reasons"]

    bad = deepcopy(contract["rows"])
    bad[0]["fixture_label"] = "tampered"
    assert "row_hash_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    event = next(row for row in bad if row["row_type"] == "event")
    event["event_id"] = "event:bad"
    mod._refresh_row(event)
    assert "event_id_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    event = next(row for row in bad if row["row_type"] == "event")
    event["event_payload_hash"] = "sha256:" + "4" * 64
    mod._refresh_row(event)
    assert "event_payload_hash_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = [row for row in deepcopy(contract["rows"]) if row.get("row_type") != "pool_state"]
    assert "pool_state_count_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = [row for row in deepcopy(contract["rows"]) if row.get("row_type") != "exact_admission"]
    assert "exact_admission_count_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    evidence = next(row for row in bad if row["row_type"] == "evidence_update")
    evidence["adaptive_peek_charged"] = False
    evidence["null_frozen_before_event"] = False
    evidence["corrected_increment"] = 9.0
    mod._refresh_row(evidence)
    reasons = mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]
    assert {"evidence_not_charged", "missing_frozen_null", "multiplicity_correction_mismatch"} <= set(reasons)

    bad = deepcopy(contract["rows"])
    evidence = next(row for row in bad if row["row_type"] == "evidence_update" and row["process_kind"] == "spawn")
    evidence["process_kind"] = "reuse"
    mod._refresh_row(evidence)
    assert "paired_evidence_count_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    decision = next(row for row in bad if row["row_type"] == "decision_action" and row["durable"])
    decision["post_state_hash"] = "sha256:" + "5" * 64
    decision["action_hash"] = mod._action_hash(decision)
    mod._refresh_row(decision)
    assert "state_hash_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    decision = next(row for row in bad if row["row_type"] == "decision_action" and row["durable"])
    decision["decision"] = "tampered_without_action_hash"
    mod._refresh_row(decision)
    assert "action_hash_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    decision = next(row for row in bad if row["row_type"] == "decision_action" and not row["durable"])
    decision["no_write_reason"] = ""
    decision["action_hash"] = mod._action_hash(decision)
    mod._refresh_row(decision)
    assert "no_write_reason_missing" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    decision = next(row for row in bad if row["row_type"] == "decision_action")
    decision["pre_state_hash"] = "sha256:" + "6" * 64
    decision["action_hash"] = mod._action_hash(decision)
    mod._refresh_row(decision)
    assert "state_hash_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    bad = deepcopy(contract["rows"])
    receipt = next(row for row in bad if row["row_type"] == "exact_admission" and row["exact_admission_passed"])
    receipt["event_id"] = "event:wrong"
    receipt["exact_admission_hash"] = "sha256:" + "7" * 64
    mod._refresh_row(receipt)
    reasons = mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]
    assert {"exact_admission_event_mismatch", "exact_admission_hash_mismatch"} <= set(reasons)

    bad = deepcopy(contract["rows"])
    state = next(row for row in bad if row["row_type"] == "pool_state" and row["fixture_id"] == "restart_replay")
    state["restart_replay_state_hash"] = "sha256:" + "8" * 64
    mod._refresh_row(state)
    assert "restart_state_mismatch" in mod.validate_controller_rows(
        bad,
        controller_spec=contract["controller_spec"],
        evidence_process_spec=contract["evidence_process_spec"],
        multiplicity_spec=contract["multiplicity_spec"],
    )["reasons"]

    artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / "artifact.json",
        write=True,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []

    bad_artifact = deepcopy(artifact)
    bad_artifact["per_unit_rows"] = bad_artifact["per_unit_rows"][:-1]
    _with_checksum(bad_artifact)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "factor_pool_controller_ready_score": 0.0})
    assert "factor_pool_controller_ready_score mismatch" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "inference_substrate": "live_llm"})
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "verifier_is_oracle": False})
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad_artifact)

    bad_artifact = deepcopy(artifact)
    bad_artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] = False
    _with_checksum(bad_artifact)
    assert "protected_files_unchanged must be true" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "field_provenance": {}})
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "field_principles": {}})
    assert "missing field_principles entry: status" in mod.validate_artifact(bad_artifact)

    bad_artifact = _with_checksum({**artifact, "honest_verdict": "done"})
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad_artifact)

    bad_artifact = {**artifact, "reproducibility_checksum": "sha256:bad"}
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_artifact)

    bad_artifact = deepcopy(artifact)
    del bad_artifact["status"]
    assert "missing required field: status" in mod.validate_artifact(bad_artifact)


def test_scenario_infra_6495_artifact_gate_receipts_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-INFRA-6495-ARTIFACT: artifact validates and gates are exact."""

    artifact = mod.build_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        write=False,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_restarted_factor_pool_controller"
    assert artifact["factor_pool_controller_ready_score"] == 1.0
    assert artifact["upstream_gate_receipt"]["field"] == "v560_lineage_lock_ready_score"
    assert artifact["upstream_gate_receipt"]["expected"] == 1.0
    assert artifact["upstream_gate_receipt"]["observed"] == 1.0
    assert artifact["dependency_receipts"]["exp6479"]["readiness_fields"] == {
        "factor_cache_shadow_adapter_ready_score": 1.0
    }
    assert artifact["dependency_receipts"]["exp6485"]["readiness_fields"] == {
        "online_transition_contract_ready_score": 1.0
    }
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"],
        controller_spec=artifact["controller_spec"],
        evidence_process_spec=artifact["evidence_process_spec"],
        multiplicity_spec=artifact["multipity_spec"]
        if "multipity_spec" in artifact
        else artifact["multiplicity_spec"],
    )
    assert artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is True
    assert artifact["preconditions_checked"]["lineage_lock_ready"] is True
    assert artifact["preconditions_checked"]["adapter_ready"] is True
    assert artifact["preconditions_checked"]["transition_contract_ready"] is True
    assert artifact["preconditions_checked"]["durable_store_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["honest_verdict"].startswith("complete_restarted_factor_pool_controller:")
    assert mod.validate_artifact(artifact) == []

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_protected_unchanged",
            lambda root, before: {"active_roadmap_and_conductor_unchanged": False, "files": {}},
        )
        blocked = mod.build_artifact(
            root=REPO,
            result_path=tmp_path / "blocked.json",
            write=False,
            duration_s=1.0,
            tests_run=[],
        )
    assert blocked["status"] == "blocked_restarted_factor_pool_controller"
    assert blocked["factor_pool_controller_ready_score"] == 0.0
    assert "protected_files_unchanged" in blocked["gate_check_summary"]["failed_gates"]
    assert blocked["honest_verdict"].startswith("blocked_restarted_factor_pool_controller:")
    assert mod.validate_artifact(blocked) == ["protected_files_unchanged must be true"]

    result = tmp_path / "run.json"
    written = mod.run(
        date="20260821",
        result_path=result,
        write=True,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )
    assert json.loads(result.read_text(encoding="utf-8")) == written
    assert written["factor_pool_controller_ready_score"] == 1.0

    cli_result = tmp_path / "cli.json"
    assert mod.main(["--date", "20260821", "--result-path", str(cli_result)]) == 0
    cli_payload = json.loads(cli_result.read_text(encoding="utf-8"))
    assert cli_payload["status"] == "complete_restarted_factor_pool_controller"

    assert mod.main(["--validate", "--result-path", str(cli_result)]) == 0
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert out == {"errors": ["artifact missing"], "ok": False}

    classification = av._classify_inference_substrate(artifact)
    floor = av.duration_floor_for_artifact(artifact)
    report_path = tmp_path / "av.json"
    report_path.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = av.verify_artifact(report_path)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert floor == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert report["flags"] == []
