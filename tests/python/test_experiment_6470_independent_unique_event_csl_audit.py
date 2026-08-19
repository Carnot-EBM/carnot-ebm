"""Tests for Exp6470 independent unique-event CSL audit.

Spec refs: REQ-LEARN-6470, SCENARIO-LEARN-6470-INVENTORY,
SCENARIO-LEARN-6470-IDENTITY, SCENARIO-LEARN-6470-CHRONOLOGY,
SCENARIO-LEARN-6470-VETO, SCENARIO-LEARN-6470-EFFECTS,
SCENARIO-LEARN-6470-LIFECYCLE, SCENARIO-LEARN-6470-READY.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6470_independent_unique_event_csl_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical(value).encode("utf-8"))


def _write_raw(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (_canonical(payload) + "\n").encode("utf-8")
    path.write_bytes(data)
    return _sha_bytes(data), len(data)


def _raw_row(
    *,
    root: Path,
    experiment: str,
    event_id: str,
    unit_id: str,
    model: str,
    arm: str,
    sequence: int,
    completion: str,
) -> dict[str, Any]:
    path = root / "raw" / experiment / f"{event_id.replace(':', '-')}.json"
    payload = {
        "schema": f"fixture.{experiment}.raw_generation",
        "event_id": event_id,
        "event_sequence": sequence,
        "model": model,
        "arm": arm,
        "unit_id": unit_id,
        "completion_text": completion,
        "runner_receipt": {"backend": "fixture", "cpu_fallback": False},
    }
    digest, size = _write_raw(path, payload)
    return {
        "event_id": event_id,
        "event_sequence": sequence,
        "model": model,
        "arm": arm,
        "unit_id": unit_id,
        "path": str(path),
        "present": True,
        "raw_output_sha256": digest,
        "byte_length": size,
        "validated_before_parse": True,
        "runner_receipt": {"backend": "fixture", "cpu_fallback": False},
        "parse_receipt": {"confidence": 0.7, "signed_direction": 1},
    }


def _row(
    *,
    experiment: str,
    event_id: str,
    sequence: int,
    unit_id: str,
    raw: dict[str, Any],
    arm: str,
    model: str,
    interval: str = "future_held",
    exact_success: bool = False,
    admitted: bool = False,
    pre_head: str = "sha256:pre0000000000000000000000000000000000000000000000000000000000000",
    post_head: str | None = None,
    corrupt: bool = False,
    rejected_child_head: str = "",
) -> dict[str, Any]:
    actual_post = post_head if post_head is not None else pre_head
    selected_id = "candidate_1" if exact_success else "candidate_0"
    post_weights = {"route_first": 0.0, "verified_binding": 0.0}
    if admitted:
        target_feature = "verified_binding" if exact_success else "route_first"
        post_weights[target_feature] = 0.1 if exact_success else -0.1
    row: dict[str, Any] = {
        "schema": f"fixture.{experiment}.per_unit_row",
        "row_id": event_id,
        "event_id": event_id,
        "event_sequence": sequence,
        "chronological_index": sequence,
        "interval": interval,
        "model": model,
        "arm": arm,
        "unit_id": unit_id,
        "unit_hash": _sha_json({"unit_id": unit_id}),
        "raw_output_path": raw["path"],
        "raw_output_sha256": raw["raw_output_sha256"],
        "raw_output_validated_before_parse": True,
        "selected_candidate": {
            "candidate_id": selected_id,
            "action": "apply_verified_binding" if exact_success else "reuse_first_visible_binding",
            "features": ["verified_binding" if exact_success else "route_first"],
        },
        "checker_result": {
            "checker": "fixture_exact_checker",
            "checker_authority_passed": True,
            "ran_before_write": True,
            "exact_success": exact_success,
            "protected_ok": True,
            "abstained": False,
            "violation_codes": [] if exact_success else ["wrong_binding"],
        },
        "model_confidence": {"confidence": 0.7, "signed_direction": 1, "sign_is_authoritative": False},
        "exact_success": exact_success,
        "exact_sign": 1 if exact_success else -1,
        "applied_update_sign": 1 if admitted and exact_success else (0 if not admitted else -1),
        "magnitude": 0.1 if admitted else 0.0,
        "future_exact_outcome": exact_success if interval == "future_held" else None,
        "pre_state": {"head": pre_head, "weights": {"route_first": 0.0, "verified_binding": 0.0}},
        "post_state": {"head": actual_post, "weights": post_weights},
        "write_decision": {
            "checker_ran_before_write": True,
            "checker_authority_passed": True,
            "admitted": admitted,
            "post_head": actual_post,
            "rollback_pointer": pre_head,
            "veto_reason": "" if admitted else "frozen_or_zero",
        },
        "rollback_pointer": pre_head,
        "protected_outcome": {"case_id": f"protected-{unit_id}", "protected_ok": True},
        "selection_used_post_update_state": False,
        "future_label_visible_before_generation": False,
        "update_visible_to_chronological_index": sequence + 1,
        "cpu_fallback": False,
    }
    if experiment == "exp6469":
        row["corruption"] = {
            "scheduled": corrupt,
            "boundary": "forged_pass" if corrupt else "",
            "detected": corrupt,
            "blocked_before_release": corrupt,
            "receipt": {"detected_reason": "fixture_corruption"} if corrupt else {},
        }
        row["quarantine"] = {"quarantined": corrupt, "quarantine_hash": _sha_json({"event_id": event_id}) if corrupt else ""}
        row["tombstone"] = {
            "written": corrupt,
            "reason": "fixture_corruption" if corrupt else "",
            "tombstone_hash": _sha_json({"event_id": event_id, "kind": "tombstone"}) if corrupt else "",
            "rejected_child_head": rejected_child_head,
        }
        row["rollback"] = {
            "rejected_child_head": rejected_child_head if corrupt else "",
            "restored_head": pre_head,
            "restored_last_valid_head": corrupt,
        }
        row["update"] = {
            "exact_sign": 1 if exact_success else -1,
            "applied_update_sign": 0 if not admitted else (1 if exact_success else -1),
            "magnitude": 0.1 if admitted else 0.0,
            "touched_features": ["verified_binding" if exact_success else "route_first"],
            "weights": row["post_state"]["weights"],
        }
    row["row_hash"] = _sha_json(row)
    return row


def _lifecycle(event_id: str, transition: str, head: str, detail: dict[str, Any]) -> dict[str, Any]:
    row = {"event_id": event_id, "transition": transition, "head": head, "detail": detail}
    row["lifecycle_hash"] = _sha_json(row)
    return row


def _effect_6468(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({row["arm"] for row in rows})
    intervals = sorted({row["interval"] for row in rows})
    out: dict[str, Any] = {}
    for interval in intervals:
        out[interval] = {}
        for arm in arms:
            arm_rows = [row for row in rows if row["interval"] == interval and row["arm"] == arm]
            count = sum(1 for row in arm_rows if row["checker_result"]["exact_success"] is True)
            out[interval][arm] = {
                "row_count": len(arm_rows),
                "exact_success_count": count,
                "exact_yield": round(count / len(arm_rows), 12) if arm_rows else 0.0,
            }
    return out


def _protected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        ok = sum(1 for row in arm_rows if row["protected_outcome"]["protected_ok"] is True)
        out[arm] = {"row_count": len(arm_rows), "protected_ok_count": ok, "retention": round(ok / len(arm_rows), 12)}
    return {"by_arm": out, "regression_count": 0}


def _fixture_upstreams(tmp_path: Path) -> dict[str, Path]:
    model = "fixture/model"
    raw6468 = [
        _raw_row(root=tmp_path, experiment="exp6468", event_id="exp6468::00::frozen_factor_weights", unit_id="u6468-0", model=model, arm="frozen_factor_weights", sequence=0, completion="confidence 70"),
        _raw_row(root=tmp_path, experiment="exp6468", event_id="exp6468::00::self_signed_updates", unit_id="u6468-1", model=model, arm="self_signed_updates", sequence=1, completion="confidence 71"),
        _raw_row(root=tmp_path, experiment="exp6468", event_id="exp6468::00::verifier_bounded_exact_sign_updates", unit_id="u6468-2", model=model, arm="verifier_bounded_exact_sign_updates", sequence=2, completion="confidence 72"),
    ]
    rows6468 = [
        _row(experiment="exp6468", event_id=raw6468[0]["event_id"], sequence=0, unit_id="u6468-0", raw=raw6468[0], arm="frozen_factor_weights", model=model),
        _row(experiment="exp6468", event_id=raw6468[1]["event_id"], sequence=1, unit_id="u6468-1", raw=raw6468[1], arm="self_signed_updates", model=model, admitted=True, post_head="sha256:self"),
        _row(experiment="exp6468", event_id=raw6468[2]["event_id"], sequence=2, unit_id="u6468-2", raw=raw6468[2], arm="verifier_bounded_exact_sign_updates", model=model, exact_success=True, admitted=True, post_head="sha256:verifier"),
    ]
    events6468 = [{key: value for key, value in row.items() if key not in {"row_id", "row_hash", "raw_output_validated_before_parse"}} for row in rows6468]
    one_raw = {
        "passed": True,
        "event_row_count": len(events6468),
        "per_unit_row_count": len(rows6468),
        "raw_output_count": len(raw6468),
        "unique_raw_hash_count": len(raw6468),
        "duplicate_raw_hash_count": 0,
        "missing_event_link_count": 0,
    }
    exp6468 = {
        "status": "success_ready",
        "honest_verdict": "success: fixture",
        "unique_event_csl_ready_score": 1.0,
        "duration_s": 90.0,
        "inference_substrate": "live_llm_inference_local_gguf_unique_event_exact_veto",
        "sealed_chronological_manifest": {"sealed": True, "split_overlap_count": 0},
        "exposure_ledger": {
            "present": True,
            "written_before_inference": True,
            "future_held_outcome_exposure_count": 0,
            "future_held_prompt_exposure_count": 0,
            "future_held_update_admission_exposure_count": 0,
        },
        "per_unit_rows": {"rows": rows6468, "row_count": len(rows6468), "row_hash": _sha_json(rows6468), "written_before_aggregates": True},
        "event_rows": {"rows": events6468, "row_count": len(events6468), "row_hash": _sha_json(events6468)},
        "raw_output_manifest": {
            "rows": raw6468,
            "raw_output_count": len(raw6468),
            "unique_raw_hash_count": len(raw6468),
            "validated_before_parse_count": len(raw6468),
            "manifest_hash": _sha_json(raw6468),
        },
        "event_identity_manifest": {
            "event_count": len(events6468),
            "unique_event_id_count": len(events6468),
            "empty_event_id_count": 0,
            "duplicate_event_id_count": 0,
        },
        "exact_veto_before_write_receipts": {
            "admitted_write_count": 2,
            "checked_first_count": 2,
            "all_admitted_writes_checked_first": True,
            "checker_authority_failed_count": 0,
            "failed_authority_head_unchanged_count": 0,
        },
        "effect_by_arm_and_interval": _effect_6468(rows6468),
        "protected_case_retention": _protected(rows6468),
        "write_and_rollback_counts": {
            "by_arm": {
                "frozen_factor_weights": {"admitted_write_count": 0, "rollback_pointer_count": 1, "checker_veto_count": 0},
                "self_signed_updates": {"admitted_write_count": 1, "rollback_pointer_count": 1, "checker_veto_count": 0},
                "verifier_bounded_exact_sign_updates": {"admitted_write_count": 1, "rollback_pointer_count": 1, "checker_veto_count": 0},
            },
            "total_admitted_write_count": 2,
            "rollback_pointer_count": 3,
            "exact_veto_failed_write_count": 0,
        },
        "one_event_one_raw_hash_check": one_raw,
        "aggregate_row_recomputation": {"matches_reported": True, "checks": {}, "mismatch_fields": [], "row_count": len(rows6468), "row_hash": _sha_json(rows6468)},
        "attack_matrix": {"rows": [{"attack_id": "exact_veto_bypass", "critical": True, "fail_closed": True}], "all_critical_fail_closed": True, "readiness_promoted_attack_count": 0},
        "current_adversarial_findings": [],
        "gate_check_summary": {"failed_check_count": 0, "failed_checks": []},
        "cpu_fallback_count": 0,
        "model_file_and_embedded_tokenizer_hashes": {"base_ggufs_frozen": True},
    }

    raw6469 = [
        _raw_row(root=tmp_path, experiment="exp6469", event_id="exp6469::00::frozen_committed_head", unit_id="u6469-0", model=model, arm="frozen_committed_head", sequence=0, completion="confidence 73"),
        _raw_row(root=tmp_path, experiment="exp6469", event_id="exp6469::00::clean_exact_veto", unit_id="u6469-1", model=model, arm="clean_exact_veto", sequence=1, completion="confidence 74"),
        _raw_row(root=tmp_path, experiment="exp6469", event_id="exp6469::00::governed_corruption_restart", unit_id="u6469-2", model=model, arm="governed_corruption_restart", sequence=2, completion="confidence 75"),
        _raw_row(root=tmp_path, experiment="exp6469", event_id="exp6469::01::governed_corruption_restart", unit_id="u6469-3", model=model, arm="governed_corruption_restart", sequence=3, completion="confidence 76"),
    ]
    bad_child = "sha256:badchild"
    rows6469 = [
        _row(experiment="exp6469", event_id=raw6469[0]["event_id"], sequence=0, unit_id="u6469-0", raw=raw6469[0], arm="frozen_committed_head", model=model),
        _row(experiment="exp6469", event_id=raw6469[1]["event_id"], sequence=1, unit_id="u6469-1", raw=raw6469[1], arm="clean_exact_veto", model=model, exact_success=True, admitted=True, post_head="sha256:clean"),
        _row(experiment="exp6469", event_id=raw6469[2]["event_id"], sequence=2, unit_id="u6469-2", raw=raw6469[2], arm="governed_corruption_restart", model=model, exact_success=True, admitted=True, post_head="sha256:governed"),
        _row(experiment="exp6469", event_id=raw6469[3]["event_id"], sequence=3, unit_id="u6469-3", raw=raw6469[3], arm="governed_corruption_restart", model=model, exact_success=True, admitted=False, corrupt=True, rejected_child_head=bad_child),
    ]
    lifecycle = [
        _lifecycle(rows6469[3]["event_id"], "generated", rows6469[3]["pre_state"]["head"], {"raw": rows6469[3]["raw_output_sha256"]}),
        _lifecycle(rows6469[3]["event_id"], "raw_persisted", rows6469[3]["pre_state"]["head"], {"present": True}),
        _lifecycle(rows6469[3]["event_id"], "exact_veto", rows6469[3]["pre_state"]["head"], {"admitted": False, "checker_authority_passed": False}),
        _lifecycle(rows6469[3]["event_id"], "quarantine", rows6469[3]["pre_state"]["head"], {"boundary": "forged_pass"}),
        _lifecycle(rows6469[3]["event_id"], "tombstone", rows6469[3]["pre_state"]["head"], {"rejected_child_head": bad_child}),
        _lifecycle(rows6469[3]["event_id"], "rollback", rows6469[3]["pre_state"]["head"], {"restored_head": rows6469[3]["pre_state"]["head"]}),
    ]
    state_path = tmp_path / "store" / "post_rollback_head.json"
    state_payload = {
        "schema": "fixture.store_state",
        "head": "sha256:clean",
        "receipt_chain": ["sha256:pre", "sha256:clean"],
        "tombstoned_heads": [bad_child],
    }
    state_payload["state_hash"] = _sha_json(
        {
            "head": state_payload["head"],
            "receipt_chain": state_payload["receipt_chain"],
            "tombstoned_heads": state_payload["tombstoned_heads"],
        }
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state_payload, sort_keys=True), encoding="utf-8")
    exp6469 = {
        "status": "success_ready",
        "honest_verdict": "success: fixture",
        "corruption_restart_ready_score": 1.0,
        "duration_s": 8.0,
        "inference_substrate": "deterministic_verifier_plus_replay",
        "sealed_new_held_manifest": {
            "sealed": True,
            "future_outcomes_visible_before_generation": False,
            "unit_ids": [row["unit_id"] for row in rows6469],
            "unit_hashes": [row["unit_hash"] for row in rows6469],
        },
        "exposure_disjointness_receipts": {
            "all_disjoint": True,
            "unit_id_overlap_with_exp6468_count": 0,
            "event_id_overlap_with_exp6468_count": 0,
            "raw_hash_overlap_with_exp6468_count": 0,
        },
        "process_restart_receipts": {
            "restart_count": 1,
            "all_recovered_heads_match": True,
            "rows": [
                {
                    "parent_pid": 10,
                    "child_pid": 11,
                    "state_path": str(state_path),
                    "expected_head": "sha256:clean",
                    "recovered_head": "sha256:clean",
                    "recovered_from_disk": True,
                    "loaded_only_committed_head_and_receipt_chain": True,
                }
            ],
        },
        "per_unit_rows": {"rows": rows6469, "row_count": len(rows6469), "row_hash": _sha_json(rows6469)},
        "lifecycle_rows": {"rows": lifecycle, "row_count": len(lifecycle), "row_hash": _sha_json(lifecycle)},
        "raw_output_manifest": {
            "rows": raw6469,
            "raw_output_count": len(raw6469),
            "unique_raw_hash_count": len(raw6469),
            "duplicate_raw_hash_count": 0,
            "validated_before_parse_count": len(raw6469),
            "manifest_hash": _sha_json(raw6469),
        },
        "event_identity_manifest": {
            "event_count": len(rows6469),
            "unique_event_id_count": len(rows6469),
            "empty_event_id_count": 0,
            "duplicate_event_id_count": 0,
        },
        "corruption_precommitment": {"corrupt_event_count": 1, "rows": [{"event_id": rows6469[3]["event_id"], "boundary": "forged_pass"}]},
        "exact_veto_before_write_receipts": {
            "admitted_write_count": 2,
            "checked_first_count": 2,
            "all_admitted_writes_checked_first": True,
            "corrupt_event_count": 1,
            "corrupt_release_count": 0,
            "all_corrupt_blocked_before_release": True,
        },
        "clean_and_corrupt_effects": {
            "frozen_exact_yield": 0.0,
            "clean_exact_yield": 1.0,
            "governed_non_corrupt_exact_yield": 1.0,
            "clean_minus_frozen": 1.0,
            "governed_non_corrupt_minus_frozen": 1.0,
            "corrupt_event_count": 1,
            "corrupt_blocked_before_release_count": 1,
            "corrupt_release_count": 0,
        },
        "protected_case_retention": _protected(rows6469),
        "quarantine_tombstone_and_rollback_receipts": {
            "corrupt_event_count": 1,
            "quarantine_count": 1,
            "tombstone_count": 1,
            "rollback_success_count": 1,
            "all_tombstones_precede_rollback": True,
            "tombstoned_child_heads": [bad_child],
        },
        "non_resurrection_check": {
            "tombstoned_head_count": 1,
            "active_head_count": 1,
            "resurrected_heads": [],
            "corrupt_state_resurrection_count": 0,
            "post_restart_active_head_clean": True,
        },
        "aggregate_row_recomputation": {"matches_reported": True, "checks": {}, "mismatch_fields": [], "row_count": len(rows6469), "row_hash": _sha_json(rows6469)},
        "attack_matrix": {"rows": [{"attack_id": "replay", "critical": True, "fail_closed": True, "promoted_readiness": False}], "all_critical_attacks_fail_closed": True, "readiness_promoted_attack_count": 0},
        "current_adversarial_findings": [],
        "gate_check_summary": {"failed_check_count": 0, "failed_checks": []},
        "device_and_runner_receipts": {"cpu_fallback_count": 0},
    }
    exp6457 = {
        "status": "complete_null",
        "honest_verdict": "complete: fixture prior audit denied eligibility",
        "csl_audit_ready_score": 0.0,
        "duration_s": 3.0,
        "inference_substrate": "aggregation_from_upstream_artifacts_no_llm",
    }
    paths = {
        "exp6457": tmp_path / "experiment_6457.json",
        "exp6468": tmp_path / "experiment_6468.json",
        "exp6469": tmp_path / "experiment_6469.json",
    }
    paths["exp6457"].write_text(json.dumps(exp6457, sort_keys=True), encoding="utf-8")
    paths["exp6468"].write_text(json.dumps(exp6468, sort_keys=True), encoding="utf-8")
    paths["exp6469"].write_text(json.dumps(exp6469, sort_keys=True), encoding="utf-8")
    return paths


def test_req_learn_6470_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6470: OpenSpec owns the independent V556 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6470") : text.index("REQ-LEARN-6444")]

    for marker in (
        "SCENARIO-LEARN-6470-INVENTORY",
        "SCENARIO-LEARN-6470-IDENTITY",
        "SCENARIO-LEARN-6470-CHRONOLOGY",
        "SCENARIO-LEARN-6470-VETO",
        "SCENARIO-LEARN-6470-EFFECTS",
        "SCENARIO-LEARN-6470-LIFECYCLE",
        "SCENARIO-LEARN-6470-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "exactly one raw path and hash per credited event",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        assert f"csl_audit_eligible_score:{condition}" in mod.FIELD_PRINCIPLES


def test_scenario_learn_6470_inventory_identity_and_raw_hashes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6470-INVENTORY/IDENTITY: raw files bind one event."""

    upstreams = _fixture_upstreams(tmp_path)
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        upstream_paths=upstreams,
        duration_s=0.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["upstream_artifact_inventory"]["all_required_present"] is True
    assert artifact["raw_file_inventory_and_hashes"]["missing_count"] == 0
    assert artifact["raw_file_inventory_and_hashes"]["path_hash_mismatch_count"] == 0
    assert artifact["independent_event_identity_recomputation"]["one_raw_per_event"] is True
    assert artifact["independent_event_identity_recomputation"]["raw_reuse_event_count"] == 0
    assert artifact["independent_event_identity_recomputation"]["credited_event_count"] == 7
    assert artifact["csl_audit_eligible_score"] == 1.0
    assert artifact["status"] == "success_ready"
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_6470_chronology_veto_effects_and_lifecycle(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6470-CHRONOLOGY/VETO/EFFECTS/LIFECYCLE: rows own gates."""

    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "audit.json",
        upstream_paths=_fixture_upstreams(tmp_path),
        duration_s=0.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    assert artifact["independent_exposure_ledger"]["held_exposure_count"] == 0
    assert artifact["independent_exposure_ledger"]["held_disjointness_passed"] is True
    assert artifact["exact_veto_order_recomputation"]["all_admitted_writes_checked_first"] is True
    assert artifact["exact_veto_order_recomputation"]["failed_checker_write_count"] == 0
    assert artifact["independent_effect_recomputation"]["exp6468"]["future_held"]["verifier_bounded_exact_sign_updates"]["exact_yield"] == 1.0
    assert artifact["independent_effect_recomputation"]["exp6469"]["clean_minus_frozen"] == 1.0
    assert artifact["protected_case_recomputation"]["regression_count"] == 0
    assert artifact["rollback_restart_and_non_resurrection_replay"]["corrupt_state_resurrection_count"] == 0
    assert artifact["rollback_restart_and_non_resurrection_replay"]["lifecycle_order_passed"] is True
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["upstream_vs_independent_field_comparison"]["critical_mismatch_count"] == 0
    assert artifact["attack_matrix"]["all_critical_attacks_fail_closed"] is True


def test_scenario_learn_6470_missing_raw_and_reuse_block_eligibility(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6470-READY: missing or reused raw evidence blocks eligibility."""

    upstreams = _fixture_upstreams(tmp_path / "missing")
    exp6468 = json.loads(upstreams["exp6468"].read_text(encoding="utf-8"))
    Path(exp6468["raw_output_manifest"]["rows"][0]["path"]).unlink()
    missing = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "missing-audit.json",
        upstream_paths=upstreams,
        duration_s=0.25,
        write=False,
    )
    assert missing["csl_audit_eligible_score"] == 0.0
    assert missing["raw_file_inventory_and_hashes"]["missing_count"] == 1
    assert any(row["kind"] == "raw_file_missing" for row in missing["critical_discrepancies"])
    assert missing["gate_check_summary"]["failed_check_count"] > 0

    upstreams = _fixture_upstreams(tmp_path / "reuse")
    exp6468 = json.loads(upstreams["exp6468"].read_text(encoding="utf-8"))
    first = exp6468["raw_output_manifest"]["rows"][0]
    second = exp6468["raw_output_manifest"]["rows"][1]
    raw_bytes = Path(first["path"]).read_bytes()
    Path(second["path"]).write_bytes(raw_bytes)
    second["raw_output_sha256"] = first["raw_output_sha256"]
    second["byte_length"] = first["byte_length"]
    for row in exp6468["per_unit_rows"]["rows"]:
        if row["event_id"] == second["event_id"]:
            row["raw_output_sha256"] = first["raw_output_sha256"]
    for row in exp6468["event_rows"]["rows"]:
        if row["event_id"] == second["event_id"]:
            row["raw_output_sha256"] = first["raw_output_sha256"]
    upstreams["exp6468"].write_text(json.dumps(exp6468, sort_keys=True), encoding="utf-8")
    reuse = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "reuse-audit.json",
        upstream_paths=upstreams,
        duration_s=0.25,
        write=False,
    )
    assert reuse["csl_audit_eligible_score"] == 0.0
    assert reuse["independent_event_identity_recomputation"]["duplicate_raw_hash_count"] == 1
    assert reuse["independent_event_identity_recomputation"]["raw_reuse_event_count"] == 2
    assert reuse["independent_event_identity_recomputation"]["credited_event_count"] == 5
    assert any(row["kind"] == "raw_output_reuse" for row in reuse["critical_discrepancies"])


def test_scenario_learn_6470_validation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6470-READY: schema and gate mutations fail validation."""

    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "audit.json",
        upstream_paths=_fixture_upstreams(tmp_path),
        duration_s=0.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=False,
    )

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        ("raw_file_inventory", lambda data: data["raw_file_inventory_and_hashes"].__setitem__("missing_count", 1)),
        ("event_identity", lambda data: data["independent_event_identity_recomputation"].__setitem__("one_raw_per_event", False)),
        ("exact_veto", lambda data: data["exact_veto_order_recomputation"].__setitem__("all_admitted_writes_checked_first", False)),
        ("aggregate", lambda data: data["aggregate_row_recomputation"].__setitem__("matches_reported", False)),
        ("attack_matrix", lambda data: data["attack_matrix"].__setitem__("all_critical_attacks_fail_closed", False)),
        ("eligible_score", lambda data: data.__setitem__("csl_audit_eligible_score", 0.0)),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"checksum", "required_fields", "eligible_score"}:
            bad["critical_discrepancies"] = mod.current_adversarial_findings(bad)
            bad["gate_check_summary"] = mod.gate_check_summary(bad)
            bad["csl_audit_eligible_score"] = mod.eligible_score(bad)
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6470_helper_edges_cover_blockers(tmp_path: Path) -> None:
    """REQ-LEARN-6470: helper edge paths preserve blocked evidence."""

    assert mod.sha256_file(tmp_path / "missing") is None
    assert mod._float_close("not-a-number", 1.0) is False
    assert mod.duration_floor("verifier_ensemble_against_cached_candidates") == 1.0
    assert mod.source_hashes()

    missing_payloads, missing_inventory = mod.upstream_inventory({"missing": tmp_path / "missing.json"})
    assert missing_payloads["missing"] == {}
    assert missing_inventory["missing_count"] == 1
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    malformed_payloads, malformed_inventory = mod.upstream_inventory({"bad": bad_json})
    assert malformed_payloads["bad"] == {}
    assert malformed_inventory["malformed_count"] == 1

    upstreams = _fixture_upstreams(tmp_path / "edges")
    payloads, _ = mod.upstream_inventory(upstreams)
    raw_path = Path(payloads["exp6468"]["raw_output_manifest"]["rows"][0]["path"])
    raw_path.write_text("", encoding="utf-8")
    raw_inventory = mod.raw_file_inventory_and_hashes(payloads)
    assert raw_inventory["zero_byte_count"] == 1
    assert raw_inventory["malformed_count"] == 1
    assert raw_inventory["path_hash_mismatch_count"] == 1

    valid = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "valid.json",
        upstream_paths=_fixture_upstreams(tmp_path / "valid"),
        duration_s=0.25,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=True,
    )
    assert mod.validate_artifact(tmp_path / "valid.json") is True

    rows = deepcopy(valid["per_unit_rows"]["rows"])
    admitted = next(row for row in rows if row["write_decision"]["admitted"] is True)
    touched = admitted["selected_candidate"]["features"][0]
    admitted["post_state"]["weights"][touched] = 99.0
    admitted["checker_result"]["exact_success"] = not admitted["checker_result"]["exact_success"]
    veto = mod.exact_veto_order_recomputation(rows)
    assert veto["write_effect_mismatch_count"] >= 1
    assert veto["checker_result_mismatch_count"] >= 1

    lifecycle_payloads, _ = mod.upstream_inventory(_fixture_upstreams(tmp_path / "lifecycle"))
    corrupt_event = lifecycle_payloads["exp6469"]["corruption_precommitment"]["rows"][0]["event_id"]
    lifecycle_rows = lifecycle_payloads["exp6469"]["lifecycle_rows"]["rows"]
    ordered = sorted(
        lifecycle_rows,
        key=lambda row: 0 if row["transition"] == "rollback" else 1,
    )
    lifecycle_payloads["exp6469"]["lifecycle_rows"]["rows"] = ordered
    bad_order = mod.rollback_restart_and_non_resurrection_replay(lifecycle_payloads)
    assert corrupt_event in bad_order["lifecycle_order_failures"]

    missing_transition_payloads, _ = mod.upstream_inventory(_fixture_upstreams(tmp_path / "missing-transition"))
    missing_transition_payloads["exp6469"]["lifecycle_rows"]["rows"] = [
        row
        for row in missing_transition_payloads["exp6469"]["lifecycle_rows"]["rows"]
        if row["transition"] != "exact_veto"
    ]
    missing_transition = mod.rollback_restart_and_non_resurrection_replay(missing_transition_payloads)
    assert corrupt_event in missing_transition["lifecycle_order_failures"]

    branch_artifact = deepcopy(valid)
    branch_artifact["upstream_artifact_inventory"]["all_required_present"] = False
    branch_artifact["raw_file_inventory_and_hashes"]["zero_byte_count"] = 1
    branch_artifact["raw_file_inventory_and_hashes"]["path_hash_mismatch_count"] = 1
    branch_artifact["independent_event_identity_recomputation"]["duplicate_event_id_count"] = 1
    branch_artifact["independent_exposure_ledger"]["held_exposure_count"] = 1
    branch_artifact["exact_veto_order_recomputation"]["checker_result_mismatch_count"] = 1
    branch_artifact["rollback_restart_and_non_resurrection_replay"]["lifecycle_order_passed"] = False
    branch_artifact["duration_recomputation"]["all_duration_floors_passed"] = False
    findings = {row["kind"] for row in mod.current_adversarial_findings(branch_artifact)}
    assert {
        "upstream_artifact_unavailable",
        "raw_file_zero_byte",
        "raw_hash_mismatch",
        "duplicate_event_id",
        "held_exposure",
        "effect_or_checker_recompute",
        "lifecycle_replay",
        "duration_floor",
    } <= findings
