"""Tests for Exp6457 independent verifier-bounded CSL audit.

Spec refs: REQ-LEARN-6457, SCENARIO-LEARN-6457-SPEC,
SCENARIO-LEARN-6457-INVENTORY, SCENARIO-LEARN-6457-REDUCERS,
SCENARIO-LEARN-6457-AUTHORITY, SCENARIO-LEARN-6457-SAFETY,
SCENARIO-LEARN-6457-READY.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot import experiment_6457_independent_verifier_bounded_csl_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _base_row(
    *,
    exp: str,
    arm: str,
    unit: str,
    index: int,
    exact_success: bool,
    future: bool,
    model: str = "fixture/model",
) -> dict[str, Any]:
    sign = 1 if exact_success else -1
    return {
        "row_id": f"{unit}::{arm}",
        "unit_id": unit,
        "model": model,
        "model_family": "fixture",
        "arm": arm,
        "chronological_index": index,
        "future_eval_unit": future,
        "future_exact_outcome": exact_success if future else None,
        "candidate_pool_path": f"/tmp/{exp}/{unit}.json",
        "candidate_pool_sha256": f"sha256:{exp}{unit}".ljust(71, "0"),
        "candidate_hashes": [f"sha256:{exp}{unit}c0".ljust(71, "0")],
        "selected_candidate": {
            "candidate_id": "candidate_0",
            "candidate_hash": f"sha256:{exp}{unit}c0".ljust(71, "0"),
            "features": ["route_first"],
        },
        "exact_result": {
            "exact_success": exact_success,
            "protected_ok": True,
            "abstained": False,
            "violation_codes": [] if exact_success else ["wrong_binding"],
        },
        "exact_sign": sign,
        "applied_update_sign": sign if arm != mod.FROZEN_ARM else 0,
        "magnitude": 0.25 if arm != mod.FROZEN_ARM else 0.0,
        "teacher_signal": {
            "signed_direction": -sign,
            "sign_is_authoritative": False,
            "nonnegative_magnitude_evidence": 0.5,
        },
        "pre_update_weights": {
            "route_first": 0.0,
            "verified_binding": 0.0,
            "protected_shortcut": 0.0,
            "abstain_guard": 0.0,
        },
        "post_update_weights": {
            "route_first": float(sign) * 0.25 if arm != mod.FROZEN_ARM else 0.0,
            "verified_binding": 0.0,
            "protected_shortcut": 0.0,
            "abstain_guard": 0.0,
        },
        "protected_outcome": {"protected_ok": True},
        "selection_used_post_update_state": False,
        "update_visible_to_chronological_index": index + 1,
        "head_before": f"sha256:{exp}{unit}{arm}before".ljust(71, "0"),
        "head_after": f"sha256:{exp}{unit}{arm}after".ljust(71, "0"),
        "transaction_hash": f"sha256:{exp}{unit}{arm}tx".ljust(71, "0"),
        "timing": {"duration_s": 0.1},
        "cpu_fallback": False,
    }


def _prospective_payload() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in (mod.FROZEN_ARM, mod.TEACHER_ARM, mod.VERIFIER_ARM):
        rows.append(
            _base_row(exp="6455", arm=arm, unit="u0", index=0, exact_success=False, future=False)
        )
        rows.append(
            _base_row(
                exp="6455",
                arm=arm,
                unit="u1",
                index=1,
                exact_success=arm == mod.VERIFIER_ARM,
                future=True,
            )
        )
    payload = {
        "status": "success_ready",
        "honest_verdict": "success: fixture",
        "duration_s": 95.0,
        "inference_substrate": "live_llm_inference_local_gguf_sota",
        "verifier_bounded_csl_ready_score": 1.0,
        "per_unit_rows": {"row_count": len(rows), "rows": rows},
        "future_exact_yield_delta": {"verifier_bounded_minus_frozen": -999.0},
    }
    return payload


def _held_payload() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in (mod.FROZEN_ARM, mod.CLEAN_ARM, mod.GOVERNED_ARM):
        rows.append(
            _base_row(exp="6456", arm=arm, unit="h0", index=0, exact_success=False, future=False)
        )
        rows.append(
            _base_row(
                exp="6456",
                arm=arm,
                unit="h1",
                index=1,
                exact_success=arm != mod.FROZEN_ARM,
                future=True,
            )
        )
    corrupt = rows[2]
    corrupt["corrupt_event"] = {"scheduled": True, "detected": True}
    corrupt["checker_response"] = {
        "authoritative": False,
        "transport_corrupted": True,
        "exact_success": True,
    }
    corrupt["quarantine"] = {"quarantined": True, "quarantine_reason": "path_hash_break"}
    corrupt["rollback"] = {
        "restored_last_good_head": True,
        "rejected_child_head": "sha256:bad".ljust(71, "0"),
        "restored_head": corrupt["head_before"],
    }
    corrupt["tombstone"] = {"written": True, "tombstone_hash": "sha256:tomb".ljust(71, "0")}
    corrupt["update"] = {"admitted": False, "exact_sign": -1, "applied_update_sign": 0}
    corrupt["path_receipts"] = {"path_hash_matches": False}
    for row in rows:
        row.setdefault("corrupt_event", {"scheduled": False, "detected": False})
        row.setdefault("checker_response", {"authoritative": True, "transport_corrupted": False})
        row.setdefault("quarantine", {"quarantined": False})
        row.setdefault("rollback", {"restored_last_good_head": False, "rejected_child_head": ""})
        row.setdefault("tombstone", {"written": False, "tombstone_hash": ""})
        row.setdefault(
            "update",
            {
                "admitted": row["arm"] != mod.FROZEN_ARM and row["exact_result"]["exact_success"],
                "exact_sign": row["exact_sign"],
                "applied_update_sign": row["applied_update_sign"],
            },
        )
        row.setdefault("path_receipts", {"path_hash_matches": True})
        row["process"] = {
            "parent_pid": 100,
            "child_pid": 101 + row["chronological_index"],
            "exit_code": 0,
            "recovered_from_disk": True,
            "head_hash_valid": True,
            "transaction_ancestry_valid": True,
            "inherited_memory_state_visible": False,
            "state_path": f"/tmp/held/{row['row_id']}.json",
        }
    return {
        "status": "success_ready",
        "honest_verdict": "success: fixture",
        "duration_s": 88.0,
        "inference_substrate": "live_llm_inference_local_gguf_sota_corrupt_feedback_held_restart",
        "csl_safety_replication_ready_score": 1.0,
        "per_unit_rows": {"row_count": len(rows), "rows": rows},
        "future_exact_yield_delta": {"clean_minus_frozen": -999.0},
    }


def test_req_learn_6457_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6457: OpenSpec owns the Exp6457 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6457") : text.index("REQ-LEARN-6444")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-LEARN-6457-SPEC",
        "SCENARIO-LEARN-6457-INVENTORY",
        "SCENARIO-LEARN-6457-REDUCERS",
        "SCENARIO-LEARN-6457-AUTHORITY",
        "SCENARIO-LEARN-6457-SAFETY",
        "SCENARIO-LEARN-6457-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "without importing upstream aggregate, readiness, gate, update, or verdict functions",
    ):
        assert marker in normalized if marker.startswith("without ") else marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        key = f"csl_audit_ready_score:{condition}"
        assert key in mod.FIELD_PRINCIPLES
        assert key in section


def test_scenario_learn_6457_reducers_ignore_upstream_aggregates() -> None:
    """SCENARIO-LEARN-6457-REDUCERS: metrics come from rows, not summaries."""

    payload = _prospective_payload()
    recomputed = mod.reduce_prospective(payload)

    assert recomputed["row_count"] == 6
    assert recomputed["future_unit_count"] == 1
    assert recomputed["future_exact_yield_delta"]["verifier_bounded_minus_frozen"] == 1.0
    assert recomputed["future_exact_yield_delta"]["verifier_bounded_minus_teacher"] == 1.0
    assert recomputed["false_accept_count"] == 0
    assert recomputed["protected_regression_count"] == 0
    assert recomputed["source"] == "per_unit_rows.rows"
    assert payload["future_exact_yield_delta"]["verifier_bounded_minus_frozen"] == -999.0


def test_scenario_learn_6457_exact_sign_authority_is_checked() -> None:
    """SCENARIO-LEARN-6457-AUTHORITY: exact checker signs own update direction."""

    payload = _prospective_payload()
    rows = payload["per_unit_rows"]["rows"]
    checks = mod.update_direction_and_chronology_checks(rows)

    assert checks["exact_sign_authority_passed"] is True
    assert checks["teacher_sign_authority_count"] == 0
    assert checks["teacher_negative_magnitude_count"] == 0
    assert checks["same_unit_update_use_count"] == 0
    assert checks["future_only_updates"] is True

    tampered = deepcopy(rows)
    tampered[4]["applied_update_sign"] = tampered[4]["teacher_signal"]["signed_direction"]
    failed = mod.update_direction_and_chronology_checks(tampered)
    assert failed["exact_sign_authority_passed"] is False
    assert failed["exact_sign_mismatch_count"] == 1


def test_scenario_learn_6457_safety_recomputes_quarantine_and_restart() -> None:
    """SCENARIO-LEARN-6457-SAFETY: corrupt feedback is contained from rows."""

    payload = _held_payload()
    held = mod.reduce_held(payload)
    safety = mod.corruption_quarantine_rollback_and_resurrection_checks(
        payload["per_unit_rows"]["rows"]
    )
    restart = mod.transaction_head_and_restart_checks(payload["per_unit_rows"]["rows"])

    assert held["future_exact_yield_delta"]["clean_minus_frozen"] == 1.0
    assert held["future_exact_yield_delta"]["governed_minus_frozen"] == 1.0
    assert safety["scheduled_corrupt_event_count"] == 1
    assert safety["detected_corrupt_event_count"] == 1
    assert safety["quarantine_precision"] == 1.0
    assert safety["quarantine_recall"] == 1.0
    assert safety["rollback_success_count"] == 1
    assert safety["corrupt_update_resurrection_count"] == 0
    assert restart["all_restart_recovery_valid"] is True
    assert restart["inherited_state_visible_count"] == 0


def test_scenario_learn_6457_blocked_artifact_populates_gate_summary(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6457-READY: blocked audits still explain failed gates."""

    artifact = mod.build_artifact(
        root=tmp_path,
        date=mod.RUN_DATE,
        write=False,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )

    assert artifact["status"] == "complete_blocked"
    assert artifact["csl_audit_ready_score"] == 0.0
    assert artifact["prospective_csl_eligibility"] is False
    assert artifact["blocked_reason"]
    assert artifact["csl_ineligibility_reasons"]
    assert artifact["gate_check_summary"]["failed_check_count"] > 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_learn_6457_source_declares_no_upstream_imports() -> None:
    """SCENARIO-LEARN-6457-INVENTORY: reducers stay independent of upstream modules."""

    receipt = mod.independent_reducer_source_and_test_hashes(REPO)

    assert receipt["module_imports_upstream_experiments"] is False
    assert receipt["forbidden_upstream_imports"] == []
    assert receipt["source_hashes"][mod.MODULE_RELATIVE_PATH.as_posix()].startswith("sha256:")
