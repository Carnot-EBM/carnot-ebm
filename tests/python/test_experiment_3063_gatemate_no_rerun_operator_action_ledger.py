"""Tests for Exp 3063 GateMate no-rerun operator-action ledger.

Spec refs: REQ-HW-089, SCENARIO-HW-089.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_no_rerun_operator_action_ledger_3063 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = {
    "gatemate_no_rerun_ledger_ready",
    "gatemate_rerun_allowed",
    "missing_operator_actions",
    "required_evidence_before_rerun",
    "downstream_tasks_blocked",
    "hardware_execution_claim_made",
    "speedup_claim_made",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _exp3048(*, ready: bool = False) -> dict[str, Any]:
    binding = {"signal_name": "done", "pin": "IO_EB_B7", "line": "Pin_out done Loc = IO_EB_B7"}
    return {
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_signal": "done",
        "ccf_binding": binding if ready else {},
        "host_reader_command": (
            ".venv/bin/python scripts/gatemate_done_gpio_reader.py --expect done=1"
            if ready
            else ""
        ),
        "expected_transcript": ["done=1 PASS"] if ready else [],
        "missing_operator_actions": []
        if ready
        else [
            "Provide an authoritative GateMate A1-EVB-2M output pinout and commit a CCF Pin_out binding for done.",
            "Commit a concrete host reader command for done: GPIO/LED read, UART serial decode, or JTAG-readable status command.",
            "Keep downstream flash smoke gated until the reader command has an expected pass/fail transcript.",
        ],
        "safety_limits": {
            "downstream_flash_gate_open": ready,
            "exp3049_gate": "require_gatemate_output_contract_ready_true",
            "exp3050_gate": "require_host_visible_io_plan_ready_true",
            "max_flash_attempts_without_operator_review": 1 if ready else 0,
        },
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": "complete: ready" if ready else "complete: blocked_gatemate_output_contract_authority_missing",
    }


def _exp3049() -> dict[str, Any]:
    return {
        "experiment": 3049,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp3048-gatemate-output-contract-operator-package.gatemate_output_contract_ready "
            "(actual=False == expected=True)"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp3048-gatemate-output-contract-operator-package",
                "artifact_field": "gatemate_output_contract_ready",
                "expected": True,
                "actual": False,
                "passed": False,
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
    }


def _exp3051() -> dict[str, Any]:
    return {
        "experiment": 3051,
        "schema": "blocked_gate_check_v1",
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: "
            "exp3050-gatemate-host-visible-flash-smoke-v5.gatemate_host_visible_smoke_passed "
            "(upstream artifact not found for task id 'exp3050-gatemate-host-visible-flash-smoke-v5')"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp3050-gatemate-host-visible-flash-smoke-v5",
                "artifact_field": "gatemate_host_visible_smoke_passed",
                "expected": True,
                "actual": None,
                "passed": False,
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
    }


def _write_sources(root: Path, *, ready: bool = False) -> None:
    _write_json(root, mod.EXP3048_REL_PATH, _exp3048(ready=ready))
    _write_json(root, mod.EXP3049_REL_PATH, _exp3049())
    _write_json(root, mod.EXP3051_BOUNDED_REL_PATH, _exp3051())
    _write_text(
        root,
        Path("ops/conductor-log.md"),
        (
            "| 2026-05-25 09:39 UTC | GateMate output shim RTL/CCF simulation v2 | GATE_BLOCK | "
            "1 of 1 gate(s) failed |\n"
            "| 2026-05-25 09:45 UTC | GateMate host-visible flash smoke v5 | GATE_BLOCK | "
            "Pre-emptive skip: upstream retired |\n"
            "| 2026-05-25 09:49 UTC | SSQA readback eligibility bounded gate v3 | GATE_BLOCK | "
            "host-visible smoke missing |\n"
        ),
    )
    _write_text(
        root,
        Path("research-hardware-wishlist.md"),
        (
            "| GateMate A1-EVB-2M | DirtyJTAG detected. | No GateMate latency or "
            "speedup claim until flash and sample-level timing evidence exists. |\n"
        ),
    )
    _write_text(root, Path("ops/changelog.md"), "GateMate stayed blocked on output-contract authority.\n")


def test_req_hw_089_spec_entry_present() -> None:
    """REQ-HW-089: OpenSpec declares the no-rerun ledger contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-089" in spec
    assert "SCENARIO-HW-089" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_089_builds_blocked_no_rerun_ledger(tmp_path: Path) -> None:
    """SCENARIO-HW-089: blocked GateMate contract keeps every branch stopped."""
    _write_sources(tmp_path, ready=False)

    artifact = mod.build_artifact(tmp_path)
    evidence = {row["evidence_id"]: row for row in artifact["required_evidence_before_rerun"]}
    downstream = {row["task_id"]: row for row in artifact["downstream_tasks_blocked"]}
    actions = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["gatemate_no_rerun_ledger_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["inference_substrate"] == {
        "kind": "hardware_contract_no_rerun_ledger",
        "source": "checked_in_local_artifacts",
        "model_inference": False,
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "flash_attempted": False,
        "rtl_run": False,
        "local_repo_only": True,
        "timing_or_speedup_claim": False,
    }

    assert artifact["operator_contract"]["selected_output_signal"] == "done"
    assert actions["authoritative_pinout_ccf_binding"]["present"] is False
    assert actions["host_reader_command"]["present"] is False
    assert actions["expected_transcript"]["present"] is False
    assert evidence["selected_output_signal"]["present"] is True
    assert evidence["authoritative_pinout_ccf_binding"]["present"] is False
    assert evidence["host_reader_command"]["present"] is False
    assert evidence["expected_transcript"]["present"] is False
    assert evidence["safety_limits"]["present"] is True

    assert set(downstream) == {
        "exp3049-gatemate-output-shim-rtl-ccf-sim-v2",
        "exp3050-gatemate-host-visible-flash-smoke-v5",
        "exp3051-ssqa-readback-eligibility-bounded-gate-v3",
    }
    assert all(row["allowed_to_rerun"] is False for row in downstream.values())
    assert downstream["exp3049-gatemate-output-shim-rtl-ccf-sim-v2"]["matrix_status"] == "blocked"
    assert downstream["exp3050-gatemate-host-visible-flash-smoke-v5"]["matrix_status"] == "gate_skipped"
    assert downstream["exp3051-ssqa-readback-eligibility-bounded-gate-v3"]["matrix_status"] == "gate_skipped"
    assert "exp3048.gatemate_output_contract_ready=false" in downstream[
        "exp3049-gatemate-output-shim-rtl-ccf-sim-v2"
    ]["upstream_blocker"]

    source_paths = {row["path"] for row in artifact["source_artifacts"]}
    assert mod.EXP3048_REL_PATH.as_posix() in source_paths
    assert mod.EXP3049_REL_PATH.as_posix() in source_paths
    assert mod.EXP3051_BOUNDED_REL_PATH.as_posix() in source_paths
    assert "ops/conductor-log.md" in source_paths
    assert "research-hardware-wishlist.md" in source_paths


def test_req_hw_089_all_evidence_present_allows_rerun_without_claiming_execution(
    tmp_path: Path,
) -> None:
    """REQ-HW-089: concrete contract evidence can open rerun permission only."""
    _write_sources(tmp_path, ready=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gatemate_no_rerun_ledger_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["missing_operator_actions"] == []
    assert all(row["present"] is True for row in artifact["required_evidence_before_rerun"])
    assert all(row["allowed_to_rerun"] is True for row in artifact["downstream_tasks_blocked"])
    assert artifact["operator_contract"]["ccf_binding"]["pin"] == "IO_EB_B7"
    assert artifact["operator_contract"]["host_reader_command"].endswith("--expect done=1")
    assert artifact["operator_contract"]["expected_transcript"] == ["done=1 PASS"]
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert "cite" in artifact["rerun_permission_basis"][0]


def test_req_hw_089_missing_required_source_fails_closed(tmp_path: Path) -> None:
    """REQ-HW-089: missing Exp 3048 prevents a ready ledger from fabricating actions."""
    _write_json(tmp_path, mod.EXP3049_REL_PATH, _exp3049())

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gatemate_no_rerun_ledger_ready"] is False
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["missing_source_artifacts"] == [mod.EXP3048_REL_PATH.as_posix()]
    assert all(row["allowed_to_rerun"] is False for row in artifact["downstream_tasks_blocked"])


def test_req_hw_089_write_artifact_emits_stable_deliverable(tmp_path: Path) -> None:
    """REQ-HW-089: write_artifact writes the stable JSON deliverable."""
    _write_sources(tmp_path, ready=False)

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["gatemate_no_rerun_ledger_ready"] is True
    assert payload["gatemate_rerun_allowed"] is False


def test_req_hw_089_helpers_handle_malformed_json_and_requested_3051_name(
    tmp_path: Path,
) -> None:
    """REQ-HW-089: helper edges use the checked-in Exp 3051 file and fail closed."""
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    _write_json(tmp_path, mod.EXP3051_REQUESTED_REL_PATH, _exp3051())

    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._existing_path(tmp_path, (mod.EXP3051_REQUESTED_REL_PATH, mod.EXP3051_BOUNDED_REL_PATH)) == (
        mod.EXP3051_REQUESTED_REL_PATH
    )
    assert mod._source_payload(tmp_path, mod.EXP3051_SOURCE)["path"] == mod.EXP3051_REQUESTED_REL_PATH.as_posix()
    assert mod._concrete("blocked_no_host_reader_command") is False
    assert mod._concrete(["done=1 PASS"]) is True

    evidence = [
        {"evidence_id": "selected_output_signal", "present": False},
        {"evidence_id": "authoritative_pinout_ccf_binding", "present": True},
        {"evidence_id": "host_reader_command", "present": True},
        {"evidence_id": "expected_transcript", "present": True},
        {"evidence_id": "safety_limits", "present": True},
    ]
    assert mod._missing_operator_actions({}, evidence)[0]["missing_item"] == "selected_output_signal"
    assert mod._blocker({"gatemate_output_contract_ready": False}, {}) == (
        "exp3048.gatemate_output_contract_ready=false"
    )
    assert (
        mod._blocker({"gatemate_output_contract_ready": True}, {})
        == "host-visible GateMate output evidence incomplete"
    )
