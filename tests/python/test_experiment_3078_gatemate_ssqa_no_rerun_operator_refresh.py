"""Tests for Exp 3078 GateMate/SSQA no-rerun operator refresh.

Spec refs: REQ-HW-091, SCENARIO-HW-091.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_ssqa_no_rerun_operator_refresh_3078 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = {
    "gatemate_ssqa_refresh_ready",
    "gatemate_rerun_allowed",
    "ssqa_readback_allowed",
    "missing_operator_actions",
    "operator_ready_artifacts",
    "next_allowed_hardware_task",
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
    binding = {
        "signal_name": "done",
        "pin": "IO_EB_B7",
        "line": "Pin_out done Loc = IO_EB_B7",
        "source_path": "hardware/gatemate/operator_done_pinout.ccf",
    }
    return {
        "artifact": "experiment_3048_gatemate_output_contract_operator_package_v1",
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_signal": "done",
        "ccf_binding": binding if ready else {},
        "host_reader_command": (
            ".venv/bin/python scripts/gatemate_done_gpio_reader.py --expect done=1" if ready else ""
        ),
        "expected_transcript": ["done=1 PASS"] if ready else [],
        "safety_limits": {
            "downstream_flash_gate_open": ready,
            "exp3049_gate": "require_gatemate_output_contract_ready_true",
            "exp3050_gate": "require_host_visible_io_plan_ready_true",
            "max_flash_attempts_without_operator_review": 1 if ready else 0,
        },
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": (
            "complete: ready"
            if ready
            else "complete: blocked_gatemate_output_contract_authority_missing"
        ),
    }


def _exp3063(*, ready: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1",
        "gatemate_no_rerun_ledger_ready": True,
        "gatemate_rerun_allowed": ready,
        "missing_operator_actions": []
        if ready
        else [
            {
                "missing_item": "authoritative_pinout_ccf_binding",
                "operator_action": (
                    "Provide an authoritative GateMate A1-EVB-2M output pinout and commit "
                    "a CCF Pin_out binding for done."
                ),
                "present": False,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "host_reader_command",
                "operator_action": "Commit a concrete host reader command for done.",
                "present": False,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "expected_transcript",
                "operator_action": (
                    "Record the expected pass/fail transcript for the done host reader command."
                ),
                "present": False,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
        ],
        "required_evidence_before_rerun": [
            {
                "evidence_id": "selected_output_signal",
                "present": True,
                "rerun_satisfied": True,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "evidence_id": "authoritative_pinout_ccf_binding",
                "present": ready,
                "rerun_satisfied": ready,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "evidence_id": "host_reader_command",
                "present": ready,
                "rerun_satisfied": ready,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "evidence_id": "expected_transcript",
                "present": ready,
                "rerun_satisfied": ready,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "evidence_id": "safety_limits",
                "present": True,
                "rerun_satisfied": ready,
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
        ],
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": f"complete: gatemate_rerun_allowed={str(ready).lower()}",
    }


def _exp3064(*, ready: bool = False) -> dict[str, Any]:
    missing = [] if ready else [field_id for field_id, _scope in mod.HOST_VISIBLE_REQUIRED_FIELDS]
    return {
        "artifact": "experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1",
        "ssqa_boundary_ledger_ready": True,
        "ssqa_readback_allowed": ready,
        "ssqa_status": (
            "clean_host_visible_smoke_transcript_present"
            if ready
            else "gated_skipped_host_visible_smoke_missing"
        ),
        "host_visible_smoke_evidence": {
            "path": mod.EXP3050_REL_PATH.as_posix(),
            "present": ready,
            "readable": ready,
            "readback_unlocks_ssqa": ready,
            "missing_required_fields": missing,
            "gatemate_host_visible_smoke_passed": ready,
            "transcript_matched": ready,
        },
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
        "honest_verdict": f"complete: ssqa_readback_allowed={str(ready).lower()}",
    }


def _passing_smoke() -> dict[str, Any]:
    return {
        "gatemate_host_visible_smoke_passed": True,
        "host_visible_transcript_path": "logs/experiment_3050/host_visible_smoke.txt",
        "transcript_sha256": "0" * 64,
        "host_reader_command": ".venv/bin/python scripts/gatemate_done_reader.py --expect done=1",
        "expected_transcript": ["done=1 PASS"],
        "observed_transcript": ["done=1 PASS"],
        "transcript_matched": True,
        "selected_output_signal": "done",
        "ccf_binding": {"signal_name": "done", "pin": "IO_EB_B7"},
        "flash_succeeded": True,
        "flash_command": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb bitstream.bit",
        "readback_supported": True,
        "readback_attempted": True,
        "readback_hash": "1" * 64,
        "sample_count": 16,
        "wall_clock_duration_s": 0.25,
        "per_sample_latency_s": 0.015625,
        "sampler_configuration": {"n_spins": 16, "top": "ising_n16_gatemate"},
        "hardware_execution_claim_made": False,
        "speedup_claim_made": False,
    }


def _write_sources(
    root: Path,
    *,
    gatemate_ready: bool = False,
    ssqa_ready: bool = False,
) -> None:
    _write_json(root, mod.EXP3048_REL_PATH, _exp3048(ready=gatemate_ready))
    _write_json(root, mod.EXP3063_REL_PATH, _exp3063(ready=gatemate_ready))
    _write_json(root, mod.EXP3064_REL_PATH, _exp3064(ready=ssqa_ready))
    if ssqa_ready:
        _write_json(root, mod.EXP3050_REL_PATH, _passing_smoke())
    _write_text(
        root,
        mod.HARDWARE_WISHLIST_REL_PATH,
        "GateMate requires host-visible sample-level timing before speedup claims.\n",
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "GateMate host-visible flash smoke v5 | GATE_BLOCK | upstream retired\n",
    )
    _write_text(root, mod.STATUS_REL_PATH, "GateMate remains no-rerun blocked.\n")


def test_req_hw_091_spec_entry_present() -> None:
    """REQ-HW-091: OpenSpec declares the matrix v21 hardware refresh contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-091" in spec
    assert "SCENARIO-HW-091" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_091_blocks_when_operator_evidence_is_missing(tmp_path: Path) -> None:
    """SCENARIO-HW-091: absent operator evidence keeps GateMate and SSQA stopped."""
    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    actions = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}
    source_paths = {row["path"] for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["gatemate_ssqa_refresh_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["operator_ready_artifacts"] == []
    assert artifact["next_allowed_hardware_task"].startswith("blocked:")
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == {
        "kind": "gatemate_ssqa_no_rerun_operator_refresh",
        "source": "checked_in_local_artifacts",
        "model_inference": False,
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "flash_attempted": False,
        "rtl_or_pnr_run": False,
        "hardware_readback_attempted": False,
        "local_repo_only": True,
        "timing_or_speedup_claim": False,
    }

    assert actions["authoritative_pinout_ccf_binding"]["present"] is False
    assert actions["host_reader_command"]["present"] is False
    assert actions["expected_transcript"]["present"] is False
    assert actions["host_visible_smoke_evidence"]["present"] is False
    assert mod.EXP3048_REL_PATH.as_posix() in source_paths
    assert mod.EXP3063_REL_PATH.as_posix() in source_paths
    assert mod.EXP3064_REL_PATH.as_posix() in source_paths
    assert mod.HARDWARE_WISHLIST_REL_PATH.as_posix() in source_paths


def test_req_hw_091_ready_evidence_records_next_task_without_execution(tmp_path: Path) -> None:
    """REQ-HW-091: ready evidence cites files but Exp 3078 still makes no claims."""
    _write_sources(tmp_path, gatemate_ready=True, ssqa_ready=True)

    artifact = mod.build_artifact(tmp_path)
    ready = {row["evidence_id"]: row for row in artifact["operator_ready_artifacts"]}

    assert artifact["gatemate_ssqa_refresh_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is True
    assert artifact["missing_operator_actions"] == []
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert artifact["next_allowed_hardware_task"] == (
        "operator_allowed: run the gated SSQA readback task using "
        "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json; "
        "do not claim speedup without a new timing transcript"
    )
    assert ready["authoritative_pinout_ccf_binding"]["path"] == mod.EXP3048_REL_PATH.as_posix()
    assert ready["host_reader_command"]["path"] == mod.EXP3048_REL_PATH.as_posix()
    assert ready["expected_transcript"]["path"] == mod.EXP3048_REL_PATH.as_posix()
    assert ready["host_visible_smoke_evidence"]["path"] == mod.EXP3050_REL_PATH.as_posix()


def test_req_hw_091_gatemate_ready_without_ssqa_names_flash_smoke_task(
    tmp_path: Path,
) -> None:
    """REQ-HW-091: GateMate readiness alone only allows the next gated smoke task."""
    _write_sources(tmp_path, gatemate_ready=True, ssqa_ready=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["next_allowed_hardware_task"] == (
        "operator_allowed: run the gated GateMate output shim RTL/CCF simulation and "
        "host-visible flash smoke using the committed output contract; do not claim "
        "speedup"
    )


def test_req_hw_091_missing_required_source_blocks_refresh(tmp_path: Path) -> None:
    """REQ-HW-091: missing required prior ledgers prevent a matrix-ready refresh."""
    _write_json(tmp_path, mod.EXP3048_REL_PATH, _exp3048())
    _write_json(tmp_path, mod.EXP3063_REL_PATH, _exp3063())
    _write_text(tmp_path, mod.HARDWARE_WISHLIST_REL_PATH, "hardware boundary\n")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gatemate_ssqa_refresh_ready"] is False
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["missing_source_artifacts"] == [mod.EXP3064_REL_PATH.as_posix()]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_091_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-091: write_artifact writes stable JSON and readers fail closed."""
    _write_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["gatemate_ssqa_refresh_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._as_text_list("done=1") == ["done=1"]
    assert mod._field_present({"items": ["done=1"]}, "items") is True
    assert mod._ccf_binding_from_text("Pin_out done Loc = IO_EB_B7\n", "done") == {
        "signal_name": "done",
        "pin": "IO_EB_B7",
        "line": "Pin_out done Loc = IO_EB_B7",
        "line_number": 1,
        "source_path": mod.GATEMATE_CCF_REL_PATH.as_posix(),
    }
