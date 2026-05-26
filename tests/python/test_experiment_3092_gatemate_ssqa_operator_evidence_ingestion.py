"""Tests for Exp 3092 GateMate/SSQA operator evidence ingestion.

Spec refs: REQ-HW-092, SCENARIO-HW-092.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_ssqa_operator_evidence_ingestion_3092 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = {
    "operator_evidence_ingestion_ready",
    "gatemate_rerun_allowed",
    "ssqa_readback_allowed",
    "operator_ready_artifacts",
    "missing_operator_actions",
    "speedup_claim_made",
    "hardware_commands_run",
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


def _exp3078_missing() -> dict[str, Any]:
    return {
        "artifact": "experiment_3078_gatemate_ssqa_no_rerun_operator_refresh_v1",
        "gatemate_ssqa_refresh_ready": True,
        "gatemate_rerun_allowed": False,
        "ssqa_readback_allowed": False,
        "operator_ready_artifacts": [],
        "missing_operator_actions": [
            {
                "missing_item": "authoritative_pinout_ccf_binding",
                "present": False,
                "operator_action": "Provide an authoritative GateMate CCF Pin_out binding.",
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "host_reader_command",
                "present": False,
                "operator_action": "Commit a concrete host reader command for done.",
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "expected_transcript",
                "present": False,
                "operator_action": "Record the expected pass/fail transcript.",
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "safety_limits",
                "present": False,
                "operator_action": "Open the downstream flash safety gate.",
                "source_artifact": mod.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "host_visible_smoke_evidence",
                "present": False,
                "operator_action": "Commit a passing host-visible smoke transcript.",
                "source_artifact": mod.EXP3050_REL_PATH.as_posix(),
            },
        ],
        "honest_verdict": "complete: gatemate_rerun_allowed=false",
    }


def _exp3048(*, ready: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3048_gatemate_output_contract_operator_package_v1",
        "gatemate_output_contract_ready": ready,
        "host_visible_io_plan_ready": ready,
        "selected_output_signal": "done",
        "ccf_binding": {"signal_name": "done", "pin": "IO_EB_B7"} if ready else {},
        "host_reader_command": (
            ".venv/bin/python scripts/gatemate_done_reader.py --expect done=1" if ready else ""
        ),
        "expected_transcript": ["done=1 PASS"] if ready else [],
        "safety_limits": {
            "downstream_flash_gate_open": ready,
            "max_flash_attempts_without_operator_review": 1 if ready else 0,
        },
        "speedup_claim_made": False,
    }


def _exp3063() -> dict[str, Any]:
    return {
        "artifact": "experiment_3063_gatemate_no_rerun_operator_action_ledger_v1",
        "gatemate_no_rerun_ledger_ready": True,
        "gatemate_rerun_allowed": False,
        "speedup_claim_made": False,
    }


def _exp3064() -> dict[str, Any]:
    return {
        "artifact": "experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1",
        "ssqa_boundary_ledger_ready": True,
        "ssqa_readback_allowed": False,
        "host_visible_smoke_evidence": {
            "path": mod.EXP3050_REL_PATH.as_posix(),
            "present": False,
            "missing_required_fields": [field for field, _scope in mod.HOST_VISIBLE_REQUIRED_FIELDS],
        },
        "speedup_claim_made": False,
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
    }


def _write_required_sources(
    root: Path,
    *,
    gatemate_ready: bool = False,
    smoke_ready: bool = False,
) -> None:
    _write_json(root, mod.EXP3078_REL_PATH, _exp3078_missing())
    _write_json(root, mod.EXP3048_REL_PATH, _exp3048(ready=gatemate_ready))
    _write_json(root, mod.EXP3063_REL_PATH, _exp3063())
    _write_json(root, mod.EXP3064_REL_PATH, _exp3064())
    if smoke_ready:
        _write_json(root, mod.EXP3050_REL_PATH, _passing_smoke())
    _write_text(root, mod.HARDWARE_WISHLIST_REL_PATH, "No speedup without transcript.\n")
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, "GateMate smoke | GATE_BLOCK\n")
    _write_text(root, mod.STATUS_REL_PATH, "GateMate/SSQA blocked on operator evidence.\n")
    _write_text(root, mod.CHANGELOG_REL_PATH, "GateMate/SSQA blocked on operator evidence.\n")
    _write_text(
        root,
        mod.GATEMATE_CCF_REL_PATH,
        "Pin_out done Loc = IO_EB_B7\n" if gatemate_ready else "# build-only CCF\n",
    )


def test_req_hw_092_spec_entry_present() -> None:
    """REQ-HW-092: OpenSpec declares the operator evidence ingestion contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-092" in spec
    assert "SCENARIO-HW-092" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_092_missing_evidence_keeps_no_rerun_boundary(tmp_path: Path) -> None:
    """SCENARIO-HW-092: unresolved Exp 3078 actions keep GateMate and SSQA blocked."""
    _write_required_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}
    checked_paths = {row["path"]: row for row in artifact["checked_paths"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["operator_evidence_ingestion_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["operator_ready_artifacts"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["hardware_commands_run"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["allowed_next_experiment_scope"].startswith("blocked:")
    assert artifact["inference_substrate"]["model_inference"] is False
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["inference_substrate"]["no_live_model_inference"] is True

    assert set(missing) == {
        "authoritative_pinout_ccf_binding",
        "host_reader_command",
        "expected_transcript",
        "safety_limits",
        "host_visible_smoke_evidence",
    }
    assert missing["host_visible_smoke_evidence"]["missing_required_fields"]
    assert checked_paths[mod.EXP3050_REL_PATH.as_posix()]["present"] is False
    assert checked_paths[mod.GATEMATE_CCF_REL_PATH.as_posix()]["present"] is True


def test_req_hw_092_ready_evidence_allows_only_next_scope(tmp_path: Path) -> None:
    """REQ-HW-092: complete evidence opens the next scope without speedup claims."""
    _write_required_sources(tmp_path, gatemate_ready=True, smoke_ready=True)

    artifact = mod.build_artifact(tmp_path)
    ready = {row["evidence_id"]: row for row in artifact["operator_ready_artifacts"]}

    assert artifact["operator_evidence_ingestion_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is True
    assert artifact["missing_operator_actions"] == []
    assert artifact["hardware_commands_run"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed: run the gated SSQA readback experiment using committed "
        "host-visible smoke evidence; do not make speedup claims without a new "
        "operator timing transcript"
    )
    assert ready["authoritative_pinout_ccf_binding"]["path"] == mod.EXP3048_REL_PATH.as_posix()
    assert ready["host_visible_smoke_evidence"]["path"] == mod.EXP3050_REL_PATH.as_posix()


def test_req_hw_092_partial_gate_only_names_gatemate_smoke_scope(tmp_path: Path) -> None:
    """REQ-HW-092: GateMate evidence alone does not unlock SSQA readback."""
    _write_required_sources(tmp_path, gatemate_ready=True, smoke_ready=False)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}

    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert list(missing) == ["host_visible_smoke_evidence"]
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed: run the gated GateMate host-visible flash smoke only; "
        "do not claim timing or speedup"
    )


def test_req_hw_092_missing_prior_refresh_blocks_ingestion(tmp_path: Path) -> None:
    """REQ-HW-092: missing Exp 3078 source blocks evidence ingestion readiness."""
    _write_json(tmp_path, mod.EXP3048_REL_PATH, _exp3048())
    _write_json(tmp_path, mod.EXP3063_REL_PATH, _exp3063())
    _write_json(tmp_path, mod.EXP3064_REL_PATH, _exp3064())
    _write_text(tmp_path, mod.HARDWARE_WISHLIST_REL_PATH, "hardware boundary\n")

    artifact = mod.build_artifact(tmp_path)

    assert artifact["operator_evidence_ingestion_ready"] is False
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert mod.EXP3078_REL_PATH.as_posix() in artifact["missing_source_artifacts"]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_092_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-092: artifact writing and fail-closed helper behavior are stable."""
    _write_required_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["operator_evidence_ingestion_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._as_text_list("done=1") == ["done=1"]
    assert mod._field_present({"items": ["done=1"]}, "items") is True
    assert mod._field_present({"flag": False}, "flag") is False
    assert mod._ccf_binding_from_text("Pin_out done Loc = IO_EB_B7\n", "done") == {
        "signal_name": "done",
        "pin": "IO_EB_B7",
        "line": "Pin_out done Loc = IO_EB_B7",
        "line_number": 1,
        "source_path": mod.GATEMATE_CCF_REL_PATH.as_posix(),
    }
