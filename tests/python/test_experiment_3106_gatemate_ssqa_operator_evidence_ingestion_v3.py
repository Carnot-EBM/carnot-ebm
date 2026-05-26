"""Tests for Exp 3106 GateMate/SSQA operator evidence ingestion v3.

Spec refs: REQ-HW-093, SCENARIO-HW-093.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import gatemate_ssqa_operator_evidence_ingestion_3092 as v2
from carnot.reporting import gatemate_ssqa_operator_evidence_ingestion_3106 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = {
    "operator_evidence_ingestion_v3_ready",
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
        "missing_operator_actions": [
            {
                "missing_item": "authoritative_pinout_ccf_binding",
                "present": False,
                "operator_action": "Provide an authoritative GateMate CCF Pin_out binding.",
                "source_artifact": v2.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "host_reader_command",
                "present": False,
                "operator_action": "Commit a concrete host reader command for done.",
                "source_artifact": v2.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "expected_transcript",
                "present": False,
                "operator_action": "Record the expected pass/fail transcript.",
                "source_artifact": v2.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "safety_limits",
                "present": False,
                "operator_action": "Open the downstream flash safety gate.",
                "source_artifact": v2.EXP3048_REL_PATH.as_posix(),
            },
            {
                "missing_item": "host_visible_smoke_evidence",
                "present": False,
                "operator_action": "Commit a passing host-visible smoke transcript.",
                "source_artifact": v2.EXP3050_REL_PATH.as_posix(),
            },
        ],
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
            "path": v2.EXP3050_REL_PATH.as_posix(),
            "present": False,
            "missing_required_fields": [field for field, _scope in v2.HOST_VISIBLE_REQUIRED_FIELDS],
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


def _capstone_v288() -> dict[str, Any]:
    hardware_rows = [
        "v20:v19:v18:exp3034",
        "v20:v19:gatemate:output_contract",
        "v20:gatemate:no_rerun_ledger",
        "v20:v19:v18:exp3035",
        "v20:v19:v18:exp3036",
        "v20:v19:v18:exp3037",
        "v20:v19:ssqa:readback_gate",
        "v20:ssqa:host_visible_readback_boundary",
        "v20:v19:gatemate:host_visible_smoke",
        "dot287:exp3078_gatemate_operator_refresh",
        "dot287:exp3078_ssqa_readback_refresh",
        "dot288:exp3092_gatemate_operator_evidence",
        "dot288:exp3092_ssqa_readback_evidence",
    ]
    return {
        "artifact": "experiment_3094_capstone_v288",
        "capstone_ready": True,
        "gatemate_status": "blocked_no_rerun_operator_actions_required",
        "ssqa_status": "gated_skipped_host_visible_smoke_missing",
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_new_synthesis_run": True,
        "prd_gap_summary": {
            "hardware_evidence": {
                "claim_boundary": "Operator evidence ingestion is not host-visible hardware speedup.",
                "publication_blocker_count": 11,
                "row_ids": hardware_rows,
                "publication_blocker_row_ids": hardware_rows[:5],
            }
        },
    }


def _write_v2_sources(
    root: Path,
    *,
    gatemate_ready: bool = False,
    smoke_ready: bool = False,
) -> None:
    _write_json(root, v2.EXP3078_REL_PATH, _exp3078_missing())
    _write_json(root, v2.EXP3048_REL_PATH, _exp3048(ready=gatemate_ready))
    _write_json(root, v2.EXP3063_REL_PATH, _exp3063())
    _write_json(root, v2.EXP3064_REL_PATH, _exp3064())
    if smoke_ready:
        _write_json(root, v2.EXP3050_REL_PATH, _passing_smoke())
    _write_text(root, v2.HARDWARE_WISHLIST_REL_PATH, "No speedup without transcript.\n")
    _write_text(root, v2.CONDUCTOR_LOG_REL_PATH, "GateMate smoke | GATE_BLOCK\n")
    _write_text(root, v2.STATUS_REL_PATH, "GateMate/SSQA blocked on operator evidence.\n")
    _write_text(root, v2.CHANGELOG_REL_PATH, "GateMate/SSQA blocked on operator evidence.\n")
    _write_text(
        root,
        v2.GATEMATE_CCF_REL_PATH,
        "Pin_out done Loc = IO_EB_B7\n" if gatemate_ready else "# build-only CCF\n",
    )
    _write_json(root, v2.GATEMATE_TEST_VECTOR_REL_PATH, {"expected_done": True})
    _write_text(root, v2.GATEMATE_RTL_REL_PATH, "module ising_n16_gatemate; endmodule\n")
    _write_text(root, v2.GATEMATE_JTAG_DOC_REL_PATH, "DirtyJTAG wiring context.\n")


def _write_required_sources(
    root: Path,
    *,
    gatemate_ready: bool = False,
    smoke_ready: bool = False,
) -> None:
    _write_v2_sources(root, gatemate_ready=gatemate_ready, smoke_ready=smoke_ready)
    _write_json(root, mod.EXP3094_REL_PATH, _capstone_v288())
    _write_json(root, mod.EXP3092_REL_PATH, v2.build_artifact(root))


def test_req_hw_093_spec_entry_present() -> None:
    """REQ-HW-093: OpenSpec declares the v3 operator evidence contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-093" in spec
    assert "SCENARIO-HW-093" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_hw_093_missing_evidence_keeps_hardware_blocked(tmp_path: Path) -> None:
    """SCENARIO-HW-093: missing operator evidence keeps rerun and readback blocked."""
    _write_required_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    checked = {row["path"]: row for row in artifact["checked_paths"]}
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}
    source_paths = {row["path"] for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["operator_evidence_ingestion_v3_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert artifact["operator_ready_artifacts"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["hardware_commands_run"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["allowed_next_experiment_scope"].startswith("blocked:")
    assert artifact["inference_substrate"]["no_live_model_inference"] is True
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["inference_substrate"]["hardware_readback_attempted"] is False

    assert set(missing) == {
        "authoritative_pinout_ccf_binding",
        "host_reader_command",
        "expected_transcript",
        "safety_limits",
        "host_visible_smoke_evidence",
    }
    assert missing["host_visible_smoke_evidence"]["missing_required_fields"]
    assert checked[v2.EXP3050_REL_PATH.as_posix()]["present"] is False
    assert checked[mod.EXP3036_REL_PATH.as_posix()]["present"] is False
    assert mod.EXP3094_REL_PATH.as_posix() in source_paths
    assert mod.EXP3092_REL_PATH.as_posix() in source_paths


def test_req_hw_093_complete_evidence_opens_scope_without_speedup_claim(tmp_path: Path) -> None:
    """REQ-HW-093: complete evidence records only the next allowed operator scope."""
    _write_required_sources(tmp_path, gatemate_ready=True, smoke_ready=True)

    artifact = mod.build_artifact(tmp_path)
    ready = {row["evidence_id"]: row for row in artifact["operator_ready_artifacts"]}

    assert artifact["operator_evidence_ingestion_v3_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is True
    assert artifact["missing_operator_actions"] == []
    assert artifact["hardware_commands_run"] == []
    assert artifact["speedup_claim_made"] is False
    assert artifact["timing_claim_made"] is False
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed: run only the gated SSQA readback-scope experiment using "
        "committed host-visible smoke evidence; no timing or speedup claim is "
        "authorized by this ingestion"
    )
    assert ready["authoritative_pinout_ccf_binding"]["path"] == v2.EXP3048_REL_PATH.as_posix()
    assert ready["host_visible_smoke_evidence"]["path"] == v2.EXP3050_REL_PATH.as_posix()


def test_req_hw_093_gatemate_only_scope_keeps_ssqa_blocked(tmp_path: Path) -> None:
    """REQ-HW-093: GateMate evidence alone only permits the flash-smoke scope."""
    _write_required_sources(tmp_path, gatemate_ready=True, smoke_ready=False)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_actions"]}

    assert artifact["operator_evidence_ingestion_v3_ready"] is True
    assert artifact["gatemate_rerun_allowed"] is True
    assert artifact["ssqa_readback_allowed"] is False
    assert list(missing) == ["host_visible_smoke_evidence"]
    assert artifact["allowed_next_experiment_scope"] == (
        "operator_allowed: run only the gated GateMate host-visible flash-smoke scope; "
        "no timing or speedup claim is authorized by this ingestion"
    )


def test_req_hw_093_missing_v2_or_capstone_blocks_v3_readiness(tmp_path: Path) -> None:
    """REQ-HW-093: missing prior ledgers honestly block the v3 boundary."""
    _write_v2_sources(tmp_path)
    _write_json(tmp_path, mod.EXP3094_REL_PATH, _capstone_v288())

    artifact = mod.build_artifact(tmp_path)

    assert artifact["operator_evidence_ingestion_v3_ready"] is False
    assert artifact["gatemate_rerun_allowed"] is False
    assert artifact["ssqa_readback_allowed"] is False
    assert mod.EXP3092_REL_PATH.as_posix() in artifact["missing_source_artifacts"]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_093_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-093: writing and fail-closed helper behavior remain deterministic."""
    _write_required_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["operator_evidence_ingestion_v3_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._as_text_list("done=1") == ["done=1"]
    assert mod._coerce_rel_path(" results/example.json ") == Path("results/example.json")
    catalog: dict[str, tuple[Path, str, bool]] = {}
    mod._add_source(catalog, Path(""), "empty", False)
    assert catalog == {}
    assert mod._path_status(tmp_path, Path("missing.txt"), "role") == {
        "path": "missing.txt",
        "present": False,
        "readable": False,
        "evidence_class": "role",
    }
