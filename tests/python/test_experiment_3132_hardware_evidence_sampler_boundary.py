"""Tests for Exp 3132 hardware evidence and sampler boundary v5.

Spec refs: REQ-HW-095, SCENARIO-HW-095.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_evidence_sampler_boundary_3132 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/fpga/spec.md"
REQUIRED_FIELDS = {
    "hardware_evidence_sampler_boundary_v5_ready",
    "hardware_commands_run",
    "gatemate_evidence_complete",
    "ssqa_readback_ready",
    "kv260_evidence_status",
    "polarfire_evidence_status",
    "thrml_tsu_claim_allowed",
    "clut_sampler_boundary",
    "missing_operator_evidence",
    "speedup_claim_allowed",
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clut_payload() -> dict[str, Any]:
    return {
        "clut_backend_integration_boundary_v2_ready": True,
        "default_backend_preserved": True,
        "hardware_claim_made": False,
        "hardware_commands_run": [],
        "flagged_adversarial": True,
        "inference_substrate": {
            "kind": "cpu_numpy_clut_backend",
            "cpu_only": True,
            "executes_hardware": False,
            "hardware_speedup_claim_eligible": False,
        },
        "honest_verdict": "complete: CPU-only cLUT backend; no hardware command",
    }


def _missing_actions() -> list[dict[str, Any]]:
    return [
        {
            "missing_item": "authoritative_pinout_ccf_binding",
            "operator_action": "Provide authoritative GateMate output pinout.",
            "source_artifact": "results/experiment_3048_gatemate_output_contract_operator_package_v1.json",
            "checked_paths": [
                "results/experiment_3048_gatemate_output_contract_operator_package_v1.json",
                "hardware/gatemate/ising_n16_gatemate.ccf",
            ],
            "missing_required_fields": [
                "gatemate_output_contract_ready",
                "host_visible_io_plan_ready",
                "ccf_binding_for_done",
            ],
        },
        {
            "missing_item": "host_visible_smoke_evidence",
            "operator_action": "Commit a passing host-visible smoke transcript.",
            "source_artifact": "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json",
            "checked_paths": ["results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json"],
            "missing_required_fields": [
                "host_visible_transcript_path",
                "readback_hash",
                "per_sample_latency_s",
            ],
        },
    ]


def _gatemate_payload(*, complete: bool = False) -> dict[str, Any]:
    return {
        "artifact": "experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4",
        "operator_evidence_ingestion_v4_ready": True,
        "gatemate_rerun_allowed": complete,
        "ssqa_readback_allowed": complete,
        "missing_operator_actions": [] if complete else _missing_actions(),
        "hardware_commands_run": [],
        "speedup_claim_made": False,
        "host_visible_smoke_evidence": {
            "ready": complete,
            "missing_required_fields": [] if complete else ["readback_hash"],
        },
        "inference_substrate": {
            "kind": "operator_evidence_ingestion_v4",
            "executes_hardware": False,
            "hardware_readback_attempted": False,
            "no_live_model_inference": True,
        },
        "honest_verdict": "complete: operator evidence v4",
    }


def _write_common_sources(root: Path, *, gatemate_complete: bool = False) -> None:
    _write_json(root, mod.EXP3118_CLUT_REL_PATH, _clut_payload())
    _write_json(root, mod.EXP3119_GATEMATE_SSQA_REL_PATH, _gatemate_payload(complete=gatemate_complete))
    _write_json(
        root,
        mod.KV260_LATENCY_REL_PATH,
        {
            "experiment_id": "exp2898-kv260-ising-sampler-hardware-latency-benchmark-v1",
            "inference_substrate": "hardware_smoke",
            "board_transcript_path": mod.KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
            "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        },
    )
    _write_text(root, mod.KV260_LATENCY_TRANSCRIPT_REL_PATH, "ssh kria uio read latency transcript\n")
    _write_json(
        root,
        mod.KV260_CLAIM_BOUNDARY_REL_PATH,
        {
            "kv260_claim_boundary_ready": True,
            "hardware_speedup_claim_eligible": True,
            "speedup_claim_made": True,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "honest_verdict": "complete: historical KV260 same-basis boundary",
        },
    )
    _write_json(
        root,
        mod.POLARFIRE_1000_REL_PATH,
        {
            "polarfire_1000_clause_hash_verified": True,
            "board_reachable": True,
            "transcript_paths": [mod.POLARFIRE_1000_TRANSCRIPT_REL_PATH.as_posix()],
            "no_speedup_claim": True,
            "no_general_acceleration_claim": True,
            "inference_substrate": "hardware_smoke",
            "honest_verdict": "complete: polarfire hash verified",
        },
    )
    _write_json(root, mod.POLARFIRE_1000_TRANSCRIPT_REL_PATH, {"stdout": "hash verified"})
    _write_json(
        root,
        mod.THRML_PARITY_REL_PATH,
        {
            "no_tsu_hardware_claim": True,
            "inference_substrate": "simulator_parity",
            "honest_verdict": "complete: thrml simulator parity no hardware claim",
        },
    )
    _write_text(root, mod.HARDWARE_WISHLIST_REL_PATH, "GateMate blocked; cLUT CPU-only\n")
    _write_text(root, mod.OPS_STATUS_REL_PATH, "GateMate/SSQA missing operator evidence\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "cLUT CPU-only and hardware blocked\n")
    _write_text(root, mod.GATEMATE_CCF_REL_PATH, "# build-only CCF; no physical Pin_out\n")
    _write_json(
        root,
        Path("results/experiment_3122_archive_v290_activate_v291.json"),
        {"carry_forward_blockers": [{"blocker_id": "missing_operator_visible_hardware_evidence"}]},
    )


def test_req_hw_095_spec_anchor_and_script_exist() -> None:
    """REQ-HW-095: OpenSpec declares the v5 hardware boundary contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-095" in spec
    assert "SCENARIO-HW-095" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_hw_095_blocked_operator_evidence_keeps_claims_bounded(tmp_path: Path) -> None:
    """SCENARIO-HW-095: incomplete GateMate/SSQA evidence blocks claims."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_evidence"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["hardware_evidence_sampler_boundary_v5_ready"] is True
    assert artifact["hardware_commands_run"] == []
    assert artifact["gatemate_evidence_complete"] is False
    assert artifact["ssqa_readback_ready"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["kv260_evidence_status"] == (
        "authenticated_historical_kv260_hardware_evidence_present_no_new_execution"
    )
    assert artifact["polarfire_evidence_status"] == (
        "authenticated_polarfire_dispatch_hash_evidence_present_no_speedup_claim"
    )
    assert artifact["thrml_tsu_claim_allowed"] is False
    assert artifact["clut_sampler_boundary"] == {
        "decision": "CPU simulation",
        "cpu_only": True,
        "ready": True,
        "hardware_claim_allowed": False,
        "hardware_commands_run": [],
        "source_artifact": mod.EXP3118_CLUT_REL_PATH.as_posix(),
        "flagged_adversarial": True,
    }
    assert set(missing) == {
        "gatemate:authoritative_pinout_ccf_binding",
        "ssqa:host_visible_smoke_evidence",
        "thrml_tsu:authenticated_tsu_hardware_evidence",
        "clut:authenticated_hardware_execution_evidence",
    }
    assert missing["ssqa:host_visible_smoke_evidence"]["missing_required_fields"] == [
        "host_visible_transcript_path",
        "readback_hash",
        "per_sample_latency_s",
    ]
    assert artifact["sampler_boundary_decisions"] == {
        "gatemate": "blocked",
        "ssqa": "blocked",
        "kv260": "authenticated hardware evidence",
        "polarfire": "authenticated hardware evidence",
        "thrml_tsu": "out-of-scope",
        "clut": "CPU simulation",
    }
    assert artifact["post_exp3119_evidence_scan"]["new_operator_evidence_found"] is False
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["inference_substrate"]["hardware_commands_run"] == []
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_095_complete_gatemate_evidence_still_does_not_allow_speedup(
    tmp_path: Path,
) -> None:
    """REQ-HW-095: complete GateMate evidence does not become a speedup claim."""
    _write_common_sources(tmp_path, gatemate_complete=True)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gatemate_evidence_complete"] is True
    assert artifact["ssqa_readback_ready"] is True
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["sampler_boundary_decisions"]["gatemate"] == "authenticated hardware evidence"
    assert artifact["sampler_boundary_decisions"]["ssqa"] == "authenticated hardware evidence"
    assert {
        row["missing_item"] for row in artifact["missing_operator_evidence"]
    } == {
        "thrml_tsu:authenticated_tsu_hardware_evidence",
        "clut:authenticated_hardware_execution_evidence",
    }


def test_req_hw_095_missing_required_sources_blocks_precondition(tmp_path: Path) -> None:
    """REQ-HW-095: missing exp3118/exp3119 sources block v5 readiness."""
    artifact = mod.build_artifact(tmp_path)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert artifact["hardware_evidence_sampler_boundary_v5_ready"] is False
    assert artifact["gatemate_evidence_complete"] is False
    assert artifact["ssqa_readback_ready"] is False
    assert sources[mod.EXP3118_CLUT_REL_PATH.as_posix()]["required"] is True
    assert sources[mod.EXP3119_GATEMATE_SSQA_REL_PATH.as_posix()]["required"] is True
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_095_post_exp3119_scan_detects_real_operator_evidence(tmp_path: Path) -> None:
    """REQ-HW-095: post-exp3119 scan notices new host-visible evidence fields."""
    _write_common_sources(tmp_path)
    _write_json(
        tmp_path,
        Path("results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"),
        {
            "host_visible_transcript_path": "logs/gatemate-smoke.txt",
            "readback_hash": "a" * 64,
        },
    )

    artifact = mod.build_artifact(tmp_path)
    scan = artifact["post_exp3119_evidence_scan"]

    assert scan["new_operator_evidence_found"] is True
    assert scan["matched_paths"] == [
        "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
    ]


def test_req_hw_095_write_artifact_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-095: writer and JSON helpers are deterministic and fail closed."""
    _write_common_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["hardware_evidence_sampler_boundary_v5_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(scalar) == {}
    assert mod.sha256_file(tmp_path / mod.KV260_LATENCY_TRANSCRIPT_REL_PATH) == _sha256_text(
        "ssh kria uio read latency transcript\n"
    )
