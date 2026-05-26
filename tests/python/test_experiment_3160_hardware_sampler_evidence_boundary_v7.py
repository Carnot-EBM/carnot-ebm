"""Tests for Exp 3160 hardware sampler evidence boundary v7.

Spec refs: REQ-HW-097, SCENARIO-HW-097.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_sampler_evidence_boundary_3160 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/fpga/spec.md"
REQUIRED_FIELDS = {
    "hardware_sampler_evidence_boundary_v7_ready",
    "authenticated_speedup_claim_allowed",
    "no_hardware_commands_run",
    "evidence_sources",
    "missing_operator_evidence",
    "cuda_status",
    "kv260_status",
    "gatemate_status",
    "polarfire_status",
    "extropic_thrml_status",
    "kona_status",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
SPEEDUP_EVIDENCE_FIELDS = {
    "command_transcript",
    "board_or_device_identity",
    "baseline",
    "artifact_checksum",
    "workload",
    "reproducibility_notes",
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


def _gatemate_payload(*, complete: bool = False) -> dict[str, Any]:
    missing_actions = [
        {
            "checked_paths": [
                "results/experiment_3048_gatemate_output_contract_operator_package_v1.json",
                "hardware/gatemate/ising_n16_gatemate.ccf",
            ],
            "missing_item": "authoritative_pinout_ccf_binding",
            "missing_required_fields": [
                "gatemate_output_contract_ready",
                "host_visible_io_plan_ready",
                "ccf_binding_for_done",
            ],
            "operator_action": "Provide an authoritative GateMate output pinout.",
            "source_artifact": "results/experiment_3048_gatemate_output_contract_operator_package_v1.json",
        },
        {
            "checked_paths": ["results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json"],
            "missing_item": "host_visible_smoke_evidence",
            "missing_required_fields": [
                "host_visible_transcript_path",
                "transcript_sha256",
                "per_sample_latency_s",
            ],
            "operator_action": "Commit a passing host-visible smoke transcript.",
            "source_artifact": "results/experiment_3050_gatemate_host_visible_flash_smoke_v5.json",
        },
    ]
    return {
        "operator_evidence_ingestion_v4_ready": True,
        "gatemate_rerun_allowed": complete,
        "ssqa_readback_allowed": complete,
        "speedup_claim_made": False,
        "hardware_commands_run": [],
        "missing_operator_actions": [] if complete else missing_actions,
        "host_visible_smoke_evidence": {
            "ready": complete,
            "missing_required_fields": [] if complete else ["readback_hash"],
        },
        "honest_verdict": "complete: operator evidence fixture",
    }


def _write_common_sources(
    root: Path,
    *,
    gatemate_complete: bool = False,
    tsu_authenticated: bool = False,
    kona_authenticated: bool = False,
    cuda_flagged: bool = True,
) -> None:
    _write_json(
        root,
        mod.EXP3146_BOUNDARY_REL_PATH,
        {
            "hardware_sampler_evidence_boundary_v6_ready": True,
            "speedup_claim_allowed": False,
            "hardware_commands_run": [],
            "honest_verdict": "complete: v6 boundary fixture",
        },
    )
    _write_json(
        root,
        mod.CUDA_RUNTIME_REL_PATH,
        {
            "sota_runtime_ready_v3": True,
            "llama_cpp_gpu_offload_verified": True,
            "usable_response_count": 1,
            "tokens_per_second": 0.232859,
            "total_tokens_generated": 2,
            "flagged_adversarial": cuda_flagged,
            "models_missing_from_cache": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "honest_verdict": "success: mandated SOTA GGUF produced usable GPU-backed output",
        },
    )
    _write_json(
        root,
        mod.CUDA_CAPSTONE_REL_PATH,
        {
            "adversarially_flagged_artifacts": ["exp2862"] if cuda_flagged else [],
            "runtime_summary": {
                "sota_runtime_artifact_clean": not cuda_flagged,
                "source_reported_sota_runtime_ready_v3": True,
            },
            "honest_verdict": "complete: capstone fixture",
        },
    )
    _write_json(root, mod.GATEMATE_EVIDENCE_REL_PATH, _gatemate_payload(complete=gatemate_complete))
    _write_json(
        root,
        mod.KV260_LATENCY_REL_PATH,
        {
            "inference_substrate": "hardware_smoke",
            "board_transcript_path": mod.KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
            "honest_verdict": "complete: kv260 transcript recorded",
        },
    )
    _write_text(
        root, mod.KV260_LATENCY_TRANSCRIPT_REL_PATH, "ssh kria uio read latency transcript\n"
    )
    _write_json(
        root,
        mod.KV260_CLAIM_BOUNDARY_REL_PATH,
        {
            "kv260_claim_boundary_ready": True,
            "speedup_claim_made": True,
            "hardware_speedup_claim_eligible": True,
            "honest_verdict": "complete: historical speedup claim fixture",
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
            "authenticated_tsu_hardware_evidence": tsu_authenticated,
            "no_tsu_hardware_claim": not tsu_authenticated,
            "inference_substrate": "simulator_parity",
            "honest_verdict": "complete: thrml simulator fixture",
        },
    )
    _write_json(
        root,
        mod.KONA_BOUNDARY_REL_PATH,
        {
            "authenticated_local_kona_access_or_execution_evidence": kona_authenticated,
            "external_dependency_claim_allowed": kona_authenticated,
            "hardware_evidence_summary": {
                "hardware_execution_claim_allowed": kona_authenticated,
            },
            "honest_verdict": "complete: kona boundary fixture",
        },
    )
    _write_text(
        root,
        mod.HARDWARE_WISHLIST_REL_PATH,
        "CUDA runtime operational; KV260, GateMate, PolarFire require transcripts before speedup.\n",
    )
    _write_text(
        root,
        mod.RESEARCH_REFERENCES_REL_PATH,
        "Extropic THRML and Logical Intelligence Kona/Aleph are public architecture references only.\n",
    )
    _write_text(root, mod.OPS_STATUS_REL_PATH, "Hardware speedup claims remain blocked.\n")
    _write_text(root, mod.OPS_CHANGELOG_REL_PATH, "Prior hardware boundary refresh recorded.\n")


def test_req_hw_097_spec_anchor_and_script_exist() -> None:
    """REQ-HW-097: OpenSpec declares the v7 evidence boundary."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-097" in spec
    assert "SCENARIO-HW-097" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_hw_097_writes_no_claim_boundary_with_source_classes(tmp_path: Path) -> None:
    """SCENARIO-HW-097: evidence is classified and speedups stay blocked."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"]: row for row in artifact["missing_operator_evidence"]}
    source_classes = {row["evidence_class"] for row in artifact["evidence_sources"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["hardware_sampler_evidence_boundary_v7_ready"] is True
    assert artifact["authenticated_speedup_claim_allowed"] is False
    assert artifact["no_hardware_commands_run"] is True
    assert artifact["cuda_status"] == "runtime_ready_no_sampler_speedup_claim_flagged_adversarial"
    assert (
        artifact["kv260_status"]
        == "authenticated_historical_board_evidence_scoped_no_fresh_speedup_claim"
    )
    assert artifact["gatemate_status"] == "blocked_operator_evidence_incomplete_no_speedup_claim"
    assert (
        artifact["polarfire_status"]
        == "authenticated_historical_dispatch_evidence_no_speedup_claim"
    )
    assert (
        artifact["extropic_thrml_status"]
        == "architecture_reference_only_no_local_tsu_or_xtr_execution"
    )
    assert artifact["kona_status"] == "architecture_reference_only_no_local_kona_or_aleph_execution"
    assert source_classes == {
        "checked_in_local_artifact",
        "local_operator_evidence",
        "ops_documentation",
        "public_architecture_reference",
        "wishlist_intent",
    }
    assert set(
        missing["authenticated_speedup_claim:complete_local_evidence_bundle"][
            "missing_required_fields"
        ]
    ) == (SPEEDUP_EVIDENCE_FIELDS)
    assert "gatemate:authoritative_pinout_ccf_binding" in missing
    assert "ssqa:host_visible_smoke_evidence" in missing
    assert "extropic_thrml:authenticated_tsu_xtr_z1_execution_evidence" in missing
    assert "kona_aleph:authenticated_local_kona_or_aleph_execution_evidence" in missing
    assert artifact["inference_substrate"] == {
        "kind": "hardware_sampler_evidence_boundary_v7",
        "source": "checked_in_local_artifacts",
        "local_repo_only": True,
        "executes_hardware": False,
        "hardware_readback_attempted": False,
        "board_flash_attempted": False,
        "synthesis_or_pnr_run": False,
        "executes_models": False,
        "no_live_model_inference": True,
        "hardware_commands_run": [],
    }
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_097_complete_local_rows_still_do_not_promote_speedup(tmp_path: Path) -> None:
    """REQ-HW-097: local readiness is not a speedup claim without the full bundle."""
    _write_common_sources(
        tmp_path,
        gatemate_complete=True,
        tsu_authenticated=True,
        kona_authenticated=True,
        cuda_flagged=False,
    )

    artifact = mod.build_artifact(tmp_path)
    missing = {row["missing_item"] for row in artifact["missing_operator_evidence"]}

    assert artifact["cuda_status"] == "runtime_ready_no_sampler_speedup_claim"
    assert artifact["gatemate_status"] == "operator_evidence_complete_no_speedup_claim"
    assert (
        artifact["extropic_thrml_status"] == "authenticated_extropic_tsu_evidence_no_speedup_claim"
    )
    assert artifact["kona_status"] == "authenticated_local_kona_or_aleph_evidence_no_speedup_claim"
    assert artifact["authenticated_speedup_claim_allowed"] is False
    assert missing == {"authenticated_speedup_claim:complete_local_evidence_bundle"}


def test_req_hw_097_missing_required_sources_blocks_precondition(tmp_path: Path) -> None:
    """REQ-HW-097: missing core sources block v7 readiness without hardware probing."""
    artifact = mod.build_artifact(tmp_path)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}

    assert artifact["hardware_sampler_evidence_boundary_v7_ready"] is False
    assert artifact["cuda_status"] == "blocked_cuda_runtime_evidence_missing"
    assert artifact["kv260_status"] == "blocked_missing_authenticated_kv260_transcript"
    assert (
        artifact["polarfire_status"] == "blocked_missing_polarfire_dispatch_or_readback_transcript"
    )
    assert sources[mod.EXP3146_BOUNDARY_REL_PATH.as_posix()]["required"] is True
    assert sources[mod.CUDA_RUNTIME_REL_PATH.as_posix()]["required"] is True
    assert sources[mod.GATEMATE_EVIDENCE_REL_PATH.as_posix()]["required"] is True
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_097_writer_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-097: writer and helper readers are deterministic and fail closed."""
    _write_common_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["hardware_sampler_evidence_boundary_v7_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(scalar) == {}
    assert mod.sha256_file(tmp_path / mod.KV260_LATENCY_TRANSCRIPT_REL_PATH) == _sha256_text(
        "ssh kria uio read latency transcript\n"
    )
    assert mod.sha256_file(tmp_path / "missing.txt") is None
