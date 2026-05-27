"""Tests for Exp 3174 hardware/tooling boundary v8.

Spec refs: REQ-HW-098, SCENARIO-HW-098.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_tooling_boundary_3174 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/fpga/spec.md"
REQUIRED_FIELDS = {
    "hardware_tooling_boundary_v8_ready",
    "authenticated_speedup_claim_allowed",
    "hardware_commands_run",
    "local_tooling_checks",
    "cuda_status",
    "kv260_status",
    "gatemate_status",
    "polarfire_status",
    "extropic_thrml_status",
    "kona_status",
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fixture_tooling_checks() -> dict[str, dict[str, Any]]:
    return {
        "thrml": {
            "module": "thrml",
            "distribution": "thrml",
            "available": True,
            "version": "0.0-test",
            "check_method": "importlib.util.find_spec + importlib.metadata.version",
            "installs_packages": False,
            "hardware_commands_run": [],
            "hardware_performance_evidence": False,
        },
        "xgrammar": {
            "module": "xgrammar",
            "distribution": "xgrammar",
            "available": False,
            "version": None,
            "check_method": "importlib.util.find_spec + importlib.metadata.version",
            "installs_packages": False,
            "hardware_commands_run": [],
            "hardware_performance_evidence": False,
        },
        "llguidance": {
            "module": "llguidance",
            "distribution": "llguidance",
            "available": False,
            "version": None,
            "check_method": "importlib.util.find_spec + importlib.metadata.version",
            "installs_packages": False,
            "hardware_commands_run": [],
            "hardware_performance_evidence": False,
        },
    }


def _write_common_sources(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3160_BOUNDARY_REL_PATH,
        {
            "hardware_sampler_evidence_boundary_v7_ready": True,
            "authenticated_speedup_claim_allowed": False,
            "hardware_commands_run": [],
            "cuda_status": "runtime_ready_no_sampler_speedup_claim_flagged_adversarial",
            "kv260_status": "authenticated_historical_board_evidence_scoped_no_fresh_speedup_claim",
            "gatemate_status": "blocked_operator_evidence_incomplete_no_speedup_claim",
            "polarfire_status": "authenticated_historical_dispatch_evidence_no_speedup_claim",
            "extropic_thrml_status": "architecture_reference_only_no_local_tsu_or_xtr_execution",
            "kona_status": "architecture_reference_only_no_local_kona_or_aleph_execution",
            "speedup_claim_made": False,
            "honest_verdict": "complete: v7 boundary fixture",
        },
    )
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
            "flagged_adversarial": True,
            "honest_verdict": "success: local GGUF runtime fixture",
        },
    )
    _write_json(
        root,
        mod.GATEMATE_EVIDENCE_REL_PATH,
        {
            "operator_evidence_ingestion_v4_ready": True,
            "gatemate_rerun_allowed": False,
            "missing_operator_actions": [
                {
                    "missing_item": "authoritative_pinout_ccf_binding",
                    "missing_required_fields": ["host_visible_io_plan_ready"],
                }
            ],
            "honest_verdict": "complete: GateMate evidence remains incomplete",
        },
    )
    _write_json(
        root,
        mod.KV260_LATENCY_REL_PATH,
        {
            "inference_substrate": "hardware_smoke",
            "board_transcript_path": mod.KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
            "honest_verdict": "complete: historical KV260 transcript fixture",
        },
    )
    _write_text(root, mod.KV260_LATENCY_TRANSCRIPT_REL_PATH, "ssh kria latency transcript\n")
    _write_json(
        root,
        mod.POLARFIRE_1000_REL_PATH,
        {
            "polarfire_1000_clause_hash_verified": True,
            "board_reachable": True,
            "transcript_paths": [mod.POLARFIRE_1000_TRANSCRIPT_REL_PATH.as_posix()],
            "no_speedup_claim": True,
            "honest_verdict": "complete: PolarFire transcript fixture",
        },
    )
    _write_json(root, mod.POLARFIRE_1000_TRANSCRIPT_REL_PATH, {"stdout": "hash verified"})
    _write_text(
        root,
        mod.RESEARCH_REFERENCES_REL_PATH,
        "\n".join(
            [
                "Extropic THRML and TSU public context only.",
                "https://extropic.ai/software",
                "https://github.com/extropic-ai/thrml",
                "https://github.com/mlc-ai/xgrammar",
                "https://github.com/guidance-ai/llguidance",
                "https://logicalintelligence.com/kona-ebms-energy-based-models",
            ]
        ),
    )
    _write_text(
        root,
        mod.HARDWARE_WISHLIST_REL_PATH,
        "CUDA runtime operational; board speedup claims require command transcripts.\n",
    )
    _write_text(root, mod.OPS_STATUS_REL_PATH, "No authenticated hardware speedup.\n")


def test_req_hw_098_spec_anchor_and_script_exist() -> None:
    """REQ-HW-098: OpenSpec declares the v8 hardware/tooling boundary."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-098" in spec
    assert "SCENARIO-HW-098" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_hw_098_writes_partitioned_no_claim_boundary(tmp_path: Path) -> None:
    """SCENARIO-HW-098: public references, local tooling, and performance stay separate."""
    _write_common_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, local_tooling_checks=_fixture_tooling_checks())
    source_classes = {row["evidence_class"] for row in artifact["source_artifacts"]}
    public_urls = {row["url"] for row in artifact["evidence_partitions"]["public_ecosystem_references"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["hardware_tooling_boundary_v8_ready"] is True
    assert artifact["authenticated_speedup_claim_allowed"] is False
    assert artifact["hardware_commands_run"] == []
    assert artifact["speedup_claim_made"] is False
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
        "local_tooling_check",
        "ops_documentation",
        "public_ecosystem_reference",
        "wishlist_intent",
    }
    assert public_urls >= {
        "https://extropic.ai/software",
        "https://github.com/extropic-ai/thrml",
        "https://github.com/mlc-ai/xgrammar",
        "https://github.com/guidance-ai/llguidance",
        "https://logicalintelligence.com/kona-ebms-energy-based-models",
    }
    assert artifact["evidence_partitions"]["local_tooling_checks"] == artifact[
        "local_tooling_checks"
    ]
    assert artifact["evidence_partitions"]["authenticated_performance_evidence"] == [
        {
            "substrate": "kv260",
            "status": "authenticated_historical_board_evidence_scoped_no_fresh_speedup_claim",
            "source_artifacts": [
                mod.KV260_LATENCY_REL_PATH.as_posix(),
                mod.KV260_LATENCY_TRANSCRIPT_REL_PATH.as_posix(),
            ],
            "fresh_speedup_claim_allowed": False,
        },
        {
            "substrate": "polarfire",
            "status": "authenticated_historical_dispatch_evidence_no_speedup_claim",
            "source_artifacts": [
                mod.POLARFIRE_1000_REL_PATH.as_posix(),
                mod.POLARFIRE_1000_TRANSCRIPT_REL_PATH.as_posix(),
            ],
            "fresh_speedup_claim_allowed": False,
        },
    ]
    assert all(
        check["hardware_performance_evidence"] is False
        for check in artifact["local_tooling_checks"].values()
    )
    assert artifact["inference_substrate"] == {
        "kind": "hardware_tooling_boundary_v8",
        "source": "checked_in_local_artifacts_and_local_import_metadata",
        "local_repo_only": True,
        "executes_hardware": False,
        "hardware_readback_attempted": False,
        "board_flash_attempted": False,
        "synthesis_or_pnr_run": False,
        "executes_models": False,
        "no_live_model_inference": True,
        "remote_hardware_called": False,
        "installs_packages": False,
        "hardware_commands_run": [],
    }
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_098_tooling_probe_records_import_metadata_without_hardware() -> None:
    """REQ-HW-098: import availability is recorded without installing or benchmarking."""

    def fake_find_spec(module_name: str) -> object | None:
        return object() if module_name in {"thrml", "xgrammar"} else None

    def fake_version(distribution_name: str) -> str:
        if distribution_name == "thrml":
            return "1.2.3"
        raise mod.metadata.PackageNotFoundError(distribution_name)

    checks = mod.probe_local_tooling(
        probes=(
            ("thrml", "thrml"),
            ("xgrammar", "xgrammar"),
            ("llguidance", "llguidance"),
        ),
        find_spec=fake_find_spec,
        version=fake_version,
    )

    assert checks["thrml"]["available"] is True
    assert checks["thrml"]["version"] == "1.2.3"
    assert checks["xgrammar"]["available"] is True
    assert checks["xgrammar"]["version"] is None
    assert checks["llguidance"]["available"] is False
    assert checks["llguidance"]["version"] is None
    assert all(check["installs_packages"] is False for check in checks.values())
    assert all(check["hardware_commands_run"] == [] for check in checks.values())
    assert (
        mod._cuda_status(
            {},
            {
                "sota_runtime_ready_v3": True,
                "llama_cpp_gpu_offload_verified": True,
                "usable_response_count": 1,
                "flagged_adversarial": True,
            },
        )
        == "runtime_ready_no_sampler_speedup_claim_flagged_adversarial"
    )
    assert (
        mod._cuda_status(
            {},
            {
                "sota_runtime_ready_v3": True,
                "llama_cpp_gpu_offload_verified": True,
                "usable_response_count": 1,
                "flagged_adversarial": False,
            },
        )
        == "runtime_ready_no_sampler_speedup_claim"
    )


def test_req_hw_098_missing_required_sources_blocks_precondition(tmp_path: Path) -> None:
    """REQ-HW-098: missing core sources block readiness without probing devices."""
    artifact = mod.build_artifact(tmp_path, local_tooling_checks=_fixture_tooling_checks())
    sources = {row["path"]: row for row in artifact["source_artifacts"] if "path" in row}

    assert artifact["hardware_tooling_boundary_v8_ready"] is False
    assert artifact["cuda_status"] == "blocked_cuda_runtime_evidence_missing"
    assert artifact["kv260_status"] == "blocked_missing_authenticated_kv260_transcript"
    assert (
        artifact["polarfire_status"] == "blocked_missing_polarfire_dispatch_or_readback_transcript"
    )
    assert sources[mod.EXP3160_BOUNDARY_REL_PATH.as_posix()]["required"] is True
    assert sources[mod.EXP3146_BOUNDARY_REL_PATH.as_posix()]["required"] is True
    assert sources[mod.RESEARCH_REFERENCES_REL_PATH.as_posix()]["required"] is True
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_098_writer_and_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-HW-098: writer and helper readers are deterministic and fail closed."""
    _write_common_sources(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(tmp_path, local_tooling_checks=_fixture_tooling_checks())
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["hardware_tooling_boundary_v8_ready"] is True
    assert mod.read_json_object(bad) == {}
    assert mod.read_json_object(scalar) == {}
    assert mod.sha256_file(tmp_path / mod.KV260_LATENCY_TRANSCRIPT_REL_PATH) == _sha256_text(
        "ssh kria latency transcript\n"
    )
    assert mod.sha256_file(tmp_path / "missing.txt") is None
