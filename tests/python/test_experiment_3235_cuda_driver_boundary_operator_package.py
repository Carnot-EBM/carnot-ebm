"""Tests for Exp 3235 CUDA driver-boundary operator package.

Spec refs: REQ-REPORT-3235, SCENARIO-REPORT-3235.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cuda_driver_boundary_operator_package_3235 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "cuda_boundary_package_ready",
    "selected_python_acceptance_contract",
    "isolated_cuda_acceptance_contract",
    "llama_cpp_acceptance_contract",
    "full_gguf_rerun_allowed_now",
    "recommended_next_task",
    "protected_files_untouched",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exp3206() -> dict[str, Any]:
    return {
        "experiment_id": "exp3206",
        "milestone": "2026.05.297",
        "selected_python": "/repo/.venv/bin/python",
        "nvidia_smi_available": True,
        "gpu_count_nvidia_smi": 1,
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "torch_cuda_available_clean_subprocess": False,
        "torch_cuda_device_count_clean_subprocess": 0,
        "llama_cpp_version": "0.3.23",
        "llama_cpp_origin": "/repo/.venv/lib/python3.14/site-packages/llama_cpp/__init__.py",
        "llama_cpp_cuda_build_detected": True,
        "cuda_env_diagnosed": True,
        "cuda_init_clean": False,
        "recommended_next_action": "repair_selected_python_torch_cuda_before_full_receipt",
        "clean_subprocess_stderr_tail": [
            "torch: CUDA initialization: CUDA unknown error",
            "llama_cpp: ggml_cuda_init: failed to initialize CUDA: unknown error",
        ],
        "honest_verdict": (
            "blocked_selected_python_torch_cuda: cuda_env_diagnosed=true; "
            "cuda_init_clean=false"
        ),
    }


def _exp3207() -> dict[str, Any]:
    return {
        "experiment_id": "exp3207",
        "milestone": "2026.05.297",
        "env_ledger_artifact": "/repo/results/experiment_3206_cuda_env_forensics_ledger_v1.json",
        "rebuild_attempted": False,
        "torch_cuda_available_after": False,
        "llama_cpp_cuda_build_detected_after": True,
        "clean_subprocess_gpu_offload_probe_passed": False,
        "cpu_fallback_only": True,
        "cuda_receipt_ready": False,
        "clean_rerun_allowed_candidate": False,
        "blocker": "selected_python_torch_cuda_unavailable: cuda_available=False; device_count=0",
        "honest_verdict": "blocked_selected_python_torch_cuda: selected Python CUDA unavailable",
    }


def _exp3220() -> dict[str, Any]:
    return {
        "experiment_id": "exp3220",
        "milestone": "2026.05.298",
        "inference_substrate": "cuda_runtime_forensics_no_model",
        "selected_python": "/repo/.venv/bin/python",
        "selected_python_cuda_ok_before": False,
        "selected_python_cuda_ok_after": False,
        "isolated_cuda_venv_created": True,
        "isolated_cuda_venv_cuda_ok": False,
        "cuda_visible_devices": "0",
        "nvidia_smi_available": True,
        "gpu_count_nvidia_smi": 1,
        "driver_version": "595.71.05",
        "torch_version_selected": "2.11.0+cu128",
        "torch_cuda_version_selected": "12.8",
        "selected_python_cuda_runtime_probe_after": {
            "cuda_bindings_import_ok": True,
            "cuda_runtime_ok": False,
            "device_count": 0,
            "cuda_error": "cudaErrorUnknown",
        },
        "isolated_cuda_venv": {
            "probe": {
                "cuda_bindings_import_ok": True,
                "cuda_runtime_ok": False,
                "device_count": 0,
                "cuda_error": "cudaErrorUnknown",
            }
        },
        "llama_cpp_linkage_probe": {
            "llama_cpp_import_ok": True,
            "llama_cpp_supports_gpu_offload": False,
            "llama_system_info": "",
            "stderr_summary": "ggml_cuda_init: failed to initialize CUDA",
        },
        "environment_pollution_findings": [
            {"kind": "path_contains_rocm", "variable": "PATH", "severity": "warn"},
            {"kind": "path_contains_xdna_tooling", "variable": "PATH", "severity": "warn"},
        ],
        "repair_actions_attempted": [
            {"action": "explicit_cuda_visible_devices_probe", "status": "cuda_failed"},
            {"action": "sanitized_selected_python_cuda_probe", "status": "cuda_failed"},
            {"action": "probe_isolated_cuda_venv", "status": "cuda_failed"},
        ],
        "cuda_receipt_ready_candidate": False,
        "recommended_next_action": "repair_system_driver_cuda_runtime_boundary",
        "honest_verdict": (
            "blocked_cuda_runtime: cuda_receipt_ready_candidate=false; "
            "recommended_next_action=repair_system_driver_cuda_runtime_boundary"
        ),
    }


def _capstone_v298() -> dict[str, Any]:
    return {
        "experiment_id": "exp3232",
        "milestone": "2026.05.298",
        "paper_ready": False,
        "publication_blocker_count": 100,
        "next_top_gap": "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt",
        "honest_verdict": (
            "complete: capstone_ready=true; paper_ready=false; publication_blocker_count=100; "
            "next_top_gap=repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"
        ),
    }


def _capstone_v299() -> dict[str, Any]:
    return {
        "experiment_id": "exp3223",
        "milestone": "2026.05.299",
        "capstone_v299_ready": True,
        "paper_ready": False,
        "publication_blocker_count": 100,
        "v4_outcome": "blocked_missing_exp3222_result",
        "next_top_gap": "cuda_chain_for_full_local_sota_receipts",
        "honest_verdict": (
            "complete: capstone_v299_ready=true; paper_ready=false; "
            "next_top_gap=cuda_chain_for_full_local_sota_receipts"
        ),
    }


def _write_upstream(root: Path, *, omit_3220: bool = False) -> None:
    _write_text(
        root,
        mod.CLAUDE_REL_PATH,
        "Every experiment artifact MUST declare an inference_substrate. "
        "aggregation_from_upstream_artifacts reads upstream JSON only.\n",
    )
    _write_text(
        root,
        mod.HARDWARE_WISHLIST_REL_PATH,
        "Dual RTX 3090 CUDA local SOTA runtime repair remains a boundary task.\n",
    )
    _write_json(root, mod.EXP3206_REL_PATH, _exp3206())
    _write_json(root, mod.EXP3207_REL_PATH, _exp3207())
    if not omit_3220:
        _write_json(root, mod.EXP3220_REL_PATH, _exp3220())
    _write_json(root, mod.CAPSTONE_V298_REL_PATH, _capstone_v298())
    _write_json(root, mod.CAPSTONE_V299_REL_PATH, _capstone_v299())
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "\n".join(
            [
                "| 2026-05-27 20:01 UTC | CUDA environment forensics ledger for selected Python | OK |",
                "| 2026-05-27 20:56 UTC | llama.cpp CUDA rebuild and clean subprocess smoke | OK |",
                "| 2026-05-27 20:58 UTC | Full local SOTA GGUF receipt v5 gated on exp3207 | GATE_BLOCK |",
                "| 2026-05-27 23:54 UTC | Hermetic CUDA runtime repair ledger for selected Python | OK |",
                "| 2026-05-27 23:56 UTC | llama.cpp CUDA offload receipt smoke gated on hermetic CUDA | GATE_BLOCK |",
            ]
        )
        + "\n",
    )


def test_req_report_3235_spec_anchor_exists() -> None:
    """REQ-REPORT-3235: OpenSpec declares the package before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3235" in spec
    assert "SCENARIO-REPORT-3235" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "cuda_python_smoke_passed" in spec
    assert "llama_cpp_cuda_receipt_ready" in spec


def test_scenario_report_3235_builds_operator_package(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3235: CUDA boundary evidence becomes a gated operator package."""

    _write_upstream(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)
    second = mod.build_artifact(tmp_path, started_s=100.0, now_s=101.0)
    sources = {row["path"]: row for row in artifact["source_artifacts"]}
    probe_map = artifact["probe_boundary_map"]
    selected_contract = artifact["selected_python_acceptance_contract"]
    isolated_contract = artifact["isolated_cuda_acceptance_contract"]
    llama_contract = artifact["llama_cpp_acceptance_contract"]

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3235"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.300"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["cuda_boundary_package_ready"] is True
    assert artifact["full_gguf_rerun_allowed_now"] is False
    assert artifact["recommended_next_task"] == "exp3236-isolated-cuda-python-smoke-v1"
    assert artifact["downstream_gate_fields"] == [
        "cuda_python_smoke_passed",
        "llama_cpp_cuda_receipt_ready",
        "clean_rerun_allowed",
    ]
    assert artifact["prior_cuda_failure_evidence"]["driver_visible"] is True
    assert artifact["prior_cuda_failure_evidence"]["selected_python_cuda_failed"] is True
    assert artifact["prior_cuda_failure_evidence"]["isolated_cuda_runtime_failed"] is True
    assert artifact["prior_cuda_failure_evidence"]["llama_cpp_offload_failed"] is True
    assert artifact["prior_cuda_failure_evidence"]["full_receipt_gate_blocked"] is True
    assert artifact["prior_cuda_failure_evidence"]["capstone_top_gaps"] == [
        "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt",
        "cuda_chain_for_full_local_sota_receipts",
    ]

    assert probe_map["driver_visibility"]["probe"] == "nvidia-smi"
    assert probe_map["driver_visibility"]["observed_passed"] is True
    assert probe_map["selected_python_torch_cuda_init"]["probe"] == (
        "selected_python clean subprocess: import torch; torch.cuda.is_available(); "
        "torch.cuda.device_count()"
    )
    assert probe_map["selected_python_torch_cuda_init"]["observed_passed"] is False
    assert probe_map["cuda_bindings_runtime_init"]["probe"] == (
        "cuda.bindings.runtime clean subprocess: cudaGetDeviceCount()"
    )
    assert probe_map["cuda_bindings_runtime_init"]["observed_passed"] is False
    assert probe_map["llama_cpp_gpu_offload_support"]["probe"] == (
        "llama_cpp clean subprocess: import llama_cpp; llama_supports_gpu_offload(); "
        "llama_system_info"
    )
    assert probe_map["llama_cpp_gpu_offload_support"]["observed_passed"] is False

    assert selected_contract["downstream_task_id"] == "exp3236-isolated-cuda-python-smoke-v1"
    assert selected_contract["gate_field"] == "cuda_python_smoke_passed"
    assert selected_contract["required_boolean_fields"] == {
        "cuda_driver_visible": True,
        "selected_python_torch_cuda_available": True,
        "selected_python_torch_device_count_gt_zero": True,
    }
    assert selected_contract["blocked_by_prior_fields"] == {
        "exp3206.torch_cuda_available_clean_subprocess": False,
        "exp3206.torch_cuda_device_count_clean_subprocess": 0,
        "exp3220.selected_python_cuda_ok_after": False,
    }
    assert isolated_contract["probe_family"] == "cuda.bindings.runtime"
    assert isolated_contract["required_boolean_fields"] == {
        "cuda_bindings_import_ok": True,
        "cuda_bindings_runtime_init_ok": True,
        "cuda_bindings_device_count_gt_zero": True,
    }
    assert isolated_contract["observed_prior_fields"]["exp3220.isolated_cuda_venv_cuda_ok"] is False
    assert llama_contract["downstream_task_id"] == "exp3237-llama-cpp-cuda-receipt-smoke-v2"
    assert llama_contract["upstream_gate_field"] == "cuda_python_smoke_passed"
    assert llama_contract["gate_field"] == "llama_cpp_cuda_receipt_ready"
    assert llama_contract["required_boolean_fields"] == {
        "upstream_cuda_python_smoke_passed": True,
        "llama_cpp_import_ok": True,
        "llama_cpp_supports_gpu_offload": True,
        "llama_cpp_smoke_completed": True,
        "llama_cpp_cpu_fallback_used": False,
        "llama_cpp_cuda_init_error_seen": False,
    }

    assert any("Do not run a blind full mandated GGUF receipt rerun" in item for item in artifact["do_not_do"])
    assert artifact["protected_files_untouched"] == {
        "scripts/research_conductor.py": True,
        "research-roadmap.yaml": True,
        "research-roadmap-next.yaml": True,
        "ops/status.md": True,
        "ops/changelog.md": True,
        "_bmad/traceability.md": True,
    }
    assert artifact["no_new_heavyweight_model_inference"] is True
    assert artifact["no_llama_cpp_rebuild"] is True
    assert artifact["no_full_gguf_load"] is True
    assert artifact["no_conductor_execution"] is True
    assert sources[mod.EXP3220_REL_PATH.as_posix()]["sha256"] == _sha256(
        tmp_path / mod.EXP3220_REL_PATH
    )
    assert artifact["source_checksums"][mod.EXP3206_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.EXP3206_REL_PATH
    )
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "repaired" not in artifact["honest_verdict"].lower()


def test_req_report_3235_writer_and_missing_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3235: writer persists JSON and missing upstream evidence blocks readiness."""

    _write_upstream(tmp_path, omit_3220=True)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["cuda_boundary_package_ready"] is False
    assert saved["full_gguf_rerun_allowed_now"] is False
    assert mod.EXP3220_REL_PATH.as_posix() in saved["missing_required_sources"]
    assert saved["recommended_next_task"] == "repair_missing_exp3235_sources_before_exp3236"
    assert saved["honest_verdict"].startswith("complete:")
    assert "source evidence incomplete" in saved["honest_verdict"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_text_file(tmp_path / "missing.txt") == ""
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._duration(3.0, 2.0) == 0.0
    assert mod._as_mapping([]) == {}
    assert mod._positive_int(1) is True
    assert mod._positive_int(0) is False
