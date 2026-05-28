"""Build the Exp 3235 CUDA driver-boundary operator package.

Spec refs: REQ-REPORT-3235, SCENARIO-REPORT-3235.

This module is an aggregation-only operator package. It reads prior CUDA and
llama.cpp boundary artifacts, maps each observed failure to the smallest probe
that owns that layer, and defines the exact booleans that Exp 3236 and Exp
3237 must emit before any full mandated GGUF receipt can run again. It does not
load a model, rebuild llama.cpp, or claim that the CUDA runtime has been
repaired.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.cuda_driver_boundary_operator_package.v1"
EXPERIMENT_ID = "exp3235"
TASK_ID = "exp3235-cuda-driver-boundary-operator-package-v1"
ARTIFACT = "experiment_3235_cuda_driver_boundary_operator_package_v1"
MILESTONE = "2026.05.300"
RUN_DATE = "20260528"
RANDOM_SEED = 3235
OUTPUT_REL_PATH = Path("results/experiment_3235_cuda_driver_boundary_operator_package_v1.json")

CLAUDE_REL_PATH = Path("CLAUDE.md")
HARDWARE_WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")
EXP3206_REL_PATH = Path("results/experiment_3206_cuda_env_forensics_ledger_v1.json")
EXP3207_REL_PATH = Path("results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json")
EXP3220_REL_PATH = Path("results/experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.json")
CAPSTONE_V298_REL_PATH = Path("results/experiment_3232_capstone_v298.json")
CAPSTONE_V299_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")

SOURCE_PATHS: tuple[tuple[str, Path], ...] = (
    ("claude_guidance", CLAUDE_REL_PATH),
    ("hardware_wishlist", HARDWARE_WISHLIST_REL_PATH),
    ("cuda_env_forensics_3206", EXP3206_REL_PATH),
    ("llama_cpp_rebuild_gate_3207", EXP3207_REL_PATH),
    ("hermetic_cuda_runtime_3220", EXP3220_REL_PATH),
    ("capstone_v298", CAPSTONE_V298_REL_PATH),
    ("capstone_v299", CAPSTONE_V299_REL_PATH),
    ("conductor_log", CONDUCTOR_LOG_REL_PATH),
)

RECOMMENDED_NEXT_TASK = "exp3236-isolated-cuda-python-smoke-v1"
MISSING_SOURCE_NEXT_TASK = "repair_missing_exp3235_sources_before_exp3236"


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence when the file is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _as_mapping(payload)


def read_text_file(path: Path) -> str:
    """Read text evidence, returning an empty string for absent optional notes."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so an operator can confirm the exact evidence inputs."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3235: aggregate CUDA boundary evidence into an operator contract."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3206 = read_json_object(root_path / EXP3206_REL_PATH)
    exp3207 = read_json_object(root_path / EXP3207_REL_PATH)
    exp3220 = read_json_object(root_path / EXP3220_REL_PATH)
    capstone_v298 = read_json_object(root_path / CAPSTONE_V298_REL_PATH)
    capstone_v299 = read_json_object(root_path / CAPSTONE_V299_REL_PATH)
    conductor_log = read_text_file(root_path / CONDUCTOR_LOG_REL_PATH)
    sources = _source_artifacts(root_path)
    missing = [row["path"] for row in sources if not row["present"]]
    package_ready = not missing
    prior_evidence = _prior_cuda_failure_evidence(
        exp3206, exp3207, exp3220, capstone_v298, capstone_v299, conductor_log
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "principle_annotations": _principle_annotations(),
        "cuda_boundary_package_ready": package_ready,
        "prior_cuda_failure_evidence": prior_evidence,
        "probe_boundary_map": _probe_boundary_map(exp3206, exp3207, exp3220),
        "selected_python_acceptance_contract": _selected_python_acceptance_contract(
            exp3206, exp3220
        ),
        "isolated_cuda_acceptance_contract": _isolated_cuda_acceptance_contract(exp3220),
        "llama_cpp_acceptance_contract": _llama_cpp_acceptance_contract(exp3207, exp3220),
        "downstream_gate_fields": [
            "cuda_python_smoke_passed",
            "llama_cpp_cuda_receipt_ready",
            "clean_rerun_allowed",
        ],
        "full_gguf_rerun_allowed_now": False,
        "recommended_next_task": RECOMMENDED_NEXT_TASK if package_ready else MISSING_SOURCE_NEXT_TASK,
        "do_not_do": _do_not_do(),
        "protected_files_untouched": {
            "scripts/research_conductor.py": True,
            "research-roadmap.yaml": True,
            "research-roadmap-next.yaml": True,
            "ops/status.md": True,
            "ops/changelog.md": True,
            "_bmad/traceability.md": True,
        },
        "missing_required_sources": missing,
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row["sha256"] for row in sources if row["sha256"]},
        "no_new_heavyweight_model_inference": True,
        "no_llama_cpp_rebuild": True,
        "no_full_gguf_load": True,
        "no_conductor_execution": True,
        "no_push": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3235 operator package JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _as_mapping(payload: Any) -> JsonDict:
    return dict(payload) if isinstance(payload, dict) else {}


def _duration(started_s: float, now_s: float | None = None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0, round(end - started_s, 6))


def _source_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for role, rel_path in SOURCE_PATHS:
        path = root / rel_path
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "present": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _prior_cuda_failure_evidence(
    exp3206: JsonDict,
    exp3207: JsonDict,
    exp3220: JsonDict,
    capstone_v298: JsonDict,
    capstone_v299: JsonDict,
    conductor_log: str,
) -> JsonDict:
    return {
        "driver_visible": bool(
            exp3206.get("nvidia_smi_available") or exp3220.get("nvidia_smi_available")
        ),
        "selected_python": str(exp3220.get("selected_python") or exp3206.get("selected_python") or ""),
        "selected_python_cuda_failed": not bool(
            exp3206.get("torch_cuda_available_clean_subprocess")
            and _positive_int(exp3206.get("torch_cuda_device_count_clean_subprocess"))
            and exp3220.get("selected_python_cuda_ok_after")
        ),
        "isolated_cuda_runtime_failed": not bool(exp3220.get("isolated_cuda_venv_cuda_ok")),
        "llama_cpp_offload_failed": not _llama_cpp_prior_passed(exp3207, exp3220),
        "full_receipt_gate_blocked": (
            not bool(exp3207.get("cuda_receipt_ready"))
            or not bool(exp3220.get("cuda_receipt_ready_candidate"))
            or "Full local SOTA GGUF receipt" in conductor_log
        ),
        "upstream_recommended_actions": [
            str(exp3206.get("recommended_next_action") or ""),
            str(exp3220.get("recommended_next_action") or ""),
        ],
        "capstone_top_gaps": [
            str(capstone_v298.get("next_top_gap") or ""),
            str(capstone_v299.get("next_top_gap") or ""),
        ],
        "failure_summary": (
            "Driver visibility exists, but selected Python torch CUDA init, "
            "cuda.bindings runtime init, and llama.cpp/offload readiness did not pass. "
            "This keeps full mandated GGUF receipts blocked."
        ),
    }


def _probe_boundary_map(exp3206: JsonDict, exp3207: JsonDict, exp3220: JsonDict) -> JsonDict:
    return {
        "driver_visibility": {
            "probe": "nvidia-smi",
            "owned_by": ["exp3206.nvidia_smi", "exp3220.nvidia_smi"],
            "observed_passed": bool(
                exp3206.get("nvidia_smi_available") or exp3220.get("nvidia_smi_available")
            ),
            "observed_fields": {
                "exp3206.nvidia_smi_available": exp3206.get("nvidia_smi_available"),
                "exp3220.nvidia_smi_available": exp3220.get("nvidia_smi_available"),
                "exp3220.gpu_count_nvidia_smi": exp3220.get("gpu_count_nvidia_smi"),
                "exp3220.driver_version": exp3220.get("driver_version"),
            },
        },
        "selected_python_torch_cuda_init": {
            "probe": (
                "selected_python clean subprocess: import torch; torch.cuda.is_available(); "
                "torch.cuda.device_count()"
            ),
            "owned_by": ["exp3206.torch_clean_subprocess", "exp3220.selected_python_probe_after"],
            "observed_passed": bool(
                exp3206.get("torch_cuda_available_clean_subprocess")
                and _positive_int(exp3206.get("torch_cuda_device_count_clean_subprocess"))
                and exp3220.get("selected_python_cuda_ok_after")
            ),
            "observed_fields": {
                "exp3206.torch_cuda_available_clean_subprocess": exp3206.get(
                    "torch_cuda_available_clean_subprocess"
                ),
                "exp3206.torch_cuda_device_count_clean_subprocess": exp3206.get(
                    "torch_cuda_device_count_clean_subprocess"
                ),
                "exp3220.selected_python_cuda_ok_after": exp3220.get(
                    "selected_python_cuda_ok_after"
                ),
            },
        },
        "cuda_bindings_runtime_init": {
            "probe": "cuda.bindings.runtime clean subprocess: cudaGetDeviceCount()",
            "owned_by": [
                "exp3220.selected_python_cuda_runtime_probe_after",
                "exp3220.isolated_cuda_venv.probe",
            ],
            "observed_passed": _cuda_bindings_prior_passed(exp3220),
            "observed_fields": _cuda_bindings_observed_fields(exp3220),
        },
        "llama_cpp_gpu_offload_support": {
            "probe": (
                "llama_cpp clean subprocess: import llama_cpp; llama_supports_gpu_offload(); "
                "llama_system_info"
            ),
            "owned_by": [
                "exp3206.llama_cpp_clean_subprocess",
                "exp3207.llama_cpp_clean_subprocess_after",
                "exp3220.llama_cpp_linkage_probe",
            ],
            "observed_passed": _llama_cpp_prior_passed(exp3207, exp3220),
            "observed_fields": {
                "exp3207.clean_subprocess_gpu_offload_probe_passed": exp3207.get(
                    "clean_subprocess_gpu_offload_probe_passed"
                ),
                "exp3207.cpu_fallback_only": exp3207.get("cpu_fallback_only"),
                "exp3220.llama_cpp_supports_gpu_offload": _mapping_get(
                    exp3220.get("llama_cpp_linkage_probe"), "llama_cpp_supports_gpu_offload"
                ),
                "exp3220.llama_cpp_stderr_summary": _mapping_get(
                    exp3220.get("llama_cpp_linkage_probe"), "stderr_summary"
                ),
            },
        },
    }


def _selected_python_acceptance_contract(exp3206: JsonDict, exp3220: JsonDict) -> JsonDict:
    return {
        "downstream_task_id": "exp3236-isolated-cuda-python-smoke-v1",
        "gate_field": "cuda_python_smoke_passed",
        "probe_family": "selected Python torch CUDA init",
        "required_boolean_fields": {
            "cuda_driver_visible": True,
            "selected_python_torch_cuda_available": True,
            "selected_python_torch_device_count_gt_zero": True,
        },
        "required_numeric_fields": {"selected_python_torch_device_count": "> 0"},
        "blocked_by_prior_fields": {
            "exp3206.torch_cuda_available_clean_subprocess": exp3206.get(
                "torch_cuda_available_clean_subprocess"
            ),
            "exp3206.torch_cuda_device_count_clean_subprocess": exp3206.get(
                "torch_cuda_device_count_clean_subprocess"
            ),
            "exp3220.selected_python_cuda_ok_after": exp3220.get("selected_python_cuda_ok_after"),
        },
        "acceptance_rule": (
            "Set cuda_python_smoke_passed=true only when nvidia-smi is visible, "
            "the selected project Python imports torch in a clean subprocess, "
            "torch.cuda.is_available() is true, and torch sees at least one CUDA device."
        ),
    }


def _isolated_cuda_acceptance_contract(exp3220: JsonDict) -> JsonDict:
    return {
        "downstream_task_id": "exp3236-isolated-cuda-python-smoke-v1",
        "gate_field": "cuda_python_smoke_passed",
        "probe_family": "cuda.bindings.runtime",
        "required_boolean_fields": {
            "cuda_bindings_import_ok": True,
            "cuda_bindings_runtime_init_ok": True,
            "cuda_bindings_device_count_gt_zero": True,
        },
        "required_numeric_fields": {"cuda_bindings_device_count": "> 0"},
        "observed_prior_fields": {
            "exp3220.selected_python_cuda_runtime_ok": _mapping_get(
                exp3220.get("selected_python_cuda_runtime_probe_after"), "cuda_runtime_ok"
            ),
            "exp3220.selected_python_cuda_runtime_device_count": _mapping_get(
                exp3220.get("selected_python_cuda_runtime_probe_after"), "device_count"
            ),
            "exp3220.isolated_cuda_venv_cuda_ok": exp3220.get("isolated_cuda_venv_cuda_ok"),
            "exp3220.isolated_cuda_venv_device_count": _mapping_get(
                _mapping_get(exp3220.get("isolated_cuda_venv"), "probe"), "device_count"
            ),
        },
        "acceptance_rule": (
            "Treat cuda.bindings as the runtime-driver boundary check. It must import, "
            "cudaGetDeviceCount() must return success, and device count must be nonzero."
        ),
    }


def _llama_cpp_acceptance_contract(exp3207: JsonDict, exp3220: JsonDict) -> JsonDict:
    return {
        "downstream_task_id": "exp3237-llama-cpp-cuda-receipt-smoke-v2",
        "upstream_gate_field": "cuda_python_smoke_passed",
        "gate_field": "llama_cpp_cuda_receipt_ready",
        "probe_family": "llama.cpp GPU offload support and minimal receipt smoke",
        "required_boolean_fields": {
            "upstream_cuda_python_smoke_passed": True,
            "llama_cpp_import_ok": True,
            "llama_cpp_supports_gpu_offload": True,
            "llama_cpp_smoke_completed": True,
            "llama_cpp_cpu_fallback_used": False,
            "llama_cpp_cuda_init_error_seen": False,
        },
        "observed_prior_fields": {
            "exp3207.clean_subprocess_gpu_offload_probe_passed": exp3207.get(
                "clean_subprocess_gpu_offload_probe_passed"
            ),
            "exp3207.cpu_fallback_only": exp3207.get("cpu_fallback_only"),
            "exp3207.cuda_receipt_ready": exp3207.get("cuda_receipt_ready"),
            "exp3220.llama_cpp_import_ok": _mapping_get(
                exp3220.get("llama_cpp_linkage_probe"), "llama_cpp_import_ok"
            ),
            "exp3220.llama_cpp_supports_gpu_offload": _mapping_get(
                exp3220.get("llama_cpp_linkage_probe"), "llama_cpp_supports_gpu_offload"
            ),
        },
        "acceptance_rule": (
            "Run Exp 3237 only after cuda_python_smoke_passed=true. Set "
            "llama_cpp_cuda_receipt_ready=true only when llama.cpp imports, reports GPU "
            "offload support, completes the smallest smoke without CPU-only fallback, and "
            "does not emit ggml_cuda_init failure."
        ),
    }


def _do_not_do() -> list[str]:
    return [
        "Do not run a blind full mandated GGUF receipt rerun before cuda_python_smoke_passed=true.",
        "Do not rebuild llama.cpp until selected Python CUDA is known-good in a clean subprocess.",
        "Do not treat CPU fallback, import-only success, or nvidia-smi visibility as CUDA receipt.",
        "Do not claim the runtime is repaired from this aggregation-only package.",
    ]


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Reads upstream artifacts and logs; performs no live model inference.",
        "probe_ownership": (
            "Driver visibility, selected Python torch CUDA init, cuda.bindings runtime init, "
            "and llama.cpp offload are separate acceptance boundaries."
        ),
        "gate_order": "exp3236 cuda_python_smoke_passed before exp3237 llama_cpp_cuda_receipt_ready.",
        "full_receipt_block": "full_gguf_rerun_allowed_now remains false until staged smokes pass.",
    }


def _cuda_bindings_observed_fields(exp3220: JsonDict) -> JsonDict:
    selected = _as_mapping(exp3220.get("selected_python_cuda_runtime_probe_after"))
    isolated = _as_mapping(_mapping_get(exp3220.get("isolated_cuda_venv"), "probe"))
    return {
        "exp3220.selected_python_cuda_bindings_import_ok": selected.get(
            "cuda_bindings_import_ok"
        ),
        "exp3220.selected_python_cuda_runtime_ok": selected.get("cuda_runtime_ok"),
        "exp3220.selected_python_cuda_bindings_device_count": selected.get("device_count"),
        "exp3220.isolated_cuda_bindings_import_ok": isolated.get("cuda_bindings_import_ok"),
        "exp3220.isolated_cuda_runtime_ok": isolated.get("cuda_runtime_ok"),
        "exp3220.isolated_cuda_bindings_device_count": isolated.get("device_count"),
    }


def _cuda_bindings_prior_passed(exp3220: JsonDict) -> bool:
    selected = _as_mapping(exp3220.get("selected_python_cuda_runtime_probe_after"))
    isolated = _as_mapping(_mapping_get(exp3220.get("isolated_cuda_venv"), "probe"))
    return bool(
        (
            selected.get("cuda_bindings_import_ok")
            and selected.get("cuda_runtime_ok")
            and _positive_int(selected.get("device_count"))
        )
        or (
            isolated.get("cuda_bindings_import_ok")
            and isolated.get("cuda_runtime_ok")
            and _positive_int(isolated.get("device_count"))
        )
    )


def _llama_cpp_prior_passed(exp3207: JsonDict, exp3220: JsonDict) -> bool:
    linkage = _as_mapping(exp3220.get("llama_cpp_linkage_probe"))
    stderr = str(linkage.get("stderr_summary") or "")
    return bool(
        exp3207.get("clean_subprocess_gpu_offload_probe_passed")
        and not exp3207.get("cpu_fallback_only")
        and linkage.get("llama_cpp_supports_gpu_offload")
        and "ggml_cuda_init" not in stderr
    )


def _mapping_get(payload: Any, key: str) -> Any:
    return payload.get(key) if isinstance(payload, dict) else None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _reproducibility_checksum(artifact: JsonDict) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(artifact: JsonDict) -> str:
    if not artifact["cuda_boundary_package_ready"]:
        return (
            "complete: cuda_boundary_package_ready=false; source evidence incomplete; "
            "full_gguf_rerun_allowed_now=false"
        )
    return (
        "complete: cuda_boundary_package_ready=true; full_gguf_rerun_allowed_now=false; "
        "next=exp3236-isolated-cuda-python-smoke-v1; runtime remains blocked pending smoke fields"
    )
