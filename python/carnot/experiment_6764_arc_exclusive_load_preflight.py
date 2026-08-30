"""Prove exclusive CUDA load, production selfparse, and clean teardown.

The experiment selects one eligible RTX 3090. Each model then runs in a fresh
worker that owns a GPU lease and one llama.cpp child. The result is an
admission receipt only. It contains no ARC score or model-quality claim.

Spec refs: REQ-INFRA-6764, SCENARIO-INFRA-6764-*,
REQ-ARC-WMTE-6764, and SCENARIO-ARC-WMTE-6764-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_6647_receipt_scoped_admission_boundary as exp6647
from carnot import experiment_6752_arc_code_carrying_tool_preflight as exp6752
from carnot import gpu_lease_phase_journal as lease_api
from carnot.agentic.arc_induction_tools import MAX_FIND_OBJECT_RESPONSE_BYTES
from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6764_arc_exclusive_load_preflight.json"
WORK_DIR = REPO_ROOT / "results/.experiment_6764_arc_exclusive_load_preflight"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
EXP6752_PATH = REPO_ROOT / "results/experiment_6752_arc_code_carrying_tool_preflight.json"
EXP6647_PATH = REPO_ROOT / "results/experiment_6647_receipt_scoped_admission_boundary.json"
SCHEMA = "carnot.experiment_6764.arc_exclusive_load_preflight.v1"
RUN_DATE = "20260829"
RANDOM_SEED = 6_764
CONTEXT_REQUESTED = 32_768
FROZEN_FREE_VRAM_THRESHOLD_MB = 22_610
VRAM_RECOVERY_TOLERANCE_MB = 512
RAM_AVAILABLE_FLOOR_BYTES = 64 * 1024**3
DISK_FREE_FLOOR_BYTES = 1024**3
WORKER_TIMEOUT_S = 1_200.0
VRAM_RECOVERY_TIMEOUT_S = 180.0
INFERENCE_SUBSTRATE = "task-owned local llama.cpp CUDA GGUF"
PRODUCTION_ROUTE = "induce_with_tool_loop/selfparse/dispatch_tool"
CLAIM_BOUNDARY = (
    "Transport and teardown admission only. It measures no ARC quality, claims no solve, "
    "and keeps model timings unpooled."
)
EXPECTED_GPU_UUIDS = (
    "GPU-b52387a2-c625-de87-8d34-e6f64e684bab",
    "GPU-7971baff-9583-eaa6-2292-393f930a28f9",
)
MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "model_id": "unsloth/Qwen3.8-27B-GGUF",
        "role": "immutable_scored_arc_generator_full_load",
        "repo_substr": "Qwen3.8-27B",
        "filename": "Qwen3.8-27B-Q4_K_M.gguf",
        "expected_sha256": "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
        "required_vram_mb": 17_815,
        "max_tokens": 1_024,
    },
    {
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_transport_canary",
        "repo_substr": "Qwen3.6-35B-A3B",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "expected_sha256": "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
        "required_vram_mb": 22_610,
        "max_tokens": 512,
    },
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "models_used",
    "model_specs",
    "live_model_invoked",
    "rows",
    "device_inventory_before",
    "device_selection_receipt",
    "lease_owner_receipts",
    "phase_rows",
    "gpu_receipts",
    "runtime_context_by_model",
    "production_selfparse_receipt",
    "owned_processes_terminated",
    "lease_release_receipts",
    "vram_recovery_receipts",
    "unrelated_processes_signaled",
    "arc_exclusive_load_ready",
    "claim_boundary",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned shape lets readers reject incompatible receipts.",
    "experiment": "The identifier binds this file to Exp6764.",
    "title": "The title states the admission-only scope.",
    "run_date": "The planning date anchors the requested hardware check.",
    "status": "The status separates ready, partial, and blocked runs.",
    "field_principles": "Each field states why it is needed.",
    "inference_substrate": "The value excludes CPU, remote, and substituted inference.",
    "duration_s": "Monotonic wall time makes real work visible.",
    "random_seed": "The fixed seed makes model requests repeatable.",
    "reproducibility_checksum": "The checksum detects receipt or input drift.",
    "models_used": "Exact paths and hashes prevent model substitution.",
    "model_specs": "The adversarial verifier consumes this exact-model evidence alias.",
    "live_model_invoked": "Readiness requires both real local decodes.",
    "rows": "One row per model phase keeps lifecycle evidence atomic.",
    "device_inventory_before": "The initial inventory exposes unrelated work and capacity.",
    "device_selection_receipt": "The receipt proves frozen least-used selection.",
    "lease_owner_receipts": "PID and start time bind each lease to its worker.",
    "phase_rows": "Ordered monotonic rows prove the complete lease lifecycle.",
    "gpu_receipts": "Separate model receipts prevent timing and evidence pooling.",
    "runtime_context_by_model": "Observed context distinguishes load evidence from intent.",
    "production_selfparse_receipt": "Parser, dispatch, bound, and transcript evidence prove the live route.",
    "owned_processes_terminated": "A true value rules out an owned process leak.",
    "lease_release_receipts": "Durable releases prove the selected device was returned.",
    "vram_recovery_receipts": "Before and after samples prove owned VRAM returned.",
    "unrelated_processes_signaled": "An empty list proves the no-preemption boundary.",
    "arc_exclusive_load_ready": "The all-admission boolean requires both complete canaries.",
    "claim_boundary": "The boundary prevents admission evidence from becoming a quality claim.",
    "gate_check_summary": "Blocked results retain each failed check and observed value.",
    "verifier_is_oracle": "False states that this transport receipt is not a correctness oracle.",
    "verdict_class": "A closed class makes the result machine-readable.",
    "honest_verdict": "The terminal text states readiness or an owned block.",
    "preconditions_checked": "The full gate record explains why workers did or did not start.",
    "owned_process_receipts": "Process identities show which task-owned PIDs ended.",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for content hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text and name the algorithm in the receipt."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a large model in chunks so the process does not copy it into RAM."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential field."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gpu_receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Hash one model receipt without its self-referential field."""

    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def _load_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _run_command(command: Sequence[str], timeout_s: float = 30.0) -> JsonDict:
    """Run one read-only host probe and retain bounded output."""

    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "duration_s": round(time.monotonic() - started, 6),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "duration_s": round(time.monotonic() - started, 6),
        "stdout": result.stdout[-16_000:],
        "stderr": result.stderr[-16_000:],
    }


def _model_path(spec: Mapping[str, Any]) -> str | None:
    if spec["model_id"] == MODEL_SPECS[1]["model_id"]:
        return resolve_cached_gguf(str(spec["model_id"]), "Q4_K_M")
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    return _resolve_gguf(str(spec["repo_substr"]))


def resolve_model_specs() -> list[JsonDict]:
    """Resolve and verify only the two mandated cached GGUF files."""

    rows: list[JsonDict] = []
    for spec in MODEL_SPECS:
        resolved = _model_path(spec) or ""
        path = Path(resolved)
        present = path.is_file() and path.name == spec["filename"]
        model_hash = sha256_file(path) if present else "missing"
        tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path) if present else None)
        rows.append(
            {
                **dict(spec),
                "model_path": str(path.resolve()) if present else resolved,
                "model_sha256": model_hash,
                "model_size_bytes": path.stat().st_size if present else 0,
                "resolved": bool(present and model_hash == spec["expected_sha256"]),
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": tokenizer_ok,
                    "detail": tokenizer_detail,
                },
            }
        )
    return rows


def nvidia_smi_inventory() -> JsonDict:
    """Read the two physical devices and all active compute rows."""

    device_receipt = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    process_receipt = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    processes: list[JsonDict] = []
    for line in process_receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    devices: list[JsonDict] = []
    for line in device_receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            uuid = parts[1]
            active = [row for row in processes if row["gpu_uuid"] == uuid]
            devices.append(
                {
                    "index": int(parts[0]),
                    "uuid": uuid,
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "temperature_c": int(parts[6]),
                    "utilization_pct": int(parts[7]),
                    "active_compute_processes": active,
                }
            )
        except ValueError:
            continue
    return {
        "device_query": device_receipt,
        "process_query": process_receipt,
        "devices": devices,
    }


def rank_eligible_devices(
    devices: Sequence[Mapping[str, Any]],
    *,
    threshold_mb: int = FROZEN_FREE_VRAM_THRESHOLD_MB,
) -> JsonDict:
    """Rank fixed RTX 3090 rows with the preregistered ordering."""

    retained: list[JsonDict] = []
    for source in devices:
        row = deepcopy(dict(source))
        active = row.get("active_compute_processes")
        active = active if isinstance(active, list) else []
        exact_identity = (
            row.get("uuid") in EXPECTED_GPU_UUIDS and row.get("name") == "NVIDIA GeForce RTX 3090"
        )
        row["active_compute_count"] = len(active)
        row["eligible"] = bool(
            exact_identity and int(row.get("memory_free_mb", -1)) >= int(threshold_mb)
        )
        row["ineligibility_reasons"] = []
        if not exact_identity:
            row["ineligibility_reasons"].append("unexpected_device_identity")
        if int(row.get("memory_free_mb", -1)) < int(threshold_mb):
            row["ineligibility_reasons"].append("free_vram_below_frozen_floor")
        retained.append(row)
    eligible = [row for row in retained if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -int(row["memory_free_mb"]),
            int(row.get("temperature_c", 10_000)),
            int(row["active_compute_count"]),
            int(row.get("index", 10_000)),
        )
    )
    return {
        "rank_policy": ["free_vram_desc", "temperature_asc", "active_compute_asc"],
        "frozen_free_vram_threshold_mb": int(threshold_mb),
        "evaluated_devices": retained,
        "ranked_eligible_devices": eligible,
        "eligible_device_count": len(eligible),
        "selected_device": deepcopy(eligible[0]) if eligible else None,
    }


def _llama_cpp_receipt() -> JsonDict:
    configured = os.environ.get("CARNOT_LLAMA_SERVER", "")
    paths = [
        Path(configured) if configured else Path("/__not_configured__"),
        Path.home() / ".cache/llama.cpp-master/build/bin/llama-server",
    ]
    server = next((path for path in paths if path.is_file()), paths[-1])
    linked = _run_command(("ldd", str(server))) if server.is_file() else {}
    linked_text = str(linked.get("stdout", "")) + str(linked.get("stderr", ""))
    try:
        from llama_cpp import llama_cpp

        python_cuda: Any = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - the artifact must keep the import failure
        python_cuda = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(server),
        "exists": server.is_file(),
        "executable": os.access(server, os.X_OK),
        "sha256": sha256_file(server),
        "dynamic_link": linked,
        "cuda_linked": "libggml-cuda" in linked_text and "libcuda" in linked_text,
        "python_cuda_offload": python_cuda,
    }


def _host_resources(root: Path) -> JsonDict:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        memory = {}
    disk = shutil.disk_usage(root)
    return {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "ram_total_bytes": memory.get("MemTotal", 0),
        "ram_available_bytes": memory.get("MemAvailable", 0),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def choose_free_ports(count: int = 2) -> list[int]:
    """Ask the kernel for distinct loopback ports and release them for workers."""

    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def port_is_free(port: int) -> bool:
    """Return true only when a loopback bind succeeds now."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def collect_preconditions(root: Path = REPO_ROOT) -> JsonDict:
    """Check every required input before any lease or model worker starts."""

    prior_6752 = _load_json(EXP6752_PATH)
    prior_6647 = _load_json(EXP6647_PATH)
    models = resolve_model_specs()
    inventory = nvidia_smi_inventory()
    selection = rank_eligible_devices(inventory["devices"])
    llama_cpp = _llama_cpp_receipt()
    resources = _host_resources(root)
    ports = choose_free_ports(len(MODEL_SPECS))
    try:
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
        from carnot.agentic.arc_induction_tool_loop import induce_with_tool_loop
        from carnot.agentic.arc_induction_tools import dispatch_tool, parse_xml_tool_calls

        production_imports = all(
            callable(value)
            for value in (
                LocalGGUFProposer,
                induce_with_tool_loop,
                dispatch_tool,
                parse_xml_tool_calls,
            )
        )
    except Exception:  # noqa: BLE001 - a failed import is a precondition value
        production_imports = False
    context_receipt = prior_6752.get("context_observed_by_model", {})
    prior_model_hashes = {
        row.get("model_id"): row.get("model_sha256")
        for row in prior_6752.get("models_used", [])
        if isinstance(row, Mapping)
    }
    checks: list[JsonDict] = [
        {
            "check": "exp6752_preflight_ready",
            "expected": True,
            "observed": prior_6752.get("arc_context_tool_preflight_ready"),
            "validator_errors": exp6752.validate_artifact(prior_6752)
            if prior_6752
            else ["missing"],
            "passed": prior_6752.get("arc_context_tool_preflight_ready") is True
            and not exp6752.validate_artifact(prior_6752),
        },
        {
            "check": "exp6647_task_owned_admission",
            "expected": 1.0,
            "observed": prior_6647.get("task_owned_admission_ready_score"),
            "validator_errors": exp6647.validate_artifact(prior_6647)
            if prior_6647
            else ["missing"],
            "passed": prior_6647.get("task_owned_admission_ready_score") == 1.0
            and not exp6647.validate_artifact(prior_6647),
        },
    ]
    for model in models:
        checks.extend(
            [
                {
                    "check": f"exact_cached_model:{model['model_id']}",
                    "expected": {
                        "filename": model["filename"],
                        "sha256": model["expected_sha256"],
                    },
                    "observed": {
                        "path": model.get("model_path"),
                        "sha256": model.get("model_sha256"),
                        "resolved": model.get("resolved"),
                    },
                    "passed": model.get("resolved") is True,
                },
                {
                    "check": f"embedded_tokenizer:{model['model_id']}",
                    "expected": "llama.cpp_embedded_gguf",
                    "observed": deepcopy(model.get("tokenizer")),
                    "passed": (model.get("tokenizer") or {}).get("loadable") is True,
                },
                {
                    "check": f"context_32k_support:{model['model_id']}",
                    "expected": {"context_at_least": CONTEXT_REQUESTED},
                    "observed": {
                        "prior_context": context_receipt.get(model["model_id"]),
                        "prior_model_sha256": prior_model_hashes.get(model["model_id"]),
                        "current_model_sha256": model.get("model_sha256"),
                    },
                    "passed": isinstance(context_receipt.get(model["model_id"]), int)
                    and context_receipt[model["model_id"]] >= CONTEXT_REQUESTED
                    and prior_model_hashes.get(model["model_id"]) == model.get("model_sha256"),
                },
            ]
        )
    identity_rows = {row.get("uuid"): row.get("name") for row in inventory["devices"]}
    checks.extend(
        [
            {
                "check": "two_fixed_rtx3090_identities",
                "expected": {uuid: "NVIDIA GeForce RTX 3090" for uuid in EXPECTED_GPU_UUIDS},
                "observed": identity_rows,
                "passed": identity_rows
                == {uuid: "NVIDIA GeForce RTX 3090" for uuid in EXPECTED_GPU_UUIDS},
            },
            {
                "check": "llama_cpp_cuda",
                "expected": True,
                "observed": llama_cpp,
                "passed": llama_cpp.get("exists") is True
                and llama_cpp.get("executable") is True
                and llama_cpp.get("cuda_linked") is True
                and llama_cpp.get("python_cuda_offload") is True,
            },
            {
                "check": "gpu_lease_api",
                "expected": True,
                "observed": {
                    "acquire": callable(getattr(lease_api.GpuLease, "acquire", None)),
                    "transition": callable(getattr(lease_api.GpuLease, "transition", None)),
                    "release": callable(getattr(lease_api.GpuLease, "release", None)),
                },
                "passed": all(
                    callable(getattr(lease_api.GpuLease, name, None))
                    for name in ("acquire", "transition", "release")
                ),
            },
            {
                "check": "ports_free",
                "expected": len(MODEL_SPECS),
                "observed": {str(port): port_is_free(port) for port in ports},
                "passed": len(ports) == len(MODEL_SPECS)
                and all(port_is_free(port) for port in ports),
            },
            {
                "check": "ram_and_disk",
                "expected": {
                    "ram_available_bytes_at_least": RAM_AVAILABLE_FLOOR_BYTES,
                    "disk_free_bytes_at_least": DISK_FREE_FLOOR_BYTES,
                },
                "observed": resources,
                "passed": resources["ram_available_bytes"] >= RAM_AVAILABLE_FLOOR_BYTES
                and resources["disk_free_bytes"] >= DISK_FREE_FLOOR_BYTES,
            },
            {
                "check": "production_selfparse_imports",
                "expected": True,
                "observed": production_imports,
                "passed": production_imports,
            },
            {
                "check": "least_used_eligible_rtx3090",
                "expected": {"free_vram_mb_at_least": FROZEN_FREE_VRAM_THRESHOLD_MB},
                "observed": deepcopy(selection.get("selected_device")),
                "passed": selection.get("selected_device") is not None,
            },
        ]
    )
    return {
        "all_passed": all(check.get("passed") is True for check in checks),
        "checks": checks,
        "models": models,
        "device_inventory_before": inventory["devices"],
        "device_inventory_commands": {
            "devices": inventory["device_query"],
            "processes": inventory["process_query"],
        },
        "device_selection_receipt": selection,
        "ports": ports,
        "llama_cpp": llama_cpp,
        "resources": resources,
        "source_receipts": {
            "exp6752": {"path": str(EXP6752_PATH), "sha256": sha256_file(EXP6752_PATH)},
            "exp6647": {"path": str(EXP6647_PATH), "sha256": sha256_file(EXP6647_PATH)},
        },
    }


def acquire_selected_lease(
    *,
    runtime_dir: Path,
    task_id: str,
    selected_device: Mapping[str, Any],
    expected_model: str,
) -> lease_api.GpuLease:
    """Acquire the selected UUID and bind it to this worker process."""

    return lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir,
        task_id=task_id,
        device_uuid=str(selected_device["uuid"]),
        expected_model=expected_model,
        vram_before_mb=int(selected_device.get("memory_used_mb", 0)),
        ttl_s=WORKER_TIMEOUT_S,
    )


def worker_environment(
    base: Mapping[str, str],
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    *,
    port: int,
) -> dict[str, str]:
    """Set the full worker environment before proposer construction."""

    env = dict(base)
    env.update(
        {
            "CARNOT_ARC_INDUCE_N_CTX": str(CONTEXT_REQUESTED),
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
            "CARNOT_ARC_INDUCE_TOOL_THINK_BUDGET": "256",
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(model["max_tokens"]),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(int(WORKER_TIMEOUT_S)),
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(selected_device["index"]),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_KV_QUANT": "q8_0",
            "CARNOT_ARC_GGUF_PATH": str(model["model_path"]),
            "CARNOT_ARC_EXCLUSIVE_PORT": str(port),
            "PYTHONPATH": str(REPO_ROOT / "python")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def terminate_owned_process(
    process: Any,
    *,
    terminate_timeout_s: float = 30.0,
) -> JsonDict:
    """Stop one recorded child without accepting any unrelated PID input."""

    receipt: JsonDict = {
        "pid": int(process.pid),
        "terminate_sent": False,
        "kill_sent": False,
        "exit_code": process.poll(),
        "absent_after_exit": process.poll() is not None,
        "unrelated_processes_signaled": [],
    }
    if process.poll() is None:
        process.terminate()
        receipt["terminate_sent"] = True
        try:
            process.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            receipt["kill_sent"] = True
            process.wait(timeout=5.0)
    receipt["exit_code"] = process.poll()
    receipt["absent_after_exit"] = process.poll() is not None
    return receipt


def build_vram_recovery_receipt(
    *,
    before_used_mb: int,
    after_used_mb: int,
    owned_pid_present: bool,
    tolerance_mb: int = VRAM_RECOVERY_TOLERANCE_MB,
) -> JsonDict:
    """Compare total device use after teardown with the frozen baseline."""

    delta = abs(int(after_used_mb) - int(before_used_mb))
    return {
        "before_used_mb": int(before_used_mb),
        "after_used_mb": int(after_used_mb),
        "absolute_delta_mb": delta,
        "tolerance_mb": int(tolerance_mb),
        "owned_pid_present": bool(owned_pid_present),
        "passed": not owned_pid_present and delta <= int(tolerance_mb),
    }


def build_production_selfparse_receipt(
    events: Sequence[Mapping[str, Any]],
    *,
    blocks_seen: int,
    blocks_unparsed: int,
) -> JsonDict:
    """Reduce production loop events into one bounded transport receipt."""

    event = next((row for row in events if row.get("parsed_tool") == "find_objects"), {})
    raw = str(event.get("raw_emission") or "")
    bounded = str(event.get("bounded_response") or "")
    arguments = deepcopy(event.get("parsed_arguments"))
    result = deepcopy(event.get("dispatch_result"))
    receipt: JsonDict = {
        "production_route": PRODUCTION_ROUTE,
        "raw_emission": raw,
        "raw_emission_sha256": sha256_text(raw),
        "parsed_tool": event.get("parsed_tool"),
        "parsed_arguments": arguments,
        "dispatch_result": result,
        "bounded_response": bounded,
        "bounded_response_bytes": len(bounded.encode("utf-8")),
        "bounded_response_sha256": sha256_text(bounded),
        "transcript_sha256": sha256_json([raw, bounded]),
        "blocks_seen": int(blocks_seen),
        "blocks_unparsed": int(blocks_unparsed),
    }
    receipt["success"] = not production_selfparse_errors(receipt)
    return receipt


def production_selfparse_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Return every parser, dispatch, response, or transcript failure."""

    errors: list[str] = []
    if receipt.get("production_route") != PRODUCTION_ROUTE:
        errors.append("production_route")
    if receipt.get("parsed_tool") != "find_objects":
        errors.append("parsed_tool")
    arguments = receipt.get("parsed_arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    if not (
        type(arguments.get("t")) is int
        and arguments.get("which") in {"before", "after"}
        and isinstance(arguments.get("predicate_code"), str)
        and type(arguments.get("max_objects")) is int
    ):
        errors.append("parsed_arguments")
    result = receipt.get("dispatch_result")
    result = result if isinstance(result, Mapping) else {}
    if result.get("ok") is not True:
        errors.append("dispatch_result")
    response_bytes = receipt.get("bounded_response_bytes")
    if (
        not isinstance(response_bytes, int)
        or response_bytes <= 0
        or response_bytes > MAX_FIND_OBJECT_RESPONSE_BYTES + 64
    ):
        errors.append("bounded_response")
    if receipt.get("blocks_seen") != 1 or receipt.get("blocks_unparsed") != 0:
        errors.append("xml_blocks")
    for field in ("raw_emission_sha256", "bounded_response_sha256", "transcript_sha256"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get(field, ""))):
            errors.append(field)
    return errors


def phase_rows_for_receipt(receipt: Mapping[str, Any]) -> list[JsonDict]:
    """Bind every lease phase to the model and owner identity."""

    owner = receipt.get("lease_owner")
    owner = owner if isinstance(owner, Mapping) else {}
    rows: list[JsonDict] = []
    for ordinal, event in enumerate(receipt.get("phase_history", [])):
        if not isinstance(event, Mapping):
            continue
        rows.append(
            {
                "model_id": receipt.get("model_id"),
                "role": receipt.get("role"),
                "ordinal": ordinal,
                "phase": event.get("phase"),
                "previous_phase": event.get("previous_phase"),
                "monotonic_ns": event.get("monotonic_ns"),
                "event_checksum": event.get("event_checksum"),
                "owner_pid": owner.get("pid"),
                "owner_pid_start_ticks": owner.get("pid_start_ticks"),
                "device_uuid": owner.get("device_uuid"),
            }
        )
    return rows


def gpu_receipt_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Return every reason one model cannot satisfy exclusive-load readiness."""

    errors: list[str] = []
    if receipt.get("receipt_sha256") != gpu_receipt_checksum(receipt):
        errors.append("receipt_sha256")
    model_id = receipt.get("model_id")
    expected = next((row for row in MODEL_SPECS if row["model_id"] == model_id), None)
    if expected is None:
        return errors + ["model_id"]
    if receipt.get("role") != expected["role"]:
        errors.append("role")
    if receipt.get("model_sha256") != expected["expected_sha256"]:
        errors.append("model_sha256")
    if receipt.get("observed_model_path") != receipt.get("model_path"):
        errors.append("observed_model_path")
    if receipt.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if receipt.get("llama_cpp_cuda") is not True:
        errors.append("llama_cpp_cuda")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get("server_sha256", ""))):
        errors.append("server_sha256")
    device = receipt.get("device")
    device = device if isinstance(device, Mapping) else {}
    if device.get("uuid") not in EXPECTED_GPU_UUIDS:
        errors.append("device_uuid")
    worker = receipt.get("worker_process")
    worker = worker if isinstance(worker, Mapping) else {}
    model_process = receipt.get("model_process")
    model_process = model_process if isinstance(model_process, Mapping) else {}
    owner = receipt.get("lease_owner")
    owner = owner if isinstance(owner, Mapping) else {}
    if not (
        owner.get("pid") == worker.get("pid")
        and owner.get("pid_start_ticks") == worker.get("pid_start_ticks")
        and owner.get("device_uuid") == device.get("uuid")
        and owner.get("expected_model") == receipt.get("model_path")
    ):
        errors.append("lease_owner")
    if worker.get("exit_code") != 0 or worker.get("absent_after_exit") is not True:
        errors.append("worker_process")
    if model_process.get("exit_code") is None or model_process.get("absent_after_exit") is not True:
        errors.append("model_process")
    sequence = [row.get("phase") for row in receipt.get("phase_history", [])]
    if sequence != list(lease_api.COMPLETE_PHASE_SEQUENCE):
        errors.append("phase_sequence")
    release = receipt.get("lease_release")
    release = release if isinstance(release, Mapping) else {}
    if release.get("released") is not True or release.get("phase") != "terminal_complete":
        errors.append("lease_release")
    if receipt.get("runtime_context") != CONTEXT_REQUESTED:
        errors.append("runtime_context")
    layers = receipt.get("gpu_layers")
    layers = layers if isinstance(layers, Mapping) else {}
    if (
        not isinstance(layers.get("offloaded"), int)
        or layers.get("offloaded", 0) <= 0
        or layers.get("offloaded") != layers.get("total")
    ):
        errors.append("gpu_layers")
    if int(receipt.get("peak_owned_vram_mb", 0) or 0) <= 0:
        errors.append("peak_owned_vram_mb")
    if int(receipt.get("resident_owned_vram_mb", 0) or 0) <= 0:
        errors.append("resident_owned_vram_mb")
    if not isinstance(receipt.get("duration_s"), (int, float)) or receipt.get("duration_s", 0) <= 0:
        errors.append("duration_s")
    if (
        receipt.get("live_model_invoked") is not True
        or receipt.get("first_token_observed") is not True
    ):
        errors.append("first_token")
    if production_selfparse_errors(receipt.get("production_selfparse", {})):
        errors.append("production_selfparse")
    if (receipt.get("vram_recovery") or {}).get("passed") is not True:
        errors.append("vram_recovery")
    if receipt.get("unrelated_processes_signaled") != []:
        errors.append("unrelated_processes_signaled")
    if receipt.get("errors") != []:
        errors.append("errors")
    if model_id == MODEL_SPECS[0]["model_id"] and receipt.get("full_load") is not True:
        errors.append("full_load")
    if model_id == MODEL_SPECS[1]["model_id"] and receipt.get("transport_canary") is not True:
        errors.append("transport_canary")
    return list(dict.fromkeys(errors))


def reduce_arc_exclusive_load_ready(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require two separate complete receipts in the fixed model order."""

    return bool(
        len(receipts) == len(MODEL_SPECS)
        and [row.get("model_id") for row in receipts] == [spec["model_id"] for spec in MODEL_SPECS]
        and all(not gpu_receipt_errors(row) for row in receipts)
    )


def _gpu_snapshot(device_uuid: str, owned_pid: int = 0) -> JsonDict:
    inventory = nvidia_smi_inventory()
    device = next((row for row in inventory["devices"] if row.get("uuid") == device_uuid), {})
    active = device.get("active_compute_processes")
    active = active if isinstance(active, list) else []
    owned = next((row for row in active if row.get("pid") == owned_pid), None)
    return {
        **deepcopy(device),
        "owned_pid": int(owned_pid),
        "owned_pid_present": owned is not None,
        "owned_pid_vram_mb": int((owned or {}).get("used_memory_mb", 0) or 0),
        "observed_monotonic_ns": time.monotonic_ns(),
    }


def _process_identity(process: Any) -> JsonDict:
    pid = int(process.pid)
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        executable = ""
    return {
        "pid": pid,
        "pid_start_ticks": lease_api.proc_start_ticks(pid),
        "parent_pid": os.getpid(),
        "executable": executable,
        "exit_code": process.poll(),
        "absent_after_exit": process.poll() is not None,
    }


def _empty_model_process() -> JsonDict:
    return {
        "pid": 0,
        "pid_start_ticks": None,
        "parent_pid": os.getpid(),
        "executable": "",
        "exit_code": None,
        "absent_after_exit": True,
    }


def _wait_for_vram_recovery(
    device_uuid: str,
    owned_pid: int,
    before_used_mb: int,
    *,
    timeout_s: float = VRAM_RECOVERY_TIMEOUT_S,
) -> tuple[JsonDict, JsonDict]:
    deadline = time.monotonic() + timeout_s
    snapshot = _gpu_snapshot(device_uuid, owned_pid)
    receipt = build_vram_recovery_receipt(
        before_used_mb=before_used_mb,
        after_used_mb=int(snapshot.get("memory_used_mb", 0) or 0),
        owned_pid_present=bool(snapshot.get("owned_pid_present")),
    )
    while not receipt["passed"] and time.monotonic() < deadline:
        time.sleep(1.0)
        snapshot = _gpu_snapshot(device_uuid, owned_pid)
        receipt = build_vram_recovery_receipt(
            before_used_mb=before_used_mb,
            after_used_mb=int(snapshot.get("memory_used_mb", 0) or 0),
            owned_pid_present=bool(snapshot.get("owned_pid_present")),
        )
    receipt["observed_monotonic_ns"] = snapshot.get("observed_monotonic_ns")
    return receipt, snapshot


def run_live_model_worker(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    *,
    port: int,
    lease_runtime_dir: Path = LEASE_RUNTIME_DIR,
) -> JsonDict:
    """Own one lease, load one model, run selfparse, and return all resources."""

    from carnot.agentic import arc_induction_tool_loop as loop
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    started_ns = time.monotonic_ns()
    worker_identity = lease_api.current_process_identity()
    device_uuid = str(selected_device["uuid"])
    before = _gpu_snapshot(device_uuid)
    fresh_selection = rank_eligible_devices(nvidia_smi_inventory()["devices"])
    errors: list[str] = []
    lease: lease_api.GpuLease | None = None
    owner: JsonDict = {}
    release: JsonDict = {}
    journal: JsonDict = {}
    proposer: Any = None
    model_process = _empty_model_process()
    cleanup: JsonDict = {
        "pid": 0,
        "terminate_sent": False,
        "kill_sent": False,
        "exit_code": None,
        "absent_after_exit": True,
        "unrelated_processes_signaled": [],
    }
    selfparse = build_production_selfparse_receipt([], blocks_seen=0, blocks_unparsed=0)
    context: int | None = None
    observed_model_path: str | None = None
    layers: JsonDict = {"requested": 999, "offloaded": 0, "total": None}
    resident = deepcopy(before)
    peak_vram = 0
    resident_vram = 0
    vram_recovery = build_vram_recovery_receipt(
        before_used_mb=int(before.get("memory_used_mb", 0) or 0),
        after_used_mb=int(before.get("memory_used_mb", 0) or 0),
        owned_pid_present=False,
    )
    stop_monitor = threading.Event()
    monitor_thread: threading.Thread | None = None
    server = _llama_cpp_receipt()
    live_model_invoked = False
    try:
        selected_now = fresh_selection.get("selected_device")
        if not isinstance(selected_now, Mapping) or selected_now.get("uuid") != device_uuid:
            raise RuntimeError("selected_device_no_longer_first_eligible")
        if not port_is_free(port):
            raise RuntimeError("selected_port_no_longer_free")
        lease = acquire_selected_lease(
            runtime_dir=lease_runtime_dir,
            task_id=f"exp6764-{model['model_id'].split('/')[-1]}",
            selected_device=before,
            expected_model=str(model["model_path"]),
        )
        owner = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        proposer = LocalGGUFProposer(
            repo_substr=str(model["repo_substr"]),
            model_path=str(model["model_path"]),
            n_ctx=CONTEXT_REQUESTED,
            max_tokens=int(model["max_tokens"]),
            timeout=int(WORKER_TIMEOUT_S),
            port=int(port),
            mtp=False,
            n_gpu_layers=999,
            use_chat_template=True,
            extra_server_args=("-v",),
        )
        if not proposer._ensure_server():
            raise RuntimeError("llama_server_load_failed")
        if proposer.port != int(port):
            raise RuntimeError("llama_server_changed_frozen_port")
        process = proposer._proc
        if process is None:
            raise RuntimeError("llama_server_process_missing")
        model_process = _process_identity(process)
        resident = _gpu_snapshot(device_uuid, int(process.pid))
        resident_vram = int(resident.get("owned_pid_vram_mb", 0) or 0)
        peak_vram = resident_vram
        context = proposer.observed_n_ctx()
        observed_model_path = proposer.observed_model_path()
        log_path = getattr(proposer, "_stderr_log_path", None)
        log_text = Path(log_path).read_text(errors="replace") if log_path else ""
        layers = exp6752._gpu_layers_from_log(log_text, 999)
        cuda_resident = (
            server.get("cuda_linked") is True
            and resident.get("owned_pid_present") is True
            and resident_vram > 0
            and layers.get("offloaded", 0) > 0
        )
        if not cuda_resident:
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))

        def monitor() -> None:
            nonlocal peak_vram
            while not stop_monitor.wait(0.1):
                sample = _gpu_snapshot(device_uuid, int(process.pid))
                peak_vram = max(peak_vram, int(sample.get("owned_pid_vram_mb", 0) or 0))

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        lease.transition("inferencing")
        events: list[JsonDict] = []
        loop.induce_with_tool_loop(
            proposer,
            "transport_fixture",
            exp6752.fixture_transitions(),
            1,
            extra_user_instruction=exp6752.build_probe_instruction(),
            tool_event_sink=events,
        )
        stats = getattr(proposer, "last_tool_loop_stats", {})
        selfparse = build_production_selfparse_receipt(
            events,
            blocks_seen=int(stats.get("selfparse_blocks_seen", 0) or 0),
            blocks_unparsed=int(stats.get("selfparse_blocks_unparsed", 0) or 0),
        )
        live_model_invoked = bool(selfparse.get("raw_emission", "").strip())
    except Exception as exc:  # noqa: BLE001 - live failures belong in the receipt
        errors.append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if proposer is not None and proposer._proc is not None:
            process = proposer._proc
            cleanup = terminate_owned_process(process)
            proposer._proc = None
            model_process.update(
                {
                    "exit_code": cleanup.get("exit_code"),
                    "absent_after_exit": cleanup.get("absent_after_exit"),
                }
            )
        owned_pid = int(model_process.get("pid", 0) or 0)
        vram_recovery, after = _wait_for_vram_recovery(
            device_uuid,
            owned_pid,
            int(before.get("memory_used_mb", 0) or 0),
        )
        if lease is not None:
            try:
                phase = lease.document.get("phase")
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("memory_used_mb", 0) or 0),
                        exit_code=int(model_process.get("exit_code", 0) or 0),
                        unload_observed=model_process.get("absent_after_exit") is True,
                    )
                    complete = bool(
                        not errors
                        and context == CONTEXT_REQUESTED
                        and observed_model_path == str(model["model_path"])
                        and layers.get("offloaded", 0) == layers.get("total")
                        and live_model_invoked
                        and not production_selfparse_errors(selfparse)
                        and vram_recovery.get("passed") is True
                    )
                    lease.transition("terminal_complete" if complete else "terminal_blocked")
                elif phase in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
                    release = lease.release()
                else:
                    lease.close()
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                lease.close()
            try:
                journal = lease_api.read_journal(lease.journal_path)
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    receipt: JsonDict = {
        "model_id": model["model_id"],
        "role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "observed_model_path": observed_model_path,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llama_cpp_cuda": server.get("cuda_linked") is True,
        "server_path": server.get("path"),
        "server_sha256": server.get("sha256"),
        "device": deepcopy(dict(selected_device)),
        "worker_process": {
            "pid": worker_identity["pid"],
            "pid_start_ticks": worker_identity["pid_start_ticks"],
            "exit_code": 0,
            "absent_after_exit": False,
        },
        "model_process": model_process,
        "lease_owner": owner,
        "phase_history": deepcopy(journal.get("phase_history", [])),
        "lease_release": release,
        "runtime_context": context,
        "gpu_layers": layers,
        "peak_owned_vram_mb": peak_vram,
        "resident_owned_vram_mb": resident_vram,
        "duration_s": round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6),
        "live_model_invoked": live_model_invoked,
        "first_token_observed": live_model_invoked,
        "production_selfparse": selfparse,
        "vram_recovery": vram_recovery,
        "full_load": model["model_id"] == MODEL_SPECS[0]["model_id"],
        "transport_canary": model["model_id"] == MODEL_SPECS[1]["model_id"],
        "owned_cleanup": cleanup,
        "unrelated_processes_signaled": [],
        "errors": errors,
    }
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def _blocked_worker_receipt(
    model: Mapping[str, Any], selected_device: Mapping[str, Any], error: str
) -> JsonDict:
    """Keep a failed worker attempt without inventing lifecycle evidence."""

    receipt: JsonDict = {
        "model_id": model["model_id"],
        "role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "observed_model_path": None,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llama_cpp_cuda": False,
        "server_path": "",
        "server_sha256": "missing",
        "device": deepcopy(dict(selected_device)),
        "worker_process": {
            "pid": 0,
            "pid_start_ticks": None,
            "exit_code": 127,
            "absent_after_exit": True,
        },
        "model_process": _empty_model_process(),
        "lease_owner": {},
        "phase_history": [],
        "lease_release": {},
        "runtime_context": None,
        "gpu_layers": {"requested": 999, "offloaded": 0, "total": None},
        "peak_owned_vram_mb": 0,
        "resident_owned_vram_mb": 0,
        "duration_s": 0.0,
        "live_model_invoked": False,
        "first_token_observed": False,
        "production_selfparse": build_production_selfparse_receipt(
            [], blocks_seen=0, blocks_unparsed=0
        ),
        "vram_recovery": build_vram_recovery_receipt(
            before_used_mb=0, after_used_mb=0, owned_pid_present=False
        ),
        "full_load": False,
        "transport_canary": False,
        "owned_cleanup": {},
        "unrelated_processes_signaled": [],
        "errors": [error],
    }
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def _terminate_worker_group(process: Any, start_ticks: int | None) -> JsonDict:
    """Stop only the new session created for one task-owned worker."""

    receipt = {
        "worker_pid": int(process.pid),
        "worker_pid_start_ticks": start_ticks,
        "term_sent": False,
        "kill_sent": False,
        "unrelated_processes_signaled": [],
    }
    if lease_api.proc_start_ticks(int(process.pid)) != start_ticks:
        receipt["identity_mismatch"] = True
        return receipt
    try:
        os.killpg(int(process.pid), signal.SIGTERM)
        receipt["term_sent"] = True
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(int(process.pid), signal.SIGKILL)
        receipt["kill_sent"] = True
        process.wait(timeout=5.0)
    except ProcessLookupError:
        pass
    return receipt


def run_model_worker(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    port: int,
    runtime_dir: Path,
    *,
    timeout_s: float = WORKER_TIMEOUT_S,
) -> JsonDict:
    """Launch one fresh leased worker and wait for its stable receipt."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(model["model_id"])).strip("-").lower()
    model_path = runtime_dir / f"{slug}.model.json"
    device_path = runtime_dir / f"{slug}.device.json"
    output_path = runtime_dir / f"{slug}.receipt.json"
    write_json_atomic(model_path, model)
    write_json_atomic(device_path, selected_device)
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6764_arc_exclusive_load_preflight",
        "--worker",
        "--worker-model",
        str(model_path),
        "--worker-device",
        str(device_path),
        "--worker-output",
        str(output_path),
        "--port",
        str(port),
        "--lease-runtime-dir",
        str(LEASE_RUNTIME_DIR),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=worker_environment(os.environ, model, selected_device, port=port),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    start_ticks = lease_api.proc_start_ticks(process.pid)
    timeout_cleanup: JsonDict = {}
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timeout_cleanup = _terminate_worker_group(process, start_ticks)
        stdout, stderr = process.communicate()
    row = _load_json(output_path)
    if not row:
        row = _blocked_worker_receipt(
            model,
            selected_device,
            f"worker_output_missing:exit={process.returncode}:stderr_sha256={sha256_text(stderr)}",
        )
    worker = row.get("worker_process")
    worker = dict(worker) if isinstance(worker, Mapping) else {}
    worker.update(
        {
            "pid": int(process.pid),
            "pid_start_ticks": start_ticks,
            "exit_code": process.returncode,
            "absent_after_exit": process.poll() is not None,
            "stdout_sha256": sha256_text(stdout),
            "stderr_sha256": sha256_text(stderr),
            "timeout_cleanup": timeout_cleanup,
        }
    )
    row["worker_process"] = worker
    row["receipt_sha256"] = gpu_receipt_checksum(row)
    return row


def _derived_gate_summary(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for spec in MODEL_SPECS:
        receipt = next((row for row in receipts if row.get("model_id") == spec["model_id"]), {})
        failures = gpu_receipt_errors(receipt) if receipt else ["receipt_missing"]
        rows.append(
            {
                "check": f"model_admission:{spec['model_id']}",
                "expected": "complete owned CUDA load, selfparse, teardown, release, recovery",
                "observed": {"failures": failures},
                "passed": not failures,
            }
        )
    rows.append(
        {
            "check": "unrelated_processes_signaled",
            "expected": [],
            "observed": [
                pid
                for receipt in receipts
                for pid in receipt.get("unrelated_processes_signaled", [])
            ],
            "passed": all(
                receipt.get("unrelated_processes_signaled") == [] for receipt in receipts
            ),
        }
    )
    return rows


def build_artifact(
    *,
    date: str,
    preflight: Mapping[str, Any],
    gpu_receipts: Sequence[Mapping[str, Any]],
    started_ns: int,
    finished_ns: int,
) -> JsonDict:
    """Reduce preconditions and separate model receipts into one admission result."""

    receipts = [deepcopy(dict(row)) for row in gpu_receipts]
    models = [deepcopy(dict(row)) for row in preflight.get("models", [])]
    selected = deepcopy(preflight.get("device_selection_receipt", {}))
    ready = bool(preflight.get("all_passed") is True and reduce_arc_exclusive_load_ready(receipts))
    if ready:
        verdict_class = "positive"
        honest_verdict = "complete_arc_exclusive_load_ready"
        status = "complete"
    elif preflight.get("all_passed") is not True:
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_arc_exclusive_load"
        status = "blocked"
    else:
        verdict_class = "partial"
        honest_verdict = "complete_partial_arc_exclusive_load"
        status = "partial"
    phase_rows = [row for receipt in receipts for row in phase_rows_for_receipt(receipt)]
    process_receipts = [
        deepcopy(receipt.get(name))
        for receipt in receipts
        for name in ("worker_process", "model_process")
    ]
    unrelated = [
        pid for receipt in receipts for pid in receipt.get("unrelated_processes_signaled", [])
    ]
    gate_summary = (
        deepcopy([row for row in preflight.get("checks", []) if row.get("passed") is not True])
        if preflight.get("all_passed") is not True
        else _derived_gate_summary(receipts)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6764,
        "title": "Exclusive ARC 32K CUDA load and selfparse admission preflight",
        "run_date": str(date),
        "status": status,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, finished_ns - started_ns) / 1_000_000_000, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "models_used": models,
        "model_specs": deepcopy(models),
        "live_model_invoked": bool(receipts)
        and len(receipts) == len(MODEL_SPECS)
        and all(receipt.get("live_model_invoked") is True for receipt in receipts),
        "rows": phase_rows,
        "device_inventory_before": deepcopy(list(preflight.get("device_inventory_before", []))),
        "device_selection_receipt": selected,
        "lease_owner_receipts": [deepcopy(row.get("lease_owner")) for row in receipts],
        "phase_rows": phase_rows,
        "gpu_receipts": receipts,
        "runtime_context_by_model": {
            str(row.get("model_id")): row.get("runtime_context") for row in receipts
        },
        "production_selfparse_receipt": {
            str(row.get("model_id")): deepcopy(row.get("production_selfparse")) for row in receipts
        },
        "owned_processes_terminated": all(
            isinstance(row, Mapping) and row.get("absent_after_exit") is True
            for row in process_receipts
        ),
        "lease_release_receipts": [deepcopy(row.get("lease_release")) for row in receipts],
        "vram_recovery_receipts": [deepcopy(row.get("vram_recovery")) for row in receipts],
        "unrelated_processes_signaled": unrelated,
        "arc_exclusive_load_ready": ready,
        "claim_boundary": CLAIM_BOUNDARY,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "preconditions_checked": deepcopy(dict(preflight)),
        "owned_process_receipts": process_receipts,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the admission result from retained evidence."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    principles = artifact.get("field_principles")
    principles = principles if isinstance(principles, Mapping) else {}
    if not set(artifact).issubset(principles):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if artifact.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("claim_boundary")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or artifact.get("duration_s", -1) < 0
    ):
        errors.append("duration_s")
    models = artifact.get("models_used")
    models = models if isinstance(models, list) else []
    model_ids_match = [row.get("model_id") for row in models] == [
        spec["model_id"] for spec in MODEL_SPECS
    ]
    exact_hashes_match = model_ids_match and all(
        row.get("model_sha256") == spec["expected_sha256"]
        for row, spec in zip(models, MODEL_SPECS, strict=True)
    )
    preconditions_passed = artifact.get("preconditions_checked", {}).get("all_passed") is True
    if not model_ids_match or (preconditions_passed and not exact_hashes_match):
        errors.append("models_used")
    if artifact.get("model_specs") != models:
        errors.append("model_specs")
    inventory = artifact.get("device_inventory_before")
    inventory = inventory if isinstance(inventory, list) else []
    expected_selection = rank_eligible_devices(inventory)
    if artifact.get("device_selection_receipt") != expected_selection:
        errors.append("device_selection_receipt")
    receipts = artifact.get("gpu_receipts")
    receipts = receipts if isinstance(receipts, list) else []
    expected_ready = bool(
        artifact.get("preconditions_checked", {}).get("all_passed") is True
        and reduce_arc_exclusive_load_ready(receipts)
    )
    if artifact.get("arc_exclusive_load_ready") is not expected_ready:
        errors.append("arc_exclusive_load_ready")
    if artifact.get("rows") != [
        row for receipt in receipts for row in phase_rows_for_receipt(receipt)
    ]:
        errors.append("rows")
    if artifact.get("phase_rows") != artifact.get("rows"):
        errors.append("phase_rows")
    if artifact.get("lease_owner_receipts") != [row.get("lease_owner") for row in receipts]:
        errors.append("lease_owner_receipts")
    if artifact.get("lease_release_receipts") != [row.get("lease_release") for row in receipts]:
        errors.append("lease_release_receipts")
    if artifact.get("vram_recovery_receipts") != [row.get("vram_recovery") for row in receipts]:
        errors.append("vram_recovery_receipts")
    if artifact.get("runtime_context_by_model") != {
        str(row.get("model_id")): row.get("runtime_context") for row in receipts
    }:
        errors.append("runtime_context_by_model")
    if artifact.get("production_selfparse_receipt") != {
        str(row.get("model_id")): row.get("production_selfparse") for row in receipts
    }:
        errors.append("production_selfparse_receipt")
    process_receipts = [
        row.get(name) for row in receipts for name in ("worker_process", "model_process")
    ]
    expected_terminated = all(
        isinstance(row, Mapping) and row.get("absent_after_exit") is True
        for row in process_receipts
    )
    if artifact.get("owned_processes_terminated") is not expected_terminated:
        errors.append("owned_processes_terminated")
    unrelated = [pid for row in receipts for pid in row.get("unrelated_processes_signaled", [])]
    if artifact.get("unrelated_processes_signaled") != unrelated or unrelated:
        errors.append("unrelated_processes_signaled")
    expected_live = (
        bool(receipts)
        and len(receipts) == len(MODEL_SPECS)
        and all(row.get("live_model_invoked") is True for row in receipts)
    )
    if artifact.get("live_model_invoked") is not expected_live:
        errors.append("live_model_invoked")
    expected_gates = (
        _derived_gate_summary(receipts)
        if artifact.get("preconditions_checked", {}).get("all_passed") is True
        else [
            row
            for row in artifact.get("preconditions_checked", {}).get("checks", [])
            if row.get("passed") is not True
        ]
    )
    if artifact.get("gate_check_summary") != expected_gates:
        errors.append("gate_check_summary")
    expected_verdict = (
        "complete_arc_exclusive_load_ready"
        if expected_ready
        else (
            "complete_blocked_arc_exclusive_load"
            if artifact.get("preconditions_checked", {}).get("all_passed") is not True
            else "complete_partial_arc_exclusive_load"
        )
    )
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict")
    selected = artifact.get("device_selection_receipt", {}).get("selected_device")
    selected_uuid = selected.get("uuid") if isinstance(selected, Mapping) else None
    if any((row.get("device") or {}).get("uuid") != selected_uuid for row in receipts):
        errors.append("selected_device_binding")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON with atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run(
    *,
    result_path: Path = RESULT_PATH,
    date: str = RUN_DATE,
    preflight_fn: Callable[[], JsonDict] = collect_preconditions,
    worker_runner: Callable[
        [Mapping[str, Any], Mapping[str, Any], int, Path], JsonDict
    ] = run_model_worker,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run all gates, then two fresh workers in fixed sequence when admitted."""

    started_ns = clock()
    preflight = preflight_fn()
    receipts: list[JsonDict] = []
    if preflight.get("all_passed") is True:
        selected = preflight.get("device_selection_receipt", {}).get("selected_device")
        models = preflight.get("models", [])
        ports = preflight.get("ports", [45_000, 45_001])
        runtime_dir = result_path.parent / ".experiment_6764_arc_exclusive_load_preflight"
        if isinstance(selected, Mapping):
            for model, port in zip(models, ports, strict=True):
                receipt = worker_runner(model, selected, int(port), runtime_dir)
                receipts.append(receipt)
                if gpu_receipt_errors(receipt):
                    break
    artifact = build_artifact(
        date=date,
        preflight=preflight,
        gpu_receipts=receipts,
        started_ns=started_ns,
        finished_ns=clock(),
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6764 artifact:" + ",".join(errors))
    write_json_atomic(result_path, artifact)
    return artifact


def _worker_entry(
    model_path: Path,
    device_path: Path,
    output_path: Path,
    port: int,
    lease_runtime_dir: Path,
) -> int:
    model = _load_json(model_path)
    device = _load_json(device_path)
    receipt = run_live_model_worker(model, device, port=port, lease_runtime_dir=lease_runtime_dir)
    write_json_atomic(output_path, receipt)
    return 0 if not receipt.get("errors") else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent experiment or one explicitly requested worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--worker-device", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--lease-runtime-dir", type=Path, default=LEASE_RUNTIME_DIR)
    args = parser.parse_args(argv)
    if args.worker:
        if not all(
            value is not None
            for value in (args.worker_model, args.worker_device, args.worker_output, args.port)
        ):
            parser.error("--worker requires model, device, output, and port")
        return _worker_entry(
            args.worker_model,
            args.worker_device,
            args.worker_output,
            args.port,
            args.lease_runtime_dir,
        )
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "ready": artifact["arc_exclusive_load_ready"],
                "verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - covered by the repository wrapper
    raise SystemExit(main())
