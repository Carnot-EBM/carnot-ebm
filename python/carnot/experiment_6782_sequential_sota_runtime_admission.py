"""Admit all three mandated SOTA GGUF families in owned CUDA windows.

This contract proves execution capability only. It does not measure model
quality, memory, proof accuracy, or ARC progress. Each model gets a fresh
worker and an owner-bound lease. The worker closes its own backend before the
next model can start.

Spec refs: REQ-INFER-SOTA-6782, SCENARIO-INFER-SOTA-6782-*,
REQ-REPORT-6782, and SCENARIO-REPORT-6782-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6782_sequential_sota_runtime_admission.json"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
MODULE_PATH = Path(__file__).resolve()
SCRIPT_PATH = REPO_ROOT / "scripts/experiments/experiment_6782_sequential_sota_runtime_admission.py"
TEST_PATH = REPO_ROOT / "tests/python/test_experiment_6782_sequential_sota_runtime_admission.py"

SCHEMA = "carnot.experiment_6782.sequential_sota_runtime_admission.v1"
EXPERIMENT_ID = "experiment_6782_sequential_sota_runtime_admission"
RUN_DATE = "20260830"
RANDOM_SEED = 6_782
INFERENCE_SUBSTRATE = "local llama.cpp CUDA with receipt-scoped RTX lease"
EXPECTED_GPU_UUIDS = (
    "GPU-b52387a2-c625-de87-8d34-e6f64e684bab",
    "GPU-7971baff-9583-eaa6-2292-393f930a28f9",
)
CANARY_CONTEXT_TOKENS = 512
CANARY_MAX_OUTPUT_TOKENS = 1
CANARY_PROMPT = "Reply with exactly one word that names the color of a clear daytime sky.\nAnswer:"
LEASE_WAIT_TIMEOUT_S = 20 * 60.0
LEASE_POLL_INTERVAL_S = 5.0
WORKER_TIMEOUT_S = 10 * 60.0
VRAM_RECOVERY_TIMEOUT_S = 180.0
VRAM_RECOVERY_TOLERANCE_MB = 512
FREE_VRAM_FLOOR_MB = 22_610
RAM_AVAILABLE_FLOOR_BYTES = 64 * 1024**3
DISK_FREE_FLOOR_BYTES = 1024**3
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE
TRAINING_ENTRYPOINT_MARKERS = ("train.py", "/nn/train", "src/nn/train")
SERVER_ENTRYPOINT_MARKERS = (
    "llama-server",
    "vllm.entrypoints.openai.api_server",
    "vllm serve",
)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "family_id": "qwen36",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_admission",
        "quantization": "Q4_K_M",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "expected_sha256": "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    },
    {
        "family_id": "gemma31",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_admission",
        "quantization": "Q4_K_M",
        "filename": "gemma-4-31B-it-Q4_K_M.gguf",
        "expected_sha256": "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1",
    },
    {
        "family_id": "gemma26",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_admission",
        "quantization": "Q4_K_M",
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "expected_sha256": "sha256:34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35",
    },
)
CANARY_PROMPT_SHA256 = "sha256:" + hashlib.sha256(CANARY_PROMPT.encode()).hexdigest()
MODEL_RECORD_FIELDS = set(MODEL_SPECS[0]) | {
    "revision",
    "model_path",
    "model_sha256",
    "model_size_bytes",
    "tokenizer",
    "context_tokens",
    "max_output_tokens",
}
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "MODEL_SPECS",
    "model_specs",
    "models_used",
    "live_model_invoked",
    "rows",
    "gpu_receipts",
    "protected_process_actions",
    "qwen36_runtime_ready",
    "gemma31_runtime_ready",
    "gemma26_runtime_ready",
    "all_mandated_runtime_ready",
    "preconditions_checked",
    "code_receipts",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned shape lets cold readers reject incompatible evidence.",
    "experiment_id": "The stable identifier binds the artifact to its producer.",
    "run_date": "The fixed execution date prevents silent protocol drift.",
    "status": "The status separates complete, partial, and blocked execution.",
    "field_principles": "Every required field states its purpose.",
    "inference_substrate": "The value excludes CPU, remote, and unowned inference.",
    "duration_s": "Monotonic task wall time records the complete execution window.",
    "random_seed": "The fixed seed makes the canary requests repeatable.",
    "reproducibility_checksum": "The hash binds models, code, rows, and GPU receipts.",
    "MODEL_SPECS": "The frozen hub IDs and roles prevent family substitution.",
    "model_specs": "Resolved paths, hashes, tokenizers, contexts, and limits bind exact files.",
    "models_used": "Only models with valid real-token receipts appear as used.",
    "live_model_invoked": "True means at least one real model emitted a token.",
    "rows": "Poll, load, inference, teardown, and recovery rows preserve partial progress.",
    "gpu_receipts": "Model-local ownership, offload, VRAM, token, and release evidence stays separate.",
    "protected_process_actions": "An empty list proves protected work was not changed.",
    "qwen36_runtime_ready": "Exp6787 consumes this exact model-local readiness field.",
    "gemma31_runtime_ready": "This field reports dense flagship readiness only.",
    "gemma26_runtime_ready": "This field reports middle MoE readiness only.",
    "all_mandated_runtime_ready": "Exp6783 consumes this conjunction of all three models.",
    "preconditions_checked": "Observed host gates explain whether live work could start.",
    "code_receipts": "Source hashes identify the producer, wrapper, and tests.",
    "gate_check_summary": "A blocked result retains the failed check and observed value.",
    "verifier_is_oracle": "False states that admission is not a correctness oracle.",
    "verdict_class": "A closed enum prevents execution evidence from becoming a science claim.",
    "honest_verdict": "A terminal prefix makes the final state machine-readable.",
}


def canonical_json(value: Any) -> str:
    """Return stable JSON for hashes and receipt comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes and retain the algorithm name in the receipt."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a large model in chunks without copying it into memory."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gpu_receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Hash one receipt without its self-referential checksum field."""

    return sha256_bytes(
        canonical_json(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        ).encode()
    )


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the frozen manifest, resolved files, code, rows, and receipts."""

    payload = {
        "MODEL_SPECS": artifact.get("MODEL_SPECS"),
        "model_specs": artifact.get("model_specs"),
        "code_receipts": artifact.get("code_receipts"),
        "rows": artifact.get("rows"),
        "gpu_receipts": artifact.get("gpu_receipts"),
    }
    return sha256_bytes(canonical_json(payload).encode())


def _revision_from_path(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-unversioned"


def _missing_model(planned: Mapping[str, Any]) -> JsonDict:
    return {
        **deepcopy(dict(planned)),
        "revision": "missing",
        "model_path": "",
        "model_sha256": "missing",
        "model_size_bytes": 0,
        "tokenizer": {
            "source": "llama.cpp_embedded_gguf",
            "loadable": False,
            "detail": "model path unresolved",
        },
        "context_tokens": CANARY_CONTEXT_TOKENS,
        "max_output_tokens": CANARY_MAX_OUTPUT_TOKENS,
    }


def resolve_model_specs(
    *,
    pair_resolver: Callable[..., list[dict] | None] = cached_sota_pair,
    single_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    tokenizer_probe: Callable[[str | None], tuple[bool, str]] = gguf_tokenizer_loadable,
    file_hasher: Callable[[str | Path], str] = sha256_file,
) -> list[JsonDict]:
    """Resolve the frozen three-model order through the shared local cache helpers."""

    pair = pair_resolver(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    paths = {
        str(row.get("hf_id")): str(row.get("model_path", ""))
        for row in pair
        if isinstance(row, Mapping)
    }
    middle_id = str(MODEL_SPECS[2]["hf_id"])
    paths[middle_id] = str(single_resolver(middle_id, "Q4_K_M") or "")
    rows: list[JsonDict] = []
    for planned in MODEL_SPECS:
        path = Path(paths.get(str(planned["hf_id"]), ""))
        if not path.is_file() or path.name != planned["filename"]:
            rows.append(_missing_model(planned))
            continue
        model_hash = file_hasher(path)
        tokenizer_ok, tokenizer_detail = tokenizer_probe(str(path))
        rows.append(
            {
                **deepcopy(planned),
                "revision": _revision_from_path(path),
                "model_path": str(path.absolute()),
                "model_sha256": model_hash,
                "model_size_bytes": path.stat().st_size,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": bool(tokenizer_ok),
                    "detail": str(tokenizer_detail),
                },
                "context_tokens": CANARY_CONTEXT_TOKENS,
                "max_output_tokens": CANARY_MAX_OUTPUT_TOKENS,
            }
        )
    return rows


def model_record_errors(record: Mapping[str, Any], planned: Mapping[str, Any]) -> list[str]:
    """Reject any missing or substituted model identity and runtime input."""

    errors: list[str] = []
    if set(record) != MODEL_RECORD_FIELDS:
        errors.append("field_set")
    for field, expected in planned.items():
        if record.get(field) != expected:
            errors.append(field)
    if record.get("model_sha256") != planned.get("expected_sha256"):
        errors.append("model_sha256")
    path = Path(str(record.get("model_path", "")))
    if not str(record.get("revision", "")) or not path.name == planned.get("filename"):
        errors.append("path_or_revision")
    if (
        not isinstance(record.get("model_size_bytes"), int)
        or record.get("model_size_bytes", 0) <= 0
    ):
        errors.append("model_size_bytes")
    tokenizer = record.get("tokenizer") if isinstance(record.get("tokenizer"), Mapping) else {}
    if (
        tokenizer.get("source") != "llama.cpp_embedded_gguf"
        or tokenizer.get("loadable") is not True
    ):
        errors.append("tokenizer")
    if record.get("context_tokens") != CANARY_CONTEXT_TOKENS:
        errors.append("context_tokens")
    if record.get("max_output_tokens") != CANARY_MAX_OUTPUT_TOKENS:
        errors.append("max_output_tokens")
    return list(dict.fromkeys(errors))


def check_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one complete precondition row with its expected value."""

    return {
        "check": check,
        "expected": expected,
        "observed": deepcopy(observed),
        "passed": bool(passed),
    }


def _run_command(  # pragma: no cover - exercised by the required live E2E run
    command: Sequence[str], timeout_s: float = 30.0
) -> JsonDict:
    """Run one read-only host probe and retain bounded output."""

    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "stdout": result.stdout[-50_000:],
            "stderr": result.stderr[-20_000:],
            "duration_s": round(time.monotonic() - started, 6),
        }
    except Exception as exc:  # noqa: BLE001 - preflight must retain probe failures
        return {
            "command": list(command),
            "exit_code": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}"[:20_000],
            "duration_s": round(time.monotonic() - started, 6),
        }


def nvidia_smi_inventory() -> JsonDict:  # pragma: no cover - live NVIDIA boundary
    """Read fixed device identities and all current compute processes."""

    device_query = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    process_query = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    processes: list[JsonDict] = []
    for line in process_query["stdout"].splitlines():
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
    for line in device_query["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            uuid = parts[1]
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
                    "active_compute_processes": [
                        row for row in processes if row["gpu_uuid"] == uuid
                    ],
                }
            )
        except ValueError:
            continue
    return {"device_query": device_query, "process_query": process_query, "devices": devices}


def llama_cpp_receipt() -> JsonDict:  # pragma: no cover - live llama.cpp boundary
    """Prove both the server binary and Python binding support CUDA offload."""

    configured = os.environ.get("CARNOT_LLAMA_SERVER", "")
    candidates = [
        Path(configured) if configured else Path("/__not_configured__"),
        Path.home() / ".cache/llama.cpp-master/build/bin/llama-server",
    ]
    server = next((path for path in candidates if path.is_file()), candidates[-1])
    linked = _run_command(("ldd", str(server))) if server.is_file() else {}
    linked_text = str(linked.get("stdout", "")) + str(linked.get("stderr", ""))
    try:
        from llama_cpp import llama_cpp

        python_cuda: Any = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - import failure is a gate value
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


def host_resources(root: Path) -> JsonDict:  # pragma: no cover - live host boundary
    """Read available RAM and disk without allocating either resource."""

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


def choose_free_ports(count: int) -> list[int]:  # pragma: no cover - live socket boundary
    """Ask the kernel for distinct loopback ports, then release them."""

    ports: list[int] = []
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            sockets.append(probe)
            ports.append(int(probe.getsockname()[1]))
    finally:
        for probe in sockets:
            probe.close()
    return ports


def port_is_free(port: int) -> bool:  # pragma: no cover - live socket boundary
    """Return true only when the requested loopback port can bind now."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def protected_process_kind(pid: int) -> str | None:  # pragma: no cover - live /proc boundary
    """Classify only the training and serving categories protected by the template."""

    try:
        cmdline = (
            Path(f"/proc/{int(pid)}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    if any(marker in cmdline for marker in TRAINING_ENTRYPOINT_MARKERS):
        return "training"
    if any(marker in cmdline for marker in SERVER_ENTRYPOINT_MARKERS):
        return "serving"
    return None


def rank_eligible_devices(
    devices: Sequence[Mapping[str, Any]],
    *,
    protected_classifier: Callable[[int], str | None] = protected_process_kind,
) -> JsonDict:
    """Select an idle fixed RTX 3090 without changing any unowned process."""

    evaluated: list[JsonDict] = []
    for source in devices:
        row = deepcopy(dict(source))
        active = row.get("active_compute_processes")
        active = active if isinstance(active, list) else []
        protected = []
        for process in active:
            if not isinstance(process, Mapping):
                continue
            pid = int(process.get("pid", 0) or 0)
            kind = protected_classifier(pid) if pid > 1 else None
            if kind:
                protected.append({**deepcopy(dict(process)), "kind": kind})
        reasons: list[str] = []
        if (
            row.get("uuid") not in EXPECTED_GPU_UUIDS
            or row.get("name") != "NVIDIA GeForce RTX 3090"
        ):
            reasons.append("unexpected_device_identity")
        if int(row.get("memory_free_mb", -1)) < FREE_VRAM_FLOOR_MB:
            reasons.append("free_vram_below_floor")
        if active:
            reasons.append("active_unowned_compute_process")
        row["protected_processes"] = protected
        row["ineligibility_reasons"] = reasons
        row["eligible"] = not reasons
        evaluated.append(row)
    eligible = [row for row in evaluated if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -int(row.get("memory_free_mb", 0)),
            int(row.get("temperature_c", 10_000)),
            int(row.get("index", 10_000)),
        )
    )
    return {
        "free_vram_floor_mb": FREE_VRAM_FLOOR_MB,
        "evaluated_devices": evaluated,
        "selected_device": deepcopy(eligible[0]) if eligible else None,
        "eligible_device_count": len(eligible),
        "protected_process_actions": [],
    }


def wait_for_eligible_device(
    *,
    deadline_ns: int,
    model_id: str = "",
    inventory_fn: Callable[[], JsonDict] = nvidia_smi_inventory,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval_s: float = LEASE_POLL_INTERVAL_S,
    protected_classifier: Callable[[int], str | None] = protected_process_kind,
) -> tuple[JsonDict | None, list[JsonDict]]:
    """Poll until one card is idle or the shared 20-minute deadline expires."""

    rows: list[JsonDict] = []
    while True:
        observed_ns = int(clock())
        inventory = inventory_fn()
        devices = inventory.get("devices", []) if isinstance(inventory, Mapping) else []
        selection = rank_eligible_devices(devices, protected_classifier=protected_classifier)
        selected = selection.get("selected_device")
        rows.append(
            {
                "row_kind": "lease_poll",
                "model_id": model_id,
                "poll_index": len(rows),
                "observed_monotonic_ns": observed_ns,
                "inventory": deepcopy(devices),
                "selection": selection,
                "passed": isinstance(selected, Mapping),
            }
        )
        if isinstance(selected, Mapping):
            return deepcopy(dict(selected)), rows
        if observed_ns >= int(deadline_ns):
            return None, rows
        remaining_s = max(0.0, (int(deadline_ns) - observed_ns) / 1_000_000_000)
        sleep_fn(min(float(poll_interval_s), remaining_s))


def code_receipts(  # pragma: no cover - live source hashing boundary
    paths: Sequence[Path] = (MODULE_PATH, SCRIPT_PATH, TEST_PATH),
) -> JsonDict:
    """Hash the implementation surfaces that define this admission contract."""

    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def collect_preconditions(
    *,
    date: str = RUN_DATE,
    model_resolver: Callable[[], list[JsonDict]] = resolve_model_specs,
    inventory_fn: Callable[[], JsonDict] = nvidia_smi_inventory,
    llama_receipt_fn: Callable[[], JsonDict] = llama_cpp_receipt,
    port_picker: Callable[[int], list[int]] = choose_free_ports,
    port_probe: Callable[[int], bool] = port_is_free,
    resource_fn: Callable[[Path], JsonDict] = host_resources,
) -> JsonDict:
    """Check identities, models, CUDA, leases, ports, RAM, and disk before waiting."""

    models = model_resolver()
    model_errors = [
        model_record_errors(row, planned) for row, planned in zip(models, MODEL_SPECS, strict=False)
    ]
    inventory = inventory_fn()
    devices = inventory.get("devices", []) if isinstance(inventory, Mapping) else []
    identities = {row.get("uuid"): row.get("name") for row in devices}
    expected_identities = {uuid: "NVIDIA GeForce RTX 3090" for uuid in EXPECTED_GPU_UUIDS}
    llama = llama_receipt_fn()
    ports = port_picker(len(MODEL_SPECS))
    port_status = {str(port): port_probe(port) for port in ports}
    resources = resource_fn(REPO_ROOT)
    checks = [
        check_row("planning_date_matches", RUN_DATE, date, date == RUN_DATE),
        check_row(
            "two_fixed_rtx3090_identities",
            expected_identities,
            identities,
            identities == expected_identities,
        ),
        check_row(
            "models_resolved",
            True,
            model_errors,
            len(models) == len(MODEL_SPECS) and all(not errors for errors in model_errors),
        ),
        check_row(
            "llama_cpp_cuda_offload",
            True,
            llama,
            llama.get("exists") is True
            and llama.get("executable") is True
            and llama.get("cuda_linked") is True
            and llama.get("python_cuda_offload") is True,
        ),
        check_row(
            "gpu_lease_api",
            True,
            {
                name: callable(getattr(lease_api.GpuLease, name, None))
                for name in ("acquire", "transition", "release")
            },
            all(
                callable(getattr(lease_api.GpuLease, name, None))
                for name in ("acquire", "transition", "release")
            ),
        ),
        check_row(
            "unused_task_ports",
            len(MODEL_SPECS),
            port_status,
            len(ports) == len(MODEL_SPECS) and all(port_status.values()),
        ),
        check_row(
            "ram_and_disk",
            {
                "ram_available_bytes": RAM_AVAILABLE_FLOOR_BYTES,
                "disk_free_bytes": DISK_FREE_FLOOR_BYTES,
            },
            resources,
            int(resources.get("ram_available_bytes", 0)) >= RAM_AVAILABLE_FLOOR_BYTES
            and int(resources.get("disk_free_bytes", 0)) >= DISK_FREE_FLOOR_BYTES,
        ),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "models": models,
        "ports": ports,
        "device_inventory_before": deepcopy(devices),
        "llama_cpp": llama,
        "resources": resources,
        "protected_process_actions": [],
    }


def _process_identity() -> JsonDict:  # pragma: no cover - live /proc boundary
    identity = lease_api.current_process_identity()
    return {
        **identity,
        "exit_code": None,
        "absent_after_exit": False,
    }


def _gpu_snapshot(  # pragma: no cover - live NVIDIA boundary
    device_uuid: str, owned_pid: int = 0
) -> JsonDict:
    inventory = nvidia_smi_inventory()
    device = next(
        (row for row in inventory.get("devices", []) if row.get("uuid") == device_uuid), {}
    )
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


def build_vram_recovery_receipt(
    before_used_mb: int, after_used_mb: int, owned_pid_present: bool
) -> JsonDict:
    """Require worker GPU absence and device use near the measured baseline."""

    delta = abs(int(after_used_mb) - int(before_used_mb))
    return {
        "before_used_mb": int(before_used_mb),
        "after_used_mb": int(after_used_mb),
        "absolute_delta_mb": delta,
        "tolerance_mb": VRAM_RECOVERY_TOLERANCE_MB,
        "owned_pid_present": bool(owned_pid_present),
        "passed": not owned_pid_present and delta <= VRAM_RECOVERY_TOLERANCE_MB,
    }


def _terminalize_lease(lease: Any, complete: bool, after: Mapping[str, Any]) -> JsonDict:
    phase = lease.document.get("phase")
    if phase in {"resident", "inferencing"}:
        lease.transition("unloading")
        phase = "unloading"
    if phase == "unloading":
        lease.transition(
            "validating",
            vram_mb=int(after.get("memory_used_mb", 0) or 0),
            exit_code=0 if complete else 1,
            unload_observed=True,
        )
        phase = "validating"
    if phase in {"preflight", "admitted", "loading"}:
        lease.transition("terminal_blocked")
    elif phase == "validating":
        lease.transition("terminal_complete" if complete else "terminal_blocked")
    if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
        return dict(lease.release())
    lease.close()
    return {}


def run_live_model_worker(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    *,
    prompt: str,
    lease_runtime_dir: Path = LEASE_RUNTIME_DIR,
    llama_factory: Callable[..., Any] | None = None,
    lease_factory: Callable[..., Any] = lease_api.GpuLease.acquire,
    snapshot_fn: Callable[[str, int], JsonDict] = _gpu_snapshot,
    process_identity_fn: Callable[[], JsonDict] = _process_identity,
    supports_gpu_offload_fn: Callable[[], bool] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    """Own one lease, emit one real token, close CUDA, and release the lease."""

    started_ns = time.monotonic_ns()
    device_uuid = str(selected_device["uuid"])
    worker = process_identity_fn()
    owned_pid = int(worker["pid"])
    before = snapshot_fn(device_uuid, owned_pid)
    unrelated = [
        deepcopy(row)
        for row in before.get("active_compute_processes", [])
        if row.get("pid") != owned_pid
    ]
    lease: Any = None
    llm: Any = None
    owner: JsonDict = {}
    release: JsonDict = {}
    errors: list[str] = []
    resident = deepcopy(before)
    peak_vram = 0
    token_text = ""
    completion_tokens = 0
    close_called = False
    close_error: str | None = None
    supports_offload = False
    after = deepcopy(before)
    recovery = build_vram_recovery_receipt(
        int(before.get("memory_used_mb", 0) or 0),
        int(before.get("memory_used_mb", 0) or 0),
        False,
    )
    try:
        if not (
            before.get("uuid") == device_uuid
            and device_uuid in EXPECTED_GPU_UUIDS
            and before.get("name") == "NVIDIA GeForce RTX 3090"
            and int(before.get("memory_free_mb", 0) or 0) >= FREE_VRAM_FLOOR_MB
            and not before.get("active_compute_processes")
        ):
            raise RuntimeError("selected_device_recheck_failed")
        lease = lease_factory(
            runtime_dir=lease_runtime_dir,
            task_id=f"exp6782-{model['family_id']}",
            device_uuid=device_uuid,
            expected_model=str(model["model_path"]),
            vram_before_mb=int(before.get("memory_used_mb", 0) or 0),
            ttl_s=WORKER_TIMEOUT_S,
        )
        owner = dict(lease.owner_receipt())
        lease.transition("admitted")
        lease.transition("loading")
        if supports_gpu_offload_fn is None:  # pragma: no cover - live binding import
            from llama_cpp import llama_cpp

            supports_gpu_offload_fn = llama_cpp.llama_supports_gpu_offload
        supports_offload = bool(supports_gpu_offload_fn())
        if not supports_offload:
            raise RuntimeError("llama_cpp_cuda_offload_unavailable")
        if llama_factory is None:  # pragma: no cover - live binding import
            from llama_cpp import Llama

            llama_factory = Llama
        llm = llama_factory(
            model_path=str(model["model_path"]),
            n_ctx=CANARY_CONTEXT_TOKENS,
            n_batch=512,
            n_gpu_layers=-1,
            main_gpu=0,
            seed=RANDOM_SEED,
            verbose=True,
        )
        resident = snapshot_fn(device_uuid, owned_pid)
        resident_vram = int(resident.get("owned_pid_vram_mb", 0) or 0)
        peak_vram = resident_vram
        if resident.get("owned_pid_present") is not True or resident_vram <= 0:
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
        lease.transition("inferencing")
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=CANARY_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            seed=RANDOM_SEED,
            stream=False,
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        token_text = str((choices[0] if choices else {}).get("text", ""))
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        sampled = snapshot_fn(device_uuid, owned_pid)
        peak_vram = max(peak_vram, int(sampled.get("owned_pid_vram_mb", 0) or 0))
        if completion_tokens < 1 or not token_text.encode("utf-8"):
            raise RuntimeError("first_token_not_observed")
    except Exception as exc:  # noqa: BLE001 - the receipt must retain live runtime failures
        errors.append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:  # pragma: no cover - live journal failure
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
        if llm is not None:
            close_called = True
            try:
                llm.close()
            except Exception as exc:  # pragma: no cover - live backend failure
                close_error = f"{type(exc).__name__}: {exc}"[:500]
                errors.append(close_error)
            llm = None
            gc.collect()
        deadline = time.monotonic() + VRAM_RECOVERY_TIMEOUT_S
        after = snapshot_fn(device_uuid, owned_pid)
        recovery = build_vram_recovery_receipt(
            int(before.get("memory_used_mb", 0) or 0),
            int(after.get("memory_used_mb", 0) or 0),
            bool(after.get("owned_pid_present")),
        )
        while not recovery["passed"] and time.monotonic() < deadline:  # pragma: no cover
            sleep_fn(1.0)
            after = snapshot_fn(device_uuid, owned_pid)
            recovery = build_vram_recovery_receipt(
                int(before.get("memory_used_mb", 0) or 0),
                int(after.get("memory_used_mb", 0) or 0),
                bool(after.get("owned_pid_present")),
            )
        non_fixture = bool(
            completion_tokens >= 1
            and token_text.encode("utf-8")
            and sha256_bytes(token_text.encode()) != CANARY_PROMPT_SHA256
        )
        complete = bool(
            not errors
            and supports_offload
            and peak_vram > 0
            and non_fixture
            and close_called
            and close_error is None
            and recovery["passed"]
        )
        if lease is not None:
            try:
                release = _terminalize_lease(lease, complete, after)
            except lease_api.LeaseError as exc:  # pragma: no cover - live journal failure
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
                lease.close()
    history = deepcopy(lease.document.get("phase_history", [])) if lease is not None else []
    receipt: JsonDict = {
        "model_id": model.get("hf_id"),
        "model_record": deepcopy(dict(model)),
        "device": deepcopy(dict(selected_device)),
        "unrelated_process_inventory": unrelated,
        "worker_process": worker,
        "lease_owner": owner,
        "phase_history": history,
        "lease_release": release,
        "cuda_offload": {
            "requested_gpu_layers": -1,
            "supports_gpu_offload": supports_offload,
            "owned_cuda_resident": int(resident.get("owned_pid_vram_mb", 0) or 0) > 0,
        },
        "resident_owned_vram_mb": int(resident.get("owned_pid_vram_mb", 0) or 0),
        "peak_owned_vram_mb": peak_vram,
        "first_token_canary": {
            "prompt_sha256": CANARY_PROMPT_SHA256,
            "first_token_observed": completion_tokens >= 1,
            "completion_tokens": completion_tokens,
            "first_token_sha256": sha256_bytes(token_text.encode()) if token_text else "missing",
            "non_fixture_token": bool(completion_tokens >= 1 and token_text.encode()),
            "bounded": completion_tokens <= CANARY_MAX_OUTPUT_TOKENS,
        },
        "backend_teardown": {"close_called": close_called, "close_error": close_error},
        "vram_recovery": recovery,
        "protected_process_actions": [],
        "unrelated_processes_signaled": [],
        "duration_s": round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6),
        "errors": errors,
    }
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def gpu_receipt_errors(
    receipt: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    require_worker_exit: bool = True,
) -> list[str]:
    """Return every reason a model-local receipt cannot satisfy readiness."""

    errors: list[str] = []
    if receipt.get("receipt_sha256") != gpu_receipt_checksum(receipt):
        errors.append("receipt_sha256")
    if receipt.get("model_id") != model.get("hf_id") or receipt.get("model_record") != model:
        errors.append("model_record")
    planned = next((row for row in MODEL_SPECS if row["hf_id"] == model.get("hf_id")), {})
    if not planned or model_record_errors(model, planned):
        errors.append("model_identity")
    device = receipt.get("device") if isinstance(receipt.get("device"), Mapping) else {}
    worker = (
        receipt.get("worker_process") if isinstance(receipt.get("worker_process"), Mapping) else {}
    )
    owner = receipt.get("lease_owner") if isinstance(receipt.get("lease_owner"), Mapping) else {}
    if not (
        device.get("uuid") in EXPECTED_GPU_UUIDS
        and owner.get("pid") == worker.get("pid")
        and owner.get("pid_start_ticks") == worker.get("pid_start_ticks")
        and owner.get("device_uuid") == device.get("uuid")
        and owner.get("expected_model") == model.get("model_path")
    ):
        errors.append("lease_owner")
    sequence = [event.get("phase") for event in receipt.get("phase_history", [])]
    if sequence != list(COMPLETE_PHASE_SEQUENCE):
        errors.append("phase_sequence")
    release = (
        receipt.get("lease_release") if isinstance(receipt.get("lease_release"), Mapping) else {}
    )
    if release.get("released") is not True or release.get("phase") != "terminal_complete":
        errors.append("lease_release")
    offload = (
        receipt.get("cuda_offload") if isinstance(receipt.get("cuda_offload"), Mapping) else {}
    )
    if not (
        offload.get("requested_gpu_layers") == -1
        and offload.get("supports_gpu_offload") is True
        and offload.get("owned_cuda_resident") is True
        and int(receipt.get("resident_owned_vram_mb", 0) or 0) > 0
        and int(receipt.get("peak_owned_vram_mb", 0) or 0) > 0
    ):
        errors.append("cuda_offload")
    canary = (
        receipt.get("first_token_canary")
        if isinstance(receipt.get("first_token_canary"), Mapping)
        else {}
    )
    if not (
        canary.get("prompt_sha256") == CANARY_PROMPT_SHA256
        and canary.get("first_token_observed") is True
        and int(canary.get("completion_tokens", 0) or 0) >= 1
        and int(canary.get("completion_tokens", 0) or 0) <= CANARY_MAX_OUTPUT_TOKENS
        and canary.get("non_fixture_token") is True
        and canary.get("bounded") is True
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(canary.get("first_token_sha256", "")))
    ):
        errors.append("first_token_canary")
    teardown = (
        receipt.get("backend_teardown")
        if isinstance(receipt.get("backend_teardown"), Mapping)
        else {}
    )
    if teardown.get("close_called") is not True or teardown.get("close_error") is not None:
        errors.append("backend_teardown")
    if require_worker_exit and (
        worker.get("exit_code") != 0 or worker.get("absent_after_exit") is not True
    ):
        errors.append("worker_process")
    if (receipt.get("vram_recovery") or {}).get("passed") is not True:
        errors.append("vram_recovery")
    if receipt.get("protected_process_actions") != []:
        errors.append("protected_process_actions")
    if receipt.get("unrelated_processes_signaled") != []:
        errors.append("unrelated_processes_signaled")
    if receipt.get("errors") != []:
        errors.append("errors")
    return list(dict.fromkeys(errors))


def _blocked_worker_receipt(
    model: Mapping[str, Any], device: Mapping[str, Any], error: str
) -> JsonDict:
    receipt: JsonDict = {
        "model_id": model.get("hf_id"),
        "model_record": deepcopy(dict(model)),
        "device": deepcopy(dict(device)),
        "unrelated_process_inventory": [],
        "worker_process": {
            "pid": 0,
            "pid_start_ticks": None,
            "exit_code": 127,
            "absent_after_exit": True,
        },
        "lease_owner": {},
        "phase_history": [],
        "lease_release": {},
        "cuda_offload": {
            "requested_gpu_layers": -1,
            "supports_gpu_offload": False,
            "owned_cuda_resident": False,
        },
        "resident_owned_vram_mb": 0,
        "peak_owned_vram_mb": 0,
        "first_token_canary": {
            "prompt_sha256": CANARY_PROMPT_SHA256,
            "first_token_observed": False,
            "completion_tokens": 0,
            "first_token_sha256": "missing",
            "non_fixture_token": False,
            "bounded": True,
        },
        "backend_teardown": {"close_called": False, "close_error": None},
        "vram_recovery": build_vram_recovery_receipt(0, 0, False),
        "protected_process_actions": [],
        "unrelated_processes_signaled": [],
        "duration_s": 0.0,
        "errors": [error],
    }
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def worker_environment(  # pragma: no cover - live subprocess boundary
    base: Mapping[str, str], model: Mapping[str, Any], selected_device: Mapping[str, Any]
) -> dict[str, str]:
    """Expose only the selected physical card to one fresh model worker."""

    env = dict(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_device["index"]),
            "CARNOT_6782_EXPECTED_GPU_UUID": str(selected_device["uuid"]),
            "CARNOT_6782_EXPECTED_MODEL": str(model["model_path"]),
            "PYTHONPATH": str(REPO_ROOT / "python")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - live subprocess boundary
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _wait_parent_recovery(  # pragma: no cover - live NVIDIA boundary
    device_uuid: str,
    worker_pid: int,
    before_used_mb: int,
    *,
    timeout_s: float = VRAM_RECOVERY_TIMEOUT_S,
) -> JsonDict:
    deadline = time.monotonic() + timeout_s
    snapshot = _gpu_snapshot(device_uuid, worker_pid)
    receipt = build_vram_recovery_receipt(
        before_used_mb,
        int(snapshot.get("memory_used_mb", 0) or 0),
        bool(snapshot.get("owned_pid_present")),
    )
    while not receipt["passed"] and time.monotonic() < deadline:
        time.sleep(1.0)
        snapshot = _gpu_snapshot(device_uuid, worker_pid)
        receipt = build_vram_recovery_receipt(
            before_used_mb,
            int(snapshot.get("memory_used_mb", 0) or 0),
            bool(snapshot.get("owned_pid_present")),
        )
    receipt["observed_monotonic_ns"] = snapshot.get("observed_monotonic_ns")
    return receipt


def _terminate_owned_worker(  # pragma: no cover - live owned-process boundary
    process: Any, start_ticks: int | None
) -> JsonDict:
    """Stop only the fresh process group whose PID identity still matches."""

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


def run_model_worker(  # pragma: no cover - exercised by the required live E2E run
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    prompt: str,
    runtime_dir: Path,
    *,
    timeout_s: float = WORKER_TIMEOUT_S,
) -> JsonDict:
    """Run one fresh worker and confirm exit and VRAM recovery from the parent."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(model["family_id"])).strip("-").lower()
    model_path = runtime_dir / f"{slug}.model.json"
    device_path = runtime_dir / f"{slug}.device.json"
    prompt_path = runtime_dir / f"{slug}.prompt.json"
    output_path = runtime_dir / f"{slug}.receipt.json"
    lease_api.write_json_atomic(model_path, model)
    lease_api.write_json_atomic(device_path, selected_device)
    lease_api.write_json_atomic(prompt_path, {"prompt": prompt})
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6782_sequential_sota_runtime_admission",
        "--worker",
        "--worker-model",
        str(model_path),
        "--worker-device",
        str(device_path),
        "--worker-prompt",
        str(prompt_path),
        "--worker-output",
        str(output_path),
        "--lease-runtime-dir",
        str(LEASE_RUNTIME_DIR),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=worker_environment(os.environ, model, selected_device),
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
        timeout_cleanup = _terminate_owned_worker(process, start_ticks)
        stdout, stderr = process.communicate()
    receipt = _load_json(output_path)
    if not receipt:
        receipt = _blocked_worker_receipt(
            model,
            selected_device,
            f"worker_output_missing:exit={process.returncode}:stderr={sha256_bytes(stderr.encode())}",
        )
    worker = (
        receipt.get("worker_process") if isinstance(receipt.get("worker_process"), Mapping) else {}
    )
    receipt["worker_process"] = {
        **dict(worker),
        "pid": int(process.pid),
        "pid_start_ticks": start_ticks,
        "exit_code": process.returncode,
        "absent_after_exit": process.poll() is not None,
        "stdout_sha256": sha256_bytes(stdout.encode()),
        "stderr_sha256": sha256_bytes(stderr.encode()),
        "timeout_cleanup": timeout_cleanup,
    }
    receipt["vram_recovery"] = _wait_parent_recovery(
        str(selected_device["uuid"]),
        int(process.pid),
        int(selected_device.get("memory_used_mb", 0) or 0),
    )
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def receipt_ready(receipt: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    """Reduce one model receipt without borrowing evidence from another model."""

    return not gpu_receipt_errors(receipt, model)


def lifecycle_rows(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create one load, inference, teardown, and recovery row per attempt."""

    rows: list[JsonDict] = []
    for receipt in receipts:
        model_id = receipt.get("model_id")
        phases = [event.get("phase") for event in receipt.get("phase_history", [])]
        rows.extend(
            [
                {
                    "row_kind": "model_load",
                    "model_id": model_id,
                    "lease_owner": deepcopy(receipt.get("lease_owner")),
                    "device_uuid": (receipt.get("device") or {}).get("uuid"),
                    "cuda_offload": deepcopy(receipt.get("cuda_offload")),
                    "resident_owned_vram_mb": receipt.get("resident_owned_vram_mb"),
                    "passed": "resident" in phases,
                },
                {
                    "row_kind": "inference",
                    "model_id": model_id,
                    "first_token_canary": deepcopy(receipt.get("first_token_canary")),
                    "passed": (receipt.get("first_token_canary") or {}).get("first_token_observed")
                    is True,
                },
                {
                    "row_kind": "teardown",
                    "model_id": model_id,
                    "backend_teardown": deepcopy(receipt.get("backend_teardown")),
                    "worker_process": deepcopy(receipt.get("worker_process")),
                    "passed": (receipt.get("backend_teardown") or {}).get("close_called") is True
                    and (receipt.get("worker_process") or {}).get("absent_after_exit") is True,
                },
                {
                    "row_kind": "recovery",
                    "model_id": model_id,
                    "lease_release": deepcopy(receipt.get("lease_release")),
                    "vram_recovery": deepcopy(receipt.get("vram_recovery")),
                    "passed": (receipt.get("lease_release") or {}).get("released") is True
                    and (receipt.get("vram_recovery") or {}).get("passed") is True,
                },
            ]
        )
    return rows


def _readiness(
    models: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    by_id = {str(row.get("model_id")): row for row in receipts}
    ready: JsonDict = {}
    for model in models:
        receipt = by_id.get(str(model.get("hf_id")))
        ready[str(model.get("family_id"))] = bool(
            isinstance(receipt, Mapping) and receipt_ready(receipt, model)
        )
    return ready


def _gate_summary(
    preconditions: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    failures = [row for row in preconditions.get("checks", []) if row.get("passed") is not True]
    ready = _readiness(models, receipts)
    failed_check: str | None = None
    observed: Any = True
    expected: Any = True
    if failures:
        failed_check = str(failures[0].get("check"))
        observed = deepcopy(failures[0].get("observed"))
        expected = deepcopy(failures[0].get("expected"))
    else:
        by_id = {str(row.get("model_id")): row for row in receipts}
        for model in models:
            if not ready.get(str(model.get("family_id")), False):
                failed_check = f"runtime_admission:{model.get('hf_id')}"
                receipt = by_id.get(str(model.get("hf_id")))
                observed = (
                    gpu_receipt_errors(receipt, model)
                    if isinstance(receipt, Mapping)
                    else "lease_wait_deadline_expired"
                )
                break
    return {
        "all_preconditions_passed": preconditions.get("all_passed") is True,
        "checks": {
            str(row.get("check")): row.get("passed") is True
            for row in preconditions.get("checks", [])
        },
        "failed_check": failed_check,
        "expected": expected,
        "observed": observed,
        "model_readiness": ready,
    }


def _state(models: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    ready = _readiness(models, receipts)
    qwen = ready.get("qwen36", False)
    dense = ready.get("gemma31", False)
    middle = ready.get("gemma26", False)
    all_ready = bool(qwen and dense and middle)
    count = sum((qwen, dense, middle))
    if all_ready:
        return {
            "status": "complete_sequential_sota_runtime",
            "verdict_class": "positive",
            "honest_verdict": "complete: all three mandated local CUDA runtime admissions passed.",
            "qwen36": True,
            "gemma31": True,
            "gemma26": True,
            "all": True,
        }
    if count:
        verdict = (
            f"complete_partial_sequential_sota_runtime: {count} of 3 model-local admissions passed."
        )
        verdict_class = "partial"
        status = "complete_partial_sequential_sota_runtime"
    else:
        verdict = (
            "complete_blocked_sequential_sota_runtime: no model completed owned CUDA admission."
        )
        verdict_class = "blocked"
        status = "complete_blocked_sequential_sota_runtime"
    return {
        "status": status,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
        "qwen36": bool(qwen),
        "gemma31": bool(dense),
        "gemma26": bool(middle),
        "all": False,
    }


def build_artifact(
    *,
    date: str,
    preconditions: Mapping[str, Any],
    gpu_receipts: Sequence[Mapping[str, Any]],
    poll_rows: Sequence[Mapping[str, Any]],
    code_receipts: Mapping[str, Any],
    started_ns: int,
    finished_ns: int,
) -> JsonDict:
    """Reduce durable model-local evidence into the required terminal schema."""

    models = [deepcopy(dict(row)) for row in preconditions.get("models", [])]
    receipts = [deepcopy(dict(row)) for row in gpu_receipts]
    retained_polls = [deepcopy(dict(row)) for row in poll_rows]
    state = _state(models, receipts)
    used = [
        deepcopy(model)
        for model in models
        if any(
            receipt.get("model_id") == model.get("hf_id") and receipt_ready(receipt, model)
            for receipt in receipts
        )
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(date),
        "status": state["status"],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, int(finished_ns) - int(started_ns)) / 1_000_000_000, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "MODEL_SPECS": deepcopy(list(MODEL_SPECS)),
        "model_specs": models,
        "models_used": used,
        "live_model_invoked": any(
            (receipt.get("first_token_canary") or {}).get("first_token_observed") is True
            for receipt in receipts
        ),
        "rows": retained_polls + lifecycle_rows(receipts),
        "gpu_receipts": receipts,
        "protected_process_actions": [],
        "qwen36_runtime_ready": state["qwen36"],
        "gemma31_runtime_ready": state["gemma31"],
        "gemma26_runtime_ready": state["gemma26"],
        "all_mandated_runtime_ready": state["all"],
        "preconditions_checked": deepcopy(dict(preconditions)),
        "code_receipts": deepcopy(dict(code_receipts)),
        "gate_check_summary": _gate_summary(preconditions, models, receipts),
        "verifier_is_oracle": False,
        "verdict_class": state["verdict_class"],
        "honest_verdict": state["honest_verdict"],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-recompute all top-level readiness, rows, verdicts, and hashes."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_field_set")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("MODEL_SPECS") != list(MODEL_SPECS):
        errors.append("MODEL_SPECS")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        errors.append("duration_s")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if not re.match(
        r"^(complete:|complete_|success:|success_|passed:|passed_|shipped:|shipped_)",
        str(artifact.get("honest_verdict", "")),
    ):
        errors.append("honest_verdict_prefix")
    models = artifact.get("model_specs") if isinstance(artifact.get("model_specs"), list) else []
    preconditions = (
        artifact.get("preconditions_checked")
        if isinstance(artifact.get("preconditions_checked"), Mapping)
        else {}
    )
    if models != preconditions.get("models") or len(models) != len(MODEL_SPECS):
        errors.append("model_specs")
    receipts = (
        artifact.get("gpu_receipts") if isinstance(artifact.get("gpu_receipts"), list) else []
    )
    poll_rows = [row for row in artifact.get("rows", []) if row.get("row_kind") == "lease_poll"]
    expected_rows = poll_rows + lifecycle_rows(receipts)
    if artifact.get("rows") != expected_rows:
        errors.append("rows")
    if (
        any(row.get("protected_process_actions") not in (None, []) for row in receipts)
        or artifact.get("protected_process_actions") != []
    ):
        errors.append("protected_process_actions")
    state = _state(models, receipts)
    mapping = {
        "status": state["status"],
        "verdict_class": state["verdict_class"],
        "honest_verdict": state["honest_verdict"],
        "qwen36_runtime_ready": state["qwen36"],
        "gemma31_runtime_ready": state["gemma31"],
        "gemma26_runtime_ready": state["gemma26"],
        "all_mandated_runtime_ready": state["all"],
    }
    for field, expected in mapping.items():
        if artifact.get(field) != expected:
            errors.append(field)
    expected_used = [
        deepcopy(model)
        for model in models
        if any(
            receipt.get("model_id") == model.get("hf_id") and receipt_ready(receipt, model)
            for receipt in receipts
        )
    ]
    if artifact.get("models_used") != expected_used:
        errors.append("models_used")
    expected_live = any(
        (receipt.get("first_token_canary") or {}).get("first_token_observed") is True
        for receipt in receipts
    )
    if artifact.get("live_model_invoked") is not expected_live:
        errors.append("live_model_invoked")
    if artifact.get("gate_check_summary") != _gate_summary(preconditions, models, receipts):
        errors.append("gate_check_summary")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and publish one checkpoint through atomic replacement."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6782 artifact: " + ",".join(errors))
    lease_api.write_json_atomic(path, artifact)


def run(
    *,
    result_path: Path = RESULT_PATH,
    date: str = RUN_DATE,
    preflight_fn: Callable[[], JsonDict] = collect_preconditions,
    device_waiter: Callable[[int, str], tuple[JsonDict | None, list[JsonDict]]] | None = None,
    worker_runner: Callable[
        [Mapping[str, Any], Mapping[str, Any], str, Path], JsonDict
    ] = run_model_worker,
    code_receipt_fn: Callable[[], JsonDict] = code_receipts,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run preconditions, bounded waits, and three sequential atomic checkpoints."""

    started_ns = int(clock())
    wait_deadline_ns = started_ns + int(LEASE_WAIT_TIMEOUT_S * 1_000_000_000)
    preconditions = preflight_fn()
    receipts: list[JsonDict] = []
    poll_rows: list[JsonDict] = []
    codes = code_receipt_fn()
    runtime_dir = result_path.parent / ".experiment_6782_sequential_sota_runtime_admission"
    if device_waiter is None:  # pragma: no cover - live bounded-wait binding
        device_waiter = lambda deadline, model_id: wait_for_eligible_device(
            deadline_ns=deadline, model_id=model_id
        )
    artifact = build_artifact(
        date=date,
        preconditions=preconditions,
        gpu_receipts=receipts,
        poll_rows=poll_rows,
        code_receipts=codes,
        started_ns=started_ns,
        finished_ns=int(clock()),
    )
    write_artifact(result_path, artifact)
    if preconditions.get("all_passed") is not True:
        return artifact
    for model in preconditions.get("models", []):
        selected, polls = device_waiter(wait_deadline_ns, str(model.get("hf_id")))
        poll_rows.extend(deepcopy(polls))
        if isinstance(selected, Mapping):
            receipt = worker_runner(model, selected, CANARY_PROMPT, runtime_dir)
            receipts.append(receipt)
        artifact = build_artifact(
            date=date,
            preconditions=preconditions,
            gpu_receipts=receipts,
            poll_rows=poll_rows,
            code_receipts=codes,
            started_ns=started_ns,
            finished_ns=int(clock()),
        )
        write_artifact(result_path, artifact)
        if not isinstance(selected, Mapping) or gpu_receipt_errors(receipts[-1], model):
            break
    return artifact


def _worker_entry(  # pragma: no cover - live subprocess entry point
    model_path: Path,
    device_path: Path,
    prompt_path: Path,
    output_path: Path,
    lease_runtime_dir: Path,
) -> int:
    model = _load_json(model_path)
    device = _load_json(device_path)
    prompt = str(_load_json(prompt_path).get("prompt", ""))
    receipt = run_live_model_worker(
        model,
        device,
        prompt=prompt,
        lease_runtime_dir=lease_runtime_dir,
    )
    lease_api.write_json_atomic(output_path, receipt)
    return 0 if not receipt.get("errors") else 2


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """Run the parent contract, one owned worker, or a cold validation pass."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--worker-device", type=Path)
    parser.add_argument("--worker-prompt", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--lease-runtime-dir", type=Path, default=LEASE_RUNTIME_DIR)
    args = parser.parse_args(argv)
    if args.worker:
        required = (args.worker_model, args.worker_device, args.worker_prompt, args.worker_output)
        if not all(value is not None for value in required):
            parser.error("--worker requires model, device, prompt, and output paths")
        return _worker_entry(
            args.worker_model,
            args.worker_device,
            args.worker_prompt,
            args.worker_output,
            args.lease_runtime_dir,
        )
    if args.validate:
        errors = validate_artifact(_load_json(args.result_path))
        if errors:
            raise ValueError("invalid Exp6782 artifact: " + ",".join(errors))
        return 0
    if args.date != RUN_DATE:
        raise ValueError(f"execution date must be {RUN_DATE}")
    artifact = run(result_path=args.result_path, date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "all_mandated_runtime_ready": artifact["all_mandated_runtime_ready"],
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns CLI coverage
    raise SystemExit(main())
