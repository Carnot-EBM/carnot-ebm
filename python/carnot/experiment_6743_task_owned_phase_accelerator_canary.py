"""Run the Exp6743 task-owned CUDA canary.

The canary proves that each mandated local GGUF can load, reach a first token,
and release its owned process. It does not measure model quality or compare
model speed.

Spec refs: REQ-INFRA-6743, SCENARIO-INFRA-6743-MONOTONIC-ELAPSED,
SCENARIO-INFRA-6743-ACCELERATOR-INTEGRITY, and
SCENARIO-INFRA-6743-BLOCKED-PREFLIGHT.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
MODULE_NAME = "carnot.experiment_6743_task_owned_phase_accelerator_canary"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6743_task_owned_phase_accelerator_canary.json"
RUN_DATE = "20260829"
RANDOM_SEED = 6_743
FIXED_PROMPT = "Reply with exactly one uppercase word: CANARY"
MAX_OUTPUT_TOKENS = 4
N_CTX = 256
N_GPU_LAYERS = -1
MODEL_TIMEOUT_S = 900.0
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "local llama.cpp CUDA GGUF"
CLAIM_BOUNDARY = (
    "This tiny canary proves local execution receipts only. It does not compare model "
    "quality or speed, rank models, or gate a science branch."
)
PROMPT_SHA256 = "sha256:" + hashlib.sha256(FIXED_PROMPT.encode("utf-8")).hexdigest()

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": PREFERRED_QUANT,
        "device_index": 0,
    },
    {
        "model_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "quantization": PREFERRED_QUANT,
        "device_index": 1,
    },
    {
        "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "quantization": PREFERRED_QUANT,
        "device_index": 0,
    },
)

MODEL_PHASES = (
    "subprocess_started",
    "model_loaded",
    "first_token",
    "decode_complete",
    "teardown_complete",
)
REQUIRED_PHASES = (
    "preflight",
    "cache_resolved",
    *MODEL_PHASES,
    "artifact_write",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "models_used",
    "live_model_invoked",
    "rows",
    "task_phase_rows",
    "phase_clock_monotonic",
    "gpu_receipts",
    "accelerator_receipt_ready",
    "claim_boundary",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why an auditor needs it.",
    "inference_substrate": "The substrate excludes CPU and remote inference substitutions.",
    "duration_s": "A monotonic wall interval prevents a synthetic zero-minute receipt.",
    "random_seed": "A fixed seed makes the bounded decode configuration repeatable.",
    "reproducibility_checksum": "The config and row hashes detect receipt drift.",
    "models_used": "Exact hub IDs prevent legacy or substitute models from counting.",
    "live_model_invoked": "This flag stays false until a real decode completes.",
    "rows": "One raw row per model preserves model-owned evidence.",
    "task_phase_rows": "Task-owned clocks show where real work occurred.",
    "phase_clock_monotonic": "A row-derived flag exposes skipped or reversed phases.",
    "gpu_receipts": "Before, during, and after samples bind CUDA use to an owned PID.",
    "accelerator_receipt_ready": "Readiness needs all three CUDA decodes and clean teardown.",
    "claim_boundary": "The boundary forbids quality, speed, ranking, and science-gate claims.",
    "gate_check_summary": "Blocked runs preserve the failed check and observed value.",
    "verdict_class": "A closed class gives automation an unambiguous terminal state.",
    "honest_verdict": "An allowed prefix reports success or a task-owned block plainly.",
}

CONFIG = {
    "model_ids": [row["model_id"] for row in MODEL_SPECS],
    "prompt_sha256": PROMPT_SHA256,
    "random_seed": RANDOM_SEED,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "n_ctx": N_CTX,
    "n_gpu_layers": N_GPU_LAYERS,
    "sequential": True,
}

GPU_LAYER_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.I)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for content hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value in canonical form."""

    return sha256_text(canonical_json(value))


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a model row without its self-referential hash field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def reproducibility_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the fixed config to the three task-owned model row hashes."""

    return sha256_json(
        {
            "config": CONFIG,
            "row_hashes": [row.get("row_sha256") for row in rows],
        }
    )


def _file_head_hash(path: Path, limit: int = 1024 * 1024) -> str:
    """Hash a bounded file prefix so cache identity does not require a full reread."""

    with path.open("rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read(limit)).hexdigest()


def _snapshot_revision(path: Path) -> str | None:
    """Read the immutable Hugging Face snapshot revision from a cache path."""

    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def model_identity_receipt(spec: Mapping[str, Any], path: Path) -> JsonDict:
    """Bind a declared model to its exact cached path and local file identity."""

    exact_path = path.absolute()
    present = exact_path.is_file()
    size = exact_path.stat().st_size if present else 0
    return {
        **dict(spec),
        "resolved": present,
        "resolved_path": str(exact_path) if present else str(path),
        "resolved_path_sha256": sha256_text(str(exact_path) if present else str(path)),
        "required_vram_mb": math.ceil(size / (1024 * 1024)) + 512 if present else 0,
        "file_identity": {
            "filename": exact_path.name,
            "size_bytes": size,
            "mtime_ns": exact_path.stat().st_mtime_ns if present else None,
            "head_1m_sha256": _file_head_hash(exact_path) if present else None,
            "snapshot_revision": _snapshot_revision(path),
            "quantization": spec["quantization"],
        },
    }


def resolve_model_specs(
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
) -> list[JsonDict]:
    """Resolve all three models through the repository's cached SOTA pair helper."""

    default_pair = (
        cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=None)
        or []
    )
    dense_pair = (
        cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(0, 2))
        or []
    )
    by_id = {
        str(row.get("hf_id")): str(row.get("model_path") or "")
        for row in [*default_pair, *dense_pair]
    }
    rows = []
    for spec in MODEL_SPECS:
        path_text = by_id.get(str(spec["model_id"]), "")
        rows.append(
            model_identity_receipt(spec, Path(path_text)) if path_text else _missing_model(spec)
        )
    return rows


def _missing_model(spec: Mapping[str, Any]) -> JsonDict:
    """Preserve the declared identity when an exact local cache is absent."""

    return {
        **dict(spec),
        "resolved": False,
        "resolved_path": "",
        "resolved_path_sha256": sha256_text(""),
        "required_vram_mb": 0,
        "file_identity": {
            "filename": None,
            "size_bytes": 0,
            "mtime_ns": None,
            "head_1m_sha256": None,
            "snapshot_revision": None,
            "quantization": spec["quantization"],
        },
    }


def _run_text_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:
    """Run one host probe and keep its exact return state."""

    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def nvidia_smi_inventory() -> JsonDict:
    """Collect physical GPU identity and free-memory evidence from nvidia-smi."""

    receipt = _run_text_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    devices = []
    for line in receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "driver_version": parts[6],
                }
            )
        except ValueError:
            continue
    return {**receipt, "devices": devices}


def live_preflight(models: list[JsonDict]) -> JsonDict:
    """Check exact caches, CUDA offload, devices, and per-model free VRAM."""

    try:
        from llama_cpp import __version__ as llama_version
        from llama_cpp import llama_cpp as llama_lib

        offload_observed: Any = bool(llama_lib.llama_supports_gpu_offload())
        version: str | None = str(llama_version)
    except Exception as exc:
        offload_observed = f"{type(exc).__name__}: {exc}"
        version = None
    inventory = nvidia_smi_inventory()
    by_index = {row["index"]: row for row in inventory["devices"]}
    checks: list[JsonDict] = [
        {
            "check": "llama_cpp_cuda_offload",
            "expected": True,
            "observed": offload_observed,
            "llama_cpp_version": version,
            "passed": offload_observed is True,
        },
        {
            "check": "nvidia_smi_available",
            "expected": True,
            "observed": inventory["ok"],
            "passed": inventory["ok"] is True,
        },
    ]
    for model in models:
        model_id = str(model["model_id"])
        checks.append(
            {
                "check": f"cache:{model_id}",
                "expected": True,
                "observed": model.get("resolved") is True,
                "resolved_path": model.get("resolved_path"),
                "passed": model.get("resolved") is True,
            }
        )
        device = by_index.get(int(model["device_index"]))
        observed_free = device.get("memory_free_mb") if device else None
        required_free = int(model.get("required_vram_mb", 0))
        checks.append(
            {
                "check": f"free_vram_mb:{model_id}",
                "expected": {"at_least": required_free},
                "observed": observed_free,
                "device_index": model["device_index"],
                "passed": observed_free is not None
                and required_free > 0
                and int(observed_free) >= required_free,
            }
        )
    return {
        "all_passed": all(check["passed"] is True for check in checks),
        "checks": checks,
        "gpu_inventory": inventory,
    }


def gpu_snapshot_for_pid(
    device_index: int,
    pid: int,
    phase: str,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Capture one PID-bound nvidia-smi sample for the assigned device."""

    inventory = nvidia_smi_inventory()
    process_receipt = _run_text_command(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        )
    )
    processes = []
    for line in process_receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            processes.append(
                {
                    "pid": int(parts[0]),
                    "device_uuid": parts[1],
                    "used_memory_mb": int(parts[2]),
                    "process_name": parts[3],
                }
            )
        except ValueError:
            continue
    device = next((row for row in inventory["devices"] if row["index"] == device_index), {})
    process = next(
        (
            row
            for row in processes
            if row["pid"] == pid and row["device_uuid"] == device.get("uuid")
        ),
        None,
    )
    return {
        "phase": phase,
        "monotonic_ns": clock(),
        "query_ok": inventory["ok"] is True and process_receipt["ok"] is True,
        "pid": pid,
        "pid_present": process is not None,
        "device_index": device_index,
        "device_uuid": device.get("uuid"),
        "device_name": device.get("name"),
        "device_memory_used_mb": device.get("memory_used_mb"),
        "device_memory_free_mb": device.get("memory_free_mb"),
        "pid_memory_mb": int(process["used_memory_mb"]) if process else 0,
        "process_name": process.get("process_name") if process else None,
        "compute_processes": processes,
    }


def _worker_decode(
    payload: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any],
    snapshot_fn: Callable[[int, int, str], JsonDict],
    supports_gpu_offload: bool,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Load and decode one model inside its dedicated worker process."""

    pid = os.getpid()
    device_index = int(payload["device_index"])
    started_ns = clock()
    before = snapshot_fn(device_index, pid, "before")
    llm = llama_factory(
        model_path=str(payload["resolved_path"]),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        main_gpu=0,
        n_batch=64,
        n_ubatch=64,
        seed=int(payload["random_seed"]),
        verbose=True,
    )
    loaded_ns = clock()
    prompt_tokens = len(llm.tokenize(str(payload["prompt"]).encode("utf-8")))
    pieces: list[str] = []
    first_token_ns: int | None = None
    during: JsonDict | None = None
    stop_reason: str | None = None
    for chunk in llm.create_completion(
        str(payload["prompt"]),
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stream=True,
    ):
        choice = (chunk.get("choices") or [{}])[0]
        piece = str(choice.get("text") or "")
        if piece and first_token_ns is None:
            first_token_ns = clock()
            during = snapshot_fn(device_index, pid, "during")
        if choice.get("finish_reason") is not None:
            stop_reason = str(choice["finish_reason"])
        pieces.append(piece)
    decode_complete_ns = clock()
    raw_output = "".join(pieces)
    output_tokens = (
        len(llm.tokenize(raw_output.encode("utf-8"), add_bos=False)) if raw_output else 0
    )
    if during is None:
        during = snapshot_fn(device_index, pid, "during")
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    return {
        "owned_pid": pid,
        "parent_pid": os.getppid(),
        "supports_gpu_offload": supports_gpu_offload,
        "clocks": {
            "subprocess_started_ns": started_ns,
            "model_loaded_ns": loaded_ns,
            "first_token_ns": first_token_ns,
            "decode_complete_ns": decode_complete_ns,
        },
        "gpu_receipts": {"before": before, "during": during},
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason or "unknown",
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "first_token_reached": first_token_ns is not None,
        "decode_completed": True,
    }


def worker_main(
    payload_text: str,
    *,
    llama_factory: Callable[..., Any] | None = None,
    snapshot_fn: Callable[[int, int, str], JsonDict] = gpu_snapshot_for_pid,
    supports_gpu_offload: bool | None = None,
    emit: Callable[[str], None] = print,
) -> int:
    """Run the child protocol and emit exactly one JSON object to stdout."""

    try:
        payload = json.loads(payload_text)
        if llama_factory is None or supports_gpu_offload is None:
            from llama_cpp import Llama
            from llama_cpp import llama_cpp as llama_lib

            llama_factory = llama_factory or Llama
            if supports_gpu_offload is None:
                supports_gpu_offload = bool(llama_lib.llama_supports_gpu_offload())
        receipt = _worker_decode(
            payload,
            llama_factory=llama_factory,
            snapshot_fn=snapshot_fn,
            supports_gpu_offload=bool(supports_gpu_offload),
        )
    except Exception as exc:
        receipt = {
            "owned_pid": os.getpid(),
            "runtime_error": f"{type(exc).__name__}: {exc}",
        }
        emit(canonical_json(receipt))
        return 1
    emit(canonical_json(receipt))
    return 0


def parse_gpu_layers(stderr_text: str) -> JsonDict:
    """Extract actual offloaded and total layer counts from llama.cpp logs."""

    matches = GPU_LAYER_RE.findall(stderr_text)
    if not matches:
        return {"requested": N_GPU_LAYERS, "offloaded": 0, "total": None}
    offloaded, total = matches[-1]
    return {"requested": N_GPU_LAYERS, "offloaded": int(offloaded), "total": int(total)}


def _parse_worker_stdout(stdout: str) -> JsonDict:
    """Read the final JSON line and reject non-object worker output."""

    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return {"runtime_error": "worker_json_receipt_missing"}


def run_live_model(
    model: Mapping[str, Any],
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    snapshot_fn: Callable[[int, int, str], JsonDict] = gpu_snapshot_for_pid,
    clock: Callable[[], int] = time.monotonic_ns,
    proc_root: Path = Path("/proc"),
) -> JsonDict:
    """Launch one fresh worker and return a parent-verified model row."""

    payload = {
        "model_id": model["model_id"],
        "resolved_path": model["resolved_path"],
        "device_index": model["device_index"],
        "prompt": FIXED_PROMPT,
        "random_seed": RANDOM_SEED,
    }
    argv = [sys.executable, "-m", MODULE_NAME, "--worker", canonical_json(payload)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(model["device_index"])
    proc = popen_factory(
        argv,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=MODEL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    child = _parse_worker_stdout(stdout)
    pid = int(child.get("owned_pid", proc.pid))
    after = snapshot_fn(int(model["device_index"]), pid, "after")
    teardown_ns = clock()
    clocks = dict(child.get("clocks", {}))
    clocks["teardown_complete_ns"] = teardown_ns
    gpu_receipts = dict(child.get("gpu_receipts", {}))
    gpu_receipts["after"] = after
    during = dict(gpu_receipts.get("during", {}))
    assigned_uuid = during.get("device_uuid") or after.get("device_uuid")
    process_absent = proc.returncode is not None and not (proc_root / str(pid)).exists()
    row = {
        **dict(model),
        "owned_pid": pid,
        "parent_pid": child.get("parent_pid"),
        "assigned_device": {
            "index": model["device_index"],
            "uuid": assigned_uuid,
            "cuda_visible_devices": str(model["device_index"]),
        },
        "gpu_layers": parse_gpu_layers(stderr),
        "peak_vram_mb": max(
            (int(dict(sample).get("pid_memory_mb", 0) or 0) for sample in gpu_receipts.values()),
            default=0,
        ),
        "prompt_sha256": PROMPT_SHA256,
        "prompt_tokens": int(child.get("prompt_tokens", 0) or 0),
        "output_tokens": int(child.get("output_tokens", 0) or 0),
        "stop_reason": child.get("stop_reason"),
        "raw_output": str(child.get("raw_output", "")),
        "raw_output_sha256": child.get("raw_output_sha256", sha256_text("")),
        "first_token_reached": child.get("first_token_reached") is True,
        "decode_completed": child.get("decode_completed") is True,
        "supports_gpu_offload": child.get("supports_gpu_offload") is True,
        "process_exit_code": proc.returncode,
        "owned_process_absent": process_absent,
        "teardown_completed": proc.returncode == 0 and process_absent and not timed_out,
        "clocks": clocks,
        "gpu_receipts": gpu_receipts,
        "backend_stderr_sha256": sha256_text(stderr),
        "runtime_error": child.get("runtime_error") or ("worker_timeout" if timed_out else None),
    }
    row["row_sha256"] = row_checksum(row)
    return row


def _blocked_model_row(model: Mapping[str, Any], reason: str) -> JsonDict:
    """Keep one declared row per model when a preflight or launch blocks."""

    row = {
        **dict(model),
        "owned_pid": None,
        "parent_pid": os.getpid(),
        "assigned_device": {
            "index": model["device_index"],
            "uuid": None,
            "cuda_visible_devices": str(model["device_index"]),
        },
        "gpu_layers": {"requested": N_GPU_LAYERS, "offloaded": 0, "total": None},
        "peak_vram_mb": 0,
        "prompt_sha256": PROMPT_SHA256,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "stop_reason": "blocked",
        "raw_output": "",
        "raw_output_sha256": sha256_text(""),
        "first_token_reached": False,
        "decode_completed": False,
        "supports_gpu_offload": False,
        "process_exit_code": None,
        "owned_process_absent": True,
        "teardown_completed": False,
        "clocks": {},
        "gpu_receipts": {},
        "backend_stderr_sha256": sha256_text(""),
        "runtime_error": reason,
    }
    row["row_sha256"] = row_checksum(row)
    return row


def phase_clock_is_monotonic(phase_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only for a positive, strictly increasing task clock."""

    values = [row.get("monotonic_ns") for row in phase_rows]
    return (
        len(values) >= 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
        and all(left < right for left, right in zip(values, values[1:], strict=False))
    )


def phase_errors(
    phase_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate global markers and the complete fixed sequential phase order."""

    errors: list[str] = []
    if not phase_clock_is_monotonic(phase_rows):
        errors.append("phase_clock_not_monotonic")
    if phase_rows:
        first = phase_rows[0].get("monotonic_ns")
        last = phase_rows[-1].get("monotonic_ns")
        if not isinstance(first, int) or not isinstance(last, int) or last <= first:
            errors.append("phase_duration_not_positive")
    else:
        errors.append("phase_duration_not_positive")
    phases = [row.get("phase") for row in phase_rows]
    for phase in ("preflight", "cache_resolved", "artifact_write"):
        if phase not in phases:
            errors.append(f"missing_phase:{phase}")
    complete = len(rows) == len(MODEL_SPECS) and all(
        row.get("teardown_completed") is True for row in rows
    )
    if complete:
        expected: list[tuple[str, str | None]] = [("preflight", None)]
        expected.extend(("cache_resolved", str(row["model_id"])) for row in rows)
        for row in rows:
            expected.extend((phase, str(row["model_id"])) for phase in MODEL_PHASES)
        expected.append(("artifact_write", None))
        observed = [(str(row.get("phase")), row.get("model_id")) for row in phase_rows]
        if observed != expected:
            errors.append("phase_sequence_mismatch")
    return errors


def model_row_errors(row: Mapping[str, Any]) -> list[str]:
    """Validate one model's cache, decode, CUDA, and teardown bindings."""

    errors: list[str] = []
    model_id = str(row.get("model_id"))
    if row.get("resolved") is not True:
        errors.append("cache_not_resolved")
    path = str(row.get("resolved_path", ""))
    if row.get("resolved_path_sha256") != sha256_text(path):
        errors.append("resolved_path_hash_mismatch")
    if row.get("prompt_sha256") != PROMPT_SHA256:
        errors.append("prompt_hash_mismatch")
    if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
        errors.append("raw_output_hash_mismatch")
    if int(row.get("prompt_tokens", 0) or 0) <= 0:
        errors.append("prompt_tokens_nonpositive")
    if int(row.get("output_tokens", 0) or 0) <= 0:
        errors.append("output_tokens_nonpositive")
    if row.get("first_token_reached") is not True:
        errors.append("first_token_missing")
    if row.get("decode_completed") is not True:
        errors.append("decode_incomplete")
    if row.get("supports_gpu_offload") is not True:
        errors.append("cuda_offload_unsupported")
    layers = dict(row.get("gpu_layers", {}))
    if int(layers.get("offloaded", 0) or 0) <= 0:
        errors.append("gpu_layers_not_offloaded")
    clocks = dict(row.get("clocks", {}))
    clock_values = [clocks.get(f"{phase}_ns") for phase in MODEL_PHASES]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in clock_values):
        errors.append("model_clock_missing")
    elif not all(left < right for left, right in zip(clock_values, clock_values[1:], strict=False)):
        errors.append("model_clock_not_monotonic")
    pid = row.get("owned_pid")
    assigned = dict(row.get("assigned_device", {}))
    receipts = dict(row.get("gpu_receipts", {}))
    before = dict(receipts.get("before", {}))
    during = dict(receipts.get("during", {}))
    after = dict(receipts.get("after", {}))
    if before.get("pid_present") is not False:
        errors.append("before_gpu_receipt_invalid")
    if (
        during.get("pid_present") is not True
        or during.get("pid") != pid
        or during.get("device_uuid") != assigned.get("uuid")
        or int(during.get("pid_memory_mb", 0) or 0) <= 0
    ):
        errors.append("during_gpu_receipt_invalid")
    if (
        after.get("pid_present") is not False
        or after.get("pid") != pid
        or after.get("device_uuid") != assigned.get("uuid")
    ):
        errors.append("after_gpu_receipt_invalid")
    if (
        row.get("process_exit_code") != 0
        or row.get("owned_process_absent") is not True
        or row.get("teardown_completed") is not True
    ):
        errors.append("teardown_incomplete")
    if row.get("runtime_error") not in (None, ""):
        errors.append(f"runtime_error:{model_id}")
    return errors


def reduce_accelerator_readiness(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require one valid distinct row for every mandated model."""

    expected = [str(spec["model_id"]) for spec in MODEL_SPECS]
    observed = [str(row.get("model_id")) for row in rows]
    return (
        observed == expected
        and len({row.get("owned_pid") for row in rows}) == len(expected)
        and all(not model_row_errors(row) for row in rows)
    )


def build_artifact(
    *,
    rows: list[JsonDict],
    task_phase_rows: list[JsonDict],
    preflight: Mapping[str, Any],
    started_ns: int,
    artifact_write_ns: int,
) -> JsonDict:
    """Build the terminal artifact only from task-owned measured rows."""

    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["row_sha256"] = row_checksum(normalized)
        normalized_rows.append(normalized)
    monotonic = not phase_errors(task_phase_rows, normalized_rows)
    receipt_ready = (
        preflight.get("all_passed") is True
        and monotonic
        and reduce_accelerator_readiness(normalized_rows)
    )
    live_count = sum(1 for row in normalized_rows if row.get("decode_completed") is True)
    live_model_invoked = live_count > 0
    failed_checks = [
        dict(check) for check in preflight.get("checks", []) if check.get("passed") is not True
    ]
    if preflight.get("all_passed") is not True:
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_accelerator_canary: preflight check failed"
    elif receipt_ready:
        verdict_class = "positive"
        honest_verdict = (
            "complete: all three mandated models reached a first token on CUDA and "
            "teardown completed"
        )
    else:
        verdict_class = "partial"
        honest_verdict = (
            f"complete_blocked_accelerator_canary: {live_count} of {len(MODEL_SPECS)} "
            "model receipts passed decode"
        )
        failed_checks.extend(
            {
                "check": f"accelerator_receipt:{row['model_id']}",
                "expected": [],
                "observed": model_row_errors(row),
                "passed": False,
            }
            for row in normalized_rows
            if model_row_errors(row)
        )
    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round((artifact_write_ns - started_ns) / 1_000_000_000, 9),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(normalized_rows),
        "models_used": [spec["model_id"] for spec in MODEL_SPECS],
        "live_model_invoked": live_model_invoked,
        "rows": normalized_rows,
        "task_phase_rows": task_phase_rows,
        "phase_clock_monotonic": monotonic,
        "gpu_receipts": [
            {"model_id": row["model_id"], **dict(row.get("gpu_receipts", {}))}
            for row in normalized_rows
        ],
        "accelerator_receipt_ready": receipt_ready,
        "claim_boundary": CLAIM_BOUNDARY,
        "gate_check_summary": failed_checks
        if failed_checks
        else [dict(check) for check in preflight.get("checks", [])],
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute all derived claims and reject drift or forged receipts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing_field:{field}" for field in missing)
    if missing:
        return errors
    if set(dict(artifact["field_principles"])) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("claim_boundary_mismatch")
    expected_models = [spec["model_id"] for spec in MODEL_SPECS]
    if artifact.get("models_used") != expected_models:
        errors.append("models_used_mismatch")
    rows = list(artifact.get("rows", []))
    if [row.get("model_id") for row in rows] != expected_models:
        errors.append("row_model_order_mismatch")
    for row in rows:
        # A blocked row can truthfully contain failed runtime checks. The row
        # hash must still bind that negative evidence without turning an honest
        # blocked artifact into a schema failure.
        if row.get("row_sha256") != row_checksum(row):
            errors.append(f"row_invalid:{row.get('model_id')}")
    phases = list(artifact.get("task_phase_rows", []))
    phase_error_rows = phase_errors(phases, rows)
    derived_monotonic = not phase_error_rows
    if artifact.get("phase_clock_monotonic") is not derived_monotonic:
        errors.append("phase_clock_boolean_mismatch")
    if phases:
        duration = round(
            (int(phases[-1]["monotonic_ns"]) - int(phases[0]["monotonic_ns"])) / 1_000_000_000,
            9,
        )
        if artifact.get("duration_s") != duration or duration <= 0:
            errors.append("duration_mismatch")
    else:
        errors.append("duration_mismatch")
    expected_live = any(row.get("decode_completed") is True for row in rows)
    if artifact.get("live_model_invoked") is not expected_live:
        errors.append("live_model_invoked_mismatch")
    derived_ready = derived_monotonic and reduce_accelerator_readiness(rows)
    if artifact.get("accelerator_receipt_ready") is not derived_ready:
        errors.append("accelerator_receipt_ready_mismatch")
    expected_gpu = [
        {"model_id": row["model_id"], **dict(row.get("gpu_receipts", {}))} for row in rows
    ]
    if artifact.get("gpu_receipts") != expected_gpu:
        errors.append("gpu_receipts_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(rows):
        errors.append("reproducibility_checksum_mismatch")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if derived_ready:
        if verdict_class != "positive" or not verdict.startswith("complete"):
            errors.append("positive_verdict_mismatch")
    elif expected_live:
        if verdict_class != "partial" or not verdict.startswith(
            "complete_blocked_accelerator_canary"
        ):
            errors.append("partial_verdict_mismatch")
    elif verdict_class != "blocked" or not verdict.startswith(
        "complete_blocked_accelerator_canary"
    ):
        errors.append("blocked_verdict_mismatch")
    if not derived_ready:
        summary = list(artifact.get("gate_check_summary", []))
        if not summary or any("observed" not in row for row in summary):
            errors.append("blocked_gate_summary_missing_observed")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write through a same-directory temporary file before atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run(
    *,
    result_path: Path = RESULT_PATH,
    resolver: Callable[[], list[JsonDict]] = resolve_model_specs,
    preflight_fn: Callable[[list[JsonDict]], JsonDict] = live_preflight,
    model_runner: Callable[[Mapping[str, Any]], JsonDict] = run_live_model,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Resolve, preflight, run sequential workers, and write one artifact."""

    started_ns = clock()
    phase_rows: list[JsonDict] = [
        {"phase": "preflight", "monotonic_ns": started_ns, "model_id": None}
    ]
    models = resolver()
    for model in models:
        phase_rows.append(
            {
                "phase": "cache_resolved",
                "monotonic_ns": clock(),
                "model_id": model["model_id"],
            }
        )
    preflight = preflight_fn(models)
    rows: list[JsonDict] = []
    if preflight.get("all_passed") is True:
        for model in models:
            try:
                row = model_runner(model)
            except Exception as exc:
                row = _blocked_model_row(model, f"{type(exc).__name__}: {exc}")
            rows.append(row)
            clocks = dict(row.get("clocks", {}))
            for phase in MODEL_PHASES:
                value = clocks.get(f"{phase}_ns")
                if isinstance(value, int) and not isinstance(value, bool):
                    phase_rows.append(
                        {
                            "phase": phase,
                            "monotonic_ns": value,
                            "model_id": model["model_id"],
                        }
                    )
    else:
        rows = [_blocked_model_row(model, "preflight_failed") for model in models]
    artifact_write_ns = clock()
    phase_rows.append(
        {"phase": "artifact_write", "monotonic_ns": artifact_write_ns, "model_id": None}
    )
    artifact = build_artifact(
        rows=rows,
        task_phase_rows=phase_rows,
        preflight=preflight,
        started_ns=started_ns,
        artifact_write_ns=artifact_write_ns,
    )
    write_artifact(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canary, validate an artifact, or serve one worker subprocess."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    if args.worker is not None:
        return worker_main(args.worker)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return 0 if not errors else 1
    artifact = run(result_path=args.result_path)
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "path": str(args.result_path),
                "accelerator_receipt_ready": artifact["accelerator_receipt_ready"],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
