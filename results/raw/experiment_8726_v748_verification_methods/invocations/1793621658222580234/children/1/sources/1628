#!/usr/bin/env python3
"""Exp 5309: SOTA GGUF runtime timeout root-cause matrix.

Spec refs: REQ-VERIFY-5309, SCENARIO-VERIFY-5309.

This module is a runtime unblock check only. It records whether the local
llama.cpp-compatible GGUF path can load, first-token, and generate eight tokens
from the mandated SOTA GGUF models with authenticated GPU offload. It makes no
model-quality, verifier-quality, solver-quality, memory, or benchmark claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import struct
import subprocess
import time
import traceback
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
GpuBackendProvider = Callable[[], JsonDict]
RuntimeProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5309_sota_runtime_timeout_rootcause_matrix_v485"
MILESTONE = "2026.07.485"
RESULT_RELATIVE_PATH = Path("results/experiment_5309_sota_runtime_timeout_rootcause_matrix_v485.json")
SCHEMA = "carnot.experiment_5309.sota_runtime_timeout_rootcause_matrix.v485"
INFERENCE_SUBSTRATE = "local_llama_cpp_gguf_runtime"
SPEC_REFS = ("REQ-VERIFY-5309", "SCENARIO-VERIFY-5309")
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
RANDOM_SEED = 5309
DEFAULT_RUNTIME_TIMEOUT_S = 180.0
PROMPT = "Return exactly OK."

RUNTIME_CONFIG: JsonDict = {
    "n_gpu_layers": "all",
    "context_size": 512,
    "batch_size": 1,
    "max_tokens": 8,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
    "split_mode": "layer",
    "fallback": "none_no_tiny_or_transformers_fallback",
}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": DEFAULT_PREFERRED_QUANT,
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5309 SOTA GGUF runtime timeout root-cause matrix.",
    "milestone": "Milestone accountability for the V485 runtime unblock decision.",
    "status": "Machine-readable terminal state for downstream gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether runtime, "
        "not quality, is unblocked."
    ),
    "inference_substrate": (
        "Declares local_llama_cpp_gguf_runtime so the artifact is read as a runtime "
        "matrix, not a quality or verifier claim."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF repository IDs and concrete local paths "
        "without AutoTokenizer fallback."
    ),
    "preconditions_checked": (
        "Records GPU, VRAM, backend command, offload flags, context, batch, cache, "
        "disk, and fallback checks before runtime interpretation."
    ),
    "gpu_backend_evidence": (
        "Records CUDA/device/build/log/memory evidence needed to distinguish real GPU "
        "offload from CPU-only runtime."
    ),
    "per_model_runtime_matrix": (
        "Separates resolution, load, prompt ingestion, first-token, bounded generation, "
        "offload, and timeout class per mandated model."
    ),
    "timeout_root_cause": (
        "Names the next concrete blocker when no mandated model completes the bounded "
        "runtime path."
    ),
    "tests_run": (
        "Commands run to validate the matrix module, artifact schema, new-code coverage, "
        "and relevant runtime checks."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "gpu_backend_evidence",
    "per_model_runtime_matrix",
    "timeout_root_cause",
    "sota_runtime_unblocked",
    "no_quality_claim",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "gpu_backend_evidence",
    "per_model_runtime_matrix",
    "timeout_root_cause",
    "tests_run",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")
CUDA_EVIDENCE_RE = re.compile(
    r"(CUDA\d|ggml_cuda|libggml-cuda|libcuda|cublas|offloaded\s+\d+/\d+\s+layers|to GPU)",
    re.IGNORECASE,
)
PROMPT_EVAL_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
LOAD_TIME_RE = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
EVAL_TIME_RE = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_run_date() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _repo_cache_path(hf_id: str) -> str:
    return str(
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"
    )


def read_gguf_header(model_path: str | Path) -> JsonDict:
    path = Path(model_path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24:
        raise ValueError("truncated GGUF header")
    if header[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", header[4:24])
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    return {
        "magic": "GGUF",
        "version": int(version),
        "tensor_count": int(tensor_count),
        "metadata_kv_count": int(metadata_kv_count),
    }


def _file_receipts(path: Path) -> JsonDict:
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
    return {
        "path": str(path),
        "size_bytes": size,
        "checksum_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        if size <= 64 * 1024 * 1024
        else None,
        "checksum_head_1m_sha256": hashlib.sha256(head).hexdigest(),
        "checksum_note": (
            "full_sha256_recorded"
            if size <= 64 * 1024 * 1024
            else "full_sha256_skipped_for_large_file_head_1m_recorded"
        ),
    }


def _missing_model_spec(spec: Mapping[str, Any], resolution_s: float = 0.0) -> JsonDict:
    hf_id = str(spec["hf_id"])
    return {
        "role": str(spec["role"]),
        "hf_id": hf_id,
        "quantization": str(spec.get("quantization", DEFAULT_PREFERRED_QUANT)),
        "cache_path": _repo_cache_path(hf_id),
        "model_path": None,
        "status": "missing_local_gguf",
        "autotokenizer_used": False,
        "metadata_model_resolution_s": round(resolution_s, 6),
        "file_receipts": None,
        "metadata": None,
        "blocked_preconditions": [],
    }


def _resolve_model_spec(spec: Mapping[str, Any], model_resolver: ModelResolver) -> JsonDict:
    started = time.perf_counter()
    path_text = model_resolver(
        str(spec["hf_id"]), str(spec.get("quantization", DEFAULT_PREFERRED_QUANT))
    )
    resolution_s = time.perf_counter() - started
    receipt = _missing_model_spec(spec, resolution_s)
    if not path_text:
        return receipt
    path = Path(path_text)
    receipt["model_path"] = str(path)
    try:
        receipt["file_receipts"] = _file_receipts(path)
        receipt["metadata"] = read_gguf_header(path)
        receipt["status"] = "local_gguf_resolved"
    except Exception as exc:
        receipt["status"] = "blocked_metadata_unreadable"
        receipt["blocked_preconditions"] = [f"metadata_unreadable:{type(exc).__name__}: {exc}"]
        receipt["traceback"] = traceback.format_exc()
    return receipt


def _run_command(command: Sequence[str], timeout_s: float = 20.0) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _candidate_llama_cli_paths() -> list[Path]:  # pragma: no cover
    candidates: list[Path] = []
    env_path = os.environ.get("CARNOT_LLAMA_CPP_CLI")
    if env_path:
        candidates.append(Path(env_path))
    which = shutil.which("llama-cli")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli",
            Path.home() / "github.com" / "ggml-org" / "llama.cpp" / "build" / "bin" / "llama-cli",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate.expanduser())
        if text not in seen:
            seen.add(text)
            unique.append(Path(text))
    return unique


def _first_executable_llama_cli() -> Path | None:  # pragma: no cover
    for candidate in _candidate_llama_cli_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _gpu_snapshot() -> list[JsonDict]:  # pragma: no cover
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_free_mb": int(float(parts[3])),
                    "utilization_gpu_pct": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return rows


def _total_used_mb(snapshot: Sequence[Mapping[str, Any]] | None) -> int:  # pragma: no cover
    return sum(int(row.get("memory_used_mb", 0)) for row in snapshot or [])


def collect_gpu_backend_evidence() -> JsonDict:  # pragma: no cover
    backend = _first_executable_llama_cli()
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    backend_version = (
        _run_command([str(backend), "--version"], timeout_s=20.0)
        if backend
        else {"ok": False, "stderr": "llama-cli not found"}
    )
    backend_devices = (
        _run_command([str(backend), "--list-devices"], timeout_s=20.0)
        if backend
        else {"ok": False, "stderr": "llama-cli not found"}
    )
    backend_dynamic_libraries = (
        _run_command(["ldd", str(backend)], timeout_s=20.0)
        if backend
        else {"ok": False, "stderr": "llama-cli not found"}
    )
    evidence_text = "\n".join(
        [
            str(backend_version.get("stdout", "")),
            str(backend_version.get("stderr", "")),
            str(backend_devices.get("stdout", "")),
            str(backend_devices.get("stderr", "")),
            str(backend_dynamic_libraries.get("stdout", "")),
            str(backend_dynamic_libraries.get("stderr", "")),
        ]
    )
    vram_before = _gpu_snapshot()
    return {
        "gpu_visible": bool(nvidia_smi.get("ok") or vram_before),
        "vram_before": vram_before,
        "vram_after": None,
        "nvidia_smi": nvidia_smi,
        "backend_kind": "native_llama_cpp_cli",
        "backend_command": str(backend) if backend else None,
        "backend_version": backend_version,
        "backend_devices": backend_devices,
        "backend_dynamic_libraries": {
            **backend_dynamic_libraries,
            "stdout": str(backend_dynamic_libraries.get("stdout", ""))[-4000:],
        },
        "cuda_backend_evidence": bool(CUDA_EVIDENCE_RE.search(evidence_text)),
    }


def _nonblocking_read(
    proc: subprocess.Popen[bytes],
    timeout_s: float,
) -> tuple[bytes, bytes, float | None, bool]:  # pragma: no cover
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    selector.register(proc.stderr, selectors.EVENT_READ)
    started = time.perf_counter()
    first_stdout_s: float | None = None
    stdout = bytearray()
    stderr = bytearray()
    timed_out = False
    while selector.get_map():
        if proc.poll() is None and time.perf_counter() - started > timeout_s:
            proc.kill()
            timed_out = True
        events = selector.select(timeout=0.25)
        if not events and proc.poll() is not None:
            for key in list(selector.get_map().values()):
                chunk = key.fileobj.read()
                if chunk:
                    if key.fileobj is proc.stdout:
                        if first_stdout_s is None:
                            first_stdout_s = time.perf_counter() - started
                        stdout.extend(chunk)
                    else:
                        stderr.extend(chunk)
                selector.unregister(key.fileobj)
            break
        for key, _event in events:
            chunk = key.fileobj.read1(8192)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if key.fileobj is proc.stdout:
                if first_stdout_s is None:
                    first_stdout_s = time.perf_counter() - started
                stdout.extend(chunk)
            else:
                stderr.extend(chunk)
        if proc.poll() is not None and not events:
            break
    return bytes(stdout), bytes(stderr), first_stdout_s, timed_out


def default_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    backend: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    timeout_s: float = DEFAULT_RUNTIME_TIMEOUT_S,
) -> JsonDict:  # pragma: no cover
    command = [
        str(backend["backend_command"]),
        "-m",
        str(model_spec["model_path"]),
        "-p",
        PROMPT,
        "-n",
        str(runtime_config["max_tokens"]),
        "-c",
        str(runtime_config["context_size"]),
        "-b",
        str(runtime_config["batch_size"]),
        "-ngl",
        str(runtime_config["n_gpu_layers"]),
        "--temp",
        str(runtime_config["temperature"]),
        "--seed",
        str(runtime_config["seed"]),
        "--no-display-prompt",
        "--simple-io",
        "--no-conversation",
    ]
    started = time.perf_counter()
    before = _gpu_snapshot()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    selector.register(proc.stderr, selectors.EVENT_READ)
    samples: list[list[JsonDict]] = [before]
    first_offload_s: float | None = None
    first_stdout_s: float | None = None
    stdout = bytearray()
    stderr = bytearray()
    timed_out = False
    while selector.get_map():
        elapsed = time.perf_counter() - started
        if proc.poll() is None and elapsed > timeout_s:
            proc.kill()
            timed_out = True
        snap = _gpu_snapshot()
        samples.append(snap)
        if first_offload_s is None and _total_used_mb(snap) - _total_used_mb(before) > 128:
            first_offload_s = time.perf_counter() - started
        events = selector.select(timeout=0.25)
        if not events and proc.poll() is not None:
            for key in list(selector.get_map().values()):
                chunk = key.fileobj.read()
                if chunk:
                    if key.fileobj is proc.stdout:
                        if first_stdout_s is None:
                            first_stdout_s = time.perf_counter() - started
                        stdout.extend(chunk)
                    else:
                        stderr.extend(chunk)
                selector.unregister(key.fileobj)
            break
        for key, _event in events:
            chunk = key.fileobj.read1(8192)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if key.fileobj is proc.stdout:
                if first_stdout_s is None:
                    first_stdout_s = time.perf_counter() - started
                stdout.extend(chunk)
            else:
                stderr.extend(chunk)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        timed_out = True
    after = _gpu_snapshot()
    samples.append(after)
    wall_clock_s = time.perf_counter() - started
    stdout_text = bytes(stdout).decode("utf-8", "replace")
    stderr_text = bytes(stderr).decode("utf-8", "replace")
    log_text = f"{stdout_text}\n{stderr_text}"
    max_used = max((_total_used_mb(sample) for sample in samples), default=0)
    max_delta = max(0, max_used - _total_used_mb(before))
    backend_log_evidence = bool(CUDA_EVIDENCE_RE.search(log_text))
    prompt_eval_s = _extract_ms(PROMPT_EVAL_RE, log_text)
    load_s = _extract_ms(LOAD_TIME_RE, log_text)
    generation_s = _extract_ms(EVAL_TIME_RE, log_text)
    token_count = min(int(runtime_config["max_tokens"]), len(stdout_text.strip().split()))
    completed = proc.returncode == 0 and not timed_out and token_count >= int(runtime_config["max_tokens"])
    return {
        "status": "completed" if completed else "timeout_or_generation_incomplete",
        "timeout_class": _classify_probe_timeout(
            completed=completed,
            timed_out=timed_out,
            first_token_latency_s=first_stdout_s,
            generated_token_count=token_count,
        ),
        "completed": completed,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "wall_clock_s": round(wall_clock_s, 6),
        "load_s": load_s,
        "gpu_offload_evidence_s": first_offload_s,
        "prompt_ingestion_s": prompt_eval_s,
        "first_token_latency_s": first_stdout_s,
        "eight_token_generation_s": generation_s,
        "generated_token_count": token_count,
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-4000:],
        "backend_gpu_log_evidence": backend_log_evidence,
        "command": command,
        "config": dict(runtime_config),
        "returncode": proc.returncode,
        "gpu_memory_receipts": {
            "before": before,
            "during": samples[1:-1],
            "after": after,
            "max_memory_delta_mb": max_delta,
            "offload_evidence": bool(max_delta > 128 or backend_log_evidence),
        },
    }


def _extract_ms(pattern: re.Pattern[str], text: str) -> float | None:  # pragma: no cover
    match = pattern.search(text)
    if not match:
        return None
    return round(float(match.group(1)) / 1000.0, 6)


def _classify_probe_timeout(
    *,
    completed: bool,
    timed_out: bool,
    first_token_latency_s: float | None,
    generated_token_count: int,
) -> str:
    if completed:
        return "completed_no_timeout"
    if timed_out and first_token_latency_s is None:
        return "timeout_before_first_token"
    if timed_out and generated_token_count < 8:
        return "timeout_during_8_token_generation"
    if first_token_latency_s is None:
        return "no_first_token"
    return "generation_incomplete"


def _precondition_blockers(
    gpu_backend: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
) -> list[str]:
    blockers: list[str] = []
    if not gpu_backend.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if not gpu_backend.get("backend_command"):
        blockers.append("backend_command_missing")
    elif not gpu_backend.get("cuda_backend_evidence"):
        blockers.append("cuda_backend_evidence_missing")
    if not any(row.get("status") == "local_gguf_resolved" for row in model_specs.values()):
        blockers.append("no_mandated_sota_gguf_resolved")
    return blockers


def _normalise_gpu_memory_receipts(value: Any) -> JsonDict:
    out = dict(value) if isinstance(value, Mapping) else {}
    out.setdefault("before", None)
    out.setdefault("during", [])
    out.setdefault("after", None)
    out.setdefault("max_memory_delta_mb", 0)
    out.setdefault("offload_evidence", False)
    return out


def _offload_authenticated(row: Mapping[str, Any]) -> bool:
    gpu_memory = _normalise_gpu_memory_receipts(row.get("gpu_memory_receipts"))
    log_text = f"{row.get('stdout_tail', '')}\n{row.get('stderr_tail', '')}"
    return bool(
        row.get("backend_gpu_log_evidence")
        or gpu_memory.get("offload_evidence")
        or int(gpu_memory.get("max_memory_delta_mb") or 0) > 128
        or CUDA_EVIDENCE_RE.search(log_text)
    )


def _not_attempted_matrix_row(
    *,
    model_spec: Mapping[str, Any],
    timeout_class: str,
    blocked_preconditions: Sequence[str],
    backend: Mapping[str, Any],
) -> JsonDict:
    return {
        "role": str(model_spec["role"]),
        "hf_id": str(model_spec["hf_id"]),
        "model_path": model_spec.get("model_path"),
        "model_status": str(model_spec.get("status")),
        "model_available": bool(model_spec.get("status") == "local_gguf_resolved"),
        "autotokenizer_used": False,
        "metadata_model_resolution_s": float(model_spec.get("metadata_model_resolution_s") or 0.0),
        "load_s": None,
        "gpu_offload_evidence_s": None,
        "prompt_ingestion_s": None,
        "first_token_latency_s": None,
        "eight_token_generation_s": None,
        "generated_token_count": 0,
        "timeout_class": timeout_class,
        "completed_load_first_token_and_8_tokens": False,
        "offload_authenticated": False,
        "backend_command": backend.get("backend_command"),
        "context_size": RUNTIME_CONFIG["context_size"],
        "batch_size": RUNTIME_CONFIG["batch_size"],
        "n_gpu_layers": RUNTIME_CONFIG["n_gpu_layers"],
        "fallback": RUNTIME_CONFIG["fallback"],
        "blocked_preconditions": list(blocked_preconditions),
        "gpu_memory_receipts": _normalise_gpu_memory_receipts(None),
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _normalise_runtime_matrix_row(
    *,
    model_spec: Mapping[str, Any],
    backend: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> JsonDict:
    gpu_memory = _normalise_gpu_memory_receipts(receipt.get("gpu_memory_receipts"))
    generated_token_count = int(receipt.get("generated_token_count") or 0)
    first_token_latency_s = _optional_float(receipt.get("first_token_latency_s"))
    completed_path = bool(
        receipt.get("completed")
        and not receipt.get("timed_out")
        and first_token_latency_s is not None
        and generated_token_count >= 8
    )
    row = {
        "role": str(model_spec["role"]),
        "hf_id": str(model_spec["hf_id"]),
        "model_path": str(model_spec["model_path"]),
        "model_status": str(model_spec.get("status")),
        "model_available": True,
        "autotokenizer_used": False,
        "metadata_model_resolution_s": float(model_spec.get("metadata_model_resolution_s") or 0.0),
        "load_s": _optional_float(receipt.get("load_s")),
        "gpu_offload_evidence_s": _optional_float(receipt.get("gpu_offload_evidence_s")),
        "prompt_ingestion_s": _optional_float(receipt.get("prompt_ingestion_s")),
        "first_token_latency_s": first_token_latency_s,
        "eight_token_generation_s": _optional_float(receipt.get("eight_token_generation_s")),
        "generated_token_count": generated_token_count,
        "timeout_class": str(
            receipt.get("timeout_class")
            or _classify_probe_timeout(
                completed=completed_path,
                timed_out=bool(receipt.get("timed_out")),
                first_token_latency_s=first_token_latency_s,
                generated_token_count=generated_token_count,
            )
        ),
        "completed_load_first_token_and_8_tokens": completed_path,
        "offload_authenticated": _offload_authenticated(receipt),
        "backend_command": backend.get("backend_command"),
        "context_size": int(receipt.get("config", {}).get("context_size", RUNTIME_CONFIG["context_size"])),
        "batch_size": int(receipt.get("config", {}).get("batch_size", RUNTIME_CONFIG["batch_size"])),
        "n_gpu_layers": receipt.get("config", {}).get("n_gpu_layers", RUNTIME_CONFIG["n_gpu_layers"]),
        "fallback": RUNTIME_CONFIG["fallback"],
        "timeout_s": receipt.get("timeout_s"),
        "wall_clock_s": _optional_float(receipt.get("wall_clock_s")),
        "command": receipt.get("command"),
        "returncode": receipt.get("returncode"),
        "gpu_memory_receipts": gpu_memory,
        "stdout_tail": str(receipt.get("stdout_tail") or "")[-2000:],
        "stderr_tail": str(receipt.get("stderr_tail") or "")[-4000:],
        "blocked_preconditions": [],
    }
    return row


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_runtime_matrix(
    *,
    model_specs: Mapping[str, JsonDict],
    backend: Mapping[str, Any],
    runtime_probe: RuntimeProbe,
    precondition_blockers: Sequence[str],
) -> JsonDict:
    matrix: JsonDict = {}
    for role, model_spec in model_specs.items():
        if model_spec.get("status") != "local_gguf_resolved" or not model_spec.get("model_path"):
            matrix[role] = _not_attempted_matrix_row(
                model_spec=model_spec,
                timeout_class="not_available",
                blocked_preconditions=model_spec.get("blocked_preconditions") or [],
                backend=backend,
            )
            continue
        if precondition_blockers:
            matrix[role] = _not_attempted_matrix_row(
                model_spec=model_spec,
                timeout_class="not_attempted_preconditions_failed",
                blocked_preconditions=precondition_blockers,
                backend=backend,
            )
            continue
        try:
            receipt = runtime_probe(
                model_spec=model_spec,
                backend=backend,
                runtime_config=RUNTIME_CONFIG,
            )
        except Exception as exc:  # pragma: no cover
            receipt = {
                "completed": False,
                "timed_out": False,
                "timeout_class": "runtime_probe_exception",
                "generated_token_count": 0,
                "stderr_tail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "gpu_memory_receipts": {"offload_evidence": False, "max_memory_delta_mb": 0},
            }
        matrix[role] = _normalise_runtime_matrix_row(
            model_spec=model_spec,
            backend=backend,
            receipt=receipt,
        )
    return matrix


def _row_unblocked(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("timeout_class") == "completed_no_timeout"
        and row.get("completed_load_first_token_and_8_tokens")
        and row.get("offload_authenticated")
    )


def _timeout_root_cause(
    *,
    matrix: Mapping[str, JsonDict],
    precondition_blockers: Sequence[str],
) -> str:
    if any(_row_unblocked(row) for row in matrix.values()):
        return "none"
    if precondition_blockers:
        return f"preconditions_failed:{','.join(precondition_blockers)}"
    available = [row for row in matrix.values() if row.get("model_available")]
    if not available:
        return "no_mandated_sota_gguf_available"
    classes = [str(row.get("timeout_class")) for row in available]
    native_abort_causes = sorted(
        {cause for row in available if (cause := _native_runtime_abort_cause(row))}
    )
    if native_abort_causes:
        return (
            "native_llama_cpp_generation_abort_after_authenticated_offload:"
            f"{','.join(native_abort_causes)}; next_root_cause="
            "the native llama.cpp CLI invocation reaches GPU offload but aborts before a "
            "complete bounded generation path; replace the unsupported llama-cli "
            "--no-conversation call with a current llama-completion-compatible prompt path "
            "or adjust batch/prompt handling before downstream quality smoke tests"
        )
    if classes and all(cls == "timeout_before_first_token" for cls in classes):
        return (
            "all_mandated_models_timeout_before_first_token; next_root_cause="
            "native llama.cpp CUDA offloads weights but does not reach first token "
            "within the bounded probe timeout"
        )
    if any(
        row.get("completed_load_first_token_and_8_tokens") and not row.get("offload_authenticated")
        for row in available
    ):
        return "no_authenticated_gpu_offload_for_completed_bounded_generation"
    if any(cls == "timeout_during_8_token_generation" for cls in classes):
        return "timeout_during_bounded_8_token_generation"
    if any(cls == "no_first_token" for cls in classes):
        return "no_first_token_observed"
    return f"no_mandated_model_completed_load_first_token_and_8_tokens:{','.join(classes)}"


def _native_runtime_abort_cause(row: Mapping[str, Any]) -> str | None:
    if row.get("returncode") not in {-6, 134}:
        return None
    log_text = f"{row.get('stdout_tail', '')}\n{row.get('stderr_tail', '')}"
    no_conversation = "--no-conversation is not supported by llama-cli" in log_text
    batch_assert = (
        "GGML_ASSERT(n_tokens_all <= cparams.n_batch)" in log_text
        or "llama-context.cpp:1712" in log_text
    )
    if no_conversation and batch_assert:
        return "llama_cli_no_conversation_unsupported_plus_batch_assert"
    if no_conversation:
        return "llama_cli_no_conversation_unsupported"
    if batch_assert:
        return "llama_context_batch_assert"
    return "native_llama_cpp_abort_signal"


def _preconditions(
    *,
    root: Path,
    backend: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    blockers: Sequence[str],
) -> JsonDict:
    total, used, free = shutil.disk_usage(root)
    return _wrap(
        "preconditions_checked",
        {
            "run_date_utc": _utc_run_date(),
            "run_on_or_after_20260706_utc": _utc_run_date() >= "20260706",
            "gpu_visibility_checked": True,
            "gpu_visible": bool(backend.get("gpu_visible")),
            "vram_before": backend.get("vram_before"),
            "vram_after": backend.get("vram_after"),
            "backend_command": backend.get("backend_command"),
            "backend_kind": backend.get("backend_kind"),
            "backend_version": backend.get("backend_version"),
            "backend_devices": backend.get("backend_devices"),
            "offload_flags": {
                "n_gpu_layers": RUNTIME_CONFIG["n_gpu_layers"],
                "split_mode": RUNTIME_CONFIG["split_mode"],
            },
            "context_size": RUNTIME_CONFIG["context_size"],
            "batch_size": RUNTIME_CONFIG["batch_size"],
            "fallback": RUNTIME_CONFIG["fallback"],
            "free_disk": {
                "path": str(root),
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
            },
            "gguf_cache_paths": {
                spec["role"]: _repo_cache_path(str(spec["hf_id"])) for spec in MANDATED_MODEL_SPECS
            },
            "resolved_roles": [
                role for role, row in model_specs.items() if row.get("status") == "local_gguf_resolved"
            ],
            "autotokenizer_used": False,
            "blocked_preconditions": list(blockers),
        },
    )


def _gpu_backend_evidence(backend: Mapping[str, Any]) -> JsonDict:
    return _wrap(
        "gpu_backend_evidence",
        {
            **dict(backend),
            "runtime_config": dict(RUNTIME_CONFIG),
            "cuda_evidence_pattern": CUDA_EVIDENCE_RE.pattern,
        },
    )


def build_artifact(
    *,
    root: Path,
    backend: JsonDict,
    model_specs: Mapping[str, JsonDict],
    matrix: Mapping[str, JsonDict],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    precondition_blockers: Sequence[str],
) -> JsonDict:
    root_cause = _timeout_root_cause(matrix=matrix, precondition_blockers=precondition_blockers)
    unblocked = root_cause == "none"
    status = "complete" if unblocked else "blocked"
    ready_roles = [role for role, row in matrix.items() if _row_unblocked(row)]
    verdict = (
        f"complete: sota_runtime_unblocked=true via {','.join(ready_roles)}"
        if unblocked
        else f"blocked_sota_runtime_unblocked_false: {root_cause}"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "preconditions_checked": _preconditions(
            root=root,
            backend=backend,
            model_specs=model_specs,
            blockers=precondition_blockers,
        ),
        "gpu_backend_evidence": _gpu_backend_evidence(backend),
        "per_model_runtime_matrix": _wrap("per_model_runtime_matrix", dict(matrix)),
        "timeout_root_cause": _wrap("timeout_root_cause", root_cause),
        "sota_runtime_unblocked": unblocked,
        "no_quality_claim": True,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "spec_refs": SPEC_REFS,
                "model_specs": artifact["MODEL_SPECS"]["value"],
                "matrix": artifact["per_model_runtime_matrix"]["value"],
                "root_cause": root_cause,
            }
        )
    )
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, Mapping) or "value" not in value or "principle" not in value:
            errors.append(f"{field} must be principle-wrapped")
    if _wrapped_value(artifact, "experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if _wrapped_value(artifact, "milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if _wrapped_value(artifact, "status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    verdict = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("sota_runtime_unblocked"), bool):
        errors.append("sota_runtime_unblocked must be a bare bool")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")
    tests_run = _wrapped_value(artifact, "tests_run")
    if not isinstance(tests_run, list):
        errors.append("tests_run.value must be a list")
    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    matrix = _wrapped_value(artifact, "per_model_runtime_matrix")
    if isinstance(model_specs, Mapping) and isinstance(matrix, Mapping):
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles:
            errors.append("MODEL_SPECS roles mismatch")
        if set(matrix) != expected_roles:
            errors.append("per_model_runtime_matrix roles mismatch")
        for spec in MANDATED_MODEL_SPECS:
            role = str(spec["role"])
            row = model_specs.get(role, {})
            runtime_row = matrix.get(role, {})
            if row.get("hf_id") != spec["hf_id"]:
                errors.append(f"MODEL_SPECS.{role}.hf_id mismatch")
            if row.get("autotokenizer_used") is not False:
                errors.append(f"MODEL_SPECS.{role}.autotokenizer_used must be false")
            if runtime_row.get("autotokenizer_used") is not False:
                errors.append(f"per_model_runtime_matrix.{role}.autotokenizer_used must be false")
            if runtime_row.get("context_size") != RUNTIME_CONFIG["context_size"]:
                errors.append(f"per_model_runtime_matrix.{role}.context_size mismatch")
            if runtime_row.get("batch_size") != RUNTIME_CONFIG["batch_size"]:
                errors.append(f"per_model_runtime_matrix.{role}.batch_size mismatch")
    else:
        errors.append("MODEL_SPECS.value and per_model_runtime_matrix.value must be objects")
    root_cause = _wrapped_value(artifact, "timeout_root_cause")
    unblocked = artifact.get("sota_runtime_unblocked")
    if unblocked is True and root_cause != "none":
        errors.append("unblocked artifact must have timeout_root_cause.value=none")
    if unblocked is False and (not isinstance(root_cause, str) or not root_cause):
        errors.append("blocked artifact must name timeout_root_cause")
    if unblocked is True and isinstance(matrix, Mapping):
        if not any(_row_unblocked(row) for row in matrix.values()):
            errors.append("unblocked artifact must have at least one complete offloaded row")
    if unblocked is False and isinstance(matrix, Mapping):
        if any(_row_unblocked(row) for row in matrix.values()):
            errors.append("blocked artifact cannot contain a complete offloaded row")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if isinstance(value, Mapping):
        return value.get("value")
    return None


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    model_resolver: ModelResolver = resolve_cached_gguf,
    gpu_backend_provider: GpuBackendProvider = collect_gpu_backend_evidence,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    artifact_path = Path(artifact_path)
    backend = gpu_backend_provider()
    model_specs = {
        str(spec["role"]): _resolve_model_spec(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }
    blockers = _precondition_blockers(backend, model_specs)
    matrix = _build_runtime_matrix(
        model_specs=model_specs,
        backend=backend,
        runtime_probe=runtime_probe,
        precondition_blockers=blockers,
    )
    if backend.get("vram_after") is None:
        backend["vram_after"] = _gpu_snapshot()  # pragma: no cover
    artifact = build_artifact(
        root=root,
        backend=backend,
        model_specs=model_specs,
        matrix=matrix,
        tests_run=tests_run or [],
        duration_s=time.perf_counter() - started,
        precondition_blockers=blockers,
    )
    validate_artifact(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _load_tests_run_argument(value: str | None) -> list[JsonDict]:  # pragma: no cover
    if not value:
        return [
            {
                "command": (
                    ".venv/bin/pytest tests/python/"
                    "test_experiment_5309_sota_runtime_timeout_rootcause_matrix_v485.py -q"
                ),
                "outcome": "not_run_in_module_invocation",
            }
        ]
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("--tests-run-json must decode to a list")
    return [dict(row) for row in parsed]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--tests-run-json", default=None)
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=Path(args.output),
        tests_run=_load_tests_run_argument(args.tests_run_json),
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
