#!/usr/bin/env python3
"""Exp 5323: native llama.cpp GGUF backend and flag bisect.

Spec refs: REQ-VERIFY-5323, SCENARIO-VERIFY-5323.

This module is a runtime-substrate check only. It records whether native
llama.cpp binaries can load one of the mandated local SOTA GGUF models, emit a
first token, and complete a bounded 8-token generation with authenticated GPU
offload. It deliberately makes no model-quality, verifier-quality, benchmark,
solver, memory, or answer-accuracy claim.
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
import socket
import struct
import subprocess
import time
import traceback
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
PreconditionsProvider = Callable[[], JsonDict]
RuntimeProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5323_native_gguf_backend_flag_bisect_v486"
MILESTONE = "2026.07.486"
RESULT_RELATIVE_PATH = Path("results/experiment_5323_native_gguf_backend_flag_bisect_v486.json")
SCHEMA = "carnot.experiment_5323.native_gguf_backend_flag_bisect.v486"
INFERENCE_SUBSTRATE = "local_native_llama_cpp_gguf_backend_bisect"
SPEC_REFS = ("REQ-VERIFY-5323", "SCENARIO-VERIFY-5323")

RANDOM_SEED = 5323
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
DEFAULT_TIMEOUT_S = 240.0
DEFAULT_CONTEXT = 512
DEFAULT_BATCH = 512
DEFAULT_UBATCH = 128
DEFAULT_GPU_LAYERS = "all"
DEFAULT_SPLIT_MODE = "layer"
DEFAULT_TENSOR_SPLIT: str | None = None
N_PREDICT = 8
PROMPT = "Write eight lowercase color words separated by spaces."
ATTEMPT_ROLE_ORDER = ("flagship_dense", "flagship_moe", "middle_moe")
NATIVE_BACKENDS = ("llama-cli", "llama-completion", "llama-server")
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

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
    "experiment_id": "Traceability for the Exp5323 native GGUF backend flag bisect.",
    "milestone": "Milestone accountability for the V486 native backend candidate decision.",
    "status": "Machine-readable terminal state for downstream runtime gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether a "
        "native bounded generation backend candidate exists."
    ),
    "inference_substrate": (
        "Declares local_native_llama_cpp_gguf_backend_bisect so the artifact is read "
        "as a native GGUF runtime bisect, not a quality or verifier claim."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF repository IDs and concrete local GGUF "
        "cache status without AutoTokenizer fallback."
    ),
    "preconditions_checked": (
        "Records GPU visibility, nvidia-smi, CUDA and driver facts, free VRAM, free "
        "disk, native binary paths, binary versions, dynamic-library evidence, and "
        "model cache status before runtime interpretation."
    ),
    "backend_matrix": (
        "Compares available native llama.cpp paths and generation modes, including "
        "attempted and skipped variants with exact flag choices."
    ),
    "per_model_runtime_matrix": (
        "Records each attempted mandated model's command, backend, model path, "
        "context, batch, ubatch, offload flags, prompt, n_predict, timeout, GPU "
        "memory delta, offload evidence, first-token time, and 8-token completion "
        "status."
    ),
    "best_backend_command": (
        "Records the exact reusable command that completed load, first token, and "
        "bounded 8-token generation with authenticated GPU offload, or null with a "
        "blocker reason."
    ),
    "timeout_or_crash_root_cause": (
        "Names the next concrete runtime blocker when no mandated model has a usable "
        "bounded generation receipt."
    ),
    "tests_run": (
        "Commands run to validate the bisect module, artifact schema, new-code "
        "coverage, and focused runtime checks."
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
    "backend_matrix",
    "per_model_runtime_matrix",
    "best_backend_command",
    "timeout_or_crash_root_cause",
    "sota_backend_candidate_ready",
    "runtime_unblocked_min_one_mandated",
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
    "backend_matrix",
    "per_model_runtime_matrix",
    "best_backend_command",
    "timeout_or_crash_root_cause",
    "tests_run",
)

CUDA_EVIDENCE_RE = re.compile(
    r"(CUDA\d|ggml_cuda|libggml-cuda|libcuda|cublas|offloaded\s+\d+/\d+\s+layers|to GPU)",
    re.IGNORECASE,
)
LOAD_TIME_RE = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
EVAL_TIME_RE = re.compile(r"eval time\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE)
EVAL_RUNS_RE = re.compile(r"eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s+runs", re.IGNORECASE)


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


def _utc_run_date() -> str:  # pragma: no cover - wall-clock metadata only
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
        "cached": False,
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
    receipt["cached"] = True
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


def _candidate_binary_paths(binary_name: str) -> list[Path]:  # pragma: no cover
    env_specific = os.environ.get(f"CARNOT_{binary_name.upper().replace('-', '_')}")
    candidates: list[Path] = []
    if env_specific:
        candidates.append(Path(env_specific))
    bin_dir = os.environ.get("CARNOT_LLAMA_CPP_BIN_DIR")
    if bin_dir:
        candidates.append(Path(bin_dir) / binary_name)
    which = shutil.which(binary_name)
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / binary_name,
            Path.home() / "github.com" / "ggml-org" / "llama.cpp" / "build" / "bin" / binary_name,
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


def _first_executable(binary_name: str) -> Path | None:  # pragma: no cover
    for candidate in _candidate_binary_paths(binary_name):
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


def _total_used_mb(snapshot: Sequence[Mapping[str, Any]] | None) -> int:
    return sum(int(row.get("memory_used_mb", 0)) for row in snapshot or [])


def _free_vram_mb(snapshot: Sequence[Mapping[str, Any]] | None) -> int:
    return sum(int(row.get("memory_free_mb", 0)) for row in snapshot or [])


def collect_preconditions(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    binary_paths: JsonDict = {}
    binary_versions: JsonDict = {}
    binary_dynamic_libraries: JsonDict = {}
    cuda_text_parts: list[str] = []
    for binary in NATIVE_BACKENDS:
        path = _first_executable(binary)
        binary_paths[binary] = str(path) if path else None
        if path:
            version = _run_command([str(path), "--version"], timeout_s=20.0)
            ldd = _run_command(["ldd", str(path)], timeout_s=20.0)
        else:
            version = {"ok": False, "stderr": f"{binary} not found", "stdout": ""}
            ldd = {"ok": False, "stderr": f"{binary} not found", "stdout": ""}
        binary_versions[binary] = version
        binary_dynamic_libraries[binary] = {
            **ldd,
            "stdout": str(ldd.get("stdout", ""))[-4000:],
        }
        cuda_text_parts.extend(
            [
                str(version.get("stdout", "")),
                str(version.get("stderr", "")),
                str(ldd.get("stdout", "")),
                str(ldd.get("stderr", "")),
            ]
        )

    raw_nvidia_smi = _run_command(["nvidia-smi"], timeout_s=10.0)
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    vram_before = _gpu_snapshot()
    disk = shutil.disk_usage(root)
    return {
        "run_date_utc": _utc_run_date(),
        "gpu_visible": bool(nvidia_smi.get("ok") or vram_before),
        "raw_nvidia_smi": raw_nvidia_smi,
        "nvidia_smi": nvidia_smi,
        "cuda_driver": _parse_cuda_driver(raw_nvidia_smi, nvidia_smi),
        "vram_before": vram_before,
        "free_vram_mb": _free_vram_mb(vram_before),
        "free_disk": {
            "path": str(root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "binary_paths": binary_paths,
        "binary_versions": binary_versions,
        "binary_dynamic_libraries": binary_dynamic_libraries,
        "cuda_backend_evidence": bool(CUDA_EVIDENCE_RE.search("\n".join(cuda_text_parts))),
        "blocked_preconditions": [],
    }


def _parse_cuda_driver(
    raw_nvidia_smi: Mapping[str, Any], nvidia_smi: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover
    raw_text = str(raw_nvidia_smi.get("stdout", "")) + "\n" + str(raw_nvidia_smi.get("stderr", ""))
    query_line = str(nvidia_smi.get("stdout", "")).splitlines()[0] if nvidia_smi.get("stdout") else ""
    parts = [part.strip() for part in query_line.split(",")]
    driver = parts[2] if len(parts) >= 3 else None
    cuda_match = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", raw_text)
    return {
        "driver_version": driver,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
    }


def build_backend_variants(
    preconditions: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    *,
    port: int = 8913,
) -> list[JsonDict]:
    binary_paths = preconditions.get("binary_paths", {})
    model_path = str(model_spec["model_path"])
    base = {
        "model_path": model_path,
        "context": DEFAULT_CONTEXT,
        "batch": DEFAULT_BATCH,
        "ubatch": DEFAULT_UBATCH,
        "gpu_layers": DEFAULT_GPU_LAYERS,
        "split_mode": DEFAULT_SPLIT_MODE,
        "tensor_split": DEFAULT_TENSOR_SPLIT,
        "prompt": PROMPT,
        "n_predict": N_PREDICT,
        "timeout_s": DEFAULT_TIMEOUT_S,
    }
    common_flags = [
        "-m",
        model_path,
        "-p",
        PROMPT,
        "-n",
        str(N_PREDICT),
        "-c",
        str(DEFAULT_CONTEXT),
        "-b",
        str(DEFAULT_BATCH),
        "-ub",
        str(DEFAULT_UBATCH),
        "-ngl",
        DEFAULT_GPU_LAYERS,
        "-sm",
        DEFAULT_SPLIT_MODE,
        "--temp",
        "0",
        "--seed",
        str(RANDOM_SEED),
        "--no-display-prompt",
        "--simple-io",
        "-st",
        "--perf",
    ]
    variants: list[JsonDict] = []
    cli = binary_paths.get("llama-cli")
    if cli:
        command = [str(cli), *common_flags]
        variants.append(
            base
            | {
                "name": "llama-cli-single-turn-batch512",
                "backend_kind": "llama-cli",
                "command": command,
                "command_form_repair": "uses -st single-turn and omits --no-conversation",
            }
        )
    completion = binary_paths.get("llama-completion")
    if completion:
        command = [str(completion), *common_flags]
        variants.append(
            base
            | {
                "name": "llama-completion-single-turn-batch512",
                "backend_kind": "llama-completion",
                "command": command,
                "command_form_repair": "uses -st single-turn and omits --no-conversation",
            }
        )
    server = binary_paths.get("llama-server")
    if server:
        command = [
            str(server),
            "-m",
            model_path,
            "-c",
            str(DEFAULT_CONTEXT),
            "-b",
            str(DEFAULT_BATCH),
            "-ub",
            str(DEFAULT_UBATCH),
            "-ngl",
            DEFAULT_GPU_LAYERS,
            "-sm",
            DEFAULT_SPLIT_MODE,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-webui",
            "-np",
            "1",
            "--metrics",
        ]
        variants.append(
            base
            | {
                "name": "llama-server-completion-api-batch512",
                "backend_kind": "llama-server",
                "command": command,
                "server_url": f"http://127.0.0.1:{port}/completion",
                "command_form_repair": "uses completion API and omits --no-conversation",
            }
        )
    return variants


def _precondition_blockers(
    preconditions: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
) -> list[str]:
    blockers = list(preconditions.get("blocked_preconditions", []))
    if not preconditions.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if int(preconditions.get("free_vram_mb") or 0) <= 0:
        blockers.append("free_vram_unavailable")
    binary_paths = preconditions.get("binary_paths", {})
    if not any(binary_paths.get(binary) for binary in NATIVE_BACKENDS):
        blockers.append("no_native_llama_cpp_binary_available")
    elif not preconditions.get("cuda_backend_evidence"):
        blockers.append("native_llama_cpp_cuda_evidence_missing")
    if not any(row.get("status") == "local_gguf_resolved" for row in model_specs.values()):
        blockers.append("no_mandated_sota_gguf_resolved")
    return list(dict.fromkeys(blockers))


def _initial_backend_matrix(preconditions: Mapping[str, Any]) -> JsonDict:
    matrix: JsonDict = {}
    for backend in NATIVE_BACKENDS:
        path = preconditions.get("binary_paths", {}).get(backend)
        version = preconditions.get("binary_versions", {}).get(backend)
        libraries = preconditions.get("binary_dynamic_libraries", {}).get(backend)
        evidence_text = "\n".join(
            [
                str((version or {}).get("stdout", "")),
                str((version or {}).get("stderr", "")),
                str((libraries or {}).get("stdout", "")),
                str((libraries or {}).get("stderr", "")),
            ]
        )
        matrix[backend] = {
            "path": path,
            "available": bool(path),
            "version": version,
            "dynamic_libraries": libraries,
            "cuda_backend_evidence": bool(CUDA_EVIDENCE_RE.search(evidence_text)),
            "attempts": [],
            "skipped": [],
        }
    return matrix


def _initial_runtime_matrix(model_specs: Mapping[str, JsonDict]) -> JsonDict:
    matrix: JsonDict = {}
    for role, spec in model_specs.items():
        available = spec.get("status") == "local_gguf_resolved"
        matrix[role] = {
            "role": role,
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "model_available": available,
            "cache_status": spec.get("status"),
            "autotokenizer_used": False,
            "attempts": [],
            "best_attempt_status": "not_attempted" if available else "not_available",
        }
    return matrix


def classify_runtime_receipt(receipt: Mapping[str, Any]) -> str:
    if (
        receipt.get("completed")
        and int(receipt.get("generated_token_count") or 0) >= N_PREDICT
        and receipt.get("first_token_latency_s") is not None
    ):
        return "completed_no_timeout"
    log_text = f"{receipt.get('stdout_tail', '')}\n{receipt.get('stderr_tail', '')}"
    if "n_tokens_all <= cparams.n_batch" in log_text or "llama-context.cpp:1712" in log_text:
        return "llama_context_batch_assert"
    if "--no-conversation" in log_text and "unsupported" in log_text.lower():
        return "llama_cli_no_conversation_unsupported"
    if receipt.get("timed_out") and receipt.get("first_token_latency_s") is None:
        return "timeout_before_first_token"
    if receipt.get("timed_out") and int(receipt.get("generated_token_count") or 0) < N_PREDICT:
        return "timeout_during_8_token_generation"
    returncode = receipt.get("returncode")
    if isinstance(returncode, int) and (returncode < 0 or returncode in {134, 137}):
        return "native_llama_cpp_abort_signal"
    if receipt.get("first_token_latency_s") is None:
        return "no_first_token"
    if int(receipt.get("generated_token_count") or 0) < N_PREDICT:
        return "generation_incomplete"
    return "generation_incomplete"


def _offload_authenticated(receipt: Mapping[str, Any]) -> bool:
    gpu_memory = receipt.get("gpu_memory_receipts") or {}
    log_text = f"{receipt.get('stdout_tail', '')}\n{receipt.get('stderr_tail', '')}"
    return bool(
        receipt.get("backend_gpu_log_evidence")
        or gpu_memory.get("offload_evidence")
        or int(gpu_memory.get("max_memory_delta_mb") or 0) > 128
        or CUDA_EVIDENCE_RE.search(log_text)
    )


def _normalise_attempt(
    receipt: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
) -> JsonDict:
    timeout_class = classify_runtime_receipt(receipt)
    generated_token_count = int(receipt.get("generated_token_count") or 0)
    eight_status = str(
        receipt.get("eight_token_completion_status")
        or ("completed_8_tokens" if generated_token_count >= N_PREDICT else "incomplete")
    )
    completed_8 = (
        timeout_class == "completed_no_timeout"
        and generated_token_count >= N_PREDICT
        and receipt.get("first_token_latency_s") is not None
    )
    gpu_memory = dict(receipt.get("gpu_memory_receipts") or {})
    offload_authenticated = _offload_authenticated(receipt)
    return {
        "backend_kind": str(receipt.get("backend_kind") or variant["backend_kind"]),
        "backend_variant": str(receipt.get("backend_variant") or variant["name"]),
        "command": list(receipt.get("command") or variant["command"]),
        "model_path": str(model_spec.get("model_path")),
        "context": int(receipt.get("context") or variant["context"]),
        "batch": int(receipt.get("batch") or variant["batch"]),
        "ubatch": int(receipt.get("ubatch") or variant["ubatch"]),
        "gpu_layers": str(receipt.get("gpu_layers") or variant["gpu_layers"]),
        "tensor_split": receipt.get("tensor_split", variant["tensor_split"]),
        "prompt": str(receipt.get("prompt") or variant["prompt"]),
        "n_predict": int(receipt.get("n_predict") or variant["n_predict"]),
        "timeout_s": float(receipt.get("timeout_s") or timeout_s),
        "status": str(receipt.get("status") or timeout_class),
        "timeout_class": timeout_class,
        "returncode": receipt.get("returncode"),
        "timed_out": bool(receipt.get("timed_out")),
        "wall_clock_s": receipt.get("wall_clock_s"),
        "load_s": receipt.get("load_s"),
        "first_token_latency_s": receipt.get("first_token_latency_s"),
        "eight_token_generation_s": receipt.get("eight_token_generation_s"),
        "generated_token_count": generated_token_count,
        "eight_token_completion_status": eight_status,
        "completed_load_first_token_and_8_tokens": completed_8,
        "offload_authenticated": offload_authenticated,
        "backend_gpu_log_evidence": bool(receipt.get("backend_gpu_log_evidence")),
        "gpu_memory_delta_mb": int(gpu_memory.get("max_memory_delta_mb") or 0),
        "gpu_memory_receipts": gpu_memory,
        "stdout_tail": str(receipt.get("stdout_tail", ""))[-2000:],
        "stderr_tail": str(receipt.get("stderr_tail", ""))[-4000:],
    }


def _attempt_is_ready(attempt: Mapping[str, Any]) -> bool:
    if not isinstance(attempt, Mapping):
        return False
    return bool(
        attempt.get("completed_load_first_token_and_8_tokens")
        and attempt.get("offload_authenticated")
    )


def _best_command_from_attempt(role: str, attempt: Mapping[str, Any]) -> JsonDict:
    return {
        "model_role": role,
        "backend_kind": attempt["backend_kind"],
        "backend_variant": attempt["backend_variant"],
        "command": attempt["command"],
        "model_path": attempt["model_path"],
        "context": attempt["context"],
        "batch": attempt["batch"],
        "ubatch": attempt["ubatch"],
        "gpu_layers": attempt["gpu_layers"],
        "tensor_split": attempt["tensor_split"],
        "prompt": attempt["prompt"],
        "n_predict": attempt["n_predict"],
        "timeout_s": attempt["timeout_s"],
        "first_token_latency_s": attempt["first_token_latency_s"],
        "eight_token_generation_s": attempt["eight_token_generation_s"],
        "gpu_memory_delta_mb": attempt["gpu_memory_delta_mb"],
    }


def timeout_or_crash_root_cause(
    matrix: Mapping[str, Any], precondition_blockers: Sequence[str]
) -> str:
    if precondition_blockers:
        return "preconditions_blocked:" + ",".join(precondition_blockers)
    if not any(row.get("model_available") for row in matrix.values()):
        return "no_mandated_sota_gguf_resolved"
    attempts = [attempt for row in matrix.values() for attempt in row.get("attempts", [])]
    if not attempts:
        return "no_native_backend_attempt_executed"
    if any(
        attempt.get("completed_load_first_token_and_8_tokens")
        and not attempt.get("offload_authenticated")
        for attempt in attempts
    ):
        return "no_authenticated_gpu_offload_after_bounded_generation"
    classes = [str(attempt.get("timeout_class")) for attempt in attempts]
    if "llama_context_batch_assert" in classes:
        return "llama_context_batch_assert_after_native_backend_attempt"
    if "llama_cli_no_conversation_unsupported" in classes:
        return "llama_cli_no_conversation_unsupported_after_native_backend_attempt"
    if all(cls == "timeout_before_first_token" for cls in classes):
        return "all_attempted_native_backends_timeout_before_first_token"
    if "timeout_during_8_token_generation" in classes:
        return "timeout_during_bounded_8_token_generation"
    if "no_first_token" in classes:
        return "no_first_token_observed"
    return "no_mandated_model_completed_load_first_token_and_8_tokens:" + ",".join(classes)


def default_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> JsonDict:  # pragma: no cover
    if variant["backend_kind"] == "llama-server":
        return _server_runtime_probe(model_spec=model_spec, variant=variant, timeout_s=timeout_s)
    return _process_runtime_probe(model_spec=model_spec, variant=variant, timeout_s=timeout_s)


def _process_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
) -> JsonDict:  # pragma: no cover
    command = list(variant["command"])
    started = time.perf_counter()
    before = _gpu_snapshot()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    selector.register(proc.stderr, selectors.EVENT_READ)
    samples: list[list[JsonDict]] = [before]
    stdout = bytearray()
    stderr = bytearray()
    first_stdout_s: float | None = None
    timed_out = False
    while selector.get_map():
        elapsed = time.perf_counter() - started
        if proc.poll() is None and elapsed > timeout_s:
            proc.kill()
            timed_out = True
        samples.append(_gpu_snapshot())
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
    generated = _extract_eval_runs(log_text)
    if generated is None and proc.returncode == 0 and stdout_text.strip():
        generated = int(variant["n_predict"])
    generated_token_count = int(generated or 0)
    generation_s = _extract_ms(EVAL_TIME_RE, log_text)
    completed = (
        proc.returncode == 0
        and not timed_out
        and first_stdout_s is not None
        and generated_token_count >= int(variant["n_predict"])
    )
    return {
        "backend_kind": variant["backend_kind"],
        "backend_variant": variant["name"],
        "status": "completed" if completed else "timeout_or_generation_incomplete",
        "completed": completed,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "wall_clock_s": round(wall_clock_s, 6),
        "load_s": _extract_ms(LOAD_TIME_RE, log_text),
        "first_token_latency_s": first_stdout_s,
        "eight_token_generation_s": generation_s,
        "generated_token_count": generated_token_count,
        "eight_token_completion_status": (
            "completed_8_tokens" if generated_token_count >= int(variant["n_predict"]) else "incomplete"
        ),
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-4000:],
        "backend_gpu_log_evidence": bool(CUDA_EVIDENCE_RE.search(log_text)),
        "command": command,
        "context": variant["context"],
        "batch": variant["batch"],
        "ubatch": variant["ubatch"],
        "gpu_layers": variant["gpu_layers"],
        "tensor_split": variant["tensor_split"],
        "prompt": variant["prompt"],
        "n_predict": variant["n_predict"],
        "returncode": proc.returncode,
        "gpu_memory_receipts": {
            "before": before,
            "during": samples[1:-1],
            "after": after,
            "max_memory_delta_mb": max_delta,
            "offload_evidence": bool(max_delta > 128 or CUDA_EVIDENCE_RE.search(log_text)),
        },
        "model_path": model_spec.get("model_path"),
    }


def _server_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
) -> JsonDict:  # pragma: no cover
    command = list(variant["command"])
    started = time.perf_counter()
    before = _gpu_snapshot()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_lines: list[str] = []
    stdout_lines: list[str] = []
    ready = False
    deadline = started + min(timeout_s, 180.0)
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            break
        line = proc.stderr.readline() if proc.stderr else ""
        if line:
            stderr_lines.append(line)
            if "server is listening" in line.lower() or "listening" in line.lower():
                ready = True
                break
        if _port_open("127.0.0.1", int(str(variant["server_url"]).rsplit(":", 1)[1].split("/", 1)[0])):
            ready = True
            break
        time.sleep(0.25)
    samples = [before, _gpu_snapshot()]
    response_text = ""
    request_error: str | None = None
    request_s: float | None = None
    if ready:
        request_started = time.perf_counter()
        payload = json.dumps(
            {
                "prompt": variant["prompt"],
                "n_predict": variant["n_predict"],
                "temperature": 0,
                "cache_prompt": False,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            str(variant["server_url"]),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=max(1.0, timeout_s / 2)) as response:
                body = response.read().decode("utf-8", "replace")
            response_json = json.loads(body)
            response_text = str(response_json.get("content", ""))
            request_s = time.perf_counter() - request_started
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            request_error = f"{type(exc).__name__}: {exc}"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if proc.stdout:
        stdout_lines.extend(proc.stdout.readlines()[-20:])
    if proc.stderr:
        stderr_lines.extend(proc.stderr.readlines()[-80:])
    after = _gpu_snapshot()
    samples.append(after)
    log_text = "".join(stdout_lines + stderr_lines)
    max_used = max((_total_used_mb(sample) for sample in samples), default=0)
    max_delta = max(0, max_used - _total_used_mb(before))
    generated = int(variant["n_predict"]) if response_text.strip() else 0
    completed = ready and request_error is None and generated >= int(variant["n_predict"])
    return {
        "backend_kind": variant["backend_kind"],
        "backend_variant": variant["name"],
        "status": "completed" if completed else "server_completion_failed",
        "completed": completed,
        "timed_out": not ready,
        "timeout_s": timeout_s,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "load_s": None,
        "first_token_latency_s": request_s,
        "eight_token_generation_s": request_s,
        "generated_token_count": generated,
        "eight_token_completion_status": "completed_8_tokens" if completed else "incomplete",
        "stdout_tail": ("".join(stdout_lines) + response_text)[-2000:],
        "stderr_tail": (log_text + (request_error or ""))[-4000:],
        "backend_gpu_log_evidence": bool(CUDA_EVIDENCE_RE.search(log_text)),
        "command": command,
        "context": variant["context"],
        "batch": variant["batch"],
        "ubatch": variant["ubatch"],
        "gpu_layers": variant["gpu_layers"],
        "tensor_split": variant["tensor_split"],
        "prompt": variant["prompt"],
        "n_predict": variant["n_predict"],
        "returncode": proc.returncode,
        "gpu_memory_receipts": {
            "before": before,
            "during": samples[1:-1],
            "after": after,
            "max_memory_delta_mb": max_delta,
            "offload_evidence": bool(max_delta > 128 or CUDA_EVIDENCE_RE.search(log_text)),
        },
        "model_path": model_spec.get("model_path"),
    }


def _port_open(host: str, port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _extract_ms(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return round(float(match.group(1)) / 1000.0, 6)


def _extract_eval_runs(text: str) -> int | None:
    match = EVAL_RUNS_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    preconditions_provider: PreconditionsProvider | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    tests_run: Sequence[Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    preconditions_provider = preconditions_provider or (lambda: collect_preconditions(root))
    preconditions = dict(preconditions_provider())
    model_specs = {
        str(spec["role"]): _resolve_model_spec(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }
    preconditions["autotokenizer_used"] = False
    preconditions["model_cache_status"] = {
        role: {
            "hf_id": spec["hf_id"],
            "cached": bool(spec.get("cached")),
            "model_path": spec.get("model_path"),
            "status": spec.get("status"),
        }
        for role, spec in model_specs.items()
    }
    preconditions["gguf_cache_paths"] = {
        role: spec["cache_path"] for role, spec in model_specs.items()
    }
    preconditions["resolved_roles"] = [
        role for role, spec in model_specs.items() if spec.get("status") == "local_gguf_resolved"
    ]
    preconditions["attempt_role_order"] = list(ATTEMPT_ROLE_ORDER)

    blockers = _precondition_blockers(preconditions, model_specs)
    preconditions["blocked_preconditions"] = blockers
    backend_matrix = _initial_backend_matrix(preconditions)
    runtime_matrix = _initial_runtime_matrix(model_specs)
    best_command: JsonDict | None = None

    if not blockers:
        for role in ATTEMPT_ROLE_ORDER:
            model_spec = model_specs[role]
            if model_spec.get("status") != "local_gguf_resolved":
                continue
            variants = build_backend_variants(preconditions, model_spec)
            if not variants:
                runtime_matrix[role]["best_attempt_status"] = "not_attempted_no_native_variant"
                continue
            for variant in variants:
                receipt = runtime_probe(model_spec=model_spec, variant=variant, timeout_s=timeout_s)
                attempt = _normalise_attempt(
                    receipt, model_spec=model_spec, variant=variant, timeout_s=timeout_s
                )
                runtime_matrix[role]["attempts"].append(attempt)
                runtime_matrix[role]["best_attempt_status"] = attempt["timeout_class"]
                backend_matrix[attempt["backend_kind"]]["attempts"].append(
                    {
                        "model_role": role,
                        "backend_variant": attempt["backend_variant"],
                        "timeout_class": attempt["timeout_class"],
                        "completed_load_first_token_and_8_tokens": attempt[
                            "completed_load_first_token_and_8_tokens"
                        ],
                        "offload_authenticated": attempt["offload_authenticated"],
                    }
                )
                if _attempt_is_ready(attempt):
                    best_command = _best_command_from_attempt(role, attempt)
                    runtime_matrix[role]["best_attempt_status"] = "ready"
                    break
            if best_command is not None:
                break

    if best_command is not None:
        for backend, row in backend_matrix.items():
            if not row["attempts"] and row["available"]:
                row["skipped"].append("skipped_after_candidate_found")

    ready = best_command is not None
    root_cause = "none" if ready else timeout_or_crash_root_cause(runtime_matrix, blockers)
    status = "complete" if ready else "blocked"
    if ready:
        honest = (
            "complete: native_llama_cpp_backend_candidate_ready="
            f"{best_command['model_role']}:{best_command['backend_kind']}"
        )
    else:
        honest = f"blocked_native_backend_candidate_false: {root_cause}"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "backend_matrix": _wrap("backend_matrix", backend_matrix),
        "per_model_runtime_matrix": _wrap("per_model_runtime_matrix", runtime_matrix),
        "best_backend_command": _wrap("best_backend_command", best_command),
        "timeout_or_crash_root_cause": _wrap("timeout_or_crash_root_cause", root_cause),
        "sota_backend_candidate_ready": ready,
        "runtime_unblocked_min_one_mandated": ready,
        "no_quality_claim": True,
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "backend_matrix": backend_matrix,
                "runtime_matrix": runtime_matrix,
                "ready": ready,
                "root_cause": root_cause,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not isinstance(value, Mapping):
        return MISSING_WRAPPED_VALUE
    if value.get("principle") != FIELD_PRINCIPLES.get(field):
        return MISSING_WRAPPED_VALUE
    return value.get("value")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        if field in artifact and _wrapped_value(artifact, field) is MISSING_WRAPPED_VALUE:
            errors.append(f"{field} must be principle-wrapped")
    if _wrapped_value(artifact, "experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if _wrapped_value(artifact, "milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if _wrapped_value(artifact, "status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")
    for field in ("sota_backend_candidate_ready", "runtime_unblocked_min_one_mandated"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare boolean")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    runtime_matrix = _wrapped_value(artifact, "per_model_runtime_matrix")
    if not isinstance(model_specs, Mapping) or not isinstance(runtime_matrix, Mapping):
        errors.append("MODEL_SPECS and per_model_runtime_matrix must be objects")
    else:
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles or set(runtime_matrix) != expected_roles:
            errors.append("roles mismatch between MODEL_SPECS and runtime matrix")
        expected_hf = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
        for role in expected_roles & set(model_specs) & set(runtime_matrix):
            spec = model_specs[role]
            row = runtime_matrix[role]
            if spec.get("hf_id") != expected_hf[role] or row.get("hf_id") != expected_hf[role]:
                errors.append("hf_id mismatch for mandated model role")
            if spec.get("autotokenizer_used") is not False or row.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")
            if not isinstance(row.get("attempts"), list):
                errors.append("runtime attempts must be a list")

    if not isinstance(_wrapped_value(artifact, "backend_matrix"), Mapping):
        errors.append("backend_matrix must be an object")
    tests_run = _wrapped_value(artifact, "tests_run")
    if not isinstance(tests_run, list):
        errors.append("tests_run must be a list")

    ready = artifact.get("sota_backend_candidate_ready")
    runtime_ready = artifact.get("runtime_unblocked_min_one_mandated")
    root_cause = _wrapped_value(artifact, "timeout_or_crash_root_cause")
    best = _wrapped_value(artifact, "best_backend_command")
    if ready != runtime_ready:
        errors.append("runtime booleans must match")
    if ready:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("ready artifact must have complete status")
        if root_cause != "none":
            errors.append("ready artifact must have root cause none")
        if not isinstance(best, Mapping):
            errors.append("ready artifact must record best_backend_command")
        if isinstance(runtime_matrix, Mapping):
            attempts = [
                attempt
                for row in runtime_matrix.values()
                if isinstance(row.get("attempts"), list)
                for attempt in row.get("attempts", [])
            ]
            if not any(_attempt_is_ready(attempt) for attempt in attempts):
                errors.append("ready artifact must contain an offloaded bounded attempt")
    else:
        if not isinstance(root_cause, str) or not root_cause:
            errors.append("blocked artifact must name root cause")
        if best is not None:
            errors.append("blocked artifact cannot contain best_backend_command")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    tests_run = json.loads(args.tests_run_json)
    artifact = run(artifact_path=args.out, timeout_s=args.timeout_s, tests_run=tests_run, write=True)
    print(
        f"[exp5323] status={artifact['status']['value']} "
        f"ready={artifact['sota_backend_candidate_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
