#!/usr/bin/env python3
"""Exp 5297: strict changed-runtime SOTA GGUF substrate gate.

Spec refs: REQ-VERIFY-5297, SCENARIO-VERIFY-5297.

This module is a runtime receipt gate, not a quality experiment.  It tries a
changed local GGUF backend, records whether a mandated SOTA GGUF can generate
or score with GPU-offload evidence, and otherwise emits an honest blocked
artifact with no quality claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairProvider = Callable[..., list[JsonDict] | None]
GpuReceiptsProvider = Callable[[], JsonDict]
RuntimeSubstrateProvider = Callable[[], JsonDict]
GenerationProbe = Callable[..., JsonDict]
SmokeTestsProvider = Callable[[], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5297
EXPERIMENT_NAME = "experiment_5297_changed_runtime_sota_substrate_gate_v484"
RESULT_RELATIVE_PATH = Path("results/experiment_5297_changed_runtime_sota_substrate_gate_v484.json")
SCHEMA = "carnot.experiment_5297.changed_runtime_sota_substrate_gate.v484"
SPEC_REFS = ("REQ-VERIFY-5297", "SCENARIO-VERIFY-5297")
LIVE_INFERENCE_SUBSTRATE = "live_llm_inference_changed_local_gguf_sota"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_preconditions_with_no_quality_claim"
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
RANDOM_SEED = 5297
MIN_LIVE_GENERATION_DURATION_S = 1.0
DEFAULT_GENERATION_TIMEOUT_S = 180.0
MINIMAL_PROMPT = "Return exactly OK."

OFFLOAD_CONFIG: JsonDict = {
    "n_gpu_layers": "all",
    "n_ctx": 512,
    "max_tokens": 2,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
    "split_mode": "layer",
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
    "honest_verdict": (
        "Terminal Exp 5297 verdict; starts with `complete:` or `blocked_` and states "
        "whether changed-runtime SOTA receipts are ready."
    ),
    "inference_substrate": (
        "Declares live changed local GGUF SOTA inference only after a mandated model "
        "actually generated or scored with changed-substrate GPU evidence; otherwise "
        "records blocked preconditions with no quality claim."
    ),
    "preconditions_checked": (
        "Records Step 0 GPU, CUDA, backend, cache, disk, and mandated-GGUF local "
        "resolution checks before any generation/scoring attempt."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs, roles, quantization/file "
        "receipts, and per-model changed-runtime status."
    ),
    "runtime_substrate_changed": (
        "Explains how the attempted backend differs from Exp 5284's CPU-only "
        "`llama-cpp-python` path and why the old path is not being counted as success."
    ),
    "duration_receipts": (
        "Captures per-model wall-clock, prompt checksum, and output checksum receipts "
        "so runtime claims are tied to live calls."
    ),
    "gpu_offload_receipts": (
        "Captures driver/device/offload settings, memory deltas, backend logs, and "
        "build/dynamic-library evidence proving whether GPU offload was available."
    ),
    "smoke_tests": (
        "Tiny or legacy model checks are labeled `smoke_test_not_headline` and cannot "
        "open the changed-runtime SOTA gate."
    ),
    "no_quality_claim": (
        "Must be true because Exp 5297 is a runtime-substrate gate, not a verifier, "
        "solver, benchmark, or model-quality experiment."
    ),
    "tests_run": (
        "Records the focused unit, coverage, and repository verification commands used "
        "for the v484 gate."
    ),
    "generation_receipts": (
        "Raw per-model changed-runtime generation or scoring receipts with prompt/output "
        "hashes, command, backend logs, offload config, and GPU-memory evidence."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "changed_runtime_sota_ready",
    "changed_runtime_sota_ready_principle",
    "runtime_substrate_changed",
    "duration_receipts",
    "gpu_offload_receipts",
    "smoke_tests",
    "no_quality_claim",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "runtime_substrate_changed",
    "duration_receipts",
    "gpu_offload_receipts",
    "smoke_tests",
    "no_quality_claim",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")
CUDA_LOG_PATTERN = re.compile(
    r"(CUDA\d|ggml_cuda|libggml-cuda|offloaded\s+\d+/\d+\s+layers|to GPU)",
    re.IGNORECASE,
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def sha16(value: str | bytes) -> str:
    """Return a compact stable checksum for prompt, output, and receipt fields."""

    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_cache_path(hf_id: str) -> str:
    return str(
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"
    )


def _utc_run_date() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _file_receipts(path: Path) -> JsonDict:
    """Return size and checksum receipts without hashing huge GGUFs in full."""

    size = path.stat().st_size
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        hasher.update(head)
        if size <= 64 * 1024 * 1024:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
            full_checksum = hasher.hexdigest()
        else:
            full_checksum = None
    return {
        "path": str(path),
        "size_bytes": size,
        "checksum_sha256": full_checksum,
        "checksum_head_1m_sha256": hashlib.sha256(head).hexdigest(),
        "checksum_note": (
            "full_sha256_recorded"
            if full_checksum is not None
            else "full_sha256_skipped_for_large_file_head_1m_recorded"
        ),
    }


def read_gguf_header(model_path: str | Path) -> JsonDict:
    """Read the fixed GGUF header so pointer files fail before runtime load."""

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


def _missing_model_spec(spec: Mapping[str, Any]) -> JsonDict:
    role = str(spec["role"])
    hf_id = str(spec["hf_id"])
    return {
        "role": role,
        "hf_id": hf_id,
        "quantization": str(spec.get("quantization", DEFAULT_PREFERRED_QUANT)),
        "cache_path": _repo_cache_path(hf_id),
        "model_path": None,
        "status": "missing_local_gguf",
        "autotokenizer_used": False,
        "headline_role": True,
        "smoke_label": None,
        "file_receipts": None,
        "metadata": None,
        "runtime_status": "not_attempted",
        "live_generation_ready": False,
        "blocked_preconditions": [],
    }


def _model_spec_receipt(spec: Mapping[str, Any], model_resolver: ModelResolver) -> JsonDict:
    receipt = _missing_model_spec(spec)
    path_text = model_resolver(
        str(spec["hf_id"]), str(spec.get("quantization", DEFAULT_PREFERRED_QUANT))
    )
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
        receipt["runtime_status"] = "not_attempted_metadata_unreadable"
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
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _first_executable_llama_cli() -> Path | None:  # pragma: no cover
    for candidate in _candidate_llama_cli_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def collect_gpu_receipts() -> JsonDict:  # pragma: no cover
    torch_cuda: JsonDict = {"import_ok": False, "available": False, "device_count": 0}
    try:
        import torch  # noqa: PLC0415

        torch_cuda = {
            "import_ok": True,
            "version": getattr(torch, "__version__", "unknown"),
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        }
    except Exception as exc:
        torch_cuda["error"] = f"{type(exc).__name__}: {exc}"

    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    return {
        "gpu_visible": bool(nvidia_smi.get("ok") or torch_cuda.get("available")),
        "nvidia_smi": nvidia_smi,
        "cuda_runtime": _run_command(["nvidia-smi"], timeout_s=10.0),
        "nvcc": _run_command(["nvcc", "--version"], timeout_s=10.0),
        "torch_cuda": torch_cuda,
    }


def collect_runtime_substrate_changed() -> JsonDict:  # pragma: no cover
    backend = _first_executable_llama_cli()
    if backend is None:
        return {
            "backend_kind": "native_llama_cpp_cli",
            "backend_path": None,
            "changed_from_exp5284": False,
            "changed_from_exp5284_principle": (
                "no executable native llama.cpp CLI found; the old Python path is not counted"
            ),
            "version": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "list_devices": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "dynamic_libraries": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "cuda_backend_evidence": False,
            "old_cpu_only_llama_cpp_python_counted_as_success": False,
        }

    version = _run_command([str(backend), "--version"], timeout_s=20.0)
    list_devices = _run_command([str(backend), "--list-devices"], timeout_s=20.0)
    dynamic_libraries = _run_command(["ldd", str(backend)], timeout_s=20.0)
    evidence_text = "\n".join(
        [
            str(version.get("stdout", "")),
            str(version.get("stderr", "")),
            str(list_devices.get("stdout", "")),
            str(list_devices.get("stderr", "")),
            str(dynamic_libraries.get("stdout", "")),
            str(dynamic_libraries.get("stderr", "")),
        ]
    )
    cuda_evidence = bool(CUDA_LOG_PATTERN.search(evidence_text))
    return {
        "backend_kind": "native_llama_cpp_cli",
        "backend_path": str(backend),
        "changed_from_exp5284": True,
        "changed_from_exp5284_principle": (
            "native llama.cpp CLI process with independent binary and CUDA library evidence, "
            "not the Exp 5284 llama-cpp-python import path"
        ),
        "version": version,
        "list_devices": list_devices,
        "dynamic_libraries": {
            **dynamic_libraries,
            "stdout": str(dynamic_libraries.get("stdout", ""))[-4000:],
        },
        "cuda_backend_evidence": cuda_evidence,
        "old_cpu_only_llama_cpp_python_counted_as_success": False,
    }


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


def _read_temp_tail(handle: Any, max_bytes: int = 8192) -> str:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(max(0, size - max_bytes), os.SEEK_SET)
    return handle.read().decode("utf-8", errors="replace")


def default_generation_probe(
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    offload_config: Mapping[str, Any],
    runtime_substrate: Mapping[str, Any],
    timeout_s: float = DEFAULT_GENERATION_TIMEOUT_S,
) -> JsonDict:  # pragma: no cover
    backend_path = runtime_substrate.get("backend_path")
    if not backend_path:
        return _normal_probe_failure(
            model_spec=model_spec,
            prompt=prompt,
            offload_config=offload_config,
            status="blocked_changed_runtime_backend_missing",
            outcome="native llama.cpp CLI path missing",
            started=time.perf_counter(),
        )

    command = [
        str(backend_path),
        "-m",
        str(model_spec["model_path"]),
        "-p",
        prompt,
        "-n",
        str(offload_config["max_tokens"]),
        "-c",
        str(offload_config["n_ctx"]),
        "-ngl",
        str(offload_config["n_gpu_layers"]),
        "--temp",
        str(offload_config["temperature"]),
        "--seed",
        str(offload_config["seed"]),
        "--no-display-prompt",
        "--simple-io",
        "--no-conversation",
    ]
    started = time.perf_counter()
    before = _gpu_snapshot()
    samples: list[list[JsonDict]] = [before]
    try:
        # llama.cpp emits verbose load logs. Temporary files avoid pipe
        # backpressure while this process polls GPU memory for offload receipts.
        with tempfile.TemporaryFile() as stdout_file:
            with tempfile.TemporaryFile() as stderr_file:
                proc = subprocess.Popen(
                    command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=False,
                )
                while proc.poll() is None:
                    samples.append(_gpu_snapshot())
                    if time.perf_counter() - started > timeout_s:
                        proc.kill()
                        proc.wait(timeout=10)
                        after = _gpu_snapshot()
                        samples.append(after)
                        stdout = _read_temp_tail(stdout_file, max_bytes=4096)
                        stderr = _read_temp_tail(stderr_file, max_bytes=8192)
                        max_used = max(
                            (_total_used_mb(sample) for sample in samples), default=0
                        )
                        max_delta = max(0, max_used - _total_used_mb(before))
                        log_text = f"{stdout}\n{stderr}"
                        backend_gpu_log_evidence = bool(CUDA_LOG_PATTERN.search(log_text))
                        return {
                            "runtime_ready": False,
                            "status": "blocked_native_cli_timeout",
                            "wall_clock_s": round(time.perf_counter() - started, 6),
                            "prompt_checksum": sha16(prompt),
                            "output_checksum": None,
                            "output_text_preview": stdout.strip()[:120],
                            "command": command,
                            "config": dict(offload_config),
                            "timeout_s": timeout_s,
                            "returncode": proc.returncode,
                            "stdout_tail": stdout[-2000:],
                            "stderr_tail": stderr[-4000:],
                            "backend_gpu_log_evidence": backend_gpu_log_evidence,
                            "gpu_memory_receipts": {
                                "before": before,
                                "during": samples[1:-1],
                                "after": after,
                                "max_memory_delta_mb": max_delta,
                                "offload_evidence": bool(
                                    max_delta > 128 or backend_gpu_log_evidence
                                ),
                            },
                            "outcome": (stderr or stdout)[-2000:],
                        }
                    time.sleep(0.5)
                stdout = _read_temp_tail(stdout_file, max_bytes=4096)
                stderr = _read_temp_tail(stderr_file, max_bytes=8192)
    except Exception as exc:
        return _normal_probe_failure(
            model_spec=model_spec,
            prompt=prompt,
            offload_config=offload_config,
            status="blocked_native_cli_subprocess_failed",
            outcome=f"{type(exc).__name__}: {exc}",
            started=started,
        )
    after = _gpu_snapshot()
    samples.append(after)
    max_used = max((_total_used_mb(sample) for sample in samples), default=0)
    max_delta = max(0, max_used - _total_used_mb(before))
    log_text = f"{stdout}\n{stderr}"
    backend_gpu_log_evidence = bool(CUDA_LOG_PATTERN.search(log_text))
    offload_evidence = bool(max_delta > 128 or backend_gpu_log_evidence)
    output_text = stdout.strip()
    return {
        "runtime_ready": proc.returncode == 0 and bool(output_text),
        "status": "generation_ready"
        if proc.returncode == 0 and output_text
        else "blocked_native_cli_generation_failed",
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "prompt_checksum": sha16(prompt),
        "output_checksum": sha16(output_text) if output_text else None,
        "output_text_preview": output_text[:120],
        "command": command,
        "config": dict(offload_config),
        "returncode": proc.returncode,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-4000:],
        "backend_gpu_log_evidence": backend_gpu_log_evidence,
        "gpu_memory_receipts": {
            "before": before,
            "during": samples[1:-1],
            "after": after,
            "max_memory_delta_mb": max_delta,
            "offload_evidence": offload_evidence,
        },
    }


def _normal_probe_failure(
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    offload_config: Mapping[str, Any],
    status: str,
    outcome: str,
    started: float,
) -> JsonDict:  # pragma: no cover
    return {
        "runtime_ready": False,
        "status": status,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "prompt_checksum": sha16(prompt),
        "output_checksum": None,
        "output_text_preview": "",
        "command": ["native_llama_cpp_cli", str(model_spec.get("model_path"))],
        "config": dict(offload_config),
        "gpu_memory_receipts": {
            "before": None,
            "during": [],
            "after": None,
            "max_memory_delta_mb": 0,
            "offload_evidence": False,
        },
        "outcome": outcome,
        "traceback": traceback.format_exc(),
    }


def default_smoke_tests() -> list[JsonDict]:
    return []


def _gpu_visible_from(gpu_receipts: Mapping[str, Any]) -> bool:
    return bool(gpu_receipts.get("gpu_visible"))


def _runtime_changed(runtime_substrate: Mapping[str, Any]) -> bool:
    return bool(
        runtime_substrate.get("changed_from_exp5284")
        and runtime_substrate.get("old_cpu_only_llama_cpp_python_counted_as_success") is False
        and runtime_substrate.get("backend_path")
    )


def _runtime_cuda_backend(runtime_substrate: Mapping[str, Any]) -> bool:
    return bool(runtime_substrate.get("cuda_backend_evidence"))


def _precondition_blockers(
    *,
    gpu_receipts: Mapping[str, Any],
    runtime_substrate: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
) -> list[str]:
    blockers: list[str] = []
    if not _gpu_visible_from(gpu_receipts):
        blockers.append("gpu_not_visible")
    if not _runtime_changed(runtime_substrate):
        blockers.append("changed_runtime_substrate_unavailable")
    elif not _runtime_cuda_backend(runtime_substrate):
        blockers.append("changed_runtime_gpu_backend_unavailable")
    if not any(spec.get("status") == "local_gguf_resolved" for spec in model_specs.values()):
        blockers.append("no_mandated_sota_gguf_resolved")
    return blockers


def _normalise_generation_receipt(
    receipt: Mapping[str, Any], model_spec: Mapping[str, Any]
) -> JsonDict:
    normal = {
        "role": str(model_spec["role"]),
        "hf_id": str(model_spec["hf_id"]),
        "model_path": str(model_spec["model_path"]),
        "runtime_ready": bool(receipt.get("runtime_ready")),
        "live_generation_ready": False,
        "status": str(
            receipt.get("status")
            or (
                "generation_ready"
                if receipt.get("runtime_ready")
                else "blocked_runtime_probe_failed"
            )
        ),
        "wall_clock_s": float(receipt.get("wall_clock_s") or receipt.get("duration_s") or 0.0),
        "command": receipt.get("command"),
        "config": dict(receipt.get("config") or OFFLOAD_CONFIG),
        "timeout_s": receipt.get("timeout_s"),
        "prompt_checksum": str(receipt.get("prompt_checksum") or sha16(MINIMAL_PROMPT)),
        "output_checksum": receipt.get("output_checksum"),
        "output_text_preview": str(receipt.get("output_text_preview") or "")[:120],
        "stdout_tail": str(receipt.get("stdout_tail") or "")[-2000:],
        "stderr_tail": str(receipt.get("stderr_tail") or "")[-4000:],
        "backend_gpu_log_evidence": bool(receipt.get("backend_gpu_log_evidence")),
        "gpu_memory_receipts": _normalise_gpu_memory_receipts(
            receipt.get("gpu_memory_receipts")
        ),
        "traceback": receipt.get("traceback"),
        "returncode": receipt.get("returncode"),
    }
    if normal["runtime_ready"] and normal["wall_clock_s"] < MIN_LIVE_GENERATION_DURATION_S:
        raise ValueError(
            f"sub-second live generation duration for {normal['role']}: {normal['wall_clock_s']}"
        )
    offload_evidence = bool(
        normal["backend_gpu_log_evidence"]
        or normal["gpu_memory_receipts"].get("offload_evidence") is True
        or int(normal["gpu_memory_receipts"].get("max_memory_delta_mb") or 0) > 128
    )
    normal["gpu_memory_receipts"]["offload_evidence"] = offload_evidence
    if normal["runtime_ready"] and offload_evidence:
        normal["live_generation_ready"] = True
    elif normal["runtime_ready"]:
        normal["status"] = "blocked_no_gpu_offload_evidence"
    return normal


def _normalise_gpu_memory_receipts(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        out = dict(value)
    else:
        out = {}
    out.setdefault("before", None)
    out.setdefault("during", [])
    out.setdefault("after", None)
    out.setdefault("max_memory_delta_mb", 0)
    out.setdefault("offload_evidence", False)
    return out


def _run_model_generation(
    *,
    model_specs: Mapping[str, JsonDict],
    runtime_substrate: Mapping[str, Any],
    generation_probe: GenerationProbe,
    precondition_blockers: Sequence[str],
) -> JsonDict:
    if precondition_blockers:
        for model_spec in model_specs.values():
            if model_spec.get("status") == "local_gguf_resolved":
                model_spec["runtime_status"] = "not_attempted_preconditions_failed"
                model_spec["blocked_preconditions"] = list(precondition_blockers)
        return {}

    receipts: JsonDict = {}
    for role, model_spec in model_specs.items():
        if model_spec.get("status") != "local_gguf_resolved" or not model_spec.get("model_path"):
            continue
        try:
            raw_receipt = generation_probe(
                model_spec=model_spec,
                prompt=MINIMAL_PROMPT,
                offload_config=OFFLOAD_CONFIG,
                runtime_substrate=runtime_substrate,
            )
        except Exception as exc:  # pragma: no cover
            raw_receipt = {
                "runtime_ready": False,
                "status": "blocked_generation_probe_exception",
                "wall_clock_s": 0.0,
                "prompt_checksum": sha16(MINIMAL_PROMPT),
                "output_checksum": None,
                "gpu_memory_receipts": {"offload_evidence": False, "max_memory_delta_mb": 0},
                "traceback": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        receipts[role] = _normalise_generation_receipt(raw_receipt, model_spec)
        model_spec["runtime_status"] = receipts[role]["status"]
        model_spec["live_generation_ready"] = bool(receipts[role]["live_generation_ready"])
        model_spec["generation_receipt"] = receipts[role]
        if not receipts[role]["live_generation_ready"] and receipts[role]["runtime_ready"]:
            model_spec["blocked_preconditions"] = ["blocked_no_gpu_offload_evidence"]
    return receipts


def _cached_pair_preview(cached_pair_provider: CachedPairProvider) -> list[JsonDict]:
    try:
        return cached_pair_provider(gpu_indices=(0, 1)) or []
    except Exception as exc:  # pragma: no cover
        return [{"status": "cached_sota_pair_error", "error": f"{type(exc).__name__}: {exc}"}]


def _preconditions(
    *,
    root: Path,
    gpu_receipts: Mapping[str, Any],
    runtime_substrate: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
    blockers: Sequence[str],
) -> JsonDict:
    total, used, free = shutil.disk_usage(root)
    resolved_roles = [role for role, spec in model_specs.items() if spec.get("model_path")]
    value = {
        "run_date_utc": _utc_run_date(),
        "run_on_or_after_20260706_utc": _utc_run_date() >= "20260706",
        "gpu_visibility_checked": True,
        "gpu_visible": _gpu_visible_from(gpu_receipts),
        "driver_cuda_checked": True,
        "driver_cuda": {
            "nvidia_smi": dict(gpu_receipts.get("nvidia_smi", {}))
            if isinstance(gpu_receipts.get("nvidia_smi"), Mapping)
            else {},
            "cuda_runtime": dict(gpu_receipts.get("cuda_runtime", {}))
            if isinstance(gpu_receipts.get("cuda_runtime"), Mapping)
            else {},
            "nvcc": dict(gpu_receipts.get("nvcc", {}))
            if isinstance(gpu_receipts.get("nvcc"), Mapping)
            else {},
            "torch_cuda": dict(gpu_receipts.get("torch_cuda", {}))
            if isinstance(gpu_receipts.get("torch_cuda"), Mapping)
            else {},
        },
        "changed_runtime_checked": True,
        "changed_runtime": dict(runtime_substrate),
        "free_disk": {
            "path": str(root),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
        },
        "gguf_cache_paths": {
            spec["role"]: _repo_cache_path(str(spec["hf_id"])) for spec in MANDATED_MODEL_SPECS
        },
        "cached_sota_pair_preview": _cached_pair_preview(cached_pair_provider),
        "at_least_one_mandated_model_resolved_without_autotokenizer": bool(resolved_roles),
        "resolved_roles": resolved_roles,
        "autotokenizer_used": False,
        "blocked_preconditions": list(blockers),
    }
    return _wrap("preconditions_checked", value)


def _duration_receipts(
    *,
    generation_receipts: Mapping[str, JsonDict],
    duration_s: float,
) -> JsonDict:
    per_model = {
        role: {
            "wall_clock_s": float(receipt["wall_clock_s"]),
            "runtime_ready": bool(receipt["runtime_ready"]),
            "live_generation_ready": bool(receipt["live_generation_ready"]),
            "status": receipt["status"],
            "prompt_checksum": receipt.get("prompt_checksum"),
            "output_checksum": receipt.get("output_checksum"),
        }
        for role, receipt in generation_receipts.items()
    }
    return _wrap(
        "duration_receipts",
        {
            "total_wall_clock_s": round(duration_s, 6),
            "minimum_live_generation_duration_s": MIN_LIVE_GENERATION_DURATION_S,
            "per_model": per_model,
        },
    )


def _gpu_offload_receipts(
    *,
    gpu_receipts: Mapping[str, Any],
    runtime_substrate: Mapping[str, Any],
    generation_receipts: Mapping[str, JsonDict],
) -> JsonDict:
    base = dict(gpu_receipts)
    base["runtime_substrate"] = dict(runtime_substrate)
    base["offload_settings"] = dict(OFFLOAD_CONFIG)
    base["per_model"] = {
        role: {
            "n_gpu_layers": receipt.get("config", {}).get("n_gpu_layers"),
            "n_ctx": receipt.get("config", {}).get("n_ctx"),
            "max_tokens": receipt.get("config", {}).get("max_tokens"),
            "backend_gpu_log_evidence": receipt.get("backend_gpu_log_evidence"),
            "stdout_tail": receipt.get("stdout_tail"),
            "stderr_tail": receipt.get("stderr_tail"),
            "gpu_memory_receipts": receipt.get("gpu_memory_receipts"),
            "offload_evidence": bool(
                receipt.get("gpu_memory_receipts", {}).get("offload_evidence")
            ),
        }
        for role, receipt in generation_receipts.items()
    }
    return _wrap("gpu_offload_receipts", base)


def _verdict(
    *,
    model_specs: Mapping[str, JsonDict],
    generation_receipts: Mapping[str, JsonDict],
    precondition_blockers: Sequence[str],
) -> tuple[bool, str, str]:
    ready_roles = [role for role, spec in model_specs.items() if spec.get("live_generation_ready")]
    if ready_roles:
        ready_list = ", ".join(ready_roles)
        return (
            True,
            f"complete: changed_runtime_sota_ready=true via {ready_list}",
            "changed_runtime_sota_ready=true because a mandated SOTA GGUF completed "
            f"changed-substrate generation or scoring with GPU-offload evidence for {ready_list}.",
        )

    attempted = [
        f"{role}:{receipt.get('status')}:offload={receipt.get('gpu_memory_receipts', {}).get('offload_evidence')}"
        for role, receipt in generation_receipts.items()
    ]
    missing = [
        f"{role}:{spec.get('status')}:{','.join(spec.get('blocked_preconditions') or [])}"
        for role, spec in model_specs.items()
        if role not in generation_receipts
    ]
    blockers = list(precondition_blockers) + attempted + missing
    first = blockers[0] if blockers else "no_changed_runtime_live_receipt"
    return (
        False,
        f"blocked_preconditions: changed_runtime_sota_ready=false {first}",
        "changed_runtime_sota_ready=false because no mandated SOTA GGUF completed "
        f"changed-substrate generation/scoring with GPU-offload evidence; blocked_preconditions={blockers}",
    )


def build_artifact(
    *,
    root: Path,
    gpu_receipts: JsonDict,
    runtime_substrate: JsonDict,
    model_specs: Mapping[str, JsonDict],
    generation_receipts: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
    smoke_tests: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    precondition_blockers: Sequence[str],
) -> JsonDict:
    ready, verdict, ready_principle = _verdict(
        model_specs=model_specs,
        generation_receipts=generation_receipts,
        precondition_blockers=precondition_blockers,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap(
            "inference_substrate",
            LIVE_INFERENCE_SUBSTRATE if ready else BLOCKED_INFERENCE_SUBSTRATE,
        ),
        "preconditions_checked": _preconditions(
            root=root,
            gpu_receipts=gpu_receipts,
            runtime_substrate=runtime_substrate,
            model_specs=model_specs,
            cached_pair_provider=cached_pair_provider,
            blockers=precondition_blockers,
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "changed_runtime_sota_ready": ready,
        "changed_runtime_sota_ready_principle": ready_principle,
        "runtime_substrate_changed": _wrap("runtime_substrate_changed", dict(runtime_substrate)),
        "duration_receipts": _duration_receipts(
            generation_receipts=generation_receipts,
            duration_s=duration_s,
        ),
        "gpu_offload_receipts": _gpu_offload_receipts(
            gpu_receipts=gpu_receipts,
            runtime_substrate=runtime_substrate,
            generation_receipts=generation_receipts,
        ),
        "generation_receipts": _wrap("generation_receipts", dict(generation_receipts)),
        "smoke_tests": _wrap("smoke_tests", [dict(row) for row in smoke_tests]),
        "no_quality_claim": _wrap("no_quality_claim", True),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "spec_refs": SPEC_REFS,
                "model_specs": artifact["MODEL_SPECS"]["value"],
                "runtime_substrate_changed": artifact["runtime_substrate_changed"]["value"],
                "generation_receipts": artifact["generation_receipts"]["value"],
            }
        )
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, Mapping) or "value" not in value or "principle" not in value:
            errors.append(f"{field} must be principle-wrapped")
    verdict = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.value must start with complete: or blocked_")
    substrate = _wrapped_value(artifact, "inference_substrate")
    if substrate not in {LIVE_INFERENCE_SUBSTRATE, BLOCKED_INFERENCE_SUBSTRATE}:
        errors.append(
            f"inference_substrate.value must be {LIVE_INFERENCE_SUBSTRATE} or {BLOCKED_INFERENCE_SUBSTRATE}"
        )
    if not isinstance(artifact.get("changed_runtime_sota_ready"), bool):
        errors.append("changed_runtime_sota_ready must be a bare bool")
    elif artifact["changed_runtime_sota_ready"] and substrate != LIVE_INFERENCE_SUBSTRATE:
        errors.append("ready artifact must use live_llm_inference_changed_local_gguf_sota")
    elif (
        artifact.get("changed_runtime_sota_ready") is False
        and substrate != BLOCKED_INFERENCE_SUBSTRATE
    ):
        errors.append("blocked artifact must use blocked_preconditions_with_no_quality_claim")
    if not artifact.get("changed_runtime_sota_ready_principle"):
        errors.append("changed_runtime_sota_ready_principle must be non-empty")
    if _wrapped_value(artifact, "no_quality_claim") is not True:
        errors.append("no_quality_claim.value must be true")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run must be a list")

    runtime_substrate = _wrapped_value(artifact, "runtime_substrate_changed")
    if isinstance(runtime_substrate, Mapping):
        if runtime_substrate.get("old_cpu_only_llama_cpp_python_counted_as_success") is not False:
            errors.append("runtime_substrate_changed must not count old CPU-only Python path")
    else:
        errors.append("runtime_substrate_changed.value must be an object")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if isinstance(model_specs, Mapping):
        for spec in MANDATED_MODEL_SPECS:
            role = str(spec["role"])
            row = model_specs.get(role)
            if not isinstance(row, Mapping):
                errors.append(f"MODEL_SPECS.value missing role {role}")
                continue
            if row.get("hf_id") != spec["hf_id"]:
                errors.append(f"MODEL_SPECS.value.{role}.hf_id mismatch")
            if row.get("autotokenizer_used") is not False:
                errors.append(f"MODEL_SPECS.value.{role}.autotokenizer_used must be false")
    else:
        errors.append("MODEL_SPECS.value must be an object")

    duration_receipts = _wrapped_value(artifact, "duration_receipts")
    if isinstance(duration_receipts, Mapping):
        per_model = duration_receipts.get("per_model")
        if not isinstance(per_model, Mapping):
            errors.append("duration_receipts.value.per_model must be an object")
        else:
            for role, receipt in per_model.items():
                if (
                    receipt.get("runtime_ready")
                    and float(receipt.get("wall_clock_s", 0.0)) < MIN_LIVE_GENERATION_DURATION_S
                ):
                    errors.append(
                        f"duration_receipts.value.per_model.{role} is below live duration floor"
                    )
                if receipt.get("runtime_ready") and not receipt.get("prompt_checksum"):
                    errors.append(
                        f"duration_receipts.value.per_model.{role}.prompt_checksum missing"
                    )
                if receipt.get("runtime_ready") and not receipt.get("output_checksum"):
                    errors.append(
                        f"duration_receipts.value.per_model.{role}.output_checksum missing"
                    )
    else:
        errors.append("duration_receipts.value must be an object")

    smoke_tests = _wrapped_value(artifact, "smoke_tests")
    if not isinstance(smoke_tests, list):
        errors.append("smoke_tests.value must be a list")
    return errors


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
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    gpu_receipts_provider: GpuReceiptsProvider = collect_gpu_receipts,
    runtime_substrate_provider: RuntimeSubstrateProvider = collect_runtime_substrate_changed,
    generation_probe: GenerationProbe = default_generation_probe,
    smoke_tests_provider: SmokeTestsProvider = default_smoke_tests,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    artifact_path = Path(artifact_path)
    gpu_receipts = gpu_receipts_provider()
    runtime_substrate = runtime_substrate_provider()
    model_specs = {
        str(spec["role"]): _model_spec_receipt(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }
    precondition_blockers = _precondition_blockers(
        gpu_receipts=gpu_receipts,
        runtime_substrate=runtime_substrate,
        model_specs=model_specs,
    )
    generation_receipts = _run_model_generation(
        model_specs=model_specs,
        runtime_substrate=runtime_substrate,
        generation_probe=generation_probe,
        precondition_blockers=precondition_blockers,
    )
    artifact = build_artifact(
        root=root,
        gpu_receipts=gpu_receipts,
        runtime_substrate=runtime_substrate,
        model_specs=model_specs,
        generation_receipts=generation_receipts,
        cached_pair_provider=cached_pair_provider,
        smoke_tests=smoke_tests_provider(),
        tests_run=tests_run or [],
        duration_s=time.perf_counter() - started,
        precondition_blockers=precondition_blockers,
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
                    "test_experiment_5297_changed_runtime_sota_substrate_gate_v484.py -q"
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
