#!/usr/bin/env python3
"""Exp 5284: strict local SOTA GGUF generation/offload receipt repair.

Spec refs: REQ-VERIFY-5284, SCENARIO-VERIFY-5284.

This is a runtime gate, not a quality experiment.  It records whether a
mandated local SOTA GGUF can actually generate or score through llama.cpp with
GPU-offload evidence.  Tiny legacy models may appear only as explicitly
non-headline smoke receipts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairProvider = Callable[..., list[JsonDict] | None]
GpuReceiptsProvider = Callable[[], JsonDict]
GenerationProbe = Callable[..., JsonDict]
SmokeTestsProvider = Callable[[], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5284
EXPERIMENT_NAME = "experiment_5284_sota_runtime_offload_receipt_repair_v483"
RESULT_RELATIVE_PATH = Path("results/experiment_5284_sota_runtime_offload_receipt_repair_v483.json")
SCHEMA = "carnot.experiment_5284.sota_runtime_offload_receipt_repair.v483"
SPEC_REFS = ("REQ-VERIFY-5284", "SCENARIO-VERIFY-5284")
LIVE_INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_preconditions_with_no_quality_claim"
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
RANDOM_SEED = 5284
MIN_LIVE_GENERATION_DURATION_S = 1.0
MINIMAL_PROMPT = "Return exactly OK."

OFFLOAD_CONFIG: JsonDict = {
    "n_gpu_layers": -1,
    "n_ctx": 512,
    "max_tokens": 2,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
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
        "Terminal Exp 5284 verdict; starts with complete: or blocked_ and states "
        "whether SOTA offload receipts are ready for exp5286/exp5288."
    ),
    "inference_substrate": (
        "Declares live local SOTA GGUF inference only after a mandated model actually "
        "generated or scored; otherwise records blocked preconditions with no quality claim."
    ),
    "preconditions_checked": (
        "Records GPU visibility, driver/CUDA or ROCm facts, llama.cpp version/origin/offload "
        "support, cache paths, disk, and local GGUF resolvability before live generation."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs, roles, quantization, local file "
        "receipts, and runtime/offload status."
    ),
    "sota_offload_ready": (
        "Bare gate for exp5286 and exp5288; true only when at least one mandated SOTA GGUF "
        "completes live generation or scoring with GPU-offload evidence."
    ),
    "sota_offload_ready_principle": (
        "Explains the exact ready model receipt or blocked precondition used by downstream gates."
    ),
    "duration_receipts": (
        "Per-model wall-clock receipts plus prompt/output checksums for live generation or "
        "scoring, preventing unsupported fast-path claims."
    ),
    "gpu_offload_receipts": (
        "Driver/device/runtime/offload settings and GPU memory receipts proving which offload "
        "path was visible or attempted."
    ),
    "smoke_tests": (
        "Tiny legacy CPU smoke tests, if any, are labeled smoke_test_not_headline and cannot "
        "open the SOTA offload gate."
    ),
    "no_quality_claim": (
        "Must be true; Exp 5284 is a runtime/offload receipt gate and makes no verifier, "
        "solver, memory, benchmark, or model-quality claim."
    ),
    "tests_run": (
        "Commands run to validate the v483 runtime-offload gate, new-code coverage, and "
        "repository test status."
    ),
    "generation_receipts": (
        "Raw per-model live generation or scoring receipts with prompt/output hashes, command, "
        "offload config, and GPU-memory evidence."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "sota_offload_ready",
    "sota_offload_ready_principle",
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
    "duration_receipts",
    "gpu_offload_receipts",
    "smoke_tests",
    "no_quality_claim",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")


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


def _llama_cpp_import_receipt() -> JsonDict:  # pragma: no cover
    llama_spec = importlib.util.find_spec("llama_cpp")
    try:
        llama_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_version = None
    support: bool | None = None
    support_error: str | None = None
    if llama_spec is not None:
        try:
            from llama_cpp import llama_cpp as llama_backend  # noqa: PLC0415

            support = bool(
                getattr(llama_backend, "llama_supports_gpu_offload", lambda: False)()
            )
        except Exception as exc:
            support_error = f"{type(exc).__name__}: {exc}"
    return {
        "import_ok": llama_spec is not None,
        "origin": llama_spec.origin if llama_spec else None,
        "version": llama_version,
        "gpu_offload_supported": support,
        "gpu_offload_support_error": support_error,
    }


def collect_gpu_offload_receipts() -> JsonDict:  # pragma: no cover
    torch_cuda: JsonDict = {"import_ok": False, "available": False, "device_count": 0}
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: PLC0415

            torch_cuda = {
                "import_ok": True,
                "version": getattr(torch, "__version__", "unknown"),
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        except Exception as exc:
            torch_cuda["error"] = f"{type(exc).__name__}: {exc}"

    value = {
        "gpu_visible": False,
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout_s=10.0,
        ),
        "cuda_runtime": _run_command(["nvidia-smi"], timeout_s=10.0),
        "rocm_smi": _run_command(["rocm-smi", "--showdriverversion"], timeout_s=10.0),
        "nvcc": _run_command(["nvcc", "--version"], timeout_s=10.0),
        "torch_cuda": torch_cuda,
        "llama_cpp": _llama_cpp_import_receipt(),
        "llama_cpp_python_distribution": "llama-cpp-python",
        "offload_settings": dict(OFFLOAD_CONFIG),
    }
    value["gpu_visible"] = bool(
        value["nvidia_smi"].get("ok")
        or torch_cuda.get("available")
        or torch_cuda.get("device_count", 0)
    )
    return _wrap("gpu_offload_receipts", value)


def default_generation_probe(
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    offload_config: Mapping[str, Any],
    timeout_s: float = 900.0,
) -> JsonDict:  # pragma: no cover
    code = r"""
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from llama_cpp import Llama


def sha16(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
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
    return {"ok": result.returncode == 0, "stderr": result.stderr.strip(), "gpus": gpus}


def total_used(snapshot):
    return sum(int(row.get("memory_used_mb", 0)) for row in snapshot.get("gpus", []))


def output_text(result):
    if isinstance(result, dict) and result.get("choices"):
        first = result["choices"][0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    return str(result)


model_path = Path(sys.argv[1])
prompt = sys.argv[2]
n_gpu_layers = int(sys.argv[3])
n_ctx = int(sys.argv[4])
max_tokens = int(sys.argv[5])
seed = int(sys.argv[6])
started = time.perf_counter()
before = gpu_snapshot()
payload = {
    "runtime_ready": False,
    "status": "blocked_runtime_load_failed",
    "wall_clock_s": 0.0,
    "prompt_checksum": sha16(prompt),
    "output_checksum": None,
    "output_text_preview": "",
    "command": ["llama_cpp.Llama", str(model_path)],
    "config": {
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    },
    "gpu_memory_receipts": {
        "before": before,
        "after_load": None,
        "after_generate": None,
        "max_memory_delta_mb": 0,
        "offload_evidence": False,
    },
    "llama_cpp_gpu_offload_supported": None,
    "traceback": None,
}
try:
    try:
        from llama_cpp import llama_cpp as llama_backend
        payload["llama_cpp_gpu_offload_supported"] = bool(
            getattr(llama_backend, "llama_supports_gpu_offload", lambda: False)()
        )
    except Exception as exc:
        payload["llama_cpp_gpu_offload_support_error"] = f"{type(exc).__name__}: {exc}"
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        seed=seed,
        verbose=False,
    )
    after_load = gpu_snapshot()
    response = llm(prompt, max_tokens=max_tokens, temperature=0.0)
    after_generate = gpu_snapshot()
    text = output_text(response)
    max_delta = max(total_used(after_load), total_used(after_generate)) - total_used(before)
    offload_evidence = bool(
        n_gpu_layers != 0 and (payload["llama_cpp_gpu_offload_supported"] or max_delta > 128)
    )
    payload["gpu_memory_receipts"] = {
        "before": before,
        "after_load": after_load,
        "after_generate": after_generate,
        "max_memory_delta_mb": max_delta,
        "offload_evidence": offload_evidence,
    }
    payload["output_text_preview"] = text[:120]
    payload["output_checksum"] = sha16(text)
    if text.strip():
        payload["runtime_ready"] = True
        payload["status"] = "generation_ready"
    else:
        payload["status"] = "blocked_empty_generation"
except Exception:
    payload["traceback"] = traceback.format_exc()
    payload["status"] = "blocked_runtime_probe_failed"
finally:
    payload["wall_clock_s"] = round(time.perf_counter() - started, 6)

print(json.dumps(payload, sort_keys=True))
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(model_spec["model_path"]),
        prompt,
        str(offload_config["n_gpu_layers"]),
        str(offload_config["n_ctx"]),
        str(offload_config["max_tokens"]),
        str(offload_config["seed"]),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except Exception as exc:
        return _normal_probe_failure(
            model_spec=model_spec,
            prompt=prompt,
            offload_config=offload_config,
            status="blocked_runtime_probe_subprocess_failed",
            outcome=f"{type(exc).__name__}: {exc}",
            started=started,
        )

    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        parsed = _normal_probe_failure(
            model_spec=model_spec,
            prompt=prompt,
            offload_config=offload_config,
            status="blocked_runtime_probe_parse_failed",
            outcome=(result.stderr or result.stdout)[-2000:],
            started=started,
        )
    parsed["returncode"] = result.returncode
    parsed["stderr_tail"] = result.stderr[-2000:]
    parsed["command"] = command[:2] + ["<probe-code>", str(model_spec["model_path"])]
    parsed.setdefault("wall_clock_s", round(time.perf_counter() - started, 6))
    return parsed


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
        "command": [sys.executable, "-c", "<probe-code>", str(model_spec["model_path"])],
        "config": dict(offload_config),
        "gpu_memory_receipts": {
            "before": None,
            "after_load": None,
            "after_generate": None,
            "max_memory_delta_mb": 0,
            "offload_evidence": False,
        },
        "outcome": outcome,
        "traceback": traceback.format_exc(),
    }


def default_smoke_tests() -> list[JsonDict]:
    return []


def _gpu_visible_from(gpu_receipts: Mapping[str, Any]) -> bool:
    value = gpu_receipts.get("value") if isinstance(gpu_receipts.get("value"), Mapping) else {}
    return bool(value.get("gpu_visible")) if isinstance(value, Mapping) else False


def _llama_cpp_import_ok(gpu_receipts: Mapping[str, Any]) -> bool:
    value = gpu_receipts.get("value") if isinstance(gpu_receipts.get("value"), Mapping) else {}
    llama = value.get("llama_cpp", {}) if isinstance(value, Mapping) else {}
    return bool(llama.get("import_ok")) if isinstance(llama, Mapping) else False


def _precondition_blockers(
    *, gpu_receipts: Mapping[str, Any], model_specs: Mapping[str, JsonDict]
) -> list[str]:
    blockers: list[str] = []
    if not _gpu_visible_from(gpu_receipts):
        blockers.append("gpu_not_visible")
    if not _llama_cpp_import_ok(gpu_receipts):
        blockers.append("llama_cpp_unavailable")
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
        "prompt_checksum": str(receipt.get("prompt_checksum") or sha16(MINIMAL_PROMPT)),
        "output_checksum": receipt.get("output_checksum"),
        "output_text_preview": str(receipt.get("output_text_preview") or "")[:120],
        "gpu_memory_receipts": _normalise_gpu_memory_receipts(
            receipt.get("gpu_memory_receipts")
        ),
        "traceback": receipt.get("traceback"),
        "returncode": receipt.get("returncode"),
        "stderr_tail": receipt.get("stderr_tail"),
    }
    if normal["runtime_ready"] and normal["wall_clock_s"] < MIN_LIVE_GENERATION_DURATION_S:
        raise ValueError(
            f"sub-second live generation duration for {normal['role']}: {normal['wall_clock_s']}"
        )
    if normal["runtime_ready"] and normal["gpu_memory_receipts"].get("offload_evidence") is True:
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
    out.setdefault("after_load", None)
    out.setdefault("after_generate", None)
    out.setdefault("max_memory_delta_mb", 0)
    out.setdefault("offload_evidence", False)
    return out


def _run_model_generation(
    *,
    model_specs: Mapping[str, JsonDict],
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
    model_specs: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
    blockers: Sequence[str],
) -> JsonDict:
    total, used, free = shutil.disk_usage(root)
    resolved_roles = [role for role, spec in model_specs.items() if spec.get("model_path")]
    gpu_value = gpu_receipts.get("value") if isinstance(gpu_receipts.get("value"), Mapping) else {}
    value = {
        "run_date_utc": _utc_run_date(),
        "run_on_or_after_20260706_utc": _utc_run_date() >= "20260706",
        "gpu_visibility_checked": True,
        "gpu_visible": _gpu_visible_from(gpu_receipts),
        "driver_cuda_or_rocm_checked": True,
        "driver_cuda_or_rocm": {
            "nvidia_smi": dict(gpu_value.get("nvidia_smi", {}))
            if isinstance(gpu_value, Mapping)
            and isinstance(gpu_value.get("nvidia_smi"), Mapping)
            else {},
            "cuda_runtime": dict(gpu_value.get("cuda_runtime", {}))
            if isinstance(gpu_value, Mapping)
            and isinstance(gpu_value.get("cuda_runtime"), Mapping)
            else {},
            "rocm_smi": dict(gpu_value.get("rocm_smi", {}))
            if isinstance(gpu_value, Mapping)
            and isinstance(gpu_value.get("rocm_smi"), Mapping)
            else {},
        },
        "llama_cpp_checked": True,
        "llama_cpp": dict(gpu_value.get("llama_cpp", {}))
        if isinstance(gpu_value, Mapping) and isinstance(gpu_value.get("llama_cpp"), Mapping)
        else {},
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
    generation_receipts: Mapping[str, JsonDict],
) -> JsonDict:
    base = dict(gpu_receipts.get("value", {})) if isinstance(gpu_receipts.get("value"), Mapping) else {}
    base["offload_settings"] = dict(base.get("offload_settings") or OFFLOAD_CONFIG)
    base["per_model"] = {
        role: {
            "n_gpu_layers": receipt.get("config", {}).get("n_gpu_layers"),
            "n_ctx": receipt.get("config", {}).get("n_ctx"),
            "max_tokens": receipt.get("config", {}).get("max_tokens"),
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
            f"complete: sota_offload_ready=true via {ready_list}",
            "sota_offload_ready=true because live mandated SOTA GGUF generation or "
            f"scoring completed with GPU-offload evidence for {ready_list}.",
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
    first = blockers[0] if blockers else "no_live_generation_receipt"
    return (
        False,
        f"blocked_preconditions: sota_offload_ready=false {first}",
        "sota_offload_ready=false because no mandated SOTA GGUF completed live "
        f"generation/scoring with GPU-offload evidence; blocked_preconditions={blockers}",
    )


def build_artifact(
    *,
    root: Path,
    gpu_receipts: JsonDict,
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
            model_specs=model_specs,
            cached_pair_provider=cached_pair_provider,
            blockers=precondition_blockers,
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "sota_offload_ready": ready,
        "sota_offload_ready_principle": ready_principle,
        "duration_receipts": _duration_receipts(
            generation_receipts=generation_receipts,
            duration_s=duration_s,
        ),
        "gpu_offload_receipts": _gpu_offload_receipts(
            gpu_receipts=gpu_receipts,
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
    if not isinstance(artifact.get("sota_offload_ready"), bool):
        errors.append("sota_offload_ready must be a bare bool")
    elif artifact["sota_offload_ready"] and substrate != LIVE_INFERENCE_SUBSTRATE:
        errors.append("ready artifact must use live_llm_inference_local_gguf_sota")
    elif artifact.get("sota_offload_ready") is False and substrate != BLOCKED_INFERENCE_SUBSTRATE:
        errors.append("blocked artifact must use blocked_preconditions_with_no_quality_claim")
    if not artifact.get("sota_offload_ready_principle"):
        errors.append("sota_offload_ready_principle must be non-empty")
    if _wrapped_value(artifact, "no_quality_claim") is not True:
        errors.append("no_quality_claim.value must be true")
    if not isinstance(artifact.get("tests_run"), list):
        errors.append("tests_run must be a list")

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
    gpu_receipts_provider: GpuReceiptsProvider = collect_gpu_offload_receipts,
    generation_probe: GenerationProbe = default_generation_probe,
    smoke_tests_provider: SmokeTestsProvider = default_smoke_tests,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    artifact_path = Path(artifact_path)
    gpu_receipts = gpu_receipts_provider()
    model_specs = {
        str(spec["role"]): _model_spec_receipt(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }
    precondition_blockers = _precondition_blockers(
        gpu_receipts=gpu_receipts,
        model_specs=model_specs,
    )
    generation_receipts = _run_model_generation(
        model_specs=model_specs,
        generation_probe=generation_probe,
        precondition_blockers=precondition_blockers,
    )
    artifact = build_artifact(
        root=root,
        gpu_receipts=gpu_receipts,
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
                    "test_experiment_5284_sota_runtime_offload_receipt_repair_v483.py -q"
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
