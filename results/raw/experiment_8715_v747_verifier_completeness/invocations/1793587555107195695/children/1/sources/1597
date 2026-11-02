"""Exp 5271: strict local SOTA GGUF telemetry receipt harness.

Spec refs: REQ-VERIFY-5271, SCENARIO-VERIFY-5271.

This module is deliberately a receipt harness, not a hallucination detector.
It records which llama.cpp telemetry surfaces are available from the mandated
local GGUF models so later experiments can decide whether to run internal-state
verifier work. Missing hidden states or attention summaries are recorded as
capability gaps, not replaced with generated-text scoring.
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
TelemetryProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5271
EXPERIMENT_NAME = "experiment_5271_sota_telemetry_receipt_harness_v482"
RESULT_RELATIVE_PATH = Path("results/experiment_5271_sota_telemetry_receipt_harness_v482.json")
SCHEMA = "carnot.experiment_5271.sota_telemetry_receipt_harness.v482"
SPEC_REFS = ("REQ-VERIFY-5271", "SCENARIO-VERIFY-5271")
INFERENCE_SUBSTRATE = "live_llm_internal_telemetry_local_gguf_sota"
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
RANDOM_SEED = 5271
MIN_LIVE_MODEL_DURATION_S = 1.0
MINIMAL_PROMPT = "Return exactly OK."

OFFLOAD_CONFIG: JsonDict = {
    "n_gpu_layers": -1,
    "n_ctx": 512,
    "max_tokens": 2,
    "temperature": 0.0,
    "seed": RANDOM_SEED,
    "logprobs": 5,
    "logits_all": True,
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
        "Terminal Exp 5271 verdict; starts with complete: or blocked_ and states "
        "whether SOTA GGUF telemetry receipts are ready for downstream verifier experiments."
    ),
    "inference_substrate": (
        "Declares live llama.cpp-backed local SOTA GGUF internal telemetry, preventing "
        "cached text scoring or non-GGUF fallbacks from being mistaken for receipts."
    ),
    "preconditions_checked": (
        "Records GPU visibility, llama.cpp version/origin, cache paths, free disk, and "
        "local GGUF resolvability before any telemetry claim."
    ),
    "MODEL_SPECS": (
        "Records mandated SOTA GGUF model IDs, roles, quantization, local file receipts, "
        "and runtime status; legacy smoke models cannot open the headline gate."
    ),
    "telemetry_harness_ready": (
        "Bare gate for exp5272 and exp5276; true only when at least one mandated local "
        "SOTA GGUF completes live telemetry with usable internal/logprob/logit receipts."
    ),
    "telemetry_harness_ready_principle": (
        "Explains the exact ready model receipts or blockers used by downstream structured gates."
    ),
    "exposed_telemetry_fields": (
        "Per-model availability map for logits, token logprobs, hidden states, and attention "
        "summaries; unavailable internal surfaces are recorded as capability_absent rather "
        "than substituted."
    ),
    "duration_receipts": (
        "Per-model and total wall-clock receipts proving live local GGUF telemetry took real "
        "time and did not produce a sub-second live-model artifact."
    ),
    "gpu_offload_receipts": (
        "Driver/device/runtime/offload settings proving which GPU and llama.cpp offload path "
        "was visible or attempted."
    ),
    "no_quality_claim": (
        "Must be true; Exp 5271 measures telemetry availability only and makes no "
        "hallucination-detection or verifier-quality claim."
    ),
    "tests_run": (
        "Commands run to validate the harness module, new-code coverage, and repository test "
        "status, with outcomes."
    ),
    "prompt_output_checksums": (
        "Per-model prompt and output checksums prove the tiny deterministic telemetry prompt "
        "and observed output without treating text content as a verifier score."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "telemetry_harness_ready",
    "telemetry_harness_ready_principle",
    "exposed_telemetry_fields",
    "duration_receipts",
    "gpu_offload_receipts",
    "no_quality_claim",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "exposed_telemetry_fields",
    "duration_receipts",
    "gpu_offload_receipts",
    "no_quality_claim",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")
TELEMETRY_KEYS = ("logits", "token_logprobs", "hidden_states", "attention_summaries")
USABLE_AVAILABILITY = {"available"}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def sha16(value: str | bytes) -> str:
    """Return a short stable checksum for prompt/output/file receipts."""

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
        "file_receipts": None,
        "metadata": None,
        "runtime_status": "not_attempted",
        "headline_role": True,
        "smoke_label": None,
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
        receipt["blocker"] = f"{type(exc).__name__}: {exc}"
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


def collect_gpu_offload_receipts() -> JsonDict:  # pragma: no cover
    llama_spec = importlib.util.find_spec("llama_cpp")
    try:
        llama_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_version = None

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
        "torch_cuda": torch_cuda,
        "llama_cpp": {
            "import_ok": llama_spec is not None,
            "origin": llama_spec.origin if llama_spec else None,
            "version": llama_version,
        },
        "llama_cpp_python_distribution": "llama-cpp-python",
        "offload_settings": dict(OFFLOAD_CONFIG),
    }
    value["gpu_visible"] = bool(
        value["nvidia_smi"].get("ok")
        or torch_cuda.get("available")
        or torch_cuda.get("device_count", 0)
    )
    return _wrap("gpu_offload_receipts", value)


def default_telemetry_probe(
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    offload_config: Mapping[str, Any],
    timeout_s: float = 900.0,
) -> JsonDict:  # pragma: no cover
    code = r"""
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

from llama_cpp import Llama


def sha16(value):
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def numeric_list(value):
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return numeric_list(list(value.values()))
    out = []
    try:
        iterator = iter(value)
    except TypeError:
        return []
    for item in iterator:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def top_logprob_rows(value):
    if not value:
        return []
    rows = []
    for row in value:
        if isinstance(row, dict):
            rows.append({str(k): float(v) for k, v in row.items() if isinstance(v, (int, float))})
    return rows


def logits_summary(eval_logits):
    if not eval_logits:
        return {"availability": "capability_absent", "reason": "eval_logits empty"}
    final = list(eval_logits[-1])
    if not final:
        return {"availability": "capability_absent", "reason": "final logits empty"}
    top = sorted(enumerate(final), key=lambda pair: float(pair[1]), reverse=True)[:8]
    payload = json.dumps([(int(i), round(float(v), 6)) for i, v in top], sort_keys=True)
    return {
        "availability": "available",
        "steps": len(eval_logits),
        "vocab_size": len(final),
        "top_k_count": len(top),
        "top_logits_checksum": sha16(payload),
    }


model_path = Path(sys.argv[1])
prompt = sys.argv[2]
n_gpu_layers = int(sys.argv[3])
n_ctx = int(sys.argv[4])
max_tokens = int(sys.argv[5])
seed = int(sys.argv[6])
logprobs_k = int(sys.argv[7])
logits_all = sys.argv[8].lower() == "true"
started = time.perf_counter()
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
        "logprobs": logprobs_k,
        "logits_all": logits_all,
    },
    "logits": {"availability": "capability_absent", "reason": "not inspected"},
    "token_logprobs": {"availability": "capability_absent", "reason": "not inspected"},
    "hidden_states": {"availability": "capability_absent", "reason": "llama_cpp_api_no_hidden_state_export"},
    "attention_summaries": {"availability": "capability_absent", "reason": "llama_cpp_api_no_attention_export"},
    "traceback": None,
}
try:
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        seed=seed,
        logits_all=logits_all,
        verbose=False,
    )
    response = llm(prompt, max_tokens=max_tokens, temperature=0.0, logprobs=logprobs_k)
    choice = {}
    if isinstance(response, dict) and response.get("choices"):
        choice = response["choices"][0]
    text = str(choice.get("text", "")) if isinstance(choice, dict) else ""
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else {}
    token_logprobs = numeric_list(logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None)
    top_rows = top_logprob_rows(logprobs.get("top_logprobs") if isinstance(logprobs, dict) else None)
    payload["output_text_preview"] = text[:80]
    payload["output_checksum"] = sha16(text)
    if token_logprobs or top_rows:
        payload["token_logprobs"] = {
            "availability": "available",
            "token_count": len(token_logprobs),
            "top_logprobs_count": len(top_rows),
            "tokens_checksum": sha16(json.dumps(logprobs.get("tokens", []), sort_keys=True))
            if isinstance(logprobs, dict)
            else None,
        }
    else:
        payload["token_logprobs"] = {
            "availability": "capability_absent",
            "reason": "llama_cpp_response_omitted_token_logprobs",
        }
    if logits_all:
        payload["logits"] = logits_summary(getattr(llm, "eval_logits", None))
    payload["runtime_ready"] = True
    payload["status"] = "telemetry_ready"
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
        str(offload_config["logprobs"]),
        str(bool(offload_config["logits_all"])),
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
    parsed["command"] = command[:3] + ["<probe-code>", str(model_spec["model_path"])]
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
        "logits": {"availability": "runtime_error", "reason": outcome},
        "token_logprobs": {"availability": "runtime_error", "reason": outcome},
        "hidden_states": {
            "availability": "capability_absent",
            "reason": "llama_cpp_api_no_hidden_state_export",
        },
        "attention_summaries": {
            "availability": "capability_absent",
            "reason": "llama_cpp_api_no_attention_export",
        },
        "traceback": traceback.format_exc(),
    }


def _normalise_telemetry_receipt(
    receipt: Mapping[str, Any], model_spec: Mapping[str, Any]
) -> JsonDict:
    normal = {
        "role": str(model_spec["role"]),
        "hf_id": str(model_spec["hf_id"]),
        "model_path": str(model_spec["model_path"]),
        "runtime_ready": bool(receipt.get("runtime_ready")),
        "status": str(
            receipt.get("status")
            or (
                "telemetry_ready"
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
        "logits": _field_receipt(receipt.get("logits")),
        "token_logprobs": _field_receipt(receipt.get("token_logprobs")),
        "hidden_states": _field_receipt(receipt.get("hidden_states"), default="capability_absent"),
        "attention_summaries": _field_receipt(
            receipt.get("attention_summaries"), default="capability_absent"
        ),
        "traceback": receipt.get("traceback"),
        "returncode": receipt.get("returncode"),
        "stderr_tail": receipt.get("stderr_tail"),
    }
    if normal["runtime_ready"] and normal["wall_clock_s"] < MIN_LIVE_MODEL_DURATION_S:
        raise ValueError(
            f"sub-second live telemetry duration for {normal['role']}: {normal['wall_clock_s']}"
        )
    return normal


def _field_receipt(value: Any, default: str = "capability_absent") -> JsonDict:
    if isinstance(value, Mapping):
        out = dict(value)
    elif value is True:
        out = {"availability": "available"}
    else:
        out = {"availability": default}
    out.setdefault("availability", default)
    return out


def _run_model_telemetry(
    *,
    model_specs: Mapping[str, JsonDict],
    telemetry_probe: TelemetryProbe,
) -> JsonDict:
    receipts: JsonDict = {}
    for role, model_spec in model_specs.items():
        if model_spec.get("status") != "local_gguf_resolved" or not model_spec.get("model_path"):
            continue
        raw_receipt = telemetry_probe(
            model_spec=model_spec,
            prompt=MINIMAL_PROMPT,
            offload_config=OFFLOAD_CONFIG,
        )
        receipts[role] = _normalise_telemetry_receipt(raw_receipt, model_spec)
        model_spec["runtime_status"] = receipts[role]["status"]
    return receipts


def _preconditions(
    *,
    root: Path,
    gpu_receipts: Mapping[str, Any],
    model_specs: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
) -> JsonDict:
    total, used, free = shutil.disk_usage(root)
    try:
        cached_pair_preview = cached_pair_provider(gpu_indices=(0, 1)) or []
    except Exception as exc:
        cached_pair_preview = [
            {"status": "cached_sota_pair_error", "error": f"{type(exc).__name__}: {exc}"}
        ]
    resolved_roles = [role for role, spec in model_specs.items() if spec.get("model_path")]
    value = {
        "gpu_visibility_checked": True,
        "gpu_visible": bool(gpu_receipts.get("value", {}).get("gpu_visible"))
        if isinstance(gpu_receipts.get("value"), Mapping)
        else False,
        "llama_cpp_checked": True,
        "llama_cpp": (
            dict(gpu_receipts.get("value", {}).get("llama_cpp", {}))
            if isinstance(gpu_receipts.get("value"), Mapping)
            else {}
        ),
        "free_disk": {
            "path": str(root),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
        },
        "gguf_cache_paths": {
            spec["role"]: _repo_cache_path(str(spec["hf_id"])) for spec in MANDATED_MODEL_SPECS
        },
        "cached_sota_pair_preview": cached_pair_preview,
        "at_least_one_mandated_model_resolved_without_autotokenizer": bool(resolved_roles),
        "resolved_roles": resolved_roles,
        "autotokenizer_used": False,
    }
    return _wrap("preconditions_checked", value)


def _exposed_telemetry_fields(
    *,
    model_specs: Mapping[str, JsonDict],
    telemetry_receipts: Mapping[str, JsonDict],
) -> JsonDict:
    fields: JsonDict = {}
    for role in model_specs:
        if role in telemetry_receipts:
            receipt = telemetry_receipts[role]
            fields[role] = {key: dict(receipt[key]) for key in TELEMETRY_KEYS}
        else:
            status = (
                "not_attempted" if model_specs[role].get("model_path") else "missing_local_gguf"
            )
            fields[role] = {
                "logits": {"availability": status},
                "token_logprobs": {"availability": status},
                "hidden_states": {"availability": status},
                "attention_summaries": {"availability": status},
            }
    return _wrap("exposed_telemetry_fields", fields)


def _duration_receipts(
    *,
    telemetry_receipts: Mapping[str, JsonDict],
    duration_s: float,
) -> JsonDict:
    per_model = {
        role: {
            "wall_clock_s": float(receipt["wall_clock_s"]),
            "runtime_ready": bool(receipt["runtime_ready"]),
            "status": receipt["status"],
        }
        for role, receipt in telemetry_receipts.items()
    }
    return _wrap(
        "duration_receipts",
        {
            "total_wall_clock_s": round(duration_s, 6),
            "minimum_live_model_duration_s": MIN_LIVE_MODEL_DURATION_S,
            "per_model": per_model,
        },
    )


def _prompt_output_checksums(telemetry_receipts: Mapping[str, JsonDict]) -> JsonDict:
    return _wrap(
        "prompt_output_checksums",
        {
            role: {
                "prompt_checksum": receipt.get("prompt_checksum"),
                "output_checksum": receipt.get("output_checksum"),
                "output_text_preview": receipt.get("output_text_preview"),
            }
            for role, receipt in telemetry_receipts.items()
        },
    )


def _receipt_has_usable_telemetry(receipt: Mapping[str, Any]) -> bool:
    if not receipt.get("runtime_ready"):
        return False
    return any(
        isinstance(receipt.get(key), Mapping)
        and str(receipt[key].get("availability")) in USABLE_AVAILABILITY
        for key in TELEMETRY_KEYS
    )


def _verdict(
    telemetry_receipts: Mapping[str, JsonDict], model_specs: Mapping[str, JsonDict]
) -> tuple[bool, str, str]:
    ready_roles = [
        role
        for role, receipt in telemetry_receipts.items()
        if _receipt_has_usable_telemetry(receipt)
    ]
    if ready_roles:
        ready_list = ", ".join(ready_roles)
        return (
            True,
            f"complete: telemetry_receipts_ready=true via {ready_list}",
            "telemetry_harness_ready=true because live mandated SOTA GGUF telemetry "
            f"completed with usable internal/logprob/logit receipts for {ready_list}; "
            "hidden or attention gaps remain recorded as capability_absent.",
        )

    attempted = [
        f"{role}:{receipt.get('status')}:{_availability_summary(receipt)}"
        for role, receipt in telemetry_receipts.items()
    ]
    missing = [
        f"{role}:{spec.get('status')}"
        for role, spec in model_specs.items()
        if role not in telemetry_receipts
    ]
    blockers = attempted + missing
    first = blockers[0] if blockers else "no_mandated_model_resolved"
    return (
        False,
        f"blocked_telemetry_receipts_not_ready: {first}",
        "telemetry_harness_ready=false because no mandated SOTA GGUF completed live "
        f"usable telemetry; blockers={blockers}",
    )


def _availability_summary(receipt: Mapping[str, Any]) -> str:
    return ",".join(
        f"{key}={receipt.get(key, {}).get('availability')}"
        for key in TELEMETRY_KEYS
        if isinstance(receipt.get(key), Mapping)
    )


def build_artifact(
    *,
    root: Path,
    gpu_receipts: JsonDict,
    model_specs: Mapping[str, JsonDict],
    telemetry_receipts: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    ready, verdict, ready_principle = _verdict(telemetry_receipts, model_specs)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _preconditions(
            root=root,
            gpu_receipts=gpu_receipts,
            model_specs=model_specs,
            cached_pair_provider=cached_pair_provider,
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "telemetry_harness_ready": ready,
        "telemetry_harness_ready_principle": ready_principle,
        "exposed_telemetry_fields": _exposed_telemetry_fields(
            model_specs=model_specs,
            telemetry_receipts=telemetry_receipts,
        ),
        "duration_receipts": _duration_receipts(
            telemetry_receipts=telemetry_receipts,
            duration_s=duration_s,
        ),
        "gpu_offload_receipts": gpu_receipts,
        "prompt_output_checksums": _prompt_output_checksums(telemetry_receipts),
        "telemetry_probe_receipts": dict(telemetry_receipts),
        "no_quality_claim": _wrap("no_quality_claim", True),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "spec_refs": SPEC_REFS,
                "model_specs": artifact["MODEL_SPECS"]["value"],
                "telemetry_fields": artifact["exposed_telemetry_fields"]["value"],
                "prompt_checksums": artifact["prompt_output_checksums"]["value"],
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
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate.value must be {INFERENCE_SUBSTRATE}")
    if not isinstance(artifact.get("telemetry_harness_ready"), bool):
        errors.append("telemetry_harness_ready must be a bare bool")
    if not artifact.get("telemetry_harness_ready_principle"):
        errors.append("telemetry_harness_ready_principle must be non-empty")
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
                    and float(receipt.get("wall_clock_s", 0.0)) < MIN_LIVE_MODEL_DURATION_S
                ):
                    errors.append(
                        f"duration_receipts.value.per_model.{role} is below live duration floor"
                    )
    else:
        errors.append("duration_receipts.value must be an object")

    fields = _wrapped_value(artifact, "exposed_telemetry_fields")
    if isinstance(fields, Mapping):
        for role, field_map in fields.items():
            if not isinstance(field_map, Mapping):
                errors.append(f"exposed_telemetry_fields.value.{role} must be an object")
                continue
            for key in TELEMETRY_KEYS:
                row = field_map.get(key)
                if not isinstance(row, Mapping) or not row.get("availability"):
                    errors.append(
                        f"exposed_telemetry_fields.value.{role}.{key}.availability missing"
                    )
    else:
        errors.append("exposed_telemetry_fields.value must be an object")
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
    telemetry_probe: TelemetryProbe = default_telemetry_probe,
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
    telemetry_receipts = _run_model_telemetry(
        model_specs=model_specs,
        telemetry_probe=telemetry_probe,
    )
    artifact = build_artifact(
        root=root,
        gpu_receipts=gpu_receipts,
        model_specs=model_specs,
        telemetry_receipts=telemetry_receipts,
        cached_pair_provider=cached_pair_provider,
        tests_run=tests_run or [],
        duration_s=time.perf_counter() - started,
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
                    "test_experiment_5271_sota_telemetry_receipt_harness_v482.py -q"
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
