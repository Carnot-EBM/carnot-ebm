"""Build the Exp 3268 SOTA receipt methodology supplement v1 artifact.

Spec refs: REQ-REPORT-3268, SCENARIO-REPORT-3268.

This supplement exists because the prior `.302` SOTA GGUF receipt proved that a
mandated local GGUF could be loaded and sampled, but its wall-clock duration was
too short to serve as a clean headline evidence row. The builder therefore keeps
the boundary explicit: either it captures a fresh, local, CUDA-offloaded receipt
whose total live inference duration reaches the 60 second floor, or it writes a
complete non-eligible artifact that explains why downstream gates must continue
to treat the prior receipt as methodology-bounded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS
from carnot.reporting.llama_cpp_cuda_receipt_smoke_3262 import (
    _default_cache_roots,
    _json_from_last_line,
    _parse_offloaded_layers,
    _read_json,
    _reproducibility_checksum,
    _run_command,
    _safe_int,
    _selected_python,
    _stderr,
    _summarize,
    _write_json,
)
from carnot.reporting.sota_gguf_receipt_3263 import (
    _candidate_records,
    _file_evidence,
    _select_candidate,
)


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.sota_receipt_methodology_supplement.v1"
EXPERIMENT_ID = "exp3268"
TASK_ID = "exp3268-sota-receipt-methodology-supplement-v1"
ARTIFACT = "experiment_3268_sota_receipt_methodology_supplement_v1"
MILESTONE = "2026.05.303"
RANDOM_SEED = 3268
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_MAX_TOKENS_PER_CALL = 512
DEFAULT_MAX_GENERATION_CALLS = 16
DEFAULT_DURATION_FLOOR_S = 60.0
DEFAULT_PROMPT = (
    "Exp 3268 SOTA GGUF methodology receipt. Produce a deterministic local "
    "CUDA receipt paragraph with numbered evidence clauses and no markdown."
)

OUTPUT_REL_PATH = Path("results/experiment_3268_sota_receipt_methodology_supplement_v1.json")
EXP3263_REL_PATH = Path("results/experiment_3263_sota_gguf_receipt_v9.json")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
_MODEL_BY_ID = {str(model["hf_id"]): dict(model) for model in SOTA_GGUF_MODELS}

CUDA_PROBE_CODE = r'''
import importlib.util
import json
import sys

print("exp3268_cuda_probe")
payload = {
    "python": sys.executable,
    "torch_import_ok": False,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_name": "",
    "llama_cpp_import_ok": False,
    "llama_cpp_supports_gpu_offload": False,
    "llama_cpp_system_info": "",
    "probe_error": "",
}
try:
    import torch

    payload["torch_import_ok"] = True
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_device_count"] = int(torch.cuda.device_count())
    if payload["cuda_available"] and payload["cuda_device_count"] > 0:
        payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
except Exception as exc:
    payload["probe_error"] = f"{type(exc).__name__}: {exc}"

try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp import llama_cpp as low

    del Llama
    supports = getattr(low, "llama_supports_gpu_offload", lambda: False)
    system_info_fn = getattr(low, "llama_print_system_info", lambda: b"")
    raw_info = system_info_fn()
    system_info = raw_info.decode() if isinstance(raw_info, bytes) else str(raw_info)
    payload["llama_cpp_import_ok"] = True
    payload["llama_cpp_supports_gpu_offload"] = bool(supports())
    payload["llama_cpp_version"] = getattr(llama_cpp, "__version__", None)
    spec = importlib.util.find_spec("llama_cpp")
    payload["llama_cpp_origin"] = spec.origin if spec is not None else ""
    payload["llama_cpp_system_info"] = system_info
except Exception as exc:
    existing = payload.get("probe_error") or ""
    payload["probe_error"] = (existing + "; " if existing else "") + f"{type(exc).__name__}: {exc}"

print(json.dumps(payload, sort_keys=True))
'''

WORKER_CODE = r'''
import argparse
import json
import os
import subprocess
import threading
import time


def _gpu_memory():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            rows.append({"index": int(parts[0]), "memory_used_mib": int(parts[1])})
        except ValueError:
            continue
    return rows


def _max_used(samples):
    values = []
    for sample in samples:
        rows = sample if isinstance(sample, list) else []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("memory_used_mib"), int):
                values.append(row["memory_used_mib"])
    return max(values) if values else 0


def _response_text(raw):
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


parser = argparse.ArgumentParser()
parser.add_argument("--exp3268-sota-methodology-worker", action="store_true")
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-tokens-per-call", type=int, required=True)
parser.add_argument("--max-generation-calls", type=int, required=True)
parser.add_argument("--n-gpu-layers", type=int, required=True)
parser.add_argument("--target-duration-s", type=float, required=True)
args = parser.parse_args()

llm = None
started = time.monotonic()
try:
    from llama_cpp import Llama

    baseline_rows = _gpu_memory()
    baseline_used = _max_used([baseline_rows])
    main_gpu = int(os.environ.get("CARNOT_SOTA_MAIN_GPU", "0"))
    llm = Llama(
        model_path=args.model_path,
        n_ctx=4096,
        n_batch=256,
        n_ubatch=128,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=main_gpu,
        verbose=True,
    )
    after_load_rows = _gpu_memory()
    during_samples = []
    outputs = []
    total_tokens = 0
    usage_rows = []
    live_generation_calls = 0

    while (
        time.monotonic() - started < args.target_duration_s
        and live_generation_calls < args.max_generation_calls
    ):
        live_generation_calls += 1
        call_prompt = (
            f"{args.prompt}\nCall {live_generation_calls}: write receipt clauses "
            "1 through 80 as compact sentences."
        )
        stop_event = threading.Event()

        def monitor():
            while not stop_event.wait(0.05):
                during_samples.append(_gpu_memory())

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        raw = llm(
            call_prompt,
            max_tokens=args.max_tokens_per_call,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.0,
            seed=args.seed + live_generation_calls,
        )
        stop_event.set()
        thread.join(timeout=1.0)
        text = _response_text(raw).strip()
        if text:
            outputs.append(text)
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, int):
            completion_tokens = len(text.split()) if text else 0
        total_tokens += int(completion_tokens)
        usage_rows.append(usage)

    after_generate_rows = _gpu_memory()
    if not during_samples:
        during_samples.append(after_generate_rows)

    output = "\n".join(outputs).strip()
    used = max(_max_used(during_samples), _max_used([after_load_rows]), _max_used([after_generate_rows]))
    print(
        json.dumps(
            {
                "ok": bool(output) and total_tokens > 0,
                "model_id": args.model_id,
                "load_status": "loaded",
                "generation_status": "generated" if output and total_tokens > 0 else "empty_response",
                "output_text": output,
                "tokens_generated": int(total_tokens),
                "n_gpu_layers_requested": args.n_gpu_layers,
                "gpu_layers_offloaded": 0,
                "gpu_mem_baseline_mib": int(baseline_used),
                "gpu_mem_used_mib": int(used),
                "gpu_mem_delta_mib": int(max(0, used - baseline_used)),
                "gpu_memory": {
                    "baseline": baseline_rows,
                    "after_load": after_load_rows,
                    "during_generate": during_samples[-10:],
                    "after_generate": after_generate_rows,
                },
                "usage": {
                    "calls": usage_rows,
                    "completion_tokens": int(total_tokens),
                },
                "duration_s": round(time.monotonic() - started, 6),
                "live_generation_calls": int(live_generation_calls),
                "target_duration_s": float(args.target_duration_s),
            },
            sort_keys=True,
        )
    )
except Exception as exc:
    print(
        json.dumps(
            {
                "ok": False,
                "model_id": args.model_id,
                "load_status": "failed",
                "generation_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "output_text": "",
                "tokens_generated": 0,
                "n_gpu_layers_requested": args.n_gpu_layers,
                "gpu_layers_offloaded": 0,
                "gpu_mem_baseline_mib": 0,
                "gpu_mem_used_mib": 0,
                "gpu_mem_delta_mib": 0,
                "duration_s": round(time.monotonic() - started, 6),
                "live_generation_calls": 0,
                "target_duration_s": float(args.target_duration_s),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1)
finally:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
'''


def _safe_float(value: Any) -> float:
    """Convert JSON-ish durations without letting malformed upstream data crash."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_nvidia_smi_csv(text: str) -> list[JsonDict]:
    """Parse the narrow `nvidia-smi` CSV shape used by the precondition check."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index = _safe_int(parts[0])
        total = _safe_int(parts[2])
        used = _safe_int(parts[3])
        util = _safe_int(parts[4])
        if index is None or total is None or used is None or util is None:
            continue
        rows.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mib": total,
                "memory_used_mib": used,
                "utilization_gpu_pct": util,
                "driver_version": parts[5],
            }
        )
    return rows


def _probe_nvidia_smi(command_runner: CommandRunner) -> JsonDict:
    """REQ-REPORT-3268: capture host GPU health before any model is loaded."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    rows = _parse_nvidia_smi_csv(str(result.get("stdout") or result.get("stdout_summary") or ""))
    passed = result.get("returncode") == 0 and bool(rows)
    return {
        "name": "nvidia_smi",
        "passed": passed,
        "gpu_count": len(rows),
        "gpus": rows,
        "returncode": result.get("returncode"),
        "stderr_summary": _summarize(_stderr(result), limit=1000),
        "command_hash": _reproducibility_checksum({"command": command}),
    }


def _probe_selected_python_cuda(
    *,
    selected_python: str,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """REQ-REPORT-3268: prove the selected venv sees CUDA and CUDA llama.cpp."""

    command = [selected_python, "-c", CUDA_PROBE_CODE, "--exp3268_cuda_probe"]
    result = command_runner(command, timeout_s=60, env=dict(env))
    payload = _json_from_last_line(result)
    cuda_count = _safe_int(payload.get("cuda_device_count")) or 0
    passed = (
        result.get("returncode") == 0
        and payload.get("cuda_available") is True
        and cuda_count > 0
        and payload.get("llama_cpp_import_ok") is True
        and payload.get("llama_cpp_supports_gpu_offload") is True
    )
    return {
        "name": "selected_python_cuda",
        "passed": passed,
        "selected_python": selected_python,
        "cuda_available": payload.get("cuda_available") is True,
        "cuda_device_count": cuda_count,
        "cuda_device_name": str(payload.get("cuda_device_name") or ""),
        "llama_cpp_import_ok": payload.get("llama_cpp_import_ok") is True,
        "llama_cpp_supports_gpu_offload": payload.get("llama_cpp_supports_gpu_offload") is True,
        "llama_cpp_system_info": str(payload.get("llama_cpp_system_info") or ""),
        "returncode": result.get("returncode"),
        "stderr_summary": _summarize(_stderr(result), limit=1000),
        "probe_error": str(payload.get("probe_error") or ""),
        "command_hash": _reproducibility_checksum({"command": command}),
    }


def _missing_model_spec(model_id: str, candidate_count: int) -> JsonDict:
    """Create the explicit missing-model row required by the supplement contract."""

    spec = _MODEL_BY_ID.get(model_id, {})
    return {
        "model_id": model_id,
        "name": spec.get("name") or model_id.split("/", 1)[-1],
        "role": spec.get("role") or "unknown",
        "expected_quantization": spec.get("quantization") or "Q4_K_M",
        "cached": False,
        "model_path": None,
        "candidate_count": int(candidate_count),
    }


def _resolve_mandated_ggufs(cache_roots: Sequence[str | Path]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Resolve local mandated GGUF files while preserving explicit missing rows."""

    roots = [Path(root).expanduser() for root in cache_roots]
    available: list[JsonDict] = []
    missing: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        records = _candidate_records(model_id, roots)
        selected = _select_candidate(records)
        if selected is None:
            missing.append(_missing_model_spec(model_id, len(records)))
            continue
        path = Path(str(selected["path"]))
        available.append(
            {
                "model_id": model_id,
                "path": str(path),
                "filename": path.name,
                "size_bytes": int(selected.get("size_bytes") or 0),
                "candidate_count": len(records),
                "candidate_paths": [str(record["path"]) for record in records],
                "file_evidence": _file_evidence(path),
            }
        )
    return available, missing


def _run_model_worker(
    *,
    selected_python: str,
    model: Mapping[str, Any],
    n_gpu_layers: int,
    max_tokens_per_call: int,
    max_generation_calls: int,
    target_duration_s: float,
    random_seed: int,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Run one selected-Python subprocess that spends real inference on a GGUF."""

    command = [
        selected_python,
        "-c",
        WORKER_CODE,
        "--exp3268-sota-methodology-worker",
        "--model-id",
        str(model["model_id"]),
        "--model-path",
        str(model["path"]),
        "--prompt",
        DEFAULT_PROMPT,
        "--seed",
        str(int(random_seed)),
        "--max-tokens-per-call",
        str(int(max_tokens_per_call)),
        "--max-generation-calls",
        str(int(max_generation_calls)),
        "--n-gpu-layers",
        str(int(n_gpu_layers)),
        "--target-duration-s",
        str(float(target_duration_s)),
    ]
    worker_env = dict(env)
    worker_env["PYTHONHASHSEED"] = str(int(random_seed))
    result = command_runner(command, timeout_s=1800, env=worker_env)
    payload = _json_from_last_line(result)
    stderr_full = _stderr(result)
    parsed_layers = _parse_offloaded_layers(stderr_full)
    if parsed_layers and not _safe_int(payload.get("gpu_layers_offloaded")):
        payload["gpu_layers_offloaded"] = parsed_layers
    return {
        "attempted": True,
        "returncode": result.get("returncode"),
        "command_hash": _reproducibility_checksum({"command": command}),
        "stderr_summary": _summarize(stderr_full),
        "payload": payload,
    }


def _receipt_from_worker(model: Mapping[str, Any], worker: Mapping[str, Any]) -> JsonDict:
    """Normalize one live worker payload into an auditable receipt row."""

    payload = dict(worker.get("payload")) if isinstance(worker.get("payload"), Mapping) else {}
    stderr_summary = str(worker.get("stderr_summary") or "")
    tokens = _safe_int(payload.get("tokens_generated")) or 0
    layers = _safe_int(payload.get("gpu_layers_offloaded")) or _parse_offloaded_layers(stderr_summary)
    baseline = _safe_int(payload.get("gpu_mem_baseline_mib")) or 0
    used = _safe_int(payload.get("gpu_mem_used_mib")) or 0
    delta = _safe_int(payload.get("gpu_mem_delta_mib"))
    if delta is None:
        delta = max(0, used - baseline)
    output = str(payload.get("output_text") or "").strip()
    clean = (
        worker.get("returncode") == 0
        and bool(output)
        and tokens > 0
        and int(layers or 0) > 0
        and used > baseline
        and Path(str(model["path"])).is_file()
    )
    return {
        "model_id": str(model["model_id"]),
        "model_path": str(model["path"]),
        "filename": str(model["filename"]),
        "size_bytes": int(model.get("size_bytes") or 0),
        "local_file_evidence": model.get("file_evidence") if isinstance(model, Mapping) else {},
        "model_load_evidence": {
            "runtime": "llama_cpp",
            "load_status": str(payload.get("load_status") or "unknown"),
            "n_gpu_layers_requested": _safe_int(payload.get("n_gpu_layers_requested")),
            "duration_s": _safe_float(payload.get("duration_s")),
        },
        "generation_evidence": {
            "generation_status": str(payload.get("generation_status") or "unknown"),
            "output_nonempty": bool(output),
            "output_preview": output[:240],
            "tokens_generated": int(tokens),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {},
            "live_generation_calls": _safe_int(payload.get("live_generation_calls")) or 0,
        },
        "gpu_evidence": {
            "gpu_layers_offloaded": int(layers or 0),
            "gpu_mem_baseline_mib": int(baseline),
            "gpu_mem_used_mib": int(used),
            "gpu_mem_delta_mib": int(delta),
        },
        "worker_attempt": {
            "attempted": bool(worker.get("attempted")),
            "returncode": worker.get("returncode"),
            "command_hash": str(worker.get("command_hash") or ""),
            "stderr_summary": stderr_summary,
        },
        "methodology_clean": clean,
    }


def _model_specs(
    available_models: Sequence[Mapping[str, Any]],
    missing_model_specs: Sequence[Mapping[str, Any]],
    n_gpu_layers: int,
) -> JsonDict:
    """Build the top-level model contract that names every mandated GGUF."""

    available_by_id = {str(model["model_id"]): model for model in available_models}
    missing_by_id = {str(model["model_id"]): model for model in missing_model_specs}
    mandated_models: JsonDict = {}
    for model_id in MANDATED_MODEL_IDS:
        spec = _MODEL_BY_ID.get(model_id, {})
        available = available_by_id.get(model_id)
        missing = missing_by_id.get(model_id, {})
        mandated_models[model_id] = {
            "name": spec.get("name") or missing.get("name") or model_id.split("/", 1)[-1],
            "role": spec.get("role") or missing.get("role") or "unknown",
            "expected_quantization": spec.get("quantization")
            or missing.get("expected_quantization")
            or "Q4_K_M",
            "cached": available is not None,
            "model_path": str(available["path"]) if available else None,
            "size_bytes": int(available.get("size_bytes") or 0) if available else 0,
        }
    return {
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "mandated_models": mandated_models,
        "runtime": "llama_cpp",
        "n_gpu_layers_requested": int(n_gpu_layers),
        "duration_floor_s": DEFAULT_DURATION_FLOOR_S,
        "prompt": DEFAULT_PROMPT,
    }


def _prior_receipt_boundary(prior: Mapping[str, Any]) -> JsonDict:
    """Summarize why the `.302` receipt is evidence but not clean reuse."""

    duration = _safe_float(prior.get("duration_s"))
    ready = prior.get("sota_gguf_receipt_ready") is True
    return {
        "path": str(REPO_ROOT / EXP3263_REL_PATH),
        "prior_sota_gguf_receipt_ready": ready,
        "prior_duration_s": duration,
        "prior_flagged_adversarial": prior.get("flagged_adversarial") is True,
        "clean_reuse_allowed": bool(ready and duration >= DEFAULT_DURATION_FLOOR_S),
        "methodology_boundary": "prior_receipt_not_clean_headline_evidence"
        if duration < DEFAULT_DURATION_FLOOR_S
        else "prior_receipt_duration_floor_met",
    }


def _models_used(
    available_models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions_ok: bool,
) -> list[JsonDict]:
    """Expose which cached models were actually attempted versus only inventoried."""

    receipt_by_id = {str(row["model_id"]): row for row in receipts}
    rows: list[JsonDict] = []
    for model in available_models:
        receipt = receipt_by_id.get(str(model["model_id"]))
        rows.append(
            {
                "model_id": str(model["model_id"]),
                "model_path": str(model["path"]),
                "filename": str(model["filename"]),
                "cached": True,
                "attempted_live_receipt": preconditions_ok and receipt is not None,
                "clean_row": bool(receipt and receipt.get("methodology_clean")),
            }
        )
    return rows


def _methodology_findings(
    *,
    preconditions_ok: bool,
    nvidia_probe: Mapping[str, Any],
    cuda_probe: Mapping[str, Any],
    available_models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    duration_floor_s: float,
    clean_eligible: bool,
) -> list[str]:
    """Explain evidence hygiene separately from model quality or output content."""

    findings: list[str] = []
    if nvidia_probe.get("passed") is not True:
        findings.append("nvidia_smi_unavailable")
    if cuda_probe.get("passed") is not True:
        findings.append("selected_python_cuda_unavailable")
    if not available_models:
        findings.append("no_mandated_sota_gguf_cached")
    if available_models and not preconditions_ok:
        findings.append("live_receipt_preconditions_failed")
    if receipts and not any(row.get("methodology_clean") for row in receipts):
        findings.append("no_methodology_clean_live_receipts")
    if duration_s < duration_floor_s:
        findings.append(f"duration_floor_not_met: duration_s={duration_s} < {duration_floor_s}")
    if clean_eligible:
        return ["methodology_clean_live_receipt_available"]
    return findings or ["clean_sota_receipt_not_eligible"]


def _honest_verdict(*, clean_eligible: bool, findings: Sequence[str]) -> str:
    """Return a terminal verdict using the success-style prefix required by specs."""

    if clean_eligible:
        return (
            "complete: sota_receipt_methodology_supplement_v1_ready=true; "
            "clean_sota_receipt_eligible=true"
        )
    return (
        "complete: sota_receipt_methodology_supplement_v1_ready=true; "
        "clean_sota_receipt_eligible=false; "
        f"methodology_findings={';'.join(findings)}"
    )


def build_artifact(
    *,
    project_root: str | Path,
    cache_roots: Sequence[str | Path] | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    max_tokens_per_call: int = DEFAULT_MAX_TOKENS_PER_CALL,
    max_generation_calls: int = DEFAULT_MAX_GENERATION_CALLS,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
) -> JsonDict:
    """REQ-REPORT-3268: build the SOTA receipt methodology supplement."""

    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    roots = [Path(path) for path in (cache_roots or _default_cache_roots(root, merged_env))]

    prior = _read_json(root / EXP3263_REL_PATH)
    prior_boundary = _prior_receipt_boundary(prior)
    nvidia_probe = _probe_nvidia_smi(command_runner)
    cuda_probe = _probe_selected_python_cuda(
        selected_python=selected,
        env=merged_env,
        command_runner=command_runner,
    )
    preconditions_ok = nvidia_probe.get("passed") is True and cuda_probe.get("passed") is True

    available_models, missing_model_specs = _resolve_mandated_ggufs(roots)
    receipts: list[JsonDict] = []
    if preconditions_ok:
        for model in available_models:
            worker = _run_model_worker(
                selected_python=selected,
                model=model,
                n_gpu_layers=int(n_gpu_layers),
                max_tokens_per_call=int(max_tokens_per_call),
                max_generation_calls=int(max_generation_calls),
                target_duration_s=float(duration_floor_s),
                random_seed=int(random_seed),
                env=merged_env,
                command_runner=command_runner,
            )
            receipts.append(_receipt_from_worker(model, worker))

    duration_s = round(max(0.0, monotonic() - start), 6)
    receipt_duration_floor_met = duration_s >= float(duration_floor_s)
    clean_rows = [row for row in receipts if row.get("methodology_clean")]
    clean_eligible = bool(clean_rows) and receipt_duration_floor_met
    gpu_mem_used_mib = (
        max(int(row["gpu_evidence"]["gpu_mem_used_mib"]) for row in clean_rows)
        if clean_rows
        else 0
    )
    tokens_generated = (
        sum(int(row["generation_evidence"]["tokens_generated"]) for row in clean_rows)
        if clean_rows
        else 0
    )
    findings = _methodology_findings(
        preconditions_ok=preconditions_ok,
        nvidia_probe=nvidia_probe,
        cuda_probe=cuda_probe,
        available_models=available_models,
        receipts=receipts,
        duration_s=duration_s,
        duration_floor_s=float(duration_floor_s),
        clean_eligible=clean_eligible,
    )
    model_specs = _model_specs(
        available_models=available_models,
        missing_model_specs=missing_model_specs,
        n_gpu_layers=int(n_gpu_layers),
    )
    models_used = _models_used(
        available_models=available_models,
        receipts=receipts,
        preconditions_ok=preconditions_ok,
    )
    inference_substrate = "live_llm_inference" if receipts else "aggregation_from_upstream_artifacts"
    checksum = _reproducibility_checksum(
        {
            "clean_sota_receipt_eligible": clean_eligible,
            "duration_s": duration_s,
            "gpu_mem_used_mib": gpu_mem_used_mib,
            "methodology_findings": findings,
            "model_specs": model_specs,
            "models_used": models_used,
            "missing_model_specs": missing_model_specs,
            "preconditions_checked": [nvidia_probe, cuda_probe],
            "random_seed": int(random_seed),
            "receipt_duration_floor_met": receipt_duration_floor_met,
            "tokens_generated": tokens_generated,
        }
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "selected_python": selected,
        "inference_substrate": inference_substrate,
        "cache_roots": [str(path) for path in roots],
        "prior_receipt_boundary": prior_boundary,
        "sota_receipt_methodology_supplement_v1_ready": True,
        "clean_sota_receipt_eligible": clean_eligible,
        "model_specs": model_specs,
        "models_used": models_used,
        "missing_model_specs": missing_model_specs,
        "preconditions_checked": [nvidia_probe, cuda_probe],
        "per_model_receipts": receipts,
        "gpu_mem_used_mib": int(gpu_mem_used_mib),
        "tokens_generated": int(tokens_generated),
        "receipt_duration_floor_met": receipt_duration_floor_met,
        "methodology_findings": findings,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(clean_eligible=clean_eligible, findings=findings),
    }


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    cache_roots: Sequence[str | Path] | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    max_tokens_per_call: int = DEFAULT_MAX_TOKENS_PER_CALL,
    max_generation_calls: int = DEFAULT_MAX_GENERATION_CALLS,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
) -> JsonDict:
    """Build and write the Exp 3268 methodology supplement artifact."""

    root = Path(project_root)
    destination = Path(output_path) if output_path is not None else root / OUTPUT_REL_PATH
    if not destination.is_absolute():
        destination = root / destination
    artifact = build_artifact(
        project_root=root,
        cache_roots=cache_roots,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        monotonic=monotonic,
        random_seed=random_seed,
        n_gpu_layers=n_gpu_layers,
        max_tokens_per_call=max_tokens_per_call,
        max_generation_calls=max_generation_calls,
        duration_floor_s=duration_floor_s,
    )
    _write_json(destination, artifact)
    return artifact


def main() -> int:
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
