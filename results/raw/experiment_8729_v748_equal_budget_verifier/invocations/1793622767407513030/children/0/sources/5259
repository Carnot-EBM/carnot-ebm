"""Build the Exp 3263 SOTA GGUF receipt v9 artifact.

Spec refs: REQ-REPORT-3263, SCENARIO-REPORT-3263.

This module turns the small-model CUDA proof from Exp 3262 into the next
required receipt: at least one mandated frontier GGUF must be present locally,
loaded through llama.cpp with CUDA offload, and asked for a tiny deterministic
generation. The code is deliberately cache-only; a missing model is reported as
an honest block rather than downloading weights or substituting a smaller model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
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


JsonDict = dict[str, Any]
CommandRunner = Any
ClockFn = Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.sota_gguf_receipt.v9"
EXPERIMENT_ID = "exp3263"
TASK_ID = "exp3263-sota-gguf-receipt-v9"
ARTIFACT = "experiment_3263_sota_gguf_receipt_v9"
MILESTONE = "2026.05.302"
RANDOM_SEED = 3263
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_MAX_TOKENS = 16
DEFAULT_PROMPT = (
    "Exp 3263 SOTA GGUF CUDA receipt. Reply with READY and one short clause."
)

OUTPUT_REL_PATH = Path("results/experiment_3263_sota_gguf_receipt_v9.json")
EXP3262_REL_PATH = Path("results/experiment_3262_llama_cpp_cuda_receipt_smoke_v4.json")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
_MODEL_BY_ID = {str(model["hf_id"]): dict(model) for model in SOTA_GGUF_MODELS}
_QUANTIZATION_TOKENS: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q8_XL",
    "Q8_0",
    "BF16",
)

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
parser.add_argument("--exp3263-sota-gguf-worker", action="store_true")
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-tokens", type=int, required=True)
parser.add_argument("--n-gpu-layers", type=int, required=True)
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
        n_ctx=512,
        n_batch=64,
        n_ubatch=64,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=main_gpu,
        verbose=True,
    )
    after_load_rows = _gpu_memory()
    during_samples = []
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            during_samples.append(_gpu_memory())
            time.sleep(0.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    raw = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=args.seed,
    )
    stop_event.set()
    thread.join(timeout=1.0)
    after_generate_rows = _gpu_memory()
    if not during_samples:
        during_samples.append(after_generate_rows)

    output = _response_text(raw).strip()
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = len(output.split()) if output else 0
    used = max(_max_used(during_samples), _max_used([after_load_rows]), _max_used([after_generate_rows]))
    print(
        json.dumps(
            {
                "ok": bool(output),
                "model_id": args.model_id,
                "load_status": "loaded",
                "generation_status": "generated" if output and completion_tokens > 0 else "empty_response",
                "output_text": output,
                "tokens_generated": int(completion_tokens),
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
                "usage": usage,
                "duration_s": round(time.monotonic() - started, 6),
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


def _safe_model_slug(model_id: str) -> str:
    """Convert a HuggingFace model ID into a stable artifact-friendly slug."""

    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_id).strip("_")


def _model_id_from_path(path: Path) -> str:
    """Infer a HuggingFace model ID from the local HuggingFace cache layout."""

    for part in path.parts:
        if part.startswith("models--"):
            pieces = part.split("--", 2)
            if len(pieces) == 3:
                return f"{pieces[1]}/{pieces[2]}"
    return f"local/{path.stem}"


def _local_model_dirs(root: Path, model_id: str) -> list[Path]:
    """Return cache subdirectories used by HF and project-local GGUF layouts."""

    owner, name = model_id.split("/", 1)
    stripped = name.removesuffix("-GGUF")
    return [
        root / f"models--{owner}--{name}",
        root / stripped,
        root / name,
        root / stripped.lower(),
        root / name.lower(),
    ]


def _candidate_record(path: Path, model_id: str) -> JsonDict:
    """Record why a GGUF path can or cannot satisfy the mandated-model cache gate."""

    try:
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
    except OSError:  # pragma: no cover - defensive filesystem metadata handling.
        exists = False
        size = 0
    filename = path.name.lower()
    token = model_id.split("/", 1)[-1].removesuffix("-GGUF").lower()
    usable = (
        exists
        and size > 0
        and token in filename
        and "mmproj" not in filename
        and ".no_exist" not in str(path)
    )
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "usable_candidate": usable,
        "is_zero_byte_marker": size == 0 or ".no_exist" in str(path),
    }


def _candidate_records(model_id: str, cache_roots: Sequence[Path]) -> list[JsonDict]:
    """Search local roots for GGUF files matching one mandated model ID."""

    records: dict[str, JsonDict] = {}
    for root in cache_roots:
        root_path = Path(root).expanduser()
        for directory in _local_model_dirs(root_path, model_id):
            if not directory.exists():
                continue
            for path in directory.rglob("*.gguf"):
                records.setdefault(str(path), _candidate_record(path, model_id))
    return list(records.values())


def _select_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Pick the preferred non-empty GGUF, favoring the mandated Q4 quantization."""

    usable = [record for record in records if record.get("usable_candidate")]
    if not usable:
        return None
    for token in _QUANTIZATION_TOKENS:
        matches = [
            record
            for record in usable
            if token.lower() in Path(str(record["path"])).name.lower()
        ]
        if matches:
            return max(matches, key=lambda record: int(record.get("size_bytes") or 0))
    return max(usable, key=lambda record: str(record["path"]))


def _file_evidence(path: str | Path | None, *, full_sha_max_bytes: int = 64 * 1024 * 1024) -> JsonDict:
    """Return checksum evidence without spending minutes hashing multi-GB GGUFs."""

    if path is None:
        return {"status": "missing", "path": None, "sha256": None}
    model_path = Path(path)
    if not model_path.is_file():
        return {"status": "missing", "path": str(model_path), "sha256": None}
    stat = model_path.stat()
    size = int(stat.st_size)
    digest = hashlib.sha256()
    if size <= full_sha_max_bytes:
        digest.update(model_path.read_bytes())
        return {
            "status": "available",
            "path": str(model_path),
            "size_bytes": size,
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
            "checksum_algorithm": "sha256_full",
        }
    chunk_size = 1024 * 1024
    with model_path.open("rb") as handle:
        head = handle.read(chunk_size)
        handle.seek(max(0, size - chunk_size))
        tail = handle.read(chunk_size)
    digest.update(str(size).encode("utf-8"))
    digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    digest.update(head)
    digest.update(tail)
    return {
        "status": "available",
        "path": str(model_path),
        "size_bytes": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": None,
        "bounded_sha256": digest.hexdigest(),
        "checksum_algorithm": "sha256_head_tail_1mib_plus_size_mtime",
    }


def _resolve_cached_mandated_ggufs(
    project_root: str | Path,
    cache_roots: Sequence[str | Path],
) -> list[JsonDict]:
    """Resolve all locally cached mandated GGUFs without downloading weights."""

    del project_root
    roots = [Path(root).expanduser() for root in cache_roots]
    resolved: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        records = _candidate_records(model_id, roots)
        selected = _select_candidate(records)
        if selected is None:
            continue
        path = Path(str(selected["path"]))
        resolved.append(
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
    return resolved


def _run_model_worker(
    *,
    selected_python: str,
    model: Mapping[str, Any],
    n_gpu_layers: int,
    max_tokens: int,
    random_seed: int,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Run one selected-Python subprocess that loads a GGUF and generates text."""

    command = [
        selected_python,
        "-c",
        WORKER_CODE,
        "--exp3263-sota-gguf-worker",
        "--model-id",
        str(model["model_id"]),
        "--model-path",
        str(model["path"]),
        "--prompt",
        DEFAULT_PROMPT,
        "--seed",
        str(int(random_seed)),
        "--max-tokens",
        str(int(max_tokens)),
        "--n-gpu-layers",
        str(int(n_gpu_layers)),
    ]
    worker_env = dict(env)
    worker_env["PYTHONHASHSEED"] = str(int(random_seed))
    result = command_runner(command, timeout_s=1800, env=worker_env)
    payload = _json_from_last_line(result)
    stderr_summary = _summarize(_stderr(result))
    parsed_layers = _parse_offloaded_layers(_stderr(result))
    if parsed_layers and not _safe_int(payload.get("gpu_layers_offloaded")):
        payload["gpu_layers_offloaded"] = parsed_layers
    return {
        "attempted": True,
        "returncode": result.get("returncode"),
        "command_hash": _reproducibility_checksum({"command": command}),
        "stderr_summary": stderr_summary,
        "payload": payload,
    }


def _receipt_from_worker(model: Mapping[str, Any], worker: Mapping[str, Any]) -> JsonDict:
    """Normalize one worker payload into the per-model receipt schema."""

    payload = dict(worker.get("payload")) if isinstance(worker.get("payload"), Mapping) else {}
    stderr_summary = str(worker.get("stderr_summary") or "")
    tokens = _safe_int(payload.get("tokens_generated")) or 0
    layers = _safe_int(payload.get("gpu_layers_offloaded")) or _parse_offloaded_layers(stderr_summary)
    baseline = _safe_int(payload.get("gpu_mem_baseline_mib")) or 0
    used = (
        _safe_int(payload.get("gpu_mem_used_mib"))
        or _safe_int(payload.get("gpu_mem_used_during_call_mib"))
        or 0
    )
    delta = _safe_int(payload.get("gpu_mem_delta_mib"))
    if delta is None:
        delta = _safe_int(payload.get("gpu_mem_delta_during_call_mib"))
    if delta is None:
        delta = max(0, used - baseline)
    output = str(payload.get("output_text") or "").strip()
    passed = (
        worker.get("returncode") == 0
        and bool(output)
        and tokens > 0
        and int(layers or 0) > 0
        and used > baseline
    )
    return {
        "model_id": str(model["model_id"]),
        "model_path": str(model["path"]),
        "filename": str(model["filename"]),
        "size_bytes": int(model["size_bytes"]),
        "model_load_evidence": {
            "runtime": "llama_cpp",
            "load_status": str(payload.get("load_status") or "unknown"),
            "n_gpu_layers_requested": _safe_int(payload.get("n_gpu_layers_requested")),
            "duration_s": float(payload.get("duration_s") or 0.0),
        },
        "generation_evidence": {
            "generation_status": str(payload.get("generation_status") or "unknown"),
            "output_nonempty": bool(output),
            "output_preview": output[:240],
            "tokens_generated": int(tokens),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {},
        },
        "gpu_evidence": {
            "gpu_layers_offloaded": int(layers or 0),
            "gpu_mem_baseline_mib": int(baseline),
            "gpu_mem_used_mib": int(used or 0),
            "gpu_mem_delta_mib": int(delta),
        },
        "worker_attempt": {
            "attempted": bool(worker.get("attempted")),
            "returncode": worker.get("returncode"),
            "command_hash": str(worker.get("command_hash") or ""),
            "stderr_summary": stderr_summary,
        },
        "receipt_passed": passed,
    }


def _model_specs(
    *,
    cached_models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    n_gpu_layers: int,
) -> JsonDict:
    """Build the required model-spec dictionary naming every mandated model."""

    cached_by_id = {str(model["model_id"]): model for model in cached_models}
    passed = [receipt for receipt in receipts if receipt.get("receipt_passed")]
    headline = passed[0] if passed else None
    mandated_models: JsonDict = {}
    for model_id in MANDATED_MODEL_IDS:
        spec = _MODEL_BY_ID.get(model_id, {})
        cached = cached_by_id.get(model_id)
        mandated_models[model_id] = {
            "name": spec.get("name") or model_id.split("/", 1)[-1],
            "role": spec.get("role") or "unknown",
            "expected_quantization": spec.get("quantization") or "Q4_K_M",
            "cached": cached is not None,
            "model_path": str(cached["path"]) if cached else None,
            "size_bytes": int(cached["size_bytes"]) if cached else 0,
        }
    return {
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "mandated_models": mandated_models,
        "headline_model_id": str(headline["model_id"]) if headline else None,
        "headline_model_path": str(headline["model_path"]) if headline else None,
        "preferred_quantization": "Q4_K_M",
        "runtime": "llama_cpp",
        "n_gpu_layers_requested": int(n_gpu_layers),
        "prompt": DEFAULT_PROMPT,
    }


def _honest_verdict(*, receipt_ready: bool, blocked_reason: str, passed_count: int) -> str:
    """Return a terminal verdict with the repo-required success-style prefix."""

    if receipt_ready:
        return (
            "complete: sota_gguf_receipt_v9_ready=true; "
            "sota_gguf_receipt_ready=true; "
            f"per_model_receipts_passed={passed_count}"
        )
    return (
        "complete: sota_gguf_receipt_v9_ready=true; "
        "sota_gguf_receipt_ready=false; "
        f"blocked_reason={blocked_reason}"
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> JsonDict:
    """REQ-REPORT-3263: build the gated SOTA GGUF receipt v9 artifact."""

    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    roots = [Path(path) for path in (cache_roots or _default_cache_roots(root, merged_env))]
    exp3262 = _read_json(root / EXP3262_REL_PATH)

    blocked_reason = ""
    cached_models: list[JsonDict] = []
    receipts: list[JsonDict] = []
    missing_model_ids: list[str] = []

    if exp3262.get("llama_cpp_cuda_receipt_ready") is not True:
        blocked_reason = "gated_exp3262_llama_cpp_cuda_receipt_not_ready"
    else:
        cached_models = _resolve_cached_mandated_ggufs(root, roots)
        cached_ids = {str(model["model_id"]) for model in cached_models}
        missing_model_ids = [model_id for model_id in MANDATED_MODEL_IDS if model_id not in cached_ids]
        if not cached_models:
            blocked_reason = "blocked_sota_gguf_not_cached"
        else:
            for model in cached_models:
                worker = _run_model_worker(
                    selected_python=selected,
                    model=model,
                    n_gpu_layers=int(n_gpu_layers),
                    max_tokens=int(max_tokens),
                    random_seed=int(random_seed),
                    env=merged_env,
                    command_runner=command_runner,
                )
                receipts.append(_receipt_from_worker(model, worker))
            if not any(receipt.get("receipt_passed") for receipt in receipts):
                blocked_reason = "sota_gguf_receipt_incomplete"

    passed_receipts = [receipt for receipt in receipts if receipt.get("receipt_passed")]
    receipt_ready = bool(passed_receipts) and blocked_reason == ""
    gpu_mem_used_mib = (
        max(int(receipt["gpu_evidence"]["gpu_mem_used_mib"]) for receipt in passed_receipts)
        if passed_receipts
        else 0
    )
    model_specs = _model_specs(
        cached_models=cached_models,
        receipts=receipts,
        n_gpu_layers=int(n_gpu_layers),
    )
    checksum = _reproducibility_checksum(
        {
            "blocked_reason": blocked_reason,
            "cached_model_ids": [model["model_id"] for model in cached_models],
            "exp3262_ready": exp3262.get("llama_cpp_cuda_receipt_ready") is True,
            "gpu_mem_used_mib": gpu_mem_used_mib,
            "model_specs": model_specs,
            "per_model_receipts": receipts,
            "random_seed": int(random_seed),
            "selected_python": selected,
        }
    )
    duration_s = round(max(0.0, monotonic() - start), 6)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "selected_python": selected,
        "exp3262_gate": {
            "path": str(root / EXP3262_REL_PATH),
            "llama_cpp_cuda_receipt_ready": exp3262.get("llama_cpp_cuda_receipt_ready") is True,
        },
        "cache_roots": [str(path) for path in roots],
        "sota_gguf_receipt_v9_ready": True,
        "sota_gguf_receipt_ready": receipt_ready,
        "blocked_reason": blocked_reason,
        "cached_model_ids": [str(model["model_id"]) for model in cached_models],
        "missing_model_ids": missing_model_ids,
        "model_specs": model_specs,
        "per_model_receipts": receipts,
        "gpu_mem_used_mib": gpu_mem_used_mib,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(
            receipt_ready=receipt_ready,
            blocked_reason=blocked_reason,
            passed_count=len(passed_receipts),
        ),
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> JsonDict:
    """Build and write the Exp 3263 SOTA GGUF receipt v9 artifact."""

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
        max_tokens=max_tokens,
    )
    _write_json(destination, artifact)
    return artifact


def main() -> int:
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
