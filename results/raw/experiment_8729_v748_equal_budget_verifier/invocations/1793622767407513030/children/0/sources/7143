#!/usr/bin/env python3
"""Exp 2989 SOTA GGUF cache provenance preflight.

**Researcher summary:**
    This preflight is the upstream gate for headline live-LLM repair claims
    after Exp 2977 only produced small-model CPU smoke evidence.  It records
    the local compute and cache state first, then asks every locally available
    mandated headline GGUF for one tiny transcript.  If no headline GGUF can
    produce a transcript, the artifact is terminally blocked.

**Detailed explanation for engineers:**
    The script does not download weights and does not promote legacy small
    models.  It inspects HuggingFace and project-local cache layouts, preserves
    bounded checksum/file evidence for each resolved GGUF, calls
    ``cached_sota_pair()`` as a compatibility check, and only attempts a
    conservative llama.cpp generation when CUDA and llama.cpp GPU offload are
    visible.  Legacy smoke-only model IDs are recorded as context when
    requested, but they can never set ``sota_headline_ready``.

Spec: REQ-INFER-SOTA-019,
      SCENARIO-INFER-SOTA-019-001,
      SCENARIO-INFER-SOTA-019-002
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf
from scripts.experiment_template import _get_repo_root


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
PromptRunnerFn = Callable[..., JsonDict]
ClockFn = Callable[[], float]

ARTIFACT_NAME = "experiment_2989_sota_gguf_cache_provenance_preflight_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
DEFAULT_ARTIFACT_PATH = Path("results") / ARTIFACT_FILENAME
RAW_TRANSCRIPT_DIR = Path("results") / "raw" / ARTIFACT_NAME
RUN_DATE = "20260524"
RANDOM_SEED = 2989
DEFAULT_PROMPT = (
    "Answer in one short sentence: this is a Carnot SOTA GGUF cache preflight."
)
HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "sota_headline_ready",
    "preconditions_checked",
    "model_specs",
    "sota_models_attempted",
    "sota_models_available",
    "cache_paths",
    "model_checksums",
    "live_transcript_paths",
    "legacy_smoke_only_used",
    "inference_substrate",
    "duration_seconds",
    "honest_verdict",
)
_MODEL_BY_HF_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
_QUANTIZATION_TOKENS: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q8_XL",
    "Q8_0",
    "BF16",
)


def _selected_python(project_root: Path) -> str:
    """Return the project venv Python when present, otherwise this interpreter."""
    candidate = project_root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 2000) -> str:
    """Keep command evidence bounded while preserving the useful prefix."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _stdout(result: Mapping[str, Any]) -> str:
    """Return command stdout, falling back to compact summaries."""
    return str(result.get("stdout") or result.get("stdout_summary") or "")


def _stderr(result: Mapping[str, Any]) -> str:
    """Return command stderr, falling back to compact summaries."""
    return str(result.get("stderr") or result.get("stderr_summary") or "")


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a bounded local diagnostic command and return structured evidence."""
    cmd = [str(part) for part in command]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_summary": "",
            "stderr_summary": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_summary": _summarize(completed.stdout),
        "stderr_summary": _summarize(completed.stderr),
    }


def _repo_commit_probe(project_root: Path, *, command_runner: CommandRunner) -> JsonDict:
    """Record the current repository commit without requiring a clean worktree."""
    command = ["git", "-C", str(project_root), "rev-parse", "HEAD"]
    result = command_runner(command, timeout_s=10)
    commit = _stdout(result).strip().splitlines()[0] if _stdout(result).strip() else None
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "commit": commit,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _python_environment(selected_python: str, project_root: Path) -> JsonDict:
    """Capture the Python environment used by the preflight and subprocesses."""
    return {
        "selected_python": selected_python,
        "current_executable": sys.executable,
        "current_version": sys.version,
        "project_root": str(project_root),
        "venv_active": bool(os.environ.get("VIRTUAL_ENV")),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def _torch_cuda_probe(selected_python: str, *, command_runner: CommandRunner) -> JsonDict:
    """Measure CUDA through the exact Python interpreter downstream should use."""
    command = [
        selected_python,
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())",
    ]
    result = command_runner(command, timeout_s=30)
    parts = _stdout(result).strip().split()
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "torch_version": parts[0] if parts else None,
        "cuda_available": bool(
            result.get("returncode") == 0 and len(parts) >= 2 and parts[1] == "True"
        ),
        "cuda_device_count": int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Record GPU inventory and free VRAM before any model load occurs."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    gpus: list[JsonDict] = []
    if result.get("returncode") == 0:
        for line in _stdout(result).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6 and parts[0].isdigit() and parts[2].isdigit():
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_used_mib": int(parts[3]) if parts[3].isdigit() else None,
                        "memory_free_mib": int(parts[4]) if parts[4].isdigit() else None,
                        "driver_version": parts[5],
                    }
                )
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "available": bool(gpus),
        "gpus": gpus,
        "free_vram_mib_total": sum(int(gpu.get("memory_free_mib") or 0) for gpu in gpus),
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _llama_cpp_probe(
    selected_python: str,
    *,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Import llama.cpp and ask whether its backend supports GPU offload."""
    script = (
        "import importlib.util, json\n"
        "payload = {'llama_cpp_import_ok': False, 'llama_cpp_supports_gpu_offload': False}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import llama_cpp as low\n"
        "    payload.update({\n"
        "        'llama_cpp_import_ok': True,\n"
        "        'llama_cpp_origin': importlib.util.find_spec('llama_cpp').origin,\n"
        "        'llama_cpp_version': getattr(llama_cpp, '__version__', None),\n"
        "        'llama_cpp_supports_gpu_offload': bool(low.llama_supports_gpu_offload()),\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-c", script]
    result = command_runner(command, timeout_s=30, env=dict(env))
    try:
        parsed = json.loads(_stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "llama_cpp_import_ok": False,
            "llama_cpp_supports_gpu_offload": False,
            "error": _stderr(result) or _stdout(result) or "llama_cpp_probe_unparseable",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stderr_summary"] = _summarize(_stderr(result))
    return parsed


def _cache_roots(project_root: Path, env: Mapping[str, str]) -> JsonDict:
    """Return cache roots that must be inspected before generation starts."""
    if env.get("HUGGINGFACE_HUB_CACHE"):
        hf_cache = Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    elif env.get("HF_HOME"):
        hf_cache = Path(env["HF_HOME"]).expanduser() / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    local_models = project_root / "models"
    return {
        "huggingface_hub_cache": str(hf_cache),
        "huggingface_hub_cache_exists": hf_cache.exists(),
        "local_models": str(local_models),
        "local_models_exists": local_models.exists(),
    }


def _model_filename_token(hf_id: str) -> str:
    """Return the model-family token expected in local GGUF filenames."""
    return hf_id.split("/", 1)[-1].removesuffix("-GGUF").lower()


def _local_model_dirs(models_root: Path, hf_id: str) -> list[Path]:
    """Mirror the project-local layouts operators have used for GGUF caches."""
    basename = hf_id.split("/", 1)[-1]
    stripped = basename.removesuffix("-GGUF")
    return [
        models_root / stripped,
        models_root / basename,
        models_root / stripped.lower(),
        models_root / basename.lower(),
    ]


def _candidate_record(path: Path, hf_id: str, source: str) -> JsonDict:
    """Convert a local GGUF path into auditable cache evidence."""
    try:
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
    except OSError:  # pragma: no cover - defensive for broken filesystem metadata.
        exists = False
        size = 0
    name = path.name.lower()
    token = _model_filename_token(hf_id)
    usable = bool(exists and size > 0 and token in name and "mmproj" not in name)
    return {
        "path": str(path),
        "source": source,
        "exists": exists,
        "size_bytes": size,
        "usable_candidate": usable,
        "is_zero_byte_marker": size == 0 or ".no_exist" in str(path),
    }


def _candidate_records(project_root: Path, roots: Mapping[str, Any], hf_id: str) -> list[JsonDict]:
    """Search local HF snapshots and project models recursively for one GGUF."""
    records: list[JsonDict] = []
    hf_cache = Path(str(roots["huggingface_hub_cache"]))
    hf_repo = hf_cache / f"models--{hf_id.replace('/', '--')}"
    if hf_repo.exists():
        records.extend(
            _candidate_record(path, hf_id, "huggingface_hub_cache")
            for path in hf_repo.rglob("*.gguf")
        )
    models_root = project_root / "models"
    for model_dir in _local_model_dirs(models_root, hf_id):
        if model_dir.exists():
            records.extend(
                _candidate_record(path, hf_id, "project_models")
                for path in model_dir.rglob("*.gguf")
            )
    unique: dict[str, JsonDict] = {}
    for record in records:
        unique.setdefault(str(record["path"]), record)
    return list(unique.values())


def _quantization_suffix(path: str | None) -> str | None:
    """Extract the visible quantization token from a GGUF filename."""
    if path is None:
        return None
    filename = Path(path).name.lower()
    for token in _QUANTIZATION_TOKENS:
        if token.lower() in filename:
            return token
    return "unknown"


def _select_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select the preferred nonzero GGUF candidate deterministically."""
    usable = [record for record in records if record.get("usable_candidate")]
    if not usable:
        return None
    for token in _QUANTIZATION_TOKENS:
        matches = [
            record for record in usable if token.lower() in Path(str(record["path"])).name.lower()
        ]
        if matches:
            return max(matches, key=lambda record: int(record.get("size_bytes") or 0))
    return max(usable, key=lambda record: str(record["path"]))


def _file_evidence(path: str | Path | None, *, full_sha_max_bytes: int = 64 * 1024 * 1024) -> JsonDict:
    """Return bounded checksum evidence for a model file.

    Full SHA256 is exact for small files.  For multi-GB GGUFs the preflight
    records a bounded head/tail digest plus size and mtime so the run stays
    fast while still tying the evidence to a concrete local artifact.
    """
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


def _inspect_cache(project_root: Path, env: Mapping[str, str], model_ids: Sequence[str]) -> list[JsonDict]:
    """Inspect requested model IDs without loading or downloading weights."""
    roots = _cache_roots(project_root, env)
    rows: list[JsonDict] = []
    for hf_id in model_ids:
        records = _candidate_records(project_root, roots, hf_id)
        selected = _select_candidate(records)
        selected_path = str(selected["path"]) if selected is not None else None
        resolver_path = resolve_cached_gguf(
            hf_id, "Q4_K_M", cache_root=str(roots["huggingface_hub_cache"])
        )
        spec = _MODEL_BY_HF_ID.get(hf_id, {})
        rows.append(
            {
                "hf_id": hf_id,
                "name": spec.get("name") or hf_id.split("/", 1)[-1],
                "role": spec.get("role") or "smoke_only",
                "expected_quantization": spec.get("quantization"),
                "cache_status": "resolved" if selected_path else "missing",
                "path": selected_path,
                "resolved_path": str(Path(selected_path).resolve()) if selected_path else None,
                "resolver_path": resolver_path,
                "observed_quantization": _quantization_suffix(selected_path),
                "candidate_count": len(records),
                "candidate_paths": [record["path"] for record in records],
            }
        )
    return rows


def _loadable_pair(model_specs: Any) -> bool:
    """Return whether cached_sota_pair yielded two concrete local GGUF specs."""
    return bool(
        isinstance(model_specs, list)
        and len(model_specs) == 2
        and all(
            isinstance(spec, dict) and spec.get("hf_id") and spec.get("model_path")
            for spec in model_specs
        )
    )


def _exercise_cached_sota_pair(cached_pair_fn: CachedPairFn) -> JsonDict:
    """Call the shared pair helper and preserve any exception as data."""
    try:
        result = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
        return {
            "called": True,
            "result": result,
            "error": None,
            "returned_two_loadable_specs": _loadable_pair(result),
        }
    except Exception as exc:
        return {
            "called": True,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "returned_two_loadable_specs": False,
        }


def _model_specs() -> JsonDict:
    """Return the headline and smoke-only model identities required by the task."""
    return {
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
        "random_seed": RANDOM_SEED,
    }


def _safe_model_slug(hf_id: str) -> str:
    """Convert a model ID to a filesystem-safe transcript filename component."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", hf_id).strip("_")


def _run_bounded_headline_prompt(
    model: Mapping[str, Any],
    *,
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
    timeout_s: int = 300,
) -> JsonDict:
    """Run one bounded llama.cpp prompt in a subprocess and parse its JSON row."""
    script = (
        "import json, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "path, hf_id, prompt = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "gpu = int(sys.argv[4])\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "started = time.monotonic()\n"
        "llm = Llama(model_path=path, n_ctx=512, n_batch=64, n_ubatch=64, n_gpu_layers=-1, main_gpu=gpu, verbose=False)\n"
        "out = llm(prompt, max_tokens=24, temperature=0.0, seed=2989)\n"
        "duration = time.monotonic() - started\n"
        "text = out.get('choices', [{}])[0].get('text', '').strip()\n"
        "tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "llm.close()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'load_status': 'loaded',\n"
        "    'generation_status': 'generated' if text and tokens > 0 else 'empty_response',\n"
        "    'usable': bool(text) and tokens > 0 and supports_gpu,\n"
        "    'gpu_backed': supports_gpu,\n"
        "    'hf_id': hf_id,\n"
        "    'model_path': path,\n"
        "    'prompt': prompt,\n"
        "    'response_text': text,\n"
        "    'tokens_generated': tokens,\n"
        "    'duration_seconds': round(duration, 6),\n"
        "    'inference_substrate': 'llama_cpp_gpu' if supports_gpu else 'llama_cpp_cpu',\n"
        "}, sort_keys=True))\n"
    )
    command = [
        selected_python,
        "-c",
        script,
        str(model["path"]),
        str(model["hf_id"]),
        DEFAULT_PROMPT,
        str(model.get("gpu", 0)),
    ]
    result = command_runner(command, timeout_s=timeout_s, env=dict(env))
    try:
        parsed = json.loads(_stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "load_status": "failed",
            "generation_status": "failed",
            "usable": False,
            "gpu_backed": False,
            "hf_id": model.get("hf_id"),
            "model_path": model.get("path"),
            "prompt": DEFAULT_PROMPT,
            "response_text": "",
            "tokens_generated": 0,
            "duration_seconds": 0.0,
            "inference_substrate": "llama_cpp_failed",
            "blocker": _stderr(result) or _stdout(result) or "bounded_prompt_failed",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stdout_summary"] = _summarize(_stdout(result))
    parsed["stderr_summary"] = _summarize(_stderr(result))
    return parsed


def _write_transcript(
    transcript_dir: Path,
    *,
    attempt: Mapping[str, Any],
    prompt_result: Mapping[str, Any],
) -> JsonDict:
    """Persist replayable live transcript evidence and return path/hash metadata."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{_safe_model_slug(str(attempt['hf_id']))}.json"
    payload = {
        "model_hf_id": attempt["hf_id"],
        "model_path": attempt["cache_path"],
        "prompt": prompt_result.get("prompt", DEFAULT_PROMPT),
        "response_text": prompt_result.get("response_text", ""),
        "tokens_generated": prompt_result.get("tokens_generated", 0),
        "duration_seconds": prompt_result.get("duration_seconds", 0.0),
        "inference_substrate": prompt_result.get("inference_substrate"),
        "load_status": prompt_result.get("load_status"),
        "generation_status": prompt_result.get("generation_status"),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _preconditions(
    *,
    project_root: Path,
    selected_python: str,
    env: Mapping[str, str],
    command_runner: CommandRunner,
    cached_pair_fn: CachedPairFn,
) -> JsonDict:
    """Collect all environment and cache-root evidence before model loading."""
    return {
        "python_environment": _python_environment(selected_python, project_root),
        "repo_commit": _repo_commit_probe(project_root, command_runner=command_runner),
        "torch_cuda": _torch_cuda_probe(selected_python, command_runner=command_runner),
        "gpu_inventory": _nvidia_smi_inventory(command_runner=command_runner),
        "llama_cpp": _llama_cpp_probe(selected_python, command_runner=command_runner, env=env),
        "cache_roots": _cache_roots(project_root, env),
        "cached_sota_pair": _exercise_cached_sota_pair(cached_pair_fn),
        "recorded_before_model_load": True,
    }


def _attempt_rows(
    *,
    cache_inventory: Sequence[Mapping[str, Any]],
    checksum_by_model: Mapping[str, Mapping[str, Any]],
    precondition_evidence: Mapping[str, Any],
    selected_python: str,
    env: Mapping[str, str],
    transcript_dir: Path,
    command_runner: CommandRunner,
    prompt_runner_fn: PromptRunnerFn,
    prompt_timeout_s: int,
) -> tuple[list[JsonDict], list[str]]:
    """Attempt one bounded live transcript for each locally available headline GGUF."""
    attempts: list[JsonDict] = []
    transcript_paths: list[str] = []
    torch_cuda = bool(precondition_evidence["torch_cuda"].get("cuda_available"))
    llama_gpu = bool(precondition_evidence["llama_cpp"].get("llama_cpp_supports_gpu_offload"))
    for index, row in enumerate(cache_inventory):
        hf_id = str(row["hf_id"])
        attempt: JsonDict = {
            "hf_id": hf_id,
            "cache_status": row["cache_status"],
            "cache_path": row["path"],
            "resolved_path": row["resolved_path"],
            "checksum_evidence": checksum_by_model[hf_id],
            "load_status": "not_attempted",
            "generation_status": "not_attempted",
            "duration_seconds": 0.0,
            "transcript_path": None,
            "transcript_sha256": None,
        }
        if row["cache_status"] != "resolved":
            attempt["load_status"] = "skipped_missing_cache"
            attempts.append(attempt)
            continue
        if not (torch_cuda and llama_gpu):
            attempt["load_status"] = "not_attempted_runtime_precondition_failed"
            attempts.append(attempt)
            continue
        prompt_result = prompt_runner_fn(
            {"hf_id": hf_id, "path": row["path"], "gpu": index},
            selected_python=selected_python,
            command_runner=command_runner,
            env=env,
            timeout_s=prompt_timeout_s,
        )
        attempt.update(
            {
                "load_status": prompt_result.get("load_status", "unknown"),
                "generation_status": prompt_result.get("generation_status", "unknown"),
                "duration_seconds": float(prompt_result.get("duration_seconds") or 0.0),
                "tokens_generated": int(prompt_result.get("tokens_generated") or 0),
                "gpu_backed": bool(prompt_result.get("gpu_backed")),
                "blocker": prompt_result.get("blocker"),
            }
        )
        if prompt_result.get("usable") and str(prompt_result.get("response_text") or "").strip():
            transcript = _write_transcript(
                transcript_dir,
                attempt=attempt,
                prompt_result=prompt_result,
            )
            attempt["transcript_path"] = transcript["path"]
            attempt["transcript_sha256"] = transcript["sha256"]
            transcript_paths.append(transcript["path"])
        attempts.append(attempt)
    return attempts, transcript_paths


def _honest_verdict(
    *,
    ready: bool,
    cached_count: int,
    torch_cuda: bool,
    llama_gpu: bool,
    attempted_live: bool,
) -> str:
    """Map the terminal gate state to an explicit ready-or-blocked verdict."""
    if ready:
        return "success: at least one mandated headline SOTA GGUF produced a live transcript"
    if cached_count == 0:
        return "blocked_model_cache: no mandated headline SOTA GGUF resolved locally"
    if not (torch_cuda and llama_gpu):
        return "blocked_runtime_preconditions: CUDA or llama.cpp GPU offload unavailable before headline load"
    if attempted_live:
        return "blocked_generation: headline GGUF cache exists but no usable transcript was produced"
    return "blocked_preconditions: headline generation did not run"


def _inference_substrate(*, ready: bool, cached_count: int, attempted_live: bool) -> str:
    """Describe the actual substrate used for the terminal claim boundary."""
    if ready:
        return "live_llm_inference"
    if cached_count == 0:
        return "blocked_no_headline_cache"
    if attempted_live:
        return "live_llm_inference_failed"
    return "blocked_runtime_preflight"


def build_preflight_artifact(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    run_legacy_smoke: bool = False,
    prompt_timeout_s: int = 300,
) -> JsonDict:
    """Build the Exp 2989 terminal preflight artifact."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    precondition_evidence = _preconditions(
        project_root=root,
        selected_python=selected,
        env=merged_env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
    )
    headline_cache = _inspect_cache(root, merged_env, HEADLINE_MODEL_IDS)
    smoke_cache = _inspect_cache(root, merged_env, SMOKE_ONLY_MODEL_IDS)
    model_checksums = {
        row["hf_id"]: _file_evidence(row["path"]) for row in [*headline_cache, *smoke_cache]
    }
    transcript_dir = root / RAW_TRANSCRIPT_DIR
    attempts, live_transcript_paths = _attempt_rows(
        cache_inventory=headline_cache,
        checksum_by_model=model_checksums,
        precondition_evidence=precondition_evidence,
        selected_python=selected,
        env=merged_env,
        transcript_dir=transcript_dir,
        command_runner=command_runner,
        prompt_runner_fn=prompt_runner_fn,
        prompt_timeout_s=prompt_timeout_s,
    )

    cached_count = sum(1 for row in headline_cache if row["cache_status"] == "resolved")
    attempted_live = any(
        row.get("load_status") not in {"skipped_missing_cache", "not_attempted_runtime_precondition_failed"}
        for row in attempts
    )
    ready = bool(live_transcript_paths)
    finished = monotonic()
    available_models = [
        {"hf_id": row["hf_id"], "path": row["path"], "status": "cache_resolved"}
        for row in headline_cache
        if row["cache_status"] == "resolved"
    ]

    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "run_date": RUN_DATE,
        "sota_headline_ready": ready,
        "preconditions_checked": True,
        "model_specs": _model_specs(),
        "sota_models_attempted": attempts,
        "sota_models_available": available_models,
        "cache_paths": {
            "roots": precondition_evidence["cache_roots"],
            "headline_models": {row["hf_id"]: row["path"] for row in headline_cache},
            "smoke_only_models": {row["hf_id"]: row["path"] for row in smoke_cache},
        },
        "model_checksums": model_checksums,
        "live_transcript_paths": live_transcript_paths,
        "legacy_smoke_only_used": bool(run_legacy_smoke and not ready),
        "inference_substrate": _inference_substrate(
            ready=ready,
            cached_count=cached_count,
            attempted_live=attempted_live,
        ),
        "duration_seconds": round(finished - started, 6),
        "honest_verdict": _honest_verdict(
            ready=ready,
            cached_count=cached_count,
            torch_cuda=bool(precondition_evidence["torch_cuda"].get("cuda_available")),
            llama_gpu=bool(precondition_evidence["llama_cpp"].get("llama_cpp_supports_gpu_offload")),
            attempted_live=attempted_live,
        ),
        "precondition_evidence": precondition_evidence,
        "tests_run": list(tests_run or []),
        "legacy_smoke_context": {
            "smoke_only": bool(run_legacy_smoke and not ready),
            "model_ids": list(SMOKE_ONLY_MODEL_IDS),
            "used_for_headline_readiness": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for conductor and downstream gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    run_legacy_smoke: bool = False,
    prompt_timeout_s: int = 300,
) -> JsonDict:
    """Build and write the Exp 2989 preflight JSON artifact."""
    root = Path(project_root) if project_root is not None else Path(_get_repo_root())
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_preflight_artifact(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        prompt_runner_fn=prompt_runner_fn,
        monotonic=monotonic,
        tests_run=tests_run,
        run_legacy_smoke=run_legacy_smoke,
        prompt_timeout_s=prompt_timeout_s,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument("--run-legacy-smoke", action="store_true")
    parser.add_argument("--prompt-timeout-s", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    kwargs: JsonDict = {
        "output_path": args.output,
        "selected_python": args.selected_python,
        "tests_run": args.test_run,
    }
    if args.run_legacy_smoke:
        kwargs["run_legacy_smoke"] = True
    if args.prompt_timeout_s != 300:
        kwargs["prompt_timeout_s"] = args.prompt_timeout_s
    run_experiment(**kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
