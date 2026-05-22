"""Exp 2862 SOTA GGUF runtime cache/offload resolver v3.

**Researcher summary:**
    This artifact is the replacement gate for the blocked Exp 2848 runtime
    evidence attempt.  It separates three facts that were previously tangled:
    whether the workstation has CUDA through the project venv, whether the
    installed ``llama_cpp`` build can offload to GPU, and whether at least one
    mandated local SOTA GGUF can produce a real bounded response.

**Detailed explanation for engineers:**
    The resolver does not download weights and does not promote legacy tiny
    models.  It inspects the local HuggingFace cache and project ``models/``
    directory, records why the two-model ``cached_sota_pair()`` helper did or
    did not return a pair, and then runs at most one bounded prompt against a
    locally available mandated model when CUDA and llama.cpp GPU offload are
    both visible.  A CPU-only llama.cpp load is recorded as a blocker with the
    exact CUDA rebuild command instead of being counted as live SOTA evidence.

Spec: REQ-INFER-SOTA-013,
      SCENARIO-INFER-SOTA-013-001,
      SCENARIO-INFER-SOTA-013-002,
      SCENARIO-INFER-SOTA-013-003
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
PromptRunnerFn = Callable[..., JsonDict]
ClockFn = Callable[[], float]

DEFAULT_ARTIFACT_PATH = Path("results/experiment_2862_sota_runtime_cache_offload_resolver_v3.json")
RANDOM_SEED = 2862
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY: tuple[str, ...] = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "sota_runtime_ready_v3",
    "model_specs",
    "selected_model_hf_id",
    "selected_model_path",
    "cached_sota_pair_returned_two_loadable_specs",
    "llama_cpp_gpu_offload_verified",
    "preconditions_checked",
    "usable_response_count",
    "total_tokens_generated",
    "tokens_per_second",
    "legacy_small_models_used_only_for_smoke",
    "random_seed",
    "reproducibility_checksum",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)
_MODEL_BY_HF_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
_QUANTIZATION_TOKENS: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q4_K_S",
    "Q8_0",
    "BF16",
)


def _repo_root() -> Path:
    """Return the repository root used by direct CLI invocations."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _selected_python(project_root: Path) -> str:
    """Return the project venv interpreter when it exists."""
    candidate = project_root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 2000) -> str:
    """Keep command evidence compact while preserving the important prefix."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a local diagnostic command and return structured evidence."""
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


def _stdout(result: Mapping[str, Any]) -> str:
    return str(result.get("stdout") or result.get("stdout_summary") or "")


def _stderr(result: Mapping[str, Any]) -> str:
    return str(result.get("stderr") or result.get("stderr_summary") or "")


def _torch_cuda_probe(selected_python: str, *, command_runner: CommandRunner) -> JsonDict:
    """Measure CUDA through the exact Python interpreter downstream must use."""
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
    """Record the NVIDIA inventory and current memory before model loading."""
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
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _llama_cpp_probe(
    selected_python: str,
    *,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Import llama.cpp and ask its low-level backend whether GPU offload exists."""
    script = (
        "import importlib.util, json\n"
        "payload = {'llama_cpp_import_ok': False}\n"
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
            "error": _stderr(result) or _stdout(result) or "llama_cpp_probe_unparseable",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stderr_summary"] = _summarize(_stderr(result))
    return parsed


def _cache_roots(project_root: Path, env: Mapping[str, str]) -> JsonDict:
    """Return the two local roots that may hold mandated GGUF files."""
    if env.get("HUGGINGFACE_HUB_CACHE"):
        hf_cache = Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    elif env.get("HF_HOME"):
        hf_cache = Path(env["HF_HOME"]).expanduser() / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    return {
        "huggingface_hub_cache": str(hf_cache),
        "huggingface_hub_cache_exists": hf_cache.exists(),
        "local_models": str(project_root / "models"),
        "local_models_exists": (project_root / "models").exists(),
    }


def _model_filename_token(hf_id: str) -> str:
    """Return the filename stem that identifies a mandated model family."""
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
    """Convert a local GGUF path into cache-inventory evidence."""
    try:
        size = int(path.stat().st_size)
        exists = path.exists()
    except OSError:
        size = 0
        exists = False
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
    """Search local HF snapshots and project models recursively for one model."""
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
    """Extract a visible quantization token from a GGUF filename."""
    if path is None:
        return None
    filename = Path(path).name.lower()
    for token in _QUANTIZATION_TOKENS:
        if token.lower() in filename:
            return token
    return "unknown"


def _select_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select the preferred nonzero model candidate deterministically."""
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


def _inspect_mandated_cache(project_root: Path, env: Mapping[str, str]) -> list[JsonDict]:
    """Inspect every mandated model and record exact resolved path or missing status."""
    roots = _cache_roots(project_root, env)
    rows: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        records = _candidate_records(project_root, roots, hf_id)
        selected = _select_candidate(records)
        resolver_path = resolve_cached_gguf(
            hf_id, "Q4_K_M", cache_root=str(roots["huggingface_hub_cache"])
        )
        spec = _MODEL_BY_HF_ID.get(hf_id, {})
        selected_path = str(selected["path"]) if selected is not None else None
        rows.append(
            {
                "hf_id": hf_id,
                "name": spec.get("name"),
                "role": spec.get("role"),
                "expected_quantization": spec.get("quantization"),
                "cache_status": "resolved" if selected_path else "missing",
                "path": selected_path,
                "resolved_path": str(Path(selected_path).resolve()) if selected_path else None,
                "resolver_path": resolver_path,
                "observed_quantization": _quantization_suffix(selected_path),
                "candidate_count": len(records),
                "hf_candidate_count": sum(
                    1
                    for record in records
                    if record["source"] == "huggingface_hub_cache" and record["usable_candidate"]
                ),
                "project_candidate_count": sum(
                    1
                    for record in records
                    if record["source"] == "project_models" and record["usable_candidate"]
                ),
                "zero_byte_marker_count": sum(
                    1 for record in records if record["is_zero_byte_marker"]
                ),
                "missing_status": None if selected_path else "missing_or_zero_byte_only",
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


def _run_bounded_prompt(
    model: Mapping[str, Any],
    *,
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Run one bounded llama.cpp prompt in a subprocess and parse its JSON row."""
    script = (
        "import json, subprocess, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "def mem():\n"
        "    try:\n"
        "        out = subprocess.check_output([\n"
        "            'nvidia-smi', '--query-gpu=index,memory.used,memory.free',\n"
        "            '--format=csv,noheader,nounits'], text=True, timeout=5)\n"
        "        rows = []\n"
        "        for line in out.splitlines():\n"
        "            parts = [p.strip() for p in line.split(',')]\n"
        "            if len(parts) == 3:\n"
        "                rows.append({'index': int(parts[0]), 'memory_used_mib': int(parts[1]), 'memory_free_mib': int(parts[2])})\n"
        "        return rows\n"
        "    except Exception as exc:\n"
        "        return [{'error': f'{type(exc).__name__}: {exc}'}]\n"
        "path, hf_id = sys.argv[1], sys.argv[2]\n"
        "gpu = int(sys.argv[3])\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "before = mem()\n"
        "started = time.monotonic()\n"
        "llm = Llama(model_path=path, n_ctx=256, n_batch=64, n_ubatch=64, n_gpu_layers=-1, main_gpu=gpu, verbose=False)\n"
        "during = mem()\n"
        "out = llm('Answer with only the number: what is 2+2?', max_tokens=16, temperature=0.0, seed=2862)\n"
        "duration = time.monotonic() - started\n"
        "text = out.get('choices', [{}])[0].get('text', '')\n"
        "tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "llm.close()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'usable': bool(text.strip()) and tokens > 0 and supports_gpu,\n"
        "    'gpu_backed': supports_gpu,\n"
        "    'hf_id': hf_id,\n"
        "    'model_path': path,\n"
        "    'response_text': text.strip(),\n"
        "    'tokens_generated': tokens,\n"
        "    'tokens_per_second': round(tokens / duration, 6) if duration > 0 else 0.0,\n"
        "    'duration_s': round(duration, 6),\n"
        "    'gpu_memory': {'before': before, 'during': during},\n"
        "}, sort_keys=True))\n"
    )
    command = [
        selected_python,
        "-c",
        script,
        str(model["path"]),
        str(model["hf_id"]),
        str(model.get("gpu", 0)),
    ]
    result = command_runner(command, timeout_s=300, env=dict(env))
    try:
        parsed = json.loads(_stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "usable": False,
            "gpu_backed": False,
            "hf_id": model.get("hf_id"),
            "model_path": model.get("path"),
            "tokens_generated": 0,
            "tokens_per_second": 0.0,
            "duration_s": 0.0,
            "blocker": _stderr(result) or _stdout(result) or "bounded_prompt_failed",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stdout_summary"] = _summarize(_stdout(result))
    parsed["stderr_summary"] = _summarize(_stderr(result))
    return parsed


def _llama_cpp_cuda_reinstall_command() -> str:
    """Return the exact local rebuild command for a CPU-only llama.cpp wheel."""
    return (
        'CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 '
        ".venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.23"
    )


def _preconditions(
    *,
    torch_probe: Mapping[str, Any],
    gpu_inventory: Mapping[str, Any],
    llama_probe: Mapping[str, Any],
    pair_result: Mapping[str, Any],
    cache_inventory: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the downstream gating checklist."""
    cached_count = sum(1 for row in cache_inventory if row.get("cache_status") == "resolved")
    return [
        {
            "resource": "venv_torch_cuda",
            "available": bool(torch_probe.get("cuda_available")),
            "detail": torch_probe.get("stdout_summary") or torch_probe.get("stderr_summary"),
            "command": torch_probe.get("command"),
        },
        {
            "resource": "nvidia_smi_inventory",
            "available": bool(gpu_inventory.get("available")),
            "detail": gpu_inventory.get("gpus", []),
            "command": gpu_inventory.get("command"),
        },
        {
            "resource": "llama_cpp_gpu_offload",
            "available": bool(llama_probe.get("llama_cpp_supports_gpu_offload")),
            "detail": llama_probe.get("llama_cpp_origin") or llama_probe.get("error"),
            "command": llama_probe.get("command"),
        },
        {
            "resource": "cached_sota_pair",
            "available": bool(pair_result.get("returned_two_loadable_specs")),
            "detail": pair_result.get("result")
            if pair_result.get("error") is None
            else pair_result.get("error"),
        },
        {
            "resource": "mandated_sota_gguf_cache",
            "available": cached_count > 0,
            "detail": f"cached_count={cached_count}",
        },
    ]


def _honest_verdict(
    *,
    ready: bool,
    torch_cuda: bool,
    llama_gpu: bool,
    cached_count: int,
    prompt_attempted: bool,
) -> str:
    """Map the gate state to an explicit terminal verdict."""
    if ready:
        return "success: mandated SOTA GGUF produced usable GPU-backed output"
    if not torch_cuda:
        return "blocked_cuda: selected .venv python did not report CUDA-capable torch"
    if not llama_gpu:
        return "blocked_llama_cpp_gpu_offload: llama_cpp imports but GPU offload support is unavailable"
    if cached_count == 0:
        return "blocked_model_cache: no mandated SOTA GGUF resolved locally"
    if prompt_attempted:
        return "blocked_prompt_smoke: mandated GGUF load/generation did not produce usable GPU-backed output"
    return "blocked_preconditions: SOTA runtime v3 preconditions were not all satisfied"


def _model_specs() -> list[JsonDict]:
    """Return the mandated candidates in the task-requested order."""
    return [
        {
            "priority": index,
            "hf_id": hf_id,
            "name": _MODEL_BY_HF_ID.get(hf_id, {}).get("name"),
            "required_role": "primary_candidate",
            "legacy_smoke_only": False,
        }
        for index, hf_id in enumerate(MANDATED_MODEL_IDS, start=1)
    ]


def _reproducibility_checksum(
    *,
    seed: int,
    selected_model: Mapping[str, Any] | None,
    cache_inventory: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the deterministic pieces of the resolver without reading huge GGUFs."""
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    for row in cache_inventory:
        digest.update(
            json.dumps(
                {k: row.get(k) for k in ("hf_id", "path", "resolved_path")}, sort_keys=True
            ).encode()
        )
    if selected_model:
        path = Path(str(selected_model.get("path")))
        digest.update(str(path).encode("utf-8"))
        if path.exists():
            digest.update(str(path.stat().st_size).encode("utf-8"))
            digest.update(str(path.resolve()).encode("utf-8"))
    return digest.hexdigest()


def build_runtime_resolver_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 2862 v3 artifact without running downstream evaluation."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    torch_probe = _torch_cuda_probe(selected, command_runner=command_runner)
    gpu_inventory = _nvidia_smi_inventory(command_runner=command_runner)
    llama_probe = _llama_cpp_probe(selected, command_runner=command_runner, env=merged_env)
    cache_inventory = _inspect_mandated_cache(root, merged_env)
    pair_result = _exercise_cached_sota_pair(cached_pair_fn)
    selected_model = next(
        (row for row in cache_inventory if row["cache_status"] == "resolved"), None
    )
    prompt_rows: list[JsonDict] = []
    can_prompt = bool(
        torch_probe.get("cuda_available")
        and llama_probe.get("llama_cpp_supports_gpu_offload")
        and selected_model is not None
    )
    if can_prompt:
        prompt_model = dict(selected_model)
        prompt_model["gpu"] = 0
        prompt_rows.append(
            prompt_runner_fn(
                prompt_model,
                selected_python=selected,
                command_runner=command_runner,
                env=merged_env,
            )
        )

    usable_rows = [row for row in prompt_rows if row.get("usable") and row.get("gpu_backed")]
    total_tokens = sum(int(row.get("tokens_generated") or 0) for row in usable_rows)
    total_prompt_duration = sum(float(row.get("duration_s") or 0.0) for row in usable_rows)
    tokens_per_second = total_tokens / total_prompt_duration if total_prompt_duration > 0 else 0.0
    ready = bool(usable_rows)
    finished = monotonic()
    missing_models = [row["hf_id"] for row in cache_inventory if row["cache_status"] != "resolved"]

    artifact: JsonDict = {
        "artifact": "experiment_2862_sota_runtime_cache_offload_resolver_v3",
        "schema_version": 1,
        "honest_verdict": _honest_verdict(
            ready=ready,
            torch_cuda=bool(torch_probe.get("cuda_available")),
            llama_gpu=bool(llama_probe.get("llama_cpp_supports_gpu_offload")),
            cached_count=len(cache_inventory) - len(missing_models),
            prompt_attempted=bool(prompt_rows),
        ),
        "sota_runtime_ready_v3": ready,
        "model_specs": _model_specs(),
        "selected_model_hf_id": str(selected_model["hf_id"]) if selected_model else "",
        "selected_model_path": str(selected_model["path"]) if selected_model else "",
        "cached_sota_pair_returned_two_loadable_specs": bool(
            pair_result["returned_two_loadable_specs"]
        ),
        "llama_cpp_gpu_offload_verified": bool(llama_probe.get("llama_cpp_supports_gpu_offload")),
        "preconditions_checked": _preconditions(
            torch_probe=torch_probe,
            gpu_inventory=gpu_inventory,
            llama_probe=llama_probe,
            pair_result=pair_result,
            cache_inventory=cache_inventory,
        ),
        "usable_response_count": len(usable_rows),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": round(tokens_per_second, 6),
        "legacy_small_models_used_only_for_smoke": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(
            seed=RANDOM_SEED,
            selected_model=selected_model,
            cache_inventory=cache_inventory,
        ),
        "tests_run": list(tests_run or []),
        "field_principles": {
            "sota_runtime_ready_v3": "True only for usable GPU-backed output from a mandated local SOTA GGUF.",
            "cached_sota_pair_returned_two_loadable_specs": "Records pair readiness separately from single-model v3 runtime readiness.",
            "llama_cpp_gpu_offload_verified": "Mirrors llama_cpp.llama_cpp.llama_supports_gpu_offload(); never inferred.",
            "legacy_small_models_used_only_for_smoke": "Legacy Qwen/Gemma IDs cannot satisfy headline readiness.",
            "duration_s": "Measured wall-clock duration; no sleep padding.",
        },
        "run_date": run_date,
        "duration_s": round(finished - started, 6),
        "selected_python": selected,
        "torch_cuda_probe": torch_probe,
        "gpu_inventory": gpu_inventory,
        "llama_cpp_probe": llama_probe,
        "llama_cpp_cuda_reinstall_command": _llama_cpp_cuda_reinstall_command(),
        "cache_locations": _cache_roots(root, merged_env),
        "cache_inventory": cache_inventory,
        "models_missing_from_cache": missing_models,
        "cached_sota_pair_result": pair_result,
        "bounded_prompt_results": prompt_rows,
        "legacy_cpu_smoke_only_model_ids": list(LEGACY_CPU_SMOKE_ONLY),
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for conductor and downstream gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    run_date: str = "20260522",
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 2862 v3 artifact."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_runtime_resolver_artifact(
        project_root=root,
        run_date=run_date,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        prompt_runner_fn=prompt_runner_fn,
        monotonic=monotonic,
        tests_run=tests_run,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default="20260522")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        output_path=args.output,
        selected_python=args.selected_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
