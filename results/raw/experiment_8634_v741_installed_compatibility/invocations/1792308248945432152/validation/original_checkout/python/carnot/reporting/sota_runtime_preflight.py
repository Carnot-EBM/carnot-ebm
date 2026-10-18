"""Fail-fast SOTA runtime/cache manifest for Exp 2836.

**Researcher summary:**
    The .268 corpus tasks failed before measuring anything useful because the
    conductor used system ``python3`` without torch while the project venv had
    the CUDA-capable runtime.  This module writes one small manifest that makes
    that runtime contract explicit: downstream live-model tasks must use the
    project ``.venv/bin/python``, must see CUDA through torch there, and must
    find at least one mandated SOTA GGUF that a local loader can smoke-load.

**Detailed explanation for engineers:**
    The code is deliberately a preflight, not an evaluation.  It performs local
    diagnostics, resolves cache paths, hashes only the selected local GGUFs, and
    runs a bounded llama.cpp load smoke.  It never asks a model a corpus question
    and it never substitutes legacy tiny models for headline readiness.  When no
    mandated model is cached, it records a terminal cache blocker rather than
    starting an unbounded download.

Spec: REQ-INFER-SOTA-012,
      SCENARIO-INFER-SOTA-012-001,
      SCENARIO-INFER-SOTA-012-002,
      SCENARIO-INFER-SOTA-012-003
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair


DEFAULT_ARTIFACT_PATH = Path("results/experiment_2836_sota_runtime_preflight.json")
PRIMARY_SOTA_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY: tuple[str, ...] = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "sota_runtime_ready",
    "selected_python",
    "venv_torch_cuda_available",
    "system_python_torch_cuda_available",
    "sota_models_cached",
    "cached_sota_pair_result",
    "model_specs",
    "preconditions_checked",
    "duration_s",
)
_QUANTIZATION_TOKENS: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q3_K_M",
    "UD-Q3_K_S",
    "UD-IQ4_NL",
    "UD-IQ4_XS",
    "UD-IQ3_S",
    "Q8_0",
    "BF16",
)
_MODEL_BY_HF_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}

JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
SmokeLoaderFn = Callable[..., JsonDict]
ClockFn = Callable[[], float]


def _repo_root() -> Path:
    """Return the repo root, preferring the canonical environment override."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _selected_python(project_root: Path) -> str:
    """Return the interpreter path downstream corpus tasks must use."""
    candidate = project_root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 2000) -> str:
    """Keep command evidence compact while preserving the failure prefix."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a diagnostic command and convert every outcome to structured data."""
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


def _torch_probe(python_executable: str, *, command_runner: CommandRunner) -> JsonDict:
    """Measure torch/CUDA with the exact interpreter named in the manifest."""
    command = [
        python_executable,
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available())",
    ]
    result = command_runner(command, timeout_s=30)
    output = _stdout(result).strip()
    parts = output.split()
    cuda_available = bool(result.get("returncode") == 0 and parts and parts[-1] == "True")
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "torch_version": parts[0] if parts else None,
        "cuda_available": cuda_available,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _disk_probe(project_root: Path, *, command_runner: CommandRunner) -> JsonDict:
    """Record free disk space because cache materialization can consume hundreds of GB."""
    command = ["df", "-k", str(project_root)]
    result = command_runner(command, timeout_s=10)
    lines = [line.split() for line in _stdout(result).splitlines() if line.strip()]
    parsed: JsonDict = {"command": result.get("command", command), "returncode": result.get("returncode")}
    if len(lines) >= 2 and len(lines[1]) >= 5:
        parsed.update(
            {
                "filesystem": lines[1][0],
                "size_kb": int(lines[1][1]),
                "used_kb": int(lines[1][2]),
                "available_kb": int(lines[1][3]),
                "use_pct": lines[1][4],
            }
        )
    parsed["stdout_summary"] = _summarize(_stdout(result))
    parsed["stderr_summary"] = _summarize(_stderr(result))
    return parsed


def _gpu_memory_probe(*, command_runner: CommandRunner) -> JsonDict:
    """Record GPU memory when NVIDIA tooling is available."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    gpus: list[JsonDict] = []
    if result.get("returncode") == 0:
        for line in _stdout(result).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_used_mib": int(parts[3]),
                        "memory_free_mib": int(parts[4]),
                    }
                )
            except ValueError:
                continue
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "available": bool(gpus),
        "gpus": gpus,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _loader_probe(
    selected_python: str,
    *,
    command_runner: CommandRunner,
    env: Mapping[str, str] | None,
) -> JsonDict:
    """Check whether the selected interpreter can import llama.cpp."""
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
        "        'llama_cpp_supports_mmap': bool(low.llama_supports_mmap()),\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-c", script]
    result = command_runner(command, timeout_s=30, env=dict(env or {}))
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
    """Locate the cache roots the resolver should inspect."""
    if env.get("HUGGINGFACE_HUB_CACHE"):
        hf_cache = Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser()
    elif env.get("HF_HOME"):
        hf_cache = Path(env["HF_HOME"]).expanduser() / "hub"
    else:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    models_root = project_root / "models"
    return {
        "huggingface_hub_cache": str(hf_cache),
        "huggingface_hub_cache_exists": hf_cache.exists(),
        "local_models": str(models_root),
        "local_models_exists": models_root.exists(),
    }


def _model_family(hf_id: str) -> str:
    """Classify the model family from the mandated hub id."""
    lowered = hf_id.lower()
    return "qwen" if "qwen" in lowered else "gemma"


def _model_filename_token(hf_id: str) -> str:
    """Return the basename token that real model GGUF files include."""
    basename = hf_id.split("/", 1)[-1]
    return basename.removesuffix("-GGUF").lower()


def _matches_model_file(path: Path, hf_id: str) -> bool:
    """Reject projector files and unrelated GGUFs in broad cache searches."""
    name = path.name.lower()
    return name.endswith(".gguf") and "mmproj" not in name and _model_filename_token(hf_id) in name


def _local_model_dirs(models_root: Path, hf_id: str) -> list[Path]:
    """Mirror the local directory conventions used by the shared resolver."""
    basename = hf_id.split("/", 1)[-1]
    stripped = basename.removesuffix("-GGUF")
    return [
        models_root / stripped,
        models_root / basename,
        models_root / stripped.lower(),
        models_root / basename.lower(),
    ]


def _candidate_paths(project_root: Path, hf_id: str, roots: Mapping[str, Any]) -> list[Path]:
    """Find local GGUF candidates for one mandated hub id."""
    candidates: list[Path] = []
    hf_cache = Path(str(roots["huggingface_hub_cache"]))
    hub_model_dir = hf_cache / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    if hub_model_dir.is_dir():
        candidates.extend(path for path in hub_model_dir.rglob("*.gguf") if _matches_model_file(path, hf_id))

    models_root = project_root / "models"
    for model_dir in _local_model_dirs(models_root, hf_id):
        if model_dir.is_dir():
            candidates.extend(path for path in model_dir.rglob("*.gguf") if _matches_model_file(path, hf_id))

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _candidate_size(path: Path) -> int:
    """Return the target size for regular files and HF cache symlinks."""
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _quantization_suffix(path: str | None) -> str | None:
    """Extract the visible quantization token from a GGUF filename."""
    if path is None:
        return None
    filename = Path(path).name.lower()
    for token in _QUANTIZATION_TOKENS:
        if token.lower() in filename:
            return token
    return "unknown"


def _select_candidate(paths: Sequence[Path]) -> Path | None:
    """Pick a deterministic preferred candidate from local cache hits."""
    existing = [path for path in paths if _candidate_size(path) > 0]
    if not existing:
        return None
    for token in _QUANTIZATION_TOKENS:
        matches = [path for path in existing if token.lower() in path.name.lower()]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    return max(existing, key=lambda path: path.stat().st_mtime)


def _sha256_file(path: Path) -> str:
    """Hash the selected local GGUF so downstream tasks can detect cache drift."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_model_cache(project_root: Path, env: Mapping[str, str], roots: Mapping[str, Any]) -> tuple[list[JsonDict], list[str]]:
    """Inspect every mandated primary model and return selected cache hits."""
    cached: list[JsonDict] = []
    missing: list[str] = []
    for hf_id in PRIMARY_SOTA_MODEL_IDS:
        candidates = _candidate_paths(project_root, hf_id, roots)
        selected = _select_candidate(candidates)
        if selected is None:
            missing.append(hf_id)
            continue
        spec = _MODEL_BY_HF_ID.get(hf_id, {})
        resolved = selected.resolve()
        cached.append(
            {
                "hf_id": hf_id,
                "name": spec.get("name"),
                "model_family": _model_family(hf_id),
                "role": spec.get("role"),
                "path": str(selected),
                "resolved_path": str(resolved),
                "size_bytes": _candidate_size(selected),
                "sha256": _sha256_file(resolved),
                "expected_quantization": spec.get("quantization"),
                "observed_quantization": _quantization_suffix(str(selected)),
                "min_vram_gb": spec.get("min_vram_gb"),
                "candidate_count": len(candidates),
            }
        )
    return cached, missing


def _json_safe(value: Any) -> Any:
    """Convert helper return values to stable JSON-compatible structures."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_json_safe(item) for item in value]
        return str(value)


def _exercise_cached_sota_pair(cached_pair_fn: CachedPairFn) -> JsonDict:
    """Import the experiment template and call the mandated cached-pair helper."""
    result: JsonDict = {
        "experiment_template_import_ok": False,
        "called": False,
        "result": None,
        "error": None,
    }
    try:
        importlib.import_module("scripts.experiment_template")
        result["experiment_template_import_ok"] = True
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
        result["called"] = True
        result["result"] = _json_safe(pair)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _hf_credentials_configured(env: Mapping[str, str]) -> bool:
    """Return whether a metadata-only HF probe is allowed by local config."""
    if env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN"):
        return True
    if env.get("HF_HOME"):
        token_path = Path(env["HF_HOME"]).expanduser() / "token"
    elif env.get("HUGGINGFACE_HUB_CACHE"):
        token_path = Path(env["HUGGINGFACE_HUB_CACHE"]).expanduser().parent / "token"
    else:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
    return token_path.is_file()


def _blocked_model_cache_attempt(
    *,
    missing_models: Sequence[str],
    cached_models_present: bool,
    env: Mapping[str, str],
) -> JsonDict:
    """Record why no model download was started."""
    if cached_models_present:
        return {"attempted": False, "status": "not_required_cache_present", "missing_models": list(missing_models)}
    if not _hf_credentials_configured(env):
        return {
            "attempted": False,
            "status": "skipped_no_local_credentials",
            "missing_models": list(missing_models),
            "principle": "No blind large download without configured local credentials/tooling.",
        }
    return {
        "attempted": False,
        "status": "metadata_only_probe_allowed_but_not_needed_for_automated_preflight",
        "missing_models": list(missing_models),
        "principle": "Avoid wedging the session on a huge cache fill.",
    }


def _smoke_load_model(
    model: Mapping[str, Any],
    *,
    selected_python: str,
    loader_probe: Mapping[str, Any],
    gpu_probe: Mapping[str, Any],
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Run a bounded llama.cpp full-load smoke in a subprocess."""
    del gpu_probe
    if not loader_probe.get("llama_cpp_import_ok"):
        return {
            "hf_id": model["hf_id"],
            "model_path": model["path"],
            "load_attempted": False,
            "load_success": False,
            "headline_usable": False,
            "blocker": loader_probe.get("error") or "llama_cpp_import_failed",
        }
    script = (
        "import gc, json, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "path = sys.argv[1]\n"
        "gpu = int(sys.argv[2])\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "kwargs = dict(model_path=path, n_ctx=64, n_batch=16, n_ubatch=16, use_mmap=True, verbose=False)\n"
        "kwargs['n_gpu_layers'] = -1 if supports_gpu else 0\n"
        "if supports_gpu:\n"
        "    kwargs['main_gpu'] = gpu\n"
        "started = time.monotonic()\n"
        "llm = Llama(**kwargs)\n"
        "elapsed = time.monotonic() - started\n"
        "n_ctx = llm.n_ctx()\n"
        "llm.close()\n"
        "gc.collect()\n"
        "print(json.dumps({\n"
        "    'load_success': True,\n"
        "    'headline_usable': True,\n"
        "    'elapsed_s': round(elapsed, 6),\n"
        "    'load_mode': 'llama_cpp_full_context_load',\n"
        "    'n_ctx': n_ctx,\n"
        "    'llama_cpp_supports_gpu_offload': supports_gpu,\n"
        "    'n_gpu_layers': kwargs['n_gpu_layers'],\n"
        "}, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", script, str(model["path"]), "0"], timeout_s=180, env=dict(env))
    row: JsonDict = {
        "hf_id": model["hf_id"],
        "model_path": model["path"],
        "load_attempted": True,
        "load_success": False,
        "headline_usable": False,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
        "blocker": None,
    }
    try:
        parsed = json.loads(_stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {}
    if result.get("returncode") == 0 and parsed.get("load_success") is True:
        row.update(parsed)
        row["blocker"] = None
    else:
        row["blocker"] = _stderr(result) or _stdout(result) or "llama_cpp_load_smoke_failed"
    return row


def _preconditions(
    *,
    venv_probe: Mapping[str, Any],
    system_probe: Mapping[str, Any],
    disk_probe: Mapping[str, Any],
    gpu_probe: Mapping[str, Any],
    roots: Mapping[str, Any],
    loader_probe: Mapping[str, Any],
    cached_count: int,
) -> list[JsonDict]:
    """Create the compact checklist every downstream task can gate on."""
    return [
        {
            "resource": "venv_torch_cuda",
            "available": bool(venv_probe.get("cuda_available")),
            "detail": venv_probe.get("stdout_summary") or venv_probe.get("stderr_summary"),
            "command": venv_probe.get("command"),
        },
        {
            "resource": "system_python_torch_cuda",
            "available": bool(system_probe.get("cuda_available")),
            "detail": system_probe.get("stdout_summary") or system_probe.get("stderr_summary"),
            "command": system_probe.get("command"),
        },
        {
            "resource": "disk_free",
            "available": bool(disk_probe.get("available_kb", 0) > 0),
            "detail": f"available_kb={disk_probe.get('available_kb')}",
            "command": disk_probe.get("command"),
        },
        {
            "resource": "gpu_memory",
            "available": bool(gpu_probe.get("available")),
            "detail": gpu_probe.get("gpus", []),
            "command": gpu_probe.get("command"),
        },
        {
            "resource": "huggingface_cache",
            "available": bool(roots.get("huggingface_hub_cache_exists")),
            "detail": roots.get("huggingface_hub_cache"),
        },
        {
            "resource": "local_models_dir",
            "available": bool(roots.get("local_models_exists")),
            "detail": roots.get("local_models"),
        },
        {
            "resource": "llama_cpp_loader",
            "available": bool(loader_probe.get("llama_cpp_import_ok")),
            "detail": loader_probe.get("llama_cpp_origin") or loader_probe.get("error"),
            "command": loader_probe.get("command"),
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
    venv_cuda: bool,
    cached_count: int,
    smoke_success: bool,
) -> str:
    """Return terminal-prefix verdict text for the manifest."""
    if ready:
        return "success: .venv CUDA torch available and at least one mandated SOTA GGUF load-smoked"
    if not venv_cuda:
        return "blocked_cuda: selected .venv python did not report CUDA-capable torch"
    if cached_count == 0:
        return "blocked_model_cache: no mandated primary SOTA GGUF resolved locally"
    if not smoke_success:
        return "blocked_loader_smoke: mandated SOTA GGUF cache exists but no loader smoke succeeded"
    return "blocked_unknown: SOTA runtime preflight did not meet readiness conditions"


def build_runtime_cache_manifest(
    *,
    project_root: str | Path,
    run_date: str,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    smoke_loader_fn: SmokeLoaderFn = _smoke_load_model,
    monotonic: ClockFn = time.monotonic,
) -> JsonDict:
    """Build the Exp 2836 manifest without running corpus evaluation."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    venv_probe = _torch_probe(selected, command_runner=command_runner)
    system_probe = _torch_probe("python3", command_runner=command_runner)
    disk = _disk_probe(root, command_runner=command_runner)
    gpu = _gpu_memory_probe(command_runner=command_runner)
    roots = _cache_roots(root, merged_env)
    loader = _loader_probe(selected, command_runner=command_runner, env=merged_env)
    cached_models, missing_models = _inspect_model_cache(root, merged_env, roots)
    cached_pair_result = _exercise_cached_sota_pair(cached_pair_fn)
    blocked_model_cache = _blocked_model_cache_attempt(
        missing_models=missing_models,
        cached_models_present=bool(cached_models),
        env=merged_env,
    )
    smoke_rows = [
        smoke_loader_fn(
            model,
            selected_python=selected,
            loader_probe=loader,
            gpu_probe=gpu,
            command_runner=command_runner,
            env=merged_env,
        )
        for model in cached_models
    ]
    smoke_success = any(row.get("load_success") and row.get("headline_usable") for row in smoke_rows)
    venv_cuda = bool(venv_probe.get("cuda_available"))
    ready = bool(venv_cuda and smoke_success)
    finished = monotonic()

    artifact: JsonDict = {
        "artifact": "experiment_2836_sota_runtime_preflight",
        "schema_version": 1,
        "run_date": run_date,
        "honest_verdict": _honest_verdict(
            ready=ready,
            venv_cuda=venv_cuda,
            cached_count=len(cached_models),
            smoke_success=smoke_success,
        ),
        "sota_runtime_ready": ready,
        "selected_python": selected,
        "venv_torch_cuda_available": venv_cuda,
        "system_python_torch_cuda_available": bool(system_probe.get("cuda_available")),
        "sota_models_cached": cached_models,
        "cached_sota_pair_result": cached_pair_result,
        "model_specs": {
            "primary": list(PRIMARY_SOTA_MODEL_IDS),
            "legacy_cpu_smoke_only": list(LEGACY_CPU_SMOKE_ONLY),
        },
        "preconditions_checked": _preconditions(
            venv_probe=venv_probe,
            system_probe=system_probe,
            disk_probe=disk,
            gpu_probe=gpu,
            roots=roots,
            loader_probe=loader,
            cached_count=len(cached_models),
        ),
        "duration_s": round(finished - started, 6),
        "models_missing_from_cache": list(missing_models),
        "blocked_model_cache": blocked_model_cache,
        "smoke_load_results": smoke_rows,
        "venv_torch_probe": venv_probe,
        "system_python_torch_probe": system_probe,
        "disk_probe": disk,
        "gpu_memory_probe": gpu,
        "cache_locations": roots,
        "loader_probe": loader,
        "field_principles": {
            "honest_verdict": 'Must start with "complete:" / "success:" or "blocked_".',
            "sota_runtime_ready": "Structured gate for downstream live-model tasks.",
            "selected_python": "Exact interpreter path downstream tasks must use.",
            "venv_torch_cuda_available": "CUDA torch readiness measured in the selected venv.",
            "system_python_torch_cuda_available": "Documents the .268 system-python mismatch.",
            "sota_models_cached": "Mandated GGUF cache evidence for headline use.",
            "cached_sota_pair_result": "Verifies the mandated template pattern.",
            "model_specs": "Records SOTA GGUF mandate and legacy smoke-test limitation.",
            "preconditions_checked": "Every expensive downstream task gates on these checks.",
            "duration_s": "Real preflight duration; no sleep padding.",
        },
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON so the conductor sees a stable terminal artifact."""
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
    smoke_loader_fn: SmokeLoaderFn = _smoke_load_model,
    monotonic: ClockFn = time.monotonic,
) -> JsonDict:
    """Build and persist the Exp 2836 manifest."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_runtime_cache_manifest(
        project_root=root,
        run_date=run_date,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        smoke_loader_fn=smoke_loader_fn,
        monotonic=monotonic,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default="20260522")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        output_path=args.output,
        selected_python=args.selected_python,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI invocation.
    raise SystemExit(main())
