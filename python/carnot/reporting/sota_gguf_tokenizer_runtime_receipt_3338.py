"""Build the Exp 3338 SOTA GGUF tokenizer/runtime receipt.

Spec refs: REQ-INFER-SOTA-3338, SCENARIO-INFER-SOTA-3338-001,
SCENARIO-INFER-SOTA-3338-002.

This is a preflight artifact, not a benchmark. It inventories the three
mandated local GGUF families, checks tokenizer and llama.cpp runtime imports,
then tries the smallest load/tokenize/generate smoke for every available
mandated model. A single successful mandated smoke is enough to prove the
runtime path for downstream preconditions; legacy small models never count.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]
CachedPairResolver = Callable[..., Sequence[Mapping[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.sota_gguf_tokenizer_runtime_receipt.v1"
EXPERIMENT_ID = "exp3338"
TASK_ID = "exp3338-sota-gguf-tokenizer-runtime-receipt-v1"
ARTIFACT = "experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1"
RUN_DATE = "20260529"
RANDOM_SEED = 3338
INFERENCE_SUBSTRATE = "live_llm_inference"

OUTPUT_REL_PATH = Path("results/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.json")
SCRIPT_REL_PATH = Path("scripts/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3338_sota_gguf_tokenizer_runtime_receipt.py")
SPEC_REL_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_BY_ID = {str(model["hf_id"]): dict(model) for model in SOTA_GGUF_MODELS}
QUANTIZATION_TOKENS: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q8_XL",
    "Q8_0",
    "BF16",
)
DEFAULT_PROMPT = "Exp 3338 runtime receipt. Reply with exactly one token: READY."
DEFAULT_MAX_TOKENS = 2
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_WORKER_TIMEOUT_S = 1800
REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "files_updated",
    "model_specs",
    "cache_status",
    "tokenizer_status",
    "loader_status",
    "gpu_status",
    "smoke_generation_status",
    "runtime_receipt_clean",
    "blocked_reasons",
}

WORKER_CODE = r'''
import argparse
import json
import time


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
parser.add_argument("--exp3338-runtime-worker", action="store_true")
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-tokens", type=int, required=True)
parser.add_argument("--n-gpu-layers", type=int, required=True)
parser.add_argument("--main-gpu", type=int, required=True)
args = parser.parse_args()

started = time.monotonic()
llm = None
try:
    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model_path,
        n_ctx=128,
        n_batch=16,
        n_ubatch=16,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=args.main_gpu,
        verbose=True,
    )
    load_status = "loaded"
    prompt_bytes = args.prompt.encode("utf-8")
    try:
        prompt_tokens = llm.tokenize(prompt_bytes, add_bos=True)
    except TypeError:
        prompt_tokens = llm.tokenize(prompt_bytes)
    raw = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=args.seed,
    )
    output = _response_text(raw).strip()
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = len(output.split()) if output else 0
    print(
        json.dumps(
            {
                "ok": bool(output) and completion_tokens > 0,
                "model_id": args.model_id,
                "load_status": load_status,
                "tokenize_status": "tokenized",
                "generation_status": "generated" if output and completion_tokens > 0 else "empty_response",
                "prompt_token_count": len(prompt_tokens),
                "output_text": output,
                "tokens_generated": int(completion_tokens),
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
                "tokenize_status": "not_attempted",
                "generation_status": "failed",
                "prompt_token_count": 0,
                "output_text": "",
                "tokens_generated": 0,
                "error": f"{type(exc).__name__}: {exc}",
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


def build_artifact(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    cache_roots: Sequence[str | Path] | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = None,
    cached_pair_resolver: CachedPairResolver = cached_sota_pair,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> JsonDict:
    """REQ-INFER-SOTA-3338: build the runtime receipt or precise blocker."""

    del output_path
    start = monotonic()
    root = Path(project_root)
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    runner = command_runner or run_command
    selected = str(selected_python or selected_python_for(root))
    roots = [Path(path).expanduser() for path in (cache_roots or default_cache_roots(root, merged_env))]

    cached_pair_result, cached_pair_error = resolve_cached_pair(cached_pair_resolver)
    cache_inventory = inspect_cache(roots)
    model_specs = build_model_specs(cache_inventory, cached_pair_result)
    tokenizer_probe = probe_tokenizer_dependencies(selected, runner)
    loader_probe = probe_llama_cpp_loader(selected, runner, merged_env)
    gpu_status = {
        "authority": "torch.cuda",
        "torch_cuda": probe_torch_cuda(selected, runner),
        "nvidia_smi": probe_nvidia_smi(runner),
    }

    per_model_smoke: dict[str, JsonDict] = {}
    loader_ok = loader_probe.get("llama_cpp_import_ok") is True
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        model_path = spec.get("model_path")
        if not model_path:
            per_model_smoke[model_id] = not_attempted_smoke(model_id, "model_not_cached")
            continue
        if not loader_ok:
            detail = str(loader_probe.get("loader_error") or "llama_cpp import failed")
            per_model_smoke[model_id] = not_attempted_smoke(model_id, detail)
            continue
        worker = run_runtime_worker(
            selected_python=selected,
            model_id=model_id,
            model_path=str(model_path),
            gpu=safe_int(spec.get("gpu")) or 0,
            random_seed=int(random_seed),
            max_tokens=int(max_tokens),
            n_gpu_layers=int(n_gpu_layers),
            env=merged_env,
            command_runner=runner,
            timeout_s=int(worker_timeout_s),
        )
        per_model_smoke[model_id] = smoke_from_worker(model_id, str(model_path), worker)

    clean_ids = [
        model_id
        for model_id, row in per_model_smoke.items()
        if row.get("runtime_receipt_passed") is True
    ]
    runtime_receipt_clean = bool(clean_ids) and gpu_status["torch_cuda"].get("cuda_available") is True
    blocked_reasons = blocked_reasons_for(
        runtime_receipt_clean=runtime_receipt_clean,
        cache_inventory=cache_inventory,
        loader_status=loader_probe,
        gpu_status=gpu_status,
        per_model_smoke=per_model_smoke,
    )
    tokenizer_status = {
        **tokenizer_probe,
        "per_model": {
            model_id: {
                "tokenize_status": row.get("tokenize_status"),
                "prompt_token_count": row.get("prompt_token_count"),
                "exception": row.get("exception"),
            }
            for model_id, row in per_model_smoke.items()
        },
    }
    smoke_generation_status = {
        "prompt": DEFAULT_PROMPT,
        "max_tokens": int(max_tokens),
        "n_gpu_layers": int(n_gpu_layers),
        "per_model": per_model_smoke,
        "attempted_model_ids": [
            model_id for model_id, row in per_model_smoke.items() if row.get("attempted") is True
        ],
        "clean_mandated_model_ids": clean_ids if runtime_receipt_clean else [],
        "legacy_cpu_loader_controls": [],
    }
    cache_status = build_cache_status(
        roots=roots,
        cache_inventory=cache_inventory,
        cached_pair_result=cached_pair_result,
        cached_pair_error=cached_pair_error,
    )
    duration_s = round(max(0.0, monotonic() - start), 6)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "selected_python": selected,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "files_updated": files_updated(),
        "model_specs": model_specs,
        "cache_status": cache_status,
        "tokenizer_status": tokenizer_status,
        "loader_status": loader_probe,
        "gpu_status": gpu_status,
        "smoke_generation_status": smoke_generation_status,
        "runtime_receipt_clean": runtime_receipt_clean,
        "blocked_reasons": blocked_reasons,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    cache_roots: Sequence[str | Path] | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = None,
    cached_pair_resolver: CachedPairResolver = cached_sota_pair,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> JsonDict:
    """Build and persist the Exp 3338 receipt JSON."""

    root = Path(project_root)
    destination = Path(output_path) if output_path is not None else root / OUTPUT_REL_PATH
    if not destination.is_absolute():
        destination = root / destination
    artifact = build_artifact(
        project_root=root,
        output_path=destination,
        cache_roots=cache_roots,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_resolver=cached_pair_resolver,
        monotonic=monotonic,
        random_seed=random_seed,
        max_tokens=max_tokens,
        n_gpu_layers=n_gpu_layers,
        worker_timeout_s=worker_timeout_s,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def resolve_cached_pair(resolver: CachedPairResolver) -> tuple[list[JsonDict], str]:
    """Call cached_sota_pair(gpu_indices=(0, 1)) and normalize its result."""

    try:
        raw = resolver(gpu_indices=(0, 1))
    except Exception as exc:  # pragma: no cover - defensive around local cache helper bugs.
        return [], f"{type(exc).__name__}: {exc}"
    if not raw:
        return [], ""
    return [dict(row) for row in raw], ""


def inspect_cache(cache_roots: Sequence[Path]) -> dict[str, JsonDict]:
    """Inspect local GGUF candidates for every mandated model without downloads."""

    inventory: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_IDS:
        records = candidate_records(model_id, cache_roots)
        selected = select_candidate(records)
        evidence = file_evidence(selected["path"]) if selected else {"status": "missing"}
        inventory[model_id] = {
            "hf_id": model_id,
            "cached": selected is not None,
            "selected_path": str(selected["path"]) if selected else None,
            "candidate_count": len(records),
            "candidate_paths": [str(row["path"]) for row in records],
            "file_evidence": evidence,
        }
    return inventory


def build_model_specs(
    cache_inventory: Mapping[str, Mapping[str, Any]],
    cached_pair_result: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return MODEL_SPECS-shaped rows while preserving all mandated IDs."""

    pair_by_id = {str(row.get("hf_id")): dict(row) for row in cached_pair_result}
    specs: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        registry = MODEL_BY_ID.get(model_id, {})
        cache_row = cache_inventory.get(model_id, {})
        pair_row = pair_by_id.get(model_id, {})
        model_path = pair_row.get("model_path") or cache_row.get("selected_path")
        specs.append(
            {
                "name": pair_row.get("name") or registry.get("name") or model_id.split("/", 1)[-1],
                "hf_id": model_id,
                "role": registry.get("role") or "unknown",
                "expected_quantization": registry.get("quantization") or "Q4_K_M",
                "min_vram_gb": registry.get("min_vram_gb"),
                "gpu": pair_row.get("gpu"),
                "model_path": str(model_path) if model_path else None,
                "cached": bool(cache_row.get("cached")),
                "source": "cached_sota_pair" if pair_row else "cache_inventory",
                "headline_model": True,
            }
        )
    return specs


def build_cache_status(
    *,
    roots: Sequence[Path],
    cache_inventory: Mapping[str, Mapping[str, Any]],
    cached_pair_result: Sequence[Mapping[str, Any]],
    cached_pair_error: str,
) -> JsonDict:
    """Build the artifact cache-status block."""

    cached_ids = [model_id for model_id in MANDATED_MODEL_IDS if cache_inventory[model_id]["cached"]]
    return {
        "cache_roots": [str(path) for path in roots],
        "cached_sota_pair_called": True,
        "cached_sota_pair_error": cached_pair_error,
        "cached_sota_pair_returned_two_loadable_specs": len(cached_pair_result) >= 2,
        "cached_sota_pair_result": [dict(row) for row in cached_pair_result],
        "mandated_models": {model_id: dict(cache_inventory[model_id]) for model_id in MANDATED_MODEL_IDS},
        "cached_model_ids": cached_ids,
        "missing_model_ids": [model_id for model_id in MANDATED_MODEL_IDS if model_id not in cached_ids],
        "downloads_performed": False,
    }


def candidate_records(model_id: str, cache_roots: Sequence[Path]) -> list[JsonDict]:
    """Search HF-cache and project-local layouts for candidate GGUF files."""

    records: dict[str, JsonDict] = {}
    for root in cache_roots:
        for directory in local_model_dirs(root.expanduser(), model_id):
            if not directory.exists():
                continue
            for path in directory.rglob("*.gguf"):
                if path.is_file():
                    records.setdefault(str(path), candidate_record(path))
    return list(records.values())


def local_model_dirs(root: Path, model_id: str) -> list[Path]:
    """Return directories that may contain a local copy of one model family."""

    owner, name = model_id.split("/", 1)
    stripped = name.removesuffix("-GGUF")
    return [
        root / f"models--{owner}--{name}",
        root / stripped,
        root / name,
        root / stripped.lower(),
        root / name.lower(),
    ]


def candidate_record(path: Path) -> JsonDict:
    """Represent one discovered GGUF candidate."""

    stat = path.stat()
    usable = stat.st_size > 0 and "mmproj" not in path.name.lower() and ".no_exist" not in str(path)
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "quantization": quantization_from_name(path.name),
        "usable_candidate": usable,
    }


def select_candidate(records: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Pick the best local GGUF candidate by quantization preference."""

    usable = [dict(record) for record in records if record.get("usable_candidate")]
    if not usable:
        return None
    for token in QUANTIZATION_TOKENS:
        matches = [record for record in usable if token.lower() in str(record["filename"]).lower()]
        if matches:
            return max(matches, key=lambda record: int(record.get("size_bytes") or 0))
    return max(usable, key=lambda record: str(record.get("path") or ""))


def file_evidence(path: str | Path, *, full_sha_max_bytes: int = 64 * 1024 * 1024) -> JsonDict:
    """Return size/checksum evidence while avoiding full hashes on huge GGUFs."""

    model_path = Path(path)
    if not model_path.is_file():
        return {"status": "missing", "path": str(model_path), "sha256": None}
    stat = model_path.stat()
    size = int(stat.st_size)
    if size <= full_sha_max_bytes:
        return {
            "status": "available",
            "path": str(model_path),
            "size_bytes": size,
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hash_bytes(model_path.read_bytes()),
            "checksum_algorithm": "sha256_full",
        }
    chunk_size = 1024 * 1024
    digest = hashlib.sha256()
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


def probe_tokenizer_dependencies(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Probe tokenizer support packages through the selected interpreter."""

    code = (
        "import importlib.metadata as md, importlib.util as util, json\n"
        "print('exp3338_tokenizer_dependency_probe')\n"
        "deps = {}\n"
        "for name in ('sentencepiece', 'tiktoken', 'tokenizers'):\n"
        "    available = util.find_spec(name) is not None\n"
        "    version = None\n"
        "    if available:\n"
        "        try:\n"
        "            version = md.version(name)\n"
        "        except Exception:\n"
        "            version = 'unknown'\n"
        "    deps[name] = {'available': available, 'version': version}\n"
        "print(json.dumps({'dependencies': deps}, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30)
    payload = first_json_line(str(result.get("stdout") or ""))
    deps = mapping(payload.get("dependencies"))
    for name in ("sentencepiece", "tiktoken", "tokenizers"):
        deps.setdefault(name, {"available": False, "version": None})
    return {
        "dependency_probe_returncode": result.get("returncode"),
        "dependencies": deps,
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_llama_cpp_loader(
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Import llama.cpp and record loader metadata without loading a model."""

    code = (
        "import importlib.util, json\n"
        "print('exp3338_llama_cpp_loader_probe')\n"
        "payload = {'llama_cpp_import_ok': False}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import Llama\n"
        "    from llama_cpp import llama_cpp as low\n"
        "    supports = getattr(low, 'llama_supports_gpu_offload', lambda: False)\n"
        "    payload.update({\n"
        "        'llama_cpp_import_ok': True,\n"
        "        'loader_name': 'llama_cpp.Llama',\n"
        "        'llama_cpp_version': getattr(llama_cpp, '__version__', None),\n"
        "        'llama_cpp_origin': importlib.util.find_spec('llama_cpp').origin,\n"
        "        'llama_cpp_supports_gpu_offload': bool(supports()),\n"
        "        'loader_error': '',\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload.update({'loader_name': 'llama_cpp.Llama', 'loader_error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30, env=dict(env))
    payload = first_json_line(str(result.get("stdout") or ""))
    import_ok = result.get("returncode") == 0 and payload.get("llama_cpp_import_ok") is True
    return {
        **payload,
        "llama_cpp_import_ok": import_ok,
        "llama_cpp_supports_gpu_offload": import_ok
        and payload.get("llama_cpp_supports_gpu_offload") is True,
        "returncode": result.get("returncode"),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_torch_cuda(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Probe GPU visibility through torch.cuda in the selected interpreter."""

    code = (
        "import importlib.util, json\n"
        "print('exp3338_torch_cuda_probe')\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    print(json.dumps({'torch_import_ok': False, 'cuda_available': False, 'device_count': 0}))\n"
        "else:\n"
        "    try:\n"
        "        import torch\n"
        "        print(json.dumps({'torch_import_ok': True, 'torch_version': getattr(torch, '__version__', None), "
        "'cuda_available': bool(torch.cuda.is_available()), 'device_count': int(torch.cuda.device_count()), "
        "'cuda_version': getattr(torch.version, 'cuda', None)}, sort_keys=True))\n"
        "    except Exception as exc:\n"
        "        print(json.dumps({'torch_import_ok': False, 'cuda_available': False, 'device_count': 0, "
        "'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30)
    payload = first_json_line(str(result.get("stdout") or ""))
    return {
        "torch_import_ok": result.get("returncode") == 0 and payload.get("torch_import_ok") is True,
        "torch_version": payload.get("torch_version"),
        "cuda_available": result.get("returncode") == 0 and payload.get("cuda_available") is True,
        "device_count": safe_int(payload.get("device_count")) or 0,
        "cuda_version": payload.get("cuda_version"),
        "error": str(payload.get("error") or ""),
        "returncode": result.get("returncode"),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_nvidia_smi(command_runner: CommandRunner) -> JsonDict:
    """Record nvidia-smi as secondary evidence, not as the authority."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    rows = parse_nvidia_smi_rows(str(result.get("stdout") or ""))
    return {
        "available": result.get("returncode") == 0 and bool(rows),
        "returncode": result.get("returncode"),
        "gpus": rows,
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def run_runtime_worker(
    *,
    selected_python: str,
    model_id: str,
    model_path: str,
    gpu: int,
    random_seed: int,
    max_tokens: int,
    n_gpu_layers: int,
    env: Mapping[str, str],
    command_runner: CommandRunner,
    timeout_s: int,
) -> JsonDict:
    """Run one isolated load/tokenize/generate smoke for a mandated GGUF."""

    command = [
        selected_python,
        "-c",
        WORKER_CODE,
        "--exp3338-runtime-worker",
        "--model-id",
        model_id,
        "--model-path",
        model_path,
        "--prompt",
        DEFAULT_PROMPT,
        "--seed",
        str(int(random_seed)),
        "--max-tokens",
        str(int(max_tokens)),
        "--n-gpu-layers",
        str(int(n_gpu_layers)),
        "--main-gpu",
        str(int(gpu)),
    ]
    worker_env = dict(env)
    worker_env["PYTHONHASHSEED"] = str(int(random_seed))
    worker_env["CARNOT_SOTA_MAIN_GPU"] = str(int(gpu))
    result = command_runner(command, timeout_s=timeout_s, env=worker_env)
    payload = first_json_line(str(result.get("stdout") or ""))
    stderr_tail = truncate_tail(str(result.get("stderr") or ""))
    return {
        "attempted": True,
        "returncode": result.get("returncode"),
        "payload": payload,
        "stderr_tail": stderr_tail,
        "command_hash": stable_hash(command),
    }


def smoke_from_worker(model_id: str, model_path: str, worker: Mapping[str, Any]) -> JsonDict:
    """Normalize a worker result into the per-model receipt schema."""

    payload = mapping(worker.get("payload"))
    exception = str(payload.get("error") or "")
    if not exception and worker.get("returncode") not in (0, None):
        exception = str(worker.get("stderr_tail") or "worker exited without JSON error")
    output = str(payload.get("output_text") or "").strip()
    tokens = safe_int(payload.get("tokens_generated")) or 0
    passed = (
        worker.get("returncode") == 0
        and payload.get("ok") is True
        and bool(output)
        and tokens > 0
        and payload.get("load_status") == "loaded"
        and payload.get("tokenize_status") == "tokenized"
    )
    return {
        "model_id": model_id,
        "model_path": model_path,
        "attempted": True,
        "returncode": worker.get("returncode"),
        "load_status": str(payload.get("load_status") or "unknown"),
        "tokenize_status": str(payload.get("tokenize_status") or "unknown"),
        "generation_status": str(payload.get("generation_status") or "unknown"),
        "prompt_token_count": safe_int(payload.get("prompt_token_count")) or 0,
        "tokens_generated": tokens,
        "output_nonempty": bool(output),
        "output_preview": output[:200],
        "usage": mapping(payload.get("usage")),
        "duration_s": safe_float(payload.get("duration_s")) or 0.0,
        "exception": exception,
        "stderr_tail": str(worker.get("stderr_tail") or ""),
        "command_hash": str(worker.get("command_hash") or ""),
        "runtime_receipt_passed": passed,
    }


def not_attempted_smoke(model_id: str, reason: str) -> JsonDict:
    """Return the per-model shape when no runtime worker was launched."""

    return {
        "model_id": model_id,
        "model_path": None,
        "attempted": False,
        "returncode": None,
        "load_status": "not_attempted",
        "tokenize_status": "not_attempted",
        "generation_status": "not_attempted",
        "prompt_token_count": 0,
        "tokens_generated": 0,
        "output_nonempty": False,
        "output_preview": "",
        "usage": {},
        "duration_s": 0.0,
        "exception": reason,
        "stderr_tail": "",
        "command_hash": "",
        "runtime_receipt_passed": False,
    }


def blocked_reasons_for(
    *,
    runtime_receipt_clean: bool,
    cache_inventory: Mapping[str, Mapping[str, Any]],
    loader_status: Mapping[str, Any],
    gpu_status: Mapping[str, Any],
    per_model_smoke: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return terminal blocker reasons only when the receipt is not clean."""

    if runtime_receipt_clean:
        return []
    cached_ids = [model_id for model_id in MANDATED_MODEL_IDS if cache_inventory[model_id]["cached"]]
    if not cached_ids:
        return ["no mandated SOTA GGUF files available locally"]
    reasons: list[str] = []
    if loader_status.get("llama_cpp_import_ok") is not True:
        reasons.append(f"llama_cpp import failed: {loader_status.get('loader_error') or 'unknown'}")
    torch_cuda = mapping(gpu_status.get("torch_cuda"))
    if torch_cuda.get("cuda_available") is not True:
        reasons.append("selected Python torch.cuda.is_available() is false")
    for model_id in cached_ids:
        row = per_model_smoke.get(model_id, {})
        if row.get("runtime_receipt_passed") is True:
            continue
        detail = str(row.get("exception") or row.get("generation_status") or "runtime smoke failed")
        reasons.append(f"{model_id}: {detail}")
    return reasons or ["no mandated SOTA GGUF completed load/tokenize/generate smoke"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the conductor-facing terminal verdict."""

    if artifact.get("runtime_receipt_clean") is True:
        clean_ids = artifact["smoke_generation_status"]["clean_mandated_model_ids"]
        return (
            "complete: sota_gguf_tokenizer_runtime_receipt_v1_ready=true; "
            "runtime_receipt_clean=true; "
            f"clean_mandated_model_ids={','.join(clean_ids)}"
        )
    detail = "; ".join(str(reason) for reason in artifact.get("blocked_reasons", []))
    return (
        "blocked_runtime_receipt: "
        "sota_gguf_tokenizer_runtime_receipt_v1_ready=true; "
        "runtime_receipt_clean=false; "
        f"detail={detail}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject ambiguous receipts before they are written."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    model_ids = [str(row.get("hf_id")) for row in artifact.get("model_specs", [])]
    if model_ids != list(MANDATED_MODEL_IDS):
        raise ValueError("model_specs must list all mandated model IDs in receipt order")
    if artifact.get("runtime_receipt_clean") is True:
        if artifact.get("blocked_reasons"):
            raise ValueError("clean receipt cannot include terminal blocked_reasons")
        clean_ids = artifact["smoke_generation_status"].get("clean_mandated_model_ids", [])
        if not clean_ids:
            raise ValueError("clean receipt requires a clean mandated model id")
    if artifact.get("runtime_receipt_clean") is not True and not artifact.get("blocked_reasons"):
        raise ValueError("blocked receipt requires blocked_reasons")


def default_cache_roots(root: Path, env: Mapping[str, str]) -> list[Path]:
    """Return local cache roots to inspect without triggering downloads."""

    roots: list[Path] = []
    if env.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(str(env["HUGGINGFACE_HUB_CACHE"])).expanduser())
    if env.get("HF_HOME"):
        roots.append(Path(str(env["HF_HOME"])).expanduser() / "hub")
    roots.extend([Path.home() / ".cache" / "huggingface" / "hub", root / "models"])
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def selected_python_for(root: Path) -> str:
    """Return the project venv Python when present, otherwise this interpreter."""

    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a bounded local command and preserve stdout/stderr."""

    cmd = [str(part) for part in command]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except Exception as exc:
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_nvidia_smi_rows(text: str) -> list[JsonDict]:
    """Parse the nvidia-smi CSV shape used by this receipt."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": safe_int(parts[2]),
                "memory_used_mib": safe_int(parts[3]),
                "memory_free_mib": safe_int(parts[4]),
                "driver_version": parts[5],
            }
        )
    return rows


def quantization_from_name(filename: str) -> str:
    """Extract a known GGUF quantization token from a filename."""

    lower = filename.lower()
    for token in QUANTIZATION_TOKENS:
        if token.lower() in lower:
            return token
    return "unknown"


def first_json_line(text: str) -> JsonDict:
    """Parse the first JSON object from command stdout."""

    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def mapping(value: Any) -> JsonDict:
    """Normalize a JSON value to a dict."""

    return dict(value) if isinstance(value, Mapping) else {}


def safe_int(value: Any) -> int | None:
    """Convert integer-like values without raising on malformed input."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    """Convert float-like values without raising on malformed input."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate_tail(text: str, *, limit: int = 2000) -> str:
    """Keep command stderr compact while preserving the newest evidence."""

    stripped = text.rstrip()
    return stripped if len(stripped) <= limit else stripped[-limit:]


def files_updated() -> list[str]:
    """List files this task owns for artifact provenance."""

    return [
        SPEC_REL_PATH.as_posix(),
        "python/carnot/reporting/sota_gguf_tokenizer_runtime_receipt_3338.py",
        SCRIPT_REL_PATH.as_posix(),
        TEST_REL_PATH.as_posix(),
        OUTPUT_REL_PATH.as_posix(),
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable runtime evidence while excluding volatile duration/verdict text."""

    return stable_hash(
        {
            "cache_status": artifact.get("cache_status"),
            "loader_status": artifact.get("loader_status"),
            "gpu_status": artifact.get("gpu_status"),
            "model_specs": artifact.get("model_specs"),
            "random_seed": artifact.get("random_seed"),
            "runtime_receipt_clean": artifact.get("runtime_receipt_clean"),
            "smoke_generation_status": artifact.get("smoke_generation_status"),
            "tokenizer_status": artifact.get("tokenizer_status"),
        }
    )


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 over JSON-serializable evidence."""

    return hash_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def hash_text(value: str) -> str:
    """Return the SHA-256 hex digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_bytes(value: bytes) -> str:
    """Return the SHA-256 hex digest for bytes."""

    return hashlib.sha256(value).hexdigest()


def main() -> int:
    """CLI entrypoint used by scripts/experiment_3338_*.py."""

    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
