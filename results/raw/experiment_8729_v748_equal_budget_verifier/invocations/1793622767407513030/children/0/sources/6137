"""Build the Exp 3193 llama.cpp CUDA/offload health probe artifact.

Spec refs: REQ-VERIFY-3193, SCENARIO-VERIFY-3193.

This module is a substrate probe, not a verifier benchmark. Its job is to
separate three cases that downstream gates must not conflate: clean CUDA
offload on a mandated SOTA GGUF, CPU-only receipt wiring, and a blocked local
precondition. It never downloads weights and never promotes legacy small-model
smoke evidence into headline fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import os
import re
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.llama_cpp_cuda_offload_health_probe.v1"
EXPERIMENT_ID = "exp3193"
ARTIFACT = "experiment_3193_llama_cpp_cuda_offload_health_probe_v1"

OUTPUT_REL_PATH = Path("results/experiment_3193_llama_cpp_cuda_offload_health_probe_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3193_llama_cpp_cuda_offload_health_probe_v1.py"

DEFAULT_PROMPT = "Exp 3193 CUDA offload health probe. Reply with exactly one word: READY."
DEFAULT_RANDOM_SEED = 20260527
DEFAULT_MAX_TOKENS = 4
DEFAULT_WORKER_TIMEOUT_S = 600
DEFAULT_N_GPU_LAYERS = -1
MIN_GPU_MEMORY_DELTA_MIB = 64

SUBSTRATE_CLASSES = (
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_backend_absent",
    "gpu_offload_unhealthy",
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "description": "Qwen 3.6 35B MoE, approximately 3B active",
        "priority": 1,
        "legacy_smoke_only": False,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "description": "Gemma 4 31B dense instruction model",
        "priority": 2,
        "legacy_smoke_only": False,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "description": "Gemma 4 26B MoE, approximately 4B active",
        "priority": 3,
        "legacy_smoke_only": False,
    },
)

QUANTIZATION_PREFERENCE = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q4_K_S",
    "Q8_0",
    "BF16",
)

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "model_specs",
    "preconditions_checked",
    "cache_inventory",
    "selected_model",
    "selected_model_file_hash",
    "llama_cpp_backend_metadata",
    "n_gpu_layers_requested",
    "n_gpu_layers_effective",
    "gpu_observations",
    "receipt_count",
    "receipts",
    "substrate_classification",
    "clean_rerun_allowed",
    "headline_claim_allowed",
    "blocker_reasons",
    "honest_verdict",
}

WORKER_CODE = r'''
import argparse
import json
import subprocess
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
parser.add_argument("--exp3193-offload-worker", action="store_true")
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-tokens", type=int, required=True)
parser.add_argument("--n-gpu-layers", type=int, required=True)
args = parser.parse_args()

started = time.monotonic()
llm = None
try:
    from llama_cpp import Llama, llama_cpp

    before = _gpu_memory()
    supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())
    load_started = time.monotonic()
    llm = Llama(
        model_path=args.model_path,
        n_ctx=256,
        n_batch=64,
        n_ubatch=64,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )
    load_wall_s = time.monotonic() - load_started
    after_load = _gpu_memory()
    raw = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=args.seed,
    )
    after_generate = _gpu_memory()
    text = _response_text(raw).strip()
    print(
        json.dumps(
            {
                "ok": bool(text),
                "model_id": args.model_id,
                "prompt": args.prompt,
                "response_text": text,
                "usage": raw.get("usage", {}) if isinstance(raw, dict) else {},
                "wall_clock_s": round(time.monotonic() - started, 6),
                "load_wall_clock_s": round(load_wall_s, 6),
                "llama_cpp_supports_gpu_offload": supports_gpu,
                "n_gpu_layers_requested": args.n_gpu_layers,
                "n_gpu_layers_effective": None,
                "gpu_memory": {
                    "before": before,
                    "after_load": after_load,
                    "after_generate": after_generate,
                },
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
                "prompt": args.prompt,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_clock_s": round(time.monotonic() - started, 6),
                "n_gpu_layers_requested": args.n_gpu_layers,
                "n_gpu_layers_effective": None,
                "gpu_memory": {},
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
    root: str | Path = REPO_ROOT,
    *,
    cache_root: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = None,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
    n_gpu_layers_requested: int = DEFAULT_N_GPU_LAYERS,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> JsonDict:
    """REQ-VERIFY-3193: build a CUDA/offload receipt or precise blocked artifact."""

    start = monotonic()
    root_path = Path(root)
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    runner = command_runner or run_command
    python_exe = str(selected_python) if selected_python is not None else selected_python_for(root_path)
    hf_cache = Path(cache_root) if cache_root is not None else default_hf_cache_root(merged_env)

    nvidia = probe_nvidia_smi(runner)
    torch_cuda = probe_torch_cuda(python_exe, runner)
    llama_meta = probe_llama_cpp_backend(python_exe, runner, merged_env)
    cache_inventory = inspect_cache(hf_cache)
    selected = select_model(cache_inventory)
    selected_hash = sha256_file(Path(str(selected["path"]))) if selected else None

    can_attempt = bool(
        selected
        and llama_meta.get("llama_cpp_import_ok") is True
        and llama_meta.get("llama_cpp_supports_gpu_offload") is True
        and nvidia.get("available") is True
        and torch_cuda.get("cuda_available") is True
        and safe_int(torch_cuda.get("device_count")) not in (None, 0)
    )

    worker = (
        run_receipt_worker(
            selected_python=python_exe,
            selected_model=selected,
            n_gpu_layers_requested=int(n_gpu_layers_requested),
            command_runner=runner,
            timeout_s=int(worker_timeout_s),
        )
        if can_attempt
        else empty_worker()
    )
    receipts = receipt_from_worker(
        selected_model=selected,
        selected_model_hash=selected_hash,
        command_hash=str(worker.get("command_hash") or ""),
        worker_payload=mapping(worker.get("payload")),
        worker_returncode=safe_int(worker.get("returncode")),
        stderr_tail=str(worker.get("stderr_tail") or ""),
    )
    gpu_observations = build_gpu_observations(nvidia, mapping(worker.get("payload")), receipts)
    n_gpu_effective = infer_effective_gpu_layers(
        payload=mapping(worker.get("payload")),
        stderr_tail=str(worker.get("stderr_tail") or ""),
        offload_evidenced=bool(gpu_observations["offload_evidenced"]),
        requested=int(n_gpu_layers_requested),
    )
    substrate = classify_substrate(
        selected_model=selected,
        llama_meta=llama_meta,
        nvidia=nvidia,
        torch_cuda=torch_cuda,
        worker=worker,
        receipts=receipts,
        offload_evidenced=bool(gpu_observations["offload_evidenced"]),
    )
    blockers = blocker_reasons(
        substrate_classification=substrate,
        selected_model=selected,
        llama_meta=llama_meta,
        nvidia=nvidia,
        torch_cuda=torch_cuda,
        worker=worker,
        receipts=receipts,
    )
    clean = substrate == "full_local_sota_receipt"
    finished = monotonic()
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "model_specs": [dict(row) for row in MANDATED_MODEL_SPECS],
        "preconditions_checked": {
            "nvidia_smi": nvidia,
            "torch_cuda": torch_cuda,
            "llama_cpp_backend": llama_meta,
            "mandated_cache": {
                "cache_root": str(hf_cache),
                "families_checked": [row["hf_id"] for row in cache_inventory],
                "available_family_count": sum(
                    1 for row in cache_inventory if row["cache_status"] == "resolved"
                ),
                "downloads_performed": False,
            },
            "protected_files": {
                "scripts/research_conductor.py": "not_modified_by_probe",
                "ops/status.md": "left_to_conductor_reconciler",
                "ops/changelog.md": "left_to_conductor_reconciler",
            },
        },
        "cache_inventory": cache_inventory,
        "selected_model": str(selected["hf_id"]) if selected else "",
        "selected_model_path": str(selected["path"]) if selected else None,
        "selected_model_file_hash": selected_hash,
        "llama_cpp_backend_metadata": llama_meta,
        "n_gpu_layers_requested": int(n_gpu_layers_requested) if selected else None,
        "n_gpu_layers_effective": n_gpu_effective,
        "gpu_observations": gpu_observations,
        "receipt_count": len(receipts),
        "receipts": receipts,
        "substrate_classification": substrate,
        "clean_rerun_allowed": clean,
        "headline_claim_allowed": clean,
        "blocker_reasons": blockers,
        "tests_run": list(tests_run or default_tests_run()),
        "worker_attempt": {
            "attempted": bool(worker.get("attempted")),
            "returncode": worker.get("returncode"),
            "stderr_tail": worker.get("stderr_tail"),
            "command_hash": worker.get("command_hash"),
        },
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: str | Path = REPO_ROOT,
    *,
    output_path: str | Path = OUTPUT_REL_PATH,
    cache_root: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = None,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
    n_gpu_layers_requested: int = DEFAULT_N_GPU_LAYERS,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> Path:
    """Build and persist the Exp 3193 terminal artifact."""

    root_path = Path(root)
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root_path / destination
    artifact = build_artifact(
        root_path,
        cache_root=cache_root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        monotonic=monotonic,
        tests_run=tests_run,
        n_gpu_layers_requested=n_gpu_layers_requested,
        worker_timeout_s=worker_timeout_s,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def selected_python_for(root: Path) -> str:
    """Return the project venv Python when present, otherwise the current interpreter."""

    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def default_hf_cache_root(env: Mapping[str, str]) -> Path:
    """Resolve the HuggingFace hub cache root without causing downloads."""

    if env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(str(env["HUGGINGFACE_HUB_CACHE"])).expanduser()
    if env.get("HF_HOME"):
        return Path(str(env["HF_HOME"])).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a bounded command and preserve stdout/stderr for the artifact."""

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
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def probe_nvidia_smi(command_runner: CommandRunner) -> JsonDict:
    """Record NVIDIA device visibility and memory before any model load."""

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
        "command": result.get("command", command),
    }


def parse_nvidia_smi_rows(text: str) -> list[JsonDict]:
    """Parse the CSV shape requested by probe_nvidia_smi."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total, used, free, driver = parts
        if not index.isdigit():
            continue
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": safe_int(total),
                "memory_used_mib": safe_int(used),
                "memory_free_mib": safe_int(free),
                "driver_version": driver,
            }
        )
    return rows


def probe_torch_cuda(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Probe torch CUDA through the same interpreter used for llama.cpp."""

    code = (
        "import importlib.util, json\n"
        "print('exp3193_torch_cuda_probe')\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    print(json.dumps({'torch_present': False, 'torch_import_ok': False, "
        "'cuda_available': False, 'device_count': 0}))\n"
        "else:\n"
        "    try:\n"
        "        import torch\n"
        "        print(json.dumps({'torch_present': True, 'torch_import_ok': True, "
        "'torch_version': getattr(torch, '__version__', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count()), "
        "'cuda_version': getattr(torch.version, 'cuda', None)}))\n"
        "    except Exception as exc:\n"
        "        print(json.dumps({'torch_present': True, 'torch_import_ok': False, "
        "'cuda_available': False, 'device_count': 0, "
        "'error': f'{type(exc).__name__}: {exc}'}))\n"
        "        raise SystemExit(1)\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30)
    payload = first_json_line(str(result.get("stdout") or ""))
    return {
        "torch_present": payload.get("torch_present") is True,
        "torch_import_ok": payload.get("torch_import_ok") is True,
        "torch_version": payload.get("torch_version"),
        "cuda_available": result.get("returncode") == 0 and payload.get("cuda_available") is True,
        "device_count": safe_int(payload.get("device_count")) or 0,
        "cuda_version": payload.get("cuda_version"),
        "returncode": result.get("returncode"),
        "error": str(payload.get("error") or ""),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_llama_cpp_backend(
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Import llama.cpp and ask the backend whether GPU offload is compiled in."""

    code = (
        "import importlib.util, json\n"
        "print('exp3193_llama_cpp_backend_probe')\n"
        "payload = {'llama_cpp_import_ok': False, 'loader_name': 'llama_cpp.Llama'}\n"
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
        "        'backend_error': '',\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload['backend_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30, env=dict(env))
    payload = first_json_line(str(result.get("stdout") or ""))
    import_ok = result.get("returncode") == 0 and payload.get("llama_cpp_import_ok") is True
    payload.setdefault("loader_name", "llama_cpp.Llama")
    payload["llama_cpp_import_ok"] = import_ok
    payload["llama_cpp_supports_gpu_offload"] = (
        import_ok and payload.get("llama_cpp_supports_gpu_offload") is True
    )
    payload["returncode"] = result.get("returncode")
    payload["stderr_tail"] = truncate_tail(str(result.get("stderr") or ""))
    return payload


def inspect_cache(cache_root: Path) -> list[JsonDict]:
    """Inventory local quant files for every mandated GGUF family."""

    rows: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        files = candidate_files(cache_root, str(spec["hf_id"]))
        rows.append(
            {
                "hf_id": spec["hf_id"],
                "role": spec["role"],
                "priority": spec["priority"],
                "cache_status": "resolved" if files else "missing",
                "available_quant_files": [candidate_record(path) for path in files],
                "selected_for_probe": False,
            }
        )
    selected = select_model(rows)
    selected_path = str(selected["path"]) if selected else None
    for row in rows:
        row["selected_for_probe"] = bool(
            selected_path
            and any(item["path"] == selected_path for item in row["available_quant_files"])
        )
    return rows


def candidate_files(cache_root: Path, hf_id: str) -> list[Path]:
    """Return non-empty GGUF candidates from the local HuggingFace cache only."""

    owner, name = hf_id.split("/", 1)
    snapshots = cache_root / f"models--{owner}--{name}" / "snapshots"
    if not snapshots.is_dir():
        return []
    paths = [
        path
        for path in snapshots.rglob("*.gguf")
        if path.is_file() and path.stat().st_size > 0 and "mmproj" not in path.name.lower()
    ]
    return sorted(paths, key=candidate_sort_key)


def candidate_sort_key(path: Path) -> tuple[int, str]:
    """Prefer the quantizations used by the local SOTA policy."""

    filename = path.name.lower()
    for index, token in enumerate(QUANTIZATION_PREFERENCE):
        if token.lower() in filename:
            return (index, str(path))
    return (len(QUANTIZATION_PREFERENCE), str(path))


def candidate_record(path: Path) -> JsonDict:
    """Represent one local GGUF candidate without hashing every large file."""

    stat = path.stat()
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "quantization": quantization_from_name(path.name),
    }


def quantization_from_name(filename: str) -> str:
    """Extract the visible quantization token from a GGUF filename."""

    lower = filename.lower()
    for token in QUANTIZATION_PREFERENCE:
        if token.lower() in lower:
            return token
    return "unknown"


def select_model(cache_inventory: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Pick the strongest available mandated model by policy priority."""

    for family in cache_inventory:
        files = family.get("available_quant_files")
        if isinstance(files, list) and files:
            first = mapping(files[0])
            if first.get("path"):
                return {
                    "hf_id": family.get("hf_id"),
                    "path": first["path"],
                    "quantization": first.get("quantization"),
                    "size_bytes": first.get("size_bytes"),
                }
    return None


def run_receipt_worker(
    *,
    selected_python: str,
    selected_model: Mapping[str, Any] | None,
    n_gpu_layers_requested: int,
    command_runner: CommandRunner,
    timeout_s: int,
) -> JsonDict:
    """Run one deterministic llama.cpp prompt after preconditions pass."""

    if selected_model is None:
        return empty_worker()
    command = [
        selected_python,
        "-c",
        WORKER_CODE,
        "--exp3193-offload-worker",
        "--model-id",
        str(selected_model["hf_id"]),
        "--model-path",
        str(selected_model["path"]),
        "--prompt",
        DEFAULT_PROMPT,
        "--seed",
        str(DEFAULT_RANDOM_SEED),
        "--max-tokens",
        str(DEFAULT_MAX_TOKENS),
        "--n-gpu-layers",
        str(int(n_gpu_layers_requested)),
    ]
    result = command_runner(command, timeout_s=timeout_s)
    stderr_tail = truncate_tail(str(result.get("stderr") or ""))
    payload = first_json_line(str(result.get("stdout") or ""))
    if stderr_tail and not payload.get("backend_log_tail"):
        payload["backend_log_tail"] = stderr_tail
    return {
        "attempted": True,
        "returncode": result.get("returncode"),
        "payload": payload,
        "stderr_tail": stderr_tail,
        "command_hash": stable_hash(command),
    }


def empty_worker() -> JsonDict:
    """Return the no-attempt worker shape used for blocked preconditions."""

    return {
        "attempted": False,
        "returncode": None,
        "payload": {},
        "stderr_tail": "",
        "command_hash": "",
    }


def receipt_from_worker(
    *,
    selected_model: Mapping[str, Any] | None,
    selected_model_hash: str | None,
    command_hash: str,
    worker_payload: Mapping[str, Any],
    worker_returncode: int | None,
    stderr_tail: str,
) -> list[JsonDict]:
    """Convert a successful worker payload into one proof receipt."""

    if selected_model is None or worker_payload.get("ok") is not True:
        return []
    response = str(worker_payload.get("response_text") or "").strip()
    if not response:
        return []
    prompt = str(worker_payload.get("prompt") or DEFAULT_PROMPT)
    prompt_hash = hash_text(prompt)
    response_hash = hash_text(response)
    token_counts = token_counts_for(prompt, response, mapping(worker_payload.get("usage")))
    effective = safe_int(worker_payload.get("n_gpu_layers_effective"))
    backend_tail = truncate_tail(
        "\n".join(
            text
            for text in (str(worker_payload.get("backend_log_tail") or ""), stderr_tail)
            if text
        )
    )
    return [
        {
            "selected_model": str(selected_model["hf_id"]),
            "model_path": str(selected_model["path"]),
            "model_file_hash": selected_model_hash,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "transcript_hash": transcript_hash(
                str(selected_model["hf_id"]), prompt_hash, response_hash, DEFAULT_RANDOM_SEED
            ),
            "token_counts": token_counts,
            "random_seed": DEFAULT_RANDOM_SEED,
            "wall_clock_s": safe_float(worker_payload.get("wall_clock_s")),
            "command_hash": command_hash,
            "subprocess_return_code": worker_returncode,
            "stderr_backend_tail": backend_tail,
            "n_gpu_layers_requested": safe_int(worker_payload.get("n_gpu_layers_requested")),
            "n_gpu_layers_effective": effective,
        }
    ]


def token_counts_for(prompt: str, response: str, usage: Mapping[str, Any]) -> JsonDict:
    """Prefer llama.cpp usage counters and fall back to a transparent estimate."""

    prompt_tokens = safe_int(usage.get("prompt_tokens"))
    completion_tokens = safe_int(usage.get("completion_tokens"))
    total_tokens = safe_int(usage.get("total_tokens"))
    if prompt_tokens is not None and completion_tokens is not None and total_tokens is not None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source": "llama_cpp_usage",
        }
    prompt_estimate = len(prompt.split())
    completion_estimate = len(response.split())
    return {
        "prompt_tokens": prompt_estimate,
        "completion_tokens": completion_estimate,
        "total_tokens": prompt_estimate + completion_estimate,
        "source": "whitespace_estimate",
    }


def build_gpu_observations(
    nvidia_probe: Mapping[str, Any],
    worker_payload: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize GPU memory and backend evidence from preflight plus worker."""

    gpu_memory = mapping(worker_payload.get("gpu_memory"))
    backend_tail = "\n".join(
        str(row.get("stderr_backend_tail") or "") for row in receipts if row.get("stderr_backend_tail")
    )
    offloaded_layers = parse_offloaded_layers(backend_tail)
    memory_delta = max_gpu_memory_delta(gpu_memory)
    return {
        "nvidia_smi_available": nvidia_probe.get("available") is True,
        "initial_gpus": nvidia_probe.get("gpus", []),
        "worker_gpu_memory": gpu_memory,
        "offloaded_layer_count_from_backend_log": offloaded_layers,
        "max_memory_delta_mib": memory_delta,
        "offload_evidenced": bool(
            offloaded_layers and offloaded_layers > 0 or memory_delta >= MIN_GPU_MEMORY_DELTA_MIB
        ),
    }


def max_gpu_memory_delta(gpu_memory: Mapping[str, Any]) -> int:
    """Return the largest observed memory increase over the worker baseline."""

    before = memory_by_index(gpu_memory.get("before"))
    deltas: list[int] = []
    for key in ("after_load", "after_generate"):
        after = memory_by_index(gpu_memory.get(key))
        for index, used in after.items():
            deltas.append(max(0, used - before.get(index, used)))
    return max(deltas) if deltas else 0


def memory_by_index(rows: Any) -> dict[int, int]:
    """Normalize worker nvidia-smi rows into index to used-memory mappings."""

    if not isinstance(rows, list):
        return {}
    out: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = safe_int(row.get("index"))
        used = safe_int(row.get("memory_used_mib"))
        if index is not None and used is not None:
            out[index] = used
    return out


def parse_offloaded_layers(text: str) -> int | None:
    """Extract layer-offload evidence from llama.cpp backend logs."""

    patterns = (
        r"offloaded\s+(\d+)\s*/\s*\d+\s+layers?\s+to\s+GPU",
        r"offloading\s+(\d+)\s+repeating\s+layers?\s+to\s+GPU",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_effective_gpu_layers(
    *,
    payload: Mapping[str, Any],
    stderr_tail: str,
    offload_evidenced: bool,
    requested: int,
) -> int | None:
    """Map worker payload and backend logs to the effective offload count."""

    explicit = safe_int(payload.get("n_gpu_layers_effective"))
    if explicit is not None:
        return explicit
    parsed = parse_offloaded_layers(str(payload.get("backend_log_tail") or "") + "\n" + stderr_tail)
    if parsed is not None:
        return parsed
    if offload_evidenced:
        return requested
    return None


def classify_substrate(
    *,
    selected_model: Mapping[str, Any] | None,
    llama_meta: Mapping[str, Any],
    nvidia: Mapping[str, Any],
    torch_cuda: Mapping[str, Any],
    worker: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    offload_evidenced: bool,
) -> str:
    """Classify the terminal substrate without hiding CPU fallback."""

    if selected_model is None:
        return "model_cache_missing"
    if llama_meta.get("llama_cpp_import_ok") is not True:
        return "loader_missing"
    if nvidia.get("available") is not True or torch_cuda.get("cuda_available") is not True:
        return "cuda_unavailable"
    if llama_meta.get("llama_cpp_supports_gpu_offload") is not True:
        return "cuda_backend_absent"
    if worker.get("attempted") and worker.get("returncode") != 0:
        return "gpu_offload_unhealthy"
    if receipts and offload_evidenced:
        return "full_local_sota_receipt"
    if receipts:
        return "cpu_fallback_receipt_only"
    return "gpu_offload_unhealthy"


def blocker_reasons(
    *,
    substrate_classification: str,
    selected_model: Mapping[str, Any] | None,
    llama_meta: Mapping[str, Any],
    nvidia: Mapping[str, Any],
    torch_cuda: Mapping[str, Any],
    worker: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return the exact precondition or runtime blockers for non-clean artifacts."""

    if substrate_classification == "full_local_sota_receipt":
        return []
    if selected_model is None:
        return ["no mandated local SOTA GGUF candidate found in HuggingFace cache"]
    if substrate_classification == "loader_missing":
        return [f"llama_cpp import failed: {llama_meta.get('backend_error') or 'unknown error'}"]
    if substrate_classification == "cuda_unavailable":
        reasons = []
        if nvidia.get("available") is not True:
            reasons.append("nvidia-smi did not report a visible NVIDIA GPU")
        if torch_cuda.get("cuda_available") is not True:
            reasons.append("selected Python torch.cuda.is_available() is false")
        if llama_meta.get("llama_cpp_supports_gpu_offload") is not True:
            detail = str(llama_meta.get("stderr_tail") or llama_meta.get("backend_error") or "").strip()
            suffix = f": {detail}" if detail else ""
            reasons.append(f"llama_cpp backend did not report GPU offload support{suffix}")
        return reasons or ["CUDA unavailable"]
    if substrate_classification == "cuda_backend_absent":
        return ["llama_cpp backend does not report GPU offload support"]
    if substrate_classification == "cpu_fallback_receipt_only":
        return ["receipt completed without CUDA/offload evidence"]
    payload = mapping(worker.get("payload"))
    detail = str(payload.get("error") or worker.get("stderr_tail") or "")
    if detail:
        return [truncate_tail(detail)]
    if not receipts:
        return ["worker did not produce a usable receipt"]
    return ["GPU offload receipt unhealthy"]


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict string required by the conductor."""

    substrate = str(artifact.get("substrate_classification") or "")
    if substrate == "full_local_sota_receipt":
        return (
            "complete: llama_cpp_cuda_offload_health_probe_v1_ready=true; "
            "substrate_classification=full_local_sota_receipt; "
            f"receipt_count={artifact.get('receipt_count')}; clean_rerun_allowed=true"
        )
    reason = "; ".join(str(item) for item in artifact.get("blocker_reasons", [])) or substrate
    return (
        f"blocked_{substrate}: "
        "llama_cpp_cuda_offload_health_probe_v1_ready=true; "
        "clean_rerun_allowed=false; "
        f"detail={reason}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject ambiguous artifacts that could unlock downstream gates incorrectly."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    substrate = artifact.get("substrate_classification")
    if substrate not in SUBSTRATE_CLASSES:
        raise ValueError("substrate_classification is not recognized")
    if artifact.get("clean_rerun_allowed") is True and substrate != "full_local_sota_receipt":
        raise ValueError("clean rerun requires full_local_sota_receipt")
    if artifact.get("headline_claim_allowed") is True and artifact.get("clean_rerun_allowed") is not True:
        raise ValueError("headline claim requires clean rerun eligibility")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if substrate != "full_local_sota_receipt" and not verdict.startswith("blocked_"):
        raise ValueError("non-full substrates must use a blocked_ honest_verdict")


def sha256_file(path: Path) -> str | None:
    """Return a full SHA-256 hash for the selected model file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript_hash(model_id: str, prompt_hash: str, response_hash: str, seed: int) -> str:
    """Hash transcript identity fields without storing the full response twice."""

    return stable_hash(
        {
            "model_id": model_id,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "seed": int(seed),
        }
    )


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 over JSON-serializable evidence."""

    return hash_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def hash_text(value: str) -> str:
    """Return the UTF-8 SHA-256 digest for artifact hashes."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_json_line(text: str) -> JsonDict:
    """Parse the first JSON object emitted by a probe or worker."""

    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def mapping(value: Any) -> JsonDict:
    """Normalize a JSON value into a dictionary."""

    return dict(value) if isinstance(value, Mapping) else {}


def safe_int(value: Any) -> int | None:
    """Convert JSON counters without raising on missing or malformed data."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    """Convert JSON durations without raising on missing or malformed data."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate_tail(text: str, *, limit: int = 2000) -> str:
    """Keep stderr/backend evidence compact while preserving the newest lines."""

    compact = text.rstrip()
    return compact if len(compact) <= limit else compact[-limit:]


def duration(started_s: float, finished_s: float) -> float:
    """Return a non-negative rounded wall-clock duration."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def default_tests_run() -> list[str]:
    """List the verification commands expected for this artifact."""

    return [
        ".venv/bin/pytest tests/python/test_experiment_3193_llama_cpp_cuda_offload_health_probe_v1.py -q -o addopts=''",
        ".venv/bin/coverage erase",
        ".venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3193_llama_cpp_cuda_offload_health_probe_v1.py -q",
        ".venv/bin/coverage report --include='python/carnot/verify/llama_cpp_cuda_offload_health_probe_v1.py' --fail-under=100 --show-missing",
        ".venv/bin/python scripts/check_spec_coverage.py",
        ".venv/bin/pytest tests/python -q",
    ]
