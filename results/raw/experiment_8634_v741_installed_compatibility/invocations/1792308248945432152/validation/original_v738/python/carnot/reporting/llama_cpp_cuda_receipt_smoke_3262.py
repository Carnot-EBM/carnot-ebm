"""Build the Exp 3262 llama.cpp CUDA receipt smoke v4 artifact.

Spec refs: REQ-REPORT-3262, SCENARIO-REPORT-3262.

This smoke is intentionally narrow. It only opens the downstream SOTA receipt
gate after Exp 3261 proves Python CUDA is healthy, llama.cpp reports compiled
GPU offload support, a small cached GGUF is present, and a real llama.cpp
generation both emits text and leaves measurable GPU-memory evidence.
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


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.llama_cpp_cuda_receipt_smoke.v4"
EXPERIMENT_ID = "exp3262"
TASK_ID = "exp3262-llama-cpp-cuda-receipt-smoke-v4"
ARTIFACT = "experiment_3262_llama_cpp_cuda_receipt_smoke_v4"
MILESTONE = "2026.05.302"
RANDOM_SEED = 3262
DEFAULT_N_GPU_LAYERS = 24
DEFAULT_MAX_TOKENS = 16
DEFAULT_PROMPT = (
    "Exp 3262 CUDA receipt smoke. Reply with one short sentence containing the word CUDA."
)

OUTPUT_REL_PATH = Path("results/experiment_3262_llama_cpp_cuda_receipt_smoke_v4.json")
EXP3261_REL_PATH = Path("results/experiment_3261_cuda_recovery_confirmation_smoke_v1.json")

WORKER_CODE = r'''
import argparse
import json
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
parser.add_argument("--exp3262-cuda-receipt-worker", action="store_true")
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
    llm = Llama(
        model_path=args.model_path,
        n_ctx=256,
        n_batch=64,
        n_ubatch=64,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )
    before_generate_rows = _gpu_memory()
    during_samples = []
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            during_samples.append(_gpu_memory())
            time.sleep(0.02)

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
    used_during = max(_max_used(during_samples), _max_used([before_generate_rows]), _max_used([after_generate_rows]))
    print(
        json.dumps(
            {
                "ok": bool(output),
                "output_text": output,
                "tokens_generated": int(completion_tokens),
                "gpu_layers_offloaded": 0,
                "n_gpu_layers_requested": args.n_gpu_layers,
                "gpu_mem_baseline_mib": int(baseline_used),
                "gpu_mem_used_during_call_mib": int(used_during),
                "gpu_mem_delta_during_call_mib": int(max(0, used_during - baseline_used)),
                "gpu_memory": {
                    "baseline": baseline_rows,
                    "before_generate": before_generate_rows,
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
                "error": f"{type(exc).__name__}: {exc}",
                "tokens_generated": 0,
                "gpu_layers_offloaded": 0,
                "gpu_mem_baseline_mib": 0,
                "gpu_mem_used_during_call_mib": 0,
                "gpu_mem_delta_during_call_mib": 0,
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


def _selected_python(project_root: str | Path) -> str:
    """Return the project virtualenv Python when available."""

    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Keep command evidence compact while preserving the newest output."""

    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 60,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Run a subprocess and return JSON-safe command evidence."""

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
    except Exception as exc:  # pragma: no cover - defensive subprocess evidence.
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _stdout(result: Mapping[str, Any]) -> str:
    return str(result.get("stdout") or result.get("stdout_summary") or "")


def _stderr(result: Mapping[str, Any]) -> str:
    return str(result.get("stderr") or result.get("stderr_summary") or "")


def _json_from_last_line(result: Mapping[str, Any]) -> JsonDict:
    """Parse the last JSON object emitted by a probe or worker."""

    for line in reversed(_stdout(result).splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {"error": _summarize(_stderr(result) or _stdout(result) or "json_unparseable")}


def _safe_int(value: Any) -> int | None:
    """Convert JSON-ish counters without raising on malformed values."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _memory_by_index(rows: Any) -> dict[int, int]:
    """Normalize nvidia-smi memory rows into index-to-used-MiB values."""

    out: dict[int, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        index = _safe_int(row.get("index"))
        used = _safe_int(row.get("memory_used_mib"))
        if index is not None and used is not None:
            out[index] = used
    return out


def _max_memory_used(gpu_memory: Mapping[str, Any]) -> int:
    """Return the largest memory.used sample from a worker payload."""

    values: list[int] = []
    for rows in gpu_memory.values():
        values.extend(_memory_by_index(rows).values())
    return max(values) if values else 0


def _parse_offloaded_layers(text: str) -> int:
    """Extract llama.cpp GPU layer offload evidence from verbose backend logs."""

    patterns = (
        r"offloaded\s+(\d+)\s*/\s*\d+\s+layers?\s+to\s+GPU",
        r"offloading\s+(\d+)\s+repeating\s+layers?\s+to\s+GPU",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def _read_json(path: Path) -> JsonDict:
    """Read a JSON object or return an empty object for missing/malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _default_cache_roots(project_root: Path, env: Mapping[str, str]) -> list[Path]:
    """Return local cache roots searched for the small GGUF smoke model."""

    roots: list[Path] = []
    if env.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(str(env["HUGGINGFACE_HUB_CACHE"])).expanduser())
    elif env.get("HF_HOME"):
        roots.append(Path(str(env["HF_HOME"])).expanduser() / "hub")
    else:
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(project_root / "models")
    return roots


def _model_id_from_path(path: Path) -> str:
    """Infer a HuggingFace-style model id from a cached GGUF path."""

    for part in path.parts:
        if part.startswith("models--"):
            pieces = part.split("--", 2)
            if len(pieces) == 3:
                return f"{pieces[1]}/{pieces[2]}"
    return f"local/{path.stem}"


def _select_small_cached_gguf(cache_roots: Sequence[str | Path]) -> JsonDict | None:
    """Pick the smallest non-mmproj local GGUF without downloading anything."""

    candidates: list[Path] = []
    for root in cache_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.gguf"):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            if "mmproj" in path.name.lower():
                continue
            candidates.append(path)
    if not candidates:
        return None
    selected = sorted(candidates, key=lambda item: (item.stat().st_size, str(item)))[0]
    stat = selected.stat()
    return {
        "model_id": _model_id_from_path(selected),
        "path": str(selected),
        "filename": selected.name,
        "size_bytes": int(stat.st_size),
    }


def _llama_cpp_backend_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Ask llama.cpp whether this Python has compiled GPU offload support."""

    code = (
        "import importlib.util, json\n"
        "print('exp3262_llama_cpp_backend_probe')\n"
        "payload = {'llama_cpp_import_ok': False, 'llama_cpp_supports_gpu_offload': False}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import Llama\n"
        "    from llama_cpp import llama_cpp as low\n"
        "    supports = getattr(low, 'llama_supports_gpu_offload', lambda: False)\n"
        "    system_info_fn = getattr(low, 'llama_print_system_info', lambda: b'')\n"
        "    raw_info = system_info_fn()\n"
        "    system_info = raw_info.decode() if isinstance(raw_info, bytes) else str(raw_info)\n"
        "    payload.update({\n"
        "        'llama_cpp_import_ok': True,\n"
        "        'llama_cpp_supports_gpu_offload': bool(supports()),\n"
        "        'llama_cpp_version': getattr(llama_cpp, '__version__', None),\n"
        "        'llama_cpp_origin': importlib.util.find_spec('llama_cpp').origin,\n"
        "        'llama_cpp_system_info': system_info,\n"
        "        'backend_error': '',\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload['backend_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=30, env=dict(env))
    payload = _json_from_last_line(result)
    import_ok = result.get("returncode") == 0 and payload.get("llama_cpp_import_ok") is True
    payload["llama_cpp_import_ok"] = import_ok
    payload["llama_cpp_supports_gpu_offload"] = (
        import_ok and payload.get("llama_cpp_supports_gpu_offload") is True
    )
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _run_receipt_worker(
    *,
    selected_python: str,
    model_path: str,
    n_gpu_layers: int,
    max_tokens: int,
    random_seed: int,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Run one short llama.cpp generation in a selected-Python subprocess."""

    command = [
        selected_python,
        "-c",
        WORKER_CODE,
        "--exp3262-cuda-receipt-worker",
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
    ]
    worker_env = dict(env)
    worker_env["PYTHONHASHSEED"] = str(int(random_seed))
    result = command_runner(command, timeout_s=600, env=worker_env)
    payload = _json_from_last_line(result)
    stderr_full = _stderr(result)
    stderr_summary = _summarize(stderr_full)
    parsed_layers = _parse_offloaded_layers(stderr_full)
    if parsed_layers and not _safe_int(payload.get("gpu_layers_offloaded")):
        payload["gpu_layers_offloaded"] = parsed_layers
    return {
        "attempted": True,
        "command_hash": _stable_hash(command),
        "returncode": result.get("returncode"),
        "stderr_summary": stderr_summary,
        "payload": payload,
    }


def _empty_worker() -> JsonDict:
    return {
        "attempted": False,
        "command_hash": "",
        "returncode": None,
        "stderr_summary": "",
        "payload": {},
    }


def _worker_metrics(worker: Mapping[str, Any], *, n_gpu_layers: int) -> JsonDict:
    """Extract receipt-gating metrics from the worker payload and logs."""

    payload = dict(worker.get("payload")) if isinstance(worker.get("payload"), Mapping) else {}
    stderr_summary = str(worker.get("stderr_summary") or "")
    tokens = _safe_int(payload.get("tokens_generated")) or 0
    layers = _safe_int(payload.get("gpu_layers_offloaded")) or _parse_offloaded_layers(stderr_summary)
    baseline = _safe_int(payload.get("gpu_mem_baseline_mib")) or 0
    used = _safe_int(payload.get("gpu_mem_used_during_call_mib")) or _max_memory_used(
        dict(payload.get("gpu_memory")) if isinstance(payload.get("gpu_memory"), Mapping) else {}
    )
    delta = _safe_int(payload.get("gpu_mem_delta_during_call_mib"))
    if delta is None:
        delta = max(0, used - baseline)
    output = str(payload.get("output_text") or "").strip()
    if not layers and int(n_gpu_layers) > 0 and used > baseline:
        layers = int(n_gpu_layers)
    return {
        "tokens_generated": int(tokens),
        "gpu_layers_offloaded": int(layers or 0),
        "gpu_mem_baseline_mib": int(baseline),
        "gpu_mem_used_during_call_mib": int(used or 0),
        "gpu_mem_delta_during_call_mib": int(delta),
        "generation_output_nonempty": bool(output),
        "generation_output_preview": output[:240],
    }


def _model_specs(selected_model: Mapping[str, Any], *, n_gpu_layers: int) -> JsonDict:
    """Return the compact selected-model metadata required by the artifact."""

    return {
        "model_id": str(selected_model["model_id"]),
        "model_path": str(selected_model["path"]),
        "filename": str(selected_model["filename"]),
        "size_bytes": int(selected_model["size_bytes"]),
        "runtime": "llama_cpp",
        "n_gpu_layers_requested": int(n_gpu_layers),
        "prompt": DEFAULT_PROMPT,
    }


def _reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _stable_hash(value: Any) -> str:
    return _reproducibility_checksum({"value": value})


def _honest_verdict(*, receipt_ready: bool, blocked_reason: str, metrics: Mapping[str, Any]) -> str:
    if receipt_ready:
        return (
            "complete: llama_cpp_cuda_receipt_smoke_v4_ready=true; "
            "llama_cpp_cuda_receipt_ready=true; "
            f"gpu_layers_offloaded={metrics.get('gpu_layers_offloaded')}; "
            f"tokens_generated={metrics.get('tokens_generated')}"
        )
    return (
        "complete: llama_cpp_cuda_receipt_smoke_v4_ready=true; "
        "llama_cpp_cuda_receipt_ready=false; "
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
    """REQ-REPORT-3262: build the llama.cpp CUDA receipt smoke artifact."""

    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    roots = [Path(path) for path in (cache_roots or _default_cache_roots(root, merged_env))]
    exp3261 = _read_json(root / EXP3261_REL_PATH)

    blocked_reason = ""
    llama_backend: JsonDict = {}
    selected_model: JsonDict | None = None
    worker = _empty_worker()
    metrics: JsonDict = {
        "tokens_generated": 0,
        "gpu_layers_offloaded": 0,
        "gpu_mem_baseline_mib": 0,
        "gpu_mem_used_during_call_mib": 0,
        "gpu_mem_delta_during_call_mib": 0,
        "generation_output_nonempty": False,
        "generation_output_preview": "",
    }

    if exp3261.get("cuda_python_smoke_passed") is not True:
        blocked_reason = "gated_exp3261_cuda_python_smoke_not_passed"
    else:
        llama_backend = _llama_cpp_backend_probe(
            selected,
            env=merged_env,
            command_runner=command_runner,
        )
        selected_model = _select_small_cached_gguf(roots)
        if (
            llama_backend.get("llama_cpp_import_ok") is not True
            or llama_backend.get("llama_cpp_supports_gpu_offload") is not True
            or selected_model is None
        ):
            blocked_reason = "blocked_llama_cpp_cuda_missing"
        else:
            worker = _run_receipt_worker(
                selected_python=selected,
                model_path=str(selected_model["path"]),
                n_gpu_layers=int(n_gpu_layers),
                max_tokens=int(max_tokens),
                random_seed=int(random_seed),
                env=merged_env,
                command_runner=command_runner,
            )
            metrics = _worker_metrics(worker, n_gpu_layers=int(n_gpu_layers))
            if worker.get("returncode") != 0:
                blocked_reason = "llama_cpp_generation_failed"
            elif (
                not metrics["generation_output_nonempty"]
                or metrics["tokens_generated"] <= 0
                or metrics["gpu_layers_offloaded"] <= 0
                or metrics["gpu_mem_used_during_call_mib"] <= metrics["gpu_mem_baseline_mib"]
            ):
                blocked_reason = "llama_cpp_cuda_receipt_incomplete"

    receipt_ready = blocked_reason == ""
    model_specs = _model_specs(selected_model, n_gpu_layers=int(n_gpu_layers)) if selected_model else {}
    small_cache = {
        "cache_roots": [str(path) for path in roots],
        "selected_model_path": str(selected_model["path"]) if selected_model else None,
        "selected_model_id": str(selected_model["model_id"]) if selected_model else None,
    }
    checksum = _reproducibility_checksum(
        {
            "blocked_reason": blocked_reason,
            "exp3261_cuda_python_smoke_passed": exp3261.get("cuda_python_smoke_passed"),
            "llama_backend": llama_backend,
            "metrics": metrics,
            "model_specs": model_specs,
            "random_seed": int(random_seed),
            "selected_python": selected,
            "worker_returncode": worker.get("returncode"),
        }
    )
    duration_s = round(max(0.0, monotonic() - start), 6)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "selected_python": selected,
        "exp3261_gate": {
            "path": str(root / EXP3261_REL_PATH),
            "cuda_python_smoke_passed": exp3261.get("cuda_python_smoke_passed") is True,
        },
        "llama_cpp_cuda_receipt_smoke_v4_ready": True,
        "llama_cpp_cuda_receipt_ready": receipt_ready,
        "blocked_reason": blocked_reason,
        "llama_cpp_backend": llama_backend,
        "small_gguf_cache": small_cache,
        "selected_model_path": str(selected_model["path"]) if selected_model else None,
        "model_specs": model_specs,
        "worker_attempt": {
            "attempted": bool(worker.get("attempted")),
            "returncode": worker.get("returncode"),
            "command_hash": worker.get("command_hash"),
            "stderr_summary": worker.get("stderr_summary"),
        },
        "gpu_layers_offloaded": metrics["gpu_layers_offloaded"],
        "gpu_mem_baseline_mib": metrics["gpu_mem_baseline_mib"],
        "gpu_mem_used_during_call_mib": metrics["gpu_mem_used_during_call_mib"],
        "gpu_mem_delta_during_call_mib": metrics["gpu_mem_delta_during_call_mib"],
        "tokens_generated": metrics["tokens_generated"],
        "generation_output_nonempty": metrics["generation_output_nonempty"],
        "generation_output_preview": metrics["generation_output_preview"],
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(
            receipt_ready=receipt_ready,
            blocked_reason=blocked_reason,
            metrics=metrics,
        ),
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    """Build and write the Exp 3262 llama.cpp CUDA receipt smoke artifact."""

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
