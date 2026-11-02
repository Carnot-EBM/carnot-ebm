"""Live SOTA GGUF repair runtime preflight for Exp 1442.

This module is a runtime gate, not a repair-quality evaluation.  It answers the
one operational question downstream experiments need before making headline
claims: can at least one mandated local SOTA GGUF model load through the local
llama.cpp path and emit a usable response to a tiny repair-style prompt?

The code is deliberately local-only.  Cache inspection uses the existing
``cached_sota_pair()`` and ``resolve_cached_gguf()`` patterns, neither of which
downloads weights.  Live inference runs in a short-lived subprocess so a failed
large-model load cannot wedge the parent experiment process.

Spec: REQ-INFER-SOTA-007,
      SCENARIO-INFER-SOTA-007-001,
      SCENARIO-INFER-SOTA-007-002
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_ARTIFACT_PATH = Path("results/experiment_1442_live_sota_repair_runtime_preflight.json")

MANDATED_MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_runtime_probe",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_runtime_probe",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_runtime_probe",
    },
]

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "local_sota_runtime_ready",
    "live_sota_model_inference_used",
    "models_found_in_cache",
    "models_missing_from_cache",
    "gpu_probe",
    "smoke_inference_results",
    "blockers",
    "honest_verdict",
)

REPAIR_STYLE_PROMPT = (
    "Carnot live SOTA repair runtime preflight.\n"
    "A certificate says: <CARNOT_CERT_STATE:REPAIR_HINT> 2+2=5.\n"
    "Return compact JSON only with keys repair_action and corrected_claim."
)

CacheResolverFn = Callable[..., str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
GpuProbeFn = Callable[[], dict[str, Any]]
SmokeProbeFn = Callable[[dict[str, Any]], dict[str, Any]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]


def _utc_run_date() -> str:
    """Return the current UTC date in Carnot's compact artifact format."""
    return time.strftime("%Y%m%d", time.gmtime())


def _repo_root() -> Path:
    """Return the project root using the repository's experiment helper when available."""
    from scripts.experiment_template import _get_repo_root  # noqa: PLC0415

    return Path(_get_repo_root())


def _default_cache_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
    """Resolve a mandated GGUF model from local cache without downloading it."""
    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    return resolve_cached_gguf(hf_id, preferred_quant=preferred_quant)


def _default_cached_pair(**kwargs: Any) -> list[dict[str, Any]] | None:
    """Call the canonical cached SOTA pair helper from the local registry."""
    from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

    return cached_sota_pair(**kwargs)


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:
    """Import llama.cpp's Python binding and return a structured status tuple."""
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised by integration hosts.
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _summarize_stream(text: str | None, *, limit: int = 1000) -> str:
    """Compact stdout/stderr while preserving the diagnostic prefix users need."""
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Return the last JSON object printed by the probe subprocess, if any."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _completion_text(result: Any) -> str:
    """Extract generated text from common llama.cpp completion shapes."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    if isinstance(text, str):
        return text
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def probe_gpu_state(
    *,
    cuda_available_fn: Callable[[], bool] | None = None,
    gpu_count_fn: Callable[[], int] | None = None,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Probe CUDA helper state and nvidia-smi memory without requiring a GPU.

    The experiment needs honest runtime context even on CPU-only machines.  The
    helper functions come from ``scripts.experiment_template`` because that file
    already carries Carnot's ROCm-aware GPU-count fallback.
    """
    if cuda_available_fn is None:
        from scripts.experiment_template import _cuda_is_available  # noqa: PLC0415

        cuda_available_fn = _cuda_is_available
    if gpu_count_fn is None:
        from scripts.experiment_template import _detect_gpu_count_rocm_aware  # noqa: PLC0415

        gpu_count_fn = _detect_gpu_count_rocm_aware

    probe: dict[str, Any] = {
        "cuda_available": bool(cuda_available_fn()),
        "gpu_count": int(gpu_count_fn()),
        "nvidia_smi_available": False,
        "nvidia_smi_error": None,
        "gpus": [],
    }
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = command_runner(cmd, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        probe["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
        return probe

    if result.returncode != 0:
        probe["nvidia_smi_error"] = _summarize_stream(result.stderr)
        return probe

    probe["nvidia_smi_available"] = True
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            total = float(parts[2])
            free = float(parts[3])
            used = float(parts[4])
        except ValueError:
            continue
        probe["gpus"].append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mb": total,
                "memory_free_mb": free,
                "memory_used_mb": used,
            }
        )
    return probe


def _inspect_model_cache(
    cache_resolver: CacheResolverFn,
) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    """Inspect mandated local model paths without triggering any model download."""
    found: list[dict[str, str]] = []
    missing: list[str] = []
    errors: dict[str, str] = {}
    for spec in MANDATED_MODEL_SPECS:
        hf_id = spec["hf_id"]
        try:
            model_path = cache_resolver(hf_id, preferred_quant="Q4_K_M")
        except Exception as exc:
            errors[hf_id] = f"{type(exc).__name__}: {exc}"
            model_path = None
        if model_path:
            found.append({"hf_id": hf_id, "role": spec["role"], "model_path": str(model_path)})
        else:
            missing.append(hf_id)
    return found, missing, errors


def _cached_pair_preview(
    cached_pair_fn: CachedPairFn,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return a JSON-safe preview of cached_sota_pair() or its exact exception."""
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not pair:
        return [], None
    preview: list[dict[str, Any]] = []
    for row in pair:
        preview.append(
            {
                "name": row.get("name"),
                "hf_id": row.get("hf_id"),
                "gpu": row.get("gpu"),
                "model_path": row.get("model_path"),
            }
        )
    return preview, None


def _model_for_probe(
    found_row: dict[str, str],
    cached_pair_preview: Sequence[dict[str, Any]],
    fallback_gpu: int,
) -> dict[str, Any]:
    """Attach a GPU index to a cache-hit row using cached_sota_pair() when possible."""
    gpu = fallback_gpu
    for row in cached_pair_preview:
        if row.get("hf_id") == found_row["hf_id"] and row.get("gpu") is not None:
            gpu = int(row["gpu"])
            break
    return {**found_row, "gpu": gpu}


def _base_artifact(*, project_root: Path, run_date: str) -> dict[str, Any]:
    """Build the complete artifact skeleton with every required gate field."""
    return {
        "status": "complete",
        "artifact": "experiment_1442_live_sota_repair_runtime_preflight",
        "run_date": run_date,
        "project_root": str(project_root),
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in MANDATED_MODEL_SPECS],
        "local_sota_runtime_ready": False,
        "live_sota_model_inference_used": False,
        "models_found_in_cache": [],
        "models_missing_from_cache": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "cached_sota_pair_preview": [],
        "cached_sota_pair_error": None,
        "gpu_probe": {},
        "smoke_inference_results": [],
        "cache_probe_errors": {},
        "blockers": [],
        "honest_verdict": "not_run",
    }


def build_live_runtime_preflight_artifact(
    *,
    project_root: Path,
    run_date: str,
    cache_resolver: CacheResolverFn = _default_cache_resolver,
    cached_pair_fn: CachedPairFn = _default_cached_pair,
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
    smoke_probe_fn: SmokeProbeFn = lambda model: run_live_probe_subprocess(model),
) -> dict[str, Any]:
    """Build the terminal Exp 1442 runtime gate artifact.

    Readiness is intentionally strict.  A cached path alone is not enough;
    readiness becomes true only after a real mandated local GGUF model completes
    a live llama.cpp generation and returns non-empty text.
    """
    artifact = _base_artifact(project_root=project_root, run_date=run_date)
    found, missing, cache_errors = _inspect_model_cache(cache_resolver)
    pair_preview, pair_error = _cached_pair_preview(cached_pair_fn)
    artifact["models_found_in_cache"] = found
    artifact["models_missing_from_cache"] = missing
    artifact["cache_probe_errors"] = cache_errors
    artifact["cached_sota_pair_preview"] = pair_preview
    artifact["cached_sota_pair_error"] = pair_error
    artifact["gpu_probe"] = gpu_probe_fn()

    blockers: list[str] = []
    if cache_errors:
        blockers.append("cache_probe_errors_present")
    if pair_error:
        blockers.append("cached_sota_pair_error")
    if not found:
        blockers.append("no_mandated_sota_models_found_in_local_cache")

    for index, found_row in enumerate(found):
        probe_model = _model_for_probe(found_row, pair_preview, fallback_gpu=index)
        result = smoke_probe_fn(probe_model)
        artifact["smoke_inference_results"].append(result)
        if result.get("truly_live") is True and result.get("usable_response") is True:
            break

    live_success = any(
        row.get("truly_live") is True and row.get("usable_response") is True
        for row in artifact["smoke_inference_results"]
    )
    if found and not live_success:
        blockers.append("no_mandated_sota_model_completed_live_inference")

    artifact["local_sota_runtime_ready"] = live_success
    artifact["live_sota_model_inference_used"] = live_success
    artifact["blockers"] = blockers
    artifact["honest_verdict"] = (
        "live_sota_runtime_ready" if live_success else "blocked_no_live_sota_runtime"
    )
    return artifact


def _in_progress_artifact(*, project_root: Path, run_date: str) -> dict[str, Any]:
    """Build the durable initial artifact written before runtime probes start."""
    artifact = _base_artifact(project_root=project_root, run_date=run_date)
    artifact["status"] = "in_progress"
    artifact["blockers"] = ["runtime preflight not completed yet"]
    artifact["honest_verdict"] = "in_progress"
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON artifact with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: Path | None = None,
    run_date: str | None = None,
    output_path: Path | None = None,
    cache_resolver: CacheResolverFn = _default_cache_resolver,
    cached_pair_fn: CachedPairFn = _default_cached_pair,
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
    smoke_probe_fn: SmokeProbeFn = lambda model: run_live_probe_subprocess(model),
) -> dict[str, Any]:
    """Write the in-progress and final Exp 1442 preflight artifacts."""
    root = project_root or _repo_root()
    date = run_date or _utc_run_date()
    path = output_path or root / DEFAULT_ARTIFACT_PATH
    _write_json(path, _in_progress_artifact(project_root=root, run_date=date))
    artifact = build_live_runtime_preflight_artifact(
        project_root=root,
        run_date=date,
        cache_resolver=cache_resolver,
        cached_pair_fn=cached_pair_fn,
        gpu_probe_fn=gpu_probe_fn,
        smoke_probe_fn=smoke_probe_fn,
    )
    _write_json(path, artifact)
    return artifact


def _build_probe_command(
    model: dict[str, Any],
    *,
    python_executable: str = sys.executable,
    max_tokens: int = 16,
) -> list[str]:
    """Build the exact isolated command used for one live model probe."""
    return [
        python_executable,
        "-m",
        "carnot.reporting.live_sota_repair_runtime_preflight",
        "--probe-one",
        "--hf-id",
        str(model["hf_id"]),
        "--role",
        str(model["role"]),
        "--model-path",
        str(model["model_path"]),
        "--gpu",
        str(model.get("gpu", 0)),
        "--max-tokens",
        str(max_tokens),
    ]


def run_live_probe_subprocess(
    model: dict[str, Any],
    *,
    command_runner: CommandRunner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    python_executable: str = sys.executable,
    timeout_s: int = 180,
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Run one live model smoke probe in a subprocess and summarize evidence."""
    command = _build_probe_command(model, python_executable=python_executable, max_tokens=max_tokens)
    start = monotonic()
    try:
        completed = command_runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - needs real process timing.
        elapsed = round(monotonic() - start, 6)
        return {
            "hf_id": model["hf_id"],
            "role": model["role"],
            "model_path": model["model_path"],
            "command": command,
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "returncode": None,
            "stdout_summary": _summarize_stream(exc.stdout, limit=1000),
            "stderr_summary": _summarize_stream(exc.stderr, limit=1000),
            "elapsed_s": elapsed,
            "truly_live": False,
            "usable_response": False,
            "response_text_preview": "",
            "blocker": "probe_subprocess_timeout",
        }

    elapsed = round(monotonic() - start, 6)
    parsed = _extract_json_from_stdout(stdout) or {}
    truly_live = (
        returncode == 0
        and parsed.get("truly_live") is True
        and parsed.get("usable_response") is True
    )
    response_text = parsed.get("response_text") if isinstance(parsed.get("response_text"), str) else ""
    if truly_live:
        blocker = None
    elif returncode != 0:
        blocker = parsed.get("blocker") or f"probe_subprocess_returncode_{returncode}"
    else:
        blocker = parsed.get("blocker") or "probe_subprocess_no_usable_live_response"

    return {
        "hf_id": model["hf_id"],
        "role": model["role"],
        "model_path": model["model_path"],
        "command": command,
        "runtime_mode": "llama_cpp_subprocess_gpu",
        "returncode": returncode,
        "stdout_summary": _summarize_stream(stdout, limit=1000),
        "stderr_summary": _summarize_stream(stderr, limit=1000),
        "elapsed_s": elapsed,
        "truly_live": truly_live,
        "usable_response": truly_live,
        "response_text_preview": _summarize_stream(response_text, limit=300),
        "blocker": blocker,
    }


def run_live_probe_one(
    *,
    hf_id: str,
    role: str,
    model_path: str,
    gpu: int,
    llama_importer: LlamaImporter = _default_llama_importer,
    monotonic: Callable[[], float] = time.monotonic,
    max_tokens: int = 16,
    prompt: str = REPAIR_STYLE_PROMPT,
) -> dict[str, Any]:
    """Load one local GGUF through llama.cpp and issue the repair-style prompt."""
    start = monotonic()
    base: dict[str, Any] = {
        "hf_id": hf_id,
        "role": role,
        "model_path": model_path,
        "gpu": gpu,
        "runtime_mode": "llama_cpp_direct_gpu",
        "load_success": False,
        "truly_live": False,
        "usable_response": False,
        "response_text": "",
        "elapsed_s": 0.0,
        "blocker": None,
    }
    ok, llama_class, import_error = llama_importer()
    if not ok or llama_class is None:
        base["elapsed_s"] = round(monotonic() - start, 6)
        base["blocker"] = import_error or "llama_cpp_import_failed"
        return base

    llm: Any | None = None
    try:
        llm = llama_class(
            model_path=model_path,
            n_gpu_layers=-1 if gpu >= 0 else 0,
            main_gpu=max(gpu, 0),
            n_ctx=512,
            verbose=False,
        )
        base["load_success"] = True
        completion = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        text = _completion_text(completion).strip()
        base["response_text"] = text
        base["usable_response"] = bool(text)
        base["truly_live"] = bool(text)
    except Exception as exc:
        base["blocker"] = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None and hasattr(llm, "close"):
            llm.close()
    base["elapsed_s"] = round(monotonic() - start, 6)
    return base


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--probe-one", action="store_true")
    parser.add_argument("--hf-id", default="")
    parser.add_argument("--role", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the conductor and the isolated subprocess probe."""
    args = _parse_args(argv)
    if args.probe_one:
        result = run_live_probe_one(
            hf_id=args.hf_id,
            role=args.role,
            model_path=args.model_path,
            gpu=args.gpu,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["truly_live"] and result["usable_response"] else 2

    run_experiment(run_date=args.run_date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m.
    raise SystemExit(main())
