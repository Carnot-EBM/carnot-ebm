"""Exp 1310 local SOTA GGUF llama.cpp smoke-load probe.

Spec: REQ-INFER-SOTA-006,
      SCENARIO-INFER-SOTA-006-001,
      SCENARIO-INFER-SOTA-006-002,
      SCENARIO-INFER-SOTA-006-003
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


DEFAULT_ARTIFACT_PATH = Path("results/experiment_1310_sota_gguf_llamacpp_smoke_load.json")
EXP1309_ARTIFACT_PATH = Path("results/experiment_1309_sota_gguf_pair_resolver_repair.json")
MANDATED_HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "models_loaded",
    "llama_cpp_import_ok",
    "tokens_per_second",
    "gpu_memory_gb",
    "model_specs_count",
    "models_used",
    "headline_result_possible",
    "honest_verdict",
)
_PROMPT = "Carnot smoke-load check. Answer with exactly one short word: ready"
_QUANTIZATION_SUFFIXES: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q8_XL",
    "Q8_0",
)

CachedPairFn = Callable[..., list[dict[str, Any]] | None]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
GpuMemoryFn = Callable[[Iterable[int]], dict[str, float]]
ClockFn = Callable[[], float]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _quantization_suffix(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    filename = Path(model_path).name.lower()
    matches = [suffix for suffix in _QUANTIZATION_SUFFIXES if suffix.lower() in filename]
    return matches[0] if matches else "unknown"


def _import_llama_class() -> tuple[bool, type[Any] | None, str | None]:
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised by live host state.
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _probe_gpu_memory_gb(gpu_indices: Iterable[int]) -> dict[str, float]:
    requested = {int(gpu) for gpu in gpu_indices}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}

    memory: dict[str, float] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpu_index = int(parts[0])
            used_mib = float(parts[1])
        except ValueError:
            continue
        if gpu_index in requested:
            memory[str(gpu_index)] = round(used_mib / 1024.0, 4)
    return memory


def _base_artifact(*, project_root: Path, run_date: str) -> dict[str, Any]:
    return {
        "status": "complete",
        "models_loaded": 0,
        "llama_cpp_import_attempted": False,
        "llama_cpp_import_ok": False,
        "llama_cpp_import_error": None,
        "tokens_per_second": 0.0,
        "gpu_memory_gb": {},
        "model_specs_count": 0,
        "models_used": [],
        "headline_result_possible": False,
        "honest_verdict": "not_run",
        "artifact": "experiment_1310_sota_gguf_llamacpp_smoke_load",
        "run_date": run_date,
        "schema_version": 1,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "preferred_quant": "Q4_K_M",
            "gpu_indices": [0, 1],
        },
        "mandated_headline_model_ids": list(MANDATED_HEADLINE_MODEL_IDS),
        "resolved_model_specs": [],
        "per_model_results": [],
    }


def _resolved_specs(raw_specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_specs, list) or len(raw_specs) != 2:
        return []

    resolved: list[dict[str, Any]] = []
    for spec in raw_specs:
        hf_id = spec.get("hf_id")
        model_path = spec.get("model_path")
        if hf_id not in MANDATED_HEADLINE_MODEL_IDS or not model_path:
            return []
        resolved.append(
            {
                "name": spec.get("name"),
                "hf_id": hf_id,
                "gpu": spec.get("gpu"),
                "model_path": model_path,
                "quantization_suffix": _quantization_suffix(str(model_path)),
            }
        )
    return resolved


def _blank_model_result(spec: dict[str, Any], *, error: str | None = None) -> dict[str, Any]:
    return {
        "name": spec.get("name"),
        "hf_id": spec.get("hf_id"),
        "gpu": spec.get("gpu"),
        "model_path": spec.get("model_path"),
        "quantization_suffix": spec.get("quantization_suffix"),
        "load_success": False,
        "generated": False,
        "token_count": 0,
        "elapsed_seconds": 0.0,
        "tokens_per_second": 0.0,
        "gpu_memory_gb": None,
        "error": error,
    }


def _completion_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    choices = result.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return str(choices[0].get("text", ""))
    return ""


def _completion_token_count(result: Any, text: str, llm: Any) -> int:
    if isinstance(result, dict):
        usage = result.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            return max(0, completion_tokens)
    tokenize = getattr(llm, "tokenize", None)
    if callable(tokenize):
        try:
            return len(tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            pass
    return len(text.split()) if text.strip() else 0


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    gc.collect()


def _smoke_one_model(
    spec: dict[str, Any],
    *,
    llama_class: type[Any],
    gpu_memory_fn: GpuMemoryFn,
    monotonic: ClockFn,
    max_tokens: int,
) -> dict[str, Any]:
    row = _blank_model_result(spec)
    llm: Any | None = None
    try:
        llm = llama_class(
            model_path=spec["model_path"],
            n_gpu_layers=-1,
            n_ctx=256,
            seed=1310,
            main_gpu=int(spec["gpu"]),
            verbose=False,
        )
        row["load_success"] = True
        started = monotonic()
        result = llm(
            _PROMPT,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        elapsed = max(monotonic() - started, 0.0)
        text = _completion_text(result)
        token_count = _completion_token_count(result, text, llm)
        row["generated"] = token_count > 0
        row["token_count"] = token_count
        row["elapsed_seconds"] = round(elapsed, 6)
        row["tokens_per_second"] = round(token_count / elapsed, 4) if elapsed else 0.0
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            _close_llama(llm)

    memory = gpu_memory_fn([int(spec["gpu"])])
    row["gpu_memory_gb"] = memory.get(str(spec["gpu"]))
    return row


def _aggregate_gpu_memory(rows: list[dict[str, Any]]) -> dict[str, float]:
    memory: dict[str, float] = {}
    for row in rows:
        value = row.get("gpu_memory_gb")
        if isinstance(value, int | float):
            memory[str(row["gpu"])] = float(value)
    return memory


def build_smoke_load_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    exp1309_path: str | Path | None = None,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    gpu_memory_fn: GpuMemoryFn = _probe_gpu_memory_gb,
    monotonic: ClockFn = time.monotonic,
    max_tokens: int = 4,
) -> dict[str, Any]:
    """Build the Exp 1310 artifact and run live smoke generation only when possible."""
    root = Path(project_root)
    artifact = _base_artifact(project_root=root, run_date=run_date)
    prior_path = Path(exp1309_path) if exp1309_path is not None else root / EXP1309_ARTIFACT_PATH
    prior = _read_json(prior_path)
    artifact["exp1309_gate"] = {
        "artifact_path": str(prior_path),
        "artifact_found": bool(prior),
        "sota_pair_ready": bool(prior.get("sota_pair_ready")),
        "status": prior.get("status"),
        "honest_verdict": prior.get("honest_verdict"),
    }

    if prior.get("sota_pair_ready") is not True:
        artifact["blocked_reason"] = "exp1309_sota_pair_not_ready"
        artifact["honest_verdict"] = "blocked_exp1309_sota_pair_not_ready"
        return artifact

    try:
        raw_specs = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:
        artifact["blocked_reason"] = "cached_sota_pair_exception"
        artifact["cached_sota_pair_error"] = f"{type(exc).__name__}: {exc}"
        artifact["honest_verdict"] = "blocked_cached_sota_pair_exception"
        return artifact

    resolved_specs = _resolved_specs(raw_specs)
    artifact["model_specs_count"] = len(resolved_specs)
    artifact["models_used"] = [spec["hf_id"] for spec in resolved_specs]
    artifact["resolved_model_specs"] = resolved_specs

    if len(resolved_specs) != 2:
        artifact["blocked_reason"] = "cached_sota_pair_not_loadable"
        artifact["honest_verdict"] = "blocked_cached_sota_pair_not_loadable"
        return artifact

    artifact["llama_cpp_import_attempted"] = True
    import_ok, llama_class, import_error = llama_importer()
    artifact["llama_cpp_import_ok"] = import_ok
    artifact["llama_cpp_import_error"] = import_error
    if not import_ok or llama_class is None:
        artifact["per_model_results"] = [
            _blank_model_result(spec, error="llama_cpp_import_failed") for spec in resolved_specs
        ]
        artifact["gpu_memory_gb"] = gpu_memory_fn([0, 1])
        artifact["honest_verdict"] = "blocked_llama_cpp_import_failed"
        return artifact

    rows = [
        _smoke_one_model(
            spec,
            llama_class=llama_class,
            gpu_memory_fn=gpu_memory_fn,
            monotonic=monotonic,
            max_tokens=max_tokens,
        )
        for spec in resolved_specs
    ]
    artifact["per_model_results"] = rows
    artifact["models_loaded"] = sum(1 for row in rows if row["load_success"])
    total_tokens = sum(int(row["token_count"]) for row in rows)
    total_elapsed = sum(float(row["elapsed_seconds"]) for row in rows)
    artifact["tokens_per_second"] = round(total_tokens / total_elapsed, 4) if total_elapsed else 0.0
    artifact["gpu_memory_gb"] = _aggregate_gpu_memory(rows)
    artifact["headline_result_possible"] = bool(
        artifact["model_specs_count"] == 2
        and artifact["llama_cpp_import_ok"]
        and all(row["load_success"] and row["generated"] and row["token_count"] > 0 for row in rows)
    )
    artifact["honest_verdict"] = (
        "sota_pair_llamacpp_smoke_loaded"
        if artifact["headline_result_possible"]
        else "sota_pair_smoke_load_failed"
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path,
    run_date: str,
    output_path: str | Path | None = None,
    exp1309_path: str | Path | None = None,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    gpu_memory_fn: GpuMemoryFn = _probe_gpu_memory_gb,
    monotonic: ClockFn = time.monotonic,
    max_tokens: int = 4,
) -> dict[str, Any]:
    """Write the in-progress marker, then overwrite it with the final artifact."""
    root = Path(project_root)
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    _write_json(destination, {"status": "in_progress", "run_date": run_date})
    artifact = build_smoke_load_artifact(
        project_root=root,
        run_date=run_date,
        exp1309_path=exp1309_path,
        cached_pair_fn=cached_pair_fn,
        llama_importer=llama_importer,
        gpu_memory_fn=gpu_memory_fn,
        monotonic=monotonic,
        max_tokens=max_tokens,
    )
    _write_json(destination, artifact)
    return artifact
