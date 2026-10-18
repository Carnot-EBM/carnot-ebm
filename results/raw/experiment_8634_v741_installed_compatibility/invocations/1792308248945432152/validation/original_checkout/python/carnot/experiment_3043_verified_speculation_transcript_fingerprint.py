"""Exp 3043 deterministic transcript fingerprint preflight.

This module answers a narrow operational question before later repair
experiments rely on live local SOTA GGUF evidence: can the current host replay
the same tiny repair-style prompts with the same model, seed, and decode
configuration and get identical transcript hashes?

The artifact deliberately records hashes and runtime provenance instead of
embedding the prompt or response text in the terminal JSON.  That keeps future
experiments auditable while avoiding a new performance or repair-quality claim.

Spec: REQ-INFER-SOTA-022,
      SCENARIO-INFER-SOTA-022-001,
      SCENARIO-INFER-SOTA-022-002,
      SCENARIO-INFER-SOTA-022-003
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[JsonDict] | None]
ResolveGgufFn = Callable[..., str | None]
LlamaFactory = Callable[..., Any]
ClockFn = Callable[[], float]
RepoCommitFn = Callable[[Path], str]

SCHEMA = "carnot.verified_speculation_transcript_fingerprint.v1"
ARTIFACT_FILENAME = "experiment_3043_verified_speculation_transcript_fingerprint_v1.json"
DEFAULT_ARTIFACT_PATH = Path("results") / ARTIFACT_FILENAME
DEFAULT_RAW_DIR = Path("results") / "raw" / ARTIFACT_FILENAME.removesuffix(".json")
DEFAULT_SEED = 304300
DEFAULT_BATCH_SIZE = 3
DEFAULT_RUN_COUNT = 2
DEFAULT_DECODE_CONFIG: JsonDict = {
    "max_tokens": 24,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
}
DEFAULT_LOAD_CONFIG: JsonDict = {
    "n_ctx": 512,
    "n_batch": 64,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "verbose": False,
}
DEFAULT_PROMPTS: tuple[str, ...] = (
    "Exp 3043 repair fingerprint prompt A.\n"
    "Function: clamp_score(x, lo, hi)\n"
    "Buggy candidate: return min(x, hi)\n"
    "Expected behavior: clamp x into the inclusive [lo, hi] range.\n"
    "Return one concise repaired implementation.",
    "Exp 3043 repair fingerprint prompt B.\n"
    "Function: unique_preserve_order(items)\n"
    "Buggy candidate: return sorted(set(items))\n"
    "Expected behavior: return each distinct value once, preserving first-seen order.\n"
    "Return one concise repaired implementation.",
    "Exp 3043 repair fingerprint prompt C.\n"
    "Function: count_vowels(text)\n"
    "Buggy candidate: count only characters in 'aeiou'\n"
    "Expected behavior: count vowels case-insensitively.\n"
    "Return one concise repaired implementation.",
)
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
FULL_SHA256_LIMIT_BYTES = 512 * 1024 * 1024
BOUNDED_HASH_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 3043.

    The defaults are intentionally tiny because this preflight is about replay
    metadata, not benchmark throughput.  Tests can inject a fake llama factory
    and a temporary model path while the real CLI uses the shared SOTA GGUF
    resolver and local llama.cpp runtime.
    """

    repo_root: Path = Path(__file__).resolve().parents[2]
    output_path: Path | None = None
    raw_dir: Path | None = None
    prompts: Sequence[str] | None = None
    seed: int = DEFAULT_SEED
    batch_size: int = DEFAULT_BATCH_SIZE
    run_count: int = DEFAULT_RUN_COUNT
    preferred_quant: str = "Q4_K_M"
    decode_config: Mapping[str, Any] | None = None
    load_config: Mapping[str, Any] | None = None
    model_checksum_full_limit_bytes: int = FULL_SHA256_LIMIT_BYTES
    selected_model_path_for_tests: Path | None = None

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / DEFAULT_ARTIFACT_PATH

    def transcript_dir(self) -> Path:
        return self.raw_dir or self.repo_root / DEFAULT_RAW_DIR

    def effective_prompts(self) -> list[str]:
        return list(self.prompts or DEFAULT_PROMPTS)[: self.batch_size]

    def effective_decode_config(self) -> JsonDict:
        config = dict(DEFAULT_DECODE_CONFIG)
        if self.decode_config:
            config.update(dict(self.decode_config))
        return config

    def effective_load_config(self) -> JsonDict:
        config = dict(DEFAULT_LOAD_CONFIG)
        if self.load_config:
            config.update(dict(self.load_config))
        return config


def sha256_text(text: str) -> str:
    """Return a full SHA-256 hash for a string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return sha256_text(_json_dumps(value))


def _run_date() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d")


def _repo_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - defensive environment probe.
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def _cuda_probe() -> JsonDict:
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({\n"
        "        'cuda_available': bool(torch.cuda.is_available()),\n"
        "        'gpu_count': int(torch.cuda.device_count()),\n"
        "        'torch_version': str(getattr(torch, '__version__', 'unknown')),\n"
        "        'torch_cuda_version': str(getattr(torch.version, 'cuda', None)),\n"
        "    }, sort_keys=True))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({\n"
        "        'cuda_available': False,\n"
        "        'gpu_count': 0,\n"
        "        'torch_error': f'{type(exc).__name__}: {exc}',\n"
        "    }, sort_keys=True))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - defensive subprocess probe.
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "torch_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "cuda_available": False,
        "gpu_count": 0,
        "torch_error": result.stderr.strip() or result.stdout.strip() or "torch_probe_failed",
    }


def _gpu_inventory() -> JsonDict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except Exception as exc:  # pragma: no cover - nvidia-smi is optional.
        return {"available": False, "gpus": [], "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode != 0:
        return {"available": False, "gpus": [], "error": result.stderr.strip()}

    gpus: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, free_mib, total_mib, used_mib, driver = parts
        gpus.append(
            {
                "index": _int_or_none(index),
                "name": name,
                "memory_free_mib": _int_or_none(free_mib),
                "memory_total_mib": _int_or_none(total_mib),
                "memory_used_mib": _int_or_none(used_mib),
                "driver_version": driver,
            }
        )
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "free_vram_mib_total": sum(gpu.get("memory_free_mib") or 0 for gpu in gpus),
    }


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _python_environment() -> JsonDict:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def _call_cached_pair(
    cached_pair_func: CachedPairFn, config: ExperimentConfig
) -> list[JsonDict] | None:
    return cached_pair_func(gpu_indices=(0, 1), preferred_quant=config.preferred_quant)


def _resolve_cache(
    resolve_gguf_func: ResolveGgufFn, config: ExperimentConfig
) -> dict[str, str | None]:
    cache: dict[str, str | None] = {}
    for hf_id in MANDATED_MODEL_IDS:
        cache[hf_id] = resolve_gguf_func(hf_id, config.preferred_quant)
    if config.selected_model_path_for_tests is not None and not any(cache.values()):
        cache[MANDATED_MODEL_IDS[0]] = str(config.selected_model_path_for_tests)
    return cache


def _select_model(
    cached_pair_result: Sequence[Mapping[str, Any]] | None,
    cache_resolution: Mapping[str, str | None],
) -> JsonDict | None:
    for entry in cached_pair_result or []:
        model_path = entry.get("model_path")
        hf_id = entry.get("hf_id")
        if hf_id in MANDATED_MODEL_IDS and model_path:
            return {
                "name": entry.get("name") or hf_id,
                "hf_id": hf_id,
                "gpu": entry.get("gpu", 0),
                "model_path": str(model_path),
                "source": "cached_sota_pair",
            }

    for model in SOTA_GGUF_MODELS:
        model_path = cache_resolution.get(model["hf_id"])
        if model_path:
            return {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "gpu": 0,
                "model_path": str(model_path),
                "source": "resolve_cached_gguf",
            }
    return None


def _file_evidence(path: str | Path, *, full_limit_bytes: int) -> JsonDict:
    file_path = Path(path)
    evidence: JsonDict = {
        "path": str(file_path),
        "exists": file_path.is_file(),
        "full_sha256_feasible": False,
        "sha256": None,
        "bounded_sha256": None,
        "model_hash_or_cache_path": str(file_path),
        "method": "cache_path_only",
    }
    if not file_path.is_file():
        return evidence

    stat = file_path.stat()
    evidence.update(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "readable": os.access(file_path, os.R_OK),
        }
    )
    if stat.st_size <= full_limit_bytes:
        digest = _hash_file(file_path)
        evidence.update(
            {
                "full_sha256_feasible": True,
                "sha256": digest,
                "model_hash_or_cache_path": f"sha256:{digest}",
                "method": "full_sha256",
            }
        )
        return evidence

    digest = _bounded_file_hash(file_path)
    evidence.update(
        {
            "bounded_sha256": digest,
            "model_hash_or_cache_path": f"bounded_sha256:{digest}",
            "method": "bounded_head_tail_sha256",
        }
    )
    return evidence


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_file_hash(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(BOUNDED_HASH_BYTES))
        if stat.st_size > BOUNDED_HASH_BYTES:
            handle.seek(max(0, stat.st_size - BOUNDED_HASH_BYTES))
            digest.update(handle.read(BOUNDED_HASH_BYTES))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _model_specs(
    selected_model: Mapping[str, Any], model_evidence: Mapping[str, Any]
) -> list[JsonDict]:
    return [
        {
            "name": selected_model.get("name"),
            "hf_id": selected_model.get("hf_id"),
            "gpu": selected_model.get("gpu", 0),
            "model_path": selected_model.get("model_path"),
            "selection_source": selected_model.get("source"),
            "model_hash_or_cache_path": model_evidence.get("model_hash_or_cache_path"),
            "checksum_feasibility": {
                "full_sha256_feasible": bool(model_evidence.get("full_sha256_feasible")),
                "method": model_evidence.get("method"),
                "size_bytes": model_evidence.get("size_bytes"),
            },
        }
    ]


def _batch_context_hash(
    *,
    selected_model: Mapping[str, Any],
    prompt_hashes: Sequence[str],
    config: ExperimentConfig,
    decode_config: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
            "model_hf_id": selected_model.get("hf_id"),
            "model_path": selected_model.get("model_path"),
            "prompt_hashes": list(prompt_hashes),
            "seed": config.seed,
            "batch_size": config.batch_size,
            "run_count": config.run_count,
            "decode_config": dict(decode_config),
            "context": "sequential_replay_same_loaded_model",
        }
    )


def _default_llama_factory(**kwargs: Any) -> Any:  # pragma: no cover - covered by live run.
    from llama_cpp import Llama  # noqa: PLC0415

    return Llama(**kwargs)


def _extract_text(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return ""


def _normalize_output(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _run_live_replay(
    *,
    selected_model: Mapping[str, Any],
    prompts: Sequence[str],
    config: ExperimentConfig,
    decode_config: Mapping[str, Any],
    batch_context_hash: str,
    model_hash_or_cache_path: str,
    llama_factory: LlamaFactory,
) -> tuple[list[JsonDict], list[JsonDict]]:
    load_config = config.effective_load_config()
    llama_kwargs = dict(load_config)
    llama_kwargs["model_path"] = str(selected_model["model_path"])
    llm = llama_factory(**llama_kwargs)
    transcript_rows: list[JsonDict] = []
    try:
        for run_index in range(config.run_count):
            for prompt_index, prompt in enumerate(prompts):
                raw_response = llm(prompt, **dict(decode_config), seed=config.seed)
                text = _extract_text(raw_response)
                normalized = _normalize_output(text)
                transcript_rows.append(
                    {
                        "run_index": run_index + 1,
                        "prompt_index": prompt_index,
                        "prompt_hash": sha256_text(prompt),
                        "model_hash_or_cache_path": model_hash_or_cache_path,
                        "decode_config": dict(decode_config),
                        "seed": config.seed,
                        "raw_output_hash": sha256_text(text),
                        "normalized_output_hash": sha256_text(normalized),
                        "tokens_observed": len(text.split()),
                        "batch_context_hash": batch_context_hash,
                    }
                )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()

    output_hashes = _output_hash_summary(transcript_rows, len(prompts), config.run_count)
    return transcript_rows, output_hashes


def _output_hash_summary(
    transcript_rows: Sequence[Mapping[str, Any]],
    n_prompts: int,
    run_count: int,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for prompt_index in range(n_prompts):
        per_prompt = [
            row for row in transcript_rows if int(row.get("prompt_index", -1)) == prompt_index
        ]
        summary: JsonDict = {
            "prompt_index": prompt_index,
            "raw_output_hashes": [row["raw_output_hash"] for row in per_prompt],
            "normalized_output_hashes": [row["normalized_output_hash"] for row in per_prompt],
        }
        for run_index in range(run_count):
            run_rows = [row for row in per_prompt if row["run_index"] == run_index + 1]
            raw_hash = run_rows[0]["raw_output_hash"] if run_rows else None
            norm_hash = run_rows[0]["normalized_output_hash"] if run_rows else None
            summary[f"run_{run_index + 1}_raw_output_hash"] = raw_hash
            summary[f"run_{run_index + 1}_normalized_output_hash"] = norm_hash
        rows.append(summary)
    return rows


def _replay_divergences(output_hashes: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    divergences: list[JsonDict] = []
    for row in output_hashes:
        raw_hashes = set(row.get("raw_output_hashes") or [])
        normalized_hashes = set(row.get("normalized_output_hashes") or [])
        if len(raw_hashes) > 1 or len(normalized_hashes) > 1:
            divergences.append(
                {
                    "prompt_index": row["prompt_index"],
                    "raw_output_hashes": sorted(raw_hashes),
                    "normalized_output_hashes": sorted(normalized_hashes),
                }
            )
    return divergences


def _write_transcripts(
    config: ExperimentConfig, transcript_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not transcript_rows:
        return []
    transcript_dir = config.transcript_dir()
    transcript_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in transcript_rows:
        path = transcript_dir / f"prompt_{row['prompt_index']:02d}_run_{row['run_index']:02d}.json"
        path.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(str(path))
    return written


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    checksum_payload = {
        "fingerprint_live_ready": artifact.get("fingerprint_live_ready"),
        "models_used": artifact.get("models_used"),
        "model_specs": artifact.get("model_specs"),
        "prompt_hashes": artifact.get("prompt_hashes"),
        "output_hashes": artifact.get("output_hashes"),
        "decode_config": artifact.get("decode_config"),
        "batch_context_hash": artifact.get("batch_context_hash"),
        "deterministic_replay_passed": artifact.get("deterministic_replay_passed"),
    }
    return _sha256_json(checksum_payload)


def _substrate(
    *,
    config: ExperimentConfig,
    cache_resolution: Mapping[str, str | None],
    selected_model: Mapping[str, Any] | None,
    model_evidence: Mapping[str, Any] | None,
    repo_commit_func: RepoCommitFn,
    duration_s: float,
) -> JsonDict:
    return {
        "recorded_before_model_load": True,
        "cuda_probe": _cuda_probe(),
        "gpu_inventory": _gpu_inventory(),
        "repo_commit": repo_commit_func(config.repo_root),
        "python_environment": _python_environment(),
        "gguf_cache_resolution": dict(cache_resolution),
        "selected_model_path": selected_model.get("model_path") if selected_model else None,
        "model_checksum_feasibility": dict(model_evidence or {}),
        "seed": config.seed,
        "batch_size": config.batch_size,
        "decode_config": config.effective_decode_config(),
        "load_config": config.effective_load_config(),
        "wall_clock_duration_s": round(duration_s, 6),
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    finished: float,
    cache_resolution: Mapping[str, str | None],
    repo_commit_func: RepoCommitFn,
) -> JsonDict:
    duration_s = round(finished - started, 6)
    artifact: JsonDict = {
        "artifact": ARTIFACT_FILENAME.removesuffix(".json"),
        "schema": SCHEMA,
        "run_date": _run_date(),
        "status": "blocked",
        "fingerprint_live_ready": False,
        "models_used": [],
        "model_specs": [],
        "legacy_smoke_only_used": False,
        "n_prompts": 0,
        "deterministic_replay_passed": False,
        "prompt_hashes": [],
        "output_hashes": [],
        "decode_config": config.effective_decode_config(),
        "batch_context_hash": "",
        "reproducibility_checksum": "",
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_model=None,
            model_evidence=None,
            repo_commit_func=repo_commit_func,
            duration_s=duration_s,
        ),
        "replay_divergences": [],
        "transcript_fingerprints": [],
        "raw_transcript_paths": [],
        "performance_or_repair_promotion_claim": False,
        "duration_s": duration_s,
        "honest_verdict": "blocked_sota_gguf_unavailable: no mandated local SOTA GGUF resolved",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Build the Exp 3043 terminal artifact without downloading model files."""

    config = config or ExperimentConfig()
    started = monotonic()
    cached_pair_result = _call_cached_pair(cached_pair_func, config)
    cache_resolution = _resolve_cache(resolve_gguf_func, config)
    selected_model = _select_model(cached_pair_result, cache_resolution)
    if selected_model is None:
        return _blocked_artifact(
            config=config,
            started=started,
            finished=monotonic(),
            cache_resolution=cache_resolution,
            repo_commit_func=repo_commit_func,
        )

    prompts = config.effective_prompts()
    prompt_hashes = [sha256_text(prompt) for prompt in prompts]
    decode_config = config.effective_decode_config()
    model_evidence = _file_evidence(
        selected_model["model_path"],
        full_limit_bytes=config.model_checksum_full_limit_bytes,
    )
    model_hash_or_cache_path = str(model_evidence.get("model_hash_or_cache_path"))
    batch_context_hash = _batch_context_hash(
        selected_model=selected_model,
        prompt_hashes=prompt_hashes,
        config=config,
        decode_config=decode_config,
    )

    try:
        transcript_rows, output_hashes = _run_live_replay(
            selected_model=selected_model,
            prompts=prompts,
            config=config,
            decode_config=decode_config,
            batch_context_hash=batch_context_hash,
            model_hash_or_cache_path=model_hash_or_cache_path,
            llama_factory=llama_factory or _default_llama_factory,
        )
        runtime_blocker = None
    except Exception as exc:
        transcript_rows = []
        output_hashes = []
        runtime_blocker = f"{type(exc).__name__}: {exc}"

    divergences = _replay_divergences(output_hashes)
    deterministic_replay_passed = (
        bool(output_hashes) and not divergences and runtime_blocker is None
    )
    fingerprint_live_ready = deterministic_replay_passed
    raw_paths = _write_transcripts(config, transcript_rows)
    finished = monotonic()
    duration_s = round(finished - started, 6)

    artifact: JsonDict = {
        "artifact": ARTIFACT_FILENAME.removesuffix(".json"),
        "schema": SCHEMA,
        "run_date": _run_date(),
        "status": "complete" if runtime_blocker is None else "blocked",
        "fingerprint_live_ready": fingerprint_live_ready,
        "models_used": [str(selected_model["hf_id"])] if runtime_blocker is None else [],
        "model_specs": _model_specs(selected_model, model_evidence),
        "legacy_smoke_only_used": False,
        "n_prompts": len(prompts) if runtime_blocker is None else 0,
        "deterministic_replay_passed": deterministic_replay_passed,
        "prompt_hashes": prompt_hashes if runtime_blocker is None else [],
        "output_hashes": output_hashes,
        "decode_config": decode_config,
        "batch_context_hash": batch_context_hash if runtime_blocker is None else "",
        "reproducibility_checksum": "",
        "inference_substrate": _substrate(
            config=config,
            cache_resolution=cache_resolution,
            selected_model=selected_model,
            model_evidence=model_evidence,
            repo_commit_func=repo_commit_func,
            duration_s=duration_s,
        ),
        "transcript_fingerprints": transcript_rows,
        "raw_transcript_paths": raw_paths,
        "replay_divergences": divergences,
        "runtime_blocker": runtime_blocker,
        "performance_or_repair_promotion_claim": False,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(
            runtime_blocker=runtime_blocker,
            fingerprint_live_ready=fingerprint_live_ready,
            deterministic_replay_passed=deterministic_replay_passed,
            n_prompts=len(prompts),
        ),
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def _honest_verdict(
    *,
    runtime_blocker: str | None,
    fingerprint_live_ready: bool,
    deterministic_replay_passed: bool,
    n_prompts: int,
) -> str:
    if runtime_blocker is not None:
        return f"blocked_sota_gguf_unavailable: live generation failed: {runtime_blocker}"
    if fingerprint_live_ready and deterministic_replay_passed:
        return (
            "complete: fingerprint_live_ready=true; "
            f"deterministic_replay_passed=true; n_prompts={n_prompts}"
        )
    return "complete_replay_diverged: live generation ran but deterministic hashes diverged"


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    resolve_gguf_func: ResolveGgufFn = resolve_cached_gguf,
    llama_factory: LlamaFactory | None = None,
    monotonic: ClockFn = time.monotonic,
    repo_commit_func: RepoCommitFn = _repo_commit,
) -> JsonDict:
    """Build and persist the Exp 3043 JSON artifact."""

    config = config or ExperimentConfig()
    artifact = build_artifact(
        config,
        cached_pair_func=cached_pair_func,
        resolve_gguf_func=resolve_gguf_func,
        llama_factory=llama_factory,
        monotonic=monotonic,
        repo_commit_func=repo_commit_func,
    )
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_artifact()
    print(
        json.dumps(
            {
                "path": str(ExperimentConfig().artifact_path()),
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0 if artifact["status"] == "complete" else 2


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
