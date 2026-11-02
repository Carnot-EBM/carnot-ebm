#!/usr/bin/env python3
"""Exp 5097: clean SOTA endpoint/logprob cache runtime provenance.

Spec refs: REQ-INFER-SOTA-027,
SCENARIO-INFER-SOTA-027-SUCCESS,
SCENARIO-INFER-SOTA-027-BLOCKED.

This module repairs the runtime-provenance gap from Exp 5085/5086.  It does
not claim a uPRM, process-verifier, or hallucination-detection result.  Its
only claim is whether the local SOTA GGUF endpoint and a tiny logprob cache are
cleanly usable, or why they are blocked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf  # noqa: E402
from scripts import experiment_3013_sota_gguf_logprob_telemetry_preflight_v1 as exp3013  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
PreconditionProbe = Callable[[Path, Mapping[str, str]], JsonDict]
EndpointProbe = Callable[[list[str], float], JsonDict]
EndpointSample = Callable[[str, float], JsonDict]
CacheSample = Callable[[int, str, JsonDict], JsonDict]
ServerFinder = Callable[[Mapping[str, str]], JsonDict]
FreePort = Callable[[str], JsonDict]
ServerStart = Callable[[list[str], dict[str, str], Path], Any]
ServerCleanup = Callable[[Any], JsonDict]
Clock = Callable[[], float]
AdversarialVerify = Callable[[Path], JsonDict]

EXPERIMENT_ID = 5097
EXPERIMENT_NAME = "experiment_5097_clean_sota_endpoint_logprob_cache"
SCHEMA = "carnot.experiment_5097_clean_sota_endpoint_logprob_cache.v1"
RESULT_RELATIVE_PATH = "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json"
CACHE_RELATIVE_PATH = "results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.jsonl"
RAW_LOG_RELATIVE_PATH = "results/raw/experiment_5097_clean_sota_endpoint_logprob_cache/llamacpp_server.log"
SPEC_REFS = [
    "REQ-INFER-SOTA-027",
    "SCENARIO-INFER-SOTA-027-SUCCESS",
    "SCENARIO-INFER-SOTA-027-BLOCKED",
]
RANDOM_SEED = 20260701
DEFAULT_ENDPOINTS = ("http://127.0.0.1:8080",)
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
DEFAULT_ENDPOINT_TIMEOUT_S = 5.0
DEFAULT_CACHE_ROWS = 10
LIVE_LLM_SUBSTRATE = "live_llm_inference"
BLOCKED_SUBSTRATE = "precondition_check_only"
SUCCESS_VERDICT = "success_clean_sota_endpoint_logprob_cache_ready"
BLOCKED_VERDICT = "blocked_clean_sota_endpoint_logprob_cache_no_live_logprobs"

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
ROLE_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal-prefix verdict separates a clean endpoint/cache result from an honest blocked runtime-precondition result."
    },
    "duration_s": {
        "principle": "wall-clock duration catches live-substrate claims that are too short to be credible."
    },
    "inference_substrate": {
        "principle": "declares live_llm_inference only when a local endpoint actually returned completion/logprob evidence."
    },
    "preconditions_checked": {
        "principle": "records CUDA/GPU, VRAM, llama.cpp, GGUF cache, free-port, cache-path, disk, and env evidence before inference."
    },
    "model_specs": {
        "principle": "names all three mandated SOTA GGUF IDs with resolved local .gguf paths or missing diagnostics."
    },
    "usable_sota_models": {
        "principle": "the exact mandated local GGUF files that can be handed to llama.cpp."
    },
    "server_command": {
        "principle": "exact command proves whether a local server was started instead of silently assuming one existed."
    },
    "endpoint_url": {
        "principle": "concrete endpoint URL makes the completion/logprob proof replayable."
    },
    "endpoint_lifetime_s": {
        "principle": "measured endpoint lifetime prevents a too-short live-substrate readiness claim."
    },
    "completion_endpoint_ready": {
        "principle": "true only when a local endpoint returned non-empty deterministic completion text."
    },
    "logprob_endpoint_ready": {
        "principle": "true only when real token logprob or top-logprob evidence was observed."
    },
    "top_logprob_or_confidence_ready": {
        "principle": "true only when top-logprob alternatives or structured confidence evidence are present."
    },
    "sample_completion": {
        "principle": "stores the deterministic completion proof used for readiness."
    },
    "sample_logprob_evidence": {
        "principle": "stores observed token/top-logprob counts and examples without fabricating absent telemetry."
    },
    "cache_rows_written": {
        "principle": "cache rows are counted only when real endpoint telemetry backs them."
    },
    "cache_path": {
        "principle": "stable JSONL path for the smoke cache or blocked no-row evidence."
    },
    "logprob_endpoint_clean": {
        "principle": "true only after endpoint, cache, and adversarial provenance checks all pass."
    },
    "live_llm_invoked": {
        "principle": "true only when the local LLM endpoint returned completion/logprob evidence during this run."
    },
    "adversarial_verify_passed": {
        "principle": "records the repository adversarial verifier outcome before readiness is declared."
    },
    "blocker_root_cause": {
        "principle": "machine-readable missing endpoint, server, runtime, cache, or telemetry evidence for blocked runs."
    },
    "flagged_adversarial": {
        "principle": "true if the artifact's own provenance or adversarial verifier detects an inconsistent claim."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
CACHE_ROW_SCHEMA = "carnot.experiment_5097_clean_sota_endpoint_logprob_cache.row.v1"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _recorded_env_vars(env: Mapping[str, str]) -> JsonDict:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "CARNOT_5097_ENDPOINTS",
        "CARNOT_LLAMA_ENDPOINTS",
        "CARNOT_JUDGE_ENDPOINTS",
        "CARNOT_JUDGE_SERVER_URL",
        "CARNOT_LLAMA_SERVER",
        "LLAMA_SERVER",
        "CARNOT_LLAMA_SERVER_ARGS",
        "CARNOT_LLAMA_SERVER_START_TIMEOUT_S",
        "HUGGINGFACE_HUB_CACHE",
    )
    return {key: env[key] for key in keys if key in env}


def _default_endpoint_list(env: Mapping[str, str]) -> list[str]:
    raw = (
        env.get("CARNOT_5097_ENDPOINTS")
        or env.get("CARNOT_LLAMA_ENDPOINTS")
        or env.get("CARNOT_JUDGE_ENDPOINTS")
        or env.get("CARNOT_JUDGE_SERVER_URL")
        or ""
    )
    endpoints = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    return endpoints or list(DEFAULT_ENDPOINTS)


def _normalize_endpoints(endpoints: Sequence[str] | None, env: Mapping[str, str]) -> list[str]:
    raw = list(endpoints) if endpoints is not None else _default_endpoint_list(env)
    normalized: list[str] = []
    for endpoint in raw:
        value = str(endpoint).strip().rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_ENDPOINTS)


def _run_command(command: list[str], timeout_s: int = 10) -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def default_precondition_probe(root: Path, env: Mapping[str, str]) -> JsonDict:  # pragma: no cover
    cuda_available = False
    gpu_count = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count())
    except Exception:
        pass

    gpus: list[JsonDict] = []
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5,
    )
    if result.get("returncode") == 0:
        for line in str(result.get("stdout") or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_vram_mb": int(float(parts[2])),
                        "used_vram_mb": int(float(parts[3])),
                        "free_vram_mb": int(float(parts[4])),
                    }
                )

    llama_cpp_python: JsonDict
    try:
        import llama_cpp

        llama_cpp_python = {
            "available": True,
            "version": getattr(llama_cpp, "__version__", None),
            "detail": "llama_cpp import ok",
        }
    except Exception as exc:
        llama_cpp_python = {
            "available": False,
            "version": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    disk = shutil.disk_usage(root)
    total_free = sum(int(row.get("free_vram_mb") or 0) for row in gpus)
    return {
        "cuda_gpu_visibility": {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count or len(gpus),
            "gpus": gpus,
            "nvidia_smi": {
                "available": result.get("returncode") == 0,
                "stderr": result.get("stderr"),
            },
        },
        "llama_cpp_python": llama_cpp_python,
        "free_vram": {
            "available": bool(gpus),
            "total_free_vram_mb": total_free,
        },
        "disk_space": {
            "available": disk.free > 0,
            "free_bytes": int(disk.free),
            "free_gib": round(disk.free / (1024**3), 3),
        },
    }


def _model_file_evidence(path: str | None) -> JsonDict:
    if not path:
        return {"exists": False, "size_bytes": None, "sha256": None}
    p = Path(path)
    return {
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "sha256": _sha256_file(p),
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
) -> tuple[JsonDict, list[JsonDict]]:
    cached_pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant=preferred_quant)
    cached_pair_rows = list(cached_pair or [])
    mandatory_models: list[JsonDict] = []
    usable: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        path = model_resolver(hf_id, preferred_quant)
        evidence = _model_file_evidence(path)
        row = {
            "role": ROLE_BY_HF_ID[hf_id],
            "hf_id": hf_id,
            "preferred_quant": preferred_quant,
            "resolved_path": path,
            "cache_status": "resolved" if path and evidence["exists"] else "missing",
            "missing_diagnostic": None if path and evidence["exists"] else f"missing cached GGUF for {hf_id}",
            "file_evidence": evidence,
        }
        mandatory_models.append(row)
        if row["cache_status"] == "resolved":
            usable.append(
                {
                    "role": row["role"],
                    "hf_id": hf_id,
                    "model_path": str(path),
                    "size_bytes": evidence["size_bytes"],
                }
            )
    return (
        {
            "mandatory_models": mandatory_models,
            "cached_sota_pair": cached_pair_rows,
            "preferred_quant": preferred_quant,
            "loader": "llama.cpp",
            "gguf_tokenizer_rule": "embedded_gguf_tokenizer_only",
        },
        usable,
    )


def _select_bringup_model(usable_sota_models: Sequence[JsonMap]) -> JsonDict | None:
    candidates = [dict(row) for row in usable_sota_models if row.get("model_path")]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row.get("size_bytes") is None,
            int(row.get("size_bytes") or 0),
            str(row.get("hf_id") or ""),
        ),
    )


def _sample_logprob_payload(sample: JsonMap) -> JsonDict:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), Mapping) else {}
    return {
        "ready": bool(sample.get("logprob_ready") or sample.get("top_logprob_ready")),
        "token_logprob_count": int(evidence.get("token_logprob_count") or 0),
        "top_logprob_row_count": int(evidence.get("top_logprob_row_count") or 0),
        "token_logprobs": list(evidence.get("token_logprobs") or []),
        "top_logprobs": list(evidence.get("top_logprobs") or []),
        "telemetry_signal": sample.get("telemetry_signal"),
        "route": sample.get("route"),
        "status": sample.get("status"),
        "error": sample.get("error"),
    }


def _sample_completion_payload(sample: JsonMap, selected_model: JsonMap | None) -> JsonDict | None:
    if not sample.get("ready"):
        return None
    return {
        "prompt": exp3013.DEFAULT_PROMPT,
        "text": str(sample.get("completion_text") or ""),
        "route": sample.get("route"),
        "status": sample.get("status"),
        "model_hf_id": (selected_model or {}).get("hf_id"),
        "model_path": (selected_model or {}).get("model_path"),
    }


def _has_real_logprob_evidence(sample: JsonMap) -> bool:
    evidence = _sample_logprob_payload(sample)
    return bool(evidence["token_logprob_count"] > 0 or evidence["top_logprob_row_count"] > 0)


def _tail_file(path: Path | None) -> str:
    return exp3013._tail_file(path)


def _default_blocker(kind: str, detail: str, **extra: Any) -> JsonDict:
    payload = {"kind": kind, "detail": detail}
    payload.update(extra)
    return payload


def default_cache_sample(row_index: int, endpoint: str, selected_model: JsonDict) -> JsonDict:  # pragma: no cover
    base = endpoint.rstrip("/")
    prompt = f"Exp5097 cache smoke row {row_index}. Return exactly: row {row_index} ok."
    payload = {
        "prompt": prompt,
        "n_predict": 16,
        "temperature": 0.0,
        "seed": RANDOM_SEED + row_index,
        "n_probs": 5,
        "top_k": 5,
    }
    try:
        status, parsed = exp3013._http_post_json(base + "/completion", payload, DEFAULT_ENDPOINT_TIMEOUT_S)
        parsed_sample = exp3013._parse_endpoint_sample_payload(parsed)
    except Exception as exc:
        return {
            "ready": False,
            "route": base + "/completion",
            "status": None,
            "completion_text": "",
            "logprob_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "telemetry_signal": None,
            "evidence": {
                "token_logprob_count": 0,
                "top_logprob_row_count": 0,
                "token_logprobs": [],
                "top_logprobs": [],
                "raw_response_keys": [],
            },
            "error": exp3013._http_error_detail(exc),
        }
    token_logprobs = parsed_sample["token_logprobs"]
    top_logprobs = parsed_sample["top_logprobs"]
    return {
        "ready": bool(200 <= status < 300 and parsed_sample["text"]),
        "route": base + "/completion",
        "status": status,
        "completion_text": parsed_sample["text"],
        "logprob_ready": bool(token_logprobs),
        "top_logprob_ready": bool(top_logprobs),
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs" if top_logprobs else None,
        "evidence": {
            "token_logprob_count": len(token_logprobs),
            "top_logprob_row_count": len(top_logprobs),
            "token_logprobs": token_logprobs[:8],
            "top_logprobs": top_logprobs[:4],
            "raw_response_keys": sorted(parsed.keys()) if isinstance(parsed, Mapping) else [],
        },
        "error": None,
        "selected_model_hf_id": selected_model.get("hf_id"),
    }


def build_cache_row(row_index: int, sample: JsonMap, endpoint: str, selected_model: JsonMap) -> JsonDict:
    evidence = sample.get("evidence") if isinstance(sample.get("evidence"), Mapping) else {}
    token_logprobs = [
        float(value)
        for value in evidence.get("token_logprobs") or []
        if _finite_float(value) is not None
    ]
    top_logprobs = [
        {str(token): float(value) for token, value in row.items() if _finite_float(value) is not None}
        for row in evidence.get("top_logprobs") or []
        if isinstance(row, Mapping)
    ]
    prompt = f"exp5097-cache-smoke-row-{row_index}"
    completion_text = str(sample.get("completion_text") or "")
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": f"exp5097-cache-smoke-{row_index:04d}",
        "row_index": int(row_index),
        "endpoint_used": str(sample.get("route") or endpoint.rstrip("/") + "/completion"),
        "model_hf_id": str(selected_model.get("hf_id") or ""),
        "gguf_path": str(selected_model.get("model_path") or ""),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "completion_text": completion_text,
        "completion_hash": hashlib.sha256(completion_text.encode("utf-8")).hexdigest(),
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "token_logprob_count": len(token_logprobs),
        "top_logprob_row_count": len(top_logprobs),
        "random_seed": RANDOM_SEED + int(row_index),
        "source_sample_status": sample.get("status"),
        "telemetry_signal": sample.get("telemetry_signal"),
    }


def validate_cache_row(row: JsonMap) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    for field in ("row_id", "endpoint_used", "model_hf_id", "gguf_path", "prompt_hash", "completion_hash"):
        if not str(row.get(field) or ""):
            errors.append(field)
    if str(row.get("model_hf_id") or "") not in MANDATED_MODEL_IDS:
        errors.append("model_hf_id")
    for field in ("prompt_hash", "completion_hash"):
        if len(str(row.get(field) or "")) != 64:
            errors.append(field)
    token_logprobs = row.get("token_logprobs")
    top_logprobs = row.get("top_logprobs")
    token_ready = isinstance(token_logprobs, list) and any(_finite_float(value) is not None for value in token_logprobs)
    top_ready = isinstance(top_logprobs, list) and bool(top_logprobs)
    if not (token_ready or top_ready):
        errors.append("logprob_evidence")
    return sorted(set(errors))


def _build_cache_rows(
    *,
    endpoint_url: str,
    selected_model: JsonDict,
    cache_sample: CacheSample,
    cache_rows_required: int,
) -> tuple[list[JsonDict], JsonDict | None]:
    rows: list[JsonDict] = []
    for index in range(cache_rows_required):
        sample = cache_sample(index, endpoint_url, selected_model)
        if not _has_real_logprob_evidence(sample):
            return [], _default_blocker(
                "cache_smoke_logprob_missing",
                "cache smoke sample lacked token/top-logprob evidence",
                row_index=index,
                sample_error=sample.get("error"),
            )
        row = build_cache_row(index, sample, endpoint_url, selected_model)
        errors = validate_cache_row(row)
        if errors:
            return [], _default_blocker(
                "cache_smoke_row_invalid",
                "cache smoke row failed validation",
                row_index=index,
                errors=errors,
            )
        rows.append(row)
    return rows, None


def _preconditions_checked(
    *,
    root: Path,
    cache_path: Path,
    env: Mapping[str, str],
    probe: JsonMap,
    model_specs: JsonMap,
    server: JsonMap,
    free_port: JsonMap,
) -> JsonDict:
    disk = probe.get("disk_space")
    if not isinstance(disk, Mapping):
        usage = shutil.disk_usage(root)
        disk = {
            "available": usage.free > 0,
            "free_bytes": int(usage.free),
            "free_gib": round(usage.free / (1024**3), 3),
        }
    resolved = {
        str(row["hf_id"]): row.get("resolved_path")
        for row in model_specs["mandatory_models"]
        if row.get("cache_status") == "resolved"
    }
    return {
        "recorded_before_live_inference": True,
        "cuda_gpu_visibility": probe.get("cuda_gpu_visibility", {}),
        "llama_cpp_python": probe.get("llama_cpp_python", {}),
        "llama_cpp_server": dict(server),
        "resolved_local_gguf_paths": resolved,
        "free_port": dict(free_port),
        "cache_path": {
            "path": cache_path.as_posix(),
            "parent_exists": cache_path.parent.exists(),
            "will_write_only_with_real_logprobs": True,
        },
        "disk_space": dict(disk),
        "free_vram": probe.get("free_vram", {}),
        "environment_variables": _recorded_env_vars(env),
    }


def _adversarial_report_flags(report: JsonMap) -> list[JsonDict]:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def default_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover
    from scripts import adversarial_verify

    report = adversarial_verify.verify_artifact(path)
    return report if isinstance(report, dict) else {"flags": []}


def _reproducibility_checksum(payload: JsonMap) -> str:
    basis = {
        "schema": payload.get("schema"),
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "inference_substrate": payload.get("inference_substrate"),
        "model_specs": payload.get("model_specs"),
        "usable_sota_models": payload.get("usable_sota_models"),
        "endpoint_url": payload.get("endpoint_url"),
        "sample_logprob_evidence": payload.get("sample_logprob_evidence"),
        "cache_rows_written": payload.get("cache_rows_written"),
        "blocker_root_cause": payload.get("blocker_root_cause"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_payload(basis)


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
        if field not in artifact.get("field_principles", {}):
            errors.append(f"field_principles.{field}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id")
    model_specs = artifact.get("model_specs")
    if not isinstance(model_specs, Mapping):
        errors.append("model_specs")
    else:
        models = model_specs.get("mandatory_models")
        if not isinstance(models, list) or [row.get("hf_id") for row in models] != list(MANDATED_MODEL_IDS):
            errors.append("model_specs.mandatory_models")
    if artifact.get("cache_rows_written", 0) and not artifact.get("logprob_endpoint_ready"):
        errors.append("cache_rows_without_logprob_ready")
    if artifact.get("logprob_endpoint_clean") and artifact.get("flagged_adversarial"):
        errors.append("clean_but_flagged")
    return sorted(set(errors))


def _initial_endpoint_probe(
    *,
    endpoint_probe: EndpointProbe,
    endpoints: list[str],
    timeout_s: float,
) -> tuple[JsonDict, str]:
    summary = endpoint_probe(endpoints, timeout_s)
    endpoint_url = str(summary.get("selected_endpoint") or endpoints[0])
    return summary, endpoint_url


def _maybe_start_server(
    *,
    root: Path,
    env: Mapping[str, str],
    selected_model: JsonDict | None,
    server: JsonMap,
    free_port: JsonMap,
    endpoint_probe: EndpointProbe,
    endpoint_sample: EndpointSample,
    timeout_s: float,
    server_start: ServerStart,
    server_cleanup: ServerCleanup,
) -> JsonDict:
    log_path = root / RAW_LOG_RELATIVE_PATH
    evidence: JsonDict = {
        "attempted": False,
        "endpoint_summary": None,
        "sample": {
            "ready": False,
            "error": "server bring-up not attempted",
            "evidence": {"token_logprob_count": 0, "top_logprob_row_count": 0},
        },
        "server_command": None,
        "server_pid": None,
        "server_logs": {"path": log_path.as_posix(), "tail": "", "exists": False},
        "cleanup_behavior": {"started_by_preflight": False, "terminated": False},
        "blocker_root_cause": None,
    }
    if selected_model is None:
        evidence["blocker_root_cause"] = _default_blocker(
            "no_usable_sota_model",
            "no mandated local GGUF resolved for endpoint bring-up",
        )
        return evidence
    if not server.get("available"):
        evidence["blocker_root_cause"] = _default_blocker(
            "llama_server_binary_unavailable",
            str(server.get("missing_diagnostic") or "llama-server unavailable"),
            candidates=server.get("candidates", []),
        )
        return evidence
    if not free_port.get("available") or not free_port.get("endpoint_url"):
        evidence["blocker_root_cause"] = _default_blocker(
            "free_port_unavailable",
            str(free_port.get("error") or "could not allocate free local port"),
            free_port=free_port,
        )
        return evidence

    command = exp3013._build_server_command(
        server_path=str(server["selected_path"]),
        model_path=str(selected_model["model_path"]),
        host=str(free_port["host"]),
        port=int(free_port["port"]),
        extra_args=env.get("CARNOT_LLAMA_SERVER_ARGS"),
    )
    evidence["attempted"] = True
    evidence["server_command"] = command
    process = None
    try:
        process = server_start(command, dict(env), log_path)
        evidence["server_pid"] = getattr(process, "pid", None)
        endpoint_url = str(free_port["endpoint_url"])
        summary = endpoint_probe([endpoint_url], timeout_s)
        evidence["endpoint_summary"] = summary
        if summary.get("completion_ready"):
            evidence["sample"] = endpoint_sample(endpoint_url, timeout_s)
        else:
            evidence["blocker_root_cause"] = _default_blocker(
                "server_started_but_endpoint_not_ready",
                "llama-server started but no completion endpoint became ready",
                endpoint_summary=summary,
            )
    except Exception as exc:
        evidence["blocker_root_cause"] = _default_blocker(
            "server_start_failed",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if process is not None:
            evidence["cleanup_behavior"] = server_cleanup(process)
        evidence["server_logs"] = {
            "path": log_path.as_posix(),
            "tail": _tail_file(log_path),
            "exists": log_path.exists(),
        }
    return evidence


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    precondition_probe: PreconditionProbe = default_precondition_probe,
    endpoint_probe: EndpointProbe = exp3013._probe_endpoint_summary,
    endpoint_sample: EndpointSample = exp3013._sample_endpoint_telemetry,
    cache_sample: CacheSample = default_cache_sample,
    server_finder: ServerFinder = exp3013._llama_server_availability,
    free_port: FreePort = exp3013._find_free_port,
    server_start: ServerStart = exp3013._start_llama_server_process,
    server_cleanup: ServerCleanup = exp3013._cleanup_llama_server_process,
    adversarial_verify: AdversarialVerify = default_adversarial_verify,
    endpoints: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    now: Clock = time.monotonic,
    duration_floor_s: float = 60.0,
    endpoint_timeout_s: float = DEFAULT_ENDPOINT_TIMEOUT_S,
    cache_rows_required: int = DEFAULT_CACHE_ROWS,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    destination = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_destination = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    started = now()
    started_at = _utc_now()

    model_specs, usable_sota_models = resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    selected_model = _select_bringup_model(usable_sota_models)
    precondition_probe_result = precondition_probe(root, merged_env)
    server = server_finder(merged_env)
    free = free_port("127.0.0.1")
    normalized_endpoints = _normalize_endpoints(endpoints, merged_env)
    endpoint_summary, endpoint_url = _initial_endpoint_probe(
        endpoint_probe=endpoint_probe,
        endpoints=normalized_endpoints,
        timeout_s=endpoint_timeout_s,
    )

    sample: JsonDict = {
        "ready": False,
        "error": "endpoint completion unavailable",
        "evidence": {"token_logprob_count": 0, "top_logprob_row_count": 0},
    }
    server_command = None
    server_pid = None
    server_logs: JsonDict = {"path": None, "tail": "", "exists": False}
    cleanup_behavior: JsonDict = {"started_by_preflight": False, "terminated": False}
    blocker_root_cause: JsonDict | None = None
    if endpoint_summary.get("completion_ready"):
        sample = endpoint_sample(endpoint_url, endpoint_timeout_s)
        if not sample.get("ready"):
            blocker_root_cause = _default_blocker(
                "endpoint_sample_failed",
                str(sample.get("error") or "endpoint probe passed but sample failed"),
            )
    else:
        bringup = _maybe_start_server(
            root=root,
            env=merged_env,
            selected_model=selected_model,
            server=server,
            free_port=free,
            endpoint_probe=endpoint_probe,
            endpoint_sample=endpoint_sample,
            timeout_s=endpoint_timeout_s,
            server_start=server_start,
            server_cleanup=server_cleanup,
        )
        server_command = bringup.get("server_command")
        server_pid = bringup.get("server_pid")
        server_logs = bringup.get("server_logs") or server_logs
        cleanup_behavior = bringup.get("cleanup_behavior") or cleanup_behavior
        blocker_root_cause = bringup.get("blocker_root_cause")
        if bringup.get("endpoint_summary") is not None:
            endpoint_summary = bringup["endpoint_summary"]
        if free.get("endpoint_url"):
            endpoint_url = str(free["endpoint_url"])
        sample = bringup.get("sample") or sample

    completion_endpoint_ready = bool(endpoint_summary.get("completion_ready") and sample.get("ready"))
    logprob_endpoint_ready = bool(_has_real_logprob_evidence(sample))
    top_logprob_or_confidence_ready = bool(
        sample.get("top_logprob_ready") or sample.get("confidence_ready")
    )
    live_llm_invoked = bool(completion_endpoint_ready and logprob_endpoint_ready)

    if live_llm_invoked and duration_floor_s > 0.0 and now is time.monotonic:
        exp3013._run_duration_floor_endpoint_probe(
            endpoint_url,
            run_started_s=started,
            target_duration_s=duration_floor_s,
            timeout_s=endpoint_timeout_s,
        )

    cache_rows: list[JsonDict] = []
    cache_blocker: JsonDict | None = None
    if live_llm_invoked and selected_model is not None:
        cache_rows, cache_blocker = _build_cache_rows(
            endpoint_url=endpoint_url,
            selected_model=selected_model,
            cache_sample=cache_sample,
            cache_rows_required=cache_rows_required,
        )
        if cache_blocker is None and write:
            write_jsonl(cache_destination, cache_rows)
    elif blocker_root_cause is None:
        blocker_root_cause = _default_blocker(
            "logprob_telemetry_unavailable",
            "no local endpoint returned token/top-logprob evidence",
            sample_logprob_evidence=_sample_logprob_payload(sample),
        )
    if cache_blocker is not None:
        blocker_root_cause = cache_blocker

    finished = now()
    finished_at = _utc_now()
    duration_s = round(float(finished - started), 6)
    endpoint_lifetime_s = round(float(finished - started), 6) if completion_endpoint_ready else 0.0
    cache_rows_written = len(cache_rows) if cache_blocker is None else 0
    success = bool(
        live_llm_invoked
        and top_logprob_or_confidence_ready
        and cache_rows_written >= cache_rows_required
    )
    inference_substrate = LIVE_LLM_SUBSTRATE if live_llm_invoked else BLOCKED_SUBSTRATE
    honest_verdict = SUCCESS_VERDICT if success else BLOCKED_VERDICT
    preconditions_checked = _preconditions_checked(
        root=root,
        cache_path=cache_destination,
        env=merged_env,
        probe=precondition_probe_result,
        model_specs=model_specs,
        server=server,
        free_port=free,
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": destination.as_posix(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "inference_substrate": inference_substrate,
        "preconditions_checked": preconditions_checked,
        "model_specs": model_specs,
        "usable_sota_models": usable_sota_models,
        "server_command": server_command,
        "server_pid": server_pid,
        "server_logs": server_logs,
        "cleanup_behavior": cleanup_behavior,
        "endpoint_url": endpoint_url,
        "endpoint_summary": endpoint_summary,
        "endpoint_lifetime_s": endpoint_lifetime_s,
        "completion_endpoint_ready": completion_endpoint_ready,
        "logprob_endpoint_ready": logprob_endpoint_ready,
        "top_logprob_or_confidence_ready": top_logprob_or_confidence_ready,
        "sample_completion": _sample_completion_payload(sample, selected_model),
        "sample_logprob_evidence": _sample_logprob_payload(sample),
        "cache_rows_written": cache_rows_written,
        "cache_path": cache_destination.as_posix(),
        "logprob_endpoint_clean": False,
        "live_llm_invoked": live_llm_invoked,
        "adversarial_verify_passed": False,
        "adversarial_verify_report": {"flags": []},
        "blocker_root_cause": None if success else blocker_root_cause,
        "flagged_adversarial": False,
        "claimed_capability_scope": "runtime_provenance_only",
        "excluded_claims": ["uPRM", "process_verifier", "hallucination_detection"],
        "honest_verdict": honest_verdict,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if write:
        write_json(destination, artifact)
        report = adversarial_verify(destination)
    else:
        probe_path = destination
        write_json(probe_path, artifact)
        report = adversarial_verify(probe_path)
    flags = _adversarial_report_flags(report)
    critical_flags = [
        flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"
    ]
    artifact["adversarial_verify_report"] = {"flags": flags}
    artifact["adversarial_verify_passed"] = not critical_flags
    artifact["flagged_adversarial"] = bool(critical_flags)
    artifact["logprob_endpoint_clean"] = bool(success and not critical_flags)
    if artifact["flagged_adversarial"] and artifact["blocker_root_cause"] is None:
        artifact["blocker_root_cause"] = _default_blocker(
            "adversarial_verify_failed",
            "repository adversarial verifier reported critical flags",
            flags=critical_flags,
        )
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if write:
        write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--endpoint-timeout-s", type=float, default=DEFAULT_ENDPOINT_TIMEOUT_S)
    parser.add_argument("--duration-floor-s", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run(
        artifact_path=args.output,
        cache_path=args.cache,
        endpoints=args.endpoint,
        endpoint_timeout_s=args.endpoint_timeout_s,
        duration_floor_s=args.duration_floor_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
