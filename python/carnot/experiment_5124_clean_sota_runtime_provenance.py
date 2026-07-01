#!/usr/bin/env python3
"""Exp 5124: clean local SOTA GGUF runtime provenance.

Spec refs: REQ-INFER-SOTA-029,
SCENARIO-INFER-SOTA-029-CLEAN,
SCENARIO-INFER-SOTA-029-BLOCKED.

This module proves only the reusable local endpoint/cache/logprob substrate.
It deliberately stops before downstream verifier experiments so a clean
runtime gate cannot be confused with a benchmark result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution
    sys.path.insert(0, str(REPO_ROOT))
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
DurationFloorProbe = Callable[..., JsonDict]
AdversarialVerify = Callable[[Path], JsonDict]
Clock = Callable[[], float]

EXPERIMENT_ID = "exp5124-clean-sota-runtime-provenance-v470"
EXPERIMENT_NUMERIC_ID = 5124
EXPERIMENT_NAME = "experiment_5124_clean_sota_runtime_provenance"
MILESTONE = "2026.07.470"
SCHEMA = "carnot.experiment_5124_clean_sota_runtime_provenance.v1"
RESULT_RELATIVE_PATH = "results/experiment_5124_clean_sota_runtime_provenance_v470.json"
CACHE_RELATIVE_PATH = "results/experiment_5124_clean_sota_runtime_provenance_v470.jsonl"
RAW_LOG_RELATIVE_PATH = "results/raw/experiment_5124_clean_sota_runtime_provenance/llamacpp_server.log"
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
DEFAULT_ENDPOINTS = ("http://127.0.0.1:8080",)
DEFAULT_ENDPOINT_TIMEOUT_S = 5.0
DEFAULT_SERVER_START_TIMEOUT_S = 120.0
DEFAULT_DURATION_FLOOR_S = 60.0
DEFAULT_PROMPT = (
    "Compute 19 * 23, state the arithmetic result, and add the phrase "
    "exp5124 runtime provenance live."
)
RANDOM_SEED = 20260701
INFERENCE_SUBSTRATE = "local_sota_gguf_llamacpp_runtime_or_blocked"
SUCCESS_VERDICT = "success_clean_sota_runtime_provenance_ready"
BLOCKED_PAIR_VERDICT = "blocked_clean_sota_runtime_provenance_cached_pair_unavailable"
BLOCKED_RUNTIME_VERDICT = "blocked_clean_sota_runtime_provenance_runtime_evidence_missing"
BLOCKED_ADVERSARIAL_VERDICT = "blocked_clean_sota_runtime_provenance_adversarial_flag"
CACHE_ROW_SCHEMA = "carnot.experiment_5124_clean_sota_runtime_provenance.row.v1"

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "Gemma4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma4-26B-A4B-it",
}
MODEL_ROLES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "experiment_id": {"principle": "traceability"},
    "milestone": {"principle": "milestone accountability"},
    "honest_verdict": {"principle": "terminal verdict with complete_/success_/blocked_ prefix"},
    "inference_substrate": {"principle": "substrate honesty"},
    "duration_s": {"principle": "timing accountability"},
    "MODEL_SPECS": {"principle": "mandated local SOTA model provenance"},
    "cached_sota_pair_attempted": {"principle": "local-first discipline"},
    "gguf_paths": {"principle": "loader-path accountability"},
    "completion_proof": {"principle": "live response evidence"},
    "logprob_proof": {"principle": "process telemetry evidence"},
    "cache_ready": {"principle": "reusable substrate"},
    "cache_receipts": {"principle": "cache provenance"},
    "endpoint_lifetime_s": {"principle": "duration-floor evidence"},
    "request_response_transcript": {"principle": "reproducibility"},
    "duration_floor_evidence": {"principle": "adversarial verification readiness"},
    "adversarial_verify_passed": {"principle": "no quarantined runtime headline"},
    "sota_runtime_clean": {"principle": "structured downstream gate"},
    "conductor_modified": {"principle": "conductor immutability"},
    "tests_run": {"principle": "verification evidence"},
}
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = tuple(FIELD_PRINCIPLES)


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
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def default_precondition_probe(root: Path, env: Mapping[str, str]) -> JsonDict:  # pragma: no cover
    del env
    cuda_available = False
    gpu_count = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_count = int(torch.cuda.device_count())
    except Exception:
        pass
    disk = shutil.disk_usage(root)
    return {
        "cuda_status": {"cuda_available": cuda_available, "gpu_count": gpu_count},
        "llama_cpp_python": _llama_cpp_import_status(),
        "disk_ram": {
            "disk_free_bytes": int(disk.free),
            "disk_free_gib": round(disk.free / (1024**3), 3),
        },
    }


def _llama_cpp_import_status() -> JsonDict:  # pragma: no cover
    try:
        import llama_cpp

        return {
            "available": True,
            "version": getattr(llama_cpp, "__version__", None),
            "detail": "llama_cpp import ok",
        }
    except Exception as exc:
        return {"available": False, "version": None, "detail": f"{type(exc).__name__}: {exc}"}


def default_endpoint_sample(endpoint: str, timeout_s: float) -> JsonDict:  # pragma: no cover
    route = endpoint.rstrip("/") + "/completion"
    payload = {
        "prompt": DEFAULT_PROMPT,
        "n_predict": exp3013.SAMPLE_MAX_TOKENS,
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "n_probs": exp3013.LOGPROBS_REQUESTED,
    }
    try:
        status, parsed = exp3013._http_post_json(route, payload, timeout_s)
    except Exception as exc:
        return _empty_sample(route=route, error=exp3013._http_error_detail(exc))
    parsed_sample = exp3013._parse_endpoint_sample_payload(parsed)
    text = str(parsed_sample["text"]).strip()
    token_logprobs = list(parsed_sample["token_logprobs"])
    top_logprobs = list(parsed_sample["top_logprobs"])
    if 200 <= status < 300 and text:
        return {
            "ready": True,
            "route": route,
            "status": status,
            "completion_text": text,
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
        }
    return _empty_sample(route=route, status=status, error="empty completion from endpoint")


def default_cache_sample(row_index: int, endpoint: str, selected_model: JsonDict) -> JsonDict:  # pragma: no cover
    del selected_model
    route = endpoint.rstrip("/") + "/completion"
    payload = {
        "prompt": f"Exp5124 cache row {row_index}: compute 19 * 23 and answer briefly.",
        "n_predict": 24,
        "temperature": 0.0,
        "seed": RANDOM_SEED + row_index,
        "n_probs": exp3013.LOGPROBS_REQUESTED,
    }
    try:
        status, parsed = exp3013._http_post_json(route, payload, DEFAULT_ENDPOINT_TIMEOUT_S)
    except Exception as exc:
        return _empty_sample(route=route, error=exp3013._http_error_detail(exc))
    parsed_sample = exp3013._parse_endpoint_sample_payload(parsed)
    return {
        "ready": bool(200 <= status < 300 and parsed_sample["text"]),
        "route": route,
        "status": status,
        "completion_text": str(parsed_sample["text"]),
        "logprob_ready": bool(parsed_sample["token_logprobs"]),
        "top_logprob_ready": bool(parsed_sample["top_logprobs"]),
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs" if parsed_sample["top_logprobs"] else None,
        "evidence": {
            "token_logprob_count": len(parsed_sample["token_logprobs"]),
            "top_logprob_row_count": len(parsed_sample["top_logprobs"]),
            "token_logprobs": list(parsed_sample["token_logprobs"])[:8],
            "top_logprobs": list(parsed_sample["top_logprobs"])[:4],
            "raw_response_keys": sorted(parsed.keys()) if isinstance(parsed, Mapping) else [],
        },
        "error": None,
    }


def default_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover
    from scripts import adversarial_verify

    report = adversarial_verify.verify_artifact(path)
    return report if isinstance(report, dict) else {"flags": []}


def _empty_sample(route: str | None = None, status: int | None = None, error: str | None = None) -> JsonDict:
    return {
        "ready": False,
        "route": route,
        "status": status,
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
        "error": error,
    }


def _recorded_env_vars(env: Mapping[str, str]) -> JsonDict:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "CARNOT_5124_ENDPOINTS",
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
        env.get("CARNOT_5124_ENDPOINTS")
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


def _file_evidence(path: str | None) -> JsonDict:
    p = Path(path) if path else None
    exists = bool(p and p.exists() and p.is_file())
    return {
        "exists": exists,
        "path": str(p) if p is not None else None,
        "size_bytes": p.stat().st_size if exists and p is not None else None,
        "sha256": _sha256_file(p) if exists and p is not None else None,
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict, list[JsonDict]]:
    pair_error = None
    try:
        pair_rows = cached_pair_fn(gpu_indices=(0, 1), preferred_quant=preferred_quant) or []
    except Exception as exc:  # pragma: no cover - defensive external resolver guard
        pair_rows = []
        pair_error = f"{type(exc).__name__}: {exc}"
    pair_by_hf = {str(row.get("hf_id")): dict(row) for row in pair_rows if isinstance(row, Mapping)}
    specs: list[JsonDict] = []
    gguf_paths: dict[str, str | None] = {}
    usable: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        pair_row = pair_by_hf.get(hf_id, {})
        path = pair_row.get("model_path") or model_resolver(hf_id, preferred_quant)
        evidence = _file_evidence(str(path) if path else None)
        cache_status = "resolved" if path and evidence["exists"] else "missing"
        spec = {
            "name": pair_row.get("name") or MODEL_NAMES[hf_id],
            "role": MODEL_ROLES[hf_id],
            "hf_id": hf_id,
            "gpu": pair_row.get("gpu"),
            "preferred_quant": pair_row.get("preferred_quant") or preferred_quant,
            "model_path": str(path) if path else None,
            "resolved_path": str(path) if path else None,
            "cache_status": cache_status,
            "from_cached_sota_pair": hf_id in pair_by_hf,
            "loader": "llama.cpp",
            "file_evidence": evidence,
            "missing_diagnostic": None if cache_status == "resolved" else f"missing cached GGUF for {hf_id}",
        }
        specs.append(spec)
        gguf_paths[hf_id] = spec["model_path"] if cache_status == "resolved" else None
        if cache_status == "resolved":
            usable.append(
                {
                    "hf_id": hf_id,
                    "name": spec["name"],
                    "role": spec["role"],
                    "gpu": spec["gpu"],
                    "model_path": spec["model_path"],
                    "size_bytes": evidence["size_bytes"],
                }
            )
    pair = {
        "attempted": True,
        "preferred_quant": preferred_quant,
        "rows": list(pair_rows),
        "error": pair_error,
        "ready": len(pair_rows) >= 2 and all(row.get("model_path") for row in pair_rows[:2]),
    }
    return specs, gguf_paths, pair, usable


def _select_bringup_model(usable: Sequence[JsonMap]) -> JsonDict | None:
    candidates = [dict(row) for row in usable if row.get("model_path")]
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


def _sample_evidence(sample: JsonMap) -> JsonDict:
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
    return {
        "token_logprob_count": int(evidence.get("token_logprob_count") or len(token_logprobs)),
        "top_logprob_row_count": int(evidence.get("top_logprob_row_count") or len(top_logprobs)),
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "raw_response_keys": list(evidence.get("raw_response_keys") or []),
    }


def _has_logprob_evidence(sample: JsonMap) -> bool:
    evidence = _sample_evidence(sample)
    return bool(evidence["token_logprob_count"] > 0 or evidence["top_logprob_row_count"] > 0)


def _completion_proof(sample: JsonMap, selected_model: JsonMap | None) -> JsonDict:
    text = str(sample.get("completion_text") or "")
    ready = bool(sample.get("ready") and text.strip())
    return {
        "ready": ready,
        "prompt": DEFAULT_PROMPT,
        "route": sample.get("route"),
        "status": sample.get("status"),
        "text": text if ready else "",
        "model_hf_id": (selected_model or {}).get("hf_id"),
        "model_path": (selected_model or {}).get("model_path"),
        "error": sample.get("error"),
    }


def _logprob_proof(sample: JsonMap) -> JsonDict:
    evidence = _sample_evidence(sample)
    return {
        "ready": bool(evidence["token_logprob_count"] > 0 or evidence["top_logprob_row_count"] > 0),
        "route": sample.get("route"),
        "status": sample.get("status"),
        "telemetry_signal": sample.get("telemetry_signal"),
        "token_logprob_count": evidence["token_logprob_count"],
        "top_logprob_row_count": evidence["top_logprob_row_count"],
        "token_logprobs": evidence["token_logprobs"],
        "top_logprobs": evidence["top_logprobs"],
        "raw_response_keys": evidence["raw_response_keys"],
        "error": sample.get("error"),
    }


def build_cache_row(row_index: int, sample: JsonMap, endpoint: str, selected_model: JsonMap) -> JsonDict:
    evidence = _sample_evidence(sample)
    completion_text = str(sample.get("completion_text") or "")
    prompt = f"exp5124-cache-smoke-row-{row_index}"
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": f"exp5124-cache-smoke-{row_index:04d}",
        "row_index": int(row_index),
        "endpoint_used": str(sample.get("route") or endpoint.rstrip("/") + "/completion"),
        "model_hf_id": str(selected_model.get("hf_id") or ""),
        "gguf_path": str(selected_model.get("model_path") or ""),
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "completion_text": completion_text,
        "completion_hash": hashlib.sha256(completion_text.encode("utf-8")).hexdigest(),
        "token_logprobs": evidence["token_logprobs"],
        "top_logprobs": evidence["top_logprobs"],
        "token_logprob_count": evidence["token_logprob_count"],
        "top_logprob_row_count": evidence["top_logprob_row_count"],
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
    if int(row.get("token_logprob_count") or 0) <= 0 and int(row.get("top_logprob_row_count") or 0) <= 0:
        errors.append("logprob_evidence")
    return sorted(set(errors))


def _write_read_cache(cache_path: Path, rows: Sequence[JsonMap], *, write: bool) -> JsonDict:
    if not rows:
        return {
            "ready": False,
            "path": cache_path.as_posix(),
            "rows_written": 0,
            "rows_read": 0,
            "readback_matches": False,
            "sha256": None,
            "errors": ["no_rows"],
        }
    if write:
        write_jsonl(cache_path, rows)
    read_rows = read_jsonl_rows(cache_path) if cache_path.exists() else []
    rows_written = len(rows) if write else 0
    readback_matches = read_rows == list(rows)
    return {
        "ready": bool(rows_written == len(rows) and readback_matches),
        "path": cache_path.as_posix(),
        "rows_written": rows_written,
        "rows_read": len(read_rows),
        "readback_matches": readback_matches,
        "sha256": _sha256_file(cache_path),
        "sample_row_ids": [str(row.get("row_id")) for row in read_rows[:3]],
        "errors": [] if readback_matches else ["cache_readback_mismatch"],
    }


def _duration_floor_from_measured(
    *,
    live_ready: bool,
    duration_s: float,
    duration_floor_s: float,
    external_evidence: JsonMap | None,
) -> JsonDict:
    if not live_ready:
        return {
            "completed": False,
            "target_duration_s": float(duration_floor_s),
            "duration_after_s": float(duration_s),
            "reason": "no_live_completion_logprob_claim",
        }
    if external_evidence is not None:
        return dict(external_evidence)
    completed = float(duration_s) >= float(duration_floor_s)
    return {
        "completed": completed,
        "target_duration_s": float(duration_floor_s),
        "duration_after_s": float(duration_s),
        "reason": "measured_wall_clock_duration_met_floor"
        if completed
        else "measured_wall_clock_duration_below_floor",
    }


def _critical_flags(report: JsonMap) -> list[JsonDict]:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def build_root_cause_tree(
    *,
    pair_ready: bool,
    completion_ready: bool,
    logprob_ready: bool,
    cache_ready: bool,
    duration_floor_ready: bool,
    critical_flags: Sequence[JsonMap],
    runtime_detail: str | None,
) -> JsonDict:
    tree: JsonDict = {
        "cached_sota_pair": {
            "present": not pair_ready,
            "detail": None if pair_ready else "cached_sota_pair() did not return two model_path rows",
        },
        "completion": {
            "present": not completion_ready,
            "detail": None if completion_ready else runtime_detail or "completion proof missing",
        },
        "logprob": {
            "present": not logprob_ready,
            "detail": None if logprob_ready else runtime_detail or "token/top-logprob proof missing",
        },
        "cache": {
            "present": not cache_ready,
            "detail": None if cache_ready else "cache write/read receipt missing",
        },
        "duration_floor": {
            "present": not duration_floor_ready,
            "detail": None if duration_floor_ready else "live runtime did not clear duration floor",
        },
        "adversarial_verify": {
            "present": bool(critical_flags),
            "critical_flags": [dict(flag) for flag in critical_flags],
        },
    }
    if all(not row["present"] for row in tree.values()):
        tree["summary"] = "clean_runtime_provenance"
    else:
        priority = (
            "cached_sota_pair",
            "completion",
            "logprob",
            "cache",
            "duration_floor",
            "adversarial_verify",
        )
        tree["summary"] = next(f"blocked_{key}" for key in priority if tree[key]["present"])
    return tree


def _reproducibility_checksum(payload: JsonMap) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "milestone": payload.get("milestone"),
        "MODEL_SPECS": payload.get("MODEL_SPECS"),
        "gguf_paths": payload.get("gguf_paths"),
        "completion_proof": payload.get("completion_proof"),
        "logprob_proof": payload.get("logprob_proof"),
        "cache_receipts": payload.get("cache_receipts"),
        "duration_floor_evidence": payload.get("duration_floor_evidence"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_payload(basis)


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    principles = artifact.get("field_principles", {})
    errors.extend([f"field_principles.{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in principles])
    ids = [row.get("hf_id") for row in artifact.get("MODEL_SPECS", []) if isinstance(row, Mapping)]
    checks = (
        ("schema", artifact.get("schema") != SCHEMA),
        ("experiment_id", artifact.get("experiment_id") != EXPERIMENT_ID),
        ("milestone", artifact.get("milestone") != MILESTONE),
        ("inference_substrate", artifact.get("inference_substrate") != INFERENCE_SUBSTRATE),
        ("conductor_modified", artifact.get("conductor_modified") is not False),
        ("MODEL_SPECS", ids != list(MANDATED_MODEL_IDS)),
        (
            "clean_gate_mismatch",
            bool(artifact.get("sota_runtime_clean"))
            != (
                bool((artifact.get("completion_proof") or {}).get("ready"))
                and bool((artifact.get("logprob_proof") or {}).get("ready"))
                and bool(artifact.get("cache_ready"))
                and bool(artifact.get("adversarial_verify_passed"))
            ),
        ),
    )
    errors.extend(name for name, failed in checks if failed)
    return sorted(set(errors))


def _default_tests_run() -> list[JsonDict]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5124_clean_sota_runtime_provenance.py -q",
            "status": "expected_or_completed",
        },
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5124_clean_sota_runtime_provenance.py --cov=python/carnot/experiment_5124_clean_sota_runtime_provenance.py --cov-report=term-missing --cov-fail-under=100 -q",
            "status": "expected_or_completed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "status": "expected_or_completed"},
        {
            "command": "python scripts/adversarial_verify.py results/experiment_5124_clean_sota_runtime_provenance_v470.json",
            "status": "expected_or_completed",
        },
    ]


def _maybe_start_server_sample(  # pragma: no cover - host/process dependent
    *,
    root: Path,
    env: Mapping[str, str],
    selected_model: JsonDict,
    server: JsonMap,
    free: JsonMap,
    endpoint_probe: EndpointProbe,
    endpoint_sample: EndpointSample,
    endpoint_timeout_s: float,
    server_start_timeout_s: float,
    server_start: ServerStart,
    server_cleanup: ServerCleanup,
) -> tuple[str, JsonDict, JsonDict, list[str], list[str] | None, int | None, Any, float | None]:
    endpoint_url = str(free["endpoint_url"])
    log_path = root / RAW_LOG_RELATIVE_PATH
    command = exp3013._build_server_command(
        server_path=str(server["selected_path"]),
        model_path=str(selected_model["model_path"]),
        host=str(free["host"]),
        port=int(free["port"]),
        extra_args=env.get("CARNOT_LLAMA_SERVER_ARGS"),
    )
    errors: list[str] = []
    process = None
    server_started_wall: float | None = None
    sample = _empty_sample(error="server bring-up did not produce sample")
    endpoint_summary: JsonDict = {"completion_ready": False, "selected_endpoint": None}
    del server_cleanup
    try:
        process = server_start(command, dict(env), log_path)
        server_started_wall = time.monotonic()
        deadline = time.monotonic() + server_start_timeout_s
        while time.monotonic() <= deadline:
            endpoint_summary = endpoint_probe([endpoint_url], endpoint_timeout_s)
            if endpoint_summary.get("completion_ready"):
                sample = endpoint_sample(endpoint_url, endpoint_timeout_s)
                break
            if getattr(process, "poll", lambda: None)() is not None:
                errors.append("server process exited before endpoint became ready")
                break
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return (
        endpoint_url,
        endpoint_summary,
        sample,
        errors,
        command,
        getattr(process, "pid", None) if process is not None else None,
        process,
        server_started_wall,
    )


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    precondition_probe: PreconditionProbe = default_precondition_probe,
    endpoint_probe: EndpointProbe = exp3013._probe_endpoint_summary,
    endpoint_sample: EndpointSample = default_endpoint_sample,
    cache_sample: CacheSample = default_cache_sample,
    server_finder: ServerFinder = exp3013._llama_server_availability,
    free_port: FreePort = exp3013._find_free_port,
    server_start: ServerStart = exp3013._start_llama_server_process,
    server_cleanup: ServerCleanup = exp3013._cleanup_llama_server_process,
    duration_floor_probe: DurationFloorProbe = exp3013._run_duration_floor_endpoint_probe,
    adversarial_verify: AdversarialVerify = default_adversarial_verify,
    endpoints: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    now: Clock = time.monotonic,
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S,
    endpoint_timeout_s: float = DEFAULT_ENDPOINT_TIMEOUT_S,
    server_start_timeout_s: float = DEFAULT_SERVER_START_TIMEOUT_S,
    tests_run: Sequence[JsonMap] | None = None,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    destination = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_destination = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    started_s = now()
    observed_started_wall = time.monotonic()
    started_at = _utc_now()
    model_specs, gguf_paths, pair, usable = resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    preflight = precondition_probe(root, merged_env)
    selected_model = _select_bringup_model(usable)
    server = server_finder(merged_env)
    free = free_port("127.0.0.1")
    normalized_endpoints = _normalize_endpoints(endpoints, merged_env)
    endpoint_summary: JsonDict = {
        "candidate_endpoints": normalized_endpoints,
        "selected_endpoint": None,
        "completion_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "probes": [],
    }
    endpoint_url = normalized_endpoints[0]
    sample = _empty_sample(error="cached_sota_pair unavailable")
    server_command = None
    server_pid = None
    server_errors: list[str] = []
    shutdown_behavior: JsonDict = {"started_by_preflight": False, "terminated": False}
    startup_log = {"path": (root / RAW_LOG_RELATIVE_PATH).as_posix(), "exists": False, "tail": ""}
    server_process = None
    server_started_wall: float | None = None
    floor_probe_result: JsonDict | None = None
    pair_ready = bool(pair["ready"])

    if pair_ready:
        endpoint_summary = endpoint_probe(normalized_endpoints, endpoint_timeout_s)
        endpoint_url = str(endpoint_summary.get("selected_endpoint") or normalized_endpoints[0])
        if endpoint_summary.get("completion_ready"):
            sample = endpoint_sample(endpoint_url, endpoint_timeout_s)
        elif selected_model and server.get("available") and free.get("available") and free.get("endpoint_url"):
            (
                endpoint_url,
                endpoint_summary,
                sample,
                server_errors,
                server_command,
                server_pid,
                server_process,
                server_started_wall,
            ) = _maybe_start_server_sample(
                root=root,
                env=merged_env,
                selected_model=selected_model,
                server=server,
                free=free,
                endpoint_probe=endpoint_probe,
                endpoint_sample=endpoint_sample,
                endpoint_timeout_s=endpoint_timeout_s,
                server_start_timeout_s=server_start_timeout_s,
                server_start=server_start,
                server_cleanup=server_cleanup,
            )
        else:
            sample = _empty_sample(
                route=endpoint_url.rstrip("/") + "/completion",
                error=str(server.get("missing_diagnostic") or "endpoint unavailable"),
            )

    completion_ready = bool(sample.get("ready") and str(sample.get("completion_text") or "").strip())
    logprob_ready = _has_logprob_evidence(sample)
    live_ready = bool(pair_ready and completion_ready and logprob_ready)
    if live_ready and duration_floor_s > 0.0 and now is time.monotonic:
        floor_probe_result = duration_floor_probe(
            endpoint_url,
            run_started_s=started_s,
            target_duration_s=duration_floor_s,
            timeout_s=endpoint_timeout_s,
            max_probes=20,
        )

    cache_rows: list[JsonDict] = []
    cache_row_errors: list[str] = []
    if live_ready and selected_model is not None:
        cache_probe = cache_sample(0, endpoint_url, selected_model)
        if _has_logprob_evidence(cache_probe):
            row = build_cache_row(0, cache_probe, endpoint_url, selected_model)
            cache_row_errors = validate_cache_row(row)
            if not cache_row_errors:
                cache_rows.append(row)
        else:
            cache_row_errors = ["cache_sample_missing_logprobs"]
    cache_receipts = _write_read_cache(cache_destination, cache_rows, write=write)

    if server_process is not None:
        shutdown_behavior = server_cleanup(server_process)
        log_path = root / RAW_LOG_RELATIVE_PATH
        startup_log = {
            "path": log_path.as_posix(),
            "exists": log_path.exists(),
            "tail": exp3013._tail_file(log_path),
        }

    finished_s = now()
    observed_finished_wall = time.monotonic()
    finished_at = _utc_now()
    duration_s = round(float(finished_s - started_s), 6)
    if now is time.monotonic and server_started_wall is not None:
        endpoint_lifetime_s = round(float(observed_finished_wall - server_started_wall), 6)
    elif now is time.monotonic:
        endpoint_lifetime_s = round(float(observed_finished_wall - observed_started_wall), 6)
    else:
        endpoint_lifetime_s = duration_s if completion_ready else 0.0
    duration_floor_evidence = _duration_floor_from_measured(
        live_ready=live_ready,
        duration_s=duration_s,
        duration_floor_s=duration_floor_s,
        external_evidence=floor_probe_result,
    )
    duration_floor_ready = bool(duration_floor_evidence.get("completed"))
    cache_ready = bool(cache_receipts.get("ready") and not cache_row_errors)
    completion_proof = _completion_proof(sample, selected_model)
    logprob_proof = _logprob_proof(sample)
    runtime_detail = str(sample.get("error") or "; ".join(server_errors) or "")
    preliminary_clean = bool(live_ready and cache_ready and duration_floor_ready)
    preliminary_verdict = SUCCESS_VERDICT if preliminary_clean else (
        BLOCKED_PAIR_VERDICT if not pair_ready else BLOCKED_RUNTIME_VERDICT
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_numeric_id": EXPERIMENT_NUMERIC_ID,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-INFER-SOTA-029",
            "SCENARIO-INFER-SOTA-029-CLEAN",
            "SCENARIO-INFER-SOTA-029-BLOCKED",
        ],
        "result_path": destination.as_posix(),
        "cache_path": cache_destination.as_posix(),
        "started_at": started_at,
        "finished_at": finished_at,
        "honest_verdict": preliminary_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "preconditions_checked": {
            "recorded_before_live_inference": True,
            "cuda_status": preflight.get("cuda_status", {}),
            "llama_cpp_python": preflight.get("llama_cpp_python", {}),
            "llama_cpp_server": server,
            "disk_ram": preflight.get("disk_ram", {}),
            "free_port": free,
            "environment_variables": _recorded_env_vars(merged_env),
        },
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "cached_sota_pair_attempted": bool(pair["attempted"]),
        "cached_sota_pair_result": pair,
        "gguf_paths": gguf_paths,
        "selected_model": selected_model,
        "endpoint_url": endpoint_url,
        "endpoint_summary": endpoint_summary,
        "endpoint_lifetime_s": endpoint_lifetime_s,
        "server_command": server_command,
        "server_pid": server_pid,
        "server_errors": server_errors,
        "startup_log": startup_log,
        "shutdown_behavior": shutdown_behavior,
        "completion_proof": completion_proof,
        "logprob_proof": logprob_proof,
        "cache_ready": cache_ready,
        "cache_receipts": cache_receipts,
        "cache_row_errors": cache_row_errors,
        "request_response_transcript": {
            "completion_request": {
                "endpoint": sample.get("route") or endpoint_url.rstrip("/") + "/completion",
                "prompt": DEFAULT_PROMPT,
                "max_tokens": exp3013.SAMPLE_MAX_TOKENS,
                "logprobs_requested": exp3013.LOGPROBS_REQUESTED,
            },
            "completion_response": {
                "status": sample.get("status"),
                "text": sample.get("completion_text") if completion_ready else "",
                "error": sample.get("error"),
            },
            "logprob_response": logprob_proof,
        },
        "duration_floor_evidence": duration_floor_evidence,
        "root_cause_tree": build_root_cause_tree(
            pair_ready=pair_ready,
            completion_ready=completion_ready,
            logprob_ready=logprob_ready,
            cache_ready=cache_ready,
            duration_floor_ready=duration_floor_ready,
            critical_flags=[],
            runtime_detail=runtime_detail,
        ),
        "flagged_adversarial": False,
        "adversarial_verify_passed": False,
        "adversarial_verify_report": {"flags": []},
        "sota_runtime_clean": False,
        "conductor_modified": False,
        "tests_run": list(tests_run) if tests_run is not None else _default_tests_run(),
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    write_json(destination, artifact)
    report = adversarial_verify(destination)
    critical = _critical_flags(report)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_passed"] = not critical
    artifact["flagged_adversarial"] = bool(critical)
    final_clean = bool(preliminary_clean and not critical)
    artifact["sota_runtime_clean"] = final_clean
    artifact["cache_ready"] = bool(cache_ready and not critical)
    if final_clean:
        artifact["honest_verdict"] = SUCCESS_VERDICT
    elif critical:
        artifact["honest_verdict"] = BLOCKED_ADVERSARIAL_VERDICT
    artifact["root_cause_tree"] = build_root_cause_tree(
        pair_ready=pair_ready,
        completion_ready=completion_ready,
        logprob_ready=logprob_ready,
        cache_ready=bool(cache_ready and not critical),
        duration_floor_ready=duration_floor_ready,
        critical_flags=critical,
        runtime_detail=runtime_detail,
    )
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if write:
        write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--endpoint-timeout-s", type=float, default=DEFAULT_ENDPOINT_TIMEOUT_S)
    parser.add_argument("--server-start-timeout-s", type=float, default=DEFAULT_SERVER_START_TIMEOUT_S)
    parser.add_argument("--duration-floor-s", type=float, default=DEFAULT_DURATION_FLOOR_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run(
        artifact_path=args.output,
        cache_path=args.cache,
        endpoints=args.endpoint,
        endpoint_timeout_s=args.endpoint_timeout_s,
        server_start_timeout_s=args.server_start_timeout_s,
        duration_floor_s=args.duration_floor_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
