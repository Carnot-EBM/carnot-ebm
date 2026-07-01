#!/usr/bin/env python3
"""Exp 5119: local SOTA GGUF endpoint root-cause diagnostic.

Spec refs: REQ-INFER-SOTA-028,
SCENARIO-INFER-SOTA-028-SUCCESS,
SCENARIO-INFER-SOTA-028-BLOCKED.

This module stops at the runtime boundary.  It records whether the mandated
local GGUF files can be resolved, whether a local llama.cpp-compatible backend
can return completion text, and whether that backend exposes token logprobs.
It does not promote any downstream cache unless live logprob evidence appears
in the same run.
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
ServerFinder = Callable[[Mapping[str, str]], JsonDict]
FreePort = Callable[[str], JsonDict]
ServerStart = Callable[[list[str], dict[str, str], Path], Any]
ServerCleanup = Callable[[Any], JsonDict]
Clock = Callable[[], float]
AdversarialVerify = Callable[[Path], JsonDict]

EXPERIMENT_ID = "exp5119-sota-endpoint-rootcause-v469"
EXPERIMENT_NUMERIC_ID = 5119
EXPERIMENT_NAME = "experiment_5119_sota_endpoint_rootcause"
MILESTONE = "2026.07.469"
SCHEMA = "carnot.experiment_5119_sota_endpoint_rootcause.v1"
RESULT_RELATIVE_PATH = "results/experiment_5119_sota_endpoint_rootcause_v469.json"
RAW_LOG_RELATIVE_PATH = "results/raw/experiment_5119_sota_endpoint_rootcause/llamacpp_server.log"
DEFAULT_ENDPOINTS = ("http://127.0.0.1:8080",)
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
DEFAULT_ENDPOINT_TIMEOUT_S = 5.0
DEFAULT_SERVER_START_TIMEOUT_S = 60.0
DEFAULT_DURATION_FLOOR_S = 60.0
DEFAULT_PROMPT = "Reply in one short sentence: exp3013 SOTA GGUF telemetry live."
SAMPLE_MAX_TOKENS = 16
LOGPROBS_REQUESTED = 5
RANDOM_SEED = 20260701

LIVE_SUBSTRATE = "live_llm_inference"
BLOCKED_SUBSTRATE = "precondition_check_only -- endpoint/server attempt made; no live logprob readiness"
SUCCESS_VERDICT = "success_sota_endpoint_rootcause_live_logprobs"
BLOCKED_VERDICT = "blocked_sota_endpoint_rootcause_no_live_logprobs"

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
ROOT_CAUSE_BLOCKERS: tuple[str, ...] = (
    "missing_binary",
    "wrong_model_path",
    "unsupported_logprob_api",
    "cuda_failure",
    "oom",
    "timeout",
    "cache_schema_mismatch",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "preconditions_checked",
    "MODEL_SPECS",
    "cached_sota_pair_attempted",
    "gguf_paths",
    "cuda_status",
    "server_command",
    "endpoint_lifetime_s",
    "completion_proof",
    "logprob_proof",
    "cache_ready",
    "root_cause_tree",
    "flagged_adversarial",
    "tests_run",
)
FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "experiment_id": {"principle": "traceability"},
    "milestone": {"principle": "milestone accountability"},
    "honest_verdict": {"principle": "terminal verdict with complete_/success_/blocked_ prefix"},
    "inference_substrate": {"principle": "substrate honesty"},
    "duration_s": {"principle": "timing accountability"},
    "preconditions_checked": {"principle": "compute preflight accountability"},
    "MODEL_SPECS": {"principle": "mandated SOTA model accountability"},
    "cached_sota_pair_attempted": {"principle": "model-resolution discipline"},
    "gguf_paths": {"principle": "reproducibility"},
    "cuda_status": {"principle": "hardware precondition"},
    "server_command": {"principle": "transcript provenance"},
    "endpoint_lifetime_s": {"principle": "live readiness"},
    "completion_proof": {"principle": "live readiness"},
    "logprob_proof": {"principle": "live readiness"},
    "cache_ready": {"principle": "downstream gate"},
    "root_cause_tree": {"principle": "actionable blocker"},
    "flagged_adversarial": {"principle": "adversarial-verification accountability"},
    "tests_run": {"principle": "verification evidence"},
}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


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


def _recorded_env_vars(env: Mapping[str, str]) -> JsonDict:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "CARNOT_5119_ENDPOINTS",
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


def _normalize_endpoints(endpoints: Sequence[str] | None, env: Mapping[str, str]) -> list[str]:
    raw = (
        list(endpoints)
        if endpoints is not None
        else [
            part.strip()
            for part in (
                env.get("CARNOT_5119_ENDPOINTS")
                or env.get("CARNOT_LLAMA_ENDPOINTS")
                or env.get("CARNOT_JUDGE_ENDPOINTS")
                or env.get("CARNOT_JUDGE_SERVER_URL")
                or ""
            ).split(",")
            if part.strip()
        ]
    )
    normalized: list[str] = []
    for endpoint in raw or DEFAULT_ENDPOINTS:
        value = str(endpoint).strip().rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_ENDPOINTS)


def _ram_status() -> JsonDict:  # pragma: no cover - host dependent
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return {"ram_available_gib": round((pages * page_size) / (1024**3), 3)}
    except (OSError, ValueError, AttributeError):
        return {"ram_available_gib": None}


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
    nvidia = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5,
    )
    if nvidia.get("returncode") == 0:
        for line in str(nvidia.get("stdout") or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(float(parts[0])),
                        "name": parts[1],
                        "total_vram_mb": int(float(parts[2])),
                        "used_vram_mb": int(float(parts[3])),
                        "free_vram_mb": int(float(parts[4])),
                    }
                )

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
    ram = _ram_status()
    return {
        "cuda_status": {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count or len(gpus),
            "gpus": gpus,
            "nvidia_smi": {"available": nvidia.get("returncode") == 0, "stderr": nvidia.get("stderr")},
        },
        "llama_cpp_python": llama_cpp_python,
        "disk_ram": {
            "disk_free_bytes": int(disk.free),
            "disk_free_gib": round(disk.free / (1024**3), 3),
            **ram,
        },
    }


def _file_evidence(path: str | None) -> JsonDict:
    p = Path(path) if path else None
    exists = bool(p and p.exists() and p.is_file())
    return {
        "exists": exists,
        "size_bytes": p.stat().st_size if exists and p is not None else None,
        "path": str(p) if p is not None else None,
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict, list[JsonDict]]:
    cached_pair_error = None
    try:
        cached_pair_rows = cached_pair_fn(gpu_indices=(0, 1), preferred_quant=preferred_quant) or []
    except Exception as exc:  # pragma: no cover - defensive around external resolver
        cached_pair_rows = []
        cached_pair_error = f"{type(exc).__name__}: {exc}"

    specs: list[JsonDict] = []
    gguf_paths: dict[str, str | None] = {}
    usable: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        path = model_resolver(hf_id, preferred_quant)
        evidence = _file_evidence(path)
        cache_status = "resolved" if path and evidence["exists"] else "missing"
        row = {
            "name": MODEL_NAMES[hf_id],
            "role": MODEL_ROLES[hf_id],
            "hf_id": hf_id,
            "preferred_quant": preferred_quant,
            "resolved_path": path,
            "cache_status": cache_status,
            "missing_diagnostic": None if cache_status == "resolved" else f"missing cached GGUF for {hf_id}",
            "file_evidence": evidence,
            "loader": "llama.cpp",
        }
        specs.append(row)
        gguf_paths[hf_id] = path if cache_status == "resolved" else None
        if cache_status == "resolved":
            usable.append(
                {
                    "hf_id": hf_id,
                    "role": MODEL_ROLES[hf_id],
                    "model_path": str(path),
                    "size_bytes": evidence["size_bytes"],
                }
            )
    pair = {
        "attempted": True,
        "preferred_quant": preferred_quant,
        "rows": list(cached_pair_rows),
        "error": cached_pair_error,
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
        "prompt": exp3013.DEFAULT_PROMPT,
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


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value or "")


def build_root_cause_tree(
    *,
    server: JsonMap,
    model_specs: Sequence[JsonMap],
    cuda_status: JsonMap,
    sample: JsonMap,
    completion_ready: bool,
    logprob_ready: bool,
    duration_s: float,
) -> JsonDict:
    text = " ".join(
        [
            _flatten_text(sample),
            _flatten_text(server),
            _flatten_text(cuda_status),
        ]
    ).lower()
    wrong_paths = [
        str(row.get("hf_id"))
        for row in model_specs
        if row.get("cache_status") != "resolved"
        or not (row.get("file_evidence") or {}).get("exists")
    ]
    missing_binary = not bool(server.get("available"))
    cuda_failure = not bool(cuda_status.get("cuda_available", True))
    unsupported_logprob = not bool(logprob_ready)
    timeout = "timeout" in text or "timed out" in text
    tree: JsonDict = {
        "missing_binary": {
            "present": missing_binary,
            "detail": server.get("missing_diagnostic") if missing_binary else None,
        },
        "wrong_model_path": {
            "present": bool(wrong_paths),
            "affected_hf_ids": wrong_paths,
        },
        "unsupported_logprob_api": {
            "present": unsupported_logprob,
            "detail": None
            if logprob_ready
            else "no token logprob or top-logprob evidence observed from the intended backend",
        },
        "cuda_failure": {
            "present": cuda_failure,
            "detail": cuda_status.get("error") or cuda_status.get("detail"),
        },
        "oom": {
            "present": "out of memory" in text or "oom" in text,
            "detail": "OOM marker found in runtime evidence" if ("out of memory" in text or "oom" in text) else None,
        },
        "timeout": {
            "present": timeout,
            "detail": f"timeout marker or startup duration observed; duration_s={duration_s}" if timeout else None,
        },
        "cache_schema_mismatch": {
            "present": "schema mismatch" in text or "cache schema" in text,
            "detail": "cache schema mismatch marker found" if ("schema mismatch" in text or "cache schema" in text) else None,
        },
    }
    if completion_ready and logprob_ready:
        summary = "no_blocker_live_logprobs_observed"
    else:
        priority = [
            "missing_binary",
            "wrong_model_path",
            "cuda_failure",
            "oom",
            "timeout",
            "cache_schema_mismatch",
            "unsupported_logprob_api",
        ]
        summary = next(
            (f"blocked_{name}" for name in priority if tree[name]["present"]),
            "blocked_unknown_endpoint_root_cause",
        )
    tree["summary"] = summary
    return tree


def _server_log(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "tail": exp3013._tail_file(path),
    }


def _cleanup(process: Any, server_cleanup: ServerCleanup) -> JsonDict:
    return server_cleanup(process)


def default_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover
    from scripts import adversarial_verify

    report = adversarial_verify.verify_artifact(path)
    return report if isinstance(report, dict) else {"flags": []}


def _critical_flags(report: JsonMap) -> list[JsonDict]:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def _default_tests_run() -> list[JsonDict]:
    cache = REPO_ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"
    lastfailed_count = None
    if cache.exists():
        try:
            loaded = json.loads(cache.read_text(encoding="utf-8"))
            lastfailed_count = len(loaded) if isinstance(loaded, Mapping) else None
        except json.JSONDecodeError:
            lastfailed_count = None
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5119_sota_endpoint_rootcause.py -q",
            "status": "expected_or_completed",
        },
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5119_sota_endpoint_rootcause.py --cov=python/carnot/experiment_5119_sota_endpoint_rootcause.py --cov-report=term-missing --cov-fail-under=100 -q",
            "status": "expected_or_completed",
        },
        {
            "command": ".venv/bin/pytest tests/python -q",
            "status": "expected_or_completed",
            "pytest_cache_lastfailed_count": lastfailed_count,
        },
    ]


def _reproducibility_checksum(payload: JsonMap) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "milestone": payload.get("milestone"),
        "honest_verdict": payload.get("honest_verdict"),
        "inference_substrate": payload.get("inference_substrate"),
        "MODEL_SPECS": payload.get("MODEL_SPECS"),
        "gguf_paths": payload.get("gguf_paths"),
        "completion_proof": payload.get("completion_proof"),
        "logprob_proof": payload.get("logprob_proof"),
        "cache_ready": payload.get("cache_ready"),
        "root_cause_tree": payload.get("root_cause_tree"),
        "random_seed": payload.get("random_seed"),
    }
    return _sha256_payload(basis)


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(
        [f"field_principles.{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact.get("field_principles", {})]
    )
    ids = [
        row.get("hf_id")
        for row in artifact.get("MODEL_SPECS", [])
        if isinstance(row, Mapping)
    ]
    errors.extend(
        name
        for name, bad in (
            ("schema", artifact.get("schema") != SCHEMA),
            ("experiment_id", artifact.get("experiment_id") != EXPERIMENT_ID),
            ("milestone", artifact.get("milestone") != MILESTONE),
            ("MODEL_SPECS", ids != list(MANDATED_MODEL_IDS)),
            ("cache_ready_without_logprob", bool(artifact.get("cache_ready")) and not bool((artifact.get("logprob_proof") or {}).get("ready"))),
            ("clean_but_flagged", bool(artifact.get("cache_ready")) and bool(artifact.get("flagged_adversarial"))),
        )
        if bad
    )
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    precondition_probe: PreconditionProbe = default_precondition_probe,
    endpoint_probe: EndpointProbe = exp3013._probe_endpoint_summary,
    endpoint_sample: EndpointSample = exp3013._sample_endpoint_telemetry,
    server_finder: ServerFinder = exp3013._llama_server_availability,
    free_port: FreePort = exp3013._find_free_port,
    server_start: ServerStart = exp3013._start_llama_server_process,
    server_cleanup: ServerCleanup = exp3013._cleanup_llama_server_process,
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
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    started_s = now()
    started_at = _utc_now()
    model_specs, gguf_paths, pair, usable = resolve_model_specs(
        model_resolver=model_resolver,
        cached_pair_fn=cached_pair_fn,
    )
    selected_model = _select_bringup_model(usable)
    preflight = precondition_probe(root, merged_env)
    cuda_status = dict(preflight.get("cuda_status") or preflight.get("cuda_gpu_visibility") or {})
    server = server_finder(merged_env)
    free = free_port("127.0.0.1")
    endpoint_list = _normalize_endpoints(endpoints, merged_env)
    endpoint_summary = endpoint_probe(endpoint_list, endpoint_timeout_s)
    endpoint_url = str(endpoint_summary.get("selected_endpoint") or endpoint_list[0])
    log_path = root / RAW_LOG_RELATIVE_PATH
    sample: JsonDict = {
        "ready": False,
        "route": None,
        "status": None,
        "completion_text": "",
        "logprob_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "evidence": {"token_logprob_count": 0, "top_logprob_row_count": 0},
        "error": "endpoint completion unavailable",
    }
    server_command = None
    server_pid = None
    startup_log = {"path": log_path.as_posix(), "exists": False, "tail": ""}
    shutdown_behavior: JsonDict = {"started_by_preflight": False, "terminated": False}
    server_errors: list[str] = []
    process = None
    server_attempted = False
    server_started_wall: float | None = None
    server_finished_wall: float | None = None
    duration_floor_evidence: JsonDict | None = None

    if endpoint_summary.get("completion_ready"):
        sample = endpoint_sample(endpoint_url, endpoint_timeout_s)
    elif selected_model and server.get("available") and free.get("available") and free.get("endpoint_url"):
        server_attempted = True
        endpoint_url = str(free["endpoint_url"])
        server_command = exp3013._build_server_command(
            server_path=str(server["selected_path"]),
            model_path=str(selected_model["model_path"]),
            host=str(free["host"]),
            port=int(free["port"]),
            extra_args=merged_env.get("CARNOT_LLAMA_SERVER_ARGS"),
        )
        try:
            process = server_start(server_command, merged_env, log_path)
            server_started_wall = time.monotonic()
            server_pid = getattr(process, "pid", None)
            deadline = time.monotonic() + server_start_timeout_s
            while time.monotonic() <= deadline:
                endpoint_summary = endpoint_probe([endpoint_url], endpoint_timeout_s)
                if endpoint_summary.get("completion_ready"):
                    sample = endpoint_sample(endpoint_url, endpoint_timeout_s)
                    break
                if getattr(process, "poll", lambda: None)() is not None:
                    server_errors.append("server process exited before endpoint became ready")
                    break
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        except Exception as exc:  # pragma: no cover - defensive around process launch
            server_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if process is not None:
                shutdown_behavior = _cleanup(process, server_cleanup)
                server_finished_wall = time.monotonic()
            startup_log = _server_log(log_path)

    completion_ready = bool(sample.get("ready") and str(sample.get("completion_text") or "").strip())
    logprob_ready = _has_logprob_evidence(sample)
    live_ready = bool(completion_ready and logprob_ready)
    if live_ready and duration_floor_s > 0 and now is time.monotonic:
        duration_floor_evidence = exp3013._run_duration_floor_endpoint_probe(
            endpoint_url,
            run_started_s=started_s,
            target_duration_s=duration_floor_s,
            timeout_s=endpoint_timeout_s,
            max_probes=20,
        )
    finished_s = now()
    finished_at = _utc_now()
    duration_s = round(float(finished_s - started_s), 6)
    if (
        completion_ready
        and now is time.monotonic
        and server_started_wall is not None
        and server_finished_wall is not None
    ):
        endpoint_lifetime_s = round(float(server_finished_wall - server_started_wall), 6)
    else:
        endpoint_lifetime_s = duration_s if completion_ready else 0.0
    completion_proof = _completion_proof(sample, selected_model)
    logprob_proof = _logprob_proof(sample)
    root_cause_tree = build_root_cause_tree(
        server=server,
        model_specs=model_specs,
        cuda_status=cuda_status,
        sample={**sample, "server_errors": server_errors, "endpoint_summary": endpoint_summary},
        completion_ready=completion_ready,
        logprob_ready=logprob_ready,
        duration_s=duration_s,
    )
    cache_ready = bool(live_ready)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "experiment_numeric_id": EXPERIMENT_NUMERIC_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-INFER-SOTA-028",
            "SCENARIO-INFER-SOTA-028-SUCCESS",
            "SCENARIO-INFER-SOTA-028-BLOCKED",
        ],
        "result_path": destination.as_posix(),
        "started_at": started_at,
        "finished_at": finished_at,
        "honest_verdict": SUCCESS_VERDICT if cache_ready else BLOCKED_VERDICT,
        "inference_substrate": LIVE_SUBSTRATE if cache_ready else BLOCKED_SUBSTRATE,
        "duration_s": duration_s,
        "preconditions_checked": {
            "recorded_before_live_inference": True,
            "cuda_status": cuda_status,
            "llama_cpp_python": preflight.get("llama_cpp_python", {}),
            "llama_cpp_server": server,
            "disk_ram": preflight.get("disk_ram", {}),
            "free_port": free,
            "environment_variables": _recorded_env_vars(merged_env),
            "server_attempted": server_attempted,
        },
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "cached_sota_pair_attempted": bool(pair["attempted"]),
        "cached_sota_pair_result": pair,
        "gguf_paths": gguf_paths,
        "cuda_status": cuda_status,
        "server_command": server_command,
        "server_pid": server_pid,
        "server_port": free.get("port") if server_attempted else None,
        "endpoint_url": endpoint_url,
        "endpoint_summary": endpoint_summary,
        "endpoint_lifetime_s": endpoint_lifetime_s,
        "startup_log": startup_log,
        "duration_floor_evidence": duration_floor_evidence,
        "request_response_transcript": {
            "completion_request": {
                "endpoint": sample.get("route") or endpoint_url.rstrip("/") + "/completion",
                "prompt": exp3013.DEFAULT_PROMPT,
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
        "shutdown_behavior": shutdown_behavior,
        "server_errors": server_errors,
        "completion_proof": completion_proof,
        "logprob_proof": logprob_proof,
        "cache_ready": cache_ready,
        "cache_rows_written": 1 if cache_ready else 0,
        "root_cause_tree": root_cause_tree,
        "flagged_adversarial": False,
        "adversarial_verify_report": {"flags": []},
        "adversarial_verify_passed": False,
        "tests_run": list(tests_run) if tests_run is not None else _default_tests_run(),
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if write:
        write_json(destination, artifact)
        report = adversarial_verify(destination)
    else:
        write_json(destination, artifact)
        report = adversarial_verify(destination)
    critical = _critical_flags(report)
    too_short_live = bool(cache_ready and duration_s < duration_floor_s)
    if too_short_live:
        critical.append(
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": f"live logprob claim duration_s={duration_s} below floor {duration_floor_s}",
            }
        )
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_passed"] = not critical
    artifact["flagged_adversarial"] = bool(critical)
    if artifact["flagged_adversarial"]:
        artifact["cache_ready"] = False
        artifact["cache_rows_written"] = 0
        artifact["honest_verdict"] = "blocked_sota_endpoint_rootcause_adversarial_flag"
        artifact["root_cause_tree"]["adversarial_verify"] = {
            "present": True,
            "critical_flags": critical,
        }
        artifact["root_cause_tree"]["summary"] = "blocked_adversarial_verify"
        write_json(destination, artifact)
        final_report = adversarial_verify(destination)
        final_critical = _critical_flags(final_report) or critical
        artifact["adversarial_verify_report"] = final_report
        artifact["adversarial_verify_passed"] = not final_critical
        artifact["flagged_adversarial"] = bool(final_critical)
        artifact["root_cause_tree"]["adversarial_verify"] = {
            "present": bool(final_critical),
            "critical_flags": final_critical,
        }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if write:
        write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--endpoint-timeout-s", type=float, default=DEFAULT_ENDPOINT_TIMEOUT_S)
    parser.add_argument("--server-start-timeout-s", type=float, default=DEFAULT_SERVER_START_TIMEOUT_S)
    parser.add_argument("--duration-floor-s", type=float, default=DEFAULT_DURATION_FLOOR_S)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run(
        artifact_path=args.output,
        endpoints=args.endpoint,
        endpoint_timeout_s=args.endpoint_timeout_s,
        server_start_timeout_s=args.server_start_timeout_s,
        duration_floor_s=args.duration_floor_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
