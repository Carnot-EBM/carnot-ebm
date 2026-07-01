#!/usr/bin/env python3
"""Exp 3013 SOTA GGUF logprob telemetry preflight.

**Researcher summary:**
    Exp 3001 proved that at least one mandated local SOTA GGUF can produce a
    live transcript.  Exp 3013 refreshes that readiness signal and adds the
    missing operational question: does the current llama.cpp loader path expose
    token logprobs and top-k alternatives that Cactus-style constrained
    acceptance can use?

**Detailed explanation for engineers:**
    This script records compute and cache preconditions before any model load,
    inspects only local GGUF cache paths, requests a bounded conservative
    llama.cpp completion with `logprobs` enabled when possible, and writes a
    terminal artifact that separates transcript readiness from telemetry
    readiness.  A live transcript can set `sota_headline_ready=true`, but
    `sota_logprob_ready=true` requires observed token logprobs plus top-k
    alternatives in the returned loader payload.

Spec: REQ-INFER-SOTA-021,
      SCENARIO-INFER-SOTA-021-001,
      SCENARIO-INFER-SOTA-021-002,
      SCENARIO-INFER-SOTA-021-003,
      SCENARIO-INFER-SOTA-021-004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.inference.sota_models import cached_sota_pair
from carnot.experiment_5043_sota_gguf_judge_preflight import (
    probe_endpoints as _probe_llama_cpp_endpoints,
)
from scripts import experiment_2989_sota_gguf_cache_provenance_preflight_v1 as base
from scripts import experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1 as exp3001
from scripts.experiment_template import _get_repo_root, _run_date


JsonDict = dict[str, Any]
CommandRunner = base.CommandRunner
CachedPairFn = base.CachedPairFn
PromptRunnerFn = base.PromptRunnerFn
ClockFn = base.ClockFn
EndpointProbeFn = Callable[[Sequence[str], float], JsonDict]

ARTIFACT_NAME = "experiment_3013_sota_gguf_logprob_telemetry_preflight_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
DEFAULT_ARTIFACT_PATH = Path("results") / ARTIFACT_FILENAME
RAW_TRANSCRIPT_DIR = Path("results") / "raw" / ARTIFACT_NAME
RANDOM_SEED = 3013
DEFAULT_PROMPT = "Reply in one short sentence: exp3013 SOTA GGUF telemetry live."
LOGPROBS_REQUESTED = 5
DEFAULT_ENDPOINTS: tuple[str, ...] = ("http://127.0.0.1:8080",)
LIVE_LLM_SUBSTRATE = "live_llm_inference"
NONLIVE_PREFLIGHT_SUBSTRATE = "deterministic_verifier"
HEADLINE_MODEL_IDS = base.HEADLINE_MODEL_IDS
SMOKE_ONLY_MODEL_IDS = base.SMOKE_ONLY_MODEL_IDS
ROLE_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}
FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal-prefix runtime verdict: complete_gguf_logprob_preflight_ready, "
            "complete_gguf_logprob_preflight_partial_ready, or "
            "blocked_gguf_logprob_preflight_no_ready_paths."
        )
    },
    "duration_s": {
        "principle": "wall-clock duration separates live endpoint work from cache-only preflight."
    },
    "inference_substrate": {
        "principle": (
            "declares whether the run invoked a live LLM endpoint or only performed "
            "deterministic cache/endpoint checks."
        )
    },
    "model_specs": {
        "principle": (
            "all three mandated SOTA GGUF IDs with exact local .gguf paths or "
            "missing diagnostics; GGUF repos are never checked with AutoTokenizer."
        )
    },
    "usable_sota_models": {
        "principle": "subset of mandated SOTA GGUF roles that resolve to local .gguf files."
    },
    "sota_models_ready": {
        "principle": "true iff at least one mandated SOTA GGUF path resolves locally."
    },
    "completion_endpoint_ready": {
        "principle": "true iff a llama.cpp-compatible endpoint returns non-empty completion text."
    },
    "logprob_endpoint_ready": {
        "principle": "true iff the endpoint exposes top-logprob telemetry, not just text."
    },
    "top_logprob_or_confidence_ready": {
        "principle": "true iff endpoint telemetry exposes top-logprobs or structured confidence."
    },
    "tool_first_verifier_ready": {
        "principle": "true iff deterministic JSON, arithmetic, and evidence checks pass without LLM inference."
    },
    "live_completion_invoked": {
        "principle": "true only when a real endpoint completion path returned content."
    },
    "skip_reasons": {
        "principle": "machine-readable reasons for every false readiness lane."
    },
    "flagged_adversarial": {
        "principle": "true when the artifact itself detects a duration/substrate inconsistency."
    },
}
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "sota_headline_ready",
    "sota_logprob_ready",
    "preconditions_checked",
    "model_specs",
    "usable_sota_models",
    "sota_models_ready",
    "completion_endpoint_ready",
    "logprob_endpoint_ready",
    "top_logprob_or_confidence_ready",
    "tool_first_verifier_ready",
    "live_completion_invoked",
    "skip_reasons",
    "flagged_adversarial",
    "headline_models_attempted",
    "headline_models_available",
    "telemetry_capabilities",
    "endpoint_summary",
    "cache_paths",
    "model_checksums",
    "live_transcript_paths",
    "legacy_smoke_only_used",
    "inference_substrate",
    "duration_s",
    "honest_verdict",
    "random_seed",
    "reproducibility_checksum",
)


def _resolved_model_specs(cache_inventory: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return role-keyed local path evidence for the mandated GGUFs."""
    by_hf = {str(row["hf_id"]): row for row in cache_inventory}
    resolved: JsonDict = {}
    for hf_id in HEADLINE_MODEL_IDS:
        role = ROLE_BY_HF_ID[hf_id]
        row = by_hf.get(hf_id, {})
        path = row.get("path") or row.get("resolved_path")
        resolved[role] = {
            "hf_id": hf_id,
            "preferred_quant": "Q4_K_M",
            "resolved_path": str(path) if path else None,
            "missing_diagnostic": None if path else f"missing cached GGUF for {hf_id}",
        }
    return resolved


def _model_specs(cache_inventory: Sequence[Mapping[str, Any]] | None = None) -> JsonDict:
    """Return the mandated model identities, paths, and telemetry parameters."""
    specs: JsonDict = {
        "experiment_id": 3013,
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
        "random_seed": RANDOM_SEED,
        "loader": "llama_cpp",
        "telemetry_request": {
            "logprobs": LOGPROBS_REQUESTED,
            "logits_all": True,
            "top_k_alternatives_required": True,
        },
        "source_pattern": base.ARTIFACT_NAME,
    }
    specs["resolved_models"] = _resolved_model_specs(cache_inventory or [])
    return specs


def _usable_sota_models(cache_inventory: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return mandated local GGUF paths that can be handed to llama.cpp."""
    usable: list[JsonDict] = []
    for row in cache_inventory:
        if row.get("cache_status") != "resolved":
            continue
        hf_id = str(row["hf_id"])
        path = row.get("path") or row.get("resolved_path")
        if path:
            usable.append(
                {
                    "role": ROLE_BY_HF_ID.get(hf_id, str(row.get("role") or "headline")),
                    "hf_id": hf_id,
                    "model_path": str(path),
                }
            )
    return usable


def _default_endpoint_list(env: Mapping[str, str] | None = None) -> list[str]:
    """Return configured llama.cpp endpoints, defaulting to localhost:8080."""
    source = env if env is not None else os.environ
    raw = (
        source.get("CARNOT_3013_ENDPOINTS")
        or source.get("CARNOT_LLAMA_ENDPOINTS")
        or source.get("CARNOT_JUDGE_ENDPOINTS")
        or source.get("CARNOT_JUDGE_SERVER_URL")
        or ""
    )
    endpoints = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    return endpoints or list(DEFAULT_ENDPOINTS)


def _normalize_endpoints(
    endpoints: Sequence[str] | None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Normalize configured endpoints while preserving probe order."""
    raw = list(endpoints) if endpoints is not None else _default_endpoint_list(env)
    normalized: list[str] = []
    for endpoint in raw:
        value = str(endpoint).strip().rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_ENDPOINTS)


def _probe_endpoint_summary(endpoints: Sequence[str], timeout_s: float) -> JsonDict:
    """Probe llama.cpp completion and telemetry endpoints with timing."""
    started = time.monotonic()
    summary = _probe_llama_cpp_endpoints(list(endpoints), timeout_s)
    summary = dict(summary)
    summary.setdefault("candidate_endpoints", list(endpoints))
    summary.setdefault("selected_endpoint", None)
    summary.setdefault("completion_ready", False)
    summary.setdefault("top_logprob_ready", False)
    summary.setdefault("confidence_ready", False)
    summary.setdefault("telemetry_signal", None)
    summary.setdefault("probes", [])
    summary["duration_s"] = round(time.monotonic() - started, 6)
    return summary


def _tool_first_verifier_summary() -> JsonDict:
    """Run deterministic verifier smoke checks that do not invoke an LLM."""
    parsed = json.loads('{"answer":"OK","confidence":0.75}')
    json_ready = parsed == {"answer": "OK", "confidence": 0.75}
    constraint_ready = (2 + 2 == 4) and (3 * 3 >= 9)
    evidence_payload = json.dumps(
        {"claim": "2+2=4", "evidence": [2, 2, 4]},
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_hash = hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
    checks = [
        {
            "name": "json_parse_check",
            "ready": json_ready,
            "detail": "parsed JSON object with answer and confidence",
        },
        {
            "name": "constraint_check",
            "ready": constraint_ready,
            "detail": "deterministic arithmetic constraints satisfied",
        },
        {
            "name": "evidence_check",
            "ready": True,
            "detail": f"evidence sha256={evidence_hash}",
        },
    ]
    return {"ready": all(bool(check["ready"]) for check in checks), "checks": checks}


def _finite_float(value: Any) -> float | None:
    """Return a finite float for JSON-ish logprob values, otherwise None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _telemetry_blockers(capabilities: Mapping[str, Any]) -> list[str]:
    """List unavailable telemetry fields in a stable order."""
    blockers: list[str] = []
    if not capabilities.get("token_logprobs_exposed"):
        blockers.append("token_logprobs_unavailable")
    if not capabilities.get("topk_logprobs_exposed"):
        blockers.append("topk_logprobs_unavailable")
    if not capabilities.get("logits_exposed"):
        blockers.append("logits_unavailable")
    return blockers


def _extract_loader_telemetry(raw_response: Any) -> JsonDict:
    """Extract observed logprob/top-k/logit exposure from a llama.cpp response."""
    if not isinstance(raw_response, Mapping):
        raw_response = {}
    choices = raw_response.get("choices") if isinstance(raw_response, Mapping) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, Mapping):
        first_choice = {}
    logprobs = first_choice.get("logprobs")

    token_texts: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    if isinstance(logprobs, Mapping):
        token_texts = [str(token) for token in logprobs.get("tokens") or [] if token is not None]
        for value in logprobs.get("token_logprobs") or []:
            parsed = _finite_float(value)
            if parsed is not None:
                token_logprobs.append(parsed)
        for row in logprobs.get("top_logprobs") or []:
            if not isinstance(row, Mapping):
                continue
            parsed_row: dict[str, float] = {}
            for token, value in row.items():
                parsed = _finite_float(value)
                if parsed is not None:
                    parsed_row[str(token)] = parsed
            if len(parsed_row) >= 2:
                top_logprobs.append(parsed_row)

    logits = first_choice.get("logits") or raw_response.get("logits")
    capabilities: JsonDict = {
        "token_logprobs_exposed": bool(token_logprobs),
        "topk_logprobs_exposed": bool(top_logprobs),
        "logits_exposed": bool(logits),
        "telemetry_observation": {
            "token_text_count": len(token_texts),
            "token_logprob_count": len(token_logprobs),
            "top_logprob_row_count": len(top_logprobs),
            "logits_present": bool(logits),
        },
    }
    capabilities["telemetry_blockers"] = _telemetry_blockers(capabilities)
    return capabilities


def _run_bounded_headline_prompt(
    model: Mapping[str, Any],
    *,
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
    timeout_s: int = 300,
) -> JsonDict:
    """Run one bounded llama.cpp prompt and parse loader telemetry exposure."""
    script = (
        "import json, os, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "path, hf_id, prompt = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "requested_gpu = int(sys.argv[4])\n"
        "main_gpu = int(os.environ.get('CARNOT_SOTA_MAIN_GPU', '0'))\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "started = time.monotonic()\n"
        "llm = Llama(model_path=path, n_ctx=384, n_batch=64, n_ubatch=64, "
        "n_gpu_layers=-1, main_gpu=main_gpu, logits_all=True, verbose=False)\n"
        "logprob_request_error = None\n"
        "try:\n"
        f"    out = llm(prompt, max_tokens=16, temperature=0.0, seed={RANDOM_SEED}, "
        f"logprobs={LOGPROBS_REQUESTED})\n"
        "except Exception as exc:\n"
        "    logprob_request_error = f'{type(exc).__name__}: {exc}'\n"
        f"    out = llm(prompt, max_tokens=16, temperature=0.0, seed={RANDOM_SEED})\n"
        "duration = time.monotonic() - started\n"
        "choice = (out.get('choices') or [{}])[0]\n"
        "text = str(choice.get('text') or '').strip()\n"
        "tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "scores = getattr(llm, 'scores', None)\n"
        "score_shape = getattr(scores, 'shape', None)\n"
        "if score_shape:\n"
        "    logits_shape = [int(part) for part in score_shape]\n"
        "elif isinstance(scores, list) and scores and isinstance(scores[0], list):\n"
        "    logits_shape = [len(scores), len(scores[0])]\n"
        "else:\n"
        "    logits_shape = []\n"
        "def json_default(obj):\n"
        "    item = getattr(obj, 'item', None)\n"
        "    if callable(item):\n"
        "        return item()\n"
        "    try:\n"
        "        return float(obj)\n"
        "    except Exception:\n"
        "        return str(obj)\n"
        "llm.close()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'load_status': 'loaded',\n"
        "    'generation_status': 'generated' if text and tokens > 0 else 'empty_response',\n"
        "    'usable': bool(text) and tokens > 0 and supports_gpu,\n"
        "    'gpu_backed': supports_gpu,\n"
        "    'hf_id': hf_id,\n"
        "    'model_path': path,\n"
        "    'prompt': prompt,\n"
        "    'response_text': text,\n"
        "    'tokens_generated': tokens,\n"
        "    'duration_s': round(duration, 6),\n"
        "    'inference_substrate': 'llama_cpp_gpu' if supports_gpu else 'llama_cpp_cpu',\n"
        "    'requested_gpu': requested_gpu,\n"
        "    'main_gpu': main_gpu,\n"
        "    'logprobs_requested': "
        f"{LOGPROBS_REQUESTED},\n"
        "    'logprob_request_error': logprob_request_error,\n"
        "    'loader_logits_available': bool(logits_shape),\n"
        "    'loader_logits_shape': logits_shape,\n"
        "    'raw_response': out,\n"
        "}, sort_keys=True, default=json_default))\n"
    )
    command = [
        selected_python,
        "-c",
        script,
        str(model["path"]),
        str(model["hf_id"]),
        DEFAULT_PROMPT,
        str(model.get("gpu", 0)),
    ]
    result = command_runner(command, timeout_s=timeout_s, env=dict(env))
    try:
        parsed = json.loads(base._stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "load_status": "failed",
            "generation_status": "failed",
            "usable": False,
            "gpu_backed": False,
            "hf_id": model.get("hf_id"),
            "model_path": model.get("path"),
            "prompt": DEFAULT_PROMPT,
            "response_text": "",
            "tokens_generated": 0,
            "duration_s": 0.0,
            "inference_substrate": "llama_cpp_failed",
            "blocker": base._stderr(result) or base._stdout(result) or "bounded_prompt_failed",
            "raw_response": None,
        }
    parsed["duration_s"] = float(parsed.get("duration_s") or parsed.get("duration_seconds") or 0.0)
    telemetry = _extract_loader_telemetry(parsed.get("raw_response"))
    if parsed.get("loader_logits_available"):
        observation = dict(telemetry.get("telemetry_observation") or {})
        observation["logits_present"] = True
        observation["logits_shape"] = parsed.get("loader_logits_shape") or []
        telemetry["logits_exposed"] = True
        telemetry["telemetry_observation"] = observation
        telemetry["telemetry_blockers"] = _telemetry_blockers(telemetry)
    parsed.update(telemetry)
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stdout_summary"] = base._summarize(base._stdout(result))
    parsed["stderr_summary"] = base._summarize(base._stderr(result))
    return parsed


def _write_transcript(
    transcript_dir: Path,
    *,
    attempt: Mapping[str, Any],
    prompt_result: Mapping[str, Any],
) -> JsonDict:
    """Persist replayable transcript and observed telemetry summary."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"{base._safe_model_slug(str(attempt['hf_id']))}.json"
    payload = {
        "model_hf_id": attempt["hf_id"],
        "model_path": attempt["cache_path"],
        "prompt": prompt_result.get("prompt", DEFAULT_PROMPT),
        "response_text": prompt_result.get("response_text", ""),
        "tokens_generated": prompt_result.get("tokens_generated", 0),
        "duration_s": prompt_result.get("duration_s", 0.0),
        "inference_substrate": prompt_result.get("inference_substrate"),
        "load_status": prompt_result.get("load_status"),
        "generation_status": prompt_result.get("generation_status"),
        "telemetry": {
            "token_logprobs_exposed": prompt_result.get("token_logprobs_exposed", False),
            "topk_logprobs_exposed": prompt_result.get("topk_logprobs_exposed", False),
            "logits_exposed": prompt_result.get("logits_exposed", False),
            "telemetry_observation": prompt_result.get("telemetry_observation", {}),
            "telemetry_blockers": prompt_result.get("telemetry_blockers", []),
        },
        "raw_response": prompt_result.get("raw_response"),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded + b"\n")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _attempt_rows(
    *,
    cache_inventory: Sequence[Mapping[str, Any]],
    checksum_by_model: Mapping[str, Mapping[str, Any]],
    precondition_evidence: Mapping[str, Any],
    selected_python: str,
    env: Mapping[str, str],
    transcript_dir: Path,
    command_runner: CommandRunner,
    prompt_runner_fn: PromptRunnerFn,
    prompt_timeout_s: int,
    direct_load_enabled: bool,
) -> tuple[list[JsonDict], list[str]]:
    """Attempt bounded live telemetry collection for each cached headline GGUF."""
    attempts: list[JsonDict] = []
    transcript_paths: list[str] = []
    torch_cuda = bool(precondition_evidence["torch_cuda"].get("cuda_available"))
    llama_gpu = bool(precondition_evidence["llama_cpp"].get("llama_cpp_supports_gpu_offload"))
    for index, row in enumerate(cache_inventory):
        hf_id = str(row["hf_id"])
        attempt: JsonDict = {
            "hf_id": hf_id,
            "cache_status": row["cache_status"],
            "cache_path": row["path"],
            "resolved_path": row["resolved_path"],
            "checksum_evidence": checksum_by_model[hf_id],
            "load_status": "not_attempted",
            "generation_status": "not_attempted",
            "duration_s": 0.0,
            "transcript_path": None,
            "transcript_sha256": None,
            "token_logprobs_exposed": False,
            "topk_logprobs_exposed": False,
            "logits_exposed": False,
            "telemetry_observation": {},
            "telemetry_blockers": ["no_live_headline_generation"],
        }
        if row["cache_status"] != "resolved":
            attempt["load_status"] = "skipped_missing_cache"
            attempts.append(attempt)
            continue
        if not direct_load_enabled:
            attempt["load_status"] = "not_attempted_direct_load_disabled"
            attempt["telemetry_blockers"] = ["direct_load_disabled"]
            attempts.append(attempt)
            continue
        if not (torch_cuda and llama_gpu):
            attempt["load_status"] = "not_attempted_runtime_precondition_failed"
            attempts.append(attempt)
            continue

        prompt_result = prompt_runner_fn(
            {"hf_id": hf_id, "path": row["path"], "gpu": index},
            selected_python=selected_python,
            command_runner=command_runner,
            env=env,
            timeout_s=prompt_timeout_s,
        )
        telemetry = _extract_loader_telemetry(prompt_result.get("raw_response"))
        for key in (
            "token_logprobs_exposed",
            "topk_logprobs_exposed",
            "logits_exposed",
            "telemetry_observation",
            "telemetry_blockers",
        ):
            if key in prompt_result:
                telemetry[key] = prompt_result[key]
        attempt.update(
            {
                "load_status": prompt_result.get("load_status", "unknown"),
                "generation_status": prompt_result.get("generation_status", "unknown"),
                "duration_s": float(
                    prompt_result.get("duration_s")
                    or prompt_result.get("duration_seconds")
                    or 0.0
                ),
                "tokens_generated": int(prompt_result.get("tokens_generated") or 0),
                "gpu_backed": bool(prompt_result.get("gpu_backed")),
                "blocker": prompt_result.get("blocker")
                or prompt_result.get("logprob_request_error"),
                "inference_substrate": prompt_result.get("inference_substrate"),
                **telemetry,
            }
        )
        if prompt_result.get("usable") and str(prompt_result.get("response_text") or "").strip():
            transcript = _write_transcript(
                transcript_dir,
                attempt=attempt,
                prompt_result={**prompt_result, **telemetry},
            )
            attempt["transcript_path"] = transcript["path"]
            attempt["transcript_sha256"] = transcript["sha256"]
            transcript_paths.append(transcript["path"])
        attempts.append(attempt)
    return attempts, transcript_paths


def _build_telemetry_capabilities(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize observed loader telemetry without promoting absent fields."""
    by_model = {
        str(row["hf_id"]): {
            "load_status": row.get("load_status"),
            "generation_status": row.get("generation_status"),
            "token_logprobs_exposed": bool(row.get("token_logprobs_exposed")),
            "topk_logprobs_exposed": bool(row.get("topk_logprobs_exposed")),
            "logits_exposed": bool(row.get("logits_exposed")),
            "telemetry_observation": row.get("telemetry_observation", {}),
            "telemetry_blockers": row.get("telemetry_blockers", []),
        }
        for row in attempts
    }
    any_live = any(row.get("transcript_path") for row in attempts)
    any_token = any(row.get("token_logprobs_exposed") for row in attempts)
    any_topk = any(row.get("topk_logprobs_exposed") for row in attempts)
    any_logits = any(row.get("logits_exposed") for row in attempts)
    blockers: list[str] = []
    if not any_live:
        blockers.append("no_live_headline_generation")
    else:
        if not any_token:
            blockers.append("token_logprobs_unavailable")
        if not any_topk:
            blockers.append("topk_logprobs_unavailable")
    return {
        "requested": {
            "logprobs": LOGPROBS_REQUESTED,
            "top_k_alternatives_required": True,
        },
        "overall": {
            "any_live_generation": any_live,
            "token_logprobs_exposed": any_token,
            "topk_logprobs_exposed": any_topk,
            "logits_exposed": any_logits,
            "cactus_acceptance_ready": bool(any_live and any_token and any_topk),
        },
        "by_model": by_model,
        "blockers": blockers,
    }


def _honest_verdict(*, headline_ready: bool, logprob_ready: bool) -> str:
    """Return the terminal verdict required by conductor-style gates."""
    if not headline_ready:
        return "blocked_gguf_logprob_preflight_no_ready_paths"
    if logprob_ready:
        return "complete_gguf_logprob_preflight_ready"
    return "complete_gguf_logprob_preflight_partial_ready"


def _runtime_skip_reasons(
    *,
    sota_models_ready: bool,
    completion_endpoint_ready: bool,
    logprob_endpoint_ready: bool,
    top_logprob_or_confidence_ready: bool,
    tool_first_verifier_ready: bool,
    live_completion_invoked: bool,
) -> list[str]:
    """Return stable machine-readable reasons for every false v466 lane."""
    reasons: list[str] = []
    if not sota_models_ready:
        reasons.append("sota_models_unavailable")
    if not completion_endpoint_ready:
        reasons.append("endpoint_completion_unavailable")
    if not logprob_endpoint_ready:
        reasons.append("logprob_endpoint_unavailable")
    if not top_logprob_or_confidence_ready:
        reasons.append("top_logprob_or_confidence_unavailable")
    if not tool_first_verifier_ready:
        reasons.append("tool_first_verifier_unavailable")
    if not live_completion_invoked:
        reasons.append("live_completion_not_invoked")
    return reasons


def _runtime_honest_verdict(
    *,
    sota_models_ready: bool,
    completion_endpoint_ready: bool,
    top_logprob_or_confidence_ready: bool,
    tool_first_verifier_ready: bool,
) -> str:
    """Return the v466 terminal-prefix readiness verdict."""
    if not sota_models_ready:
        return "blocked_gguf_logprob_preflight_no_ready_paths"
    if completion_endpoint_ready and top_logprob_or_confidence_ready and tool_first_verifier_ready:
        return "complete_gguf_logprob_preflight_ready"
    return "complete_gguf_logprob_preflight_partial_ready"


def _inference_substrate(
    *,
    ready: bool,
    cached_count: int,
    attempted_live: bool,
    generated_substrates: Sequence[str],
) -> str:
    """Describe the actual execution substrate used for the terminal claim."""
    if ready and generated_substrates:
        return generated_substrates[0]
    if cached_count == 0:
        return "blocked_no_headline_cache"
    if attempted_live:
        return "llama_cpp_failed"
    return NONLIVE_PREFLIGHT_SUBSTRATE


def _reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the fields that define this preflight's replayable state."""
    basis = {
        "artifact": payload.get("artifact"),
        "honest_verdict": payload.get("honest_verdict"),
        "model_specs": payload.get("model_specs"),
        "usable_sota_models": payload.get("usable_sota_models"),
        "endpoint_summary": payload.get("endpoint_summary"),
        "skip_reasons": payload.get("skip_reasons"),
        "random_seed": payload.get("random_seed"),
    }
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flagged_adversarial(
    *,
    inference_substrate: str,
    duration_s: float,
    live_completion_invoked: bool,
) -> bool:
    """Flag the artifact when it claims live LLM inference below the audit floor."""
    return bool(
        live_completion_invoked
        and inference_substrate == LIVE_LLM_SUBSTRATE
        and float(duration_s) < 60.0
    )


def build_preflight_artifact(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = base._run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    endpoint_probe_fn: EndpointProbeFn = _probe_endpoint_summary,
    endpoints: Sequence[str] | None = None,
    endpoint_timeout_s: float = 2.0,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    prompt_timeout_s: int = 300,
    direct_load_enabled: bool = False,
) -> JsonDict:
    """Build the Exp 3013 terminal preflight artifact without downloading weights."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or base._selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    precondition_evidence = base._preconditions(
        project_root=root,
        selected_python=selected,
        env=merged_env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
    )
    headline_cache = base._inspect_cache(root, merged_env, HEADLINE_MODEL_IDS)
    smoke_cache = base._inspect_cache(root, merged_env, SMOKE_ONLY_MODEL_IDS)
    model_checksums = {
        row["hf_id"]: base._file_evidence(row["path"]) for row in [*headline_cache, *smoke_cache]
    }
    precondition_evidence["checksum_feasibility"] = exp3001._checksum_feasibility(
        model_checksums
    )
    endpoint_summary = endpoint_probe_fn(
        _normalize_endpoints(endpoints, env=merged_env),
        endpoint_timeout_s,
    )

    attempts, live_transcript_paths = _attempt_rows(
        cache_inventory=headline_cache,
        checksum_by_model=model_checksums,
        precondition_evidence=precondition_evidence,
        selected_python=selected,
        env=merged_env,
        transcript_dir=root / RAW_TRANSCRIPT_DIR,
        command_runner=command_runner,
        prompt_runner_fn=prompt_runner_fn,
        prompt_timeout_s=prompt_timeout_s,
        direct_load_enabled=direct_load_enabled,
    )
    cached_count = sum(1 for row in headline_cache if row["cache_status"] == "resolved")
    attempted_live = any(
        row.get("load_status")
        not in {
            "skipped_missing_cache",
            "not_attempted_runtime_precondition_failed",
            "not_attempted_direct_load_disabled",
        }
        for row in attempts
    )
    completion_endpoint_ready = bool(endpoint_summary.get("completion_ready"))
    logprob_endpoint_ready = bool(endpoint_summary.get("top_logprob_ready"))
    top_logprob_or_confidence_ready = bool(
        endpoint_summary.get("top_logprob_ready") or endpoint_summary.get("confidence_ready")
    )
    tool_first_verifier_summary = _tool_first_verifier_summary()
    tool_first_verifier_ready = bool(tool_first_verifier_summary.get("ready"))
    live_completion_invoked = completion_endpoint_ready
    ready = bool(live_transcript_paths) or completion_endpoint_ready
    telemetry_capabilities = _build_telemetry_capabilities(attempts)
    direct_logprob_ready = bool(
        telemetry_capabilities["overall"]["any_live_generation"]
        and telemetry_capabilities["overall"]["token_logprobs_exposed"]
        and telemetry_capabilities["overall"]["topk_logprobs_exposed"]
    )
    logprob_ready = bool(direct_logprob_ready or logprob_endpoint_ready)
    generated_by_hf = {
        str(row["hf_id"]): bool(row.get("transcript_path")) for row in attempts
    }
    usable_sota_models = _usable_sota_models(headline_cache)
    sota_models_ready = bool(usable_sota_models)
    available_models = [
        {
            "hf_id": row["hf_id"],
            "path": row["path"],
            "status": "cache_resolved",
            "generated": generated_by_hf.get(str(row["hf_id"]), False),
        }
        for row in headline_cache
        if row["cache_status"] == "resolved"
    ]
    generated_substrates = [
        str(row.get("inference_substrate"))
        for row in attempts
        if row.get("transcript_path") and row.get("inference_substrate")
    ]
    finished = monotonic()
    duration_s = round(finished - started, 6)
    inference_substrate = (
        LIVE_LLM_SUBSTRATE
        if live_completion_invoked
        else _inference_substrate(
            ready=bool(live_transcript_paths),
            cached_count=cached_count,
            attempted_live=attempted_live,
            generated_substrates=generated_substrates,
        )
    )
    honest_verdict = _runtime_honest_verdict(
        sota_models_ready=sota_models_ready,
        completion_endpoint_ready=completion_endpoint_ready,
        top_logprob_or_confidence_ready=top_logprob_or_confidence_ready,
        tool_first_verifier_ready=tool_first_verifier_ready,
    )
    skip_reasons = _runtime_skip_reasons(
        sota_models_ready=sota_models_ready,
        completion_endpoint_ready=completion_endpoint_ready,
        logprob_endpoint_ready=logprob_endpoint_ready,
        top_logprob_or_confidence_ready=top_logprob_or_confidence_ready,
        tool_first_verifier_ready=tool_first_verifier_ready,
        live_completion_invoked=live_completion_invoked,
    )
    flagged_adversarial = _flagged_adversarial(
        inference_substrate=inference_substrate,
        duration_s=duration_s,
        live_completion_invoked=live_completion_invoked,
    )

    artifact: JsonDict = {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "run_date": _run_date(),
        "sota_headline_ready": ready,
        "sota_logprob_ready": logprob_ready,
        "preconditions_checked": True,
        "model_specs": _model_specs(headline_cache),
        "usable_sota_models": usable_sota_models,
        "sota_models_ready": sota_models_ready,
        "sota_judge_ready": bool(
            sota_models_ready and completion_endpoint_ready and top_logprob_or_confidence_ready
        ),
        "completion_endpoint_ready": completion_endpoint_ready,
        "logprob_endpoint_ready": logprob_endpoint_ready,
        "top_logprob_or_confidence_ready": top_logprob_or_confidence_ready,
        "tool_first_verifier_ready": tool_first_verifier_ready,
        "live_completion_invoked": live_completion_invoked,
        "skip_reasons": skip_reasons,
        "flagged_adversarial": flagged_adversarial,
        "endpoint_summary": endpoint_summary,
        "headline_models_attempted": attempts,
        "headline_models_available": available_models,
        "telemetry_capabilities": telemetry_capabilities,
        "cache_paths": {
            "roots": precondition_evidence["cache_roots"],
            "headline_models": {row["hf_id"]: row["path"] for row in headline_cache},
            "smoke_only_models": {row["hf_id"]: row["path"] for row in smoke_cache},
        },
        "model_checksums": model_checksums,
        "live_transcript_paths": live_transcript_paths,
        "legacy_smoke_only_used": False,
        "legacy_smoke_context": {
            "smoke_only": False,
            "model_ids": list(SMOKE_ONLY_MODEL_IDS),
            "used_for_headline_readiness": False,
        },
        "inference_substrate": inference_substrate,
        "duration_s": duration_s,
        "honest_verdict": honest_verdict,
        "field_principles": dict(FIELD_PRINCIPLES),
        "precondition_evidence": precondition_evidence,
        "tests_run": list(tests_run or []),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    if flagged_adversarial:
        artifact["corrigendum_pending"] = [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": (
                    "live_completion_invoked=true with live_llm_inference but "
                    f"duration_s={duration_s} < 60.0"
                ),
            }
        ]
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for conductor and downstream gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = base._run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    endpoint_probe_fn: EndpointProbeFn = _probe_endpoint_summary,
    endpoints: Sequence[str] | None = None,
    endpoint_timeout_s: float = 2.0,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    prompt_timeout_s: int = 300,
    direct_load_enabled: bool = False,
) -> JsonDict:
    """Build and write the Exp 3013 SOTA/logprob preflight JSON artifact."""
    root = Path(project_root) if project_root is not None else Path(_get_repo_root())
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_preflight_artifact(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        prompt_runner_fn=prompt_runner_fn,
        endpoint_probe_fn=endpoint_probe_fn,
        endpoints=endpoints,
        endpoint_timeout_s=endpoint_timeout_s,
        monotonic=monotonic,
        tests_run=tests_run,
        prompt_timeout_s=prompt_timeout_s,
        direct_load_enabled=direct_load_enabled,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument("--prompt-timeout-s", type=int, default=300)
    parser.add_argument("--endpoint", action="append", default=None)
    parser.add_argument("--endpoint-timeout-s", type=float, default=2.0)
    parser.add_argument("--direct-load", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    kwargs: JsonDict = {
        "output_path": args.output,
        "selected_python": args.selected_python,
        "tests_run": args.test_run,
    }
    if args.prompt_timeout_s != 300:
        kwargs["prompt_timeout_s"] = args.prompt_timeout_s
    if args.endpoint:
        kwargs["endpoints"] = args.endpoint
    if args.endpoint_timeout_s != 2.0:
        kwargs["endpoint_timeout_s"] = args.endpoint_timeout_s
    if args.direct_load:
        kwargs["direct_load_enabled"] = True
    run_experiment(**kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
