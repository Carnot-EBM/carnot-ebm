#!/usr/bin/env python3
"""Exp 5043: SOTA GGUF cache and judge endpoint preflight.

Spec refs: REQ-VERIFY-5043, SCENARIO-VERIFY-5043.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
EndpointProbe = Callable[[list[str], float], JsonDict]
Clock = Callable[[], float]

EXPERIMENT_ID = 5043
EXPERIMENT_NAME = "experiment_5043_sota_gguf_judge_preflight"
RESULT_RELATIVE_PATH = "results/experiment_5043_sota_gguf_judge_preflight.json"
SCHEMA = "carnot.experiment_5043_sota_gguf_judge_preflight.v1"
SPEC_REFS = ["REQ-VERIFY-5043", "SCENARIO-VERIFY-5043"]
DEFAULT_ENDPOINTS = ("http://127.0.0.1:8080",)
DEFAULT_PREFERRED_QUANT = "Q4_K_M"

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix for the preflight: blocked_sota_gguf_unavailable, "
            "blocked_judge_server, or complete_sota_gguf_judge_preflight_ready."
        )
    },
    "model_specs": {
        "principle": (
            "all three mandated SOTA GGUF hub IDs with exact resolved .gguf paths "
            "or missing."
        )
    },
    "usable_sota_models": {
        "principle": "the subset of mandated SOTA GGUFs resolved to local .gguf paths."
    },
    "sota_models_ready": {
        "principle": "true iff at least one mandated SOTA GGUF is locally usable."
    },
    "sota_judge_ready": {
        "principle": (
            "true iff a local endpoint can complete and expose top-logprob or "
            "confidence telemetry for a mandated SOTA model."
        )
    },
    "top_logprob_or_confidence_ready": {
        "principle": "true iff the probed judge route exposes top-logprob or confidence telemetry."
    },
    "endpoint_summary": {
        "principle": (
            "machine-readable completion and telemetry diagnostics for every probed "
            "llama.cpp-compatible endpoint."
        )
    },
    "legacy_models_smoke_only": {
        "principle": "legacy small models are allowed only for smoke tests, never headline evidence."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "duration_s",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class HttpProbe:
    """One HTTP probe result for the preflight artifact."""

    ready: bool
    status: int | None
    detail: str
    signal: str | None = None
    route: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "ready": bool(self.ready),
            "status": self.status,
            "detail": self.detail,
        }
        if self.signal is not None:
            payload["signal"] = self.signal
        if self.route is not None:
            payload["route"] = self.route
        return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _response_text(parsed: Any) -> str:
    if isinstance(parsed, Mapping):
        for key in ("content", "response", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
    return ""


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> tuple[int, Any]:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout_s) as response:
        status = int(getattr(response, "status", 0) or 0)
        raw = response.read().decode("utf-8", "replace")
    try:
        return status, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return status, {"raw": raw}


def _http_error_detail(exc: BaseException) -> tuple[int | None, str]:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        suffix = f": {body[:240]}" if body else ""
        return int(exc.code), f"HTTPError {exc.code}{suffix}"
    if isinstance(exc, (URLError, OSError, TimeoutError)):
        return None, f"{type(exc).__name__}: {exc}"
    return None, f"{type(exc).__name__}: {exc}"


def _completion_probe(endpoint: str, *, timeout_s: float) -> HttpProbe:
    base = endpoint.rstrip("/")
    attempts = (
        (
            base + "/completion",
            {"prompt": "Return exactly OK.", "n_predict": 4, "temperature": 0.0},
        ),
        (
            base + "/v1/completions",
            {
                "model": "local",
                "prompt": "Return exactly OK.",
                "max_tokens": 4,
                "temperature": 0.0,
            },
        ),
    )
    failures: list[str] = []
    for route, payload in attempts:
        try:
            status, parsed = _http_post_json(route, payload, timeout_s)
        except Exception as exc:
            status, detail = _http_error_detail(exc)
            failures.append(f"{route}: {detail}")
            continue
        text = _response_text(parsed)
        if 200 <= status < 300 and text.strip():
            return HttpProbe(
                True,
                status,
                f"completion returned non-empty content ({len(text.strip())} chars)",
                route=route,
            )
        failures.append(f"{route}: status={status} empty_or_unrecognized_completion")
    return HttpProbe(False, None, "; ".join(failures) or "no completion route attempted")


def _mapping_walk(value: Any) -> list[Any]:
    seen: list[Any] = [value]
    if isinstance(value, Mapping):
        for child in value.values():
            seen.extend(_mapping_walk(child))
    elif isinstance(value, list):
        for child in value:
            seen.extend(_mapping_walk(child))
    return seen


def _has_top_logprob_signal(parsed: Any) -> bool:
    for node in _mapping_walk(parsed):
        if not isinstance(node, Mapping):
            continue
        top_logprobs = node.get("top_logprobs")
        if isinstance(top_logprobs, (list, Mapping)) and bool(top_logprobs):
            return True
        token_logprobs = node.get("token_logprobs")
        if isinstance(token_logprobs, list) and bool(token_logprobs):
            return True
        probs = node.get("probs")
        if isinstance(probs, list) and bool(probs):
            return True
    return False


def _parse_confidence_from_text(text: str) -> float | None:
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidates = [json_match.group(0)] if json_match else []
    if text.strip() and text.strip() not in candidates:
        candidates.append(text.strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            value = parsed.get("confidence")
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= number <= 1.0:
                return number
    match = re.search(r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, flags=re.I)
    if match:
        return float(match.group(1))
    return None


def _telemetry_probe(endpoint: str, *, timeout_s: float) -> HttpProbe:
    base = endpoint.rstrip("/")
    top_logprob_attempts = (
        (
            base + "/completion",
            {
                "prompt": "Return one token: OK",
                "n_predict": 1,
                "temperature": 0.0,
                "n_probs": 5,
            },
        ),
        (
            base + "/v1/completions",
            {
                "model": "local",
                "prompt": "Return one token: OK",
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 5,
            },
        ),
    )
    failures: list[str] = []
    for route, payload in top_logprob_attempts:
        try:
            status, parsed = _http_post_json(route, payload, timeout_s)
        except Exception as exc:
            _status, detail = _http_error_detail(exc)
            failures.append(f"{route}: {detail}")
            continue
        if 200 <= status < 300 and _has_top_logprob_signal(parsed):
            return HttpProbe(True, status, "top-logprob telemetry present", "top_logprobs", route)
        failures.append(f"{route}: status={status} top_logprobs_absent")

    confidence_attempts = (
        (
            base + "/completion",
            {
                "prompt": (
                    'Return JSON only: {"answer":"OK","confidence":0.5}. '
                    "Use a confidence number between 0 and 1."
                ),
                "n_predict": 48,
                "temperature": 0.0,
            },
        ),
        (
            base + "/v1/completions",
            {
                "model": "local",
                "prompt": (
                    'Return JSON only: {"answer":"OK","confidence":0.5}. '
                    "Use a confidence number between 0 and 1."
                ),
                "max_tokens": 48,
                "temperature": 0.0,
            },
        ),
    )
    for route, payload in confidence_attempts:
        try:
            status, parsed = _http_post_json(route, payload, timeout_s)
        except Exception as exc:
            _status, detail = _http_error_detail(exc)
            failures.append(f"{route}: {detail}")
            continue
        text = _response_text(parsed)
        confidence = _parse_confidence_from_text(text)
        if 200 <= status < 300 and confidence is not None:
            return HttpProbe(
                True,
                status,
                f"structured confidence present ({confidence:.3f})",
                "confidence",
                route,
            )
        failures.append(f"{route}: status={status} confidence_absent")
    return HttpProbe(False, None, "; ".join(failures) or "no telemetry route attempted")


def _default_endpoint_list() -> list[str]:
    raw = (
        os.environ.get("CARNOT_5043_ENDPOINTS")
        or os.environ.get("CARNOT_JUDGE_ENDPOINTS")
        or os.environ.get("CARNOT_LLAMA_ENDPOINTS")
        or os.environ.get("CARNOT_JUDGE_SERVER_URL")
        or ""
    )
    endpoints = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    if endpoints:
        return endpoints
    return list(DEFAULT_ENDPOINTS)


def _normalize_endpoints(endpoints: Sequence[str] | None) -> list[str]:
    raw = list(endpoints) if endpoints is not None else _default_endpoint_list()
    normalized: list[str] = []
    for endpoint in raw:
        value = str(endpoint).strip().rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_ENDPOINTS)


def probe_endpoints(endpoints: list[str], timeout_s: float = 2.0) -> JsonDict:
    """Probe llama.cpp-compatible endpoints for completion and telemetry."""

    probes: list[JsonDict] = []
    completion_ready = False
    top_logprob_ready = False
    confidence_ready = False
    selected_endpoint: str | None = None
    telemetry_signal: str | None = None

    for endpoint in _normalize_endpoints(endpoints):
        completion = _completion_probe(endpoint, timeout_s=timeout_s)
        completion_ready = completion_ready or completion.ready
        if completion.ready:
            telemetry = _telemetry_probe(endpoint, timeout_s=timeout_s)
        else:
            telemetry = HttpProbe(False, None, "skipped: completion probe failed")
        probe = {
            "endpoint": endpoint,
            "completion_probe": completion.as_dict(),
            "telemetry_probe": telemetry.as_dict(),
        }
        probes.append(probe)
        if completion.ready and telemetry.ready:
            selected_endpoint = endpoint
            telemetry_signal = telemetry.signal
            top_logprob_ready = telemetry.signal == "top_logprobs"
            confidence_ready = telemetry.signal == "confidence"
            break

    return {
        "candidate_endpoints": _normalize_endpoints(endpoints),
        "selected_endpoint": selected_endpoint,
        "completion_ready": completion_ready,
        "top_logprob_ready": top_logprob_ready,
        "confidence_ready": confidence_ready,
        "telemetry_signal": telemetry_signal,
        "probes": probes,
    }


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
) -> tuple[JsonDict, list[JsonDict]]:
    """Resolve mandated model specs to exact paths or ``missing``."""

    model_specs: JsonDict = {}
    usable: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        hf_id = str(spec["hf_id"])
        quant = str(spec.get("preferred_quant") or preferred_quant)
        path = model_resolver(hf_id, quant)
        resolved_path = str(path) if path else "missing"
        model_specs[role] = {
            "hf_id": hf_id,
            "preferred_quant": quant,
            "resolved_path": resolved_path,
        }
        if path:
            usable.append({"role": role, "hf_id": hf_id, "model_path": str(path)})
    return model_specs, usable


def _checksum(payload: Mapping[str, Any]) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "model_specs": payload.get("model_specs"),
        "endpoint_summary": payload.get("endpoint_summary"),
    }
    return hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _verdict(
    *,
    sota_models_ready: bool,
    completion_ready: bool,
    top_logprob_or_confidence_ready: bool,
) -> str:
    if not sota_models_ready:
        return "blocked_sota_gguf_unavailable"
    if not completion_ready or not top_logprob_or_confidence_ready:
        return "blocked_judge_server"
    return "complete_sota_gguf_judge_preflight_ready"


def build_artifact(
    *,
    model_specs: JsonDict,
    usable_sota_models: list[JsonDict],
    endpoint_summary: JsonDict,
    duration_s: float,
    result_path: Path,
) -> JsonDict:
    sota_models_ready = bool(usable_sota_models)
    completion_ready = bool(endpoint_summary.get("completion_ready"))
    telemetry_ready_raw = bool(
        endpoint_summary.get("top_logprob_ready") or endpoint_summary.get("confidence_ready")
    )
    top_logprob_or_confidence_ready = bool(sota_models_ready and telemetry_ready_raw)
    sota_judge_ready = bool(sota_models_ready and completion_ready and telemetry_ready_raw)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(result_path),
        "honest_verdict": _verdict(
            sota_models_ready=sota_models_ready,
            completion_ready=completion_ready,
            top_logprob_or_confidence_ready=top_logprob_or_confidence_ready,
        ),
        "model_specs": model_specs,
        "usable_sota_models": usable_sota_models,
        "sota_models_ready": sota_models_ready,
        "sota_judge_ready": sota_judge_ready,
        "top_logprob_or_confidence_ready": top_logprob_or_confidence_ready,
        "endpoint_summary": endpoint_summary,
        "legacy_models_smoke_only": True,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing field: {field}")
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only must be true")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be a mapping")
    else:
        for spec in MANDATED_MODEL_SPECS:
            role = str(spec["role"])
            entry = artifact["model_specs"].get(role)
            if not isinstance(entry, Mapping):
                errors.append(f"model_specs.{role} missing")
                continue
            if entry.get("hf_id") != spec["hf_id"]:
                errors.append(f"model_specs.{role}.hf_id mismatch")
            if not entry.get("resolved_path"):
                errors.append(f"model_specs.{role}.resolved_path missing")
    endpoint_summary = artifact.get("endpoint_summary")
    if not isinstance(endpoint_summary, Mapping):
        errors.append("endpoint_summary must be a mapping")
    elif not isinstance(endpoint_summary.get("probes"), list):
        errors.append("endpoint_summary.probes must be a list")
    verdict = artifact.get("honest_verdict")
    if verdict not in {
        "blocked_sota_gguf_unavailable",
        "blocked_judge_server",
        "complete_sota_gguf_judge_preflight_ready",
    }:
        errors.append(f"unexpected honest_verdict: {verdict!r}")
    if artifact.get("sota_judge_ready") and not artifact.get("top_logprob_or_confidence_ready"):
        errors.append("sota_judge_ready requires top_logprob_or_confidence_ready")
    return errors


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    endpoint_probe: EndpointProbe = probe_endpoints,
    endpoints: Sequence[str] | None = None,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
    timeout_s: float = 2.0,
    now: Clock = time.monotonic,
    write: bool = True,
) -> JsonDict:
    """Run the 5043 preflight and optionally write the terminal artifact."""

    start = now()
    result_path = artifact_path if artifact_path is not None else root / RESULT_RELATIVE_PATH
    model_specs, usable = resolve_model_specs(
        model_resolver=model_resolver,
        preferred_quant=preferred_quant,
    )
    endpoint_summary = endpoint_probe(_normalize_endpoints(endpoints), timeout_s)
    artifact = build_artifact(
        model_specs=model_specs,
        usable_sota_models=usable,
        endpoint_summary=endpoint_summary,
        duration_s=now() - start,
        result_path=result_path,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
    if write:
        write_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": artifact["result_path"],
                "honest_verdict": artifact["honest_verdict"],
                "sota_models_ready": artifact["sota_models_ready"],
                "sota_judge_ready": artifact["sota_judge_ready"],
                "top_logprob_or_confidence_ready": artifact[
                    "top_logprob_or_confidence_ready"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))
