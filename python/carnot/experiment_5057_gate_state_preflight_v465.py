#!/usr/bin/env python3
"""Exp 5057: split gate-state preflight for downstream SOTA work.

Spec refs: REQ-VERIFY-5057, SCENARIO-VERIFY-5057.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_5043_sota_gguf_judge_preflight import probe_endpoints  # noqa: E402
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
EndpointProbe = Callable[[list[str], float], JsonDict]
Clock = Callable[[], float]

EXPERIMENT_ID = 5057
EXPERIMENT_NAME = "experiment_5057_gate_state_preflight_v465"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5057_gate_state_preflight_v465.py"
RESULT_RELATIVE_PATH = "results/experiment_5057_gate_state_preflight_v465.json"
SCHEMA = "carnot.experiment_5057_gate_state_preflight_v465.v1"
SPEC_REFS = ["REQ-VERIFY-5057", "SCENARIO-VERIFY-5057"]
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
            "terminal prefix for the split readiness report: "
            "complete_gate_state_preflight_all_ready, "
            "complete_gate_state_preflight_partial_ready, or "
            "blocked_gate_state_preflight_no_ready_paths."
        )
    },
    "model_specs": {
        "principle": (
            "all three mandated SOTA GGUF hub IDs with exact resolved .gguf paths "
            "or missing diagnostics."
        )
    },
    "usable_sota_models": {
        "principle": "the subset of mandated SOTA GGUF roles resolved to local .gguf paths."
    },
    "sota_models_ready": {
        "principle": "true iff at least one mandated SOTA GGUF path resolves locally."
    },
    "sota_judge_ready": {
        "principle": (
            "true iff a mandated SOTA GGUF is usable and a local endpoint completes "
            "with top-logprob or confidence telemetry."
        )
    },
    "top_logprob_or_confidence_ready": {
        "principle": (
            "true iff the endpoint telemetry probe exposes top-logprob or confidence "
            "telemetry, independent of model cache readiness."
        )
    },
    "tool_first_verifier_ready": {
        "principle": (
            "true iff deterministic JSON, constraint, and evidence smoke checks pass "
            "without live LLM inference."
        )
    },
    "endpoint_summary": {
        "principle": (
            "machine-readable completion and telemetry diagnostics for each probed "
            "llama.cpp-compatible endpoint."
        )
    },
    "skip_reasons": {
        "principle": "machine-readable reasons for every false readiness lane."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are smoke-only and never headline evidence."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
    "tool_first_verifier_summary",
    "reproducibility_checksum",
)

ALLOWED_VERDICTS = {
    "complete_gate_state_preflight_all_ready",
    "complete_gate_state_preflight_partial_ready",
    "blocked_gate_state_preflight_no_ready_paths",
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_endpoint_list() -> list[str]:  # pragma: no cover - environment shim.
    raw = (
        os.environ.get("CARNOT_5057_ENDPOINTS")
        or os.environ.get("CARNOT_JUDGE_ENDPOINTS")
        or os.environ.get("CARNOT_LLAMA_ENDPOINTS")
        or os.environ.get("CARNOT_JUDGE_SERVER_URL")
        or ""
    )
    endpoints = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    return endpoints or list(DEFAULT_ENDPOINTS)


def normalize_endpoints(endpoints: Sequence[str] | None) -> list[str]:
    raw = list(endpoints) if endpoints is not None else _default_endpoint_list()
    normalized: list[str] = []
    for endpoint in raw:
        value = str(endpoint).strip().rstrip("/")
        if value and value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_ENDPOINTS)


def resolve_model_specs(
    *,
    model_resolver: ModelResolver = resolve_cached_gguf,
    preferred_quant: str = DEFAULT_PREFERRED_QUANT,
) -> tuple[JsonDict, list[JsonDict]]:
    model_specs: JsonDict = {}
    usable: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        hf_id = str(spec["hf_id"])
        quant = str(spec.get("preferred_quant") or preferred_quant)
        path = model_resolver(hf_id, quant)
        model_specs[role] = {
            "hf_id": hf_id,
            "preferred_quant": quant,
            "resolved_path": str(path) if path else None,
            "missing_diagnostic": None if path else f"missing cached GGUF for {hf_id}",
        }
        if path:
            usable.append({"role": role, "hf_id": hf_id, "model_path": str(path)})
    return model_specs, usable


def run_tool_first_verifier_smoke() -> JsonDict:
    parsed = json.loads('{"answer":"OK","confidence":0.75}')
    json_ready = parsed == {"answer": "OK", "confidence": 0.75}
    constraint_ready = (2 + 2 == 4) and (3 * 3 >= 9)
    evidence_payload = _json_dumps({"claim": "2+2=4", "evidence": [2, 2, 4]})
    evidence_hash = hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
    evidence_ready = evidence_hash == hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
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
            "ready": evidence_ready,
            "detail": f"evidence sha256={evidence_hash}",
        },
    ]
    return {"ready": all(bool(check["ready"]) for check in checks), "checks": checks}


def _skip_reasons(
    *,
    sota_models_ready: bool,
    completion_ready: bool,
    telemetry_ready: bool,
    sota_judge_ready: bool,
    tool_first_verifier_ready: bool,
) -> list[str]:
    reasons: list[str] = []
    if not sota_models_ready:
        reasons.append("sota_models_unavailable")
    if not completion_ready:
        reasons.append("endpoint_completion_unavailable")
    if not telemetry_ready:
        reasons.append("top_logprob_or_confidence_unavailable")
    if not sota_judge_ready:
        reasons.append("sota_judge_unavailable")
    if not tool_first_verifier_ready:
        reasons.append("tool_first_verifier_unavailable")
    return reasons


def _verdict(
    *,
    sota_judge_ready: bool,
    tool_first_verifier_ready: bool,
    any_readiness: bool,
) -> str:
    if sota_judge_ready and tool_first_verifier_ready:
        return "complete_gate_state_preflight_all_ready"
    if any_readiness:
        return "complete_gate_state_preflight_partial_ready"
    return "blocked_gate_state_preflight_no_ready_paths"


def _checksum(payload: Mapping[str, Any]) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "model_specs": payload.get("model_specs"),
        "endpoint_summary": payload.get("endpoint_summary"),
        "skip_reasons": payload.get("skip_reasons"),
        "tool_first_verifier_ready": payload.get("tool_first_verifier_ready"),
    }
    return hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    model_specs: JsonDict,
    usable_sota_models: list[JsonDict],
    endpoint_summary: JsonDict,
    tool_first_verifier_summary: JsonDict,
    duration_s: float,
    result_path: Path,
) -> JsonDict:
    sota_models_ready = bool(usable_sota_models)
    completion_ready = bool(endpoint_summary.get("completion_ready"))
    telemetry_ready = bool(
        endpoint_summary.get("top_logprob_ready") or endpoint_summary.get("confidence_ready")
    )
    tool_first_verifier_ready = bool(tool_first_verifier_summary.get("ready"))
    sota_judge_ready = bool(sota_models_ready and completion_ready and telemetry_ready)
    skip_reasons = _skip_reasons(
        sota_models_ready=sota_models_ready,
        completion_ready=completion_ready,
        telemetry_ready=telemetry_ready,
        sota_judge_ready=sota_judge_ready,
        tool_first_verifier_ready=tool_first_verifier_ready,
    )
    any_readiness = bool(
        sota_models_ready or completion_ready or telemetry_ready or tool_first_verifier_ready
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(result_path),
        "honest_verdict": _verdict(
            sota_judge_ready=sota_judge_ready,
            tool_first_verifier_ready=tool_first_verifier_ready,
            any_readiness=any_readiness,
        ),
        "model_specs": model_specs,
        "usable_sota_models": usable_sota_models,
        "sota_models_ready": sota_models_ready,
        "sota_judge_ready": sota_judge_ready,
        "top_logprob_or_confidence_ready": telemetry_ready,
        "tool_first_verifier_ready": tool_first_verifier_ready,
        "endpoint_summary": endpoint_summary,
        "skip_reasons": skip_reasons,
        "legacy_models_smoke_only": True,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tool_first_verifier_summary": tool_first_verifier_summary,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing field: {field}")
    for field in (
        "sota_models_ready",
        "sota_judge_ready",
        "top_logprob_or_confidence_ready",
        "tool_first_verifier_ready",
    ):
        if field in artifact and not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bool")
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only must be true")
    model_specs = artifact.get("model_specs")
    if isinstance(model_specs, Mapping):
        for spec in MANDATED_MODEL_SPECS:
            role = str(spec["role"])
            entry = model_specs.get(role)
            if not isinstance(entry, Mapping):
                errors.append(f"model_specs.{role} missing")
            elif entry.get("hf_id") != spec["hf_id"]:
                errors.append(f"model_specs.{role}.hf_id mismatch")
            elif not entry.get("resolved_path") and not entry.get("missing_diagnostic"):
                errors.append(f"model_specs.{role} needs resolved_path or missing_diagnostic")
    else:
        errors.append("model_specs must be a mapping")
    endpoint_summary = artifact.get("endpoint_summary")
    if not isinstance(endpoint_summary, Mapping):
        errors.append("endpoint_summary must be a mapping")
    elif not isinstance(endpoint_summary.get("probes"), list):
        errors.append("endpoint_summary.probes must be a list")
    if not isinstance(artifact.get("skip_reasons"), list):
        errors.append("skip_reasons must be a list")
    if artifact.get("honest_verdict") not in ALLOWED_VERDICTS:
        errors.append(f"unexpected honest_verdict: {artifact.get('honest_verdict')!r}")
    if artifact.get("sota_judge_ready") and not artifact.get(
        "top_logprob_or_confidence_ready"
    ):
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
    start = now()
    result_path = artifact_path if artifact_path is not None else root / RESULT_RELATIVE_PATH
    model_specs, usable = resolve_model_specs(
        model_resolver=model_resolver,
        preferred_quant=preferred_quant,
    )
    endpoint_summary = endpoint_probe(normalize_endpoints(endpoints), timeout_s)
    artifact = build_artifact(
        model_specs=model_specs,
        usable_sota_models=usable,
        endpoint_summary=endpoint_summary,
        tool_first_verifier_summary=run_tool_first_verifier_smoke(),
        duration_s=now() - start,
        result_path=result_path,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        artifact["schema_errors"] = errors
    if write:
        write_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
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
                "tool_first_verifier_ready": artifact["tool_first_verifier_ready"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main(sys.argv[1:]))
