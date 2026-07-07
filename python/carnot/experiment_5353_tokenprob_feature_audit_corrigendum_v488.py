#!/usr/bin/env python3
"""Exp5353: token-probability feature audit corrigendum.

Spec refs: REQ-VERIFY-5353, SCENARIO-VERIFY-5353.

This module audits backend feature availability only. It records whether the
selected local GGUF backend exposes internal token-probability surfaces such as
per-token logprobs and top-k alternatives. It does not score generated text,
train a reward model, rerank candidates, or make an answer-quality claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as exp5323
from carnot import experiment_5331_internal_energy_receipt_harness_v486 as exp5331
from carnot import experiment_5337_sota_runtime_corrigendum_multimodel_v487 as exp5337


JsonDict = dict[str, Any]
FeatureProbe = Callable[..., JsonDict]
PreconditionsProvider = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5353_tokenprob_feature_audit_corrigendum_v488"
EXPERIMENT_NUMBER = 5353
MILESTONE = "2026.07.488"
RUN_DATE = "20260707"
RESULT_RELATIVE_PATH = Path("results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json")
SCHEMA = "carnot.experiment_5353.tokenprob_feature_audit_corrigendum.v488"
INFERENCE_SUBSTRATE_LIVE = "live_llm_inference"
INFERENCE_SUBSTRATE_AUDIT_ONLY = "feature_audit_only"
ALLOWED_INFERENCE_SUBSTRATES = (INFERENCE_SUBSTRATE_LIVE, INFERENCE_SUBSTRATE_AUDIT_ONLY)
SPEC_REFS = ("REQ-VERIFY-5353", "SCENARIO-VERIFY-5353")
TERMINAL_PREFIXES = ("complete:", "blocked_")
RANDOM_SEED = 5353
N_PROBS = 8
N_PREDICT = 1
MANDATED_MODEL_SPECS = exp5323.MANDATED_MODEL_SPECS
EXPECTED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
EXPECTED_ROLES = tuple(str(spec["role"]) for spec in MANDATED_MODEL_SPECS)
EXPECTED_HF_BY_ROLE = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents `.487` flagged internal-energy evidence from being reused.",
    "status": "Lets downstream gates skip if feature rows are absent.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous feature availability."
    ),
    "inference_substrate": (
        "Expected value must truthfully reflect live_llm_inference or feature_audit_only."
    ),
    "MODEL_SPECS": "Confirms mandated local SOTA GGUF models are included for any LLM call.",
    "preconditions_checked": "Records model/backend/GPU checks before feature probing.",
    "selected_model_spec": "Identifies which mandated model was probed.",
    "tests_run": "Lists local checks for schema and receipt validation.",
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "tests_run",
)
REQUIRED_ARTIFACT_FIELDS = REQUIRED_WRAPPED_FIELDS + (
    "per_token_logprob_available",
    "topk_alternatives_available",
    "logits_available",
    "attention_available",
    "hidden_states_available",
    "tokenprob_feature_row_count",
    "missing_feature_names",
    "methodology_duration_s",
    "feature_audit_duration_s",
    "external_text_scorer_reopened",
    "no_quality_claim",
    "tokenprob_feature_rows_ready",
)

AUDIT_PROMPTS: tuple[JsonDict, ...] = (
    {
        "prompt_id": "receipt_alpha",
        "prompt": "Token-probability feature audit only. Return exactly alpha.",
    },
    {
        "prompt_id": "receipt_blue",
        "prompt": "Token-probability feature audit only. Return exactly blue.",
    },
    {
        "prompt_id": "receipt_17",
        "prompt": "Token-probability feature audit only. Return exactly 17.",
    },
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _round_float(value: float | None) -> float | None:
    return None if value is None else round(float(value), 9)


def _model_specs_from_runtime(runtime_artifact: Mapping[str, Any]) -> JsonDict:
    prior = _raw_or_wrapped_value(runtime_artifact, "MODEL_SPECS")
    prior = prior if isinstance(prior, Mapping) else {}
    out: JsonDict = {}
    for spec in MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        row = prior.get(role) if isinstance(prior.get(role), Mapping) else {}
        out[role] = {
            "role": role,
            "hf_id": str(spec["hf_id"]),
            "quantization": str(row.get("quantization") or spec.get("quantization", "Q4_K_M")),
            "model_path": row.get("model_path"),
            "status": str(row.get("status") or "missing_local_gguf"),
            "autotokenizer_used": False,
            "file_receipts": row.get("file_receipts"),
            "metadata": row.get("metadata"),
        }
    return out


def _selected_model_from_sources(
    runtime_artifact: Mapping[str, Any],
    internal_artifact: Mapping[str, Any],
    tiny_receipt: Mapping[str, Any],
    model_specs: Mapping[str, Any],
) -> JsonDict | None:
    receipt = _raw_or_wrapped_value(runtime_artifact, "runtime_corrigendum_receipt")
    role = "flagship_dense"
    if isinstance(receipt, Mapping) and receipt.get("model_role"):
        role = str(receipt["model_role"])
    selected = model_specs.get(role)
    if isinstance(selected, Mapping) and selected.get("hf_id") in EXPECTED_MODEL_IDS:
        return dict(selected)

    prior_selected = _raw_or_wrapped_value(internal_artifact, "selected_model_spec")
    if isinstance(prior_selected, Mapping) and prior_selected.get("hf_id") in EXPECTED_MODEL_IDS:
        out = dict(prior_selected)
        out["autotokenizer_used"] = False
        return out

    hf_id = tiny_receipt.get("model_hf_id")
    model_path = tiny_receipt.get("model_path")
    if hf_id in EXPECTED_MODEL_IDS:
        return {
            "role": str(tiny_receipt.get("model_role") or role),
            "hf_id": str(hf_id),
            "model_path": model_path,
            "status": "local_gguf_resolved" if model_path else "missing_local_gguf",
            "autotokenizer_used": False,
        }
    return None


def _selected_backend_command(runtime_artifact: Mapping[str, Any]) -> JsonDict | None:
    selected = _raw_or_wrapped_value(runtime_artifact, "selected_backend_command")
    if isinstance(selected, Mapping):
        return dict(selected)
    return None


def _completion_probability_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    response = receipt.get("response_json") if isinstance(receipt.get("response_json"), Mapping) else {}
    rows = response.get("completion_probabilities") or receipt.get("completion_probabilities") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _top_logprob_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for completion_row in _completion_probability_rows(receipt):
        top = completion_row.get("top_logprobs")
        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)):
            rows.extend(dict(row) for row in top if isinstance(row, Mapping))
        elif _numeric(completion_row.get("logprob")) is not None:
            rows.append(completion_row)
    return rows


def _availability_true(payload: Mapping[str, Any], field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, Mapping):
        return value.get("availability") == "available"
    return False


def _schema_availability(schema_artifact: Mapping[str, Any], field: str) -> bool:
    availability = schema_artifact.get("availability")
    if isinstance(availability, Mapping):
        return availability.get(field) is True
    return False


def audit_backend_features(
    tiny_receipt: Mapping[str, Any],
    schema_artifact: Mapping[str, Any],
    internal_artifact: Mapping[str, Any],
) -> JsonDict:
    """Return the exact backend feature surfaces visible to the audit."""

    completion_rows = _completion_probability_rows(tiny_receipt)
    top_rows = _top_logprob_rows(tiny_receipt)
    raw_output = tiny_receipt.get("raw_output")
    raw_output = raw_output if isinstance(raw_output, Mapping) else {}
    option_surface = _raw_or_wrapped_value(internal_artifact, "backend_option_surface")
    option_flags = {}
    if isinstance(option_surface, Mapping) and isinstance(option_surface.get("option_flags"), Mapping):
        option_flags = dict(option_surface["option_flags"])

    per_token = any(_numeric(row.get("logprob")) is not None for row in completion_rows)
    topk = any(_numeric(row.get("logprob")) is not None for row in top_rows)
    logits = bool(
        _availability_true(tiny_receipt, "logits")
        or _schema_availability(schema_artifact, "logits_available")
        or option_flags.get("logit_export_option") is True
    )
    attention = bool(
        _availability_true(tiny_receipt, "attention")
        or _schema_availability(schema_artifact, "attention_available")
        or option_flags.get("attention_export_option") is True
    )
    hidden_states = bool(
        _availability_true(tiny_receipt, "hidden_states")
        or _availability_true(tiny_receipt, "hidden_state_proxy")
        or _schema_availability(schema_artifact, "hidden_state_proxy_available")
    )
    token_timing = bool(
        _availability_true(tiny_receipt, "token_timing")
        or _schema_availability(schema_artifact, "token_timing_available")
        or option_flags.get("aggregate_timing_option") is True
    )
    prompt_completion_split = bool(
        _numeric(raw_output.get("tokens_evaluated")) is not None
        and _numeric(raw_output.get("tokens_predicted")) is not None
    )
    feature_flags = {
        "per_token_logprob_available": per_token,
        "topk_alternatives_available": topk,
        "logits_available": logits,
        "attention_available": attention,
        "hidden_states_available": hidden_states,
        "token_timing_available": token_timing,
        "prompt_completion_token_split_available": prompt_completion_split,
    }
    missing_names = [
        name
        for name, available in (
            ("per_token_logprob", per_token),
            ("topk_alternatives", topk),
            ("logits", logits),
            ("attention", attention),
            ("hidden_states", hidden_states),
            ("token_timing", token_timing),
            ("prompt_completion_token_split", prompt_completion_split),
        )
        if not available
    ]
    return {
        **feature_flags,
        "completion_probability_count": len(completion_rows),
        "top_logprob_row_count": len(top_rows),
        "token_probability_api_available": bool(per_token and tiny_receipt.get("endpoint")),
        "missing_feature_names": missing_names,
    }


def _token_checksum(row: Mapping[str, Any]) -> str | None:
    if row.get("token_checksum"):
        return str(row["token_checksum"])
    if row.get("token") is not None:
        return sha16(str(row["token"]))
    if row.get("id") is not None:
        return sha16(str(row["id"]))
    return None


def _normalise_top_alternatives(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    alternatives: list[JsonDict] = []
    for rank, row in enumerate(rows):
        logprob = _numeric(row.get("logprob"))
        if logprob is None:
            continue
        alternatives.append(
            {
                "rank": rank,
                "token_id": row.get("id"),
                "token_checksum": _token_checksum(row),
                "logprob": _round_float(logprob),
            }
        )
    return alternatives


def _receipt_prompt_split(receipt: Mapping[str, Any]) -> tuple[int | None, int | None]:
    response = receipt.get("response_json") if isinstance(receipt.get("response_json"), Mapping) else {}
    evaluated = response.get("tokens_evaluated", receipt.get("tokens_evaluated"))
    predicted = response.get("tokens_predicted", receipt.get("tokens_predicted"))
    evaluated_f = _numeric(evaluated)
    predicted_f = _numeric(predicted)
    return (
        int(evaluated_f) if evaluated_f is not None else None,
        int(predicted_f) if predicted_f is not None else None,
    )


def _receipt_token_timing_ms(receipt: Mapping[str, Any]) -> float | None:
    response = receipt.get("response_json") if isinstance(receipt.get("response_json"), Mapping) else {}
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    return _numeric(timings.get("predicted_per_token_ms"))


def build_feature_rows(
    raw_probe: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]] = AUDIT_PROMPTS
) -> list[JsonDict]:
    """Normalize live backend token-probability rows into a tiny receipt table."""

    receipts = raw_probe.get("prompt_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        receipts = raw_probe.get("case_receipts")
    receipts = receipts if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)) else []
    prompt_by_id = {str(prompt["prompt_id"]): prompt for prompt in prompts}
    rows: list[JsonDict] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            continue
        prompt_id = str(receipt.get("prompt_id") or receipt.get("case_id") or "")
        prompt = prompt_by_id.get(prompt_id, {})
        prompt_checksum = str(receipt.get("prompt_checksum") or sha16(str(prompt.get("prompt", ""))))
        prompt_tokens, completion_tokens = _receipt_prompt_split(receipt)
        timing_ms = _receipt_token_timing_ms(receipt)
        for token_index, completion_row in enumerate(_completion_probability_rows(receipt)):
            logprob = _numeric(completion_row.get("logprob"))
            if logprob is None:
                continue
            top_rows = completion_row.get("top_logprobs")
            if not isinstance(top_rows, Sequence) or isinstance(top_rows, (str, bytes)):
                top_rows = [completion_row]
            top_alternatives = _normalise_top_alternatives(
                [row for row in top_rows if isinstance(row, Mapping)]
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_checksum": prompt_checksum,
                    "token_index": token_index,
                    "token_id": completion_row.get("id"),
                    "token_checksum": _token_checksum(completion_row),
                    "logprob": _round_float(logprob),
                    "top_alternative_count": len(top_alternatives),
                    "top_alternatives": top_alternatives,
                    "token_timing_ms": _round_float(timing_ms),
                    "prompt_tokens_evaluated": prompt_tokens,
                    "completion_tokens_predicted": completion_tokens,
                    "feature_source": "backend_completion_probabilities",
                    "quality_interpretation": None,
                }
            )
    return rows


def _retired_scope_check(root: Path) -> JsonDict:
    manifest_path = root / "ops" / "exclusion_manifest.yaml"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        manifest_text = ""
    marker_present = "phase_d_external_text_scorer_retired" in manifest_text
    return {
        "manifest_path": str(manifest_path),
        "phase_d_external_text_scorer_retired_marker_present": marker_present,
        "external_text_scorer_reopened": False,
        "retired_scope_reopened": False,
    }


def _precondition_blockers(
    *,
    selected_model_spec: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if selected_model_spec is None:
        blockers.append("selected_model_spec_unavailable")
    elif selected_model_spec.get("hf_id") not in EXPECTED_MODEL_IDS:
        blockers.append("selected_model_not_mandated")
    else:
        model_path = selected_model_spec.get("model_path")
        if not model_path or not Path(str(model_path)).is_file():
            blockers.append("selected_model_file_missing")
        if selected_model_spec.get("autotokenizer_used") is True:
            blockers.append("autotokenizer_used_for_gguf")
    if preconditions.get("gpu_visible") is not True:
        blockers.append("gpu_not_visible")
    if feature_audit.get("per_token_logprob_available") is not True:
        blockers.append("per_token_logprob")
    if feature_audit.get("topk_alternatives_available") is not True:
        blockers.append("topk_alternatives")
    return list(dict.fromkeys(blockers))


def _build_preconditions_record(
    *,
    root: Path,
    exp5337_artifact_path: Path,
    exp5331_artifact_path: Path,
    exp5331_schema_path: Path,
    exp5331_tiny_receipt_path: Path,
    selected_backend_command: Mapping[str, Any] | None,
    selected_model_spec: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
    blockers: Sequence[str],
    live_probe_attempted: bool,
    raw_probe: Mapping[str, Any],
) -> JsonDict:
    binary_paths = preconditions.get("binary_paths")
    binary_paths = binary_paths if isinstance(binary_paths, Mapping) else {}
    backend_kind = raw_probe.get("backend_kind") or "llama-server"
    return {
        "exp5337_artifact_path": str(exp5337_artifact_path),
        "exp5331_artifact_path": str(exp5331_artifact_path),
        "exp5331_receipt_schema_path": str(exp5331_schema_path),
        "exp5331_tiny_receipt_path": str(exp5331_tiny_receipt_path),
        "gpu_visible": preconditions.get("gpu_visible"),
        "nvidia_smi": preconditions.get("nvidia_smi"),
        "free_vram_mb": preconditions.get("free_vram_mb"),
        "selected_runtime_backend_kind": (selected_backend_command or {}).get("backend_kind"),
        "selected_backend_kind": backend_kind,
        "selected_backend_path": binary_paths.get(str(backend_kind)),
        "selected_model_hf_id": (selected_model_spec or {}).get("hf_id"),
        "selected_model_path": (selected_model_spec or {}).get("model_path"),
        "selected_model_file_present": bool(
            selected_model_spec
            and selected_model_spec.get("model_path")
            and Path(str(selected_model_spec["model_path"])).is_file()
        ),
        "token_probability_api_available": feature_audit.get("token_probability_api_available")
        is True,
        "retired_scope_check": _retired_scope_check(root),
        "external_text_scorer_reopened": False,
        "blocked_preconditions": list(blockers),
        "live_probe_attempted": live_probe_attempted,
    }


def _honest_verdict(rows_ready: bool, blockers: Sequence[str], live_probe_attempted: bool) -> str:
    if rows_ready:
        return "complete: tokenprob_feature_rows_ready"
    if "per_token_logprob" in blockers:
        return "blocked_tokenprob_features_unavailable"
    if not live_probe_attempted and blockers:
        return "blocked_preconditions:" + ",".join(blockers)
    return "blocked_tokenprob_feature_rows_not_ready"


def build_artifact(
    *,
    root: Path,
    runtime_artifact: Mapping[str, Any],
    internal_artifact: Mapping[str, Any],
    schema_artifact: Mapping[str, Any],
    tiny_receipt: Mapping[str, Any],
    exp5337_artifact_path: Path,
    exp5331_artifact_path: Path,
    exp5331_schema_path: Path,
    exp5331_tiny_receipt_path: Path,
    preconditions: Mapping[str, Any],
    feature_probe: FeatureProbe,
    tests_run: Sequence[Any],
) -> JsonDict:
    """Build the terminal feature-audit artifact."""

    started = time.perf_counter()
    audit_started = time.perf_counter()
    model_specs = _model_specs_from_runtime(runtime_artifact)
    selected_model_spec = _selected_model_from_sources(
        runtime_artifact, internal_artifact, tiny_receipt, model_specs
    )
    selected_backend_command = _selected_backend_command(runtime_artifact)
    feature_audit = audit_backend_features(tiny_receipt, schema_artifact, internal_artifact)
    feature_audit_duration_s = round(time.perf_counter() - audit_started, 6)
    blockers = _precondition_blockers(
        selected_model_spec=selected_model_spec,
        preconditions=preconditions,
        feature_audit=feature_audit,
    )
    live_probe_attempted = not blockers
    raw_probe: JsonDict = {}
    feature_rows: list[JsonDict] = []
    if live_probe_attempted:
        raw_probe = dict(
            feature_probe(
                selected_model_spec=selected_model_spec,
                selected_backend_command=selected_backend_command,
                preconditions=preconditions,
                prompts=AUDIT_PROMPTS,
                n_probs=N_PROBS,
                n_predict=N_PREDICT,
            )
        )
        feature_rows = build_feature_rows(raw_probe, AUDIT_PROMPTS)
        feature_audit_duration_s = round(
            float(raw_probe.get("feature_audit_wall_clock_s") or feature_audit_duration_s), 6
        )
    methodology_duration_s = (
        round(float(raw_probe.get("wall_clock_s") or 0.0), 6)
        if live_probe_attempted
        else round(time.perf_counter() - started, 6)
    )
    observed_per_token = any(_numeric(row.get("logprob")) is not None for row in feature_rows)
    observed_topk = any(int(row.get("top_alternative_count") or 0) > 0 for row in feature_rows)
    per_token_available = bool(feature_audit["per_token_logprob_available"] and observed_per_token)
    topk_available = bool(feature_audit["topk_alternatives_available"] and observed_topk)
    missing_feature_names = list(blockers)
    if live_probe_attempted:
        if not observed_per_token:
            missing_feature_names.append("per_token_logprob")
        if not observed_topk:
            missing_feature_names.append("topk_alternatives")
        if methodology_duration_s <= 0:
            missing_feature_names.append("methodology_duration_s")
        if feature_audit_duration_s <= 0:
            missing_feature_names.append("feature_audit_duration_s")
        if not tests_run:
            missing_feature_names.append("tests_run")
    missing_feature_names = list(dict.fromkeys(missing_feature_names))
    tokenprob_feature_rows_ready = bool(
        live_probe_attempted
        and per_token_available
        and methodology_duration_s > 0
        and feature_audit_duration_s > 0
        and methodology_duration_s != feature_audit_duration_s
        and not missing_feature_names
        and tests_run
    )
    substrate = INFERENCE_SUBSTRATE_LIVE if live_probe_attempted else INFERENCE_SUBSTRATE_AUDIT_ONLY
    precondition_record = _build_preconditions_record(
        root=root,
        exp5337_artifact_path=exp5337_artifact_path,
        exp5331_artifact_path=exp5331_artifact_path,
        exp5331_schema_path=exp5331_schema_path,
        exp5331_tiny_receipt_path=exp5331_tiny_receipt_path,
        selected_backend_command=selected_backend_command,
        selected_model_spec=selected_model_spec,
        preconditions=preconditions,
        feature_audit=feature_audit,
        blockers=blockers,
        live_probe_attempted=live_probe_attempted,
        raw_probe=raw_probe,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", "complete" if tokenprob_feature_rows_ready else "blocked"),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(tokenprob_feature_rows_ready, missing_feature_names, live_probe_attempted),
        ),
        "inference_substrate": _wrap("inference_substrate", substrate),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap("preconditions_checked", precondition_record),
        "selected_model_spec": _wrap("selected_model_spec", selected_model_spec),
        "per_token_logprob_available": per_token_available,
        "topk_alternatives_available": topk_available,
        "logits_available": bool(feature_audit["logits_available"]),
        "attention_available": bool(feature_audit["attention_available"]),
        "hidden_states_available": bool(feature_audit["hidden_states_available"]),
        "token_timing_available": bool(feature_audit["token_timing_available"]),
        "prompt_completion_token_split_available": bool(
            feature_audit["prompt_completion_token_split_available"]
        ),
        "tokenprob_feature_row_count": len(feature_rows),
        "tokenprob_feature_rows": feature_rows,
        "feature_audit": feature_audit,
        "missing_feature_names": [] if tokenprob_feature_rows_ready else missing_feature_names,
        "methodology_duration_s": methodology_duration_s,
        "feature_audit_duration_s": feature_audit_duration_s,
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
        "tokenprob_feature_rows_ready": tokenprob_feature_rows_ready,
        "raw_probe_summary": {
            "status": raw_probe.get("status"),
            "backend_kind": raw_probe.get("backend_kind"),
            "endpoint": raw_probe.get("endpoint"),
            "prompt_receipt_count": len(raw_probe.get("prompt_receipts") or []),
            "quality_interpretation": None,
        },
        "prompt_count": len(AUDIT_PROMPTS) if live_probe_attempted else 0,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": FIELD_PRINCIPLES,
        "tests_run": _wrap("tests_run", list(tests_run)),
    }
    artifact["duration_s"] = methodology_duration_s
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "selected_model_spec": selected_model_spec,
                "feature_rows": feature_rows,
                "methodology_duration_s": methodology_duration_s,
                "feature_audit_duration_s": feature_audit_duration_s,
                "missing_feature_names": missing_feature_names,
            }
        )
    )
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    exp5337_artifact_path: Path | None = None,
    exp5331_artifact_path: Path | None = None,
    exp5331_schema_path: Path | None = None,
    exp5331_tiny_receipt_path: Path | None = None,
    preconditions_provider: PreconditionsProvider | None = None,
    feature_probe: FeatureProbe | None = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5353 and write the requested result artifact."""

    result_path = result_path or root / RESULT_RELATIVE_PATH
    exp5337_artifact_path = exp5337_artifact_path or root / exp5337.RESULT_RELATIVE_PATH
    exp5331_artifact_path = exp5331_artifact_path or root / exp5331.RESULT_RELATIVE_PATH
    exp5331_schema_path = exp5331_schema_path or root / exp5331.RECEIPT_SCHEMA_RELATIVE_PATH
    exp5331_tiny_receipt_path = (
        exp5331_tiny_receipt_path or root / exp5331.TINY_RECEIPT_RELATIVE_PATH
    )
    preconditions_provider = preconditions_provider or (lambda: exp5323.collect_preconditions(root))
    feature_probe = feature_probe or default_feature_probe
    artifact = build_artifact(
        root=root,
        runtime_artifact=_read_json(exp5337_artifact_path),
        internal_artifact=_read_json(exp5331_artifact_path),
        schema_artifact=_read_json(exp5331_schema_path),
        tiny_receipt=_read_json(exp5331_tiny_receipt_path),
        exp5337_artifact_path=exp5337_artifact_path,
        exp5331_artifact_path=exp5331_artifact_path,
        exp5331_schema_path=exp5331_schema_path,
        exp5331_tiny_receipt_path=exp5331_tiny_receipt_path,
        preconditions=dict(preconditions_provider()),
        feature_probe=feature_probe,
        tests_run=list(tests_run or []),
    )
    if write:
        _write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate fields downstream gates depend on."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    for field in REQUIRED_WRAPPED_FIELDS:
        value = artifact.get(field)
        if (
            not isinstance(value, Mapping)
            or value.get("principle") != FIELD_PRINCIPLES[field]
            or "value" not in value
        ):
            errors.append(f"{field} must be principle wrapped")

    if (artifact.get("experiment_id") or {}).get("value") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if (artifact.get("milestone") or {}).get("value") != MILESTONE:
        errors.append("milestone mismatch")
    status = (artifact.get("status") or {}).get("value")
    if status not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = (artifact.get("honest_verdict") or {}).get("value")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    substrate = (artifact.get("inference_substrate") or {}).get("value")
    if substrate not in ALLOWED_INFERENCE_SUBSTRATES:
        errors.append("inference_substrate must be live_llm_inference or feature_audit_only")

    for field in (
        "per_token_logprob_available",
        "topk_alternatives_available",
        "logits_available",
        "attention_available",
        "hidden_states_available",
        "external_text_scorer_reopened",
        "no_quality_claim",
        "tokenprob_feature_rows_ready",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare boolean")
    if artifact.get("external_text_scorer_reopened") is not False:
        errors.append("external_text_scorer_reopened must be bare false")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")

    row_count = artifact.get("tokenprob_feature_row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        errors.append("tokenprob_feature_row_count must be a bare integer")
    missing = artifact.get("missing_feature_names")
    if not isinstance(missing, list):
        errors.append("missing_feature_names must be a bare list")
        missing = []
    methodology_duration_s = artifact.get("methodology_duration_s")
    feature_audit_duration_s = artifact.get("feature_audit_duration_s")
    if not isinstance(methodology_duration_s, int | float) or isinstance(
        methodology_duration_s, bool
    ):
        errors.append("methodology_duration_s must be numeric")
    if not isinstance(feature_audit_duration_s, int | float) or isinstance(
        feature_audit_duration_s, bool
    ):
        errors.append("feature_audit_duration_s must be numeric")

    model_specs = (artifact.get("MODEL_SPECS") or {}).get("value")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        if set(model_specs) != set(EXPECTED_ROLES):
            errors.append("MODEL_SPECS roles mismatch")
        for role in set(model_specs) & set(EXPECTED_ROLES):
            row = model_specs[role]
            if not isinstance(row, Mapping) or row.get("hf_id") != EXPECTED_HF_BY_ROLE[role]:
                errors.append("MODEL_SPECS hf_id mismatch")
            if isinstance(row, Mapping) and row.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")
    selected = (artifact.get("selected_model_spec") or {}).get("value")
    if selected is not None:
        if not isinstance(selected, Mapping):
            errors.append("selected_model_spec must be null or object")
        elif selected.get("hf_id") not in EXPECTED_MODEL_IDS:
            errors.append("selected_model_spec must name a mandated model")
        elif selected.get("autotokenizer_used") is True:
            errors.append("selected_model_spec autotokenizer_used must stay false")
    tests_run = (artifact.get("tests_run") or {}).get("value")
    if not isinstance(tests_run, list):
        errors.append("tests_run must be a list")

    rows_ready = artifact.get("tokenprob_feature_rows_ready")
    if rows_ready is True:
        if status != "complete":
            errors.append("ready artifact must have complete status")
        if substrate != INFERENCE_SUBSTRATE_LIVE:
            errors.append("ready artifact must use live_llm_inference")
        if artifact.get("per_token_logprob_available") is not True:
            errors.append("ready artifact requires per_token_logprob_available")
        if row_count == 0:
            errors.append("ready artifact requires tokenprob_feature_row_count > 0")
        if missing:
            errors.append("ready artifact must not have missing_feature_names")
        if not tests_run:
            errors.append("ready artifact requires tests_run")
        if not isinstance(methodology_duration_s, int | float) or methodology_duration_s <= 0:
            errors.append("ready artifact requires positive methodology_duration_s")
        if not isinstance(feature_audit_duration_s, int | float) or feature_audit_duration_s <= 0:
            errors.append("ready artifact requires positive feature_audit_duration_s")
        if methodology_duration_s == feature_audit_duration_s:
            errors.append("duration fields must be independent")
    elif rows_ready is False:
        if status != "blocked":
            errors.append("blocked artifact must have blocked status")
    else:
        errors.append("tokenprob_feature_rows_ready must be a bare boolean")

    if errors:
        raise ValueError("; ".join(errors))


def default_feature_probe(
    *,
    selected_model_spec: Mapping[str, Any],
    selected_backend_command: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]] = AUDIT_PROMPTS,
    n_probs: int = N_PROBS,
    n_predict: int = N_PREDICT,
) -> JsonDict:  # pragma: no cover - live llama-server integration
    """Run a tiny live llama-server probe for backend token-probability rows."""

    _ = selected_backend_command
    binary_paths = preconditions.get("binary_paths")
    server = None
    if isinstance(binary_paths, Mapping):
        server = binary_paths.get("llama-server")
    if not server:
        return {"status": "blocked_llama_server_missing", "wall_clock_s": 0.0}
    port = _free_port()
    command = [
        str(server),
        "-m",
        str(selected_model_spec["model_path"]),
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "128",
        "-ngl",
        "all",
        "-sm",
        "layer",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "-np",
        "1",
        "--metrics",
        "--props",
        "--slots",
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    prompt_receipts: list[JsonDict] = []
    runtime_error: str | None = None
    feature_started = time.perf_counter()
    try:
        if not _wait_for_health(port, 180.0):
            runtime_error = "llama-server health endpoint did not become ready"
        else:
            feature_started = time.perf_counter()
            for prompt in prompts:
                request_started = time.perf_counter()
                response = _post_completion(
                    port,
                    str(prompt["prompt"]),
                    n_probs=n_probs,
                    n_predict=n_predict,
                    timeout_s=90.0,
                )
                prompt_receipts.append(
                    {
                        "prompt_id": str(prompt["prompt_id"]),
                        "prompt_checksum": sha16(str(prompt["prompt"])),
                        "wall_clock_s": round(time.perf_counter() - request_started, 6),
                        "response_json": response,
                    }
                )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return {
        "status": "completed" if runtime_error is None and prompt_receipts else "blocked_probe_failed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "command": command,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "feature_audit_wall_clock_s": round(time.perf_counter() - feature_started, 6),
        "prompt_receipts": prompt_receipts,
        "runtime_error": runtime_error,
    }


def _free_port() -> int:  # pragma: no cover - live integration helper
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout_s: float) -> bool:  # pragma: no cover
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib_request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib_error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def _post_completion(
    port: int,
    prompt: str,
    *,
    n_probs: int,
    n_predict: int,
    timeout_s: float,
) -> JsonDict:  # pragma: no cover
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
        "n_probs": n_probs,
    }
    req = urllib_request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", "replace")
    data = json.loads(body)
    return {
        "content_checksum": sha16(str(data.get("content", ""))),
        "tokens_predicted": data.get("tokens_predicted"),
        "tokens_evaluated": data.get("tokens_evaluated"),
        "timings": data.get("timings"),
        "completion_probabilities": data.get("completion_probabilities") or [],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--exp5337", type=Path, default=REPO_ROOT / exp5337.RESULT_RELATIVE_PATH)
    parser.add_argument("--exp5331", type=Path, default=REPO_ROOT / exp5331.RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--exp5331-schema", type=Path, default=REPO_ROOT / exp5331.RECEIPT_SCHEMA_RELATIVE_PATH
    )
    parser.add_argument(
        "--exp5331-tiny", type=Path, default=REPO_ROOT / exp5331.TINY_RECEIPT_RELATIVE_PATH
    )
    parser.add_argument("--tests-run-json", default="[]")
    args = parser.parse_args(argv)
    artifact = run(
        result_path=args.out,
        exp5337_artifact_path=args.exp5337,
        exp5331_artifact_path=args.exp5331,
        exp5331_schema_path=args.exp5331_schema,
        exp5331_tiny_receipt_path=args.exp5331_tiny,
        tests_run=json.loads(args.tests_run_json),
    )
    print(
        f"[exp5353] status={artifact['status']['value']} "
        f"rows_ready={artifact['tokenprob_feature_rows_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
