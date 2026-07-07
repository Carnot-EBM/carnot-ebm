#!/usr/bin/env python3
"""Exp5345: clean token-probability energy corrigendum.

Spec refs: REQ-VERIFY-5345, SCENARIO-VERIFY-5345.

This module measures a very small internal-signal receipt, not answer quality.
It only uses backend-provided token probabilities for known synthetic
arithmetic/factual target tokens. The generated text is never scored, reranked,
or converted into a hallucination claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import contextlib
import hashlib
import json
import math
from pathlib import Path
import re
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
PreconditionsProvider = Callable[[], JsonDict]
TokenProbabilityProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5345_tokenprob_energy_corrigendum_v487"
EXPERIMENT_NUMBER = 5345
MILESTONE = "2026.07.487"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5345.tokenprob_energy_corrigendum.v487"
RESULT_RELATIVE_PATH = Path("results/experiment_5345_tokenprob_energy_corrigendum_v487.json")
INFERENCE_SUBSTRATE_LIVE = "live_llm_inference"
INFERENCE_SUBSTRATE_AGGREGATION = "aggregation_from_upstream_artifacts"
ALLOWED_INFERENCE_SUBSTRATES = (INFERENCE_SUBSTRATE_LIVE, INFERENCE_SUBSTRATE_AGGREGATION)
SPEC_REFS = ("REQ-VERIFY-5345", "SCENARIO-VERIFY-5345")
TERMINAL_PREFIXES = ("complete:", "blocked_")
RANDOM_SEED = 5345
MIN_LIVE_DURATION_S = 60.0
N_PROBS = 20
N_PREDICT = 1
MANDATED_MODEL_SPECS = exp5323.MANDATED_MODEL_SPECS

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5345 token-probability energy corrigendum receipt.",
    "milestone": "Milestone accountability for the V487 token-probability energy corrigendum gate.",
    "status": "Machine-readable terminal state for downstream internal-energy corrigendum consumers.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether a clean "
        "token-probability energy diagnostic exists without reopening retired scorer scope."
    ),
    "inference_substrate": (
        "Declares whether Exp5345 ran live_llm_inference or only aggregation blocked before "
        "generation so duration checks match the actual substrate."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs so the diagnostic cannot silently "
        "substitute a legacy, tiny, API, or non-GGUF model."
    ),
    "selected_model_spec": (
        "Binds the token-probability receipt to the clean Exp5337 selected local GGUF "
        "model/backend."
    ),
    "token_energy_feature_rows": (
        "Records only transparent token-probability-derived energies for controlled "
        "known-target cases, not generated-text quality scores."
    ),
    "tests_run": (
        "Commands run to validate the Exp5345 module, artifact schema, new-code coverage, "
        "repository tests, and applicable e2e checks."
    ),
    "preconditions_checked": (
        "Records Exp5337 runtime cleanliness, current GPU/backend visibility, Exp5331 receipt "
        "schema, and no-retired-scope checks before any clean token-energy claim."
    ),
    "token_probability_receipt": (
        "Records the live probe receipt summary and upstream token-probability receipt paths "
        "without interpreting generated text as quality."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "selected_model_spec",
    "token_energy_feature_rows",
    "tests_run",
)
WRAPPED_FIELDS = REQUIRED_WRAPPED_FIELDS + ("preconditions_checked", "token_probability_receipt")
REQUIRED_ARTIFACT_FIELDS = WRAPPED_FIELDS + (
    "token_probability_available",
    "logits_available",
    "attention_available",
    "diagnostic_case_count",
    "methodology_duration_s",
    "external_text_scorer_reopened",
    "no_quality_claim",
    "internal_energy_corrigendum_clean",
)


@dataclass(frozen=True)
class DiagnosticCase:
    """One known-target prompt used only for token-probability feature extraction."""

    case_id: str
    domain: str
    prompt: str
    correct_aliases: tuple[str, ...]
    perturbed_aliases: tuple[str, ...]


DIAGNOSTIC_CASES: tuple[DiagnosticCase, ...] = (
    DiagnosticCase(
        case_id="arithmetic_true_2_plus_2",
        domain="arithmetic",
        prompt="Answer with one lowercase token, yes or no: 2 + 2 = 4. Answer:",
        correct_aliases=("yes", "true"),
        perturbed_aliases=("no", "false"),
    ),
    DiagnosticCase(
        case_id="arithmetic_false_3_plus_5",
        domain="arithmetic",
        prompt="Answer with one lowercase token, yes or no: 3 + 5 = 9. Answer:",
        correct_aliases=("no", "false"),
        perturbed_aliases=("yes", "true"),
    ),
    DiagnosticCase(
        case_id="factual_true_paris_capital",
        domain="factual",
        prompt="Answer with one lowercase token, yes or no: Paris is the capital of France. Answer:",
        correct_aliases=("yes", "true"),
        perturbed_aliases=("no", "false"),
    ),
    DiagnosticCase(
        case_id="factual_false_mars_star",
        domain="factual",
        prompt="Answer with one lowercase token, yes or no: Mars is a star. Answer:",
        correct_aliases=("no", "false"),
        perturbed_aliases=("yes", "true"),
    ),
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _round_float(value: float | None) -> float | None:
    return None if value is None else round(float(value), 9)


def _normalise_token(value: Any) -> str:
    text = re.sub(r"^[^\w+-]+", "", str(value or "")).strip().lower()
    return text.strip(".,:;!?\"'()[]{}")


def _normalise_aliases(values: Sequence[str]) -> set[str]:
    return {_normalise_token(value) for value in values}


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


def _selected_model_from_runtime(
    runtime_artifact: Mapping[str, Any], model_specs: Mapping[str, Any]
) -> JsonDict | None:
    receipt = _raw_or_wrapped_value(runtime_artifact, "runtime_corrigendum_receipt")
    role = (
        str((receipt or {}).get("model_role") or "flagship_dense")
        if isinstance(receipt, Mapping)
        else "flagship_dense"
    )
    selected = model_specs.get(role)
    if isinstance(selected, Mapping):
        return dict(selected)
    return None


def _selected_backend_command(runtime_artifact: Mapping[str, Any]) -> JsonDict | None:
    selected = _raw_or_wrapped_value(runtime_artifact, "selected_backend_command")
    if not isinstance(selected, Mapping):
        return None
    return dict(selected)


def _runtime_clean(runtime_artifact: Mapping[str, Any]) -> bool:
    return bool(
        _raw_or_wrapped_value(runtime_artifact, "sota_runtime_clean_receipt_ready") is True
        and _raw_or_wrapped_value(runtime_artifact, "runtime_unblocked_min_one_mandated") is True
        and _raw_or_wrapped_value(runtime_artifact, "inference_substrate")
        == INFERENCE_SUBSTRATE_LIVE
        and float(_raw_or_wrapped_value(runtime_artifact, "methodology_duration_s") or 0.0)
        >= MIN_LIVE_DURATION_S
    )


def _receipt_schema_clean(schema_artifact: Mapping[str, Any]) -> bool:
    availability = schema_artifact.get("availability")
    availability = availability if isinstance(availability, Mapping) else {}
    return bool(
        schema_artifact.get("schema") == exp5331.RECEIPT_SCHEMA
        and schema_artifact.get("internal_signal_receipt_ready") is True
        and schema_artifact.get("receipt_kind") == "token_probability"
        and availability.get("token_probability_available") is True
        and schema_artifact.get("external_text_scorer_reopened") is False
        and schema_artifact.get("no_quality_claim") is True
    )


def _tiny_receipt_clean(tiny_receipt: Mapping[str, Any]) -> bool:
    token_probability = tiny_receipt.get("token_probability")
    token_probability = token_probability if isinstance(token_probability, Mapping) else {}
    return bool(
        tiny_receipt.get("schema") == exp5331.TINY_RECEIPT_SCHEMA
        and tiny_receipt.get("receipt_kind") == "token_probability"
        and token_probability.get("availability") == "available"
        and tiny_receipt.get("quality_interpretation") is None
    )


def _internal_artifact_clean(internal_artifact: Mapping[str, Any]) -> bool:
    return bool(
        _raw_or_wrapped_value(internal_artifact, "token_probability_available") is True
        and _raw_or_wrapped_value(internal_artifact, "internal_signal_receipt_ready") is True
        and _raw_or_wrapped_value(internal_artifact, "external_text_scorer_reopened") is False
        and _raw_or_wrapped_value(internal_artifact, "no_quality_claim") is True
    )


def _server_path(preconditions: Mapping[str, Any]) -> str | None:
    binary_paths = preconditions.get("binary_paths")
    if isinstance(binary_paths, Mapping) and binary_paths.get("llama-server"):
        return str(binary_paths["llama-server"])
    return None


def _precondition_blockers(
    *,
    runtime_artifact: Mapping[str, Any],
    internal_artifact: Mapping[str, Any],
    schema_artifact: Mapping[str, Any],
    tiny_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    selected_model_spec: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if not _runtime_clean(runtime_artifact):
        blockers.append("exp5337_clean_runtime_unavailable")
    if not _internal_artifact_clean(internal_artifact):
        blockers.append("exp5331_token_probability_receipt_unavailable")
    if not _receipt_schema_clean(schema_artifact):
        blockers.append("exp5331_receipt_schema_not_token_probability")
    if not _tiny_receipt_clean(tiny_receipt):
        blockers.append("exp5331_tiny_token_probability_receipt_unavailable")
    if selected_model_spec is None:
        blockers.append("selected_model_spec_unavailable")
    else:
        model_path = selected_model_spec.get("model_path")
        if not model_path or not Path(str(model_path)).is_file():
            blockers.append("selected_model_file_missing")
    if preconditions.get("gpu_visible") is not True:
        blockers.append("gpu_not_visible")
    server = _server_path(preconditions)
    if not server or not Path(server).is_file():
        blockers.append("llama_server_binary_missing")
    if _raw_or_wrapped_value(internal_artifact, "external_text_scorer_reopened") is True:
        blockers.append("retired_external_text_scorer_reopened")
    if _raw_or_wrapped_value(internal_artifact, "no_quality_claim") is not True:
        blockers.append("quality_claim_reopened")
    return list(dict.fromkeys(blockers))


def _completion_probability_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    response = (
        receipt.get("response_json") if isinstance(receipt.get("response_json"), Mapping) else {}
    )
    rows = response.get("completion_probabilities") or receipt.get("completion_probabilities") or []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _top_logprob_rows(receipt: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for completion_row in _completion_probability_rows(receipt):
        top = completion_row.get("top_logprobs")
        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)):
            rows.extend(dict(row) for row in top if isinstance(row, Mapping))
        elif "token" in completion_row and "logprob" in completion_row:
            rows.append(completion_row)
    return rows


def _target_logprob(rows: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> float | None:
    target_aliases = _normalise_aliases(aliases)
    for row in rows:
        logprob = row.get("logprob")
        if _normalise_token(row.get("token")) in target_aliases and isinstance(
            logprob, int | float
        ):
            return float(logprob)
    return None


def _case_receipts_by_id(
    case_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(receipt.get("case_id")): receipt
        for receipt in case_receipts
        if isinstance(receipt, Mapping) and receipt.get("case_id")
    }


def build_token_energy_feature_rows(case_receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compute transparent negative-logprob energy rows for known targets."""

    receipts_by_id = _case_receipts_by_id(case_receipts)
    rows: list[JsonDict] = []
    for case in DIAGNOSTIC_CASES:
        receipt = receipts_by_id.get(case.case_id, {})
        top_rows = _top_logprob_rows(receipt)
        correct_logprob = _target_logprob(top_rows, case.correct_aliases)
        perturbed_logprob = _target_logprob(top_rows, case.perturbed_aliases)
        correct_energy = -correct_logprob if correct_logprob is not None else None
        perturbed_energy = -perturbed_logprob if perturbed_logprob is not None else None
        margin = (
            perturbed_energy - correct_energy
            if perturbed_energy is not None and correct_energy is not None
            else None
        )
        rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "prompt_checksum": sha16(case.prompt),
                "correct_target_aliases": list(case.correct_aliases),
                "perturbed_target_aliases": list(case.perturbed_aliases),
                "top_logprob_row_count": len(top_rows),
                "correct_target_logprob": _round_float(correct_logprob),
                "perturbed_target_logprob": _round_float(perturbed_logprob),
                "correct_token_energy": _round_float(correct_energy),
                "perturbed_token_energy": _round_float(perturbed_energy),
                "energy_margin_perturbed_minus_correct": _round_float(margin),
                "feature_complete": bool(
                    correct_logprob is not None and perturbed_logprob is not None
                ),
                "feature_source": "backend_top_logprobs",
                "quality_interpretation": None,
            }
        )
    return rows


def _feature_missing_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id"))
        if row.get("top_logprob_row_count") == 0:
            missing.append(f"{case_id}:completion_probabilities")
        if row.get("correct_target_logprob") is None:
            missing.append(f"{case_id}:correct_target_logprob")
        if row.get("perturbed_target_logprob") is None:
            missing.append(f"{case_id}:perturbed_target_logprob")
    return missing


def _token_probability_available(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(int(row.get("top_logprob_row_count") or 0) > 0 for row in rows)


def _receipt_summary(
    *,
    raw_probe: Mapping[str, Any],
    live_probe_attempted: bool,
    tiny_receipt_path: Path,
    schema_path: Path,
) -> JsonDict:
    return {
        "live_probe_attempted": live_probe_attempted,
        "backend_kind": raw_probe.get("backend_kind"),
        "endpoint": raw_probe.get("endpoint"),
        "status": raw_probe.get("status"),
        "round_count": raw_probe.get("round_count", 0),
        "case_receipt_count": len(raw_probe.get("case_receipts") or []),
        "wall_clock_s": raw_probe.get("wall_clock_s", 0.0),
        "exp5331_tiny_receipt_path": str(tiny_receipt_path),
        "exp5331_receipt_schema_path": str(schema_path),
        "raw_output_quality_interpretation": None,
    }


def _honest_verdict(
    clean: bool, live_probe_attempted: bool, missing_feature_names: Sequence[str]
) -> str:
    if clean:
        return "complete: token_probability_energy_corrigendum_clean"
    if not live_probe_attempted:
        return "blocked_preconditions:" + ",".join(
            missing_feature_names or ["unknown_precondition"]
        )
    if "methodology_duration_below_60s" in missing_feature_names:
        return "blocked_methodology_duration_below_60s"
    return "blocked_token_probability_energy_features_insufficient:" + ",".join(
        missing_feature_names or ["unknown_feature"]
    )


def _build_preconditions_record(
    *,
    exp5337_path: Path,
    exp5331_path: Path,
    schema_path: Path,
    tiny_receipt_path: Path,
    runtime_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    selected_model_spec: Mapping[str, Any] | None,
    blockers: Sequence[str],
    live_probe_attempted: bool,
) -> JsonDict:
    return {
        "exp5337_artifact_path": str(exp5337_path),
        "exp5337_clean_runtime": _runtime_clean(runtime_artifact),
        "exp5331_artifact_path": str(exp5331_path),
        "exp5331_receipt_schema_path": str(schema_path),
        "exp5331_tiny_receipt_path": str(tiny_receipt_path),
        "selected_model_hf_id": (selected_model_spec or {}).get("hf_id"),
        "selected_model_file_present": bool(
            selected_model_spec
            and selected_model_spec.get("model_path")
            and Path(str(selected_model_spec["model_path"])).is_file()
        ),
        "selected_backend_kind": (_selected_backend_command(runtime_artifact) or {}).get(
            "backend_kind"
        ),
        "gpu_visible": preconditions.get("gpu_visible"),
        "llama_server_path": _server_path(preconditions),
        "external_text_scorer_reopened": False,
        "retired_scope_reopened": False,
        "blocked_preconditions": list(blockers),
        "no_live_generation_reason": None
        if live_probe_attempted
        else ",".join(blockers or ["preconditions_failed"]),
    }


def build_artifact(
    *,
    runtime_artifact: Mapping[str, Any],
    internal_artifact: Mapping[str, Any],
    schema_artifact: Mapping[str, Any],
    tiny_receipt: Mapping[str, Any],
    exp5337_artifact_path: Path,
    exp5331_artifact_path: Path,
    exp5331_schema_path: Path,
    exp5331_tiny_receipt_path: Path,
    preconditions: Mapping[str, Any],
    token_probability_probe: TokenProbabilityProbe,
    tests_run: Sequence[Any],
) -> JsonDict:
    """Build the terminal artifact, running the live probe only after gates pass."""

    started = time.perf_counter()
    model_specs = _model_specs_from_runtime(runtime_artifact)
    selected_model_spec = _selected_model_from_runtime(runtime_artifact, model_specs)
    selected_backend_command = _selected_backend_command(runtime_artifact)
    blockers = _precondition_blockers(
        runtime_artifact=runtime_artifact,
        internal_artifact=internal_artifact,
        schema_artifact=schema_artifact,
        tiny_receipt=tiny_receipt,
        preconditions=preconditions,
        selected_model_spec=selected_model_spec,
    )
    live_probe_attempted = not blockers
    raw_probe: JsonDict = {}
    if live_probe_attempted:
        raw_probe = dict(
            token_probability_probe(
                selected_model_spec=selected_model_spec,
                selected_backend_command=selected_backend_command,
                preconditions=preconditions,
                diagnostic_cases=DIAGNOSTIC_CASES,
                minimum_duration_s=MIN_LIVE_DURATION_S,
            )
        )
    case_receipts = (
        raw_probe.get("case_receipts") if isinstance(raw_probe.get("case_receipts"), list) else []
    )
    feature_rows = build_token_energy_feature_rows(case_receipts) if live_probe_attempted else []
    methodology_duration_s = (
        round(float(raw_probe.get("wall_clock_s") or 0.0), 6) if live_probe_attempted else 0.0
    )
    token_probability_available = live_probe_attempted and _token_probability_available(
        feature_rows
    )
    missing_feature_names = list(blockers)
    if live_probe_attempted:
        if methodology_duration_s < MIN_LIVE_DURATION_S:
            missing_feature_names.append("methodology_duration_below_60s")
        if not token_probability_available:
            missing_feature_names.append("token_probability_top_logprobs_unavailable")
        missing_feature_names.extend(_feature_missing_names(feature_rows))
    if not tests_run:
        missing_feature_names.append("tests_run_unrecorded")
    missing_feature_names = list(dict.fromkeys(missing_feature_names))
    features_complete = bool(
        feature_rows
        and len(feature_rows) == len(DIAGNOSTIC_CASES)
        and all(row.get("feature_complete") is True for row in feature_rows)
    )
    clean = bool(
        live_probe_attempted
        and methodology_duration_s >= MIN_LIVE_DURATION_S
        and token_probability_available
        and features_complete
        and not missing_feature_names
    )
    substrate = (
        INFERENCE_SUBSTRATE_LIVE if live_probe_attempted else INFERENCE_SUBSTRATE_AGGREGATION
    )
    precondition_record = _build_preconditions_record(
        exp5337_path=exp5337_artifact_path,
        exp5331_path=exp5331_artifact_path,
        schema_path=exp5331_schema_path,
        tiny_receipt_path=exp5331_tiny_receipt_path,
        runtime_artifact=runtime_artifact,
        preconditions=preconditions,
        selected_model_spec=selected_model_spec,
        blockers=blockers,
        live_probe_attempted=live_probe_attempted,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_NAME),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", "complete" if clean else "blocked"),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(clean, live_probe_attempted, missing_feature_names),
        ),
        "inference_substrate": _wrap("inference_substrate", substrate),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "selected_model_spec": _wrap("selected_model_spec", selected_model_spec),
        "preconditions_checked": _wrap("preconditions_checked", precondition_record),
        "token_probability_receipt": _wrap(
            "token_probability_receipt",
            _receipt_summary(
                raw_probe=raw_probe,
                live_probe_attempted=live_probe_attempted,
                tiny_receipt_path=exp5331_tiny_receipt_path,
                schema_path=exp5331_schema_path,
            ),
        ),
        "token_probability_available": token_probability_available,
        "logits_available": False,
        "attention_available": False,
        "diagnostic_case_count": len(feature_rows),
        "token_energy_feature_rows": _wrap("token_energy_feature_rows", feature_rows),
        "methodology_duration_s": methodology_duration_s,
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
        "internal_energy_corrigendum_clean": clean,
        "missing_feature_names": missing_feature_names,
        "diagnostic_cases": [
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "prompt_checksum": sha16(case.prompt),
                "correct_aliases": list(case.correct_aliases),
                "perturbed_aliases": list(case.perturbed_aliases),
            }
            for case in DIAGNOSTIC_CASES
        ],
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_run": _wrap("tests_run", list(tests_run)),
    }
    artifact["duration_s"] = (
        methodology_duration_s if live_probe_attempted else round(time.perf_counter() - started, 6)
    )
    artifact["reproducibility_checksum"] = sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_NAME,
                "selected_model_spec": selected_model_spec,
                "feature_rows": feature_rows,
                "methodology_duration_s": methodology_duration_s,
                "missing_feature_names": missing_feature_names,
                "seed": RANDOM_SEED,
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
    token_probability_probe: TokenProbabilityProbe = None,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5345 and write the requested result artifact."""

    result_path = result_path or root / RESULT_RELATIVE_PATH
    exp5337_artifact_path = exp5337_artifact_path or root / exp5337.RESULT_RELATIVE_PATH
    exp5331_artifact_path = exp5331_artifact_path or root / exp5331.RESULT_RELATIVE_PATH
    exp5331_schema_path = exp5331_schema_path or root / exp5331.RECEIPT_SCHEMA_RELATIVE_PATH
    exp5331_tiny_receipt_path = (
        exp5331_tiny_receipt_path or root / exp5331.TINY_RECEIPT_RELATIVE_PATH
    )
    preconditions_provider = preconditions_provider or (lambda: exp5323.collect_preconditions(root))
    token_probability_probe = token_probability_probe or default_token_probability_probe
    artifact = build_artifact(
        runtime_artifact=_read_json(exp5337_artifact_path),
        internal_artifact=_read_json(exp5331_artifact_path),
        schema_artifact=_read_json(exp5331_schema_path),
        tiny_receipt=_read_json(exp5331_tiny_receipt_path),
        exp5337_artifact_path=exp5337_artifact_path,
        exp5331_artifact_path=exp5331_artifact_path,
        exp5331_schema_path=exp5331_schema_path,
        exp5331_tiny_receipt_path=exp5331_tiny_receipt_path,
        preconditions=dict(preconditions_provider()),
        token_probability_probe=token_probability_probe,
        tests_run=list(tests_run or []),
    )
    if write:
        _write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate fields that downstream artifact gates depend on."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if (
            not isinstance(value, Mapping)
            or value.get("principle") != FIELD_PRINCIPLES[field]
            or "value" not in value
        ):
            errors.append(f"{field} must be principle wrapped")

    if (artifact.get("experiment_id") or {}).get("value") != EXPERIMENT_NAME:
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
        errors.append(
            "inference_substrate must be live_llm_inference or aggregation_from_upstream_artifacts"
        )

    for field in (
        "token_probability_available",
        "logits_available",
        "attention_available",
        "external_text_scorer_reopened",
        "no_quality_claim",
        "internal_energy_corrigendum_clean",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare boolean")
    if artifact.get("external_text_scorer_reopened") is not False:
        errors.append("external_text_scorer_reopened must be bare false")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")
    if not isinstance(artifact.get("diagnostic_case_count"), int) or isinstance(
        artifact.get("diagnostic_case_count"), bool
    ):
        errors.append("diagnostic_case_count must be a bare integer")
    if not isinstance(artifact.get("methodology_duration_s"), int | float):
        errors.append("methodology_duration_s must be numeric")

    model_specs = (artifact.get("MODEL_SPECS") or {}).get("value")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles:
            errors.append("MODEL_SPECS roles mismatch")
        expected_hf = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
        for role in expected_roles & set(model_specs):
            row = model_specs[role]
            if not isinstance(row, Mapping) or row.get("hf_id") != expected_hf[role]:
                errors.append("MODEL_SPECS hf_id mismatch")
            if isinstance(row, Mapping) and row.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    selected = (artifact.get("selected_model_spec") or {}).get("value")
    if selected is not None and not isinstance(selected, Mapping):
        errors.append("selected_model_spec must be null or object")
    rows = (artifact.get("token_energy_feature_rows") or {}).get("value")
    if not isinstance(rows, list):
        errors.append("token_energy_feature_rows must be a principle-wrapped list")
        rows = []
    tests_run = (artifact.get("tests_run") or {}).get("value")
    if not isinstance(tests_run, list):
        errors.append("tests_run must be a list")

    clean = artifact.get("internal_energy_corrigendum_clean")
    duration = artifact.get("methodology_duration_s")
    missing = artifact.get("missing_feature_names", [])
    if clean is True:
        if status != "complete":
            errors.append("clean artifact must have complete status")
        if substrate != INFERENCE_SUBSTRATE_LIVE:
            errors.append("clean artifact must use live_llm_inference")
        if not isinstance(duration, int | float) or duration < MIN_LIVE_DURATION_S:
            errors.append("clean artifact requires methodology_duration_s >= 60")
        if artifact.get("token_probability_available") is not True:
            errors.append("clean artifact requires token_probability_available")
        if artifact.get("diagnostic_case_count") != len(DIAGNOSTIC_CASES):
            errors.append("clean artifact requires all diagnostic cases")
        if missing:
            errors.append("clean artifact must not have missing_feature_names")
        if not tests_run:
            errors.append("clean artifact requires tests_run")
        if not rows or not all(
            isinstance(row, Mapping) and row.get("feature_complete") is True for row in rows
        ):
            errors.append("clean artifact requires complete token energy rows")
    elif clean is False:
        if status != "blocked":
            errors.append("blocked artifact must have blocked status")
    else:
        errors.append("internal_energy_corrigendum_clean must be a bare boolean")

    if errors:
        raise ValueError("; ".join(errors))


def default_token_probability_probe(
    *,
    selected_model_spec: Mapping[str, Any],
    selected_backend_command: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    diagnostic_cases: Sequence[DiagnosticCase] = DIAGNOSTIC_CASES,
    minimum_duration_s: float = MIN_LIVE_DURATION_S,
) -> JsonDict:  # pragma: no cover - live llama-server integration
    """Run a real llama-server token-probability probe until the duration floor."""

    _ = selected_backend_command
    server = _server_path(preconditions)
    if not server:
        return {"status": "blocked_llama_server_missing", "wall_clock_s": 0.0, "case_receipts": []}
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
    case_receipts: list[JsonDict] = []
    all_request_count = 0
    runtime_error: str | None = None
    try:
        if not _wait_for_health(port, 180.0):
            runtime_error = "llama-server health endpoint did not become ready"
        else:
            round_count = 0
            while time.perf_counter() - started < minimum_duration_s or round_count == 0:
                round_count += 1
                for case in diagnostic_cases:
                    response = _post_completion(port, case.prompt, timeout_s=90.0)
                    all_request_count += 1
                    if round_count == 1:
                        case_receipts.append(
                            {
                                "case_id": case.case_id,
                                "prompt": case.prompt,
                                "response_json": response,
                            }
                        )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        round_count = len(case_receipts) // max(1, len(diagnostic_cases))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return {
        "status": "completed"
        if runtime_error is None and case_receipts
        else "blocked_probe_failed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "command": command,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "round_count": round_count,
        "all_request_count": all_request_count,
        "case_receipts": case_receipts,
        "runtime_error": runtime_error,
    }


def _free_port() -> int:  # pragma: no cover - live integration helper
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(
    port: int, timeout_s: float
) -> bool:  # pragma: no cover - live integration helper
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib_request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib_error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def _post_completion(port: int, prompt: str, timeout_s: float) -> JsonDict:  # pragma: no cover
    payload = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "cache_prompt": False,
        "n_probs": N_PROBS,
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
        "content": data.get("content", ""),
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
        f"[exp5345] status={artifact['status']['value']} "
        f"clean={artifact['internal_energy_corrigendum_clean']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
