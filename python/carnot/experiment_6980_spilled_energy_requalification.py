"""Requalify spilled energy from frozen adjacent-step logit statistics.

The reducer reads no model. It localizes the semantic values in each recorded
ConstraintIR mapping, reproduces the paper energies from stored scalars, and
uses exact certification only as an external error label.

Spec refs: REQ-VERIFY-6980 and SCENARIO-VERIFY-6980-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
MetricFunction = Callable[[Sequence[Mapping[str, Any]], str, Mapping[str, Any]], float | None]

EXPERIMENT_ID = 6980
RUN_DATE = "20260904"
RANDOM_SEED = 6_980_202_609_04
SCHEMA = "carnot.exp6980.spilled_energy_requalification.v1"
INFERENCE_SUBSTRATE = "deterministic_recorded_logit_energy_reducer"
RESULT_PATH = Path("results/experiment_6980_spilled_energy_requalification.json")
BANK_PATH = Path("results/experiment_6975_delayed_constraint_candidate_bank.json")
CERTIFICATION_PATH = Path("results/experiment_6976_exact_candidate_certification.json")
PRIOR_PATH = Path("results/experiment_2497_phase4_spilled_energy.json")

SIGNALS = ("spilled_energy", "marginalized_energy", "entropy", "top_probability")
CONTROL_SIGNALS = ("entropy", "top_probability")
MODEL_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SCHEDULES = ("direct", "trigger_switched", "draft_conditioned")
FORMULATION_FAMILIES = (
    "bounded_integer_linear",
    "boolean_cardinality",
    "bounded_piecewise_linear",
)
TRACE_ABS_TOLERANCE = 1e-12
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_CONFIDENCE = 0.95

REQUIRED_TRACE_FIELDS = (
    "emitted_token_id",
    "selected_token_logit",
    "selected_token_logprob",
    "full_vocabulary_logsumexp",
    "full_vocabulary_size",
    "finite_logit_count",
    "full_vocabulary_logits_sha256",
    "entropy",
    "top_probability",
    "phase_id",
    "phase_step_index",
    "attempt_step_index",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "formula_config",
    "calibration_rows",
    "rows",
    "per_span_results",
    "trace_reproduction_rows",
    "heldout_metric_rows",
    "per_model_metric_rows",
    "per_schedule_metric_rows",
    "per_formulation_metric_rows",
    "per_error_class_metric_rows",
    "control_comparison_rows",
    "bootstrap_interval_rows",
    "coverage_rows",
    "abstention_rows",
    "prior_verdict_comparison",
    "spilled_energy_evaluation_complete_score",
    "spilled_energy_requalified_score",
    "retirement_recommendation",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents incompatible result shapes from being compared.",
    "experiment_id": "A stable identifier binds this record to its preregistered task.",
    "run_date": "The execution date distinguishes this reduction from later evidence.",
    "field_principles": "One reason per field makes omissions and weak evidence visible.",
    "preconditions_checked": "Expected and observed values show whether frozen evidence was usable.",
    "inference_substrate": "The substrate states that this run reduced recorded logits without model inference.",
    "duration_s": "Measured wall time makes the deterministic execution auditable.",
    "source_artifact_hashes": "Hashes bind every conclusion to immutable upstream bytes.",
    "formula_config": "A frozen configuration prevents held-out results from choosing the analysis.",
    "calibration_rows": "Calibration receipts prove that labels selected signs and thresholds only before testing.",
    "rows": "One terminal denominator row prevents malformed candidates from disappearing.",
    "per_span_results": "Span-level signals let every aggregate be rebuilt from local evidence.",
    "trace_reproduction_rows": "Reproduction residuals reject scalar traces that cannot recreate log probability.",
    "heldout_metric_rows": "Overall held-out rows state the one-shot scientific result for every signal.",
    "per_model_metric_rows": "Model rows show whether an aggregate hides family-specific failure.",
    "per_schedule_metric_rows": "Schedule rows test whether prompt timing changes the diagnostic behavior.",
    "per_formulation_metric_rows": "Formulation rows expose task-family dependence.",
    "per_error_class_metric_rows": "Error-class rows separate distinct exact failure mechanisms.",
    "control_comparison_rows": "Paired control deltas test whether spilled energy adds information.",
    "bootstrap_interval_rows": "Clustered intervals quantify uncertainty without splitting shared pair evidence.",
    "coverage_rows": "Coverage states how much of the frozen bank supported exact span scoring.",
    "abstention_rows": "Explicit abstentions preserve missing and malformed cases in the record.",
    "prior_verdict_comparison": "The prior null makes this final requalification decision irreversible.",
    "spilled_energy_evaluation_complete_score": "One means every candidate and aggregate reached a terminal state.",
    "spilled_energy_requalified_score": "One requires the full preregistered held-out superiority gate.",
    "retirement_recommendation": "A terminal null must close this diagnostic instead of creating another retry.",
    "random_seed": "One seed fixes every pair-cluster bootstrap draw.",
    "reproducibility_checksum": "A timing-free digest detects any scientific payload drift.",
    "gate_check_summary": "Named failures retain both sides of each blocked comparison.",
    "verifier_is_oracle": "False records that exact checking labels the signal but never selects candidates.",
    "verdict_class": "A closed class prevents favorable prose from disguising a null or blocked run.",
    "honest_verdict": "A class-consistent prefix gives automation an unambiguous terminal result.",
}

BASE_FORMULA_CONFIG = {
    "paper": "Spilled Energy in Large Language Models, ICLR 2026, Eq. 7-8",
    "paper_url": "https://arxiv.org/abs/2602.18671",
    "temperature": 1.0,
    "formula_variant": "paper_official_code_token_energy_minus_next_step_marginalized_energy",
    "token_energy_formula": "-selected_token_logit[i]",
    "marginalized_energy_formula": "-full_vocabulary_logsumexp[i+1]",
    "spilled_energy_formula": "full_vocabulary_logsumexp[i+1]-selected_token_logit[i]",
    "entropy_formula": "recorded_full_vocabulary_entropy[i]",
    "top_probability_formula": "recorded_max_softmax_probability[i]",
    "mapping_span": "utf8_hull_of_top_level_objective_map_and_variable_map_values",
    "pooling": "arithmetic_mean",
    "missing_step_policy": "abstain_entire_span",
    "trace_abs_tolerance": TRACE_ABS_TOLERANCE,
    "directionality": "one_sign_per_signal_selected_on_calibration_only",
    "threshold_selection": "maximize_balanced_accuracy_at_observed_signed_score_cutpoints",
    "threshold_tie_break": "positive_sign_then_lower_threshold",
    "degenerate_calibration_policy": "observed_class_recall_then_registered_tie_break",
    "family_aggregation": "micro_average_all_eligible_heldout_spans",
    "bootstrap_unit": "pair_id",
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
    "requalification_gate": {
        "heldout_spilled_auroc_at_least": 0.65,
        "spilled_auroc_interval_excludes": 0.50,
        "paired_delta_lower_above": 0.0,
        "required_controls": list(CONTROL_SIGNALS),
        "minimum_model_families_beating_both_controls": 2,
    },
    "selection_split": "calibration",
    "evaluation_split": "heldout",
    "configuration_frozen_before_heldout": True,
}


class TraceError(ValueError):
    """Report a trace defect that makes a span score inadmissible."""


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for hashes and equality checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a repository-style SHA-256 string for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash one file while keeping absence visible to preflight."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def sha256_text(value: str) -> str:
    """Hash the UTF-8 bytes that the token spans address."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value with stable key ordering."""

    return sha256_bytes(canonical_json(value))


def _sha256_format(value: Any) -> bool:
    """Accept only an explicit lower-case SHA-256 string."""

    text = str(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text[7:])
    )


def _finite_number(value: Any) -> bool:
    """Return true only for a finite real scalar, excluding booleans."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object and reject arrays or scalar placeholders."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_inputs(repo_root: Path) -> JsonDict:
    """Load the three frozen inputs without altering their tracked files."""

    paths = {
        "bank": repo_root / BANK_PATH,
        "certification": repo_root / CERTIFICATION_PATH,
        "prior": repo_root / PRIOR_PATH,
    }
    loaded: JsonDict = {"paths": paths, "source_file_hashes": {}}
    for name, path in paths.items():
        loaded["source_file_hashes"][name] = sha256_path(path)
        try:
            loaded[name] = _read_object(path)
        except (OSError, json.JSONDecodeError, ValueError):
            loaded[name] = {}
    return loaded


def source_artifact_hashes(repo_root: Path, inputs: Mapping[str, Any]) -> JsonDict:
    """Bind upstream data and the complete reducer contract to exact bytes."""

    return {
        "experiment_6975": inputs.get("source_file_hashes", {}).get("bank"),
        "experiment_6976": inputs.get("source_file_hashes", {}).get("certification"),
        "experiment_2497": inputs.get("source_file_hashes", {}).get("prior"),
        "module": sha256_path(
            repo_root / "python/carnot/experiment_6980_spilled_energy_requalification.py"
        ),
        "test": sha256_path(
            repo_root / "tests/python/test_experiment_6980_spilled_energy_requalification.py"
        ),
        "wrapper": sha256_path(
            repo_root / "scripts/experiments/experiment_6980_spilled_energy_requalification.py"
        ),
        "spec": sha256_path(repo_root / "openspec/capabilities/verification/spec.md"),
        "paper_official_code": sha256_bytes(
            b"https://github.com/OmnAI-Lab/spilled-energy/src/spilled_energy/energy.py"
        ),
    }


def _check(name: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact precondition comparison row."""

    return {
        "check": name,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
    }


def _attempt_index(bank: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index valid attempt objects without hiding duplicate keys."""

    rows = bank.get("per_attempt_rows", [])
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return {}
    keys = [str(row.get("attempt_key")) for row in rows]
    if len(set(keys)) != len(keys):
        return {}
    return {str(row["attempt_key"]): row for row in rows}


def _certification_index(certification: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Index exact outcome rows only when every key is unique."""

    rows = certification.get("per_candidate_rows", [])
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return {}
    keys = [str(row.get("attempt_key")) for row in rows]
    if len(set(keys)) != len(keys):
        return {}
    return {str(row["attempt_key"]): row for row in rows}


def _trace_fields_complete(row: Mapping[str, Any]) -> bool:
    """Check that one trace has every scalar needed by the frozen formulas."""

    if not all(field in row for field in REQUIRED_TRACE_FIELDS):
        return False
    scalar_fields = (
        "selected_token_logit",
        "selected_token_logprob",
        "full_vocabulary_logsumexp",
        "entropy",
        "top_probability",
    )
    if not all(_finite_number(row.get(field)) for field in scalar_fields):
        return False
    vocabulary_size = row.get("full_vocabulary_size")
    finite_count = row.get("finite_logit_count")
    if (
        not isinstance(vocabulary_size, int)
        or isinstance(vocabulary_size, bool)
        or not isinstance(finite_count, int)
        or isinstance(finite_count, bool)
        or vocabulary_size <= 0
        or finite_count <= 0
        or finite_count > vocabulary_size
    ):
        return False
    return (
        isinstance(row.get("emitted_token_id"), int)
        and 0 <= int(row["emitted_token_id"]) < vocabulary_size
        and 0.0 <= float(row["top_probability"]) <= 1.0
        and float(row["entropy"]) >= 0.0
        and _sha256_format(row.get("full_vocabulary_logits_sha256"))
        and isinstance(row.get("phase_step_index"), int)
        and isinstance(row.get("attempt_step_index"), int)
    )


def _span_fields_complete(row: Mapping[str, Any]) -> bool:
    """Check one recorded token interval for exact integer byte offsets."""

    fields = (
        "phase_byte_start",
        "phase_byte_end",
        "attempt_byte_start",
        "attempt_byte_end",
        "phase_step_index",
        "attempt_step_index",
        "emitted_token_id",
    )
    return all(
        isinstance(row.get(field), int) and not isinstance(row.get(field), bool) for field in fields
    ) and (
        int(row["phase_byte_start"]) <= int(row["phase_byte_end"])
        and int(row["attempt_byte_start"]) <= int(row["attempt_byte_end"])
    )


def _mapping_span_observation(attempts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize exact semantic spans for every syntax-valid candidate."""

    parsed_count = 0
    valid_count = 0
    for attempt in attempts.values():
        diagnostic = attempt.get("parser_diagnostic", {})
        if (
            not isinstance(diagnostic, Mapping)
            or diagnostic.get("constraintir_shape_valid") is not True
        ):
            continue
        parsed_count += 1
        phase_id = attempt.get("candidate_phase_id")
        raw = str(attempt.get("candidate_raw_text", ""))
        mapping_span = extract_mapping_span(raw)
        phases = attempt.get("phase_outputs", [])
        phase = next(
            (row for row in phases if isinstance(row, Mapping) and row.get("phase_id") == phase_id),
            None,
        )
        if phase is None or mapping_span is None or phase.get("raw_text") != raw:
            continue
        spans = phase.get("token_span_rows", [])
        if (
            not isinstance(spans, list)
            or not spans
            or not all(isinstance(row, Mapping) and _span_fields_complete(row) for row in spans)
        ):
            continue
        start, end = mapping_span
        raw_length = len(raw.encode("utf-8"))
        if 0 <= start < end <= raw_length and any(
            int(row["phase_byte_end"]) > start and int(row["phase_byte_start"]) < end
            for row in spans
        ):
            valid_count += 1
    return {"parsed_candidate_count": parsed_count, "valid_mapping_span_count": valid_count}


def _candidate_hash_observation(
    attempts: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Compare candidate bytes and hashes across both frozen artifacts."""

    common = sorted(set(attempts) & set(candidates))
    matches = 0
    for key in common:
        attempt = attempts[key]
        candidate = candidates[key]
        raw = str(attempt.get("candidate_raw_text", ""))
        saved_hash = attempt.get("candidate_raw_sha256")
        if saved_hash == sha256_text(raw) and saved_hash == candidate.get("raw_sha256"):
            matches += 1
    return {
        "attempt_count": len(attempts),
        "candidate_count": len(candidates),
        "common_count": len(common),
        "matching_count": matches,
    }


def check_preconditions(inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Check all frozen evidence before any scientific metric is computed."""

    bank = inputs.get("bank", {})
    certification = inputs.get("certification", {})
    if not isinstance(bank, Mapping):
        bank = {}
    if not isinstance(certification, Mapping):
        certification = {}
    attempts = _attempt_index(bank)
    candidates = _certification_index(certification)
    source_hashes = inputs.get("source_file_hashes", {})
    available = {
        name: _sha256_format(source_hashes.get(name)) for name in ("bank", "certification", "prior")
    }
    checks = [
        _check("source_artifacts_available", {name: True for name in available}, available),
        _check("candidate_bank_complete_score", 1, bank.get("candidate_bank_complete_score")),
        _check(
            "candidate_certification_complete_score",
            1,
            certification.get("candidate_certification_complete_score"),
        ),
    ]
    model_observed = sorted({str(row.get("hf_id")) for row in attempts.values()})
    checks.append(_check("all_three_model_families", sorted(MODEL_FAMILIES), model_observed))
    checks.append(
        _check(
            "candidate_hash_agreement",
            {
                "attempt_count": 108,
                "candidate_count": 108,
                "common_count": 108,
                "matching_count": 108,
            },
            _candidate_hash_observation(attempts, candidates),
        )
    )
    traces = [
        trace
        for attempt in attempts.values()
        for trace in attempt.get("energy_trace_rows", [])
        if isinstance(trace, Mapping)
    ]
    span_rows = [
        span
        for attempt in attempts.values()
        for span in attempt.get("token_span_rows", [])
        if isinstance(span, Mapping)
    ]
    trace_observed = {
        "trace_count": len(traces),
        "span_count": len(span_rows),
        "all_trace_fields_complete": bool(traces)
        and all(_trace_fields_complete(row) for row in traces),
        "all_span_fields_complete": bool(span_rows)
        and all(_span_fields_complete(row) for row in span_rows),
    }
    checks.append(
        _check(
            "sufficient_energy_trace_fields",
            {
                "trace_count": len(traces),
                "span_count": len(traces),
                "all_trace_fields_complete": True,
                "all_span_fields_complete": True,
            },
            trace_observed,
        )
    )
    mapping_observed = _mapping_span_observation(attempts)
    checks.append(
        _check(
            "exact_mapping_span_offsets",
            {
                "parsed_candidate_count": mapping_observed["parsed_candidate_count"],
                "valid_mapping_span_count": mapping_observed["parsed_candidate_count"],
            },
            mapping_observed,
        )
    )
    heldout = [row for row in candidates.values() if row.get("split") == "heldout"]
    label_observed = {
        "heldout_count": len(heldout),
        "all_labels_terminal": bool(heldout)
        and all(type(row.get("exact_semantic_success")) is bool for row in heldout),
    }
    checks.append(
        _check(
            "heldout_exact_labels",
            {"heldout_count": 54, "all_labels_terminal": True},
            label_observed,
        )
    )
    return checks


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project failed preconditions into the required compact gate schema."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def _skip_ws(text: str, offset: int) -> int:
    """Advance over JSON whitespace from one character offset."""

    while offset < len(text) and text[offset] in " \t\r\n":
        offset += 1
    return offset


def _top_level_value_spans(raw_text: str) -> dict[str, tuple[int, int]]:
    """Return exact character spans for each top-level JSON value."""

    decoder = json.JSONDecoder()
    offset = _skip_ws(raw_text, 0)
    if offset >= len(raw_text) or raw_text[offset] != "{":
        return {}
    offset += 1
    spans: dict[str, tuple[int, int]] = {}
    while True:
        offset = _skip_ws(raw_text, offset)
        if offset >= len(raw_text):
            return {}
        if raw_text[offset] == "}":
            offset = _skip_ws(raw_text, offset + 1)
            return spans if offset == len(raw_text) else {}
        try:
            key, key_end = decoder.raw_decode(raw_text, offset)
        except json.JSONDecodeError:
            return {}
        if not isinstance(key, str):
            return {}
        offset = _skip_ws(raw_text, key_end)
        if offset >= len(raw_text) or raw_text[offset] != ":":
            return {}
        value_start = _skip_ws(raw_text, offset + 1)
        try:
            _, value_end = decoder.raw_decode(raw_text, value_start)
        except json.JSONDecodeError:
            return {}
        spans[key] = (value_start, value_end)
        offset = _skip_ws(raw_text, value_end)
        if offset >= len(raw_text):
            return {}
        if raw_text[offset] == ",":
            offset += 1
            continue
        if raw_text[offset] != "}":
            return {}


def extract_mapping_span(raw_text: str) -> tuple[int, int] | None:
    """Locate the exact UTF-8 byte hull of both semantic mapping values."""

    try:
        parsed = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    if not isinstance(parsed.get("objective_map"), Mapping) or not isinstance(
        parsed.get("variable_map"), list
    ):
        return None
    spans = _top_level_value_spans(raw_text)
    if "objective_map" not in spans or "variable_map" not in spans:
        return None
    char_start = min(spans["objective_map"][0], spans["variable_map"][0])
    char_end = max(spans["objective_map"][1], spans["variable_map"][1])
    return (
        len(raw_text[:char_start].encode("utf-8")),
        len(raw_text[:char_end].encode("utf-8")),
    )


def compute_token_signals(
    current: Mapping[str, Any],
    following: Mapping[str, Any],
    *,
    tolerance: float = TRACE_ABS_TOLERANCE,
) -> JsonDict:
    """Reproduce paper energies for one token and its next model step."""

    missing = [field for field in REQUIRED_TRACE_FIELDS if field not in current]
    missing.extend(field for field in REQUIRED_TRACE_FIELDS if field not in following)
    if missing:
        raise TraceError(f"missing_trace_fields:{sorted(set(missing))}")
    if not _trace_fields_complete(current) or not _trace_fields_complete(following):
        scalar_fields = (
            "selected_token_logit",
            "selected_token_logprob",
            "full_vocabulary_logsumexp",
            "entropy",
            "top_probability",
        )
        if any(not _finite_number(current.get(field)) for field in scalar_fields):
            raise TraceError("finite_trace_scalars_required")
        raise TraceError("vocabulary_trace_fields_invalid")
    if current.get("phase_id") != following.get("phase_id"):
        raise TraceError("adjacent_step_phase_mismatch")
    if int(following["phase_step_index"]) != int(current["phase_step_index"]) + 1:
        raise TraceError("adjacent_step_index_mismatch")
    current_key = current.get("attempt_key")
    following_key = following.get("attempt_key")
    if current_key is not None and following_key is not None and current_key != following_key:
        raise TraceError("adjacent_step_attempt_mismatch")
    reproduced = float(current["selected_token_logit"]) - float(
        current["full_vocabulary_logsumexp"]
    )
    residual = reproduced - float(current["selected_token_logprob"])
    if abs(residual) > tolerance:
        raise TraceError(f"selected_logprob_not_reproduced:{residual}")
    token_energy = -float(current["selected_token_logit"])
    marginalized = -float(following["full_vocabulary_logsumexp"])
    return {
        "token_energy": token_energy,
        "marginalized_energy": marginalized,
        "spilled_energy": token_energy - marginalized,
        "entropy": float(current["entropy"]),
        "top_probability": float(current["top_probability"]),
        "selected_token_logprob_reproduced": reproduced,
        "selected_token_logprob_residual": residual,
    }


def _empty_span_result(reason: str, mapping_span: tuple[int, int] | None) -> JsonDict:
    """Build one terminal abstention without dropping the source candidate."""

    return {
        "mapping_byte_start": mapping_span[0] if mapping_span else None,
        "mapping_byte_end": mapping_span[1] if mapping_span else None,
        "overlapping_token_count": 0,
        "token_results": [],
        "signals": {signal: None for signal in SIGNALS},
        "trace_reproduced": False,
        "eligible": False,
        "abstention_reason": reason,
        "terminal": True,
    }


def pool_token_span(
    *,
    attempt_key: str,
    phase_id: str,
    mapping_span: tuple[int, int],
    trace_rows: Sequence[Mapping[str, Any]],
    token_span_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Mean-pool all mapping tokens or abstain on the complete span."""

    start, end = mapping_span
    if start < 0 or end <= start:
        return _empty_span_result("mapping_span_invalid", mapping_span)
    traces = [
        {**row, "attempt_key": row.get("attempt_key", attempt_key)}
        for row in trace_rows
        if row.get("phase_id") == phase_id
    ]
    spans = [
        {**row, "attempt_key": row.get("attempt_key", attempt_key)}
        for row in token_span_rows
        if row.get("phase_id") == phase_id
    ]
    trace_steps = [row.get("phase_step_index") for row in traces]
    span_steps = [row.get("phase_step_index") for row in spans]
    if len(set(trace_steps)) != len(trace_steps) or len(set(span_steps)) != len(span_steps):
        return _empty_span_result("duplicate_phase_step", mapping_span)
    trace_by_step = {int(row["phase_step_index"]): row for row in traces}
    span_by_step = {int(row["phase_step_index"]): row for row in spans}
    overlapping = sorted(
        step
        for step, row in span_by_step.items()
        if int(row["phase_byte_end"]) > start and int(row["phase_byte_start"]) < end
    )
    if not overlapping:
        return _empty_span_result("mapping_span_has_no_token_overlap", mapping_span)
    token_results: list[JsonDict] = []
    for step in overlapping:
        current = trace_by_step.get(step)
        following = trace_by_step.get(step + 1)
        span = span_by_step[step]
        if current is None:
            return _empty_span_result("missing_current_step", mapping_span)
        if following is None:
            return _empty_span_result("missing_adjacent_step", mapping_span)
        if current.get("emitted_token_id") != span.get("emitted_token_id"):
            return _empty_span_result("trace_span_token_mismatch", mapping_span)
        try:
            values = compute_token_signals(current, following)
        except TraceError as exc:
            reason = str(exc).split(":", maxsplit=1)[0]
            return _empty_span_result(reason, mapping_span)
        token_results.append(
            {
                "phase_step_index": step,
                "attempt_step_index": current.get("attempt_step_index"),
                "emitted_token_id": current.get("emitted_token_id"),
                "phase_byte_start": span.get("phase_byte_start"),
                "phase_byte_end": span.get("phase_byte_end"),
                **values,
                "terminal": True,
            }
        )
    signals = {
        signal: float(np.mean([float(row[signal]) for row in token_results])) for signal in SIGNALS
    }
    return {
        "mapping_byte_start": start,
        "mapping_byte_end": end,
        "overlapping_token_count": len(token_results),
        "token_results": token_results,
        "signals": signals,
        "trace_reproduced": True,
        "eligible": True,
        "abstention_reason": None,
        "terminal": True,
    }


def reduce_candidate_span(attempt: Mapping[str, Any]) -> JsonDict:
    """Reduce one frozen candidate while preserving malformed outcomes."""

    metadata = {
        "attempt_key": str(attempt.get("attempt_key")),
        "ordinal": attempt.get("ordinal"),
        "hf_id": attempt.get("hf_id"),
        "pair_id": attempt.get("pair_id"),
        "split": attempt.get("split"),
        "schedule_id": attempt.get("schedule_id"),
        "formulation_family": attempt.get("formulation_family"),
        "candidate_raw_sha256": attempt.get("candidate_raw_sha256"),
        "candidate_phase_id": attempt.get("candidate_phase_id"),
        "parse_success": bool(
            isinstance(attempt.get("parser_diagnostic"), Mapping)
            and attempt["parser_diagnostic"].get("constraintir_shape_valid") is True
        ),
    }
    if not metadata["parse_success"]:
        return {**metadata, **_empty_span_result("malformed_or_non_constraintir", None)}
    phase_id = attempt.get("candidate_phase_id")
    if not isinstance(phase_id, str) or not phase_id:
        return {**metadata, **_empty_span_result("candidate_phase_missing", None)}
    mapping_span = extract_mapping_span(str(attempt.get("candidate_raw_text", "")))
    if mapping_span is None:
        return {**metadata, **_empty_span_result("mapping_span_missing", None)}
    reduced = pool_token_span(
        attempt_key=str(attempt.get("attempt_key")),
        phase_id=phase_id,
        mapping_span=mapping_span,
        trace_rows=attempt.get("energy_trace_rows", []),
        token_span_rows=attempt.get("token_span_rows", []),
    )
    return {**metadata, **reduced}


def classify_error(row: Mapping[str, Any]) -> str:
    """Assign one deterministic exact-outcome class to each candidate."""

    if row.get("exact_semantic_success") is True:
        return "correct"
    ordered = (
        ("parse_outcome", "rejected", "parse"),
        ("schema_outcome", "rejected", "schema"),
        ("domain_correspondence_outcome", "failed", "domain_correspondence"),
        ("objective_direction_outcome", "failed", "objective_direction"),
        ("objective_order_outcome", "failed", "objective_order"),
        ("satisfiability_outcome", "failed", "satisfiability"),
        ("optimum_outcome", "failed", "optimum"),
        ("solution_space_equivalence_outcome", "failed", "solution_space"),
    )
    for field, failure, name in ordered:
        if row.get(field) == failure:
            return name
    if any(bool(value) for value in row.get("timeout_outcome", {}).values()):
        return "timeout"
    if any(bool(value) for value in row.get("unknown_outcome", {}).values()):
        return "unknown"
    if any(bool(value) for value in row.get("exception_outcome", {}).values()):
        return "exception"
    return "other_exact_failure"


class ExactOutcomeVault:
    """Expose exact outcomes by split and record the required opening order."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = [deepcopy(dict(row)) for row in rows]
        self._opened: set[str] = set()
        self.opening_rows: list[JsonDict] = []

    def open(self, split: str, *, formula_config_hash: str | None = None) -> dict[str, int]:
        """Open one label surface, requiring a frozen hash for held-out data."""

        if split in self._opened:
            raise ValueError(f"labels_already_opened:{split}")
        if split == "heldout" and not _sha256_format(formula_config_hash):
            raise ValueError("formula_config_hash_required_before_heldout")
        selected = [row for row in self._rows if row.get("split") == split]
        if not selected or any(
            type(row.get("exact_semantic_success")) is not bool for row in selected
        ):
            raise ValueError(f"terminal_exact_labels_required:{split}")
        self._opened.add(split)
        self.opening_rows.append(
            {
                "split": split,
                "opening_sequence": len(self.opening_rows) + 1,
                "formula_config_hash": formula_config_hash,
                "label_count": len(selected),
            }
        )
        return {
            str(row["attempt_key"]): int(not bool(row["exact_semantic_success"]))
            for row in selected
        }


def _attach_outcomes(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    certification_index: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Attach opened labels and exact failure classes to one split."""

    attached: list[JsonDict] = []
    for row in rows:
        key = str(row.get("attempt_key"))
        if key not in labels or key not in certification_index:
            raise ValueError(f"opened_outcome_missing:{key}")
        exact = certification_index[key]
        attached.append(
            {
                **dict(row),
                "error_label": int(labels[key]),
                "error_class": classify_error(exact),
            }
        )
    return attached


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float | None:
    """Average recall over observed classes for deterministic calibration."""

    classes = sorted(set(labels))
    if not classes:
        return None
    recalls = []
    for label in classes:
        indices = [index for index, value in enumerate(labels) if value == label]
        recalls.append(sum(predictions[index] == label for index in indices) / len(indices))
    return float(np.mean(recalls))


def select_calibration_policies(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], str]:
    """Select each signal sign and threshold from calibration rows only."""

    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("calibration_rows_only")
    policies: list[JsonDict] = []
    ordered_rows = sorted(rows, key=lambda row: str(row.get("attempt_key")))
    for signal in SIGNALS:
        eligible = [
            row
            for row in ordered_rows
            if row.get("eligible") is True
            and _finite_number(row.get("signals", {}).get(signal))
            and row.get("error_label") in (0, 1)
        ]
        if not eligible:
            policies.append(
                {
                    "signal": signal,
                    "selection_split": "calibration",
                    "direction_sign": 1,
                    "threshold": None,
                    "calibration_balanced_accuracy": None,
                    "eligible_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "degenerate_class": True,
                    "selection_frozen": True,
                }
            )
            continue
        labels = [int(row["error_label"]) for row in eligible]
        best: tuple[float, int, float] | None = None
        selected: tuple[int, float, float] | None = None
        for sign in (1, -1):
            signed_scores = [sign * float(row["signals"][signal]) for row in eligible]
            for threshold in sorted(set(signed_scores)):
                predictions = [int(score >= threshold) for score in signed_scores]
                balanced = _balanced_accuracy(labels, predictions)
                if balanced is None:
                    continue
                rank = (balanced, int(sign == 1), -threshold)
                if best is None or rank > best:
                    best = rank
                    selected = (sign, threshold, balanced)
        if selected is None:
            raise ValueError(f"calibration_policy_selection_failed:{signal}")
        sign, threshold, balanced = selected
        policies.append(
            {
                "signal": signal,
                "selection_split": "calibration",
                "direction_sign": sign,
                "threshold": threshold,
                "calibration_balanced_accuracy": balanced,
                "eligible_count": len(eligible),
                "positive_count": sum(labels),
                "negative_count": len(labels) - sum(labels),
                "degenerate_class": len(set(labels)) < 2,
                "selection_frozen": True,
            }
        )
    return policies, sha256_json(policies)


def _policy_index(policies: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index exactly one frozen policy per required signal."""

    result = {str(row.get("signal")): row for row in policies}
    if set(result) != set(SIGNALS) or len(result) != len(policies):
        raise ValueError("signal_policy_roster_mismatch")
    return result


def _eligible_rows(rows: Sequence[Mapping[str, Any]], signal: str) -> list[Mapping[str, Any]]:
    """Select finite labeled rows without changing their frozen denominator."""

    return [
        row
        for row in rows
        if row.get("eligible") is True
        and row.get("error_label") in (0, 1)
        and _finite_number(row.get("signals", {}).get(signal))
    ]


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute pairwise AUROC with half credit for exact ties."""

    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return credit / (len(positives) * len(negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute tie-stable non-interpolated average precision."""

    positive_count = sum(labels)
    if positive_count == 0 or positive_count == len(labels):
        return None
    groups: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        groups.setdefault(float(score), []).append(int(label))
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        true_positive += sum(group)
        false_positive += len(group) - sum(group)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def paired_ranking_accuracy(
    rows: Sequence[Mapping[str, Any]], signal: str, policy: Mapping[str, Any]
) -> float | None:
    """Compare error and correct scores only inside the same model-pair group."""

    sign = int(policy["direction_sign"])
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in _eligible_rows(rows, signal):
        group = (str(row.get("hf_id")), str(row.get("pair_id")))
        groups.setdefault(group, []).append(row)
    pair_credits: list[float] = []
    for group_rows in groups.values():
        positives = [
            sign * float(row["signals"][signal])
            for row in group_rows
            if row.get("error_label") == 1
        ]
        negatives = [
            sign * float(row["signals"][signal])
            for row in group_rows
            if row.get("error_label") == 0
        ]
        pair_credits.extend(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
    return float(np.mean(pair_credits)) if pair_credits else None


def metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    policy: Mapping[str, Any],
) -> JsonDict:
    """Compute all requested metrics over one fixed denominator."""

    eligible = _eligible_rows(rows, signal)
    labels = [int(row["error_label"]) for row in eligible]
    sign = int(policy["direction_sign"])
    scores = [sign * float(row["signals"][signal]) for row in eligible]
    threshold = policy.get("threshold")
    predictions = (
        [int(score >= float(threshold)) for score in scores] if _finite_number(threshold) else []
    )
    degenerate = len(set(labels)) < 2
    return {
        "signal": signal,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "abstention_count": len(rows) - len(eligible),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
        "direction_sign": sign,
        "threshold": threshold,
        "auroc": None if degenerate else binary_auroc(labels, scores),
        "auprc": None if degenerate else average_precision(labels, scores),
        "calibration_error": (
            float(
                np.mean(
                    [
                        abs(prediction - label)
                        for prediction, label in zip(predictions, labels, strict=True)
                    ]
                )
            )
            if predictions
            else None
        ),
        "paired_ranking_accuracy": (
            None if degenerate else paired_ranking_accuracy(eligible, signal, policy)
        ),
        "degenerate_class": degenerate,
        "terminal": True,
    }


def _metric_value(
    rows: Sequence[Mapping[str, Any]],
    signal: str,
    policy: Mapping[str, Any],
    metric: str,
) -> float | None:
    """Read one recomputed metric for a bootstrap replicate."""

    summary = metric_summary(rows, signal=signal, policy=policy)
    value = summary.get(metric)
    return float(value) if _finite_number(value) else None


def _pair_clusters(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group complete row clusters under their frozen pair ID."""

    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: str(item.get("attempt_key"))):
        clusters.setdefault(str(row.get("pair_id")), []).append(row)
    return clusters


def _percentile_interval(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Compute the frozen two-sided percentile interval."""

    if not values:
        return None, None
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def bootstrap_metric_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    policy: Mapping[str, Any],
    metric: str,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    """Bootstrap one metric by resampling complete pair-ID clusters."""

    ordered_rows = sorted(rows, key=lambda row: str(row.get("attempt_key")))
    point = _metric_value(ordered_rows, signal, policy, metric)
    clusters = _pair_clusters(ordered_rows)
    pair_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    if point is not None and pair_ids:
        for _ in range(resamples):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            replicate = [row for pair_id in sampled for row in clusters[str(pair_id)]]
            value = _metric_value(replicate, signal, policy, metric)
            if value is not None:
                values.append(value)
    lower, upper = _percentile_interval(values)
    return {
        "signal": signal,
        "metric": metric,
        "point_estimate": point,
        "lower": lower,
        "upper": upper,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_unit": "pair_id",
        "pair_count": len(pair_ids),
        "requested_resamples": resamples,
        "eligible_resamples": len(values),
        "random_seed": seed,
        "terminal": True,
    }


def bootstrap_control_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_signal: str,
    policies: Mapping[str, Mapping[str, Any]],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    """Bootstrap the paired spilled-minus-control AUROC difference."""

    matched = [
        row
        for row in rows
        if row.get("eligible") is True
        and row.get("error_label") in (0, 1)
        and _finite_number(row.get("signals", {}).get("spilled_energy"))
        and _finite_number(row.get("signals", {}).get(control_signal))
    ]
    matched = sorted(matched, key=lambda row: str(row.get("attempt_key")))

    def delta(sample: Sequence[Mapping[str, Any]]) -> float | None:
        spilled = _metric_value(sample, "spilled_energy", policies["spilled_energy"], "auroc")
        control = _metric_value(sample, control_signal, policies[control_signal], "auroc")
        return spilled - control if spilled is not None and control is not None else None

    point = delta(matched)
    clusters = _pair_clusters(matched)
    pair_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    if point is not None and pair_ids:
        for _ in range(resamples):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            replicate = [row for pair_id in sampled for row in clusters[str(pair_id)]]
            value = delta(replicate)
            if value is not None:
                values.append(value)
    lower, upper = _percentile_interval(values)
    return {
        "signal": "spilled_energy",
        "control_signal": control_signal,
        "metric": "auroc_delta",
        "delta_auroc": point,
        "delta_lower": lower,
        "delta_upper": upper,
        "paired_row_count": len(matched),
        "pair_count": len(pair_ids),
        "confidence": BOOTSTRAP_CONFIDENCE,
        "bootstrap_unit": "pair_id",
        "requested_resamples": resamples,
        "eligible_resamples": len(values),
        "random_seed": seed,
        "terminal": True,
    }


def _dimension_subsets(
    rows: Sequence[Mapping[str, Any]], dimension: str
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Build deterministic grouped denominators for one reporting dimension."""

    values = sorted({str(row.get(dimension)) for row in rows})
    return [(value, [row for row in rows if str(row.get(dimension)) == value]) for value in values]


def _error_class_subsets(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Compare each error mechanism with the same correct-row reference set."""

    correct = [row for row in rows if row.get("error_label") == 0]
    classes = sorted({str(row.get("error_class")) for row in rows if row.get("error_label") == 1})
    return [
        (
            error_class,
            correct + [row for row in rows if row.get("error_class") == error_class],
        )
        for error_class in classes
    ]


def build_group_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policies: Mapping[str, Mapping[str, Any]],
    dimension: str,
) -> list[JsonDict]:
    """Compute every signal for each value of one fixed dimension."""

    subsets = (
        _error_class_subsets(rows)
        if dimension == "error_class"
        else _dimension_subsets(rows, dimension)
    )
    result = []
    for value, subset in subsets:
        for signal in SIGNALS:
            result.append(
                {
                    "dimension": dimension,
                    "value": value,
                    **metric_summary(subset, signal=signal, policy=policies[signal]),
                }
            )
    return result


def build_metric_surfaces(
    heldout_rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Build overall and all preregistered grouped metric surfaces."""

    overall = [
        {
            "dimension": "overall",
            "value": "all",
            **metric_summary(heldout_rows, signal=signal, policy=policies[signal]),
        }
        for signal in SIGNALS
    ]
    return {
        "heldout_metric_rows": overall,
        "per_model_metric_rows": build_group_metric_rows(
            heldout_rows, policies=policies, dimension="hf_id"
        ),
        "per_schedule_metric_rows": build_group_metric_rows(
            heldout_rows, policies=policies, dimension="schedule_id"
        ),
        "per_formulation_metric_rows": build_group_metric_rows(
            heldout_rows, policies=policies, dimension="formulation_family"
        ),
        "per_error_class_metric_rows": build_group_metric_rows(
            heldout_rows, policies=policies, dimension="error_class"
        ),
    }


def build_bootstrap_rows(
    heldout_rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Build overall intervals for every requested metric and signal."""

    result = []
    for signal_index, signal in enumerate(SIGNALS):
        for metric_index, metric in enumerate(
            ("auroc", "auprc", "calibration_error", "paired_ranking_accuracy")
        ):
            result.append(
                {
                    "dimension": "overall",
                    "value": "all",
                    **bootstrap_metric_interval(
                        heldout_rows,
                        signal=signal,
                        policy=policies[signal],
                        metric=metric,
                        seed=RANDOM_SEED + signal_index * 10 + metric_index,
                    ),
                }
            )
    return result


def build_control_comparisons(
    heldout_rows: Sequence[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare spilled energy with both controls overall and by model."""

    groups = [("overall", "all", list(heldout_rows))]
    groups.extend(
        ("hf_id", value, subset) for value, subset in _dimension_subsets(heldout_rows, "hf_id")
    )
    result = []
    for group_index, (dimension, value, subset) in enumerate(groups):
        for control_index, control in enumerate(CONTROL_SIGNALS):
            result.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "hf_id": value if dimension == "hf_id" else None,
                    **bootstrap_control_delta(
                        subset,
                        control_signal=control,
                        policies=policies,
                        seed=RANDOM_SEED + 100 + group_index * 10 + control_index,
                    ),
                }
            )
    return result


def _coverage_rows(metric_surfaces: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Project coverage and class counts from every saved metric group."""

    result = []
    for surface, rows in metric_surfaces.items():
        for row in rows:
            result.append(
                {
                    "surface": surface,
                    "dimension": row.get("dimension"),
                    "value": row.get("value"),
                    "signal": row.get("signal"),
                    "candidate_count": row.get("candidate_count"),
                    "eligible_count": row.get("eligible_count"),
                    "positive_count": row.get("positive_count"),
                    "negative_count": row.get("negative_count"),
                    "coverage": row.get("coverage"),
                    "terminal": True,
                }
            )
    return result


def _abstention_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one explicit receipt for every span that could not be scored."""

    return [
        {
            "attempt_key": row.get("attempt_key"),
            "split": row.get("split"),
            "hf_id": row.get("hf_id"),
            "pair_id": row.get("pair_id"),
            "schedule_id": row.get("schedule_id"),
            "abstention_reason": row.get("abstention_reason"),
            "terminal": True,
        }
        for row in rows
        if row.get("eligible") is not True
    ]


def evaluation_complete_score(
    source_count: int,
    span_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    formula_config: Mapping[str, Any],
    metric_surfaces: Mapping[str, Sequence[Mapping[str, Any]]],
) -> int:
    """Return one only when every candidate and aggregate is terminal."""

    required_surfaces = {
        "heldout_metric_rows",
        "per_model_metric_rows",
        "per_schedule_metric_rows",
        "per_formulation_metric_rows",
        "per_error_class_metric_rows",
    }
    complete = (
        source_count == 108
        and len(span_rows) == source_count
        and len(trace_rows) == source_count
        and len({str(row.get("attempt_key")) for row in span_rows}) == source_count
        and all(row.get("terminal") is True for row in span_rows)
        and all(row.get("terminal") is True for row in trace_rows)
        and all(
            row.get("trace_reproduced") is True for row in span_rows if row.get("eligible") is True
        )
        and formula_config.get("configuration_frozen_before_heldout") is True
        and _sha256_format(formula_config.get("formula_config_hash"))
        and required_surfaces == set(metric_surfaces)
        and all(metric_surfaces[name] for name in required_surfaces)
        and all(
            row.get("terminal") is True
            for name in required_surfaces
            for row in metric_surfaces[name]
        )
    )
    return int(complete)


def decide_requalification(
    *,
    evaluation_complete_score: int,
    overall_spilled: Mapping[str, Any],
    spilled_interval: Mapping[str, Any],
    per_model_control_comparisons: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Apply the final registered superiority gate and retirement rule."""

    auroc = overall_spilled.get("auroc")
    lower = spilled_interval.get("lower")
    families_passing = []
    for model in MODEL_FAMILIES:
        rows = [
            row
            for row in per_model_control_comparisons
            if row.get("hf_id") == model and row.get("control_signal") in CONTROL_SIGNALS
        ]
        if len(rows) == len(CONTROL_SIGNALS) and all(
            _finite_number(row.get("delta_lower")) and float(row["delta_lower"]) > 0.0
            for row in rows
        ):
            families_passing.append(model)
    passed = (
        type(evaluation_complete_score) is int
        and evaluation_complete_score == 1
        and _finite_number(auroc)
        and float(auroc) >= 0.65
        and _finite_number(lower)
        and float(lower) > 0.50
        and len(families_passing) >= 2
    )
    if passed:
        return {
            "spilled_energy_requalified_score": 1,
            "verdict_class": "positive",
            "honest_verdict": "complete_spilled_energy_requalified",
            "families_beating_both_controls": families_passing,
            "retirement_recommendation": {
                "retire": False,
                "retire_if_same_verdict": True,
                "propose_retry": False,
                "reason": "The sole preregistered requalification gate passed.",
            },
        }
    return {
        "spilled_energy_requalified_score": 0,
        "verdict_class": "null" if evaluation_complete_score == 1 else "partial",
        "honest_verdict": (
            "complete_null_spilled_energy_requalification_retired"
            if evaluation_complete_score == 1
            else "partial_spilled_energy_requalification"
        ),
        "families_beating_both_controls": families_passing,
        "retirement_recommendation": {
            "retire": evaluation_complete_score == 1,
            "retire_if_same_verdict": True,
            "propose_retry": False,
            "reason": (
                "The final preregistered attempt repeated the prior null; no pooling, prompt, corpus, or verifier retry is proposed."
                if evaluation_complete_score == 1
                else "Evaluation did not reach a scientific verdict."
            ),
        },
    }


def _trace_reproduction_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project one terminal trace decision for each source span."""

    return [
        {
            "attempt_key": row.get("attempt_key"),
            "split": row.get("split"),
            "mapping_byte_start": row.get("mapping_byte_start"),
            "mapping_byte_end": row.get("mapping_byte_end"),
            "overlapping_token_count": row.get("overlapping_token_count"),
            "trace_reproduced": row.get("trace_reproduced"),
            "eligible": row.get("eligible"),
            "abstention_reason": row.get("abstention_reason"),
            "max_abs_logprob_residual": (
                max(
                    (
                        abs(float(token["selected_token_logprob_residual"]))
                        for token in row.get("token_results", [])
                    ),
                    default=None,
                )
            ),
            "terminal": True,
        }
        for row in rows
    ]


def prior_verdict_comparison(prior: Mapping[str, Any], current_auroc: Any) -> JsonDict:
    """Record the prior null and the changed exact-span evidence surface."""

    prior_auroc = prior.get("auroc_spilled")
    return {
        "prior_experiment": 2497,
        "prior_honest_verdict": prior.get("honest_verdict"),
        "prior_spilled_auroc": prior_auroc,
        "prior_pearson_spilled": prior.get("pearson_spilled"),
        "current_spilled_auroc": current_auroc,
        "auroc_delta": (
            float(current_auroc) - float(prior_auroc)
            if _finite_number(current_auroc) and _finite_number(prior_auroc)
            else None
        ),
        "changed_evidence_surface": "three_current_model_families_with_exact_constraintir_mapping_spans",
        "retire_if_same_verdict": True,
        "terminal": True,
    }


def formula_config_digest(config: Mapping[str, Any]) -> str:
    """Hash the policy-bearing configuration without later opening receipts."""

    payload = {
        key: value
        for key, value in config.items()
        if key not in {"formula_config_hash", "label_opening_rows"}
    }
    return sha256_json(payload)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the scientific payload while excluding runtime wall time and self hash."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(payload)


def _base_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> JsonDict:
    """Build the complete field roster used by blocked and evaluated runs."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "formula_config": deepcopy(BASE_FORMULA_CONFIG),
        "calibration_rows": [],
        "rows": [],
        "per_span_results": [],
        "trace_reproduction_rows": [],
        "heldout_metric_rows": [],
        "per_model_metric_rows": [],
        "per_schedule_metric_rows": [],
        "per_formulation_metric_rows": [],
        "per_error_class_metric_rows": [],
        "control_comparison_rows": [],
        "bootstrap_interval_rows": [],
        "coverage_rows": [],
        "abstention_rows": [],
        "prior_verdict_comparison": prior_verdict_comparison(prior, None),
        "spilled_energy_evaluation_complete_score": 0,
        "spilled_energy_requalified_score": 0,
        "retirement_recommendation": {
            "retire": False,
            "retire_if_same_verdict": True,
            "propose_retry": False,
            "reason": "No scientific verdict exists because preflight failed.",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_spilled_energy_requalification",
    }
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-complete blocked record for any failed precondition."""

    artifact = _base_artifact(
        date=date,
        duration_s=duration_s,
        checks=checks,
        hashes=hashes,
        prior=prior,
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_evaluated_artifact(
    *,
    date: str,
    duration_s: float,
    inputs: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
) -> JsonDict:
    """Reduce calibration first, then open and evaluate held-out outcomes once."""

    bank = inputs["bank"]
    certification = inputs["certification"]
    attempts = list(bank["per_attempt_rows"])
    certification_rows = list(certification["per_candidate_rows"])
    certification_by_key = _certification_index(certification)
    vault = ExactOutcomeVault(certification_rows)
    unlabeled = [reduce_candidate_span(attempt) for attempt in attempts]

    calibration_labels = vault.open("calibration")
    calibration_unlabeled = [row for row in unlabeled if row.get("split") == "calibration"]
    calibration = _attach_outcomes(
        calibration_unlabeled,
        calibration_labels,
        certification_by_key,
    )
    calibration_rows, policy_hash = select_calibration_policies(calibration)
    policies = _policy_index(calibration_rows)
    formula_config = {
        **deepcopy(BASE_FORMULA_CONFIG),
        "signal_policies": deepcopy(calibration_rows),
        "calibration_policy_hash": policy_hash,
    }
    formula_config_hash = formula_config_digest(formula_config)
    formula_config["formula_config_hash"] = formula_config_hash

    heldout_labels = vault.open("heldout", formula_config_hash=formula_config_hash)
    heldout_unlabeled = [row for row in unlabeled if row.get("split") == "heldout"]
    heldout = _attach_outcomes(heldout_unlabeled, heldout_labels, certification_by_key)
    formula_config["label_opening_rows"] = deepcopy(vault.opening_rows)
    calibration_rows = [
        {**row, "formula_config_hash": formula_config_hash} for row in calibration_rows
    ]
    all_rows_by_key = {str(row["attempt_key"]): row for row in [*calibration, *heldout]}
    all_rows = [all_rows_by_key[str(row["attempt_key"])] for row in unlabeled]

    metric_surfaces = build_metric_surfaces(heldout, policies)
    bootstrap_rows = build_bootstrap_rows(heldout, policies)
    control_rows = build_control_comparisons(heldout, policies)
    trace_rows = _trace_reproduction_rows(all_rows)
    complete_score = evaluation_complete_score(
        len(attempts),
        all_rows,
        trace_rows,
        formula_config,
        metric_surfaces,
    )
    overall_spilled = next(
        row for row in metric_surfaces["heldout_metric_rows"] if row["signal"] == "spilled_energy"
    )
    spilled_interval = next(
        row
        for row in bootstrap_rows
        if row["signal"] == "spilled_energy" and row["metric"] == "auroc"
    )
    model_control_rows = [row for row in control_rows if row.get("dimension") == "hf_id"]
    decision = decide_requalification(
        evaluation_complete_score=complete_score,
        overall_spilled=overall_spilled,
        spilled_interval=spilled_interval,
        per_model_control_comparisons=model_control_rows,
    )
    artifact = _base_artifact(
        date=date,
        duration_s=duration_s,
        checks=checks,
        hashes=hashes,
        prior=inputs["prior"],
    )
    artifact.update(
        {
            "formula_config": formula_config,
            "calibration_rows": calibration_rows,
            "rows": all_rows,
            "per_span_results": deepcopy(all_rows),
            "trace_reproduction_rows": trace_rows,
            **metric_surfaces,
            "control_comparison_rows": control_rows,
            "bootstrap_interval_rows": bootstrap_rows,
            "coverage_rows": _coverage_rows(metric_surfaces),
            "abstention_rows": _abstention_rows(all_rows),
            "prior_verdict_comparison": prior_verdict_comparison(
                inputs["prior"], overall_spilled.get("auroc")
            ),
            "spilled_energy_evaluation_complete_score": complete_score,
            "spilled_energy_requalified_score": decision["spilled_energy_requalified_score"],
            "retirement_recommendation": decision["retirement_recommendation"],
            "gate_check_summary": [],
            "verdict_class": decision["verdict_class"],
            "honest_verdict": decision["honest_verdict"],
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Recompute every saved surface and reject aggregate or verdict drift."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"required_fields_missing:{sorted(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not str(principles.get(field, "")).strip() for field in REQUIRED_ARTIFACT_FIELDS
    ):
        raise ValueError("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_oracle_declaration_mismatch")
    for field in (
        "spilled_energy_evaluation_complete_score",
        "spilled_energy_requalified_score",
    ):
        if type(artifact.get(field)) is not int or artifact[field] not in (0, 1):
            raise ValueError(f"not_bare_int:{field}")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("verdict_class_invalid")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("honest_verdict") != "blocked_spilled_energy_requalification":
            raise ValueError("blocked_verdict_mismatch")
        if not artifact.get("gate_check_summary"):
            raise ValueError("blocked_gate_summary_missing")
    else:
        rows = artifact.get("rows", [])
        if rows != artifact.get("per_span_results"):
            raise ValueError("rows_projection_mismatch")
        if len(rows) != 108:
            raise ValueError("span_row_count_mismatch")
        trace_rows = _trace_reproduction_rows(rows)
        if trace_rows != artifact.get("trace_reproduction_rows"):
            raise ValueError("trace_reproduction_rows_mismatch")
        config = artifact.get("formula_config", {})
        if config.get("formula_config_hash") != formula_config_digest(config):
            raise ValueError("formula_config_hash_mismatch")
        policies = _policy_index(artifact.get("calibration_rows", []))
        heldout = [row for row in rows if row.get("split") == "heldout"]
        surfaces = build_metric_surfaces(heldout, policies)
        for name, expected in surfaces.items():
            if artifact.get(name) != expected:
                raise ValueError(f"{name}_mismatch")
        bootstrap_rows = build_bootstrap_rows(heldout, policies)
        if artifact.get("bootstrap_interval_rows") != bootstrap_rows:
            raise ValueError("bootstrap_interval_rows_mismatch")
        control_rows = build_control_comparisons(heldout, policies)
        if artifact.get("control_comparison_rows") != control_rows:
            raise ValueError("control_comparison_rows_mismatch")
        if artifact.get("coverage_rows") != _coverage_rows(surfaces):
            raise ValueError("coverage_rows_mismatch")
        if artifact.get("abstention_rows") != _abstention_rows(rows):
            raise ValueError("abstention_rows_mismatch")
        complete = evaluation_complete_score(len(rows), rows, trace_rows, config, surfaces)
        if artifact.get("spilled_energy_evaluation_complete_score") != complete:
            raise ValueError("evaluation_complete_score_mismatch")
        overall = next(
            row for row in surfaces["heldout_metric_rows"] if row["signal"] == "spilled_energy"
        )
        interval = next(
            row
            for row in bootstrap_rows
            if row["signal"] == "spilled_energy" and row["metric"] == "auroc"
        )
        decision = decide_requalification(
            evaluation_complete_score=complete,
            overall_spilled=overall,
            spilled_interval=interval,
            per_model_control_comparisons=[
                row for row in control_rows if row.get("dimension") == "hf_id"
            ],
        )
        for field in (
            "spilled_energy_requalified_score",
            "retirement_recommendation",
            "verdict_class",
            "honest_verdict",
        ):
            if artifact.get(field) != decision[field]:
                raise ValueError(f"decision_mismatch:{field}")
        if artifact.get("gate_check_summary") != []:
            raise ValueError("completed_gate_summary_not_empty")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum_mismatch")


def _atomic_write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace one result only after complete JSON bytes are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(artifact, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(
    *,
    date: str,
    repo_root: Path,
    output_path: Path | None = None,
) -> JsonDict:
    """Run preflight, reduce frozen evidence once, validate, and write atomically."""

    started = time.monotonic()
    inputs = load_inputs(repo_root)
    checks = check_preconditions(inputs)
    hashes = source_artifact_hashes(repo_root, inputs)
    duration = time.monotonic() - started
    if gate_summary(checks):
        artifact = build_blocked_artifact(
            date=date,
            duration_s=duration,
            checks=checks,
            hashes=hashes,
            prior=inputs.get("prior", {}),
        )
    else:
        artifact = build_evaluated_artifact(
            date=date,
            duration_s=duration,
            inputs=inputs,
            checks=checks,
            hashes=hashes,
        )
        artifact["duration_s"] = time.monotonic() - started
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    destination = output_path if output_path is not None else repo_root / RESULT_PATH
    _atomic_write(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the required command-line surface with no model access."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    artifact = run(date=args.date, repo_root=repo_root, output_path=args.output)
    print(
        json.dumps(
            {
                "artifact": str(args.output or repo_root / RESULT_PATH),
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
                "spilled_energy_evaluation_complete_score": artifact[
                    "spilled_energy_evaluation_complete_score"
                ],
                "spilled_energy_requalified_score": artifact["spilled_energy_requalified_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the required command surface.
    raise SystemExit(main())
