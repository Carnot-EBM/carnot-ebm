"""Measure one bounded label-free evidence extraction pilot.

The experiment uses the eight records that Experiment 7602 froze. It measures
transport and cost only. It does not use human labels and does not claim that a
pointer is semantically correct.

Spec: REQ-VERIFY-7604 and SCENARIO-VERIFY-7604-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any
import urllib.request

from carnot import experiment_7588_v663_evidence_protocol as evidence_protocol
from carnot import experiment_7602_v664_evidence_requalification as source_protocol
from carnot.inference.sota_models import cached_current_model, gguf_tokenizer_loadable
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7604-v664-evidence-pilot"
SCHEMA = "carnot.exp7604.v664.evidence_pilot.v1"
RESULT_PATH = Path("results/experiment_7604_v664_evidence_pilot.json")
RAW_DIR = Path("results/raw/experiment_7604_v664_evidence_pilot")
SOURCE_RESULT = Path("results/experiment_7602_v664_evidence_requalification.json")
SOURCE_RAW = Path("results/raw/experiment_7602_v664_evidence_requalification")
PILOT_INPUT = SOURCE_RAW / "pilot_model_inputs.jsonl"
PROTOCOL_PATH = SOURCE_RAW / "protocol.json"
MODULE_PATH = Path("python/carnot/experiment_7604_v664_evidence_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7604_v664_evidence_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7604_v664_evidence_pilot.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
RANDOM_SEED = 7_604_001
MAX_TOKENS = 512
MAX_LINKS = 6
CAPTURE_GROUPS = 120
CAPTURE_LIMIT_S = 3000.0
VALIDATION_RESERVE_S = 300.0
GPU_REQUIRED_FREE_MB = 20_000

SYSTEM_PROMPT = """You extract evidence pointers from complete text.
Return one JSON array and no prose. Each item must have exactly these fields:
response_sentence_id, source_sentence_ids, relation, entity_type,
abstention_reason. Use only supplied IDs. relation is supports, contradicts, or
unknown. Unknown items use no source IDs and give a reason. Linked items use at
least one source ID and an empty reason. Return at most six items. Do not emit
replacement text or reasoning. /no_think"""


def text_sha256(value: str) -> str:
    """Hash exact UTF-8 text so input loss cannot look like model behavior."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes with the common artifact prefix."""

    return current_work_receipt.sha256_file(path)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for requests, reductions, and configuration."""

    return current_work_receipt.canonical_hash(value)


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase boundary so long local work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7604] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}{(' ' + suffix) if suffix else ''}",
        flush=True,
    )


def validate_input_record(record: Mapping[str, Any]) -> bool:
    """Prove that one predictor record is complete and cannot expose outcomes."""

    forbidden = {"label", "human_label", "probability", "raw_probability_offset"}
    if forbidden & set(record):
        raise ValueError("predictor_label_access")
    if record.get("labels_accessible") is not False:
        raise ValueError("predictor_label_access")
    if record.get("raw_probability_accessible") is not False:
        raise ValueError("predictor_probability_access")
    if record.get("maximum_proposed_links") != MAX_LINKS:
        raise ValueError("evidence_link_budget_changed")
    expected_relations = {"supports", "contradicts", "unknown"}
    if set(record.get("allowed_relations") or []) != expected_relations:
        raise ValueError("evidence_relations_changed")
    for name, prefix in (("source", "S"), ("question", "S"), ("answer", "R")):
        text = record.get(f"complete_{name}")
        if not isinstance(text, str) or not text:
            raise ValueError(f"complete_{name}_absent")
        if record.get(f"{name}_sha256") != text_sha256(text):
            raise ValueError(f"{name}_hash_invalid")
        evidence_protocol.roundtrip_segments(text, record.get(f"{name}_sentences") or [])
        ids = [str(row.get("sentence_id")) for row in record[f"{name}_sentences"]]
        if len(ids) != len(set(ids)) or any(not value.startswith(prefix) for value in ids):
            raise ValueError(f"{name}_sentence_ids_invalid")
    if not str(record.get("component_hash") or ""):
        raise ValueError("component_hash_absent")
    return True


def _pointer_rows(record: Mapping[str, Any], field: str) -> list[JsonDict]:
    """Keep only the supplied ID and exact text in the model request."""

    return [{"id": str(row["sentence_id"]), "text": str(row["text"])} for row in record[field]]


def build_extraction_request(record: Mapping[str, Any]) -> JsonDict:
    """Build the frozen non-thinking request without outcome-bearing fields."""

    validate_input_record(record)
    content = {
        "complete_source": record["complete_source"],
        "complete_question": record["complete_question"],
        "complete_answer": record["complete_answer"],
        "source_sentences": _pointer_rows(record, "source_sentences"),
        "answer_sentences": _pointer_rows(record, "answer_sentences"),
        "maximum_links": MAX_LINKS,
        "relations": ["supports", "contradicts", "unknown"],
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(content, ensure_ascii=False, sort_keys=True)},
        ],
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "max_tokens": MAX_TOKENS,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_extraction_response(
    record: Mapping[str, Any], response_text: str, finish_reason: str
) -> JsonDict:
    """Reduce one completion once and retain every failure in the denominator."""

    validate_input_record(record)
    if not response_text.strip():
        return {
            "parser_outcome": "empty_output",
            "usable_schema": False,
            "invalid_pointer_accepted": False,
            "evidence": [],
            "parser_error": "empty_output",
            "censoring": "empty_output",
        }
    if finish_reason == "length":
        return {
            "parser_outcome": "truncated_output",
            "usable_schema": False,
            "invalid_pointer_accepted": False,
            "evidence": [],
            "parser_error": "finish_reason_length",
            "censoring": "output_truncated",
        }
    try:
        proposals = json.loads(response_text)
        if not isinstance(proposals, list) or any(
            not isinstance(row, Mapping) for row in proposals
        ):
            raise ValueError("evidence_output_not_array")
        contract = {
            "source_sentences": record["source_sentences"],
            "response_sentences": record["answer_sentences"],
        }
        normalized = evidence_protocol.normalize_evidence_output(contract, proposals)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "parser_outcome": "invalid_output",
            "usable_schema": False,
            "invalid_pointer_accepted": False,
            "evidence": [],
            "parser_error": f"{type(exc).__name__}:{exc}",
            "censoring": "invalid_output",
        }
    return {
        "parser_outcome": "usable_schema",
        "usable_schema": True,
        "invalid_pointer_accepted": False,
        "evidence": normalized,
        "parser_error": None,
        "censoring": "none",
    }


def reduce_pilot_rows(
    rows: Sequence[Mapping[str, Any]], *, transport_authenticated: bool
) -> JsonDict:
    """Recompute transport readiness from eight independent call outcomes."""

    if len(rows) != 8:
        raise ValueError("exactly_eight_pilot_rows_required")
    components = [str(row.get("component_hash") or "") for row in rows]
    if not all(components) or len(set(components)) != 8:
        raise ValueError("pilot_components_not_disjoint")
    usable = sum(row.get("usable_schema") is True for row in rows)
    invalid_accepted = sum(row.get("invalid_pointer_accepted") is True for row in rows)
    lossless = all(row.get("lossless_input") is True for row in rows)
    ready = bool(transport_authenticated and usable >= 6 and invalid_accepted == 0 and lossless)
    return {
        "pilot_count": 8,
        "usable_schema_count": usable,
        "invalid_pointer_accepted_count": invalid_accepted,
        "lossless_input_count": sum(row.get("lossless_input") is True for row in rows),
        "transport_authenticated": bool(transport_authenticated),
        "evidence_transport_ready_score": int(ready),
        "parser_outcome_counts": {
            name: sum(row.get("parser_outcome") == name for row in rows)
            for name in (
                "usable_schema",
                "invalid_output",
                "empty_output",
                "truncated_output",
                "transport_failure",
            )
        },
    }


def estimate_capture(
    pilot_rows: Sequence[Mapping[str, Any]],
    target_prompt_tokens: Sequence[int],
    *,
    model_load_s: float,
    validation_reserve_s: float = VALIDATION_RESERVE_S,
) -> JsonDict:
    """Project a capture from upper measured rates and the fixed output budget."""

    if len(pilot_rows) != 8 or len(target_prompt_tokens) != CAPTURE_GROUPS:
        raise ValueError("capture_estimator_sample_size_invalid")
    if any(int(value) <= 0 for value in target_prompt_tokens):
        raise ValueError("target_prompt_length_invalid")
    prefill_rates = [
        float(row["prefill_s"]) / int(row["prompt_tokens"])
        for row in pilot_rows
        if int(row.get("prompt_tokens") or 0) > 0 and float(row.get("prefill_s") or 0) >= 0
    ]
    decode_rates = [
        float(row["decode_s"]) / max(1, int(row.get("output_tokens") or 0))
        for row in pilot_rows
        if float(row.get("decode_s") or 0) >= 0
    ]
    fixed_costs = [
        max(
            0.0,
            float(row.get("call_s") or 0)
            - float(row.get("prefill_s") or 0)
            - float(row.get("decode_s") or 0),
        )
        for row in pilot_rows
    ]
    if not prefill_rates or not decode_rates:
        raise ValueError("pilot_cost_components_absent")
    prefill_upper = max(prefill_rates)
    decode_upper = max(decode_rates)
    fixed_upper = max(fixed_costs)
    uncertainty = 1.25
    inference_s = sum(
        fixed_upper + prefill_upper * int(tokens) + decode_upper * MAX_TOKENS
        for tokens in target_prompt_tokens
    )
    projected = float(model_load_s) + float(validation_reserve_s) + uncertainty * inference_s
    return {
        "group_count": CAPTURE_GROUPS,
        "actual_target_prompt_tokens": [int(value) for value in target_prompt_tokens],
        "actual_target_prompt_token_total": sum(int(value) for value in target_prompt_tokens),
        "output_tokens_per_group": MAX_TOKENS,
        "prefill_upper_s_per_token": prefill_upper,
        "decode_upper_s_per_token": decode_upper,
        "fixed_upper_s_per_call": fixed_upper,
        "uncertainty_multiplier": uncertainty,
        "model_load_s": float(model_load_s),
        "validation_reserve_s": float(validation_reserve_s),
        "projected_s": projected,
        "threshold_s": CAPTURE_LIMIT_S,
        "feasible_score": int(projected <= CAPTURE_LIMIT_S),
        "method": "max_observed_component_rates_times_actual_prompt_lengths_plus_25pct",
    }


def prompt_token_projections(
    records: Sequence[Mapping[str, Any]], pilot_rows: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Scale exact target request bytes by the worst measured prompt-token ratio."""

    ratios = [
        int(row["prompt_tokens"]) / int(row["request_bytes"])
        for row in pilot_rows
        if int(row.get("prompt_tokens") or 0) > 0 and int(row.get("request_bytes") or 0) > 0
    ]
    if not ratios:
        raise ValueError("prompt_token_ratio_absent")
    upper = max(ratios)
    return [
        max(
            1,
            math.ceil(
                len(
                    json.dumps(
                        build_extraction_request(record),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
                * upper
            ),
        )
        for record in records
    ]


def gate_row(
    check: str,
    *,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    category: str = "validity",
    principle: str = "A failed operand stays explicit and blocks dependent claims.",
) -> JsonDict:
    """Keep every operand needed to diagnose one acceptance check."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": str(Path(path).resolve()),
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failures and retain the first complete diagnostic row."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def field_principles() -> dict[str, str]:
    """Place the requested interpretation rule beside each governed field."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; completion alone is not scientific benefit.",
        "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence never opens readiness.",
        "gate_check_summary": "Every block names check, upstream, path, field, operator, expected, and observed.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness remain separate.",
        "rows": "Each independent unit and arm keeps absolute metrics and raw counting operands.",
        "sample_size_budget": "Intended, observed, excluded, and censored units never multiply groups.",
        "inference_substrate": "Describe current execution; historical GPU evidence is not a current call.",
        "inference_substrate_class": "Record the actual closed-enum class here and the planned class separately; blocked_no_run means no model work.",
        "MODEL_SPECS": "Every current model call names unsloth/Qwen3.8-27B-GGUF.",
        "invocation_counts": "Count loads, forwards, generations, and tokens separately from history.",
        "duration_s": "Use current monotonic time without inherited work or padding.",
        "random_seed": "Persist every stochastic stage seed.",
        "reproducibility_checksum": "Bind configuration, immutable raw evidence, and reduction.",
        "source_artifact_hashes": "Separate producer bytes, conductor records, and missing producers.",
        "validation_receipts": "Bind commands, exits, worktrees, logs, and terminal readers.",
        "verifier_is_oracle": "Exact fixtures cannot prove learned semantic correctness or gain.",
        "field_principles": "Carry each one-line interpretation rule beside its governed field.",
        "evidence_transport_ready_score": "One requires owned offload and at least six valid rows of eight.",
        "fit_capture_feasible_score": "One requires a frozen 120-group fit projection within 3000 seconds.",
        "eval_capture_feasible_score": "One requires a frozen 120-group evaluation projection within 3000 seconds.",
        "capture_configuration": "Pin prompt, parser, model hash, output budget, seeds, and tokenizer.",
        "pilot_rows": "Keep all eight call outcomes, including invalid, unknown, and censored output.",
    }


def capture_configuration(model_sha256: str | None = None) -> JsonDict:
    """Return the immutable request and parser choices used by every row."""

    return {
        "model_id": MODEL_ID,
        "model_sha256": model_sha256,
        "quantization": "Q4_K_M",
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": text_sha256(SYSTEM_PROMPT),
        "parser": "carnot.experiment_7588_v663_evidence_protocol.normalize_evidence_output",
        "parser_relations": ["supports", "contradicts", "unknown"],
        "maximum_links": MAX_LINKS,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": RANDOM_SEED,
        "thinking": False,
        "tokenizer": "embedded_gguf",
        "chat_template": "embedded_gguf",
        "retry_count": 0,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _acceptance_gates(*, valid: bool, ready: bool, retained: bool) -> list[JsonDict]:
    return [
        {
            "gate": "validity",
            "principle": "Only authenticated current bytes and execution can support reduction.",
            "passed": bool(valid),
        },
        {
            "gate": "readiness",
            "principle": "Transport readiness requires six usable rows and zero accepted bad pointers.",
            "passed": bool(ready),
        },
        {
            "gate": "benefit",
            "principle": "This feasibility pilot does not measure oracle-distinct semantic benefit.",
            "passed": False,
        },
        {
            "gate": "retention",
            "principle": "Every attempted unit remains in the fixed denominator without retries.",
            "passed": bool(retained),
        },
        {
            "gate": "freshness",
            "principle": "Historically exposed fixtures cannot support a fresh confirmatory claim.",
            "passed": False,
        },
    ]


def _unstarted_rows(reason: str) -> list[JsonDict]:
    return [
        {
            "pilot_index": index,
            "component_hash": f"unstarted-{index}",
            "arm": "evidence_link",
            "status": "unstarted",
            "usable_schema": False,
            "invalid_pointer_accepted": False,
            "lossless_input": False,
            "parser_outcome": "blocked_no_run",
            "censoring": reason,
            "numerator": 0,
            "denominator": 1,
            "seed": RANDOM_SEED,
            "direction": "higher_is_more_transport_usable",
            "provenance": {"source": "exp7602", "call_id": None},
        }
        for index in range(8)
    ]


def _zero_invocations() -> JsonDict:
    return {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "model_loads_failed": 0,
        "forward_calls_attempted": 0,
        "forward_calls_completed": 0,
        "forward_calls_failed": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "generation_calls_failed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "warmup_calls": 0,
    }


def build_blocked_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    duration_s: float,
    reason: str,
    current_receipt: Mapping[str, Any] | None = None,
    invocation_counts: Mapping[str, Any] | None = None,
    actual_substrate_class: str = "blocked_no_run",
) -> JsonDict:
    """Build complete no-run evidence for an external or resource block."""

    rows = _unstarted_rows(reason)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": gate_summary(checks),
        "acceptance_gate_results": _acceptance_gates(valid=False, ready=False, retained=True),
        "rows": rows,
        "pilot_rows": rows,
        "sample_size_budget": {"intended": 8, "observed": 0, "excluded": 0, "censored": 8},
        "inference_substrate": actual_substrate_class,
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": actual_substrate_class,
        "inference_substrate_plausibility_floor_s": 10.0,
        "MODEL_SPECS": [] if actual_substrate_class == "blocked_no_run" else MODEL_SPECS,
        "planned_MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invoked": actual_substrate_class != "blocked_no_run",
        "invocation_counts": deepcopy(dict(invocation_counts or _zero_invocations())),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": [RANDOM_SEED],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": True,
        "field_principles": field_principles(),
        "evidence_transport_ready_score": 0,
        "fit_capture_feasible_score": 0,
        "eval_capture_feasible_score": 0,
        "capture_configuration": capture_configuration(),
        "pilot_reduction": {
            "pilot_count": 8,
            "usable_schema_count": 0,
            "invalid_pointer_accepted_count": 0,
            "lossless_input_count": 0,
            "transport_authenticated": False,
            "evidence_transport_ready_score": 0,
        },
        "fit_capture_projection": {"status": "blocked_no_run", "reason": reason},
        "eval_capture_projection": {"status": "blocked_no_run", "reason": reason},
        "current_work_receipt": deepcopy(dict(current_receipt or {})),
        "fresh_confirmatory_claim_allowed": False,
        "empirical_benefit_measured": False,
        "scope_retirement": {
            "retired": False,
            "reason": "resource_or_external_block_does_not_retire_scientific_hypothesis",
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "active_research_roadmap_modified": False,
        "applicable_numbered_e2e": [],
        "capability_e2e": {
            "applicable": False,
            "reason": "Read-only extraction reporting has no numbered model E2E.",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _invocations(pilot_rows: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]) -> JsonDict:
    attempts = sum(row.get("status") != "unstarted" for row in pilot_rows)
    completed = sum(row.get("transport_completed") is True for row in pilot_rows)
    failed = attempts - completed
    return {
        "model_loads_attempted": 1,
        "model_loads_completed": int(runtime.get("model_load_completed") is True),
        "model_loads_failed": int(runtime.get("model_load_completed") is not True),
        "forward_calls_attempted": attempts,
        "forward_calls_completed": completed,
        "forward_calls_failed": failed,
        "generation_calls_attempted": attempts,
        "generation_calls_completed": completed,
        "generation_calls_failed": failed,
        "input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in pilot_rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in pilot_rows),
        "warmup_calls": int(runtime.get("warmup_calls") or 0),
    }


def build_complete_artifact(
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    source_hashes: Sequence[Mapping[str, Any]],
    fit_projection: Mapping[str, Any],
    eval_projection: Mapping[str, Any],
    duration_s: float,
    validation_receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal feasibility result without a semantic benefit claim."""

    reduction = reduce_pilot_rows(
        pilot_rows,
        transport_authenticated=runtime_receipt.get("transport_authenticated") is True,
    )
    invocations = _invocations(pilot_rows, runtime_receipt)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": "complete_null_transport_feasibility_measured",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": ["empirical_benefit"],
            "first_failure": {
                "check": "empirical_benefit",
                "category": "benefit",
                "upstream": EXPERIMENT_ID,
                "path": str((REPO_ROOT / RESULT_PATH).resolve()),
                "field": "empirical_benefit_measured",
                "operator": "eq",
                "expected": True,
                "observed": False,
                "passed": False,
                "principle": "Transport feasibility does not establish semantic benefit.",
            },
        },
        "acceptance_gate_results": _acceptance_gates(
            valid=runtime_receipt.get("transport_authenticated") is True,
            ready=reduction["evidence_transport_ready_score"] == 1,
            retained=len(pilot_rows) == 8,
        ),
        "rows": [deepcopy(dict(row)) for row in pilot_rows],
        "pilot_rows": [deepcopy(dict(row)) for row in pilot_rows],
        "sample_size_budget": {
            "intended": 8,
            "observed": 8,
            "excluded": 0,
            "censored": sum(row.get("censoring") != "none" for row in pilot_rows),
        },
        "inference_substrate": "model_bounded_generation",
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": "model_bounded_generation",
        "inference_substrate_plausibility_floor_s": 10.0,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [deepcopy(dict(runtime_receipt.get("model_spec") or {}))],
        "model_invoked": True,
        "invocation_counts": invocations,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": [RANDOM_SEED],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": True,
        "field_principles": field_principles(),
        "evidence_transport_ready_score": reduction["evidence_transport_ready_score"],
        "fit_capture_feasible_score": int(fit_projection.get("feasible_score") or 0),
        "eval_capture_feasible_score": int(eval_projection.get("feasible_score") or 0),
        "capture_configuration": capture_configuration(runtime_receipt.get("model_sha256")),
        "pilot_reduction": reduction,
        "fit_capture_projection": deepcopy(dict(fit_projection)),
        "eval_capture_projection": deepcopy(dict(eval_projection)),
        "inference_runtime_receipt": deepcopy(dict(runtime_receipt)),
        "current_work_receipt": deepcopy(dict(current_receipt or {})),
        "fresh_confirmatory_claim_allowed": False,
        "empirical_benefit_measured": False,
        "scope_retirement": {
            "retired": False,
            "reason": "first_completed_transport_feasibility_measurement",
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "active_research_roadmap_modified": False,
        "applicable_numbered_e2e": [],
        "capability_e2e": {
            "applicable": False,
            "reason": "Read-only extraction reporting has no numbered model E2E.",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, counters, reductions, principles, and provenance."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema_invalid")
    verdict = value.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("flagged_adversarial") not in {True, False}:
        errors.append("flagged_adversarial_invalid")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_invalid")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(field_principles()) - set(principles):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    rows = value.get("pilot_rows")
    if not isinstance(rows, list) or len(rows) != 8 or value.get("rows") != rows:
        errors.append("pilot_rows_invalid")
        rows = []
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("intended") != 8:
        errors.append("sample_size_budget_invalid")
    invocations = value.get("invocation_counts")
    if not isinstance(invocations, Mapping):
        errors.append("invocation_counts_invalid")
        invocations = {}
    planned_substrate_class = value.get("planned_inference_substrate_class")
    if planned_substrate_class != "model_bounded_generation":
        errors.append("planned_substrate_invalid")
    substrate_class = value.get("inference_substrate_class")
    if not isinstance(substrate_class, str):
        errors.append("inference_substrate_class_invalid")
        substrate_class = ""
    if value.get("verdict_class") == "blocked":
        actual_blocked_class = substrate_class
        if actual_blocked_class not in {"blocked_no_run", "model_load_no_generation"}:
            errors.append("blocked_substrate_invalid")
        expected_specs = [] if actual_blocked_class == "blocked_no_run" else MODEL_SPECS
        if (
            value.get("MODEL_SPECS") != expected_specs
            or invocations.get("generation_calls_attempted") != 0
        ):
            errors.append("blocked_invocations_nonzero")
        first = (value.get("gate_check_summary") or {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(first, Mapping) or required - set(first):
            errors.append("blocked_gate_diagnostic_incomplete")
    elif rows:
        try:
            reduction = reduce_pilot_rows(
                rows,
                transport_authenticated=(value.get("inference_runtime_receipt") or {}).get(
                    "transport_authenticated"
                )
                is True,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"pilot_reduction_invalid:{exc}")
        else:
            if value.get("pilot_reduction") != reduction:
                errors.append("pilot_reduction_mismatch")
            if (
                value.get("evidence_transport_ready_score")
                != reduction["evidence_transport_ready_score"]
            ):
                errors.append("evidence_transport_ready_score_mismatch")
        if value.get("MODEL_SPECS") != MODEL_SPECS:
            errors.append("model_specs_mandate_invalid")
        if invocations.get("generation_calls_attempted") != 8:
            errors.append("generation_call_count_invalid")
        if invocations.get("model_loads_attempted") != 1:
            errors.append("model_load_count_invalid")
        if substrate_class != "model_bounded_generation":
            errors.append("measured_substrate_invalid")
        duration = value.get("duration_s")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 10.0:
            errors.append("model_bounded_generation_duration_implausible")
        for name, score in (
            ("fit", "fit_capture_feasible_score"),
            ("eval", "eval_capture_feasible_score"),
        ):
            projection = value.get(f"{name}_capture_projection")
            projected = (projection or {}).get("projected_s")
            expected = int(
                isinstance(projected, (int, float))
                and not isinstance(projected, bool)
                and projected <= CAPTURE_LIMIT_S
                and (projection or {}).get("status", "measured_projection") == "measured_projection"
            )
            if value.get(score) != expected:
                errors.append(f"{score}_mismatch")
    receipt = value.get("current_work_receipt")
    if isinstance(receipt, Mapping) and receipt:
        errors.extend(
            f"current_work_receipt:{error}"
            for error in current_work_receipt.validate_current_work_receipt(receipt, root=root)
        )
    return list(dict.fromkeys(errors))


def load_json(path: Path) -> JsonDict:
    """Read one JSON object and reject scalar or list top levels."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("json_object_required")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read complete JSONL rows without tolerating blank or scalar records."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("jsonl_object_required")
        rows.append(value)
    return rows


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload exact candidate bytes and return an independent validity receipt."""

    value = load_json(path)
    errors = validate_artifact(value, root=root)
    return {
        "mode": "cold_replay",
        "valid": not errors,
        "errors": errors,
        "candidate_sha256": sha256_file(path),
        "pilot_count": len(value.get("pilot_rows") or []),
    }


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reduce persisted rows without trusting proposer summary fields."""

    value = load_json(path)
    errors = validate_artifact(value, root=root)
    rows = value.get("pilot_rows") or []
    blocked = value.get("verdict_class") == "blocked"
    reduction = (
        {"blocked_no_run": True, "pilot_count": len(rows)}
        if blocked
        else reduce_pilot_rows(
            rows,
            transport_authenticated=(value.get("inference_runtime_receipt") or {}).get(
                "transport_authenticated"
            )
            is True,
        )
    )
    return {
        "mode": "independent_reduce",
        "passed": not errors,
        "errors": errors,
        "candidate_sha256": sha256_file(path),
        "reduction": reduction,
        "row_reduction_sha256": canonical_hash(rows),
    }


def _source_hash_row(path: Path, *, producer: str, source_class: str) -> JsonDict:
    return {
        "path": str(path.resolve()),
        "producer": producer,
        "source_class": source_class,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _check(
    checks: list[JsonDict],
    name: str,
    *,
    upstream: str,
    path: Path,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    category: str = "validity",
) -> None:
    checks.append(
        gate_row(
            name,
            upstream=upstream,
            path=str(path),
            field=field,
            operator=operator,
            expected=expected,
            observed=observed,
            passed=passed,
            category=category,
        )
    )


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate the exact upstream terminal record and all model-facing bytes."""

    resolved = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    context: JsonDict = {}
    required = [SOURCE_RESULT, PROTOCOL_PATH, PILOT_INPUT, SPEC_PATH]
    for relative in required:
        path = resolved / relative
        exists = path.is_file() and path.stat().st_size > 0
        _check(
            checks,
            f"source_exists:{relative.name}",
            upstream="exp7602"
            if relative in {SOURCE_RESULT, PROTOCOL_PATH, PILOT_INPUT}
            else "spec",
            path=path,
            field="readable_nonempty_file",
            operator="eq",
            expected=True,
            observed=exists,
            passed=exists,
        )
    if any(row["passed"] is not True for row in checks):
        return checks, hashes, context
    artifact_path = resolved / SOURCE_RESULT
    artifact = load_json(artifact_path)
    hashes.append(
        _source_hash_row(
            artifact_path, producer="exp7602_terminal", source_class="authenticated_producer"
        )
    )
    expected_fields = {
        "honest_verdict": "complete_null_evidence_requalification_ready",
        "verdict_class": "null",
        "evidence_protocol_ready_score": 1,
        "flagged_adversarial": False,
    }
    for field, expected in expected_fields.items():
        observed = artifact.get(field)
        _check(
            checks,
            f"exp7602_{field}",
            upstream="exp7602_terminal",
            path=artifact_path,
            field=field,
            operator="eq",
            expected=expected,
            observed=observed,
            passed=observed == expected,
        )
    checksum = artifact.get("reproducibility_checksum")
    recomputed = source_protocol.reproducibility_checksum(artifact)
    _check(
        checks,
        "exp7602_reproducibility_checksum",
        upstream="exp7602_terminal",
        path=artifact_path,
        field="reproducibility_checksum",
        operator="eq",
        expected=checksum,
        observed=recomputed,
        passed=checksum == recomputed,
    )
    outcomes = artifact.get("terminal_reader_outcomes") or []
    expected_readers = {
        "declared_entrypoint_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    observed_readers = {
        str(row.get("name"))
        for row in outcomes
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    _check(
        checks,
        "exp7602_terminal_receipt",
        upstream="exp7602_terminal",
        path=artifact_path,
        field="terminal_reader_outcomes",
        operator="superset",
        expected=sorted(expected_readers),
        observed=sorted(observed_readers),
        passed=expected_readers <= observed_readers,
    )
    validation = artifact.get("validation_receipts") or []
    validation_ok = bool(validation) and all(
        isinstance(row, Mapping)
        and row.get("passed") is True
        and row.get("exit_code") == 0
        and isinstance(row.get("log_sha256"), str)
        for row in validation
    )
    _check(
        checks,
        "exp7602_validation_receipt",
        upstream="exp7602_terminal",
        path=artifact_path,
        field="validation_receipts",
        operator="all_exit_zero_with_log_hash",
        expected=True,
        observed=validation_ok,
        passed=validation_ok,
    )
    sidecars = artifact.get("raw_sidecars") or {}
    model_inputs = sidecars.get("model_inputs") if isinstance(sidecars, Mapping) else {}
    for role in ("fit", "tune", "policy", "online", "evaluation", "pilot"):
        receipt = model_inputs.get(role) if isinstance(model_inputs, Mapping) else None
        path = resolved / str(receipt.get("path") if isinstance(receipt, Mapping) else "missing")
        observed = (
            {
                "rows": len(load_jsonl(path)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file()
            else None
        )
        expected = (
            {key: receipt.get(key) for key in ("rows", "bytes", "sha256")}
            if isinstance(receipt, Mapping)
            else None
        )
        _check(
            checks,
            f"exp7602_{role}_model_inputs",
            upstream="exp7602_raw_sidecars",
            path=path,
            field=role,
            operator="eq",
            expected=expected,
            observed=observed,
            passed=observed == expected,
        )
        if observed is not None:
            hashes.append(
                _source_hash_row(
                    path, producer="exp7602_model_inputs", source_class="authenticated_producer"
                )
            )
    pilots = load_jsonl(resolved / PILOT_INPUT)
    pilot_valid = len(pilots) == 8 and len({row.get("component_hash") for row in pilots}) == 8
    try:
        pilot_valid = pilot_valid and all(validate_input_record(row) for row in pilots)
    except ValueError:
        pilot_valid = False
    _check(
        checks,
        "exp7602_complete_pilot_records",
        upstream="exp7602_pilot_model_inputs",
        path=resolved / PILOT_INPUT,
        field="complete_label_free_disjoint_records",
        operator="eq",
        expected={"rows": 8, "valid": True},
        observed={"rows": len(pilots), "valid": pilot_valid},
        passed=pilot_valid,
    )
    spec_text = (resolved / SPEC_PATH).read_text(encoding="utf-8")
    _check(
        checks,
        "driving_requirement",
        upstream="verification_spec",
        path=resolved / SPEC_PATH,
        field="REQ-*",
        operator="contains",
        expected="REQ-VERIFY-7604",
        observed="REQ-VERIFY-7604" if "REQ-VERIFY-7604" in spec_text else None,
        passed="REQ-VERIFY-7604" in spec_text,
    )
    hashes.append(
        _source_hash_row(
            resolved / SPEC_PATH, producer="capability_spec", source_class="conductor_pre_gate"
        )
    )
    context.update({"source_artifact": artifact, "pilots": pilots, "model_inputs": model_inputs})
    return checks, hashes, context


def _runtime_build_receipt() -> JsonDict:  # pragma: no cover - installed runtime boundary.
    """Bind the imported llama.cpp package and its native shared library bytes."""

    import llama_cpp

    module_path = Path(str(llama_cpp.__file__)).resolve()
    native_value = getattr(getattr(llama_cpp, "llama_cpp", None), "_lib", None)
    native_name = getattr(native_value, "_name", None)
    native_path = Path(str(native_name)).resolve() if native_name else None
    return {
        "llama_cpp_version": str(getattr(llama_cpp, "__version__", "unknown")),
        "module_path": str(module_path),
        "module_sha256": sha256_file(module_path),
        "native_library_path": str(native_path) if native_path and native_path.is_file() else None,
        "native_library_sha256": (
            sha256_file(native_path) if native_path and native_path.is_file() else None
        ),
    }


def _select_gpu() -> tuple[JsonDict | None, list[JsonDict]]:  # pragma: no cover - host boundary.
    """Select one idle card without changing or signaling any foreign process."""

    from carnot.experiment_7581_v662_arc_bounded_canary import gpu_inventory

    inventory = gpu_inventory()
    candidates = [
        deepcopy(dict(row))
        for row in inventory
        if int(row.get("memory_free_mb") or 0) >= GPU_REQUIRED_FREE_MB and not row.get("processes")
    ]
    selected = min(candidates, key=lambda row: int(row["index"])) if candidates else None
    return selected, inventory


def _post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> bytes:  # pragma: no cover
    """Send one exact JSON request and return the untouched response bytes."""

    request_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _timing_value(raw: Mapping[str, Any], name: str, fallback: str) -> float:
    timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
    value = timings.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) / 1000.0
    fallback_value = timings.get(fallback)
    if isinstance(fallback_value, (int, float)) and not isinstance(fallback_value, bool):
        return float(fallback_value) / 1000.0
    return 0.0


def _token_count(raw: Mapping[str, Any], timing_name: str, usage_name: str) -> int:
    timings = raw.get("timings") if isinstance(raw.get("timings"), Mapping) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    value = timings.get(timing_name)
    if not isinstance(value, int) or isinstance(value, bool):
        value = usage.get(usage_name)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _offload_receipt(
    log_path: Path | None, owned_vram_mb: int | None, parsed: Mapping[str, Any]
) -> JsonDict:
    """Authenticate CUDA offload from layer logs or owned VRAM plus load boundaries."""

    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path and log_path.is_file()
        else ""
    )
    buffer_matches = re.findall(r"CUDA\d+ model buffer size\s*=\s*([0-9.]+) MiB", text)
    layer_count = int(parsed.get("loaded_layers") or 0)
    fallback = bool(
        int(owned_vram_mb or 0) >= 15_000
        and "- CUDA0" in text
        and "model loaded" in text
        and "loading model" in text
    )
    return {
        **deepcopy(dict(parsed)),
        "actual_offload": layer_count > 0 or fallback,
        "authentication_method": (
            "reported_layer_count" if layer_count > 0 else "owned_vram_and_cuda_load_log"
        ),
        "owned_vram_mb": owned_vram_mb,
        "cuda_model_buffer_mb": float(buffer_matches[-1]) if buffer_matches else None,
        "warmup_calls": int("warming up the model with an empty run" in text),
        "log_path": str(log_path) if log_path else None,
        "log_sha256": sha256_file(log_path) if log_path and log_path.is_file() else None,
    }


def _completed_row(
    *,
    index: int,
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    request_path: Path,
    response_path: Path,
    response_bytes: bytes,
    call_start_ns: int,
    call_end_ns: int,
) -> JsonDict:
    """Reduce one raw server response and retain exact bytes and call costs."""

    raw = json.loads(response_bytes)
    if not isinstance(raw, dict):
        raise ValueError("response_object_required")
    choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    finish_reason = str(choice.get("finish_reason") or "unknown")
    parsed = parse_extraction_response(record, content, finish_reason)
    prompt_tokens = _token_count(raw, "prompt_n", "prompt_tokens")
    output_tokens = _token_count(raw, "predicted_n", "completion_tokens")
    request_bytes = request_path.read_bytes()
    return {
        "pilot_index": index,
        "component_hash": record["component_hash"],
        "arm": "evidence_link",
        "status": "completed",
        "transport_completed": True,
        "lossless_input": validate_input_record(record),
        **parsed,
        "request": deepcopy(dict(request_payload)),
        "input_record": deepcopy(dict(record)),
        "response_text": content,
        "reasoning_text": reasoning,
        "raw_response": raw,
        "request_path": str(request_path),
        "request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": len(request_bytes),
        "response_path": str(response_path),
        "response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
        "response_bytes": len(response_bytes),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "call_start_monotonic_ns": call_start_ns,
        "call_end_monotonic_ns": call_end_ns,
        "call_s": (call_end_ns - call_start_ns) / 1_000_000_000,
        "prefill_s": _timing_value(raw, "prompt_ms", "prompt_per_second_ms"),
        "decode_s": _timing_value(raw, "predicted_ms", "predicted_per_token_ms"),
        "numerator": int(parsed["usable_schema"] is True),
        "denominator": 1,
        "seed": RANDOM_SEED,
        "direction": "higher_is_more_transport_usable",
        "provenance": {"source": "exp7602", "call_id": f"generation-{index}"},
    }


def _failed_row(
    *,
    index: int,
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    request_path: Path,
    error: BaseException,
    call_start_ns: int,
    call_end_ns: int,
) -> JsonDict:
    """Retain one transport exception as its one charged pilot outcome."""

    request_bytes = request_path.read_bytes()
    return {
        "pilot_index": index,
        "component_hash": record["component_hash"],
        "arm": "evidence_link",
        "status": "failed",
        "transport_completed": False,
        "lossless_input": validate_input_record(record),
        "parser_outcome": "transport_failure",
        "usable_schema": False,
        "invalid_pointer_accepted": False,
        "evidence": [],
        "parser_error": f"{type(error).__name__}:{error}",
        "censoring": "transport_failure",
        "request": deepcopy(dict(request_payload)),
        "input_record": deepcopy(dict(record)),
        "response_text": "",
        "reasoning_text": "",
        "raw_response": None,
        "request_path": str(request_path),
        "request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": len(request_bytes),
        "response_path": None,
        "response_sha256": None,
        "response_bytes": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "finish_reason": "transport_failure",
        "call_start_monotonic_ns": call_start_ns,
        "call_end_monotonic_ns": call_end_ns,
        "call_s": (call_end_ns - call_start_ns) / 1_000_000_000,
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "numerator": 0,
        "denominator": 1,
        "seed": RANDOM_SEED,
        "direction": "higher_is_more_transport_usable",
        "provenance": {"source": "exp7602", "call_id": f"generation-{index}"},
    }


def _atomic_bytes(path: Path, value: bytes) -> None:  # pragma: no cover - durable runtime I/O.
    """Publish exact captured bytes after a flush and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _event(run_id: str, call_id: str, operation: str, state: str) -> JsonDict:
    return {
        "scope": "current",
        "transport": "owned_runtime",
        "run_id": run_id,
        "owner_pid": os.getpid(),
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": time.monotonic_ns(),
    }


def run_owned_pilot(
    *,
    root: Path,
    records: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    model_sha256: str,
    selected_gpu: Mapping[str, Any],
    lease: Any,
    started: float,
    run_dir: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - owned CUDA integration.
    """Load one owned server, issue eight calls once, and release only that server."""

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _call_with_heartbeats,
        _observed_offload_layers,
        _owned_vram_mb,
        process_start_tick,
    )

    if len(records) != 8:
        raise ValueError("exactly_eight_pilot_records_required")
    run_dir.mkdir(parents=True, exist_ok=False)
    requests_dir = run_dir / "requests"
    responses_dir = run_dir / "responses"
    requests_dir.mkdir()
    responses_dir.mkdir()
    model_path = Path(str(model_spec["model_path"])).resolve()
    previous_env = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CARNOT_ARC_GENERATOR_CUDA_GPU",
            "CARNOT_ARC_INDUCE_THINK",
            "CARNOT_ARC_INDUCE_THINKING_BUDGET",
            "CARNOT_ARC_SAMPLING_SEED",
            "CARNOT_ARC_SERVER_LOG_DIR",
        )
    }
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu["index"]),
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(selected_gpu["index"]),
            "CARNOT_ARC_INDUCE_THINK": "0",
            "CARNOT_ARC_SAMPLING_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_SERVER_LOG_DIR": str(run_dir / "server_logs"),
        }
    )
    os.environ.pop("CARNOT_ARC_INDUCE_THINKING_BUDGET", None)
    try:
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=str(model_path),
            port=_free_port(),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=32_768,
            max_tokens=MAX_TOKENS,
            timeout=900,
            tries=1,
            extra_server_args=("-lv", "4"),
        )
    except BaseException:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        raise
    proposer.model_repository = MODEL_ID
    proposer.requested_model_path = str(model_path)
    run_id = f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}"
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    runtime: JsonDict = {
        "transport_authenticated": False,
        "model_load_completed": False,
        "model_spec": deepcopy(dict(model_spec)),
        "model_sha256": model_sha256,
        "model_bytes": model_path.stat().st_size,
        "gpu_uuid": selected_gpu["uuid"],
        "gpu_index": selected_gpu["index"],
        "owner_pid": os.getpid(),
        "owner_pid_start_ticks": process_start_tick(os.getpid()),
        "lease_owner": lease.owner_receipt(),
        "lease_release": None,
        "signals_sent_to_foreign_processes": [],
        "warmup_calls": 0,
    }
    receipt_start_ns = time.monotonic_ns()
    load_start_ns = time.monotonic_ns()
    terminal_error: BaseException | None = None
    try:
        lease.transition("admitted")
        lease.transition("loading")
        events.append(_event(run_id, "model-load-1", "model_load", "attempted"))
        progress(started, "model_load", "before", gpu_uuid=selected_gpu["uuid"])
        try:
            healthy = _call_with_heartbeats(
                proposer._ensure_server, started=started, phase="model_load"
            )
        except BaseException:
            events.append(_event(run_id, "model-load-1", "model_load", "failed"))
            raise
        load_end_ns = time.monotonic_ns()
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
        server_log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        owned_vram_mb = _owned_vram_mb(server_pid)
        offload = _offload_receipt(
            server_log_path,
            owned_vram_mb,
            _observed_offload_layers(server_log_path),
        )
        runtime.update(
            {
                "model_load_completed": bool(healthy),
                "model_load_start_monotonic_ns": load_start_ns,
                "model_load_end_monotonic_ns": load_end_ns,
                "model_load_s": (load_end_ns - load_start_ns) / 1_000_000_000,
                "server_pid": server_pid,
                "server_pid_start_ticks": process_start_tick(server_pid),
                "server_command": list(getattr(proposer, "last_launch_argv", ()) or ()),
                "owned_vram_mb": owned_vram_mb,
                "offload_layers": offload,
                "runtime_build": _runtime_build_receipt(),
                "observed_model_path": proposer.observed_model_path(),
                "server_props": proposer.server_props() if healthy else {},
                "warmup_calls": offload["warmup_calls"],
            }
        )
        progress(
            started,
            "model_load",
            "after",
            healthy=healthy,
            server_pid=server_pid,
            offload=offload.get("loaded_layers") or offload.get("authentication_method"),
        )
        load_authenticated = bool(
            healthy
            and runtime["server_pid_start_ticks"] is not None
            and int(runtime.get("owned_vram_mb") or 0) > 0
            and offload.get("actual_offload") is True
            and runtime["runtime_build"].get("native_library_sha256")
        )
        if not load_authenticated:
            events.append(_event(run_id, "model-load-1", "model_load", "failed"))
            raise RuntimeError("owned_model_load_not_authenticated")
        events.append(_event(run_id, "model-load-1", "model_load", "completed"))
        lease.transition("resident", vram_mb=int(runtime["owned_vram_mb"]))
        lease.transition("inferencing")
        for index, record in enumerate(records, 1):
            payload = build_extraction_request(record)
            request_bytes = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            request_path = requests_dir / f"{index:02d}.json"
            response_path = responses_dir / f"{index:02d}.json"
            _atomic_bytes(request_path, request_bytes)
            events.append(_event(run_id, f"generation-{index}", "generation", "attempted"))
            call_start_ns = time.monotonic_ns()
            progress(started, "generation", "before", unit=f"{index}/8")
            try:
                response_bytes = _call_with_heartbeats(
                    lambda payload=payload: _post_json(
                        proposer._url() + "/v1/chat/completions", payload, 900.0
                    ),
                    started=started,
                    phase=f"generation_{index}",
                )
                call_end_ns = time.monotonic_ns()
                _atomic_bytes(response_path, response_bytes)
                row = _completed_row(
                    index=index,
                    record=record,
                    request_payload=payload,
                    request_path=request_path,
                    response_path=response_path,
                    response_bytes=response_bytes,
                    call_start_ns=call_start_ns,
                    call_end_ns=call_end_ns,
                )
                events.append(_event(run_id, f"generation-{index}", "generation", "completed"))
            except BaseException as exc:  # one failed call remains one outcome.
                call_end_ns = time.monotonic_ns()
                row = _failed_row(
                    index=index,
                    record=record,
                    request_payload=payload,
                    request_path=request_path,
                    error=exc,
                    call_start_ns=call_start_ns,
                    call_end_ns=call_end_ns,
                )
                events.append(_event(run_id, f"generation-{index}", "generation", "failed"))
            rows.append(row)
            current_work_receipt.atomic_json(run_dir / "checkpoint.json", {"pilot_rows": rows})
            progress(
                started,
                "generation",
                "after",
                unit=f"{index}/8",
                parser_outcome=row["parser_outcome"],
            )
    except BaseException as exc:
        terminal_error = exc
    finally:
        progress(started, "model_unload", "before")
        proposer.stop()
        progress(started, "model_unload", "after")
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete" if len(rows) == 8 else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    receipt_end_ns = time.monotonic_ns()
    if terminal_error is not None:
        raise terminal_error
    costs_complete = all(
        int(row.get("prompt_tokens") or 0) > 0
        and float(row.get("prefill_s") or 0) > 0
        and float(row.get("decode_s") or 0) >= 0
        for row in rows
        if row.get("transport_completed") is True
    )
    runtime["transport_authenticated"] = bool(
        len(rows) == 8
        and all(row.get("transport_completed") is True for row in rows)
        and costs_complete
    )
    receipt = current_work_receipt.build_current_work_receipt(
        run_id=run_id,
        owner_pid=os.getpid(),
        events=events,
        inference_substrate="model_bounded_generation",
        inference_substrate_details={
            "model_id": MODEL_ID,
            "gpu_uuid": selected_gpu["uuid"],
            "fixed_output_budget": MAX_TOKENS,
        },
        inference_substrate_class="model_bounded_generation",
        execution_venue="host",
        started_monotonic_ns=receipt_start_ns,
        ended_monotonic_ns=receipt_end_ns,
        phase_spans=[],
    )
    receipt["MODEL_SPECS"] = MODEL_SPECS
    current_work_receipt.atomic_json(run_dir / "current_work_receipt.json", receipt)
    current_work_receipt.atomic_json(run_dir / "runtime_receipt.json", runtime)
    return rows, runtime, receipt


def affected_validation_manifest() -> JsonDict:
    """Freeze the only files admitted to current scoped validation."""

    return {
        "schema": "carnot.exp7604.affected_validation_manifest.v1",
        "requirement": "REQ-VERIFY-7604",
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
        "spec_paths": [SPEC_PATH.as_posix()],
        "numbered_e2e": [],
        "numbered_e2e_non_applicability": (
            "This task reports bounded extraction calls and has no numbered model E2E."
        ),
    }


def run_scoped_checks(
    root: Path, private_root: Path
) -> list[JsonDict]:  # pragma: no cover - subprocess integration.
    """Run the fixed serial checks with private pytest and coverage state."""

    manifest = affected_validation_manifest()
    basetemp = private_root / "basetemp"
    coverage_file = private_root / "coverage" / ".coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        manifest["test_paths"],
        manifest["changed_modules"],
        static_paths=manifest["static_paths"],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    return validation_scope.run_commands(
        root,
        commands,
        log_dir=private_root / "logs" / "scoped",
        heartbeat_s=60.0,
    )


def terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare the fresh exact-candidate readers that control publication."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _all_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def _reader_outcomes(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "name": row.get("name"),
            "exit_code": row.get("exit_code"),
            "passed": row.get("passed") is True,
            "log_sha256": row.get("log_sha256"),
            "worktree": row.get("worktree", str(REPO_ROOT)),
        }
        for row in receipts
    ]


def _finalize_candidate(
    *,
    root: Path,
    artifact: JsonDict,
    private_root: Path,
    started: float,
) -> int:  # pragma: no cover - exact candidate subprocess boundary.
    """Run fresh readers and atomically publish only their accepted candidate."""

    candidate = private_root / "terminal_candidate.json"
    current_work_receipt.atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before")
    receipts = validation_scope.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=private_root / "logs" / "terminal",
        heartbeat_s=60.0,
    )
    progress(started, "terminal_readers", "after", passed=_all_passed(receipts))
    if not _all_passed(receipts):
        return 1
    artifact["validation_receipts"] = [
        *list(artifact.get("validation_receipts") or []),
        *[deepcopy(dict(row)) for row in receipts],
    ]
    artifact["terminal_reader_outcomes"] = _reader_outcomes(receipts)
    artifact["flagged_adversarial"] = False
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, root=root)
    if errors:
        print(json.dumps({"publication_blocked": errors}, sort_keys=True), flush=True)
        return 1
    publication = root / RESULT_PATH
    current_work_receipt.atomic_json(publication, artifact)
    progress(started, "publish", "after", path=publication, sha256=sha256_file(publication))
    return 0


def _empty_current_receipt(start_ns: int, end_ns: int, reason: str) -> JsonDict:
    receipt = current_work_receipt.build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}:{os.getpid()}:{start_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="blocked_no_run",
        inference_substrate_details={"reason": reason},
        inference_substrate_class="blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=end_ns,
    )
    receipt["MODEL_SPECS"] = []
    return receipt


def _add_changed_source_hashes(root: Path, hashes: list[JsonDict]) -> None:
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = root / relative
        if path.is_file() and not any(row.get("path") == str(path.resolve()) for row in hashes):
            hashes.append(
                _source_hash_row(
                    path,
                    producer="exp7604_current_source",
                    source_class="conductor_pre_gate",
                )
            )


def _private_root() -> Path:  # pragma: no cover - host temporary directory.
    return Path(tempfile.mkdtemp(prefix="carnot-exp7604-", dir="/tmp")).resolve()


def _scoped_receipts(
    root: Path, private_root: Path, started: float
) -> list[JsonDict]:  # pragma: no cover
    progress(started, "scoped_validation", "before")
    receipts = run_scoped_checks(root, private_root)
    for row in receipts:
        row["worktree"] = str(root)
        row["command_category"] = "required_validation"
    progress(started, "scoped_validation", "after", passed=_all_passed(receipts))
    return receipts


def _publish_blocked(
    *,
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    hashes: list[JsonDict],
    reason: str,
    started: float,
    started_ns: int,
    private_root: Path,
) -> int:  # pragma: no cover - production publication flow.
    """Validate and publish a complete unchanged external no-run block."""

    _add_changed_source_hashes(root, hashes)
    receipts = _scoped_receipts(root, private_root, started)
    if not _all_passed(receipts):
        return 1
    end_ns = time.monotonic_ns()
    artifact = build_blocked_artifact(
        checks=checks,
        source_hashes=hashes,
        duration_s=(end_ns - started_ns) / 1_000_000_000,
        reason=reason,
        current_receipt=_empty_current_receipt(started_ns, end_ns, reason),
    )
    artifact["validation_receipts"] = receipts
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return _finalize_candidate(
        root=root,
        artifact=artifact,
        private_root=private_root,
        started=started,
    )


def _capture_records(
    root: Path, model_inputs: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load the frozen 120 fit-side and 120 evaluation-side request rosters."""

    def role(name: str) -> list[JsonDict]:
        receipt = model_inputs[name]
        rows = load_jsonl(root / str(receipt["path"]))
        if len(rows) != int(receipt["rows"]):
            raise ValueError(f"capture_role_count_changed:{name}")
        if not all(validate_input_record(row) for row in rows):
            raise ValueError(f"capture_role_invalid:{name}")
        return rows

    fit = [*role("fit"), *role("tune"), *role("policy")]
    evaluation = [*role("online"), *role("evaluation")]
    if len(fit) != CAPTURE_GROUPS or len(evaluation) != CAPTURE_GROUPS:
        raise ValueError("capture_roster_not_120_groups")
    return fit, evaluation


def _projection_or_blocked(
    rows: Sequence[Mapping[str, Any]],
    target_tokens: Sequence[int],
    runtime: Mapping[str, Any],
) -> JsonDict:
    """Return a zero score when measured cost components are not complete."""

    try:
        projection = estimate_capture(
            rows,
            target_tokens,
            model_load_s=float(runtime.get("model_load_s") or 0),
            validation_reserve_s=VALIDATION_RESERVE_S,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "insufficient_measured_cost",
            "reason": f"{type(exc).__name__}:{exc}",
            "group_count": CAPTURE_GROUPS,
            "projected_s": None,
            "threshold_s": CAPTURE_LIMIT_S,
            "feasible_score": 0,
        }
    if runtime.get("transport_authenticated") is not True:
        projection["status"] = "transport_not_authenticated"
        projection["feasible_score"] = 0
    else:
        projection["status"] = "measured_projection"
    return projection


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - declared E2E.
    """Authenticate, measure eight calls, validate, and publish one terminal result."""

    resolved = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    private_root = _private_root()
    progress(started, "start", "begin", root=resolved, run_date=run_date)
    if not resolved.is_dir() or run_date != RUN_DATE:
        print("root_or_run_date_invalid", flush=True)
        return 2
    checks, hashes, context = collect_preconditions(resolved)
    progress(
        started,
        "preconditions",
        "after",
        passed=all(row.get("passed") is True for row in checks),
    )
    if any(row.get("passed") is not True for row in checks):
        first = next(row for row in checks if row.get("passed") is not True)
        return _publish_blocked(
            root=resolved,
            checks=checks,
            hashes=hashes,
            reason=str(first["check"]).replace(":", "_"),
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    progress(started, "model_cache", "before")
    cached = cached_current_model(gpu_index=0, preferred_quant="Q4_K_M")
    cache_ok = isinstance(cached, Mapping) and Path(str(cached.get("model_path"))).is_file()
    cache_path = (
        Path(str(cached.get("model_path"))) if isinstance(cached, Mapping) else resolved / "missing"
    )
    _check(
        checks,
        "model_cache",
        upstream="cached_current_model",
        path=cache_path,
        field="Q4_K_M_cached_path",
        operator="is_file",
        expected=True,
        observed=cache_ok,
        passed=cache_ok,
        category="readiness",
    )
    progress(started, "model_cache", "after", passed=cache_ok, path=cache_path)
    if not cache_ok:
        return _publish_blocked(
            root=resolved,
            checks=checks,
            hashes=hashes,
            reason="model_cache",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    selected, inventory = _select_gpu()
    gpu_ok = selected is not None
    _check(
        checks,
        "exclusive_cuda_capacity",
        upstream="nvidia-smi",
        path=resolved / RAW_DIR,
        field="idle_gpu_with_required_free_mb",
        operator="gte",
        expected=GPU_REQUIRED_FREE_MB,
        observed=selected if selected is not None else inventory,
        passed=gpu_ok,
        category="readiness",
    )
    progress(
        started,
        "cuda_capacity",
        "after",
        passed=gpu_ok,
        gpu_uuid=selected.get("uuid") if selected else None,
    )
    if selected is None:
        return _publish_blocked(
            root=resolved,
            checks=checks,
            hashes=hashes,
            reason="exclusive_cuda_capacity",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    cached = dict(cached)
    cached["gpu"] = int(selected["index"])
    model_path = Path(str(cached["model_path"])).resolve()
    from carnot.gpu_lease_phase_journal import GpuLease

    lease_root = resolved / RAW_DIR / "gpu_leases"
    progress(started, "gpu_lease", "before", gpu_uuid=selected["uuid"])
    try:
        lease = GpuLease.acquire(
            runtime_dir=lease_root,
            task_id=EXPERIMENT_ID,
            device_uuid=str(selected["uuid"]),
            expected_model=str(model_path),
            vram_before_mb=int(selected["memory_used_mb"]),
            ttl_s=4800.0,
        )
    except Exception as exc:  # exact lease failure becomes terminal evidence.
        _check(
            checks,
            "exclusive_gpu_lease",
            upstream="carnot.gpu_lease_phase_journal.GpuLease",
            path=lease_root,
            field="exclusive_owner",
            operator="acquired",
            expected={"gpu_uuid": selected["uuid"], "task_id": EXPERIMENT_ID},
            observed={"error": f"{type(exc).__name__}:{exc}"},
            passed=False,
            category="readiness",
        )
        progress(started, "gpu_lease", "after", acquired=False)
        return _publish_blocked(
            root=resolved,
            checks=checks,
            hashes=hashes,
            reason="exclusive_gpu_lease",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    progress(started, "gpu_lease", "after", acquired=True, lease_id=lease.lease_id)
    from carnot.experiment_7581_v662_arc_bounded_canary import _call_with_heartbeats

    declared_model_path = Path(str(cached["model_path"]))
    progress(started, "model_hash", "before", bytes=model_path.stat().st_size)
    model_hash = _call_with_heartbeats(
        lambda: sha256_file(model_path), started=started, phase="model_hash"
    )
    progress(started, "model_hash", "after", sha256=model_hash)
    model_bytes_ok = (
        "Q4_K_M" in declared_model_path.name and model_path.stat().st_size > 15_000_000_000
    )
    _check(
        checks,
        "q4_k_m_model_bytes",
        upstream="cached_current_model",
        path=model_path,
        field="name_and_bytes",
        operator="authenticated",
        expected={"quantization": "Q4_K_M", "minimum_bytes": 15_000_000_000},
        observed={
            "name": declared_model_path.name,
            "resolved_path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": model_hash,
        },
        passed=model_bytes_ok,
        category="readiness",
    )
    hashes.append(
        _source_hash_row(
            model_path, producer="cached_current_model", source_class="authenticated_model"
        )
    )
    progress(started, "tokenizer_preflight", "before")
    tokenizer_ok, tokenizer_detail = _call_with_heartbeats(
        lambda: gguf_tokenizer_loadable(str(model_path)),
        started=started,
        phase="tokenizer_preflight",
    )
    progress(started, "tokenizer_preflight", "after", passed=tokenizer_ok)
    _check(
        checks,
        "embedded_tokenizer",
        upstream="gguf_tokenizer_loadable",
        path=model_path,
        field="embedded_gguf_tokenizer",
        operator="loadable",
        expected=True,
        observed={"passed": tokenizer_ok, "detail": tokenizer_detail},
        passed=tokenizer_ok,
        category="readiness",
    )
    if not model_bytes_ok or not tokenizer_ok:
        lease.transition("terminal_blocked")
        lease.release()
        return _publish_blocked(
            root=resolved,
            checks=checks,
            hashes=hashes,
            reason="model_bytes_or_tokenizer",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    run_dir = resolved / RAW_DIR / "runs" / f"{int(time.time())}-{os.getpid()}"
    progress(started, "owned_pilot", "before", run_dir=run_dir)
    try:
        rows, runtime, receipt = run_owned_pilot(
            root=resolved,
            records=context["pilots"],
            model_spec=cached,
            model_sha256=model_hash,
            selected_gpu=selected,
            lease=lease,
            started=started,
            run_dir=run_dir,
        )
    except BaseException as exc:
        _check(
            checks,
            "owned_model_transport",
            upstream="LocalGGUFProposer",
            path=run_dir,
            field="one_load_eight_calls",
            operator="completed",
            expected={"loads": 1, "calls": 8},
            observed={"error": f"{type(exc).__name__}:{exc}"},
            passed=False,
            category="readiness",
        )
        progress(started, "owned_pilot", "after", passed=False, error=type(exc).__name__)
        failed_counts = _zero_invocations()
        failed_counts.update({"model_loads_attempted": 1, "model_loads_failed": 1})
        _add_changed_source_hashes(resolved, hashes)
        validation = _scoped_receipts(resolved, private_root, started)
        if not _all_passed(validation):
            return 1
        end_ns = time.monotonic_ns()
        artifact = build_blocked_artifact(
            checks=checks,
            source_hashes=hashes,
            duration_s=(end_ns - started_ns) / 1_000_000_000,
            reason="owned_model_transport",
            invocation_counts=failed_counts,
            actual_substrate_class="model_load_no_generation",
        )
        artifact["validation_receipts"] = validation
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return _finalize_candidate(
            root=resolved, artifact=artifact, private_root=private_root, started=started
        )
    progress(started, "owned_pilot", "after", passed=True, calls=len(rows))
    fit_records, eval_records = _capture_records(resolved, context["model_inputs"])
    progress(started, "capture_estimation", "before", fit=120, evaluation=120)
    fit_tokens = prompt_token_projections(fit_records, rows)
    eval_tokens = prompt_token_projections(eval_records, rows)
    fit_projection = _projection_or_blocked(rows, fit_tokens, runtime)
    eval_projection = _projection_or_blocked(rows, eval_tokens, runtime)
    progress(
        started,
        "capture_estimation",
        "after",
        fit_s=fit_projection.get("projected_s"),
        eval_s=eval_projection.get("projected_s"),
    )
    for path in sorted(run_dir.rglob("*.json")):
        hashes.append(
            _source_hash_row(
                path,
                producer="exp7604_owned_runtime",
                source_class="immutable_raw_evidence",
            )
        )
    _add_changed_source_hashes(resolved, hashes)
    manifest_path = resolved / RAW_DIR / "affected_validation_manifest.json"
    current_work_receipt.atomic_json(manifest_path, affected_validation_manifest())
    hashes.append(
        _source_hash_row(
            manifest_path,
            producer="exp7604_validation_manifest",
            source_class="conductor_pre_gate",
        )
    )
    receipts = _scoped_receipts(resolved, private_root, started)
    if not _all_passed(receipts):
        return 1
    end_ns = time.monotonic_ns()
    artifact = build_complete_artifact(
        pilot_rows=rows,
        runtime_receipt=runtime,
        source_hashes=hashes,
        fit_projection=fit_projection,
        eval_projection=eval_projection,
        duration_s=(end_ns - started_ns) / 1_000_000_000,
        validation_receipts=receipts,
        current_receipt=receipt,
    )
    errors = validate_artifact(artifact, root=resolved)
    if errors:
        print(json.dumps({"candidate_invalid": errors}, sort_keys=True), flush=True)
        return 1
    return _finalize_candidate(
        root=resolved,
        artifact=artifact,
        private_root=private_root,
        started=started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse production and read-only candidate modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the producer or one fresh-process reader without new inference."""

    args = parse_args(argv)
    root = args.root.resolve()
    selected = [args.validate, args.cold_replay, args.independent_reduce]
    if sum(value is not None for value in selected) > 1:
        print("reader_modes_are_mutually_exclusive", flush=True)
        return 2
    if args.validate is not None:
        path = _argument_path(args.validate, root)
        errors = validate_artifact(load_json(path), root=root)
        print(json.dumps({"mode": "validate", "valid": not errors, "errors": errors}), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        outcome = cold_replay(_argument_path(args.cold_replay, root), root=root)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(outcome["valid"] is not True)
    if args.independent_reduce is not None:
        outcome = independent_reduce_artifact(
            _argument_path(args.independent_reduce, root), root=root
        )
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(outcome["passed"] is not True)
    return run_experiment(root, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
