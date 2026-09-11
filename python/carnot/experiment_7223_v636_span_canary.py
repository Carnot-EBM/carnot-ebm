"""Run one authenticated Qwen3.8 span extraction canary.

The model sees eight public calibration units. Source relations and claims use
separate requests. The same raw replies then feed syntax and semantic checks.

Spec refs: REQ-VERIFY-7223 and SCENARIO-VERIFY-7223-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

from carnot import experiment_7208_v635_span_fixture as fixture
from carnot import experiment_7209_v635_span_canary as capture
from carnot import experiment_7222_v636_span_fixture as upstream_fixture
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
Tokenize = Callable[[bytes], Sequence[int]]

RUN_DATE = "20260911"
TASK_ID = "experiment_7223_v636_span_canary"
RANDOM_SEED = 7_223_001
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]

RESULT_PATH = Path("results/experiment_7223_v636_span_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7223")
RAW_DIR = Path("results/raw/experiment_7223")
UPSTREAM_PATH = Path("results/experiment_7222_v636_span_fixture.json")
PUBLIC_PATH = Path("results/raw/experiment_7222/public.jsonl")
AUTHORITY_PATH = Path("results/raw/experiment_7222/authority.jsonl")
FIXTURE_MANIFEST_PATH = Path("results/raw/experiment_7222/manifest.json")
HISTORICAL_EXP7209_PATH = Path("results/experiment_7209_v635_span_canary.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7223_v636_span_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7223_v636_span_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7223_v636_span_canary.py")

PINNED_UPSTREAM_SHA256 = "sha256:50f3f05fe04b95911e528fda6f08936589b4e423d8dc19b2c372c5c50e44ad94"
PINNED_PUBLIC_SHA256 = "sha256:e063654faf39d76e9dbfdf63df01119871def3c0c0ad853dfea98b6c3fbbbe0a"
PINNED_AUTHORITY_SHA256 = "sha256:569bd663b08ada100dd41c04d52ff530b55ef6649c331380cb624301ff2f0615"
PINNED_MANIFEST_SHA256 = "sha256:5467dcdf1a889e4adc0ebf2c64791aec024102a06c0d4555101b7d27dcc2becb"
PINNED_EXP7209_SHA256 = "sha256:19bb1d3bd2dd142a0ca31469fd7c5992c94464485cfcab8b25cfaa3e6ad1afe2"

UPSTREAM_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "verdict_class": "circular_positive",
    "honest_verdict": "complete_circular_positive_span_fixture_ready_no_distinct_verifier_value",
    "span_fixture_ready_score": 1,
    "public_view_path": PUBLIC_PATH.as_posix(),
    "authority_sidecar_path": AUTHORITY_PATH.as_posix(),
    "fixture_manifest_path": FIXTURE_MANIFEST_PATH.as_posix(),
}

SOURCE_TOKEN_BUDGET = 384
CLAIM_TOKEN_BUDGET = 128
CONTEXT_TOKEN_BUDGET = 8192
MODEL_LOAD_CAP_S = 240.0
REQUEST_CAP_S = 90.0
INFERENCE_DEADLINE_S = 1500.0
TOKEN_BUDGETS = {"source": SOURCE_TOKEN_BUDGET, "claim": CLAIM_TOKEN_BUDGET}
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": RANDOM_SEED,
    "cache_prompt": False,
}
PROMPT_TEMPLATES = deepcopy(capture.PROMPT_TEMPLATES)
REQUIRED_VARIANTS = ("supported", "reversal", "joint_support", "support_removed")
RELATION_FAMILIES = ("precedes", "starts before", "ends before", "occurs before")
AUTHORITY_ONLY_FIELDS = {
    "alpha_rename_base_id",
    "alpha_rename_variant",
    "base_id",
    "expected_decision",
    "relation_family",
    "source_relation_count",
    "split",
    "variant",
    "claim_relation",
}

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Annotate actual values in this map; do not wrap arbitrary dictionaries as principle/value records.",
    "status": "Write a terminal artifact only when done or externally blocked; running checkpoints use a different path.",
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": "Use the recognized literal for the work actually executed; custom free text caused the Exp7208 quarantine.",
    "inference_substrate_class": "Match actual generation, load-only, CPU or aggregation work and its duration floor.",
    "inference_mode": "Use live_gpu only for actual task-owned CUDA generation.",
    "execution_venue": "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host.",
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": "Per unit/arm/seed metric, error and abstention for every comparison; retain full denominators.",
    "sample_size_budget": "Planned, attempted, completed, censored and independent units; no silent removal.",
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when correctness authority is reused as the verifier; independent code alone is not distinct authority.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished own work only.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. A failed acceptance gate forbids positive.",
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "span_canary_ready_score": "Exact 7/8 parse and 6/8 semantic calibration gate plus clean provenance.",
    "span_canary_complete_score": "All scheduled bounded rows are retained.",
    "canary_rows": "Every selected base, raw output, parse and semantics result.",
    "frozen_decoding_contract": "Model, prompt, grammar, parameters and hashes for the capture.",
    "token_budget_receipt": "Actual generated tokens and bounded stops.",
    "model_identity_receipt": "GGUF revision/hash, exact loader and tokenizer, and actual CUDA execution.",
    "gpu_receipts": "Task-owned lease and correlated CUDA observations.",
    "phase_spans": "Monotonic load/generate/score/cleanup spans.",
    "runner_receipt": "One model, runner choice, PID identity and cleanup.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

canonical_json = capture.canonical_json
sha256_bytes = capture.sha256_bytes
sha256_json = capture.sha256_json
sha256_file = capture.sha256_file
unwrap_principle = capture.unwrap_principle
is_quarantined = capture.is_quarantined
load_yaml = capture.load_yaml
gate_row = capture.gate_row
gate_summary = capture.gate_summary
build_completion_row = capture.build_completion_row
measure_token_budgets = capture.measure_token_budgets


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local duration."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read exact object rows without repairing malformed evidence."""

    return capture._read_jsonl(path)


def load_calibration_selection(
    public_path: Path, authority_path: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Choose eight balanced calibration units without reading a model result."""

    public_rows = _read_jsonl(public_path)
    authority_rows = _read_jsonl(authority_path)
    if len(public_rows) != 320 or len(authority_rows) != 320:
        raise ValueError("exp7222_panel_denominator")
    public_by_id = {str(row.get("unit_id")): row for row in public_rows}
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if len(public_by_id) != 320 or set(public_by_id) != set(authority_ids):
        raise ValueError("exp7222_public_authority_identity")

    variant_plan = (
        ("supported", "reversal"),
        ("joint_support", "support_removed"),
        ("supported", "reversal"),
        ("joint_support", "support_removed"),
    )
    selected_authority: list[JsonDict] = []
    for family, variants in zip(RELATION_FAMILIES, variant_plan, strict=True):
        family_rows = [
            row
            for row in authority_rows
            if row.get("split") == "canary" and row.get("relation_family") == family
        ]
        bases = list(dict.fromkeys(str(row["base_id"]) for row in family_rows))
        if len(bases) != 2:
            raise ValueError("canary_family_base_denominator")
        for base_id, variant in zip(bases, variants, strict=True):
            matches = [
                row
                for row in family_rows
                if row.get("base_id") == base_id and row.get("variant") == variant
            ]
            if len(matches) != 1:
                raise ValueError("calibration_selection_cell")
            selected_authority.append(deepcopy(matches[0]))
    selected_public = [deepcopy(public_by_id[str(row["unit_id"])]) for row in selected_authority]
    if any(set(row) != {"unit_id", "source_text", "claim_text"} for row in selected_public):
        raise ValueError("public_view_contains_private_fields")
    return selected_public, selected_authority


def _call_prompt(call_type: str, input_text: str) -> str:
    """Render one request from only the source or only the claim text."""

    return PROMPT_TEMPLATES[call_type].format(input_text=input_text)


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 16 public-only calls before opening model weights."""

    if len(public_rows) != 8 or len(authority_rows) != 8:
        raise ValueError("calibration_selection_denominator")
    authority_ids = [str(row["unit_id"]) for row in authority_rows]
    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    if len(public_by_id) != 8 or set(public_by_id) != set(authority_ids):
        raise ValueError("calibration_public_authority_identity")
    if any(row.get("split") != "canary" for row in authority_rows):
        raise ValueError("non_calibration_authority")

    schedule: list[JsonDict] = []
    for unit_id in authority_ids:
        public = public_by_id[unit_id]
        for call_type, field in (("source", "source_text"), ("claim", "claim_text")):
            input_text = str(public[field])
            input_bytes = input_text.encode("utf-8")
            grammar = fixture.compile_grammar(input_bytes, call_type, "grammar_only")
            prompt = _call_prompt(call_type, input_text)
            call_id = sha256_json(
                {
                    "unit_id": unit_id,
                    "call_type": call_type,
                    "seed": RANDOM_SEED,
                    "schedule": TASK_ID,
                }
            )
            schedule.append(
                {
                    "call_order": len(schedule),
                    "call_id": call_id,
                    "unit_id": unit_id,
                    "arm": "syntax_only_capture",
                    "call_type": call_type,
                    "seed": RANDOM_SEED,
                    "model_input": {field: input_text},
                    "input_text": input_text,
                    "input_sha256": sha256_bytes(input_bytes),
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "grammar": grammar["grammar"],
                    "grammar_sha256": grammar["grammar_sha256"],
                    "reference_grammar_sha256": fixture.compile_grammar(
                        input_bytes, call_type, "reference"
                    )["grammar_sha256"],
                    "output_token_budget": TOKEN_BUDGETS[call_type],
                    "context_token_budget": CONTEXT_TOKEN_BUDGET,
                    "request_timeout_s": REQUEST_CAP_S,
                    "decoding_parameters": deepcopy(DECODING_PARAMETERS),
                    "cold_request": True,
                }
            )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Name each change from the one predeclared schedule."""

    try:
        expected = build_schedule(public_rows, authority_rows)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != len(expected):
        errors.append("schedule_count")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field, value in wanted.items():
            if observed.get(field) != value:
                errors.append(f"call_{index}:{field}")
        if set(observed) - set(wanted):
            errors.append(f"call_{index}:extra_fields")
    return errors


def selection_receipt(
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Seal why the eight calibration rows were eligible before inference."""

    variants = Counter(str(row["variant"]) for row in authority_rows)
    return {
        "selection_frozen_before_inference": True,
        "selection_rule": "two fixed variants per relation family in Exp7222 canary order",
        "selected_unit_ids": [str(row["unit_id"]) for row in authority_rows],
        "selected_unit_count": len(authority_rows),
        "variant_counts": dict(sorted(variants.items())),
        "relation_families": sorted({str(row["relation_family"]) for row in authority_rows}),
        "public_rows_sha256": sha256_json(list(public_rows)),
        "authority_rows_sha256": sha256_json(list(authority_rows)),
        "schedule_sha256": sha256_json(list(schedule)),
        "held_out_rows_read_for_selection": 0,
        "model_outcomes_read_for_selection": 0,
        "authority_fields_in_model_schedule": sum(
            bool(set(row) & AUTHORITY_ONLY_FIELDS) for row in schedule
        ),
    }


def upstream_gate_rows(
    upstream: Mapping[str, Any],
    upstream_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    manifest_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate Exp7222 bytes and preserve the real Exp7209 diagnosis."""

    hashes = {
        "artifact": sha256_bytes(upstream_bytes),
        "public": sha256_bytes(public_bytes),
        "authority": sha256_bytes(authority_bytes),
        "manifest": sha256_bytes(manifest_bytes),
    }
    expected_hashes = {
        "artifact": PINNED_UPSTREAM_SHA256,
        "public": PINNED_PUBLIC_SHA256,
        "authority": PINNED_AUTHORITY_SHA256,
        "manifest": PINNED_MANIFEST_SHA256,
    }
    rows = [
        gate_row(
            "exact_upstream_bytes",
            expected_hashes,
            hashes,
            hashes == expected_hashes,
            upstream="experiment_7222",
            field="artifact_and_sidecar_bytes",
        )
    ]
    quarantined = is_quarantined(upstream)
    rows.append(
        gate_row(
            "structured_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="experiment_7222",
            field="flagged_adversarial|quarantined|fabricated",
        )
    )
    excluded = capture._manifest_hits(
        exclusion_manifest,
        {
            "Exp7222",
            "experiment_7222_v636_span_fixture",
            UPSTREAM_PATH.as_posix(),
        },
    )
    rows.append(
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="experiment_7222",
            field="experiment_ids",
        )
    )
    observed_fields = {
        field: unwrap_principle(upstream.get(field)) for field in UPSTREAM_EXPECTED_FIELDS
    }
    rows.append(
        gate_row(
            "producer_gate_fields",
            UPSTREAM_EXPECTED_FIELDS,
            observed_fields,
            observed_fields == UPSTREAM_EXPECTED_FIELDS,
            upstream="experiment_7222",
            field="terminal_fields",
        )
    )
    checksum_ok = upstream.get("reproducibility_checksum") == upstream_fixture.artifact_checksum(
        upstream
    )
    rows.append(
        gate_row(
            "upstream_authentication",
            True,
            checksum_ok,
            checksum_ok,
            upstream="experiment_7222",
            field="reproducibility_checksum",
        )
    )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        manifest = {}
    source_hashes = upstream.get("source_artifact_hashes")
    observed_sidecars = {
        "artifact_public": source_hashes.get("public_view")
        if isinstance(source_hashes, Mapping)
        else None,
        "artifact_authority": source_hashes.get("authority_sidecar")
        if isinstance(source_hashes, Mapping)
        else None,
        "artifact_manifest": source_hashes.get("fixture_manifest")
        if isinstance(source_hashes, Mapping)
        else None,
        "manifest_public": manifest.get("public_view_sha256"),
        "manifest_authority": manifest.get("authority_sidecar_sha256"),
        "manifest_paths": manifest.get("paths"),
        "manifest_counts": manifest.get("counts"),
    }
    expected_sidecars = {
        "artifact_public": PINNED_PUBLIC_SHA256,
        "artifact_authority": PINNED_AUTHORITY_SHA256,
        "artifact_manifest": PINNED_MANIFEST_SHA256,
        "manifest_public": PINNED_PUBLIC_SHA256,
        "manifest_authority": PINNED_AUTHORITY_SHA256,
        "manifest_paths": {
            "public_view": PUBLIC_PATH.as_posix(),
            "authority_sidecar": AUTHORITY_PATH.as_posix(),
            "fixture_manifest": FIXTURE_MANIFEST_PATH.as_posix(),
        },
        "manifest_counts": {
            "public_rows": 320,
            "authority_rows": 320,
            "base_rows": 80,
            "row_receipts": 320,
            "grammar_requests": 640,
            "unique_grammar_requests": 400,
        },
    }
    rows.append(
        gate_row(
            "sidecar_authentication",
            expected_sidecars,
            observed_sidecars,
            observed_sidecars == expected_sidecars,
            upstream="experiment_7222",
            field="source_artifact_hashes_and_manifest",
        )
    )
    historical_path = Path(__file__).resolve().parents[2] / HISTORICAL_EXP7209_PATH
    try:
        historical_bytes = historical_path.read_bytes()
        historical = json.loads(historical_bytes)
        historical_observed = {
            "sha256": sha256_bytes(historical_bytes),
            "status": unwrap_principle(historical.get("status")),
            "model_invoked": unwrap_principle(historical.get("model_invoked")),
            "inference_substrate_class": unwrap_principle(
                historical.get("inference_substrate_class")
            ),
            "used_as_extraction_failure_evidence": False,
        }
    except (OSError, json.JSONDecodeError) as exc:
        historical_observed = {"error": f"{type(exc).__name__}:{exc}"}
    historical_expected = {
        "sha256": PINNED_EXP7209_SHA256,
        "status": "blocked",
        "model_invoked": False,
        "inference_substrate_class": "blocked_no_run",
        "used_as_extraction_failure_evidence": False,
    }
    rows.append(
        gate_row(
            "historical_exp7209_interpretation",
            historical_expected,
            historical_observed,
            historical_observed == historical_expected,
            upstream="experiment_7209",
            field="model_execution_and_interpretation",
        )
    )
    return rows


def _relation_view(relation: Mapping[str, Any]) -> JsonDict:
    """Keep only the tuple fields that the calibration authority defines."""

    return {field: relation.get(field) for field in fixture.RELATION_FIELDS}


def score_offline_arms(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Run syntax and typed semantic checks on the same 16 raw replies."""

    calls = {(str(row["unit_id"]), str(row["call_type"])): row for row in completion_rows}
    scheduled = {(str(row["unit_id"]), str(row["call_type"])): row for row in schedule}
    output: list[JsonDict] = []
    for private in authority_rows:
        unit_id = str(private["unit_id"])
        source = calls.get((unit_id, "source"))
        claim = calls.get((unit_id, "claim"))
        source_call = scheduled.get((unit_id, "source"))
        claim_call = scheduled.get((unit_id, "claim"))
        raw_hashes = [
            source.get("raw_response_sha256") if source else None,
            claim.get("raw_response_sha256") if claim else None,
        ]
        syntax_ok = bool(
            source
            and claim
            and source.get("transport_error") is None
            and claim.get("transport_error") is None
            and source.get("parse_valid") is True
            and claim.get("parse_valid") is True
            and source.get("truncated") is False
            and claim.get("truncated") is False
            and source.get("reasoning_disabled_observed") is True
            and claim.get("reasoning_disabled_observed") is True
        )
        common = {
            "unit_id": unit_id,
            "seed": RANDOM_SEED,
            "raw_call_hashes": raw_hashes,
            "expected_prediction": str(private["expected_decision"]),
            "variant": str(private["variant"]),
            "relation_family": str(private["relation_family"]),
        }
        output.append(
            {
                **common,
                "arm": "syntax_only",
                "metric": int(syntax_ok),
                "error": None if syntax_ok else "source_or_claim_syntax_incomplete",
                "abstention": not syntax_ok,
                "prediction": None,
                "source_relation_agreement": None,
                "claim_relation_agreement": None,
                "executor_invoked": False,
                "executor_errors": [],
            }
        )
        source_compiled = dict(source.get("compiled_completion") or {}) if source else {}
        claim_compiled = dict(claim.get("compiled_completion") or {}) if claim else {}
        expected_source = (
            fixture.extract_public_completion(
                str(source_call["input_text"]).encode("utf-8"), "source"
            )
            if source_call
            else {"relations": []}
        )
        observed_source = [_relation_view(row) for row in source_compiled.get("relations", [])]
        observed_claim = [_relation_view(row) for row in claim_compiled.get("relations", [])]
        expected_claim = dict(private.get("claim_relation") or {})
        source_agreement = observed_source == expected_source.get("relations", [])
        claim_agreement = observed_claim == [expected_claim]
        exact_references = bool(
            source
            and claim
            and source.get("exact_reference_valid") is True
            and claim.get("exact_reference_valid") is True
        )
        executor_invoked = bool(source_call and claim_call and source and claim)
        execution = (
            capture._execute_compiled_pair(
                str(source_call["input_text"]), source_compiled, claim_compiled
            )
            if executor_invoked
            else {"decision": "unknown", "abstention": True, "errors": ["missing_call"]}
        )
        semantic_ok = bool(
            syntax_ok
            and exact_references
            and source_agreement
            and claim_agreement
            and execution["decision"] == private["expected_decision"]
        )
        output.append(
            {
                **common,
                "arm": "reference_type_semantics",
                "metric": int(semantic_ok),
                "error": None if semantic_ok else "source_claim_or_decision_disagreement",
                "abstention": bool(execution["abstention"]),
                "prediction": execution["decision"],
                "source_relation_agreement": source_agreement,
                "claim_relation_agreement": claim_agreement,
                "executor_invoked": executor_invoked,
                "executor_errors": execution["errors"],
            }
        )
    return output


def readiness_receipt(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    authority_leakage_count: int,
    model_identity_errors: Sequence[str],
) -> JsonDict:
    """Apply the one fixed 7-of-8 parse and 6-of-8 semantic gate."""

    calls = {(str(row.get("unit_id")), str(row.get("call_type"))): row for row in completion_rows}
    unit_ids = list(dict.fromkeys(str(row.get("unit_id")) for row in schedule))
    parse_complete = 0
    for unit_id in unit_ids:
        pair = [calls.get((unit_id, call_type)) for call_type in ("source", "claim")]
        if all(
            row
            and row.get("transport_error") is None
            and row.get("terminal_state") == "complete"
            and row.get("parse_valid") is True
            and row.get("truncated") is False
            for row in pair
        ):
            parse_complete += 1
    semantic_rows = [row for row in comparison_rows if row.get("arm") == "reference_type_semantics"]
    semantic_correct = sum(row.get("metric") == 1 for row in semantic_rows)
    scheduled_ids = {str(row.get("call_id")) for row in schedule}
    observed_ids = {str(row.get("call_id")) for row in completion_rows}
    complete = len(schedule) == 16 and len(completion_rows) == 16 and observed_ids == scheduled_ids
    ready = bool(
        complete
        and len(unit_ids) == 8
        and len(semantic_rows) == 8
        and parse_complete >= 7
        and semantic_correct >= 6
        and authority_leakage_count == 0
        and not model_identity_errors
    )
    return {
        "scheduled_call_denominator": 16,
        "observed_calls": len(completion_rows),
        "unit_denominator": 8,
        "parse_complete_units": parse_complete,
        "minimum_parse_complete_units": 7,
        "semantic_correct_units": semantic_correct,
        "minimum_semantic_correct_units": 6,
        "authority_leakage_count": authority_leakage_count,
        "model_identity_errors": list(model_identity_errors),
        "span_canary_complete_score": int(complete),
        "span_canary_ready_score": int(ready),
    }


def _frozen_decoding_contract() -> JsonDict:
    """Return the only decoding settings authorized for this canary."""

    return {
        "model_specs": deepcopy(MODEL_SPECS),
        "context_tokens": CONTEXT_TOKEN_BUDGET,
        "source_tokens": SOURCE_TOKEN_BUDGET,
        "claim_tokens": CLAIM_TOKEN_BUDGET,
        "maximum_tokens_per_call": max(TOKEN_BUDGETS.values()),
        "decoding_parameters": deepcopy(DECODING_PARAMETERS),
        "prompt_templates": deepcopy(PROMPT_TEMPLATES),
        "prompt_template_hashes": {
            key: sha256_bytes(value.encode("utf-8")) for key, value in PROMPT_TEMPLATES.items()
        },
        "capture_grammar_arm": "grammar_only",
        "offline_arms": ["syntax_only", "reference_type_semantics"],
        "separate_source_and_claim_calls": True,
        "call_count": 16,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "request_cap_s": REQUEST_CAP_S,
        "inference_deadline_s": INFERENCE_DEADLINE_S,
        "non_thinking_required": True,
        "scheduled_canary_count": 1,
        "rerun_variants": 0,
        "held_out_run_authorized": False,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete running checkpoint before any gate can fail."""

    return {
        "schema": "carnot.experiment_7223_v636_span_canary.v1",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "preconditions_checked": [],
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_calls": 16,
            "attempted_calls": 0,
            "completed_calls": 0,
            "usable_calls": 0,
            "censored_calls": 16,
            "planned_rows": 16,
            "completed_rows": 0,
            "censored_rows": 16,
            "independent_units": 8,
            "calibration_units": 8,
            "held_out_units_selected": 0,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_exp7223_running_preconditions",
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "span_canary_ready_score": 0,
        "span_canary_complete_score": 0,
        "canary_rows": [],
        "frozen_decoding_contract": _frozen_decoding_contract(),
        "token_budget_receipt": {"measurement_status": "not_measured"},
        "model_identity_receipt": {"identity_errors": ["not_checked"]},
        "gpu_receipts": {},
        "phase_spans": [],
        "runner_receipt": {
            "model_count": 1,
            "runner": "native_llama.cpp_server",
            "dual_gpu_runner_used": False,
        },
        "readiness_receipt": {},
        "selection_receipt": {},
        "schedule": [],
        "raw_manifest": {},
        "historical_exp7209_interpretation": {
            "model_invoked": False,
            "used_as_extraction_failure_evidence": False,
        },
        "scope_statement": "calibration-only bounded canary; no held-out outcome selected or scored",
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish an external block without converting absence into a null result."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(failure)
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_exp7223_{failure['check'] if failure else 'unknown'}"
    artifact["span_canary_ready_score"] = 0
    artifact["duration_s"] = duration_s
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finalize_measured_artifact(
    artifact: JsonDict,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    token_receipt: Mapping[str, Any],
    *,
    duration_s: float,
    live_evidence: bool,
    authority_leakage_count: int,
    model_identity_errors: Sequence[str],
) -> JsonDict:
    """Finish the single scheduled canary as ready or as a measured null."""

    artifact["schedule"] = [deepcopy(dict(row)) for row in schedule]
    artifact["canary_rows"] = [deepcopy(dict(row)) for row in completion_rows]
    artifact["rows"] = [deepcopy(dict(row)) for row in comparison_rows]
    artifact["token_budget_receipt"] = deepcopy(dict(token_receipt))
    budgets_by_call = {str(row["call_id"]): int(row["output_token_budget"]) for row in schedule}
    artifact["token_budget_receipt"].update(
        {
            "observed_call_count": len(completion_rows),
            "generated_token_total": sum(
                int(row.get("completion_tokens", 0) or 0) for row in completion_rows
            ),
            "bounded_stop_counts": dict(
                sorted(Counter(str(row.get("finish_reason")) for row in completion_rows).items())
            ),
            "all_observed_calls_within_budget": all(
                0
                <= int(row.get("completion_tokens", 0) or 0)
                <= budgets_by_call.get(str(row.get("call_id")), -1)
                for row in completion_rows
            ),
        }
    )
    artifact["selection_receipt"].setdefault(
        "authority_fields_in_model_schedule", authority_leakage_count
    )
    receipt = readiness_receipt(
        schedule,
        completion_rows,
        comparison_rows,
        authority_leakage_count=authority_leakage_count,
        model_identity_errors=model_identity_errors,
    )
    artifact["readiness_receipt"] = receipt
    ready = int(receipt["span_canary_ready_score"] == 1 and live_evidence)
    artifact["span_canary_ready_score"] = ready
    artifact["span_canary_complete_score"] = int(receipt["span_canary_complete_score"])
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": len(completion_rows),
            "completed_calls": sum(
                row.get("terminal_state") == "complete" for row in completion_rows
            ),
            "usable_calls": sum(
                row.get("terminal_state") == "complete"
                and row.get("transport_error") is None
                and row.get("parse_valid") is True
                and row.get("truncated") is False
                for row in completion_rows
            ),
            "censored_calls": 16 - len(completion_rows),
            "completed_rows": len(comparison_rows),
            "censored_rows": 16 - len(comparison_rows),
        }
    )
    artifact["status"] = "complete"
    artifact["model_invoked"] = bool(live_evidence)
    if live_evidence:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
        artifact["gpu_receipts"].setdefault("provenance_ok", True)
    artifact["gate_check_summary"] = gate_summary(None)
    artifact["verdict_class"] = "circular_positive" if ready else "null"
    artifact["honest_verdict"] = (
        "complete_circular_positive_span_canary_ready_no_held_out_value_claim"
        if ready
        else "complete_null_span_canary_not_ready_no_held_out_value_claim"
    )
    artifact["duration_s"] = duration_s
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check fixed counts, provenance, verdicts, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("MODEL_SPECS")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("frozen_decoding_contract") != _frozen_decoding_contract():
        errors.append("frozen_decoding_contract")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    budget = value.get("sample_size_budget")
    canary_rows = value.get("canary_rows")
    rows = value.get("rows")
    if not isinstance(canary_rows, list) or not isinstance(rows, list):
        errors.append("rows")
        canary_rows, rows = [], []
    if not isinstance(budget, Mapping):
        errors.append("sample_size_budget")
    else:
        expected_counts = {
            "planned_calls": 16,
            "attempted_calls": len(canary_rows),
            "completed_calls": sum(row.get("terminal_state") == "complete" for row in canary_rows),
            "censored_calls": 16 - len(canary_rows),
            "planned_rows": 16,
            "completed_rows": len(rows),
            "censored_rows": 16 - len(rows),
            "independent_units": 8,
            "calibration_units": 8,
            "held_out_units_selected": 0,
        }
        if "usable_calls" in budget:
            expected_counts["usable_calls"] = sum(
                row.get("terminal_state") == "complete"
                and row.get("transport_error") is None
                and row.get("parse_valid") is True
                and row.get("truncated") is False
                for row in canary_rows
            )
        if any(budget.get(field) != expected for field, expected in expected_counts.items()):
            errors.append("sample_size_budget")
    status = value.get("status")
    if status == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or value.get("span_canary_ready_score") != 0
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
        ):
            errors.append("blocked_terminal_state")
        if (
            value.get("model_invoked") is not True
            and value.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_substrate")
        return list(dict.fromkeys(errors))
    if status != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    schedule = value.get("schedule")
    if not isinstance(schedule, list):
        errors.append("schedule")
        schedule = []
    identity = value.get("model_identity_receipt")
    identity_errors = (
        list(identity.get("identity_errors") or [])
        if isinstance(identity, Mapping)
        else ["missing"]
    )
    selection = value.get("selection_receipt")
    authority_leakage = (
        int(selection.get("authority_fields_in_model_schedule", -1))
        if isinstance(selection, Mapping)
        else -1
    )
    expected_readiness = readiness_receipt(
        schedule,
        canary_rows,
        rows,
        authority_leakage_count=authority_leakage,
        model_identity_errors=identity_errors,
    )
    if value.get("readiness_receipt") != expected_readiness:
        errors.append("readiness_receipt")
    if value.get("span_canary_ready_score") != expected_readiness["span_canary_ready_score"]:
        errors.append("span_canary_ready_score")
    if value.get("span_canary_complete_score") != expected_readiness["span_canary_complete_score"]:
        errors.append("span_canary_complete_score")
    expected_class = (
        "circular_positive" if expected_readiness["span_canary_ready_score"] else "null"
    )
    if value.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    token_receipt = value.get("token_budget_receipt")
    if not isinstance(token_receipt, Mapping) or token_receipt.get("measurement_status") != (
        "measured_embedded_gguf_tokenizer"
    ):
        errors.append("token_budget_receipt")
    if (
        value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
        or not isinstance(value.get("gpu_receipts"), Mapping)
        or value["gpu_receipts"].get("provenance_ok") is not True
    ):
        errors.append("live_inference_provenance")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration < 10.0:
        errors.append("bounded_generation_duration_floor")
    if identity_errors:
        errors.append("model_identity_receipt")
    return list(dict.fromkeys(errors))


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Seal prompts, selection, model identity, and every raw response hash."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema": "carnot.exp7223.raw_manifest.v1",
        "status": "complete" if len(completion_rows) == 16 else "partial",
        "raw_row_count": len(completion_rows),
        "worker_input_paths": [PUBLIC_PATH.as_posix()],
        "authority_path_opened_by_model_worker": False,
        "held_out_outcome_count": 0,
        "schedule": [deepcopy(dict(row)) for row in schedule],
        "schedule_sha256": sha256_json(list(schedule)),
        "selection_receipt": deepcopy(dict(selection)),
        "decoding_contract": _frozen_decoding_contract(),
        "model_identity": deepcopy(dict(model_identity)),
        "raw_rows": [
            {
                "call_order": row.get("call_order"),
                "call_id": row.get("call_id"),
                "unit_id": row.get("unit_id"),
                "call_type": row.get("call_type"),
                "raw_completion_sha256": row.get("raw_completion_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "grammar_sha256": row.get("grammar_sha256"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "terminal_state": row.get("terminal_state"),
                "row_sha256": sha256_json(row),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(
        raw_dir / "schedule.json",
        {"schedule": list(schedule)},
        allow_override=False,
        sort_keys=True,
    )
    atomic_write_json(
        raw_dir / "selection.json", dict(selection), allow_override=False, sort_keys=True
    )
    atomic_write_json(
        raw_dir / "frozen_contract.json",
        {"contract": _frozen_decoding_contract(), "model_identity": dict(model_identity)},
        allow_override=False,
        sort_keys=True,
    )
    atomic_write_json(raw_dir / "raw_manifest.json", frozen, allow_override=False, sort_keys=True)
    return frozen


def _source_hashes(root: Path) -> JsonDict:
    """Bind each code, contract, producer, and evidence file used here."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "research_references": Path("research-references.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "constraint_spec": SPEC_PATH,
        "upstream_artifact": UPSTREAM_PATH,
        "public_view": PUBLIC_PATH,
        "authority_sidecar": AUTHORITY_PATH,
        "fixture_manifest": FIXTURE_MANIFEST_PATH,
        "historical_exp7209": HISTORICAL_EXP7209_PATH,
        "v635_span_capture": Path("python/carnot/experiment_7209_v635_span_canary.py"),
        "v636_span_fixture": Path("python/carnot/experiment_7222_v636_span_fixture.py"),
        "sota_models": Path("python/carnot/inference/sota_models.py"),
        "llama_server_supervisor": Path("python/carnot/inference/llama_server_supervisor.py"),
        "gpu_lease_journal": Path("python/carnot/gpu_lease_phase_journal.py"),
        "module": MODULE_PATH,
        "entrypoint": WRAPPER_PATH,
        "focused_tests": TEST_PATH,
    }
    return {
        name: sha256_file(root / path) if (root / path).is_file() else "missing"
        for name, path in paths.items()
    }


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Flush actual phase and progress observations for external monitoring."""

    print(
        "exp7223 " + canonical_json({"phase": phase, "event": event, **fields}),
        flush=True,
    )


def _collect_preflight(
    root: Path, run_date: str, result_path: Path, checkpoint_dir: Path, raw_dir: Path
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Use the shipped resource checks with the authenticated V636 inputs."""

    checks, public_rows, authority_rows, context = capture._collect_preflight(
        root,
        run_date,
        result_path,
        checkpoint_dir,
        raw_dir,
        contract={
            "upstream_path": UPSTREAM_PATH,
            "public_path": PUBLIC_PATH,
            "authority_path": AUTHORITY_PATH,
            "manifest_path": FIXTURE_MANIFEST_PATH,
            "module_path": MODULE_PATH,
            "wrapper_path": WRAPPER_PATH,
            "test_path": TEST_PATH,
            "spec_req": "REQ-VERIFY-7223",
            "expected_calls": 16,
            "expected_units": 8,
            "task_id": TASK_ID,
            "upstream_id": "experiment_7222",
            "split_id": "experiment_7222_canary_split",
            "upstream_gate_rows": upstream_gate_rows,
            "load_split": load_calibration_selection,
            "build_schedule": build_schedule,
            "schedule_errors": schedule_errors,
            "request_cap_s": REQUEST_CAP_S,
            "live_window_cap_s": INFERENCE_DEADLINE_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
        },
    )
    if context.get("schedule"):
        context["selection_receipt"] = selection_receipt(
            public_rows, authority_rows, context["schedule"]
        )
    return checks, public_rows, authority_rows, context


def _identity_errors(identity: Mapping[str, Any], gpu_receipts: Mapping[str, Any]) -> list[str]:
    """Name unresolved model or CUDA identity fields without repairing them."""

    errors = []
    if identity.get("hf_id") != QWEN_MODEL_ID:
        errors.append("hf_id")
    if identity.get("quantization") != QUANTIZATION:
        errors.append("quantization")
    if not identity.get("revision") or not identity.get("gguf_sha256"):
        errors.append("gguf_revision_or_hash")
    if identity.get("embedded_chat_template_present") is not True:
        errors.append("embedded_chat_template")
    if gpu_receipts.get("provenance_ok") is not True:
        errors.append("actual_cuda_execution")
    return errors


def _checkpoint(path: Path, artifact: Mapping[str, Any], started: float) -> None:
    """Write provisional state only below the checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _terminal(
    artifact: JsonDict, result_path: Path, checkpoint_path: Path, started: float
) -> JsonDict:
    """Cold-check once and atomically publish one terminal artifact."""

    _progress(9, "validation_start", path=str(result_path))
    validation_started = time.monotonic()
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    preliminary = validate_artifact(artifact)
    artifact["phase_spans"].append(
        {
            "phase": 9,
            "name": "cold_artifact_verification",
            "duration_s": time.monotonic() - validation_started,
        }
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    _progress(9, "validation_end", preliminary_errors=preliminary, errors=errors)
    if errors:
        raise ValueError(f"invalid Exp7223 artifact: {errors}")
    _progress(10, "write_start", path=str(result_path))
    _checkpoint(checkpoint_path, artifact, started)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(10, "write_end", path=str(result_path))
    return artifact


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
) -> JsonDict:
    """Run the one finite canary or preserve an exact external block."""

    _progress(0, "start", detail="print before checking any prerequisite")
    started = time.monotonic()
    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    result_path = destination / RESULT_PATH
    checkpoint_dir = destination / CHECKPOINT_DIR
    raw_dir = destination / RAW_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "terminal_candidate.json"
    artifact = base_artifact(run_date)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase_started = time.monotonic()
    _progress(1, "start", operation="preconditions_and_exp7222_authentication")
    checks, public_rows, authority_rows, context = _collect_preflight(
        repo, run_date, result_path, checkpoint_dir, raw_dir
    )
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "preconditions_and_exp7222_authentication",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(1, "end", passed=failure is None)
    if failure is not None:
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)

    schedule = list(context["schedule"])
    selection = dict(context["selection_receipt"])
    artifact["schedule"] = deepcopy(schedule)
    artifact["selection_receipt"] = deepcopy(selection)
    _progress(2, "start", operation="freeze_selection_schedule_and_contract")
    atomic_write_json(raw_dir / "selection.json", selection, allow_override=False, sort_keys=True)
    atomic_write_json(
        raw_dir / "schedule.json", {"schedule": schedule}, allow_override=False, sort_keys=True
    )
    _progress(2, "end", selected_units=8, scheduled_calls=16)

    _progress(3, "model_load_start", operation="embedded_tokenizer_vocab_only")
    token_started = time.monotonic()
    owner, tokenizer, tokenizer_load = capture._load_embedded_tokenizer(
        Path(context["model_path"]), context["tokenizer_loader"]
    )
    _progress(
        3,
        "model_load_end",
        operation="embedded_tokenizer_vocab_only",
        available=tokenizer_load["embedded_tokenizer_available"],
    )
    if tokenizer is None:
        check = gate_row(
            "embedded_tokenizer_load",
            True,
            False,
            False,
            upstream="cached_qwen_gguf",
            field="embedded_tokenizer",
        )
        checks.append(check)
        artifact["phase_spans"].append(
            {
                "phase": 3,
                "name": "embedded_tokenizer_load",
                "duration_s": time.monotonic() - token_started,
            }
        )
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)
    token_receipt = measure_token_budgets(schedule, tokenizer)
    token_receipt["tokenizer_load_receipt"] = tokenizer_load
    owner.close()
    artifact["token_budget_receipt"] = token_receipt
    artifact["phase_spans"].append(
        {
            "phase": 3,
            "name": "embedded_tokenizer_load",
            "duration_s": time.monotonic() - token_started,
        }
    )
    if token_receipt.get("all_forms_fit_with_20_percent_headroom") is not True:
        check = gate_row(
            "representation_size_fit",
            True,
            False,
            False,
            upstream="embedded_qwen_tokenizer",
            field="token_budget_receipt",
        )
        checks.append(check)
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        return _terminal(artifact, result_path, checkpoint_path, started)

    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    artifact["model_identity_receipt"].update(
        {
            "serving_path": str(context["server_path"]),
            "tokenizer_path": str(context["model_path"]),
            "tokenizer_load_receipt": tokenizer_load,
            "identity_errors": [],
        }
    )
    _checkpoint(checkpoint_path, artifact, started)

    _progress(4, "benchmark_start", operation="single_scheduled_16_call_canary")
    capture_result = capture._live_capture(
        context, checkpoint_dir, raw_dir, artifact["phase_spans"]
    )
    _progress(
        4,
        "benchmark_end",
        completed_calls=len(capture_result["rows"]),
        runtime_error=capture_result["runtime_error"],
    )
    completion_rows = list(capture_result["rows"])
    _progress(7, "benchmark_start", operation="offline_syntax_reference_type_semantics")
    score_started = time.monotonic()
    comparison_rows = score_offline_arms(schedule, completion_rows, authority_rows)
    artifact["phase_spans"].append(
        {
            "phase": 7,
            "name": "offline_syntax_reference_type_semantics",
            "duration_s": time.monotonic() - score_started,
        }
    )
    _progress(7, "benchmark_end", completed_rows=len(comparison_rows))
    artifact["gpu_receipts"] = deepcopy(capture_result["gpu_receipts"])
    artifact["runner_receipt"] = deepcopy(capture_result["runner_receipt"])
    artifact["runner_receipt"].update(
        {
            "model_count": 1,
            "dual_gpu_runner_used": False,
            "pid_identity": deepcopy(capture_result["gpu_receipts"].get("server_identity", {})),
            "decoding_parameters": deepcopy(DECODING_PARAMETERS),
            "cleanup_ok": capture_result["gpu_receipts"].get("cleanup", {}).get("leak_free")
            is True,
        }
    )
    artifact["model_invoked"] = bool(capture_result["model_invoked"])
    artifact["model_identity_receipt"]["actual_cuda_execution"] = capture_result[
        "gpu_receipts"
    ].get("provenance_ok")
    identity_errors = _identity_errors(
        artifact["model_identity_receipt"], capture_result["gpu_receipts"]
    )
    artifact["model_identity_receipt"]["identity_errors"] = identity_errors

    manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, selection, artifact["model_identity_receipt"]
    )
    artifact["raw_manifest"] = manifest
    artifact["source_artifact_hashes"]["raw_manifest"] = sha256_file(raw_dir / "raw_manifest.json")
    runtime_failed = bool(
        capture_result["runtime_error"]
        or len(completion_rows) != 16
        or capture_result["gpu_receipts"].get("provenance_ok") is not True
    )
    if runtime_failed:
        check = gate_row(
            "live_runtime_completion_and_cuda_provenance",
            {"calls": 16, "runtime_error": None, "provenance_ok": True},
            {
                "calls": len(completion_rows),
                "runtime_error": capture_result["runtime_error"],
                "provenance_ok": capture_result["gpu_receipts"].get("provenance_ok"),
            },
            False,
            upstream="owned_native_llama_server",
            field="live_capture",
        )
        checks.append(check)
        finalize_measured_artifact(
            artifact,
            schedule,
            completion_rows,
            comparison_rows,
            token_receipt,
            duration_s=time.monotonic() - started,
            live_evidence=bool(completion_rows),
            authority_leakage_count=int(selection["authority_fields_in_model_schedule"]),
            model_identity_errors=identity_errors,
        )
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
    else:
        finalize_measured_artifact(
            artifact,
            schedule,
            completion_rows,
            comparison_rows,
            token_receipt,
            duration_s=time.monotonic() - started,
            live_evidence=True,
            authority_leakage_count=int(selection["authority_fields_in_model_schedule"]),
            model_identity_errors=identity_errors,
        )
    return _terminal(artifact, result_path, checkpoint_path, started)


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the experiment contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canary and accept any cold-valid terminal result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7223] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7223] terminal verdict={artifact['honest_verdict']} "
        f"ready={artifact['span_canary_ready_score']} "
        f"complete={artifact['span_canary_complete_score']}",
        flush=True,
    )
    return 0
