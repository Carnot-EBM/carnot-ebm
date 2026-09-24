"""Measure paired explicit-schema transport with and without grammar.

Both arms use the lossless input and schema authority frozen by Experiment
7616. This module measures syntax transport and runtime cost only. It does not
claim that a syntactically valid evidence pointer is factually correct.

Spec: REQ-REPORT-7617 and SCENARIO-REPORT-7617-*.
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
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

from carnot import experiment_7604_v664_evidence_pilot as v664
from carnot import experiment_7616_v665_evidence_schema as schema_protocol
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7617-v665-schema-pilot"
SCHEMA = "carnot.exp7617.v665.schema_pilot.v1"
RESULT_PATH = Path("results/experiment_7617_v665_schema_pilot.json")
RAW_DIR = Path("results/raw/experiment_7617_v665_schema_pilot")
MODULE_PATH = Path("python/carnot/experiment_7617_v665_schema_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7617_v665_schema_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7617_v665_schema_pilot.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
SCHEMA_RESULT = Path("results/experiment_7616_v665_evidence_schema.json")
V664_RESULT = Path("results/experiment_7604_v664_evidence_pilot.json")
PILOT_INPUT = Path(
    "results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl"
)
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
CONTROL_ARM = "schema_prompt_control"
GRAMMAR_ARM = "schema_prompt_with_grammar"
ARMS = (CONTROL_ARM, GRAMMAR_ARM)
RANDOM_SEED = 7_617_001
ARM_ORDER_SEED = 7_617_002
MAX_TOKENS = 512
VALID_THRESHOLD = 6
GENERATION_HARD_STOP_S = 3000.0
TOTAL_TASK_LIMIT_S = 4500.0
VALIDATION_RESERVE_S = 300.0
GPU_REQUIRED_FREE_MB = 20_000
TERMINAL_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so every reduction can be reproduced byte-for-byte."""

    return current_work_receipt.canonical_hash(value)


def sha256_file(path: Path) -> str:  # pragma: no cover - shared file helper.
    """Hash exact file bytes with the repository's common prefix."""

    return current_work_receipt.sha256_file(path)


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print each boundary immediately so a long local run stays observable."""

    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_ID,
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **details,
            },
            default=str,
            sort_keys=True,
        ),
        flush=True,
    )


def build_arm_request(record: Mapping[str, Any], arm: str) -> JsonDict:
    """Build paired requests whose only arm difference is installed grammar."""

    if arm not in ARMS:
        raise ValueError("pilot_arm_invalid")
    request = schema_protocol.build_extraction_request(record)
    request.pop("response_format", None)
    request["seed"] = RANDOM_SEED
    if arm == GRAMMAR_ARM:
        request["grammar"] = schema_protocol.compile_decoder_grammar(
            schema_protocol.build_schema_authority(record)
        )
    return request


def seeded_paired_schedule(
    records: Sequence[Mapping[str, Any]], *, seed: int = ARM_ORDER_SEED
) -> list[JsonDict]:
    """Freeze one deterministic within-group arm order for all eight groups."""

    if len(records) != 8 or len({str(row.get("component_hash")) for row in records}) != 8:
        raise ValueError("exactly_eight_disjoint_pilot_groups_required")
    rng = random.Random(seed)
    schedule: list[JsonDict] = []
    for group_index, record in enumerate(records, 1):
        order = list(ARMS)
        rng.shuffle(order)
        for pair_position, arm in enumerate(order, 1):
            schedule.append(
                {
                    "group_index": group_index,
                    "pair_position": pair_position,
                    "component_hash": str(record["component_hash"]),
                    "arm": arm,
                    "seed": seed,
                }
            )
    return schedule


def parse_pilot_response(
    record: Mapping[str, Any], response_text: str, *, finish_reason: str
) -> JsonDict:
    """Parse saved response text without trusting decoder-side acceptance."""

    outcome = schema_protocol.validate_evidence_output(
        record, response_text, finish_reason=finish_reason
    )
    error = outcome.get("error")
    invalid_id = error in {"response_sentence_id_invalid", "source_sentence_id_invalid"}
    evidence = outcome.get("evidence") if isinstance(outcome.get("evidence"), list) else []
    unknown_count = sum(
        isinstance(row, Mapping) and row.get("relation") == "unknown" for row in evidence
    )
    valid_completed = bool(outcome.get("accepted") is True and finish_reason != "length")
    return {
        "accepted": outcome.get("accepted") is True,
        "valid_completed": valid_completed,
        "parser_error": error,
        "invalid_id_reference": invalid_id,
        "evidence": deepcopy(evidence),
        "unknown_numerator": unknown_count,
        "unknown_denominator": len(evidence),
        "unknown_fraction": unknown_count / len(evidence) if evidence else 0.0,
        "syntax_coverage_only": True,
        "semantic_correctness_measured": False,
    }


def _arm_reduction(rows: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
    arm_rows = [row for row in rows if row.get("arm") == arm]
    components = {str(row.get("component_hash") or "") for row in arm_rows}
    if len(arm_rows) != 8 or len(components) != 8 or "" in components:
        raise ValueError(f"paired_arm_groups_invalid:{arm}")
    valid = sum(
        row.get("status") == "completed"
        and row.get("transport_completed") is True
        and isinstance(row.get("parser_result"), Mapping)
        and row["parser_result"].get("valid_completed") is True
        for row in arm_rows
    )
    accepted_bad_ids = sum(
        isinstance(row.get("parser_result"), Mapping)
        and row["parser_result"].get("accepted") is True
        and row["parser_result"].get("invalid_id_reference") is True
        for row in arm_rows
    )
    passed = valid >= VALID_THRESHOLD and accepted_bad_ids == 0
    return {
        "arm": arm,
        "independent_groups": 8,
        "valid_completed_count": valid,
        "valid_completed_threshold": VALID_THRESHOLD,
        "accepted_invalid_id_reference_count": accepted_bad_ids,
        "passed": passed,
    }


def select_configuration(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prefer grammar, then control, only under the frozen six-of-eight rule."""

    if len(rows) != 16:
        raise ValueError("exactly_sixteen_paired_rows_required")
    grammar = _arm_reduction(rows, GRAMMAR_ARM)
    control = _arm_reduction(rows, CONTROL_ARM)
    selected = GRAMMAR_ARM if grammar["passed"] else CONTROL_ARM if control["passed"] else None
    return {
        "selection_order": [GRAMMAR_ARM, CONTROL_ARM],
        "arm_reductions": {GRAMMAR_ARM: grammar, CONTROL_ARM: control},
        "selected_arm": selected,
        "scale_decision": "selected_for_scaling" if selected else "stop_scaling",
        "evidence_transport_ready_score": int(selected is not None),
        "semantic_benefit_claim": False,
    }


def _nearest_rank_p90(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("completed_generation_timing_absent")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def project_fixed_rosters(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_load_s: float,
    validation_reserve_s: float = VALIDATION_RESERVE_S,
) -> dict[str, JsonDict]:
    """Project three fixed rosters from load plus 1.25 times row p90."""

    completed = [
        float(row["generation_s"])
        for row in rows
        if row.get("status") == "completed"
        and isinstance(row.get("generation_s"), (int, float))
        and not isinstance(row.get("generation_s"), bool)
        and float(row["generation_s"]) >= 0
    ]
    p90 = _nearest_rank_p90(completed)
    projections: dict[str, JsonDict] = {}
    for name, count in (("fit_tune_policy", 120), ("online", 80), ("evaluation", 40)):
        generation = 1.25 * p90 * count
        total = float(model_load_s) + generation + float(validation_reserve_s)
        feasible = generation <= GENERATION_HARD_STOP_S and total <= TOTAL_TASK_LIMIT_S
        projections[name] = {
            "roster_size": count,
            "roster_resized": False,
            "per_row_p90_s": p90,
            "uncertainty_multiplier": 1.25,
            "model_load_s": float(model_load_s),
            "validation_reserve_s": float(validation_reserve_s),
            "projected_generation_s": generation,
            "generation_limit_s": GENERATION_HARD_STOP_S,
            "projected_total_task_s": total,
            "total_task_limit_s": TOTAL_TASK_LIMIT_S,
            "feasible_score": int(feasible),
            "method": "measured_load_plus_1.25_times_per_row_nearest_rank_p90",
        }
    return projections


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind the artifact while excluding its self-referential checksum."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def gate_row(
    check: str,
    *,
    category: str,
    upstream: str,
    path: Path,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    condition: str,
    principle: str,
) -> JsonDict:
    """Keep all operands needed to diagnose one gate."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "condition": condition,
        "principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failure name and the first complete diagnostic row."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def field_principles() -> dict[str, str]:
    """State how each governed artifact field must be interpreted."""

    return {
        "honest_verdict": "A complete_ prefix reports terminal execution, not scientific benefit.",
        "verdict_class": "Only positive, circular_positive, null, blocked, disqualified, or partial is valid.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence opens no downstream gate.",
        "gate_check_summary": "Every failure keeps its upstream, path, field, operator, expected, and observed operands.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness remain separate.",
        "rows": "Each group-arm unit keeps absolute counts, seed, direction, censoring, and raw provenance.",
        "paired_pilot_rows": "Eight groups produce exactly two rows; retries and replays never add samples.",
        "sample_size_budget": "Intended, observed, excluded, and censored counts use independent groups.",
        "preconditions_checked": "Only checks actually run before inference can authenticate prerequisites.",
        "inference_substrate": "Describe current execution, never historical GPU work.",
        "inference_substrate_class": "Bounded generation needs at least ten measured seconds without padding.",
        "MODEL_SPECS": "Every current LLM call uses unsloth/Qwen3.8-27B-GGUF.",
        "model_invoked": "True means a current load or generation was attempted.",
        "execution_venue": "Name the current host and physical GPU UUID.",
        "phase_spans": "Stages are disjoint and keep planned, completed, pending, and checkpoint state.",
        "invocation_counts": "Loads, forwards, generations, and tokens count only current work.",
        "duration_s": "Use current monotonic elapsed time and never inherit or pad runtime.",
        "random_seed": "Persist each stochastic seed and its single purpose.",
        "reproducibility_checksum": "Bind immutable inputs, configuration, raw rows, and reductions.",
        "source_artifact_hashes": "Distinguish producers, pre-gate records, model bytes, and current raw evidence.",
        "validation_receipts": "Record actual commands, exits, worktrees, log hashes, and final readers.",
        "verifier_is_oracle": "Syntax validation does not establish oracle-distinct semantic advantage.",
        "field_principles": "Every governed field carries a nearby interpretation rule.",
        "evidence_transport_ready_score": "One requires a selected arm under the frozen six-of-eight rule.",
        "fit_capture_feasible_score": "One requires the fixed 120-row roster within both measured limits.",
        "online_capture_feasible_score": "One requires the fixed 80-row roster within both measured limits.",
        "evaluation_capture_feasible_score": "One requires the fixed 40-row roster within both measured limits.",
        "selected_config_path": "Bind the immutable prompt, decoder, backend, parser, and model configuration.",
        "semantic_benefit_claim": "Syntax coverage alone cannot claim factual correctness or evidence utility.",
    }


def _complete_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    principles = field_principles()
    for key in artifact:
        principles.setdefault(
            key,
            "Interpret this field only within the authenticated current-run scope and its gates.",
        )
    principles.setdefault(
        "reproducibility_checksum",
        "Bind immutable inputs, configuration, raw rows, and reductions.",
    )
    return principles


def _acceptance_gates(
    *, valid: bool, ready: bool, retained: bool, result_path: Path
) -> list[JsonDict]:
    return [
        gate_row(
            "authenticated_current_execution",
            category="validity",
            upstream=EXPERIMENT_ID,
            path=result_path,
            field="preconditions_and_validation",
            operator="all_true",
            expected=True,
            observed=valid,
            passed=valid,
            condition="all preconditions and affected validation pass",
            principle="Only authenticated current bytes and checks support a result.",
        ),
        gate_row(
            "independent_parse_transport_ready",
            category="readiness",
            upstream=EXPERIMENT_ID,
            path=result_path,
            field="evidence_transport_ready_score",
            operator="eq",
            expected=1,
            observed=int(ready),
            passed=ready,
            condition="one arm has at least six valid completed groups and zero accepted invalid IDs",
            principle="Decoder acceptance never replaces independent parsing.",
        ),
        gate_row(
            "semantic_benefit",
            category="benefit",
            upstream=EXPERIMENT_ID,
            path=result_path,
            field="semantic_benefit_claim",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            condition="fresh labeled evidence shows semantic correctness and downstream utility",
            principle="Syntax coverage is not factual correctness or learned benefit.",
        ),
        gate_row(
            "paired_row_retention",
            category="retention",
            upstream=EXPERIMENT_ID,
            path=result_path,
            field="paired_pilot_rows",
            operator="len_eq",
            expected=16,
            observed=16 if retained else None,
            passed=retained,
            condition="all 16 requested group-arm outcomes remain present",
            principle="Failures stay in the denominator and are never retried or filtered.",
        ),
        gate_row(
            "fresh_confirmatory_claim",
            category="freshness",
            upstream="exp7616-v665-evidence-schema",
            path=result_path,
            field="fresh_confirmatory_claim_allowed",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            condition="evaluated groups have no historical exposure",
            principle="The frozen pilots are protocol evidence, not fresh confirmation.",
        ),
    ]


def _invocation_counts(rows: Sequence[Mapping[str, Any]], *, loads: int = 1) -> JsonDict:
    attempted = len(rows)
    completed = sum(row.get("transport_completed") is True for row in rows)
    return {
        "model_loads_attempted": loads,
        "model_loads_completed": loads,
        "model_loads_failed": 0,
        "forward_calls_attempted": attempted,
        "forward_calls_completed": completed,
        "forward_calls_failed": attempted - completed,
        "generation_calls_attempted": attempted,
        "generation_calls_completed": completed,
        "generation_calls_failed": attempted - completed,
        "input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
    }


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    selected_config_path: Mapping[str, Any] | str | None,
    duration_s: float,
    current_receipt: Mapping[str, Any] | None = None,
    e2e_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal pilot record without claiming semantic benefit."""

    paired = [deepcopy(dict(row)) for row in rows]
    selection = select_configuration(paired)
    selected_rows = [row for row in paired if row.get("arm") == selection["selected_arm"]]
    timing_rows = selected_rows or paired
    projections = project_fixed_rosters(
        timing_rows,
        model_load_s=float(runtime.get("model_load_s") or 0.0),
        validation_reserve_s=float(runtime.get("validation_reserve_s") or VALIDATION_RESERVE_S),
    )
    valid = all(row.get("passed") is True for row in preconditions) and all(
        row.get("passed") is True for row in validation_receipts
    )
    retained = len(paired) == 16
    ready = selection["evidence_transport_ready_score"] == 1
    gates = _acceptance_gates(
        valid=valid, ready=ready, retained=retained, result_path=ROOT / RESULT_PATH
    )
    censored_groups = {
        str(row.get("component_hash"))
        for row in paired
        if row.get("censored") is True
        or row.get("status") != "completed"
        or not isinstance(row.get("parser_result"), Mapping)
        or row["parser_result"].get("valid_completed") is not True
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7617,
        "title": "V665 explicit shared-schema paired transport pilot",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": "complete_null_schema_transport_measured",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": gate_summary(gates),
        "acceptance_gate_results": gates,
        "rows": paired,
        "paired_pilot_rows": paired,
        "sample_size_budget": {
            "independent_unit": "frozen_pilot_group",
            "intended": 8,
            "observed": len({str(row.get("component_hash")) for row in paired}),
            "excluded": 0,
            "censored": len(censored_groups),
            "excluded_from_all_measured_roles": 8,
            "seeds_views_replays_multiply_independent_samples": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": "owned_local_llama_cpp_paired_bounded_generation",
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": "model_bounded_generation",
        "inference_substrate_plausibility_floor_s": 10.0,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [deepcopy(dict(runtime.get("model_spec") or {}))],
        "model_invoked": True,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "gpu_uuid": runtime.get("gpu_uuid"),
            "physical_device": runtime.get("gpu_uuid"),
            "gpu_index": runtime.get("gpu_index"),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": deepcopy(
            dict((current_receipt or {}).get("invocation_counts") or _invocation_counts(paired))
        ),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds": [
            {"seed": RANDOM_SEED, "purpose": "deterministic_model_generation"},
            {"seed": ARM_ORDER_SEED, "purpose": "within_group_paired_arm_order"},
        ],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": [],
        "current_work_receipt": deepcopy(dict(current_receipt or {})),
        "verifier_is_oracle": False,
        "field_principles": field_principles(),
        "evidence_transport_ready_score": selection["evidence_transport_ready_score"],
        "fit_capture_feasible_score": projections["fit_tune_policy"]["feasible_score"],
        "online_capture_feasible_score": projections["online"]["feasible_score"],
        "evaluation_capture_feasible_score": projections["evaluation"]["feasible_score"],
        "selected_config_path": deepcopy(selected_config_path),
        "selection_reduction": selection,
        "capture_projections": projections,
        "inference_runtime_receipt": deepcopy(dict(runtime)),
        "task_e2e_result": deepcopy(dict(e2e_result or {})),
        "semantic_benefit_claim": False,
        "syntax_coverage_measured": True,
        "semantic_correctness_measured": False,
        "downstream_evidence_utility_measured": False,
        "fresh_confirmatory_claim_allowed": False,
        "scope_retirement": {
            "explicit_schema_transport_failure_retired": ready,
            "scientific_hypothesis_retired": False,
            "principle": "Transport readiness can retire only its exact protocol failure.",
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "default_promotion_authorized": False,
        "generator_weights_immutable": True,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
    }
    artifact["field_principles"] = _complete_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a complete in-memory record for changed-behavior tests."""

    checks = [{"passed": True, "check": "test_fixture"}]
    return build_artifact(
        rows=rows,
        runtime={
            "model_load_s": 10.0,
            "validation_reserve_s": 300.0,
            "model_spec": {"hf_id": MODEL_ID, "quantization": "Q4_K_M"},
            "gpu_uuid": "GPU-test",
            "gpu_index": 0,
        },
        preconditions=checks,
        source_hashes=[],
        validation_receipts=checks,
        phase_spans=[],
        selected_config_path="test://selected-config",
        duration_s=10.0,
    )


def validate_artifact(  # pragma: no cover - fresh-process defensive reader.
    value: object, *, root: Path = ROOT, require_terminal: bool = False
) -> list[str]:
    """Cold-check paired rows, reductions, projections, and terminal custody."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_identity_mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_required")
    if artifact.get("verdict_class") not in TERMINAL_CLASSES:
        errors.append("verdict_class_invalid")
    if artifact.get("planned_inference_substrate_class") != "model_bounded_generation":
        errors.append("planned_substrate_invalid")
    if artifact.get("semantic_benefit_claim") is not False:
        errors.append("semantic_benefit_claim_invalid")
    if artifact.get("flagged_adversarial") not in {True, False}:
        errors.append("flagged_adversarial_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not principles.get(key) for key in artifact if key != "reproducibility_checksum"
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
            errors.append("blocked_model_identity_invalid")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(first, Mapping) or required - set(first):
            errors.append("blocked_gate_summary_incomplete")
        return list(dict.fromkeys(errors))
    if artifact.get("MODEL_SPECS") != MODEL_SPECS or artifact.get("model_invoked") is not True:
        errors.append("current_model_identity_invalid")
    if artifact.get("inference_substrate_class") != "model_bounded_generation":
        errors.append("actual_substrate_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 10.0:
        errors.append("bounded_generation_duration_implausible")
    rows = artifact.get("paired_pilot_rows")
    if not isinstance(rows, list) or len(rows) != 16 or artifact.get("rows") != rows:
        errors.append("paired_rows_invalid")
        rows = []
    if rows:
        try:
            selection = select_configuration(rows)
        except (TypeError, ValueError) as exc:
            errors.append(f"selection_invalid:{exc}")
        else:
            if artifact.get("selection_reduction") != selection:
                errors.append("selection_reduction_mismatch")
            if (
                artifact.get("evidence_transport_ready_score")
                != selection["evidence_transport_ready_score"]
            ):
                errors.append("transport_ready_score_mismatch")
            selected = [row for row in rows if row.get("arm") == selection["selected_arm"]]
            timing = selected or rows
            runtime = artifact.get("inference_runtime_receipt") or {}
            expected_projections = project_fixed_rosters(
                timing,
                model_load_s=float(runtime.get("model_load_s") or 0),
                validation_reserve_s=float(
                    runtime.get("validation_reserve_s") or VALIDATION_RESERVE_S
                ),
            )
            if artifact.get("capture_projections") != expected_projections:
                errors.append("capture_projection_mismatch")
            for name, score in (
                ("fit_tune_policy", "fit_capture_feasible_score"),
                ("online", "online_capture_feasible_score"),
                ("evaluation", "evaluation_capture_feasible_score"),
            ):
                if artifact.get(score) != expected_projections[name]["feasible_score"]:
                    errors.append(f"{score}_mismatch")
    budget = artifact.get("sample_size_budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("intended") != 8
        or budget.get("observed") != 8
        or budget.get("excluded_from_all_measured_roles") != 8
    ):
        errors.append("sample_size_budget_invalid")
    gates = artifact.get("acceptance_gate_results") or []
    categories = {row.get("category") for row in gates if isinstance(row, Mapping)}
    if categories != {"validity", "readiness", "benefit", "retention", "freshness"}:
        errors.append("acceptance_gate_shape_invalid")
    elif next(row for row in gates if row["category"] == "benefit")["passed"] is not False:
        errors.append("benefit_gate_invalid")
    first = (artifact.get("gate_check_summary") or {}).get("first_failure")
    diagnostic = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
    if not isinstance(first, Mapping) or diagnostic - set(first):
        errors.append("gate_summary_incomplete")
    if require_terminal:
        names = {
            str(row.get("name"))
            for row in artifact.get("validation_receipts") or []
            if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
        }
        required = {
            *validation_scope.REQUIRED_CHECK_NAMES,
            *TERMINAL_CHECK_NAMES,
            "fresh_process_selected_request",
        }
        if not required <= names:
            errors.append("terminal_validation_incomplete")
    receipt = artifact.get("current_work_receipt")
    if isinstance(receipt, Mapping) and receipt:
        errors.extend(
            f"current_work_receipt:{error}"
            for error in current_work_receipt.validate_current_work_receipt(receipt, root=root)
        )
    return list(dict.fromkeys(errors))


def load_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    """Read one JSON object and reject other top-level types."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - filesystem boundary.
    """Read non-empty JSONL rows without tolerating scalar records."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_object_required:{path}")
        rows.append(value)
    return rows


def _source_row(  # pragma: no cover - filesystem receipt boundary.
    path: Path, *, producer: str, source_class: str
) -> JsonDict:
    return {
        "producer": producer,
        "source_class": source_class,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _append_check(  # pragma: no cover - precondition integration.
    checks: list[JsonDict],
    check: str,
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
            check,
            category=category,
            upstream=upstream,
            path=path,
            field=field,
            operator=operator,
            expected=expected,
            observed=observed,
            passed=passed,
            condition=f"{field} {operator} expected value",
            principle="Unavailable or changed input bytes block current evidence.",
        )
    )


def _tool_versions(root: Path) -> JsonDict:  # pragma: no cover - installed tool boundary.
    commands = {
        "python": [str(root / ".venv/bin/python"), "--version"],
        "pytest": [str(root / ".venv/bin/pytest"), "--version"],
        "ruff": [str(root / ".venv/bin/ruff"), "--version"],
        "mypy": [str(root / ".venv/bin/mypy"), "--version"],
    }
    return {
        name: subprocess.check_output(argv, cwd=root, text=True, timeout=30).strip()
        for name, argv in commands.items()
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover - filesystem custody.
    """Authenticate named inputs, V665 authority, historical identity, and tools."""

    resolved = root.resolve()
    checks: list[JsonDict] = []
    hashes: list[JsonDict] = []
    context: JsonDict = {}
    named_inputs = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7604_v664_evidence_pilot.py"),
        SCHEMA_RESULT,
        V664_RESULT,
        SPEC_PATH,
        PILOT_INPUT,
        Path("results/raw/experiment_7616_v665_evidence_schema/schema_authority.json"),
        Path("results/raw/experiment_7616_v665_evidence_schema/capture_configuration.json"),
    )
    for relative in named_inputs:
        path = resolved / relative
        present = path.is_file() and path.stat().st_size > 0
        _append_check(
            checks,
            f"named_input:{relative.name}",
            upstream="declared_task_input",
            path=path,
            field="readable_nonempty_file",
            operator="eq",
            expected=True,
            observed=present,
            passed=present,
        )
        if present:
            hashes.append(
                _source_row(path, producer="declared_task_input", source_class="pre_gate_input")
            )
    if not all(row["passed"] for row in checks):
        return checks, hashes, context
    spec_text = (resolved / SPEC_PATH).read_text(encoding="utf-8")
    _append_check(
        checks,
        "driving_requirement",
        upstream="research_reporting_spec",
        path=resolved / SPEC_PATH,
        field="REQ-*",
        operator="contains",
        expected="REQ-REPORT-7617",
        observed="REQ-REPORT-7617" if "REQ-REPORT-7617" in spec_text else None,
        passed="REQ-REPORT-7617" in spec_text,
    )
    schema_artifact = load_json(resolved / SCHEMA_RESULT)
    expected_schema = {
        "honest_verdict": "complete_null_evidence_schema_ready",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "evidence_schema_ready_score": 1,
        "role_contract_ready_score": 1,
        "guarded_update_ready_score": 1,
    }
    for field, expected in expected_schema.items():
        observed = schema_artifact.get(field)
        _append_check(
            checks,
            f"exp7616_{field}",
            upstream="exp7616_terminal",
            path=resolved / SCHEMA_RESULT,
            field=field,
            operator="eq",
            expected=expected,
            observed=observed,
            passed=observed == expected,
        )
    checksum = schema_protocol.reproducibility_checksum(schema_artifact)
    _append_check(
        checks,
        "exp7616_reproducibility_checksum",
        upstream="exp7616_terminal",
        path=resolved / SCHEMA_RESULT,
        field="reproducibility_checksum",
        operator="eq",
        expected=schema_artifact.get("reproducibility_checksum"),
        observed=checksum,
        passed=checksum == schema_artifact.get("reproducibility_checksum"),
    )
    historical = load_json(resolved / V664_RESULT)
    expected_v664 = {
        "verdict_class": "null",
        "flagged_adversarial": False,
        "MODEL_SPECS": MODEL_SPECS,
    }
    for field, expected in expected_v664.items():
        observed = historical.get(field)
        _append_check(
            checks,
            f"exp7604_{field}",
            upstream="exp7604_historical_context",
            path=resolved / V664_RESULT,
            field=field,
            operator="eq",
            expected=expected,
            observed=observed,
            passed=observed == expected,
        )
    historical_checksum = v664.reproducibility_checksum(historical)
    _append_check(
        checks,
        "exp7604_reproducibility_checksum",
        upstream="exp7604_historical_context",
        path=resolved / V664_RESULT,
        field="reproducibility_checksum",
        operator="eq",
        expected=historical.get("reproducibility_checksum"),
        observed=historical_checksum,
        passed=historical_checksum == historical.get("reproducibility_checksum"),
    )
    records = load_jsonl(resolved / PILOT_INPUT)
    record_ids = [str(row.get("component_hash") or "") for row in records]
    canonical_inputs = json.loads(
        (
            resolved / "results/raw/experiment_7616_v665_evidence_schema/canonical_inputs.json"
        ).read_text(encoding="utf-8")
    )
    if not isinstance(canonical_inputs, list):
        raise ValueError("exp7616_canonical_inputs_array_required")
    canonical_ids = [str(row.get("component_hash") or "") for row in canonical_inputs]
    records_valid = bool(
        len(records) == 8
        and len(set(record_ids)) == 8
        and record_ids == canonical_ids
        and all(v664.validate_input_record(row) for row in records)
        and all(schema_protocol.build_canonical_input(row) for row in records)
    )
    _append_check(
        checks,
        "frozen_disjoint_pilot_groups",
        upstream="exp7616_canonical_inputs",
        path=resolved / PILOT_INPUT,
        field="component_hashes",
        operator="ordered_eq",
        expected=canonical_ids,
        observed=record_ids,
        passed=records_valid,
    )
    role_receipt = schema_artifact.get("role_contract_receipt") or {}
    observed_exclusion = {
        "pilot_count": role_receipt.get("pilot_count"),
        "pilot_disjoint": role_receipt.get("pilot_disjoint"),
    }
    exclusion_valid = observed_exclusion == {"pilot_count": 8, "pilot_disjoint": True}
    _append_check(
        checks,
        "pilot_groups_excluded_from_measured_roles",
        upstream="exp7616_role_contract",
        path=resolved / SCHEMA_RESULT,
        field="role_contract_receipt.disjoint_pilot_count",
        operator="eq",
        expected={"pilot_count": 8, "pilot_disjoint": True},
        observed=observed_exclusion,
        passed=exclusion_valid,
    )
    versions = _tool_versions(resolved)
    _append_check(
        checks,
        "declared_tool_versions",
        upstream="worktree_virtualenv",
        path=resolved / ".venv",
        field="python_pytest_ruff_mypy",
        operator="all_declared",
        expected=["python", "pytest", "ruff", "mypy"],
        observed=versions,
        passed=set(versions) == {"python", "pytest", "ruff", "mypy"},
    )
    historical_runtime = historical.get("inference_runtime_receipt") or {}
    context.update(
        {
            "records": records,
            "schema_artifact": schema_artifact,
            "historical_artifact": historical,
            "historical_model_sha256": historical_runtime.get("model_sha256"),
            "historical_model_path": (historical_runtime.get("model_spec") or {}).get("model_path"),
            "tool_versions": versions,
        }
    )
    return checks, hashes, context


def build_blocked_artifact(  # pragma: no cover - external block integration.
    *,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    reason: str,
    duration_s: float,
) -> JsonDict:
    """Build a complete external-block record without fabricated model work."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else gate_row(
            reason,
            category="readiness",
            upstream=EXPERIMENT_ID,
            path=ROOT / RESULT_PATH,
            field="external_precondition",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            condition="external precondition is available",
            principle="An external block is complete blocked work, not partial work.",
        )
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7617,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed) or 1,
            "failed_checks": [str(row.get("check")) for row in failed] or [reason],
            "first_failure": first,
        },
        "acceptance_gate_results": [],
        "rows": [],
        "paired_pilot_rows": [],
        "sample_size_budget": {
            "independent_unit": "frozen_pilot_group",
            "intended": 8,
            "observed": 0,
            "excluded": 0,
            "censored": 8,
            "excluded_from_all_measured_roles": 8,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": "blocked_no_run",
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": "blocked_no_run",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "gpu_uuid": None},
        "phase_spans": [],
        "invocation_counts": deepcopy(v664._zero_invocations()),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds": [{"seed": ARM_ORDER_SEED, "purpose": "planned_arm_order"}],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "terminal_reader_outcomes": [],
        "verifier_is_oracle": False,
        "field_principles": field_principles(),
        "evidence_transport_ready_score": 0,
        "fit_capture_feasible_score": 0,
        "online_capture_feasible_score": 0,
        "evaluation_capture_feasible_score": 0,
        "selected_config_path": None,
        "semantic_benefit_claim": False,
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
    }
    artifact["field_principles"] = _complete_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def cold_replay(  # pragma: no cover - fresh-process reader.
    path: Path, *, root: Path = ROOT
) -> JsonDict:
    """Reload exact candidate bytes and validate without invoking a model."""

    artifact = load_json(path)
    errors = validate_artifact(artifact, root=root, require_terminal=False)
    return {
        "mode": "cold_replay",
        "passed": not errors,
        "errors": errors,
        "paired_row_count": len(artifact.get("paired_pilot_rows") or []),
        "candidate_sha256": sha256_file(path),
    }


def independent_reduce_artifact(  # pragma: no cover - fresh-process reader.
    path: Path, *, root: Path = ROOT
) -> JsonDict:
    """Recompute comparative arm metrics from rows, not proposer summaries."""

    artifact = load_json(path)
    errors = validate_artifact(artifact, root=root, require_terminal=False)
    if artifact.get("verdict_class") == "blocked":
        return {
            "mode": "independent_reduction",
            "passed": not errors,
            "errors": errors,
            "blocked_without_fabricated_rows": True,
            "comparative_metrics_recomputed_from_rows": False,
            "row_reduction_sha256": canonical_hash([]),
        }
    rows = artifact.get("paired_pilot_rows") or []
    try:
        selection = select_configuration(rows)
    except (TypeError, ValueError) as exc:
        return {"mode": "independent_reduction", "passed": False, "errors": [str(exc)]}
    metrics = {
        arm: {
            "valid_completed_count": selection["arm_reductions"][arm]["valid_completed_count"],
            "accepted_invalid_id_reference_count": selection["arm_reductions"][arm][
                "accepted_invalid_id_reference_count"
            ],
            "unknown_numerator": sum(
                int((row.get("parser_result") or {}).get("unknown_numerator") or 0)
                for row in rows
                if row.get("arm") == arm
            ),
            "unknown_denominator": sum(
                int((row.get("parser_result") or {}).get("unknown_denominator") or 0)
                for row in rows
                if row.get("arm") == arm
            ),
        }
        for arm in ARMS
    }
    return {
        "mode": "independent_reduction",
        "passed": not errors and selection == artifact.get("selection_reduction"),
        "errors": errors,
        "selection": selection,
        "comparative_metrics": metrics,
        "comparative_metrics_recomputed_from_rows": True,
        "row_reduction_sha256": canonical_hash(rows),
    }


def affected_validation_manifest() -> JsonDict:  # pragma: no cover - validation wiring.
    """Freeze the only files admitted to current scoped validation."""

    return {
        "schema": "carnot.exp7617.affected_files.v1",
        "requirement": "REQ-REPORT-7617",
        "files": [
            SPEC_PATH.as_posix(),
            MODULE_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
            TEST_PATH.as_posix(),
        ],
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }


def build_validation_commands(
    root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - subprocess integration.
    """Build focused pytest, coverage, Ruff, mypy, and spec checks."""

    return validation_scope.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / ".coverage-exp7617",
    )


def terminal_commands(  # pragma: no cover - validation wiring.
    root: Path, candidate: Path
) -> list[validation_scope.CommandSpec]:
    """Build fresh readers that decide whether exact candidate bytes publish."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--root", str(root), "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (
                python,
                "-u",
                wrapper,
                "--root",
                str(root),
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            300.0,
        ),
    ]


def _all_passed(  # pragma: no cover - subprocess receipt reduction.
    receipts: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def _reader_outcomes(  # pragma: no cover - subprocess receipt reduction.
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        {
            "name": row.get("name"),
            "exit_code": row.get("exit_code"),
            "passed": row.get("passed") is True,
            "log_sha256": row.get("log_sha256"),
            "worktree": row.get("worktree", str(ROOT)),
        }
        for row in receipts
    ]


def _atomic_bytes(path: Path, value: bytes) -> None:  # pragma: no cover - durable I/O.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _event(  # pragma: no cover - current runtime receipt.
    run_id: str, call_id: str, operation: str, state: str
) -> JsonDict:
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


def _response_parts(  # pragma: no cover - live transport boundary.
    raw: Mapping[str, Any],
) -> tuple[str, str, str]:
    choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    return (
        str(message.get("content") or ""),
        str(message.get("reasoning_content") or ""),
        str(choice.get("finish_reason") or "unknown"),
    )


def _completed_row(
    *,
    schedule_row: Mapping[str, Any],
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    request_path: Path,
    response_path: Path,
    response_bytes: bytes,
    call_start_ns: int,
    call_end_ns: int,
) -> JsonDict:  # pragma: no cover - live response boundary.
    raw = json.loads(response_bytes)
    if not isinstance(raw, dict):
        raise ValueError("response_object_required")
    content, reasoning, finish_reason = _response_parts(raw)
    parsed = parse_pilot_response(record, content, finish_reason=finish_reason)
    request_bytes = request_path.read_bytes()
    generation_s = (call_end_ns - call_start_ns) / 1_000_000_000
    return {
        **deepcopy(dict(schedule_row)),
        "status": "completed",
        "transport_completed": True,
        "lossless_input": True,
        "request": deepcopy(dict(request_payload)),
        "input_record": deepcopy(dict(record)),
        "response_text": content,
        "reasoning_text": reasoning,
        "raw_response": raw,
        "finish_reason": finish_reason,
        "parser_result": parsed,
        "unknown_fraction": parsed["unknown_fraction"],
        "request_path": str(request_path),
        "request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": len(request_bytes),
        "response_path": str(response_path),
        "response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
        "response_bytes": len(response_bytes),
        "call_start_monotonic_ns": call_start_ns,
        "call_end_monotonic_ns": call_end_ns,
        "generation_s": generation_s,
        "prefill_s": v664._timing_value(raw, "prompt_ms", "prompt_per_second_ms"),
        "decode_s": v664._timing_value(raw, "predicted_ms", "predicted_per_token_ms"),
        "prompt_tokens": v664._token_count(raw, "prompt_n", "prompt_tokens"),
        "output_tokens": v664._token_count(raw, "predicted_n", "completion_tokens"),
        "exception": None,
        "numerator": int(parsed["valid_completed"] is True),
        "denominator": 1,
        "direction": "higher_is_more_independently_valid_syntax",
        "censored": parsed["valid_completed"] is not True,
        "raw_provenance": {
            "producer": EXPERIMENT_ID,
            "call_id": f"paired-{schedule_row['group_index']}-{schedule_row['arm']}",
        },
    }


def _failed_row(
    *,
    schedule_row: Mapping[str, Any],
    record: Mapping[str, Any],
    request_payload: Mapping[str, Any],
    request_path: Path,
    error: BaseException,
    call_start_ns: int,
    call_end_ns: int,
) -> JsonDict:  # pragma: no cover - live failure boundary.
    request_bytes = request_path.read_bytes()
    return {
        **deepcopy(dict(schedule_row)),
        "status": "failed",
        "transport_completed": False,
        "lossless_input": True,
        "request": deepcopy(dict(request_payload)),
        "input_record": deepcopy(dict(record)),
        "response_text": "",
        "reasoning_text": "",
        "raw_response": None,
        "finish_reason": "transport_failure",
        "parser_result": {
            "accepted": False,
            "valid_completed": False,
            "parser_error": "transport_failure",
            "invalid_id_reference": False,
            "evidence": [],
            "unknown_numerator": 0,
            "unknown_denominator": 0,
            "unknown_fraction": 0.0,
            "syntax_coverage_only": True,
            "semantic_correctness_measured": False,
        },
        "unknown_fraction": 0.0,
        "request_path": str(request_path),
        "request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": len(request_bytes),
        "response_path": None,
        "response_sha256": None,
        "response_bytes": 0,
        "call_start_monotonic_ns": call_start_ns,
        "call_end_monotonic_ns": call_end_ns,
        "generation_s": (call_end_ns - call_start_ns) / 1_000_000_000,
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "exception": f"{type(error).__name__}:{error}",
        "numerator": 0,
        "denominator": 1,
        "direction": "higher_is_more_independently_valid_syntax",
        "censored": True,
        "raw_provenance": {
            "producer": EXPERIMENT_ID,
            "call_id": f"paired-{schedule_row['group_index']}-{schedule_row['arm']}",
        },
    }


def run_fresh_request(spec_path: Path, output_path: Path) -> int:  # pragma: no cover
    """Issue one selected request in a new process and independently reduce it."""

    started = time.monotonic()
    spec = load_json(spec_path)
    record = spec.get("record")
    request = spec.get("request")
    if not isinstance(record, Mapping) or not isinstance(request, Mapping):
        print("fresh_request_spec_invalid", flush=True)
        return 2
    progress(started, "fresh_process_selected_request", "before_generation")
    call_start_ns = time.monotonic_ns()
    try:
        response_bytes = v664._post_json(
            str(spec["url"]), request, float(spec.get("timeout_s") or 900.0)
        )
    except BaseException as exc:
        print(f"fresh_request_failed:{type(exc).__name__}:{exc}", flush=True)
        return 1
    call_end_ns = time.monotonic_ns()
    progress(started, "fresh_process_selected_request", "after_generation")
    raw = json.loads(response_bytes)
    if not isinstance(raw, dict):
        print("fresh_response_object_required", flush=True)
        return 1
    content, reasoning, finish_reason = _response_parts(raw)
    parsed = parse_pilot_response(record, content, finish_reason=finish_reason)
    timing_reduction = {
        "generation_s": (call_end_ns - call_start_ns) / 1_000_000_000,
        "prefill_s": v664._timing_value(raw, "prompt_ms", "prompt_per_second_ms"),
        "decode_s": v664._timing_value(raw, "predicted_ms", "predicted_per_token_ms"),
        "prompt_tokens": v664._token_count(raw, "prompt_n", "prompt_tokens"),
        "output_tokens": v664._token_count(raw, "predicted_n", "completion_tokens"),
    }
    result = {
        "arm": spec.get("arm"),
        "component_hash": record.get("component_hash"),
        "request_sha256": canonical_hash(request),
        "raw_response": raw,
        "raw_response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
        "response_text": content,
        "reasoning_text": reasoning,
        "finish_reason": finish_reason,
        "parser_result": parsed,
        "independent_timing_reduction": timing_reduction,
        "passed": parsed["valid_completed"] is True,
    }
    current_work_receipt.atomic_json(output_path, result)
    print(json.dumps({"fresh_process_selected_request": result["passed"]}), flush=True)
    return int(result["passed"] is not True)


def _e2e_command(  # pragma: no cover - subprocess wiring.
    root: Path, spec_path: Path, output_path: Path
) -> validation_scope.CommandSpec:
    return validation_scope.CommandSpec(
        "fresh_process_selected_request",
        (
            str(root / ".venv/bin/python"),
            "-u",
            WRAPPER_PATH.as_posix(),
            "--root",
            str(root),
            "--fresh-request",
            str(spec_path),
            "--fresh-request-output",
            str(output_path),
        ),
        "selected_config_task_e2e",
        900.0,
    )


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
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:  # pragma: no cover
    """Load one owned server, run 16 calls once, then one fresh-process E2E."""

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _call_with_heartbeats,
        _observed_offload_layers,
        _owned_vram_mb,
        process_start_tick,
    )

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
    proposer.model_repository = MODEL_ID
    proposer.requested_model_path = str(model_path)
    run_id = f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}"
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    e2e_bundle: JsonDict = {}
    receipt_start_ns = time.monotonic_ns()
    load_start_ns = time.monotonic_ns()
    generation_started = time.monotonic()
    runtime: JsonDict = {
        "model_spec": deepcopy(dict(model_spec)),
        "model_sha256": model_sha256,
        "model_bytes": model_path.stat().st_size,
        "quantization": "Q4_K_M",
        "gpu_uuid": selected_gpu["uuid"],
        "gpu_index": selected_gpu["index"],
        "owner_pid": os.getpid(),
        "owner_pid_start_ticks": process_start_tick(os.getpid()),
        "lease_owner": lease.owner_receipt(),
        "signals_sent_to_foreign_processes": [],
        "validation_reserve_s": VALIDATION_RESERVE_S,
    }
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
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        owned_vram_mb = _owned_vram_mb(server_pid)
        offload = v664._offload_receipt(log_path, owned_vram_mb, _observed_offload_layers(log_path))
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
                "runtime_build": v664._runtime_build_receipt(),
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
        authenticated = bool(
            healthy
            and runtime["server_pid_start_ticks"] is not None
            and int(runtime.get("owned_vram_mb") or 0) > 0
            and offload.get("actual_offload") is True
            and runtime["runtime_build"].get("native_library_sha256")
        )
        if not authenticated:
            events.append(_event(run_id, "model-load-1", "model_load", "failed"))
            raise RuntimeError("owned_model_load_not_authenticated")
        events.append(_event(run_id, "model-load-1", "model_load", "completed"))
        lease.transition("resident", vram_mb=int(runtime["owned_vram_mb"]))
        lease.transition("inferencing")
        schedule = seeded_paired_schedule(records)
        by_component = {str(row["component_hash"]): row for row in records}
        for call_index, schedule_row in enumerate(schedule, 1):
            elapsed_generation = time.monotonic() - generation_started
            if elapsed_generation >= GENERATION_HARD_STOP_S:
                raise TimeoutError("generation_hard_stop_reached")
            record = by_component[str(schedule_row["component_hash"])]
            payload = build_arm_request(record, str(schedule_row["arm"]))
            request_bytes = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            request_path = requests_dir / f"{call_index:02d}_{schedule_row['arm']}.json"
            response_path = responses_dir / f"{call_index:02d}_{schedule_row['arm']}.json"
            _atomic_bytes(request_path, request_bytes)
            call_id = f"generation-{call_index}"
            events.append(_event(run_id, call_id, "generation", "attempted"))
            call_start_ns = time.monotonic_ns()
            progress(
                started, "generation", "before", unit=f"{call_index}/16", arm=schedule_row["arm"]
            )
            try:
                timeout_s = max(
                    1.0,
                    min(900.0, GENERATION_HARD_STOP_S - (time.monotonic() - generation_started)),
                )
                response_bytes = _call_with_heartbeats(
                    lambda payload=payload, timeout_s=timeout_s: v664._post_json(
                        proposer._url() + "/v1/chat/completions", payload, timeout_s
                    ),
                    started=started,
                    phase=f"generation_{call_index}",
                )
                call_end_ns = time.monotonic_ns()
                _atomic_bytes(response_path, response_bytes)
                row = _completed_row(
                    schedule_row=schedule_row,
                    record=record,
                    request_payload=payload,
                    request_path=request_path,
                    response_path=response_path,
                    response_bytes=response_bytes,
                    call_start_ns=call_start_ns,
                    call_end_ns=call_end_ns,
                )
                events.append(_event(run_id, call_id, "generation", "completed"))
            except BaseException as exc:
                call_end_ns = time.monotonic_ns()
                row = _failed_row(
                    schedule_row=schedule_row,
                    record=record,
                    request_payload=payload,
                    request_path=request_path,
                    error=exc,
                    call_start_ns=call_start_ns,
                    call_end_ns=call_end_ns,
                )
                events.append(_event(run_id, call_id, "generation", "failed"))
            rows.append(row)
            current_work_receipt.atomic_json(run_dir / "checkpoint.json", {"paired_rows": rows})
            progress(
                started,
                "generation",
                "after",
                unit=f"{call_index}/16",
                arm=schedule_row["arm"],
                valid=row["parser_result"]["valid_completed"],
            )
        selection = select_configuration(rows)
        if selection["selected_arm"] is not None:
            e2e_record = records[0]
            e2e_request = build_arm_request(e2e_record, str(selection["selected_arm"]))
            e2e_spec_path = run_dir / "fresh_process_selected_request_spec.json"
            e2e_output_path = run_dir / "fresh_process_selected_request_result.json"
            current_work_receipt.atomic_json(
                e2e_spec_path,
                {
                    "url": proposer._url() + "/v1/chat/completions",
                    "arm": selection["selected_arm"],
                    "record": e2e_record,
                    "request": e2e_request,
                    "timeout_s": min(
                        900.0,
                        max(
                            1.0,
                            GENERATION_HARD_STOP_S - (time.monotonic() - generation_started),
                        ),
                    ),
                },
            )
            events.append(_event(run_id, "generation-e2e", "generation", "attempted"))
            progress(started, "fresh_process_selected_request", "before")
            e2e_receipts = validation_scope.run_commands(
                root,
                [_e2e_command(root, e2e_spec_path, e2e_output_path)],
                log_dir=run_dir / "fresh_process_logs",
                heartbeat_s=60.0,
            )
            progress(
                started,
                "fresh_process_selected_request",
                "after",
                passed=_all_passed(e2e_receipts),
            )
            if not _all_passed(e2e_receipts) or not e2e_output_path.is_file():
                events.append(_event(run_id, "generation-e2e", "generation", "failed"))
                raise RuntimeError("fresh_process_selected_request_failed")
            events.append(_event(run_id, "generation-e2e", "generation", "completed"))
            e2e_bundle = {
                "receipt": e2e_receipts[0],
                "result": load_json(e2e_output_path),
            }
        else:
            e2e_bundle = {
                "receipt": {
                    "name": "fresh_process_selected_request",
                    "exit_code": 0,
                    "passed": True,
                    "timed_out": False,
                    "scope": "not_applicable_no_selected_arm",
                    "log_sha256": canonical_hash("not_applicable_no_selected_arm"),
                    "worktree": str(root),
                },
                "result": {"passed": True, "not_applicable": "no_selected_arm"},
            }
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
            lease.transition("terminal_complete" if len(rows) == 16 else "terminal_blocked")
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
    runtime["transport_authenticated"] = bool(
        len(rows) == 16
        and all(row.get("transport_completed") is True for row in rows)
        and all(int(row.get("prompt_tokens") or 0) > 0 for row in rows)
    )
    runtime["generation_elapsed_s"] = time.monotonic() - generation_started
    runtime["generation_hard_stop_s"] = GENERATION_HARD_STOP_S
    runtime["chat_template"] = (runtime.get("server_props") or {}).get("chat_template")
    runtime["chat_template_sha256"] = schema_protocol.text_sha256(
        str(runtime.get("chat_template") or "")
    )
    receipt = current_work_receipt.build_current_work_receipt(
        run_id=run_id,
        owner_pid=os.getpid(),
        events=events,
        inference_substrate="owned_local_llama_cpp_paired_bounded_generation",
        inference_substrate_details={
            "model_id": MODEL_ID,
            "gpu_uuid": selected_gpu["uuid"],
            "fixed_output_budget": MAX_TOKENS,
            "paired_generation_calls": 16,
            "fresh_process_e2e_calls": int("result" in e2e_bundle),
        },
        inference_substrate_class="model_bounded_generation",
        execution_venue="host",
        started_monotonic_ns=receipt_start_ns,
        ended_monotonic_ns=receipt_end_ns,
        phase_spans=[],
    )
    receipt["MODEL_SPECS"] = MODEL_SPECS
    current_work_receipt.atomic_json(run_dir / "current_work_receipt.json", receipt)
    return rows, runtime, receipt, e2e_bundle


def capture_configuration(  # pragma: no cover - live runtime custody.
    records: Sequence[Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> JsonDict:
    """Freeze the exact shared prompt, grammar, parser, model, and backend."""

    prompt_hashes: dict[str, str] = {}
    grammar_hashes: dict[str, str] = {}
    schema_hashes: dict[str, str] = {}
    for record in records:
        component = str(record["component_hash"])
        authority = schema_protocol.build_schema_authority(record)
        prompt_hashes[component] = schema_protocol.text_sha256(
            schema_protocol.render_system_prompt(authority)
        )
        grammar_hashes[component] = schema_protocol.text_sha256(
            schema_protocol.compile_decoder_grammar(authority)
        )
        schema_hashes[component] = canonical_hash(authority["json_schema"])
    runtime_build = runtime.get("runtime_build") or {}
    return {
        "schema": "carnot.exp7617.selected_config.v1",
        "selected_arm": selection.get("selected_arm"),
        "scale_decision": selection.get("scale_decision"),
        "selection_rule": "grammar_first_then_control_if_valid_completed_gte_6_and_bad_ids_eq_0",
        "model_id": MODEL_ID,
        "model_sha256": runtime.get("model_sha256"),
        "quantization": runtime.get("quantization"),
        "backend": "llama.cpp_openai_compatible_server",
        "backend_version": runtime_build.get("llama_cpp_version"),
        "backend_native_sha256": runtime_build.get("native_library_sha256"),
        "chat_template_sha256": runtime.get("chat_template_sha256"),
        "temperature": 0.0,
        "max_new_tokens": MAX_TOKENS,
        "thinking": False,
        "no_think": True,
        "retry_count": 0,
        "arm_order_seed": ARM_ORDER_SEED,
        "generation_seed": RANDOM_SEED,
        "canonical_input_representation": "ordered_exact_sentence_arrays_once",
        "independent_parser": (
            "carnot.experiment_7616_v665_evidence_schema.validate_evidence_output"
        ),
        "prompt_sha256_by_component": prompt_hashes,
        "compiled_grammar_sha256_by_component": grammar_hashes,
        "decoder_schema_sha256_by_component": schema_hashes,
    }


def _span(  # pragma: no cover - live monotonic timing.
    phase: str,
    *,
    task_started: float,
    phase_started: float,
    planned: int,
    completed: int,
    pending: str | None = None,
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - task_started,
        "ended_offset_s": ended - task_started,
        "duration_s": ended - phase_started,
        "planned_units": planned,
        "completed_units": completed,
        "pending_operation": pending,
        "checkpoint_position": completed,
    }


def _private_root() -> Path:  # pragma: no cover - host temporary directory.
    return Path(tempfile.mkdtemp(prefix="carnot-exp7617-"))


def _select_gpu_allowing_current_pid(  # pragma: no cover - live GPU boundary.
) -> tuple[JsonDict | None, list[JsonDict]]:
    """Select an idle card while ignoring only this task's tokenizer context."""

    from carnot.experiment_7581_v662_arc_bounded_canary import gpu_inventory

    inventory = gpu_inventory()
    candidates = [
        deepcopy(dict(row))
        for row in inventory
        if int(row.get("memory_free_mb") or 0) >= GPU_REQUIRED_FREE_MB
        and all(
            int(process.get("pid") or -1) == os.getpid() for process in row.get("processes") or []
        )
    ]
    return (
        min(candidates, key=lambda row: int(row["index"])) if candidates else None,
        inventory,
    )


def _normalize_receipts(
    receipts: Sequence[Mapping[str, Any]], root: Path
) -> list[JsonDict]:  # pragma: no cover - receipt paths.
    normalized: list[JsonDict] = []
    for row in receipts:
        value = deepcopy(dict(row))
        log_path = Path(str(value.get("log_path") or ""))
        if log_path.is_absolute() and log_path.is_relative_to(root):
            value["log_path"] = log_path.relative_to(root).as_posix()
        normalized.append(value)
    return normalized


def _finalize_candidate(
    *,
    root: Path,
    artifact: JsonDict,
    private_root: Path,
    destination: Path,
    started: float,
) -> int:  # pragma: no cover - exact subprocess and publication boundary.
    """Persist reader outcomes, then atomically publish exact accepted bytes."""

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
        *_normalize_receipts(receipts, root),
    ]
    artifact["terminal_reader_outcomes"] = _reader_outcomes(receipts)
    artifact["flagged_adversarial"] = False
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        print(json.dumps({"publication_blocked": errors}, sort_keys=True), flush=True)
        return 1
    raw_root = root / RAW_DIR
    current_work_receipt.atomic_json(raw_root / "exact_terminal_candidate.json", artifact)
    current_work_receipt.atomic_json(
        raw_root / "exact_terminal_reader_outcomes.json",
        artifact["terminal_reader_outcomes"],
    )
    current_work_receipt.atomic_json(destination, artifact)
    progress(started, "publish", "after", path=destination, sha256=sha256_file(destination))
    return 0


def _publish_blocked(
    *,
    root: Path,
    destination: Path,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    reason: str,
    started: float,
    started_ns: int,
    private_root: Path,
) -> int:  # pragma: no cover - external block path.
    artifact = build_blocked_artifact(
        checks=checks,
        source_hashes=source_hashes,
        reason=reason,
        duration_s=(time.monotonic_ns() - started_ns) / 1_000_000_000,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        print(json.dumps({"blocked_artifact_invalid": errors}, sort_keys=True), flush=True)
        return 1
    return _finalize_candidate(
        root=root,
        artifact=artifact,
        private_root=private_root,
        destination=destination,
        started=started,
    )


def run_experiment(
    root: Path, run_date: str, *, output_path: Path | None = None
) -> int:  # pragma: no cover - declared capability E2E.
    """Authenticate, measure paired calls, validate, and atomically publish."""

    resolved = root.resolve()
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    destination = output_path or resolved / RESULT_PATH
    if not destination.is_absolute():
        destination = resolved / destination
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    private_root = _private_root()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "before", root=resolved, run_date=run_date)
    phase_started = time.monotonic()
    checks, source_hashes, context = collect_preconditions(resolved)
    spans.append(
        _span(
            "preconditions",
            task_started=started,
            phase_started=phase_started,
            planned=len(checks),
            completed=sum(row.get("passed") is True for row in checks),
        )
    )
    preconditions_ok = all(row.get("passed") is True for row in checks)
    progress(started, "preconditions", "after", passed=preconditions_ok)
    if not preconditions_ok:
        first = next(row for row in checks if row.get("passed") is not True)
        return _publish_blocked(
            root=resolved,
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason=str(first["check"]).replace(":", "_"),
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )

    phase_started = time.monotonic()
    progress(started, "model_discovery", "before")
    pair = cached_sota_pair(preferred_quant="Q4_K_M")
    current = (
        next((dict(row) for row in pair or [] if row.get("hf_id") == MODEL_ID), None)
        if pair
        else None
    )
    declared_model_path = Path(str((current or {}).get("model_path") or resolved / "missing"))
    model_path = declared_model_path.resolve()
    cache_ok = bool(
        current
        and model_path.is_file()
        and "Q4_K_M" in declared_model_path.name
        and model_path.stat().st_size > 15_000_000_000
    )
    _append_check(
        checks,
        "cached_sota_pair_q4_k_m",
        upstream="cached_sota_pair",
        path=model_path,
        field="hf_id_quantization_bytes",
        operator="authenticated",
        expected={"hf_id": MODEL_ID, "quantization": "Q4_K_M", "minimum_bytes": 15_000_000_000},
        observed={
            "hf_id": (current or {}).get("hf_id"),
            "declared_name": declared_model_path.name,
            "resolved_name": model_path.name,
            "bytes": model_path.stat().st_size if model_path.is_file() else None,
        },
        passed=cache_ok,
        category="readiness",
    )
    progress(started, "model_discovery", "after", passed=cache_ok, path=model_path)
    if not cache_ok:
        return _publish_blocked(
            root=resolved,
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason="cached_sota_pair_q4_k_m",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )

    from carnot.experiment_7581_v662_arc_bounded_canary import _call_with_heartbeats

    progress(started, "model_hash", "before", bytes=model_path.stat().st_size)
    model_sha256 = _call_with_heartbeats(
        lambda: sha256_file(model_path), started=started, phase="model_hash"
    )
    progress(started, "model_hash", "after", sha256=model_sha256)
    historical_sha = context.get("historical_model_sha256")
    same_model = model_sha256 == historical_sha
    _append_check(
        checks,
        "identical_v664_model_bytes",
        upstream="exp7604_model_receipt",
        path=model_path,
        field="model_sha256",
        operator="eq",
        expected=historical_sha,
        observed=model_sha256,
        passed=same_model,
        category="readiness",
    )
    source_hashes.append(
        _source_row(model_path, producer="cached_sota_pair", source_class="authenticated_model")
    )
    progress(started, "tokenizer_preflight", "before")
    tokenizer_ok, tokenizer_detail = _call_with_heartbeats(
        lambda: gguf_tokenizer_loadable(str(model_path)),
        started=started,
        phase="tokenizer_preflight",
    )
    progress(started, "tokenizer_preflight", "after", passed=tokenizer_ok)
    _append_check(
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
    if not same_model or not tokenizer_ok:
        return _publish_blocked(
            root=resolved,
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason="model_identity_or_tokenizer",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    selected, inventory = _select_gpu_allowing_current_pid()
    _append_check(
        checks,
        "exclusive_cuda_capacity",
        upstream="nvidia-smi",
        path=resolved / RAW_DIR,
        field="idle_gpu_with_required_free_mb",
        operator="gte",
        expected=GPU_REQUIRED_FREE_MB,
        observed=selected if selected is not None else inventory,
        passed=selected is not None,
        category="readiness",
    )
    progress(
        started,
        "cuda_capacity",
        "after",
        passed=selected is not None,
        gpu_uuid=selected.get("uuid") if selected else None,
    )
    if selected is None:
        return _publish_blocked(
            root=resolved,
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason="exclusive_cuda_capacity",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    assert current is not None
    current["gpu"] = int(selected["index"])
    current["quantization"] = "Q4_K_M"
    current["sha256"] = model_sha256
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
    except Exception as exc:
        _append_check(
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
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason="exclusive_gpu_lease",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    _append_check(
        checks,
        "exclusive_gpu_lease",
        upstream="carnot.gpu_lease_phase_journal.GpuLease",
        path=lease_root,
        field="exclusive_owner",
        operator="acquired",
        expected={"gpu_uuid": selected["uuid"], "task_id": EXPERIMENT_ID},
        observed={"gpu_uuid": selected["uuid"], "task_id": EXPERIMENT_ID},
        passed=True,
        category="readiness",
    )
    progress(started, "gpu_lease", "after", acquired=True, lease_id=lease.lease_id)
    spans.append(
        _span(
            "model_preflight",
            task_started=started,
            phase_started=phase_started,
            planned=5,
            completed=5,
        )
    )

    run_dir = resolved / RAW_DIR / "runs" / f"{int(time.time())}-{os.getpid()}"
    phase_started = time.monotonic()
    progress(started, "paired_pilot", "before", run_dir=run_dir)
    try:
        rows, runtime, receipt, e2e_bundle = run_owned_pilot(
            root=resolved,
            records=context["records"],
            model_spec=current,
            model_sha256=model_sha256,
            selected_gpu=selected,
            lease=lease,
            started=started,
            run_dir=run_dir,
        )
    except BaseException as exc:
        _append_check(
            checks,
            "owned_paired_transport",
            upstream="LocalGGUFProposer",
            path=run_dir,
            field="one_load_sixteen_paired_calls",
            operator="completed",
            expected={"loads": 1, "paired_calls": 16},
            observed={"error": f"{type(exc).__name__}:{exc}"},
            passed=False,
            category="readiness",
        )
        progress(started, "paired_pilot", "after", passed=False, error=type(exc).__name__)
        return _publish_blocked(
            root=resolved,
            destination=destination,
            checks=checks,
            source_hashes=source_hashes,
            reason="owned_paired_transport",
            started=started,
            started_ns=started_ns,
            private_root=private_root,
        )
    _append_check(
        checks,
        "owned_paired_transport",
        upstream="LocalGGUFProposer",
        path=run_dir,
        field="one_load_sixteen_paired_calls",
        operator="completed",
        expected={"loads": 1, "paired_calls": 16},
        observed={"loads": 1, "paired_calls": len(rows)},
        passed=len(rows) == 16,
        category="readiness",
    )
    spans.append(
        _span(
            "paired_generation_and_e2e",
            task_started=started,
            phase_started=phase_started,
            planned=17,
            completed=16 + int(bool(e2e_bundle.get("result"))),
        )
    )
    progress(started, "paired_pilot", "after", passed=True, calls=len(rows))

    selection = select_configuration(rows)
    config = capture_configuration(context["records"], selection=selection, runtime=runtime)
    config_path = resolved / RAW_DIR / "selected_configuration.json"
    current_work_receipt.atomic_json(config_path, config)
    config_receipt = {"path": str(config_path), "sha256": sha256_file(config_path)}
    manifest_path = resolved / RAW_DIR / "affected_validation_manifest.json"
    current_work_receipt.atomic_json(manifest_path, affected_validation_manifest())
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            source_hashes.append(
                _source_row(
                    path,
                    producer="exp7617_owned_runtime",
                    source_class="immutable_raw_evidence",
                )
            )
    source_hashes.extend(
        [
            _source_row(
                config_path,
                producer=EXPERIMENT_ID,
                source_class="current_frozen_output",
            ),
            _source_row(
                manifest_path,
                producer=EXPERIMENT_ID,
                source_class="conductor_pre_gate_record",
            ),
        ]
    )
    for relative in (SPEC_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = resolved / relative
        if path.is_file():
            source_hashes.append(
                _source_row(
                    path,
                    producer=EXPERIMENT_ID,
                    source_class="conductor_pre_gate_record",
                )
            )

    _append_check(
        checks,
        "authenticated_runtime_transport",
        upstream="exp7617_current_runtime",
        path=run_dir,
        field="transport_authenticated",
        operator="eq",
        expected=True,
        observed=runtime.get("transport_authenticated"),
        passed=runtime.get("transport_authenticated") is True,
        category="validity",
    )
    phase_started = time.monotonic()
    progress(started, "scoped_validation", "before")
    scoped_receipts = validation_scope.run_commands(
        resolved,
        build_validation_commands(resolved, private_root),
        log_dir=private_root / "logs" / "scoped",
        heartbeat_s=60.0,
    )
    progress(started, "scoped_validation", "after", passed=_all_passed(scoped_receipts))
    spans.append(
        _span(
            "scoped_validation",
            task_started=started,
            phase_started=phase_started,
            planned=len(scoped_receipts),
            completed=sum(row.get("passed") is True for row in scoped_receipts),
        )
    )
    if not _all_passed(scoped_receipts):
        return 1
    validation_receipts = [
        deepcopy(dict(e2e_bundle["receipt"])),
        *_normalize_receipts(scoped_receipts, resolved),
    ]
    ended_ns = time.monotonic_ns()
    artifact = build_artifact(
        rows=rows,
        runtime=runtime,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=validation_receipts,
        phase_spans=spans,
        selected_config_path=config_receipt,
        duration_s=(ended_ns - started_ns) / 1_000_000_000,
        current_receipt=receipt,
        e2e_result=e2e_bundle["result"],
    )
    errors = validate_artifact(artifact, root=resolved, require_terminal=False)
    if errors:
        print(json.dumps({"candidate_invalid": errors}, sort_keys=True), flush=True)
        return 1
    return _finalize_candidate(
        root=resolved,
        artifact=artifact,
        private_root=private_root,
        destination=destination,
        started=started,
    )


def _argument_path(path: Path, root: Path) -> Path:  # pragma: no cover - CLI boundary.
    return path if path.is_absolute() else root / path


def parse_args(  # pragma: no cover - CLI boundary.
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse production and read-only candidate modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--fresh-request", type=Path)
    parser.add_argument("--fresh-request-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the producer or one fresh-process reader without extra inference."""

    args = parse_args(argv)
    root = args.root.resolve()
    modes = [args.validate, args.cold_replay, args.independent_reduce, args.fresh_request]
    if sum(value is not None for value in modes) > 1:
        print("reader_modes_are_mutually_exclusive", flush=True)
        return 2
    if args.validate is not None:
        errors = validate_artifact(load_json(_argument_path(args.validate, root)), root=root)
        print(json.dumps({"mode": "validate", "valid": not errors, "errors": errors}), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        outcome = cold_replay(_argument_path(args.cold_replay, root), root=root)
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(outcome["passed"] is not True)
    if args.independent_reduce is not None:
        outcome = independent_reduce_artifact(
            _argument_path(args.independent_reduce, root), root=root
        )
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return int(outcome["passed"] is not True)
    if args.fresh_request is not None:
        if args.fresh_request_output is None:
            print("fresh_request_output_required", flush=True)
            return 2
        return run_fresh_request(
            _argument_path(args.fresh_request, root),
            _argument_path(args.fresh_request_output, root),
        )
    return run_experiment(root, args.date, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
