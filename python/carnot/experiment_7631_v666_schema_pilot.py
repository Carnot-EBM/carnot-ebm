"""Measure V666 explicit-schema and schema-constrained evidence transport.

The pure reducers in this module preserve all eight frozen pilot groups and
keep syntax transport separate from semantic benefit. Live orchestration uses
the Exp7630 ownership checks and blocks before model work when capacity is not
owned.

Spec: REQ-REPORT-7631 and SCENARIO-REPORT-7631-*.
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
import tempfile
import time
from typing import Any

from carnot import experiment_7616_v665_evidence_schema as schema_protocol
from carnot import experiment_7617_v665_schema_pilot as prior_pilot
from carnot import experiment_7630_v666_cuda_ownership as ownership
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.666"
EXPERIMENT_ID = "experiment_7631_v666_schema_pilot"
SCHEMA = "carnot.exp7631.v666.schema_pilot.v1"
RESULT_PATH = Path("results/experiment_7631_v666_schema_pilot.json")
RAW_DIR = Path("results/raw/experiment_7631_v666_schema_pilot")
SELECTED_CONFIG_PATH = RAW_DIR / "selected_config.json"
MODULE_PATH = Path("python/carnot/experiment_7631_v666_schema_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7631_v666_schema_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7631_v666_schema_pilot.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
SCHEMA_RESULT = Path("results/experiment_7616_v665_evidence_schema.json")
OWNERSHIP_RESULT = Path("results/experiment_7630_v666_cuda_ownership.json")
PILOT_INPUT = Path(
    "results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl"
)
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
EXPLICIT_ARM = "explicit_schema_prompt"
CONSTRAINED_ARM = "schema_constrained_decoding"
ARMS = (EXPLICIT_ARM, CONSTRAINED_ARM)
RANDOM_SEED = 7631
MAX_TOKENS = 512
GENERATION_LIMIT_S = 3000.0
TOTAL_LIMIT_S = 4500.0
VALIDATION_RESERVE_S = 300.0
TERMINAL_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


def text_hash(value: str) -> str:
    """Hash exact UTF-8 bytes retained by one transport row."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON for exact cold reductions."""

    return current_work_receipt.canonical_hash(value)


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit a flushed monotonic phase boundary."""

    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_ID,
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **details,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def build_arm_request(record: Mapping[str, Any], arm: str) -> Json:
    """Build paired requests whose only difference is constrained decoding."""

    if arm not in ARMS:
        raise ValueError("pilot_arm_invalid")
    request = schema_protocol.build_extraction_request(record)
    request.pop("response_format", None)
    request["seed"] = RANDOM_SEED
    request["max_tokens"] = MAX_TOKENS
    if arm == CONSTRAINED_ARM:
        authority = schema_protocol.build_schema_authority(record)
        request["grammar"] = schema_protocol.compile_decoder_grammar(authority)
    return request


def counterbalanced_schedule(
    records: Sequence[Mapping[str, Any]], *, seed: int = RANDOM_SEED
) -> list[Json]:
    """Freeze one deterministic paired order without changing sample size."""

    identities = [str(row.get("component_hash") or "") for row in records]
    if len(records) != 8 or len(set(identities)) != 8 or "" in identities:
        raise ValueError("exactly_eight_disjoint_pilot_groups_required")
    rng = random.Random(seed)
    schedule: list[Json] = []
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
) -> Json:
    """Independently validate saved bytes and retain every rejection reason."""

    outcome = schema_protocol.validate_evidence_output(
        record, response_text, finish_reason=finish_reason
    )
    error = str(outcome.get("error") or "")
    reasons = [error] if error else []
    accepted = outcome.get("accepted") is True
    pointer_valid = accepted and error not in {
        "response_sentence_id_invalid",
        "source_sentence_id_invalid",
    }
    evidence = outcome.get("evidence") if isinstance(outcome.get("evidence"), list) else []
    unknown = sum(isinstance(row, Mapping) and row.get("relation") == "unknown" for row in evidence)
    return {
        "accepted": accepted,
        "schema_valid": accepted,
        "pointer_valid": pointer_valid,
        "valid_completed": pointer_valid and finish_reason != "length",
        "parser_error": error or None,
        "rejection_reasons": reasons,
        "evidence": deepcopy(evidence),
        "unknown_numerator": unknown,
        "unknown_denominator": len(evidence),
        "unknown_fraction": unknown / len(evidence) if evidence else 0.0,
        "semantic_success_measured": False,
    }


def _arm_reduction(rows: Sequence[Mapping[str, Any]], arm: str) -> Json:
    arm_rows = [row for row in rows if row.get("arm") == arm]
    components = {str(row.get("component_hash") or "") for row in arm_rows}
    if len(arm_rows) != 8 or len(components) != 8 or "" in components:
        raise ValueError(f"paired_arm_groups_invalid:{arm}")
    valid = sum(
        row.get("status") == "completed"
        and row.get("transport_completed") is True
        and row.get("pointer_valid") is True
        and row.get("finish_reason") != "length"
        and row.get("censored") is not True
        for row in arm_rows
    )
    truncations = sum(row.get("finish_reason") == "length" for row in arm_rows)
    rejected = sum(row.get("pointer_valid") is not True for row in arm_rows)
    return {
        "arm": arm,
        "independent_groups": 8,
        "pointer_valid_numerator": valid,
        "pointer_valid_denominator": 8,
        "truncation_count": truncations,
        "independent_rejection_count": rejected,
        "passed": valid == 8 and truncations == 0,
    }


def select_configuration(rows: Sequence[Mapping[str, Any]], *, decoder_supported: bool) -> Json:
    """Prefer constrained decoding only at 8/8, then apply the same explicit gate."""

    if len(rows) != 16:
        raise ValueError("exactly_sixteen_paired_rows_required")
    constrained = _arm_reduction(rows, CONSTRAINED_ARM)
    explicit = _arm_reduction(rows, EXPLICIT_ARM)
    constrained["decoder_support_authenticated"] = bool(decoder_supported)
    constrained["passed"] = bool(constrained["passed"] and decoder_supported)
    selected = (
        CONSTRAINED_ARM if constrained["passed"] else EXPLICIT_ARM if explicit["passed"] else None
    )
    return {
        "selection_order": [CONSTRAINED_ARM, EXPLICIT_ARM],
        "arm_reductions": {
            CONSTRAINED_ARM: constrained,
            EXPLICIT_ARM: explicit,
        },
        "selected_arm": selected,
        "scale_decision": "selected_for_scaling" if selected else "stop_scaling",
        "evidence_transport_ready_score": int(selected is not None),
        "semantic_benefit_claim": False,
    }


def _p90(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("completed_timing_absent")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def project_fixed_rosters(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_load_s: float,
    validation_reserve_s: float = VALIDATION_RESERVE_S,
) -> dict[str, Json]:
    """Project fixed rosters from real load, tokenization, and p90 request costs."""

    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and isinstance(row.get("request_s"), (int, float))
        and not isinstance(row.get("request_s"), bool)
        and isinstance(row.get("tokenization_s"), (int, float))
        and not isinstance(row.get("tokenization_s"), bool)
    ]
    request_p90 = _p90([float(row["request_s"]) for row in completed])
    tokenization_p90 = _p90([float(row["tokenization_s"]) for row in completed])
    projections: dict[str, Json] = {}
    for name, count in (("fit_tune_policy", 120), ("online", 80), ("evaluation", 40)):
        capture = float(model_load_s) + count * (request_p90 + tokenization_p90)
        total = capture + float(validation_reserve_s)
        projections[name] = {
            "roster_size": count,
            "roster_resized": False,
            "model_load_s": float(model_load_s),
            "request_p90_s": request_p90,
            "tokenization_p90_s": tokenization_p90,
            "projected_capture_s": capture,
            "capture_limit_s": GENERATION_LIMIT_S,
            "validation_reserve_s": float(validation_reserve_s),
            "projected_total_s": total,
            "total_limit_s": TOTAL_LIMIT_S,
            "feasible_score": int(capture <= GENERATION_LIMIT_S and total <= TOTAL_LIMIT_S),
        }
    return projections


def _load_json(path: Path) -> Json:  # pragma: no cover - filesystem boundary
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _load_records(path: Path) -> list[Json]:  # pragma: no cover - filesystem boundary
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"jsonl_object_required:{path}")
    return values


def authenticate_schema_custody(root: Path) -> Json:
    """Recompute the exact schema, role, and eight-group custody contract."""

    artifact = _load_json(root / SCHEMA_RESULT)
    receipt = artifact.get("role_contract_receipt")
    role = dict(receipt) if isinstance(receipt, Mapping) else {}
    records = _load_records(root / PILOT_INPUT)
    identities = [str(row.get("component_hash") or "") for row in records]
    reconstructions = artifact.get("reconstruction_receipts") or []
    reconstruction_ids = [
        str(row.get("component_hash") or "")
        for row in reconstructions
        if isinstance(row, Mapping) and row.get("exact_reconstruction") is True
    ]
    expected_roles = {
        "evaluation": 40,
        "fit": 80,
        "online": 80,
        "pilot": 8,
        "policy": 20,
        "tune": 20,
    }
    expected_fit = {"anchor": 16, "optimization": 64}
    sidecars = role.get("sidecars") or []
    passed = bool(
        artifact.get("evidence_schema_ready_score") == 1
        and artifact.get("role_contract_ready_score") == 1
        and role.get("selection_salt") == "v663-evidence-20260924"
        and role.get("restored_group_count") == 480
        and role.get("selected_scored_group_count") == 240
        and role.get("role_counts") == expected_roles
        and role.get("fit_partition_counts") == expected_fit
        and role.get("pilot_disjoint") is True
        and len(records) == 8
        and len(set(identities)) == 8
        and identities == reconstruction_ids
        and sidecars
        and all(isinstance(row, Mapping) and row.get("authenticated") is True for row in sidecars)
    )
    return {
        "passed": passed,
        "selection_salt": role.get("selection_salt"),
        "restored_group_count": role.get("restored_group_count"),
        "selected_scored_group_count": role.get("selected_scored_group_count"),
        "role_counts": deepcopy(role.get("role_counts")),
        "fit_partition_counts": deepcopy(role.get("fit_partition_counts")),
        "pilot_disjoint": role.get("pilot_disjoint"),
        "pilot_component_hashes": identities,
        "schema_path": deepcopy(artifact.get("schema_path")),
        "canonical_inputs_path": deepcopy(artifact.get("canonical_inputs_path")),
        "role_manifest_path": role.get("role_manifest_path"),
        "role_manifest_sha256": role.get("role_manifest_sha256"),
    }


def gate_row(
    check: str,
    *,
    category: str,
    upstream: str,
    path: str | Path,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str = "Exact observed operands fail closed and remain auditable.",
) -> Json:
    """Keep the complete operands required for every failed gate."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "condition": f"{field} {operator} expected value",
        "governing_principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Expose every failed name and the first full diagnostic row."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind immutable content without recursively hashing the digest."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _acceptance_gates(*, valid: bool, ready: bool, retained: bool) -> list[Json]:
    values = (
        ("validity", "authenticated_inputs_and_execution", True, valid),
        ("readiness", "eight_of_eight_transport", True, ready),
        ("probability_benefit", "held_out_probability_benefit", True, False),
        ("utility", "downstream_evidence_utility", True, False),
        ("retention", "all_paired_rows_retained", True, retained),
        ("freshness", "fresh_confirmatory_groups", True, False),
    )
    return [
        gate_row(
            check,
            category=category,
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH,
            field=check,
            operator="eq",
            expected=expected,
            observed=observed,
            passed=observed == expected,
            principle="Transport, benefit, utility, retention, and freshness are independent.",
        )
        for category, check, expected, observed in values
    ]


def field_principles(keys: Sequence[str] = ()) -> dict[str, str]:
    """Give each terminal field an explicit interpretation rule."""

    principles = {
        "honest_verdict": "A complete_ prefix means terminal accounting, not benefit.",
        "verdict_class": "Only the closed six-value verdict enum is accepted.",
        "flagged_adversarial": "Persist the terminal reader result; flags open no gate.",
        "gate_check_summary": "Blocked work names exact expected and observed operands.",
        "acceptance_gate_results": "Validity, readiness, probability, utility, retention, and freshness stay separate.",
        "rows": "Each independent group-arm row keeps absolute operands and raw provenance.",
        "paired_pilot_rows": "Sixteen requests represent eight independent source groups.",
        "sample_size_budget": "Repeated views, arms, and seeds do not multiply sample size.",
        "preconditions_checked": "Only actually observed inputs and resources authenticate work.",
        "inference_substrate": "Describe current work; inherited GPU evidence is not invocation.",
        "inference_substrate_class": "Actual no-load or bounded-generation work is declared without padding.",
        "planned_inference_substrate_class": "The planned class remains visible when resources block.",
        "MODEL_SPECS": "Current model identities are empty until a current load is attempted.",
        "planned_MODEL_SPECS": "Every planned LLM request uses the mandated Qwen3.8 model.",
        "model_invoked": "True requires a current real load or generation attempt.",
        "execution_venue": "Current host, owned PID, and physical UUID bound resource claims.",
        "phase_spans": "Disjoint spans retain units, pending operations, and checkpoints.",
        "invocation_counts": "Current loads, forwards, generations, and tokens exclude history.",
        "duration_s": "Measured monotonic current work is never inherited or padded.",
        "random_seed": "Seed 7631 controls arm order and deterministic generation.",
        "reproducibility_checksum": "Immutable inputs, configuration, rows, and reduction are content-bound.",
        "source_artifact_hashes": "Inputs, pre-gates, missing evidence, and planned outputs stay distinct.",
        "validation_receipts": "Actual commands, exits, log hashes, and reader outcomes remain durable.",
        "verifier_is_oracle": "Transport validity cannot establish oracle-distinct advantage.",
        "field_principles": "Every governed field carries its interpretation in the artifact.",
        "evidence_transport_ready_score": "One requires one predeclared arm to pass all eight pointers without truncation.",
        "fit_capture_feasible_score": "One requires the unchanged 120-group role within both budgets.",
        "online_capture_feasible_score": "One requires the unchanged 80-group role within both budgets.",
        "evaluation_capture_feasible_score": "One requires the unchanged 40-group role within both budgets.",
        "selected_config_path": "The immutable selected prompt, decoder, model, and parser path is fixed.",
        "semantic_benefit_claim": "Syntax transport is not factual or predictive benefit.",
    }
    for key in keys:
        principles.setdefault(
            key, "Interpret this field only within authenticated current-run scope and gates."
        )
    return principles


def _zero_counts() -> Json:
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
    }


def _base_artifact(duration_s: float) -> Json:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7631,
        "title": "V666 paired schema transport pilot",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "flagged_adversarial": False,
        "planned_inference_substrate_class": "model_bounded_generation",
        "planned_MODEL_SPECS": MODEL_SPECS,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds": [{"seed": RANDOM_SEED, "purpose": "counterbalanced_order_and_generation"}],
        "verifier_is_oracle": False,
        "semantic_benefit_claim": False,
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "default_promotion_authorized": False,
        "generator_weights_immutable": True,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
    }


def build_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    decoder_supported: bool,
    duration_s: float,
) -> Json:
    """Build one terminal measured transport record without a benefit claim."""

    paired = [deepcopy(dict(row)) for row in rows]
    selection = select_configuration(paired, decoder_supported=decoder_supported)
    selected = [row for row in paired if row.get("arm") == selection["selected_arm"]]
    projections = project_fixed_rosters(
        selected or paired,
        model_load_s=float(runtime.get("model_load_s") or 0.0),
        validation_reserve_s=float(runtime.get("validation_reserve_s") or VALIDATION_RESERVE_S),
    )
    ready = selection["evidence_transport_ready_score"] == 1
    valid = all(row.get("passed") is True for row in preconditions) and all(
        row.get("passed") is True for row in validation_receipts
    )
    retained = len(paired) == 16
    gates = _acceptance_gates(valid=valid, ready=ready, retained=retained)
    censored = {
        str(row.get("component_hash"))
        for row in paired
        if row.get("censored") is True or row.get("pointer_valid") is not True
    }
    artifact = {
        **_base_artifact(duration_s),
        "honest_verdict": (
            "complete_null_schema_transport_ready"
            if ready
            else "complete_null_schema_transport_not_ready"
        ),
        "verdict_class": "null",
        "gate_check_summary": gate_summary(gates),
        "acceptance_gate_results": gates,
        "rows": paired,
        "paired_pilot_rows": paired,
        "sample_size_budget": {
            "independent_unit": "frozen_source_group",
            "intended": 8,
            "observed": len({str(row.get("component_hash")) for row in paired}),
            "excluded": 0,
            "censored": len(censored),
            "repeated_orders_views_seeds_multiply_sample_size": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": "owned_local_llama_cpp_paired_bounded_generation",
        "inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [deepcopy(dict(runtime.get("model_spec") or {}))],
        "model_invoked": True,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "device_uuid": runtime.get("gpu_uuid"),
            "owned_pid": runtime.get("owner_pid"),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": deepcopy(dict(runtime.get("invocation_counts") or {})),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": {},
        "evidence_transport_ready_score": int(ready),
        "fit_capture_feasible_score": projections["fit_tune_policy"]["feasible_score"],
        "online_capture_feasible_score": projections["online"]["feasible_score"],
        "evaluation_capture_feasible_score": projections["evaluation"]["feasible_score"],
        "selected_config_path": SELECTED_CONFIG_PATH.as_posix(),
        "selection_reduction": selection,
        "capture_projections": projections,
        "inference_runtime_receipt": deepcopy(dict(runtime)),
        "pilot_groups_excluded_from_learning_and_evaluation": True,
    }
    artifact["field_principles"] = field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Build an in-memory measured fixture through production reducers."""

    check = {"check": "fixture", "passed": True}
    runtime = {
        "model_load_s": 10.0,
        "validation_reserve_s": 300.0,
        "model_spec": {"hf_id": MODEL_ID, "quantization": "Q4_K_M"},
        "gpu_uuid": "GPU-test",
        "owner_pid": 123,
        "invocation_counts": {
            **_zero_counts(),
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "forward_calls_attempted": 16,
            "forward_calls_completed": 16,
            "generation_calls_attempted": 16,
            "generation_calls_completed": 16,
            "input_tokens": 1600,
            "output_tokens": 320,
        },
    }
    return build_artifact(
        rows,
        runtime=runtime,
        preconditions=[check],
        source_hashes=[],
        validation_receipts=[check],
        phase_spans=[],
        decoder_supported=True,
        duration_s=60.0,
    )


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    source_hashes: Sequence[Mapping[str, Any]] = (),
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> Json:
    """Build complete blocked work with no fabricated load or generation."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = [row for row in copied if row.get("passed") is not True]
    if not failed:
        raise ValueError("blocked_artifact_requires_failed_check")
    first = failed[0]
    gates = _acceptance_gates(valid=False, ready=False, retained=False)
    artifact = {
        **_base_artifact(duration_s),
        "honest_verdict": f"complete_blocked_{first['check']}",
        "verdict_class": "blocked",
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed),
            "failed_checks": [str(row.get("check")) for row in failed],
            "first_failure": first,
        },
        "acceptance_gate_results": gates,
        "rows": [],
        "paired_pilot_rows": [],
        "sample_size_budget": {
            "independent_unit": "frozen_source_group",
            "intended": 8,
            "observed": 0,
            "excluded": 0,
            "censored": 8,
            "repeated_orders_views_seeds_multiply_sample_size": False,
        },
        "preconditions_checked": copied,
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "device_uuid": None,
            "owned_pid": os.getpid(),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": _zero_counts(),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [],
        "terminal_reader_outcomes": {},
        "evidence_transport_ready_score": 0,
        "fit_capture_feasible_score": 0,
        "online_capture_feasible_score": 0,
        "evaluation_capture_feasible_score": 0,
        "selected_config_path": SELECTED_CONFIG_PATH.as_posix(),
        "selection_reduction": None,
        "capture_projections": {},
        "inference_runtime_receipt": {
            "planned_model_id": MODEL_ID,
            "actual_loads": 0,
            "actual_generations": 0,
            "no_model_load": True,
        },
        "pilot_groups_excluded_from_learning_and_evaluation": True,
    }
    artifact["field_principles"] = field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(  # pragma: no cover - fresh-process defensive reader
    value: object, *, require_terminal: bool = False
) -> list[str]:
    """Cold-check terminal claims, rows, reductions, and blocked diagnostics."""

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
    if artifact.get("planned_MODEL_SPECS") != MODEL_SPECS:
        errors.append("planned_model_identity_invalid")
    if artifact.get("semantic_benefit_claim") is not False:
        errors.append("semantic_benefit_claim_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not principles.get(key) for key in artifact if key != "reproducibility_checksum"
    ):
        errors.append("field_principles_incomplete")
    categories = {
        row.get("category")
        for row in artifact.get("acceptance_gate_results") or []
        if isinstance(row, Mapping)
    }
    expected_categories = {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }
    if categories != expected_categories:
        errors.append("acceptance_gate_shape_invalid")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("inference_substrate_class") != "no_model_load":
            errors.append("blocked_substrate_invalid")
        if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
            errors.append("blocked_model_identity_invalid")
        if any(value != 0 for value in (artifact.get("invocation_counts") or {}).values()):
            errors.append("blocked_invocation_counts_invalid")
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(first, Mapping) or required - set(first):
            errors.append("blocked_gate_summary_incomplete")
        return list(dict.fromkeys(errors))
    if artifact.get("inference_substrate_class") != "model_bounded_generation":
        errors.append("actual_substrate_invalid")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS or artifact.get("model_invoked") is not True:
        errors.append("current_model_identity_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 10.0:
        errors.append("bounded_generation_duration_implausible")
    rows = artifact.get("paired_pilot_rows")
    if not isinstance(rows, list) or len(rows) != 16 or artifact.get("rows") != rows:
        errors.append("paired_rows_invalid")
        rows = []
    if rows:
        runtime = artifact.get("inference_runtime_receipt") or {}
        decoder_supported = bool(runtime.get("decoder_supported", True))
        try:
            selection = select_configuration(rows, decoder_supported=decoder_supported)
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
            projections = project_fixed_rosters(
                selected or rows,
                model_load_s=float(runtime.get("model_load_s") or 0.0),
                validation_reserve_s=float(
                    runtime.get("validation_reserve_s") or VALIDATION_RESERVE_S
                ),
            )
            if artifact.get("capture_projections") != projections:
                errors.append("capture_projection_mismatch")
            score_fields = (
                ("fit_tune_policy", "fit_capture_feasible_score"),
                ("online", "online_capture_feasible_score"),
                ("evaluation", "evaluation_capture_feasible_score"),
            )
            for name, field in score_fields:
                if artifact.get(field) != projections[name]["feasible_score"]:
                    errors.append(f"{field}_mismatch")
    budget = artifact.get("sample_size_budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("intended") != 8
        or budget.get("observed") != 8
    ):
        errors.append("sample_size_budget_invalid")
    if artifact.get("selected_config_path") != SELECTED_CONFIG_PATH.as_posix():
        errors.append("selected_config_path_invalid")
    if require_terminal:
        names = {
            str(row.get("name"))
            for row in artifact.get("validation_receipts") or []
            if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
        }
        required_names = {
            *validation_scope.REQUIRED_CHECK_NAMES,
            "fresh_process_cold_replay",
            "independent_reduction",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        }
        if not required_names <= names:
            errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> Json:  # pragma: no cover - fresh process reader
    """Reload an exact candidate and validate it without model work."""

    artifact = _load_json(path)
    errors = validate_artifact(artifact)
    return {
        "mode": "cold_replay",
        "passed": not errors,
        "errors": errors,
        "candidate_sha256": current_work_receipt.sha256_file(path),
        "paired_row_count": len(artifact.get("paired_pilot_rows") or []),
    }


def independent_reduce_artifact(path: Path) -> Json:
    """Recompute arm comparison from persisted rows, never stored scores."""

    artifact = _load_json(path)
    errors = validate_artifact(artifact)
    rows = artifact.get("paired_pilot_rows") or []
    if artifact.get("verdict_class") == "blocked":
        return {
            "mode": "independent_reduction",
            "passed": not errors and rows == [],
            "errors": errors,
            "blocked_without_fabricated_rows": rows == [],
            "row_reduction_sha256": canonical_hash(rows),
        }
    runtime = artifact.get("inference_runtime_receipt") or {}
    try:
        selection = select_configuration(
            rows, decoder_supported=bool(runtime.get("decoder_supported", True))
        )
    except (TypeError, ValueError) as exc:
        return {"mode": "independent_reduction", "passed": False, "errors": [str(exc)]}
    return {
        "mode": "independent_reduction",
        "passed": not errors and selection == artifact.get("selection_reduction"),
        "errors": errors,
        "selection": selection,
        "row_reduction_sha256": canonical_hash(rows),
    }


def _source_row(path: Path, *, producer: str, source_class: str) -> Json:  # pragma: no cover
    return {
        "path": str(path.resolve()),
        "producer": producer,
        "source_class": source_class,
        "bytes": path.stat().st_size,
        "sha256": current_work_receipt.sha256_file(path),
    }


def collect_preconditions(  # pragma: no cover - host and filesystem boundary
    root: Path,
) -> tuple[list[Json], list[Json], Json]:
    """Authenticate declared inputs, schema custody, model bytes, and owned capacity."""

    named = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7616_v665_evidence_schema.py"),
        Path("python/carnot/experiment_7617_v665_schema_pilot.py"),
        Path("python/carnot/experiment_7630_v666_cuda_ownership.py"),
        SCHEMA_RESULT,
        Path("results/experiment_7617_v665_schema_pilot.json"),
        OWNERSHIP_RESULT,
        PILOT_INPUT,
        Path("results/raw/experiment_7616_v665_evidence_schema/schema_authority.json"),
        Path("results/raw/experiment_7616_v665_evidence_schema/canonical_inputs.json"),
        Path("results/raw/experiment_7630_v666_cuda_ownership/launch_config.json"),
        SPEC_PATH,
    )
    checks: list[Json] = []
    hashes: list[Json] = []
    context: Json = {}
    for relative in named:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"named_input:{relative.name}",
                category="validity",
                upstream="declared_task_input",
                path=path,
                field="readable_nonempty_file",
                operator="eq",
                expected=True,
                observed=present,
                passed=present,
            )
        )
        if present:
            hashes.append(
                _source_row(path, producer="declared_task_input", source_class="pre_gate_input")
            )
    if not all(row["passed"] for row in checks):
        return checks, hashes, context
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        gate_row(
            "driving_requirement",
            category="validity",
            upstream="research_reporting_spec",
            path=root / SPEC_PATH,
            field="REQ-*",
            operator="contains",
            expected="REQ-REPORT-7631",
            observed="REQ-REPORT-7631" if "REQ-REPORT-7631" in spec_text else None,
            passed="REQ-REPORT-7631" in spec_text,
        )
    )
    custody = authenticate_schema_custody(root)
    checks.append(
        gate_row(
            "schema_role_custody",
            category="validity",
            upstream="experiment_7616_v665_evidence_schema",
            path=root / SCHEMA_RESULT,
            field="schema_role_and_eight_group_custody",
            operator="eq",
            expected=True,
            observed=custody["passed"],
            passed=custody["passed"] is True,
        )
    )
    launch = _load_json(root / OWNERSHIP_RESULT)
    launch_ready = bool(
        launch.get("launch_protocol_ready_score") == 1
        and launch.get("flagged_adversarial") is False
    )
    checks.append(
        gate_row(
            "cuda_ownership_protocol",
            category="validity",
            upstream="experiment_7630_v666_cuda_ownership",
            path=root / OWNERSHIP_RESULT,
            field="launch_protocol_ready_score_and_flag",
            operator="eq",
            expected={"ready": 1, "flagged": False},
            observed={
                "ready": launch.get("launch_protocol_ready_score"),
                "flagged": launch.get("flagged_adversarial"),
            },
            passed=launch_ready,
        )
    )
    model = cached_current_model(preferred_quant="Q4_K_M")
    declared_model_path = Path(str((model or {}).get("model_path") or root / "missing"))
    model_path = declared_model_path.resolve()
    quantization = "Q4_K_M" if "Q4_K_M" in declared_model_path.name else None
    cache_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and quantization == "Q4_K_M"
        and model_path.is_file()
        and model_path.stat().st_size > 15_000_000_000
    )
    model_sha = current_work_receipt.sha256_file(model_path) if cache_ok else None
    checks.append(
        gate_row(
            "cached_q4_k_m_model_hash",
            category="readiness",
            upstream="cached_current_model",
            path=model_path,
            field="hf_id_quantization_sha256",
            operator="authenticated",
            expected={"hf_id": MODEL_ID, "quantization": "Q4_K_M"},
            observed={
                "hf_id": (model or {}).get("hf_id"),
                "quantization": quantization,
                "sha256": model_sha,
            },
            passed=cache_ok,
        )
    )
    registry = ownership.ProcessRegistry.current()
    inventory = ownership._current_inventory()
    selected, ownership_rows = ownership.select_owned_capacity(inventory, registry)
    checks.append(
        gate_row(
            "owned_cuda_capacity",
            category="readiness",
            upstream="nvidia-smi_and_exp7630_launcher",
            path=root / RAW_DIR,
            field="owned_idle_device_with_20000_mb",
            operator="eq",
            expected=True,
            observed={
                "available": selected is not None,
                "inventory": inventory,
                "ownership_rows": ownership_rows,
            },
            passed=selected is not None,
            principle="Foreign compute and insufficient free VRAM block before model work.",
        )
    )
    context.update(
        custody=custody,
        model_spec={**(model or {}), "quantization": quantization},
        model_path=str(model_path),
        model_sha256=model_sha,
        registry=registry,
        inventory=inventory,
        selected_gpu=selected,
    )
    return checks, hashes, context


def affected_validation_manifest() -> Json:  # pragma: no cover - validation wiring
    """Freeze the exact files admitted to changed-behavior validation."""

    return {
        "requirement": "REQ-REPORT-7631",
        "tests": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }


def build_validation_commands(  # pragma: no cover - validation wiring
    root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:
    manifest = affected_validation_manifest()
    basetemp = private_root / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        manifest["tests"],
        manifest["changed_modules"],
        static_paths=manifest["static_paths"],
        basetemp=basetemp,
        coverage_file=private_root / ".coverage.exp7631",
    )


def terminal_commands(  # pragma: no cover - validation wiring
    root: Path, candidate: Path
) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "persisted_rows_only",
            180.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
    ]


def _span(  # pragma: no cover - live monotonic receipt
    phase: str,
    *,
    task_started: float,
    phase_started: float,
    planned: int,
    completed: int,
    pending: str | None = None,
) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - task_started,
        "ended_offset_s": ended - task_started,
        "duration_s": ended - phase_started,
        "planned_units": planned,
        "completed_units": completed,
        "pending_operations": [pending] if pending else [],
        "checkpoint_position": completed,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:  # pragma: no cover
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in receipts
    )


def _normalize_receipts(  # pragma: no cover - path normalization
    receipts: Sequence[Mapping[str, Any]], root: Path
) -> list[Json]:
    normalized: list[Json] = []
    for row in receipts:
        copied = deepcopy(dict(row))
        log_path = Path(str(copied.get("log_path") or ""))
        if log_path.is_absolute() and log_path.is_relative_to(root):
            copied["log_path"] = log_path.relative_to(root).as_posix()
        normalized.append(copied)
    return normalized


def _refresh(artifact: Json) -> None:  # pragma: no cover - terminal mutation
    artifact["field_principles"] = field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _reader_outcomes(receipts: Sequence[Mapping[str, Any]]) -> Json:  # pragma: no cover
    return {
        str(row.get("name")): {
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
    }


def _live_measurement(  # pragma: no cover - live model/GPU integration
    root: Path,
    context: Mapping[str, Any],
    *,
    started: float,
) -> tuple[list[Json], Json]:
    """Acquire, recheck, qualify in a CPU child, then reuse the paired transport."""

    from carnot.gpu_lease_phase_journal import GpuLease

    selected = dict(context["selected_gpu"])
    registry = context["registry"]
    lease_root = root / RAW_DIR / "gpu_leases"
    progress(started, "gpu_lease", "before", gpu_uuid=selected["uuid"])
    lease = GpuLease.acquire(
        runtime_dir=lease_root,
        task_id=EXPERIMENT_ID,
        device_uuid=str(selected["uuid"]),
        expected_model=str(context["model_path"]),
        vram_before_mb=int(selected["memory_used_mb"]),
        ttl_s=4800.0,
    )
    progress(started, "gpu_lease", "after", lease_id=lease.lease_id)
    snapshots = [ownership._current_inventory(), ownership._current_inventory()]
    recheck = ownership.recheck_before_launch(str(selected["uuid"]), snapshots, registry)
    if recheck.get("passed") is not True:
        lease.transition("terminal_blocked")
        lease.release()
        raise RuntimeError(f"owned_capacity_recheck_failed:{recheck.get('reason')}")
    tokenizer_code = (
        "import json,sys; "
        "from carnot.inference.sota_models import gguf_tokenizer_loadable; "
        "ok,detail=gguf_tokenizer_loadable(sys.argv[1]); "
        "print(json.dumps({'passed':ok,'detail':detail}),flush=True); "
        "raise SystemExit(0 if ok else 1)"
    )
    progress(started, "tokenizer_cpu_child", "before")
    tokenizer = subprocess.run(
        [str(root / ".venv/bin/python"), "-u", "-c", tokenizer_code, str(context["model_path"])],
        cwd=root,
        env=ownership.tokenizer_qualification_environment(),
        capture_output=True,
        text=True,
        timeout=180.0,
        check=False,
    )
    progress(started, "tokenizer_cpu_child", "after", exit_code=tokenizer.returncode)
    if tokenizer.returncode != 0:
        lease.transition("terminal_blocked")
        lease.release()
        raise RuntimeError("embedded_tokenizer_cpu_child_failed")
    records = _load_records(root / PILOT_INPUT)
    prior_pilot.RANDOM_SEED = RANDOM_SEED
    prior_pilot.ARM_ORDER_SEED = RANDOM_SEED
    prior_pilot.VALID_THRESHOLD = 8
    run_dir = root / RAW_DIR / "runs" / f"{int(time.time())}-{os.getpid()}"
    model_spec = dict(context["model_spec"])
    model_spec["gpu"] = int(selected["index"])
    progress(started, "paired_pilot", "before", units=16)
    old_rows, runtime, _receipt, _e2e = prior_pilot.run_owned_pilot(
        root=root,
        records=records,
        model_spec=model_spec,
        model_sha256=str(context["model_sha256"]),
        selected_gpu=selected,
        lease=lease,
        started=started,
        run_dir=run_dir,
    )
    mapped_rows: list[Json] = []
    arm_map = {
        prior_pilot.CONTROL_ARM: EXPLICIT_ARM,
        prior_pilot.GRAMMAR_ARM: CONSTRAINED_ARM,
    }
    for row in old_rows:
        copied = deepcopy(row)
        copied["arm"] = arm_map[str(row["arm"])]
        parsed = parse_pilot_response(
            copied.get("input_record") or {},
            str(copied.get("response_text") or ""),
            finish_reason=str(copied.get("finish_reason") or "unknown"),
        )
        copied.update(
            request_s=float(copied.get("generation_s") or 0.0),
            first_token_latency_s=float(copied.get("prefill_s") or 0.0),
            tokenization_s=float(copied.get("prefill_s") or 0.0),
            grammar_compile_s=0.0,
            raw_request_bytes=int(copied.get("request_bytes") or 0),
            raw_response_bytes=int(copied.get("response_bytes") or 0),
            raw_request_sha256=copied.get("request_sha256"),
            raw_response_sha256=copied.get("response_sha256"),
            parser_result=parsed,
            schema_valid=parsed["schema_valid"],
            pointer_valid=parsed["pointer_valid"],
            independent_rejection_reasons=parsed["rejection_reasons"],
        )
        mapped_rows.append(copied)
    runtime["decoder_supported"] = True
    runtime["model_spec"] = model_spec
    runtime["gpu_uuid"] = selected["uuid"]
    runtime["owner_pid"] = os.getpid()
    runtime["validation_reserve_s"] = VALIDATION_RESERVE_S
    progress(started, "paired_pilot", "after", completed=len(mapped_rows))
    return mapped_rows, runtime


def run_experiment(  # pragma: no cover - declared capability E2E
    root: Path, run_date: str, output: Path
) -> int:
    """Authenticate, measure or block, validate, and publish atomically."""

    resolved = root.resolve()
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    destination = output if output.is_absolute() else resolved / output
    started = time.monotonic()
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7631-"))
    spans: list[Json] = []
    progress(started, "preconditions", "before", root=resolved)
    phase_started = time.monotonic()
    checks, source_hashes, context = collect_preconditions(resolved)
    spans.append(
        _span(
            "preconditions",
            task_started=started,
            phase_started=phase_started,
            planned=len(checks),
            completed=sum(row.get("passed") is True for row in checks),
            pending=(
                str(next(row["check"] for row in checks if row.get("passed") is not True))
                if any(row.get("passed") is not True for row in checks)
                else None
            ),
        )
    )
    preconditions_ok = all(row.get("passed") is True for row in checks)
    progress(started, "preconditions", "after", passed=preconditions_ok)
    if preconditions_ok:
        phase_started = time.monotonic()
        try:
            rows, runtime = _live_measurement(resolved, context, started=started)
        except BaseException as exc:
            checks.append(
                gate_row(
                    "owned_model_transport",
                    category="readiness",
                    upstream="exp7630_owned_launcher",
                    path=resolved / RAW_DIR,
                    field="owned_load_and_sixteen_requests",
                    operator="completed",
                    expected=True,
                    observed=f"{type(exc).__name__}:{exc}",
                    passed=False,
                )
            )
            artifact = build_blocked_artifact(
                checks,
                duration_s=time.monotonic() - started,
                source_hashes=source_hashes,
                phase_spans=spans,
            )
        else:
            spans.append(
                _span(
                    "paired_measurement",
                    task_started=started,
                    phase_started=phase_started,
                    planned=16,
                    completed=len(rows),
                )
            )
            selection = select_configuration(rows, decoder_supported=True)
            selected_config = {
                "schema": "carnot.exp7631.selected_config.v1",
                "selected_arm": selection["selected_arm"],
                "model_id": MODEL_ID,
                "model_sha256": context["model_sha256"],
                "temperature": 0.0,
                "seed": RANDOM_SEED,
                "max_tokens": MAX_TOKENS,
                "schema_path": context["custody"]["schema_path"],
                "pilot_component_hashes": context["custody"]["pilot_component_hashes"],
            }
            current_work_receipt.atomic_json(resolved / SELECTED_CONFIG_PATH, selected_config)
            source_hashes.append(
                _source_row(
                    resolved / SELECTED_CONFIG_PATH,
                    producer=EXPERIMENT_ID,
                    source_class="current_frozen_output",
                )
            )
            artifact = build_artifact(
                rows,
                runtime=runtime,
                preconditions=checks,
                source_hashes=source_hashes,
                validation_receipts=[],
                phase_spans=spans,
                decoder_supported=True,
                duration_s=time.monotonic() - started,
            )
    else:
        artifact = build_blocked_artifact(
            checks,
            duration_s=time.monotonic() - started,
            source_hashes=[
                *source_hashes,
                {
                    "path": str(destination),
                    "producer": EXPERIMENT_ID,
                    "source_class": "planned_output_not_a_precondition",
                    "present_before_publication": False,
                },
            ],
            phase_spans=spans,
        )
    progress(started, "scoped_validation", "before")
    phase_started = time.monotonic()
    validation = validation_scope.run_commands(
        resolved,
        build_validation_commands(resolved, private_root),
        log_dir=private_root / "logs" / "scoped",
        heartbeat_s=60.0,
    )
    progress(started, "scoped_validation", "after", passed=_receipts_pass(validation))
    if not _receipts_pass(validation):
        return 1
    spans.append(
        _span(
            "scoped_validation",
            task_started=started,
            phase_started=phase_started,
            planned=len(validation),
            completed=len(validation),
        )
    )
    artifact["validation_receipts"] = _normalize_receipts(validation, resolved)
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    _refresh(artifact)
    candidate = private_root / "terminal_candidate.json"
    current_work_receipt.atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before")
    terminal = validation_scope.run_commands(
        resolved,
        terminal_commands(resolved, candidate),
        log_dir=private_root / "logs" / "terminal",
        heartbeat_s=60.0,
    )
    progress(started, "terminal_readers", "after", passed=_receipts_pass(terminal))
    if not _receipts_pass(terminal):
        return 1
    artifact["validation_receipts"] = [
        *artifact["validation_receipts"],
        *_normalize_receipts(terminal, resolved),
    ]
    artifact["terminal_reader_outcomes"] = _reader_outcomes(terminal)
    artifact["flagged_adversarial"] = False
    artifact["duration_s"] = time.monotonic() - started
    _refresh(artifact)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"publication_blocked": errors}, sort_keys=True), flush=True)
        return 1
    current_work_receipt.atomic_json(
        resolved / RAW_DIR / "terminal_reader_outcomes.json", artifact["terminal_reader_outcomes"]
    )
    current_work_receipt.atomic_json(destination, artifact)
    progress(
        started,
        "publish",
        "after",
        path=destination,
        sha256=current_work_receipt.sha256_file(destination),
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse production and read-only terminal modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary
    """Run the producer or one exact read-only candidate operation."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        result = cold_replay(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    if args.independent_reduce is not None:
        result = independent_reduce_artifact(args.independent_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    return run_experiment(ROOT, args.date, args.output)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
