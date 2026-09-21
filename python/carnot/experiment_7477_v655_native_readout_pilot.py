"""Measure native Qwen option-logit transport without generating tokens.

The pilot checks the boundary between a complete prompt and two option tokens.
It does not use prediction accuracy as a transport requirement.

Spec refs: REQ-VERIFY-7477 and SCENARIO-VERIFY-7477-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as option_protocol
from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]

RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7477-native-readout-pilot"
SCHEMA = "carnot.exp7477.v655.native_readout_pilot.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_native_llama_cpp_raw_logit_readout_no_generation"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
MAX_LIVE_SECONDS = 600.0
FORWARD_BUDGET = 28
CAPTURE_HARD_CAP_S = 3300.0
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7477_v655_native_readout_pilot.json")
RAW_DIR = Path("results/raw/experiment_7477_v655_native_readout_pilot")
MODULE_PATH = Path("python/carnot/experiment_7477_v655_native_readout_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7477_v655_native_readout_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7477_v655_native_readout_pilot.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
V654_GROUPS_PATH = Path("results/raw/experiment_7462_v654_option_protocol/cohort_groups.jsonl")
V655_QUALIFICATION_PATH = Path("results/experiment_7476_v655_option_qualification.json")
V654_PARITY_PATH = Path("results/experiment_7463_v654_semif_e0_logprob_parity.json")
CAPTURE_LIFECYCLE_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("scripts/jevbench_readout_eval.py"),
    Path("python/carnot/experiment_7463_v654_semif_e0_logprob_parity.py"),
    V654_PARITY_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_7448_v653_capture_lifecycle.py"),
    CAPTURE_LIFECYCLE_PATH,
    V655_QUALIFICATION_PATH,
    V654_GROUPS_PATH,
    Path("openspec/capabilities/verification/spec.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks and also emit lowercase model_specs.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name the actual native raw-logit readout without claiming generation or parity.",
    "inference_substrate_class": "Use model_load_no_generation because the model loads and evaluates prompts but generates zero tokens.",
    "execution_venue": "Use host and record actual CPU and CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding; separate load, forward, generation, numeric work and validation.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering and audit seeds; the deterministic transport reducer uses no outcome-selected seed.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw rows and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags and classes.",
    "rows": "One row per forward includes failures and censoring so difficult prompts cannot disappear.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted units.",
    "acceptance_gate_results": "Each check carries category, expected, observed, operator, result and a failure-prevention principle.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream field, expected value and observed value.",
    "honest_verdict": "Use a complete terminal finding while keeping predictive benefit and scored deployment deferred.",
    "verdict_class": "Use the closed verdict enum; this valid transport-only result is a scientific null.",
    "verifier_is_oracle": "False prevents transport fixtures from becoming positive predictive evidence.",
    "flagged_adversarial": "Keep real reader flags; never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exit codes, log hashes and required status establish validation scope.",
    "field_principles": "Explain why each required field exists so the evidence stands alone.",
    "native_readout_ready_score": "Bare 0 or 1 for live native transport; accuracy is not a validity precondition.",
    "prompt_identity": "Exact model, template, token IDs and logit positions make scores reproducible.",
    "pilot_cost_rows": "Per-prompt prefill and load costs support honest capture feasibility.",
    "scored_runtime_parity_score": "Remain 0 because no exact scored-runtime comparison occurs.",
}
REQUIRED_PRINCIPLE_FIELDS = frozenset(FIELD_PRINCIPLES)


def _development_inputs() -> list[tuple[str, str]]:
    """Return label-free source and response text frozen before model outcomes."""

    return [
        ("The observatory opened in 1912.", "The observatory opened in 1912."),
        ("The archive contains blue and green maps.", "The archive contains red maps."),
        ("Mira bought two pears and one plum.", "Mira bought three pieces of fruit."),
        ("The north gate closes at 18:00.", "The north gate closes at 19:00."),
        ("Water freezes at zero degrees Celsius.", "Water freezes at zero degrees Celsius."),
        ("The report names Ada as the project lead.", "Ada leads the project."),
        ("The train stops at Elm, Oak, and Pine.", "The train stops at Cedar."),
        ("Copper conducts electricity.", "Copper is an electrical conductor."),
        ("The box weighs four kilograms.", "The box weighs fourteen kilograms."),
        ("Ravi sent the letter on Tuesday.", "Ravi sent the letter on Tuesday."),
        ("The garden has six rose bushes.", "The garden has six tulip bushes."),
        ("The permit expires in November.", "The permit expires in November."),
    ]


def _freeze_groups() -> list[JsonDict]:
    groups: list[JsonDict] = []
    for index, (source, response) in enumerate(_development_inputs()):
        inputs = {"source": source, "response": response}
        group_hash = canonical_hash(inputs)
        groups.append(
            {
                "group_id": f"exp7477-development-{index:02d}-{group_hash[-12:]}",
                "group_hash": group_hash,
                "source": source,
                "response": response,
                "prompt_inputs": inputs,
                "role": "development_transport_outside_sealed_cohort",
            }
        )
    return groups


_DEVELOPMENT_GROUPS = _freeze_groups()
DEVELOPMENT_GROUPS_SHA256 = canonical_hash(_DEVELOPMENT_GROUPS)


def freeze_development_groups() -> list[JsonDict]:
    """Return a copy so a caller cannot alter the registered prompt panel."""

    return deepcopy(_DEVELOPMENT_GROUPS)


def controlled_transport_fixture() -> list[JsonDict]:
    """Build four valid rows that exercise mapping, position, diversity, and reset."""

    vectors = ((4.0, 1.0), (1.0, 5.0), (2.0, 3.5), (4.0, 1.0))
    orders = (
        option_protocol.OPTION_IDS,
        tuple(reversed(option_protocol.OPTION_IDS)),
        option_protocol.OPTION_IDS,
        tuple(reversed(option_protocol.OPTION_IDS)),
    )
    rows: list[JsonDict] = []
    for index, (vector, order) in enumerate(zip(vectors, orders, strict=True)):
        prompt_ids = [1, 20 + index, 30 + index]
        label_to_option = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=True))
        raw = dict(zip(order, vector, strict=True))
        rows.append(
            {
                "call_id": f"fixture-{index}",
                "row_kind": "transport_control",
                "disposition": "complete",
                "option_ids_in_prompt": list(order),
                "display_labels": list(option_protocol.DISPLAY_LABELS),
                "label_token_ids": [11, 17],
                "label_to_option_id": label_to_option,
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "requested_score_position": len(prompt_ids) - 1,
                "actual_last_evaluated_position": len(prompt_ids) - 1,
                "token_boundary_receipts": {" A": True, " B": True},
                "raw_logits_by_option_id": raw,
                "prefill_s": 0.05 + index / 100.0,
                "generated_tokens": 0,
                "state_reset": True,
                "kv_state_id": f"fresh-{index}",
                "prediction_correct": index % 2 == 0,
                "predicted_option_id": max(raw, key=raw.__getitem__),
            }
        )
    return rows


def mutate_transport_fixture(rows: Sequence[Mapping[str, Any]], mutation: str) -> list[JsonDict]:
    """Apply one named defect so tests use the same production reducer."""

    altered = deepcopy([dict(row) for row in rows])
    if mutation == "label_swap":
        altered[0]["label_to_option_id"] = {
            " A": "contains_unsupported",
            " B": "supported",
        }
    elif mutation == "token_position":
        altered[0]["actual_last_evaluated_position"] += 1
    elif mutation == "constant_vector":
        vector = deepcopy(altered[0]["raw_logits_by_option_id"])
        for row in altered:
            row["raw_logits_by_option_id"] = deepcopy(vector)
    elif mutation == "state_reuse":
        altered[1]["kv_state_id"] = altered[0]["kv_state_id"]
    else:
        raise ValueError(f"unknown_mutation:{mutation}")
    return altered


def reduce_transport_rows(rows: Sequence[Mapping[str, Any]], *, expected_forwards: int) -> JsonDict:
    """Reduce raw rows without treating prediction quality as transport validity."""

    complete = [row for row in rows if row.get("disposition") == "complete"]
    completeness_valid = len(rows) == len(complete) == expected_forwards
    option_set = set(option_protocol.OPTION_IDS)
    label_mapping_valid = True
    token_position_valid = True
    finite_logits_valid = True
    token_boundaries_valid = True
    zero_generation_valid = True
    vectors: list[tuple[float, float]] = []
    states: list[str] = []
    uniform_rows = 0
    for row in complete:
        order = list(row.get("option_ids_in_prompt") or [])
        mapping = row.get("label_to_option_id")
        expected_mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=False))
        label_mapping_valid &= (
            len(order) == 2
            and set(order) == option_set
            and mapping == expected_mapping
            and len(set(row.get("label_token_ids") or [])) == 2
        )
        expected_position = int(row.get("prompt_token_count", 0)) - 1
        token_position_valid &= (
            row.get("requested_score_position") == expected_position
            and row.get("actual_last_evaluated_position") == expected_position
        )
        boundaries = row.get("token_boundary_receipts")
        token_boundaries_valid &= isinstance(boundaries, Mapping) and all(
            boundaries.get(label) is True for label in option_protocol.DISPLAY_LABELS
        )
        logits = row.get("raw_logits_by_option_id")
        if isinstance(logits, Mapping) and set(logits) == option_set:
            values = tuple(float(logits[key]) for key in option_protocol.OPTION_IDS)
            finite_logits_valid &= all(math.isfinite(value) for value in values)
            uniform_rows += int(math.isclose(values[0], values[1], abs_tol=1e-12))
            vectors.append(values)
        else:
            finite_logits_valid = False
        zero_generation_valid &= row.get("generated_tokens") == 0
        states.append(str(row.get("kv_state_id")))
    constant_stub_absent = len(set(vectors)) > 1
    fresh_state_valid = len(states) == len(set(states)) == len(complete) and all(
        row.get("state_reset") is True for row in complete
    )
    checks = {
        "completeness_valid": completeness_valid,
        "label_mapping_valid": label_mapping_valid,
        "token_position_valid": token_position_valid,
        "finite_logits_valid": finite_logits_valid,
        "token_boundaries_valid": token_boundaries_valid,
        "constant_stub_absent": constant_stub_absent,
        "fresh_state_valid": fresh_state_valid,
        "zero_generation_valid": zero_generation_valid,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "planned_forwards": expected_forwards,
        "observed_rows": len(rows),
        "complete_forwards": len(complete),
        "uniform_rows": uniform_rows,
        "inaccurate_rows_observed": sum(row.get("prediction_correct") is False for row in complete),
        "predictive_accuracy_is_gate": False,
        "real_order_agreement_is_gate": False,
    }


def project_capture_costs(prefill_s: Sequence[float], *, load_s: float) -> list[JsonDict]:
    """Project fixed downstream rosters without changing their hard caps."""

    if not prefill_s or any(not math.isfinite(value) or value <= 0 for value in prefill_s):
        raise ValueError("positive_finite_prefill_costs_required")
    median_s = float(statistics.median(prefill_s))
    plans = (
        ("fit_capture", {"training": 180, "calibration_tuning": 60}, 520),
        ("evaluation_capture", {"internal_test": 60, "online": 160, "external": 74}, 628),
    )
    available = max(0, math.floor((CAPTURE_HARD_CAP_S - load_s) / median_s))
    rows: list[JsonDict] = []
    for name, roster, planned in plans:
        completed = min(planned, available)
        projected_s = float(load_s + planned * median_s)
        rows.append(
            {
                "capture_shard": name,
                "group_roster": roster,
                "planned_forwards": planned,
                "hard_cap_s": CAPTURE_HARD_CAP_S,
                "load_s": float(load_s),
                "median_prefill_s": median_s,
                "projected_s": projected_s,
                "feasible": projected_s <= CAPTURE_HARD_CAP_S,
                "projected_completed_forwards": completed,
                "unstarted_forwards": planned - completed,
                "unstarted_cells": [
                    f"{name}-forward-{index:04d}" for index in range(completed, planned)
                ],
            }
        )
    return rows


def fixture_invocation_events(*, forwards: int) -> list[JsonDict]:
    """Build one balanced event ledger for reducer and artifact tests."""

    events = [
        {"call_id": "load", "operation": "model_loads", "state": "attempted"},
        {"call_id": "load", "operation": "model_loads", "state": "completed"},
    ]
    for index in range(forwards):
        events.extend(
            (
                {"call_id": f"forward-{index}", "operation": "forward_calls", "state": "attempted"},
                {"call_id": f"forward-{index}", "operation": "forward_calls", "state": "completed"},
            )
        )
    return events


def reduce_invocation_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count terminal and unfinished operations from the owned current ledger."""

    counts: JsonDict = {}
    for operation in ("model_loads", "forward_calls", "generation_calls"):
        attempted_ids = {
            str(row.get("call_id"))
            for row in events
            if row.get("operation") == operation and row.get("state") == "attempted"
        }
        terminal: dict[str, str] = {}
        for row in events:
            if row.get("operation") == operation and row.get("state") in {
                "completed",
                "failed",
                "cancelled",
            }:
                terminal[str(row.get("call_id"))] = str(row.get("state"))
        operation_counts = {"attempted": len(attempted_ids)}
        for state in ("completed", "failed", "cancelled"):
            operation_counts[state] = sum(value == state for value in terminal.values())
        operation_counts["in_flight"] = len(attempted_ids - set(terminal))
        counts[operation] = operation_counts
    return counts


def invocation_counts_balanced(counts: Mapping[str, Any]) -> bool:
    """Require each attempt to have exactly one terminal outcome and no generation."""

    for operation in ("model_loads", "forward_calls", "generation_calls"):
        operation_counts = counts.get(operation)
        if not isinstance(operation_counts, Mapping):
            return False
        attempted = int(operation_counts.get("attempted", -1))
        terminal = sum(
            int(operation_counts.get(state, -1))
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        if attempted != terminal:
            return False
    return counts["generation_calls"].get("attempted") == 0 and all(
        counts[operation].get("in_flight") == 0
        for operation in ("model_loads", "forward_calls", "generation_calls")
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    """Create one gate with operands and the failure mode it prevents."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": principle,
    }


def build_acceptance_gates(
    *, transport_passed: bool, validation_passed: bool, benefit_measured: bool
) -> list[JsonDict]:
    """Keep evidence validity, transport readiness, and benefit independent."""

    valid = transport_passed and validation_passed
    return [
        _gate(
            "authenticated_transport",
            "required_validity",
            True,
            valid,
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "native_readout_ready",
            "readiness",
            True,
            valid,
            "A valid null must not suppress an independent measurement.",
        ),
        _gate(
            "scientific_benefit_deferred",
            "scientific_benefit",
            False,
            benefit_measured,
            "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
    ]


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed check while preserving the complete gate list."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"passed": True, "failed_check": None, "failed_count": 0}
    first = failed[0]
    return {
        "passed": False,
        "failed_check": first.get("check"),
        "category": first.get("category"),
        "expected": first.get("expected"),
        "observed": first.get("observed"),
        "failed_count": len(failed),
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the complete artifact without its self-referential checksum."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _fixture_prompt_identity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "model_id": MODEL_HF_ID,
        "tokenizer": "embedded_gguf",
        "chat_template_bytes_base64": "",
        "chat_template_sha256": canonical_hash(""),
        "development_groups_sha256": DEVELOPMENT_GROUPS_SHA256,
        "prompt_token_ids_by_call": {
            str(row["call_id"]): deepcopy(row["prompt_token_ids"]) for row in rows
        },
        "actual_last_evaluated_position_by_call": {
            str(row["call_id"]): row["actual_last_evaluated_position"] for row in rows
        },
    }


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    rows = controlled_transport_fixture()
    reduction = reduce_transport_rows(rows, expected_forwards=4)
    events = fixture_invocation_events(forwards=4)
    counts = reduce_invocation_events(events)
    gates = build_acceptance_gates(
        transport_passed=reduction["passed"] and invocation_counts_balanced(counts),
        validation_passed=True,
        benefit_measured=False,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_native_readout_transport_ready_predictive_benefit_deferred",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-21T00:00:00Z",
        "ended_at_utc": "2026-09-21T00:00:03Z",
        "started_monotonic_ns": 1_000_000_000,
        "ended_monotonic_ns": 4_000_000_000,
        "clock_identity": {"wall": "datetime.now(UTC)", "monotonic": "time.monotonic_ns", "pid": 1},
        "preconditions_checked": [
            {"check": "fixture", "expected": True, "observed": True, "passed": True}
        ],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": deepcopy(MODEL_SPECS),
        "model_invoked": True,
        "invocation_counts": counts,
        "current_invocation_events": events,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": {"cpu": "fixture", "cuda": [{"uuid": "fixture-gpu"}]},
        "duration_s": 3.0,
        "duration_components_s": {
            "model_load": 2.0,
            "forward": 1.0,
            "generation": 0.0,
            "validation": 0.0,
        },
        "phase_spans": [{"phase": "fixture", "start_s": 0.0, "end_s": 3.0, "completed_units": 4}],
        "random_seed": {"ordering": 65_500, "audit": 65_509, "bootstrap": None},
        "source_artifact_hashes": {},
        "rows": rows,
        "sample_size_budget": {
            "planned": 4,
            "attempted": 4,
            "complete": 4,
            "failed": 0,
            "censored": 0,
            "excluded": 0,
            "unstarted": 0,
        },
        "transport_reduction": reduction,
        "independent_reduction": deepcopy(reduction),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": "complete_null_native_readout_transport_ready_predictive_benefit_deferred",
        "verdict_class": "null",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "native_readout_ready_score": 1,
        "prompt_identity": _fixture_prompt_identity(rows),
        "pilot_cost_rows": [
            {"kind": "model_load", "elapsed_s": 2.0},
            *[
                {"kind": "prompt_prefill", "call_id": row["call_id"], "elapsed_s": row["prefill_s"]}
                for row in rows
            ],
        ],
        "capture_feasibility": project_capture_costs(
            [float(row["prefill_s"]) for row in rows], load_s=2.0
        ),
        "scored_runtime_parity_score": 0,
        "server_runtime_parity_claimed": False,
        "blackwell_parity_claimed": False,
        "e0_disposition": "deferred_not_reopened",
        "scientific_benefit_measured": False,
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "fixture",
            "numbered_runtime_e2e": "not_applicable_isolated_readout_experiment",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "promotion_score": 0,
    }
    artifact["field_principles"] = deepcopy(FIELD_PRINCIPLES)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validation_names_passed(receipts: object, names: Sequence[str]) -> bool:
    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    return all(
        sum(row.get("name") == name and row.get("passed") is True for row in rows) == 1
        for name in names
    )


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, row reduction, readiness, counters, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "model_invoked": True,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "scored_runtime_parity_score": 0,
        "server_runtime_parity_claimed": False,
        "blackwell_parity_claimed": False,
        "scientific_benefit_measured": False,
        "verifier_is_oracle": False,
    }
    for key, wanted in expected.items():
        if artifact.get(key) != wanted:
            errors.append(f"{key}_mismatch")
    rows = artifact.get("rows")
    budget = artifact.get("sample_size_budget")
    expected_forwards = int(budget.get("planned", -1)) if isinstance(budget, Mapping) else -1
    if not isinstance(rows, list):
        errors.append("rows_invalid")
    else:
        reduced = reduce_transport_rows(
            [row for row in rows if isinstance(row, Mapping)],
            expected_forwards=expected_forwards,
        )
        if reduced != artifact.get("transport_reduction") or reduced != artifact.get(
            "independent_reduction"
        ):
            errors.append("transport_reduction_mismatch")
    counts = artifact.get("invocation_counts")
    events = artifact.get("current_invocation_events")
    if not isinstance(counts, Mapping) or not isinstance(events, list):
        errors.append("invocation_evidence_invalid")
    elif counts != reduce_invocation_events([row for row in events if isinstance(row, Mapping)]):
        errors.append("invocation_counts_mismatch")
    elif not invocation_counts_balanced(counts):
        errors.append("invocation_counts_unbalanced")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not REQUIRED_PRINCIPLE_FIELDS.issubset(principles):
        errors.append("field_principles_missing")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or not row.get("principle") for row in gates
    ):
        errors.append("acceptance_gates_invalid")
    else:
        summary = gate_check_summary([row for row in gates if isinstance(row, Mapping)])
        if summary != artifact.get("gate_check_summary"):
            errors.append("gate_summary_mismatch")
    ready = int(
        isinstance(artifact.get("transport_reduction"), Mapping)
        and artifact["transport_reduction"].get("passed") is True
        and isinstance(counts, Mapping)
        and invocation_counts_balanced(counts)
        and artifact.get("flagged_adversarial") is False
    )
    if artifact.get("native_readout_ready_score") != ready:
        errors.append("native_readout_ready_score_mismatch")
    if require_validation and not _validation_names_passed(
        artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ):
        errors.append("required_validation_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - live evidence boundary.
    """Return an aware UTC timestamp for one real boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and potentially long operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7477] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _event(call_id: str, operation: str, state: str) -> JsonDict:  # pragma: no cover
    return {
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current",
    }


def _load_object(path: Path) -> JsonDict:  # pragma: no cover
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _precondition(
    check: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover
    return {
        "check": check,
        "path": path,
        "upstream": path,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
    }


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    receipt: JsonDict = {"path": relative, "sha256": sha256_file(path)}
    if path.suffix == ".json":
        value = _load_object(path)
        receipt["original_flagged_adversarial"] = value.get("flagged_adversarial")
        receipt["original_verdict_class"] = value.get("verdict_class")
    else:
        receipt["original_flagged_adversarial"] = None
        receipt["original_verdict_class"] = None
    return receipt


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:  # pragma: no cover
    """Authenticate source bytes, historical flags, model, runtime, and CUDA."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_receipt(path, root)

    qualification = _load_object(root / V655_QUALIFICATION_PATH)
    historical = _load_object(root / V654_PARITY_PATH)
    lifecycle = _load_object(root / CAPTURE_LIFECYCLE_PATH)
    expected_fields = (
        (V655_QUALIFICATION_PATH, qualification, "option_protocol_ready_score", 1),
        (V655_QUALIFICATION_PATH, qualification, "verdict_class", "null"),
        (V655_QUALIFICATION_PATH, qualification, "flagged_adversarial", False),
        (V654_PARITY_PATH, historical, "native_readout_ready_score", 1),
        (V654_PARITY_PATH, historical, "local_runtime_parity_score", 0),
        (V654_PARITY_PATH, historical, "scored_runtime_parity_score", 0),
        (V654_PARITY_PATH, historical, "flagged_adversarial", False),
        (CAPTURE_LIFECYCLE_PATH, lifecycle, "capture_lifecycle_ready_score", 1),
        (CAPTURE_LIFECYCLE_PATH, lifecycle, "flagged_adversarial", False),
    )
    for relative, artifact, field, expected in expected_fields:
        checks.append(
            _precondition(
                f"upstream:{relative.name}:{field}",
                relative.as_posix(),
                field,
                expected,
                artifact.get(field),
            )
        )

    model = cached_current_model(gpu_index=0)
    model_path = Path(str(model.get("model_path"))) if model else Path("/absent-model")
    gpus = parity._gpu_inventory()
    idle = [
        row
        for row in gpus
        if row["memory_free_mb"] >= 20_000
        and row["memory_used_mb"] <= 1_024
        and row["utilization_pct"] <= 5
    ]
    selected = deepcopy(idle[0]) if idle else {}
    checks.extend(
        (
            _precondition(
                "force_live",
                "environment:CARNOT_FORCE_LIVE",
                "value",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
            _precondition(
                "model_identity",
                str(model_path),
                "hf_id",
                MODEL_HF_ID,
                model.get("hf_id") if model else None,
            ),
            _precondition("cached_gguf", str(model_path), "is_file", True, model_path.is_file()),
            _precondition(
                "embedded_tokenizer", str(model_path), "suffix", ".gguf", model_path.suffix.lower()
            ),
            _precondition("idle_cuda_device", "nvidia-smi", "idle_candidate", True, bool(idle)),
            _precondition(
                "current_task_not_quarantined",
                "ops/exclusion_manifest.yaml",
                EXPERIMENT_ID,
                False,
                "7477" in (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"),
            ),
            _precondition(
                "driving_requirement",
                str(SPEC_PATH.relative_to(root)),
                "REQ-*",
                "REQ-VERIFY-7477",
                "REQ-VERIFY-7477"
                if "REQ-VERIFY-7477" in SPEC_PATH.read_text(encoding="utf-8")
                else None,
            ),
        )
    )
    sealed_ids = {
        json.loads(line)["group_id"]
        for line in (root / V654_GROUPS_PATH).read_text(encoding="utf-8").splitlines()
    }
    development_ids = {row["group_id"] for row in freeze_development_groups()}
    checks.append(
        _precondition(
            "development_groups_disjoint",
            V654_GROUPS_PATH.as_posix(),
            "group_id_overlap",
            [],
            sorted(sealed_ids & development_ids),
        )
    )
    if model_path.is_file():
        model_receipt = _source_receipt(model_path, root)
        historical_model = historical.get("runtime_manifest", {}).get("native_llama_cpp", {})
        checks.append(
            _precondition(
                "historical_weight_hash_matches_current_bytes",
                str(model_path),
                "model_sha256",
                historical_model.get("model_sha256"),
                model_receipt["sha256"],
            )
        )
        hashes[str(model_path)] = model_receipt
    return (
        checks,
        {
            "model": model,
            "model_path": model_path,
            "gpu_inventory": gpus,
            "selected_gpu": selected,
            "qualification": qualification,
            "historical_parity": historical,
            "capture_lifecycle": lifecycle,
        },
        hashes,
    )


class NativeReadoutRunner:  # pragma: no cover - live Qwen boundary.
    """Add V654 option identity receipts around the existing native scorer."""

    def __init__(self, model_path: Path, gpu_index: int) -> None:
        self.scorer = parity.NativeScorer(model_path, gpu_index)
        self.reset_epoch = 0
        self.identity: JsonDict = {}

    def load(self, *, weight_sha256: str) -> JsonDict:
        """Load Qwen once and capture embedded tokenizer and build identity."""

        self.scorer.load()
        metadata = dict(getattr(self.scorer.llm, "metadata", {}) or {})
        template = str(metadata.get("tokenizer.chat_template", ""))
        template_bytes = template.encode("utf-8")
        backend_path = Path(self.scorer.backend.__file__).resolve()
        system_info = ""
        printer = getattr(
            getattr(self.scorer.backend, "llama_cpp", None), "llama_print_system_info", None
        )
        if callable(printer):
            printed = printer()
            system_info = (
                printed.decode(errors="replace") if isinstance(printed, bytes) else str(printed)
            )
        self.identity = {
            "model_id": MODEL_HF_ID,
            "model_path": str(self.scorer.model_path),
            "model_sha256": weight_sha256,
            "tokenizer_identity": {
                "representation": "embedded_gguf",
                "metadata_model": metadata.get("tokenizer.ggml.model"),
                "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
                "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
            },
            "chat_template_bytes_base64": base64.b64encode(template_bytes).decode("ascii"),
            "chat_template_byte_count": len(template_bytes),
            "chat_template_sha256": "sha256:" + hashlib.sha256(template_bytes).hexdigest(),
            "llama_cpp_build": {
                "version": getattr(self.scorer.backend, "__version__", "unknown"),
                "module_path": str(backend_path),
                "module_sha256": sha256_file(backend_path),
                "system_info": system_info,
            },
            "requested_offload": {"n_gpu_layers": -1, "split_mode": 0, "main_gpu": 0},
            "prefix_cache_used": False,
            "http_adapter_used": False,
        }
        return deepcopy(self.identity)

    def score(
        self,
        *,
        call_id: str,
        source: str,
        response: str,
        option_order: Sequence[str],
        row_kind: str,
        group_id: str,
    ) -> JsonDict:
        """Reset KV state and retain exact token-boundary and score-position receipts."""

        prompt = option_protocol.build_option_prompt(source, response, option_order)
        prompt_ids = self.scorer.tokenize(prompt, add_bos=True)
        boundaries: dict[str, bool] = {}
        label_ids: list[int] = []
        for label in option_protocol.DISPLAY_LABELS:
            ids = self.scorer.tokenize(label, add_bos=False)
            label_ids.append(ids[0] if len(ids) == 1 else -1)
            boundaries[label] = len(ids) == 1 and self.scorer.tokenize(
                prompt + label, add_bos=True
            ) == [*prompt_ids, *ids]
        started = time.monotonic()
        scored = self.scorer.score(prompt)
        elapsed = time.monotonic() - started
        self.reset_epoch += 1
        actual_tokens = getattr(
            self.scorer.llm,
            "n_tokens",
            getattr(self.scorer.llm, "_n_tokens", len(prompt_ids)),
        )
        actual_position = int(actual_tokens) - 1
        labels = ("A", "B")
        raw_by_option = {
            option_id: float(scored["option_logits"][label])
            for label, option_id in zip(labels, option_order, strict=True)
        }
        label_probabilities = parity.option_distribution(scored["option_logits"])
        probabilities = {
            option_id: float(label_probabilities[label])
            for label, option_id in zip(labels, option_order, strict=True)
        }
        return {
            "call_id": call_id,
            "group_id": group_id,
            "row_kind": row_kind,
            "disposition": "complete",
            "source_sha256": canonical_hash(source),
            "response_sha256": canonical_hash(response),
            "prompt_sha256": canonical_hash(prompt),
            "option_ids_in_prompt": list(option_order),
            "display_labels": list(option_protocol.DISPLAY_LABELS),
            "label_token_ids": label_ids,
            "label_to_option_id": dict(
                zip(option_protocol.DISPLAY_LABELS, option_order, strict=True)
            ),
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "requested_score_position": len(prompt_ids) - 1,
            "actual_last_evaluated_position": actual_position,
            "token_boundary_receipts": boundaries,
            "raw_logits_by_option_id": raw_by_option,
            "probabilities_by_option_id": probabilities,
            "predicted_option_id": max(probabilities, key=probabilities.__getitem__),
            "prediction_correct": None,
            "prefill_s": elapsed,
            "generated_tokens": 0,
            "state_reset": scored["state_reset"],
            "kv_state_id": f"reset-epoch-{self.reset_epoch:03d}",
            "model_sha256": self.identity["model_sha256"],
            "tokenizer_identity": deepcopy(self.identity["tokenizer_identity"]),
        }

    def close(self) -> None:
        """Release only this in-process model; no server process exists."""

        self.scorer.close()


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str
) -> Any:  # pragma: no cover
    """Run a blocking model load while reporting truthful pending heartbeats."""

    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the owner thread.
            error.append(exc)

    worker = threading.Thread(target=target, name=f"{EXPERIMENT_ID}-{phase}", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=60.0)
        if worker.is_alive():
            progress(started, phase, "pending", operation="model_load")
    if error:
        raise error[0]
    return result[0] if result else None


def _phase_span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": time.monotonic() - run_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _transport_controls() -> list[JsonDict]:  # pragma: no cover
    """Return four frozen calls, including a separated prompt replay."""

    first = {
        "control_id": "supported-original",
        "source": "The access code is 4821.",
        "response": "The access code is 4821.",
        "order": option_protocol.OPTION_IDS,
    }
    return [
        first,
        {
            "control_id": "unsupported-reversed",
            "source": "The access code is 4821.",
            "response": "The access code is 9000.",
            "order": tuple(reversed(option_protocol.OPTION_IDS)),
        },
        {
            "control_id": "supported-reversed",
            "source": "The marker is blue.",
            "response": "The marker is blue.",
            "order": tuple(reversed(option_protocol.OPTION_IDS)),
        },
        {**first, "control_id": "supported-original-replay"},
    ]


def _source_hashes_valid(root: Path, values: object) -> bool:  # pragma: no cover
    if not isinstance(values, Mapping) or not values:
        return False
    for receipt in values.values():
        if not isinstance(receipt, Mapping):
            return False
        path = Path(str(receipt.get("path")))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
            return False
    return True


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:  # pragma: no cover
    """Reload per-forward evidence without trusting headline fields."""

    rows = artifact.get("rows")
    budget = artifact.get("sample_size_budget")
    if not isinstance(rows, list) or not isinstance(budget, Mapping):
        return {"passed": False, "errors": ["rows_or_budget_invalid"]}
    reduction = reduce_transport_rows(
        [row for row in rows if isinstance(row, Mapping)],
        expected_forwards=int(budget.get("planned", -1)),
    )
    errors = []
    if reduction != artifact.get("transport_reduction"):
        errors.append("transport_reduction_mismatch")
    if not _source_hashes_valid(root, artifact.get("source_artifact_hashes")):
        errors.append("source_hashes_invalid")
    if require_terminal and not _validation_names_passed(
        artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ):
        errors.append("validation_receipts_invalid")
    return {"passed": not errors, "errors": errors, "transport_reduction": reduction}


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(root).as_posix()
    python = ".venv/bin/python"
    commands = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE, "--cold-replay", relative),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    )
    return [
        PlannedCommand(
            command, "safety" if command.name == "adversarial_verify" else "completion", True
        )
        for command in commands
    ]


def _build_live_artifact(
    *,
    preconditions: list[JsonDict],
    source_hashes: dict[str, JsonDict],
    rows: list[JsonDict],
    events: list[JsonDict],
    model_identity: JsonDict,
    lease_receipt: JsonDict,
    device_identity: JsonDict,
    spans: list[JsonDict],
    durations: JsonDict,
    validation_receipts: list[JsonDict],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    require_terminal: bool,
) -> JsonDict:  # pragma: no cover
    reduction = reduce_transport_rows(rows, expected_forwards=FORWARD_BUDGET)
    counts = reduce_invocation_events(events)
    required_names = (
        (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES) if require_terminal else AFFECTED_CHECK_NAMES
    )
    validation_passed = _validation_names_passed(validation_receipts, required_names)
    authenticated = (
        all(row.get("passed") is True for row in preconditions)
        and reduction["passed"]
        and invocation_counts_balanced(counts)
        and model_identity.get("observed_offload", {}).get("cuda_offload") is True
        and lease_receipt.get("release", {}).get("released") is True
    )
    gates = build_acceptance_gates(
        transport_passed=authenticated,
        validation_passed=validation_passed,
        benefit_measured=False,
    )
    ready = int(all(row["passed"] for row in gates))
    prefill = [float(row["prefill_s"]) for row in rows if row.get("disposition") == "complete"]
    load_s = float(durations["model_load"])
    cost_rows = [
        {"kind": "model_load", "call_id": "native-load", "elapsed_s": load_s},
        *[
            {
                "kind": "prompt_prefill",
                "call_id": row["call_id"],
                "group_id": row["group_id"],
                "prompt_tokens": row["prompt_token_count"],
                "elapsed_s": row["prefill_s"],
            }
            for row in rows
        ],
    ]
    prompt_identity = {
        **deepcopy(model_identity),
        "development_groups_sha256": DEVELOPMENT_GROUPS_SHA256,
        "prompt_token_ids_by_call": {row["call_id"]: row["prompt_token_ids"] for row in rows},
        "actual_last_evaluated_position_by_call": {
            row["call_id"]: row["actual_last_evaluated_position"] for row in rows
        },
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_null_native_readout_transport_ready_predictive_benefit_deferred"
        if ready
        else "complete_disqualified_native_readout_evidence",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {
            "wall": "datetime.now(UTC)",
            "monotonic": "time.monotonic_ns",
            "pid": os.getpid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        },
        "preconditions_checked": preconditions,
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": deepcopy(MODEL_SPECS),
        "model_invoked": counts["model_loads"]["attempted"] > 0
        or counts["forward_calls"]["attempted"] > 0,
        "invocation_counts": counts,
        "current_invocation_events": events,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": device_identity,
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_components_s": durations,
        "phase_spans": spans,
        "random_seed": {"ordering": 65_500, "audit": 65_509, "bootstrap": None},
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        "sample_size_budget": {
            "planned": FORWARD_BUDGET,
            "attempted": counts["forward_calls"]["attempted"],
            "complete": counts["forward_calls"]["completed"],
            "failed": counts["forward_calls"]["failed"],
            "censored": counts["forward_calls"]["cancelled"],
            "excluded": 0,
            "unstarted": FORWARD_BUDGET - counts["forward_calls"]["attempted"],
        },
        "transport_reduction": reduction,
        "independent_reduction": deepcopy(reduction),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": "complete_null_native_readout_transport_ready_predictive_benefit_deferred"
        if ready
        else "complete_disqualified_native_readout_evidence",
        "verdict_class": "null" if ready else "disqualified",
        "verifier_is_oracle": False,
        "flagged_adversarial": not ready,
        "validation_receipts": validation_receipts,
        "native_readout_ready_score": ready,
        "prompt_identity": prompt_identity,
        "pilot_cost_rows": cost_rows,
        "pilot_cost_summary": {
            "load_s": load_s,
            "prefill_min_s": min(prefill),
            "prefill_median_s": statistics.median(prefill),
            "prefill_max_s": max(prefill),
            "prefill_total_s": sum(prefill),
        },
        "capture_feasibility": project_capture_costs(prefill, load_s=load_s),
        "gpu_lease": lease_receipt,
        "scored_runtime_parity_score": 0,
        "server_runtime_parity_claimed": False,
        "blackwell_parity_claimed": False,
        "e0_disposition": "deferred_not_reopened",
        "scored_deployment_authorized": False,
        "scientific_benefit_measured": False,
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "passed" if require_terminal else "pending",
            "independent_raw_reduction": "passed" if require_terminal else "pending",
            "numbered_runtime_e2e": "not_applicable_isolated_readout_experiment",
        },
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "promotion_score": 0,
    }
    artifact["field_principles"] = deepcopy(FIELD_PRINCIPLES)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared live capability E2E.
    """Run one owned native load, 28 no-generation forwards, and scoped checks."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    durations: JsonDict = {
        "model_load": 0.0,
        "forward": 0.0,
        "generation": 0.0,
        "numeric_reduction": 0.0,
        "validation": 0.0,
    }

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before_source_and_weight_hashes")
    preconditions, context, source_hashes = collect_preconditions(root)
    progress(
        run_started, "preconditions", "after_source_and_weight_hashes", checked=len(preconditions)
    )
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise RuntimeError(f"precondition_failed:{failed['check']}:{failed['observed']}")
    spans.append(
        _phase_span(
            "preconditions", phase_started, run_started, len(preconditions), "authenticated"
        )
    )
    progress(run_started, "preconditions", "complete", checked=len(preconditions))

    model_path = Path(context["model_path"])
    gpu = dict(context["selected_gpu"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=MAX_LIVE_SECONDS + 300.0,
    )
    owner = lease.owner_receipt()
    runner = NativeReadoutRunner(model_path, int(gpu["index"]))
    model_identity: JsonDict = {}
    release: JsonDict = {}
    inference_ok = False
    live_started = time.monotonic()
    try:
        lease.transition("admitted")
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(run_started, "model_load", "before_model_load", gpu=gpu["index"])
        events.append(_event("native-load", "model_loads", "attempted"))
        load_started = time.monotonic()
        try:
            model_identity = _call_with_heartbeats(
                lambda: runner.load(weight_sha256=source_hashes[str(model_path)]["sha256"]),
                started=run_started,
                phase="model_load",
            )
        except BaseException:
            events.append(_event("native-load", "model_loads", "failed"))
            raise
        events.append(_event("native-load", "model_loads", "completed"))
        durations["model_load"] = time.monotonic() - load_started
        resident_mb = parity._gpu_memory(int(gpu["index"]))
        model_identity["observed_offload"] = {
            "device_uuid": gpu["uuid"],
            "resident_vram_mb": resident_mb,
            "baseline_vram_mb": gpu["memory_used_mb"],
            "cuda_offload": resident_mb - int(gpu["memory_used_mb"]) > 1_000,
        }
        model_identity["lease_id"] = owner["lease_id"]
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, run_started, 1, "resident"))
        progress(run_started, "model_load", "after_model_load", resident_vram_mb=resident_mb)

        schedule: list[JsonDict] = []
        for group in freeze_development_groups():
            for order_index, order in enumerate(
                (option_protocol.OPTION_IDS, tuple(reversed(option_protocol.OPTION_IDS)))
            ):
                schedule.append(
                    {
                        "call_id": f"development-{len(schedule):02d}",
                        "group_id": group["group_id"],
                        "source": group["source"],
                        "response": group["response"],
                        "order": order,
                        "row_kind": "development_source",
                        "order_index": order_index,
                    }
                )
        for control in _transport_controls():
            schedule.append(
                {
                    "call_id": f"transport-control-{len(schedule) - 24:02d}",
                    "group_id": control["control_id"],
                    "source": control["source"],
                    "response": control["response"],
                    "order": control["order"],
                    "row_kind": "transport_control",
                }
            )
        if len(schedule) != FORWARD_BUDGET:
            raise RuntimeError("forward_schedule_budget_mismatch")

        phase_started = time.monotonic()
        progress(run_started, "native_readout", "before_forward_loop", planned=len(schedule))
        checkpoint_path = root / RAW_DIR / "checkpoint.json"
        for index, request in enumerate(schedule):
            if time.monotonic() - live_started >= MAX_LIVE_SECONDS:
                raise TimeoutError("live_work_cap_reached")
            call_id = str(request["call_id"])
            events.append(_event(call_id, "forward_calls", "attempted"))
            call_started = time.monotonic()
            try:
                row = runner.score(
                    call_id=call_id,
                    source=str(request["source"]),
                    response=str(request["response"]),
                    option_order=request["order"],
                    row_kind=str(request["row_kind"]),
                    group_id=str(request["group_id"]),
                )
            except BaseException:
                events.append(_event(call_id, "forward_calls", "failed"))
                raise
            durations["forward"] += time.monotonic() - call_started
            events.append(_event(call_id, "forward_calls", "completed"))
            rows.append(row)
            atomic_json(
                checkpoint_path,
                {
                    "schema": "carnot.exp7477.checkpoint.v1",
                    "development_groups_sha256": DEVELOPMENT_GROUPS_SHA256,
                    "model_sha256": model_identity["model_sha256"],
                    "completed_units": len(rows),
                    "rows_sha256": canonical_hash(rows),
                },
            )
            progress(
                run_started,
                "native_readout",
                "unit_complete",
                completed=index + 1,
                planned=len(schedule),
            )
        spans.append(
            _phase_span(
                "native_readout", phase_started, run_started, len(rows), "all_forwards_complete"
            )
        )
        progress(run_started, "native_readout", "after_forward_loop", completed=len(rows))
        inference_ok = reduce_transport_rows(rows, expected_forwards=FORWARD_BUDGET)["passed"]
        if not inference_ok:
            raise RuntimeError("native_transport_reduction_failed")
    finally:
        progress(run_started, "model_unload", "before_model_unload")
        runner.close()
        gc.collect()
        progress(run_started, "model_unload", "after_model_unload")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_mb = parity._gpu_memory(int(gpu["index"]))
            unload_observed = after_mb <= int(gpu["memory_used_mb"]) + 1_024
            lease.transition(
                "validating",
                vram_mb=after_mb,
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition(
                "terminal_complete" if inference_ok and unload_observed else "terminal_blocked"
            )
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
    lease_receipt = {**owner, "release": release}

    phase_started = time.monotonic()
    reduction_started = time.monotonic()
    progress(run_started, "reduction", "before_benchmark", completed=len(rows))
    durations["numeric_reduction"] = time.monotonic() - reduction_started
    spans.append(_phase_span("reduction", phase_started, run_started, len(rows), "candidate_ready"))
    progress(run_started, "reduction", "after_benchmark", completed=len(rows))

    private_root = Path(tempfile.mkdtemp(prefix="exp7477-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError("validation_plan_invalid:" + ",".join(plan_errors))
    validation_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "affected_checks"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if not affected_reduction["passed"]:
        raise RuntimeError("affected_validation_failed")

    device_identity = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "cuda_inventory": context["gpu_inventory"],
        "selected_cuda": gpu,
    }
    candidate = _build_live_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        events=events,
        model_identity=model_identity,
        lease_receipt=lease_receipt,
        device_identity=device_identity,
        spans=spans,
        durations=durations,
        validation_receipts=affected,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        require_terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(root, candidate_path)
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "terminal_readers"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")
    durations["validation"] = time.monotonic() - validation_started

    final = _build_live_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=rows,
        events=events,
        model_identity=model_identity,
        lease_receipt=lease_receipt,
        device_identity=device_identity,
        spans=spans,
        durations=durations,
        validation_receipts=[*affected, *terminal],
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        require_terminal=True,
    )
    independent = independent_reduce(final, root=root, require_terminal=True)
    if independent.get("passed") is not True:
        raise RuntimeError(f"independent_reduction_failed:{independent}")
    errors = validate_artifact(final, require_validation=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    progress(run_started, "publish", "before_atomic_publish", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_publish", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed live command and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the live pilot or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        artifact = _load_object(args.cold_replay)
        errors = (
            validate_artifact(artifact, require_validation=False)
            if artifact
            else ["artifact_unreadable"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = _load_object(args.independent_reduce)
        reduced = (
            independent_reduce(artifact, root=root, require_terminal=False)
            if artifact
            else {"passed": False}
        )
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return int(reduced.get("passed") is not True)
    result = run_experiment(root, args.date, output=args.output)
    print(
        json.dumps(
            {
                "artifact": str(root / args.output),
                "honest_verdict": result["honest_verdict"],
                "native_readout_ready_score": result["native_readout_ready_score"],
                "scored_runtime_parity_score": result["scored_runtime_parity_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
