"""Measure the sealed native option readout after a fresh GPU admission.

The module reuses the qualified collection and native transport code. It adds
the sealed online role only for forecasting and does not open outcome labels.

Spec refs: REQ-VERIFY-7563 and SCENARIO-VERIFY-7563-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7535_v659_native_pilot as native
from carnot import experiment_7548_v660_capture_runner as capture
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
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
VERSION = "v661"
EXPERIMENT_ID = "exp7563-v661-native-pilot"
SCHEMA = "carnot.exp7563.v661.native_pilot.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
DATASET_ID = "KRLabsOrg/lettucedetect-code-hallucination"
DATASET_REVISION = "866a7c5392c3cf87e4fbc2b3808815d524f54331"
ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40, "online": 160, "test": 80}
DEVELOPMENT_GROUPS = 12
PILOT_FORWARDS = 72
CAPTURE_GROUPS = 240
CAPTURE_FORWARDS = 1440
ACQUISITION_LIMIT_S = 3000.0
EXECUTION_LIMIT_S = 4200.0
VALIDATION_RESERVE_S = 900.0
N_CTX = native.N_CTX
CONDITIONS = capture.CONDITIONS
OPTION_ORDERS = capture.OPTION_ORDERS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7563_v661_native_pilot.json")
RAW_DIR = Path("results/raw/experiment_7563_v661_native_pilot")
ROWS_PATH = RAW_DIR / "native_rows.jsonl"
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
CHECKPOINT_PATH = RAW_DIR / "checkpoint.json"
MODULE_PATH = Path("python/carnot/experiment_7563_v661_native_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7563_v661_native_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7563_v661_native_pilot.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
PROTOCOL_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
RUNNER_PATH = Path("results/experiment_7548_v660_capture_runner.json")

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

NativePilotError = native.NativePilotError
NativeOptionRunner = native.NativeOptionRunner
sha256_text = native.sha256_text
sha256_file = native.sha256_file
canonical_hash = native.canonical_hash
gate = native.gate
gate_check_summary = native.gate_check_summary


def reduce_invocation_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count each operation and state without making a zero generation count a no-model claim."""

    buckets = {
        "model_loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0},
        "forwards": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0},
        "generations": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0},
    }
    operation_names = {
        "model_load": "model_loads",
        "forward": "forwards",
        "generation": "generations",
    }
    for event in events:
        operation = operation_names.get(str(event.get("operation")))
        state = str(event.get("state"))
        if operation is not None and state in buckets[operation]:
            buckets[operation][state] += 1
    in_flight = sum(
        bucket["attempted"] - bucket["completed"] - bucket["failed"] - bucket["cancelled"]
        for bucket in buckets.values()
    )
    return {
        **buckets,
        "model_loads_attempted": buckets["model_loads"]["attempted"],
        "model_loads_completed": buckets["model_loads"]["completed"],
        "forward_calls_attempted": buckets["forwards"]["attempted"],
        "forward_calls_completed": buckets["forwards"]["completed"],
        "failures": sum(bucket["failed"] for bucket in buckets.values()),
        "cancellations": sum(bucket["cancelled"] for bucket in buckets.values()),
        "in_flight": in_flight,
    }


def _contract_gate(
    check: str,
    expected: object,
    observed: object,
    *,
    upstream: str,
    field: str,
    path: str | None = None,
) -> JsonDict:
    """Make one exact prerequisite row so a block has an actionable route."""

    return gate(
        check,
        "external_precondition",
        expected,
        observed,
        "==",
        observed == expected,
        "Changed or missing sealed evidence must stop dependent model work.",
        upstream=upstream,
        field=field,
        path=path,
    )


def _role_contract(protocol: Mapping[str, Any]) -> tuple[dict[str, int], bool, bool]:
    manifest = protocol.get("role_manifest")
    groups = list(manifest.get("groups") or []) if isinstance(manifest, Mapping) else []
    counts = dict(Counter(str(row.get("role")) for row in groups if isinstance(row, Mapping)))
    hashes = [str(row.get("component_hash")) for row in groups if isinstance(row, Mapping)]
    connected = len(hashes) == len(set(hashes)) == sum(ROLE_COUNTS.values())
    membership = all(
        row.get("official_split") == ("test" if row.get("role") == "test" else "train")
        for row in groups
        if isinstance(row, Mapping)
    )
    return counts, connected, membership


def authenticate_upstreams(
    protocol: Mapping[str, Any], runner: Mapping[str, Any]
) -> tuple[list[JsonDict], JsonDict]:
    """Reduce sealed corpus and runner evidence without trusting either headline."""

    checks: list[JsonDict] = []
    protocol_identity = {
        "experiment_id": protocol.get("experiment_id"),
        "tool_protocol_ready_score": protocol.get("tool_protocol_ready_score"),
        "verdict_class": protocol.get("verdict_class"),
        "flagged_adversarial": protocol.get("flagged_adversarial"),
    }
    checks.append(
        _contract_gate(
            "sealed_protocol_identity",
            {
                "experiment_id": "exp7533-v659-tool-protocol",
                "tool_protocol_ready_score": 1,
                "verdict_class": "null",
                "flagged_adversarial": False,
            },
            protocol_identity,
            upstream="exp7533-v659-tool-protocol",
            field="terminal_identity_and_readiness",
            path=PROTOCOL_PATH.as_posix(),
        )
    )
    license_row = protocol.get("license_attribution")
    license_map = dict(license_row) if isinstance(license_row, Mapping) else {}
    checks.append(
        _contract_gate(
            "dataset_revision",
            {"dataset": DATASET_ID, "revision": DATASET_REVISION},
            {"dataset": license_map.get("dataset"), "revision": license_map.get("revision")},
            upstream="exp7533-v659-tool-protocol",
            field="license_attribution.dataset_revision",
        )
    )
    role_counts, connected, official_membership = _role_contract(protocol)
    checks.extend(
        (
            _contract_gate(
                "sealed_role_counts",
                ROLE_COUNTS,
                role_counts,
                upstream="exp7533-v659-tool-protocol",
                field="role_manifest.groups.role_counts",
            ),
            _contract_gate(
                "connected_component_grouping",
                True,
                connected,
                upstream="exp7533-v659-tool-protocol",
                field="role_manifest.unique_component_membership",
            ),
            _contract_gate(
                "official_test_membership",
                True,
                official_membership,
                upstream="exp7533-v659-tool-protocol",
                field="role_manifest.official_split_membership",
            ),
        )
    )
    role_manifest = protocol.get("role_manifest")
    role_groups = (
        list(role_manifest.get("groups") or []) if isinstance(role_manifest, Mapping) else []
    )
    donor_manifest = protocol.get("donor_manifest")
    donor_groups = (
        list(donor_manifest.get("groups") or []) if isinstance(donor_manifest, Mapping) else []
    )
    roles = {
        (str(row.get("role")), str(row.get("component_hash"))): str(row.get("tool_type"))
        for row in role_groups
        if isinstance(row, Mapping)
    }
    donors_valid = bool(
        len(donor_groups) == sum(ROLE_COUNTS.values())
        and isinstance(donor_manifest, Mapping)
        and donor_manifest.get("altered_sources_have_labels") is False
        and all(
            isinstance(row, Mapping)
            and str(row.get("donor_component_hash")) != str(row.get("component_hash"))
            and roles.get((str(row.get("role")), str(row.get("component_hash"))))
            == str(row.get("tool_type"))
            for row in donor_groups
        )
    )
    checks.append(
        _contract_gate(
            "whole_prompt_same_role_donors",
            True,
            donors_valid,
            upstream="exp7533-v659-tool-protocol",
            field="donor_manifest.complete_same_role_tool_donors",
        )
    )
    readers = protocol.get("reader_access_contract")
    reader_map = dict(readers) if isinstance(readers, Mapping) else {}
    label_access = reader_map.get("actual_capture_label_access")
    checks.append(
        _contract_gate(
            "capture_label_access",
            [],
            label_access,
            upstream="exp7533-v659-tool-protocol",
            field="reader_access_contract.actual_capture_label_access",
        )
    )
    exposure = protocol.get("exposure_inventory")
    events = list(exposure.get("events") or []) if isinstance(exposure, Mapping) else []
    exposure_ok = bool(events) and all(
        row.get("labels_consumed") == 0
        for row in events
        if isinstance(row, Mapping) and row.get("event") in {"capture_reader", "role_selection"}
    )
    checks.append(
        _contract_gate(
            "exposure_exclusions",
            True,
            exposure_ok,
            upstream="exp7533-v659-tool-protocol",
            field="exposure_inventory.capture_and_selection",
        )
    )
    checks.extend(
        (
            _contract_gate(
                "capture_runner_identity",
                "exp7548-capture-runner",
                runner.get("experiment_id"),
                upstream="exp7548-capture-runner",
                field="experiment_id",
                path=RUNNER_PATH.as_posix(),
            ),
            _contract_gate(
                "capture_runner_ready",
                1,
                runner.get("capture_runner_ready_score"),
                upstream="exp7548-capture-runner",
                field="capture_runner_ready_score",
            ),
            _contract_gate(
                "capture_runner_adversarial",
                False,
                runner.get("flagged_adversarial"),
                upstream="exp7548-capture-runner",
                field="flagged_adversarial",
            ),
        )
    )
    controls = runner.get("qualification_controls")
    control_map = dict(controls) if isinstance(controls, Mapping) else {}
    mutations = control_map.get("mutation_failures")
    mutation_map = dict(mutations) if isinstance(mutations, Mapping) else {}
    reduction = control_map.get("reduction")
    reduction_map = dict(reduction) if isinstance(reduction, Mapping) else {}
    transport_ready = bool(
        control_map.get("passed") is True
        and control_map.get("resume_parity") is True
        and reduction_map.get("passed") is True
        and mutation_map
        and all(value is True for value in mutation_map.values())
    )
    checks.append(
        _contract_gate(
            "transport_controls",
            True,
            transport_ready,
            upstream="exp7548-capture-runner",
            field="qualification_controls.independent_transport_controls",
        )
    )
    expected_runner_roles = {
        "fit": 160,
        "tune": 40,
        "policy": 40,
        "test": 80,
        "development": DEVELOPMENT_GROUPS,
    }
    checks.append(
        _contract_gate(
            "excluded_development_groups",
            expected_runner_roles,
            runner.get("role_counts"),
            upstream="exp7548-capture-runner",
            field="role_counts",
        )
    )
    return checks, {
        "role_counts": role_counts,
        "development_excluded_count": DEVELOPMENT_GROUPS,
        "historical_runner_verdict": runner.get("honest_verdict"),
        "historical_runner_verdict_class": runner.get("verdict_class"),
        "runner_transport_ready": transport_ready,
        "source_label_access": deepcopy(label_access),
    }


def compile_capture_roles(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> JsonDict:
    """Add the sealed online role while retaining the qualified six-cell wrapper."""

    observed = Counter(str(row.get("role")) for row in groups)
    if dict(observed) != dict(expected_role_counts):
        raise NativePilotError(f"capture_role_counts:{dict(expected_role_counts)}:{dict(observed)}")
    fit_roles = ("fit", "tune", "policy")
    evaluation_roles = ("test", "online")

    def one_capture(roles: tuple[str, ...]) -> JsonDict:
        selected = [deepcopy(dict(row)) for row in groups if row.get("role") in roles]
        counts = {role: int(expected_role_counts[role]) for role in roles}
        schedule = capture.compile_capture_schedule(selected, expected_role_counts=counts)
        reduced = capture.validate_capture_schedule(schedule, expected_role_counts=counts)
        if reduced.get("passed") is not True:
            raise NativePilotError("capture_schedule_invalid")
        return {
            "roles": list(roles),
            "group_count": sum(counts.values()),
            "forward_count": len(schedule),
            "schedule": schedule,
            "token_lengths": [int(row["prompt_token_count"]) for row in schedule],
            "role_schedule_hashes": reduced["role_schedule_hashes"],
        }

    return {
        "fit": one_capture(fit_roles),
        "evaluation": one_capture(evaluation_roles),
        "source_label_access": [],
        "native_transport_rebuilt": False,
    }


def forecast_captures(
    *,
    load_seconds: float,
    pilot_rows: Sequence[Mapping[str, Any]],
    fit_token_lengths: Sequence[int],
    evaluation_token_lengths: Sequence[int],
    fit_checkpoint_seconds: float,
    evaluation_checkpoint_seconds: float,
) -> JsonDict:
    """Forecast both fixed captures from measured time and complete prompt lengths."""

    samples = [
        row
        for row in pilot_rows
        if row.get("disposition") in {None, "complete"}
        and isinstance(row.get("forward_seconds"), int | float)
        and isinstance(row.get("prompt_token_count"), int | float)
        and float(row["forward_seconds"]) >= 0.0
        and int(row["prompt_token_count"]) > 0
    ]
    if not samples:
        raise NativePilotError("pilot_durations_missing")
    if (
        len(fit_token_lengths) != CAPTURE_FORWARDS
        or len(evaluation_token_lengths) != CAPTURE_FORWARDS
    ):
        raise NativePilotError("capture_length_count")
    all_lengths = [*fit_token_lengths, *evaluation_token_lengths]
    if any(length <= 0 or length > N_CTX for length in all_lengths):
        raise NativePilotError("capture_prompt_overlength")
    p95_forward = native._p95([float(row["forward_seconds"]) for row in samples])
    p95_pilot_tokens = native._p95([float(row["prompt_token_count"]) for row in samples])

    def one(lengths: Sequence[int], checkpoint_seconds: float) -> JsonDict:
        projected_forward = sum(p95_forward * length / p95_pilot_tokens for length in lengths)
        acquisition = float(load_seconds) + projected_forward + float(checkpoint_seconds)
        total = acquisition + VALIDATION_RESERVE_S
        return {
            "planned_groups": CAPTURE_GROUPS,
            "planned_forwards": CAPTURE_FORWARDS,
            "p95_forward_seconds": p95_forward,
            "p95_pilot_prompt_tokens": p95_pilot_tokens,
            "full_prompt_token_count": len(lengths),
            "full_prompt_token_sum": sum(lengths),
            "full_prompt_token_min": min(lengths),
            "full_prompt_token_max": max(lengths),
            "projected_forward_seconds": projected_forward,
            "model_load_seconds": float(load_seconds),
            "checkpoint_seconds": float(checkpoint_seconds),
            "acquisition_seconds": acquisition,
            "total_with_validation_seconds": total,
            "acquisition_within_limit": acquisition <= ACQUISITION_LIMIT_S,
            "execution_within_limit": total <= EXECUTION_LIMIT_S,
        }

    fit = one(fit_token_lengths, fit_checkpoint_seconds)
    evaluation = one(evaluation_token_lengths, evaluation_checkpoint_seconds)
    return {
        "formula": "load + sum(p95_forward * tokens / p95_pilot_tokens) + checkpoint",
        "fit": fit,
        "evaluation": evaluation,
        "fit_capture_feasible_score": int(
            fit["acquisition_within_limit"] and fit["execution_within_limit"]
        ),
        "eval_capture_feasible_score": int(
            evaluation["acquisition_within_limit"] and evaluation["execution_within_limit"]
        ),
        "acquisition_limit_seconds": ACQUISITION_LIMIT_S,
        "execution_limit_seconds": EXECUTION_LIMIT_S,
        "validation_reserve_seconds": VALIDATION_RESERVE_S,
        "sample_shrinking_allowed": False,
        "prompt_shortening_allowed": False,
        "role_shrinking_allowed": False,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind the terminal artifact while excluding only its checksum field."""

    stable = deepcopy(dict(value))
    stable.pop("reproducibility_checksum", None)
    return canonical_hash(stable)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "experiment_id": "The exact identity prevents evidence from moving between tasks.",
        "preconditions_checked": "Recorded prerequisites prevent fabricated fallback evidence.",
        "MODEL_SPECS": "The planned model name prevents silent model substitution.",
        "model_specs": "Resolved model bytes bind actual model work.",
        "invocation_counts": "Typed call counts prevent hidden generation or missing work.",
        "inference_substrate_class": "The actual class prevents implausible duration claims.",
        "duration_s": "Measured monotonic time exposes fabricated compute claims.",
        "random_seed": "Frozen seeds make selection and fitting reproducible.",
        "reproducibility_checksum": "The checksum detects changed code, inputs, and evidence.",
        "rows": "Absolute rows prevent missing observations from becoming zeros.",
        "sample_size_budget": "The full budget keeps exclusions and unstarted work visible.",
        "acceptance_gate_results": "Typed gates keep validity separate from readiness and benefit.",
        "gate_check_summary": "Exact failed operands route a blocked task without guesswork.",
        "honest_verdict": "A terminal prefix prevents false retry classification.",
        "verdict_class": "The closed class prevents external absence from becoming partial work.",
        "native_tool_ready_score": "A bare score requires authentic zero-generation custody.",
        "fit_capture_feasible_score": "A bare score requires the full fit capture to meet budget.",
        "eval_capture_feasible_score": "A bare score requires the full evaluation capture to meet budget.",
        "source_label_access": "An empty list proves capture did not consume outcome labels.",
        "current_gpu_admission": "Fresh ownership prevents stale capacity evidence from authorizing CUDA.",
    }
    return {
        field: specific.get(field, f"The {field} field preserves one auditable terminal operand.")
        for field in fields
    }


def _validation_passed(receipts: object) -> bool:
    if not isinstance(receipts, list):
        return False
    passed = {
        str(row.get("name"))
        for row in receipts
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)) <= passed


def _base_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "schema_binding": {
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "version": VERSION,
            "run_date": RUN_DATE,
        },
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "title": "V661 sealed native option-forward pilot",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": reduce_invocation_events([]),
        "historical_invocation_counts": {},
        "inference_substrate_class": "model_load_no_generation",
        "inference_substrate": "live_llm_embedding_extraction",
        "readout_kind": "option_logits",
        "execution_venue": "host",
        "execution_device": "real_cuda_required",
        "duration_s": max(0.000001, float(duration_s)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "duration_breakdown_s": {
            "current_measurement_and_validation": max(0.000001, float(duration_s)),
            "historical": 0.0,
        },
        "process_identity": {
            "pid": os.getpid(),
            "process_start_ticks": lease_api.proc_start_ticks(os.getpid()),
            "python": os.path.realpath(os.sys.executable),
            "clock": "time.monotonic",
        },
        "random_seed": {
            "model": 661563,
            "fitting": 661564,
            "ordering": 661565,
            "bootstrap": 661566,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "positive_claim": False,
        "benefit_measured": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": "not_applicable_isolated_readout_pilot",
        "source_label_access": [],
        "role_manifest": {
            "role_counts": ROLE_COUNTS,
            "development_excluded_count": DEVELOPMENT_GROUPS,
            "fit_capture_roles": ["fit", "tune", "policy"],
            "evaluation_capture_roles": ["test", "online"],
            "labels_read_for_capture": 0,
        },
    }


def build_blocked_artifact(
    *,
    failed_gate: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Record external absence with no invented load, forward, or generation."""

    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    failed = deepcopy(dict(failed_gate))
    reason = str(failed.get("check") or "external_precondition")
    artifact.update(
        {
            "inference_substrate_class": "blocked_no_run",
            "inference_substrate": "blocked_no_run",
            "planned_inference_substrate_class": "model_load_no_generation",
            "planned_inference_substrate": "live_llm_embedding_extraction",
            "execution_device": "none_precondition_blocked",
            "rows": [],
            "raw_native_rows": None,
            "current_invocation_events": [],
            "sample_size_budget": {
                "planned": PILOT_FORWARDS,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": PILOT_FORWARDS,
            },
            "acceptance_gate_results": [failed],
            "gate_check_summary": gate_check_summary([failed]),
            "honest_verdict": f"complete_blocked_{reason}",
            "verdict_class": "blocked",
            "native_tool_ready_score": 0,
            "fit_capture_feasible_score": 0,
            "eval_capture_feasible_score": 0,
            "capture_forecasts": None,
            "pilot_cost_rows": [],
            "current_gpu_admission": None,
            "ownership_receipt": None,
            "transport_reduction": None,
            "validation_receipts": [],
            "complete": 1,
            "ready": 0,
        }
    )
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _ready_gates(
    reduction: Mapping[str, Any], forecasts: Mapping[str, Any], validation_ok: bool
) -> list[JsonDict]:
    return [
        gate(
            "native_logit_custody",
            "validity",
            True,
            reduction.get("passed"),
            "is",
            reduction.get("passed") is True,
            "Invalid native rows cannot support transport readiness.",
            upstream="raw_native_rows",
            field="transport_reduction.passed",
        ),
        gate(
            "fit_capture_forecast",
            "readiness",
            1,
            forecasts.get("fit_capture_feasible_score"),
            "==",
            forecasts.get("fit_capture_feasible_score") == 1,
            "A full fixed fit capture must meet both time limits.",
            upstream="capture_forecasts.fit",
            field="fit_capture_feasible_score",
        ),
        gate(
            "evaluation_capture_forecast",
            "readiness",
            1,
            forecasts.get("eval_capture_feasible_score"),
            "==",
            forecasts.get("eval_capture_feasible_score") == 1,
            "A full fixed evaluation capture must meet both time limits.",
            upstream="capture_forecasts.evaluation",
            field="eval_capture_feasible_score",
        ),
        gate(
            "required_validation",
            "validity",
            True,
            validation_ok,
            "is",
            validation_ok,
            "Required scoped and terminal checks must pass before publication.",
            upstream="validation_receipts",
            field="all_required_checks_passed",
        ),
        gate(
            "positive_claim",
            "benefit",
            False,
            False,
            "is",
            True,
            "Transport and feasibility do not prove predictive benefit.",
            upstream="development_only_pilot",
            field="positive_claim",
        ),
    ]


def build_complete_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
    raw_receipts: Mapping[str, Any],
    reduction: Mapping[str, Any],
    forecasts: Mapping[str, Any],
    ownership: Mapping[str, Any],
    pilot_cost_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    require_validation: bool,
) -> JsonDict:
    """Build one complete transport result without converting features to labels."""

    validation_ok = _validation_passed(list(validation_receipts)) if require_validation else True
    gates = _ready_gates(reduction, forecasts, validation_ok)
    custody = reduction.get("passed") is True
    verdict_class = "null" if custody and validation_ok else "disqualified"
    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    counts = reduce_invocation_events(events)
    comparative = [deepcopy(dict(row)) for row in reduction.get("comparative_rows") or []]
    for row in comparative:
        row.setdefault("disposition", "complete_feature_only_no_truth_label")
    artifact.update(
        {
            "model_specs": [deepcopy(dict(row)) for row in model_specs],
            "model_invoked": counts["model_loads_attempted"] > 0,
            "invocation_counts": counts,
            "current_invocation_events": [deepcopy(dict(row)) for row in events],
            "rows": comparative,
            "raw_native_rows": deepcopy(dict(raw_receipts)),
            "sample_size_budget": {
                "planned": PILOT_FORWARDS,
                "attempted": reduction.get("attempted_forward_count"),
                "completed": reduction.get("complete_forward_count"),
                "excluded": 0,
                "failed": reduction.get("error_forward_count"),
                "censored": 0,
                "unstarted": PILOT_FORWARDS - int(reduction.get("attempted_forward_count") or 0),
            },
            "development_panel": deepcopy(dict(panel)),
            "transport_reduction": deepcopy(dict(reduction)),
            "capture_forecasts": deepcopy(dict(forecasts)),
            "fit_capture_feasible_score": int(forecasts["fit_capture_feasible_score"]),
            "eval_capture_feasible_score": int(forecasts["eval_capture_feasible_score"]),
            "native_tool_ready_score": int(custody and validation_ok),
            "pilot_cost_rows": [deepcopy(dict(row)) for row in pilot_cost_rows],
            "current_gpu_admission": deepcopy(dict(ownership)),
            "ownership_receipt": deepcopy(dict(ownership)),
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "honest_verdict": (
                "complete_null_native_transport_ready_benefit_unmeasured"
                if verdict_class == "null"
                else "complete_disqualified_native_transport_validation"
            ),
            "verdict_class": verdict_class,
            "complete": 1,
            "ready": int(custody and validation_ok),
        }
    )
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test() -> JsonDict:
    """Build a production-shaped fixture without writing tracked evidence."""

    prior = native.build_artifact_for_test()
    pilot_rows = [
        {
            "disposition": row.get("disposition", "complete"),
            "forward_seconds": row["measured_seconds"],
            "prompt_token_count": row["full_prompt_length"],
        }
        for row in prior["pilot_cost_rows"]
        if row.get("kind") == "native_forward"
    ]
    forecasts = forecast_captures(
        load_seconds=2.0,
        pilot_rows=pilot_rows,
        fit_token_lengths=[100] * CAPTURE_FORWARDS,
        evaluation_token_lengths=[100] * CAPTURE_FORWARDS,
        fit_checkpoint_seconds=1.0,
        evaluation_checkpoint_seconds=1.0,
    )
    return build_complete_artifact(
        preconditions=[],
        source_hashes={},
        model_specs=prior["model_specs"],
        events=prior["current_invocation_events"],
        panel=prior["development_panel"],
        raw_receipts=prior["raw_native_rows"],
        reduction=prior["transport_reduction"],
        forecasts=forecasts,
        ownership=prior["ownership_receipt"],
        pilot_cost_rows=prior["pilot_cost_rows"],
        validation_receipts=[],
        duration_s=3.0,
        phase_spans=[],
        require_validation=False,
    )


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check terminal shape, counters, custody, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID or artifact.get("run_date") != RUN_DATE:
        errors.append("identity_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    for field in (
        "native_tool_ready_score",
        "fit_capture_feasible_score",
        "eval_capture_feasible_score",
    ):
        if type(artifact.get(field)) is not int or artifact.get(field) not in {0, 1}:
            errors.append(f"{field}_not_bare_binary")
    if artifact.get("source_label_access") != []:
        errors.append("capture_label_access_invalid")
    if artifact.get("readout_kind") != "option_logits":
        errors.append("readout_kind_invalid")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("planned_model_specs_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not principles:
        errors.append("field_principles_missing")
    events = artifact.get("current_invocation_events")
    if not isinstance(events, list):
        errors.append("invocation_evidence_invalid")
    elif artifact.get("invocation_counts") != reduce_invocation_events(events):
        errors.append("invocation_counts_mismatch")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("planned") != PILOT_FORWARDS:
        errors.append("sample_size_budget_invalid")
    role_manifest = artifact.get("role_manifest")
    if not isinstance(role_manifest, Mapping) or role_manifest.get("role_counts") != ROLE_COUNTS:
        errors.append("role_manifest_invalid")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or artifact.get("gate_check_summary") != gate_check_summary(
        gates
    ):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        if artifact.get("planned_inference_substrate_class") != "model_load_no_generation":
            errors.append("blocked_planned_substrate_invalid")
        if artifact.get("rows") != [] or artifact.get("model_specs") != []:
            errors.append("blocked_artifact_fabricated_work")
        if artifact.get("native_tool_ready_score") != 0:
            errors.append("blocked_artifact_ready")
    else:
        if artifact.get("inference_substrate_class") != "model_load_no_generation":
            errors.append("inference_substrate_class_invalid")
        counts = artifact.get("invocation_counts")
        if isinstance(counts, Mapping):
            generations = counts.get("generations")
            if not isinstance(generations, Mapping) or generations.get("attempted") != 0:
                errors.append("generation_call_detected")
            if artifact.get("native_tool_ready_score") == 1 and (
                counts.get("model_loads_completed") != 1
                or counts.get("forward_calls_completed") != PILOT_FORWARDS
            ):
                errors.append("ready_invocation_count_invalid")
    if require_validation and not _validation_passed(artifact.get("validation_receipts")):
        errors.append("required_validation_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _receipt_path(receipt: Mapping[str, Any]) -> Path:
    path = Path(str(receipt.get("path") or ""))
    return path if path.is_absolute() else REPO_ROOT / path


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Reload exact schedule and row bytes before trusting the artifact reduction."""

    if value.get("verdict_class") == "blocked":
        budget = value.get("sample_size_budget")
        passed = bool(
            value.get("rows") == []
            and value.get("model_invoked") is False
            and isinstance(budget, Mapping)
            and budget.get("unstarted") == PILOT_FORWARDS
        )
        return {"passed": passed, "mode": "blocked_no_measurement", "forward_count": 0}
    raw = value.get("raw_native_rows")
    raw_map = dict(raw) if isinstance(raw, Mapping) else {}
    schedule_receipt = raw_map.get("forward_schedule")
    rows_receipt = raw_map.get("native_rows")
    if not isinstance(schedule_receipt, Mapping) or not isinstance(rows_receipt, Mapping):
        return {"passed": False, "error": "sidecar_receipt_missing"}
    for receipt in (schedule_receipt, rows_receipt):
        path = _receipt_path(receipt)
        if (
            not path.is_file()
            or path.stat().st_size != receipt.get("bytes")
            or sha256_file(path) != receipt.get("sha256")
        ):
            return {"passed": False, "error": "sidecar_receipt_invalid"}
    schedule = native._read_jsonl(_receipt_path(schedule_receipt))
    rows = native._read_jsonl(_receipt_path(rows_receipt))
    reduction = native.reduce_native_rows(rows, schedule)
    declared = value.get("transport_reduction")
    return {
        "passed": reduction.get("passed") is True and reduction == declared,
        "forward_count": reduction.get("complete_forward_count"),
        "generated_token_count": reduction.get("generated_token_count"),
        "schedule_sha256": schedule_receipt.get("sha256"),
        "rows_sha256": rows_receipt.get("sha256"),
        "reduction_sha256": canonical_hash(reduction),
        "declared_reduction_sha256": canonical_hash(declared),
    }


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
    Path("python/carnot/experiment_7548_v660_capture_runner.py"),
    Path("python/carnot/experiment_7533_v659_tool_protocol.py"),
    Path("python/carnot/experiment_7535_v659_native_pilot.py"),
    Path("python/carnot/inference/sota_models.py"),
    PROTOCOL_PATH,
    RUNNER_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def _load_object(path: Path) -> JsonDict:  # pragma: no cover - live file boundary.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Flush every slow boundary so the conductor sees truthful progress."""

    print(
        json.dumps(
            {"phase": phase, "event": event, "elapsed_s": time.monotonic() - started, **details},
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "start_monotonic_offset_s": phase_started - run_started,
        "end_monotonic_offset_s": time.monotonic() - run_started,
        "duration_s": time.monotonic() - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def collect_preconditions(
    root: Path, started: float, run_date: str
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:  # pragma: no cover
    """Authenticate source bytes, sealed inputs, cached model, and runtime."""

    checks = [
        _contract_gate(
            "run_date",
            RUN_DATE,
            run_date,
            upstream="command_line",
            field="--date",
        )
    ]
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _contract_gate(
                f"required_input:{relative.as_posix()}",
                "readable_nonempty_bytes",
                observed,
                upstream=relative.as_posix(),
                field="path.bytes",
                path=relative.as_posix(),
            )
        )
        if observed is not None:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _contract_gate(
            "driving_requirement",
            "REQ-VERIFY-7563",
            "REQ-VERIFY-7563" if "REQ-VERIFY-7563" in spec_text else None,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-*",
            path=SPEC_PATH.as_posix(),
        )
    )
    protocol_artifact = _load_object(root / PROTOCOL_PATH)
    runner_artifact = _load_object(root / RUNNER_PATH)
    contract_checks, contract_context = authenticate_upstreams(protocol_artifact, runner_artifact)
    checks.extend(contract_checks)
    sealed_checks, sealed_context = capture.authenticate_protocol(root)
    checks.extend(sealed_checks)
    model = cached_current_model(preferred_quant="Q4_K_M")
    model_path = Path(str(model.get("model_path"))) if model else Path("/nonexistent")
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and "Q4_K_M" in model_path.name
        and model_path.is_file()
    )
    checks.append(
        _contract_gate(
            "cached_current_model",
            {"hf_id": MODEL_ID, "quantization": "Q4_K_M", "exists": True},
            {
                "hf_id": model.get("hf_id") if model else None,
                "quantization": "Q4_K_M" if "Q4_K_M" in model_path.name else None,
                "exists": model_path.is_file(),
            },
            upstream="carnot.inference.sota_models.cached_current_model",
            field="hf_id_quantization_path",
            path=str(model_path),
        )
    )
    model_hash: str | None = None
    if model_ok:
        progress(started, "preconditions", "before_model_hash", bytes=model_path.stat().st_size)
        model_hash = native._call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            pending="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_hash)
        hashes[str(model_path)] = model_hash
    try:
        import llama_cpp  # noqa: PLC0415

        runtime = {
            "available": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "module_path": str(Path(llama_cpp.__file__).resolve()),
        }
    except Exception as exc:  # noqa: BLE001
        runtime = {"available": False, "error": f"{type(exc).__name__}:{exc}"}
    checks.append(
        _contract_gate(
            "llama_cpp_runtime",
            True,
            runtime["available"],
            upstream="python_environment",
            field="llama_cpp.available",
            path=str(runtime.get("module_path")),
        )
    )
    return (
        checks,
        hashes,
        {
            "protocol": protocol_artifact,
            "runner": runner_artifact,
            "contract": contract_context,
            "sealed": sealed_context,
            "model": model or {},
            "model_path": model_path,
            "model_hash": model_hash,
            "runtime": runtime,
        },
    )


def _sealed_capture_groups(
    context: Mapping[str, Any], *, root: Path, started: float
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - sealed sidecar boundary.
    """Read predictor-only intervention bytes and retain the existing online role."""

    sealed = dict(context["sealed"])
    groups: list[JsonDict] = []
    for name, receipt in sorted(dict(sealed["intervention_receipts"]).items()):
        progress(started, "capture_schedule", "before_intervention_read", shard=name)
        rows = capture._read_jsonl_receipt(dict(receipt), root)
        progress(
            started,
            "capture_schedule",
            "after_intervention_read",
            shard=name,
            completed_units=len(rows),
        )
        for source in rows:
            role = str(source.get("role"))
            if role not in ROLE_COUNTS:
                continue
            groups.append(
                {
                    "component_hash": source.get("component_hash"),
                    "donor_component_hash": source.get("donor_component_hash"),
                    "role": role,
                    "tool_type": source.get("tool_type"),
                    "label_scope": "original_source_only",
                    "requests": deepcopy(list(source.get("requests") or [])),
                }
            )
    captures = compile_capture_roles(groups)
    manifest = {
        "role_counts": dict(Counter(str(row["role"]) for row in groups)),
        "fit": {
            key: value
            for key, value in captures["fit"].items()
            if key not in {"schedule", "token_lengths"}
        },
        "evaluation": {
            key: value
            for key, value in captures["evaluation"].items()
            if key not in {"schedule", "token_lengths"}
        },
        "source_label_access": [],
        "native_transport_rebuilt": False,
    }
    return captures, manifest


def _event(operation: str, state: str, call_id: str) -> JsonDict:  # pragma: no cover
    return {
        "operation": operation,
        "state": state,
        "call_id": call_id,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current",
    }


def _wait_for_owned_unload(
    pid: int, gpu_uuid: str, *, started: float, timeout_s: float = 60.0
) -> int | None:  # pragma: no cover - live NVIDIA boundary.
    """Wait for owned model memory to fall below the registered idle threshold."""

    wait_started = time.monotonic()
    next_heartbeat = wait_started
    while True:
        owned_vram = native._owned_vram_mb(pid, gpu_uuid)
        if owned_vram is None or owned_vram < native.GPU_IDLE_MAX_USED_MB:
            return owned_vram
        now = time.monotonic()
        if now - wait_started >= timeout_s:
            return owned_vram
        if now >= next_heartbeat:
            progress(
                started,
                "model_unload",
                "pending_driver_release",
                owned_vram_mb=owned_vram,
                waited_s=now - wait_started,
            )
            next_heartbeat = now + 10.0
        time.sleep(1.0)


def _run_affected_validation(
    root: Path, started: float
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7563-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if errors:
        raise NativePilotError("validation_plan_invalid:" + ",".join(errors))
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(receipts),
    )
    reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return [dict(row) for row in receipts], reduction.get("passed") is True


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
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
    return [PlannedCommand(spec, spec.scope, True) for spec in specs]


def _refresh_artifact(
    artifact: JsonDict,
    *,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> None:  # pragma: no cover
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in receipts]
    artifact["phase_spans"] = [deepcopy(dict(row)) for row in spans]
    artifact["duration_s"] = max(0.000001, float(duration_s))
    artifact["duration_breakdown_s"]["current_measurement_and_validation"] = artifact["duration_s"]
    validation_ok = _validation_passed(artifact["validation_receipts"])
    for row in artifact.get("acceptance_gate_results") or []:
        if isinstance(row, dict) and row.get("check") == "required_validation":
            row["observed"] = validation_ok
            row["passed"] = validation_ok
    if artifact.get("verdict_class") != "blocked" and not validation_ok:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
    artifact["gate_check_summary"] = gate_check_summary(
        artifact.get("acceptance_gate_results") or []
    )
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def _run_terminal_validation(
    root: Path,
    artifact: JsonDict,
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    candidate = root / CANDIDATE_PATH
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in affected]
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_json(candidate, artifact)
    commands = _terminal_commands(candidate)
    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", planned=len(commands))
    terminal = run_categorized_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal),
            CANDIDATE_PATH.as_posix(),
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )
    receipts = [*affected, *[dict(row) for row in terminal]]
    return receipts, all(row.get("passed") is True for row in terminal)


def _finish_candidate(
    root: Path,
    artifact: JsonDict,
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> int:  # pragma: no cover
    receipts, terminal_ok = _run_terminal_validation(root, artifact, affected, spans, started)
    if not terminal_ok and artifact.get("verdict_class") != "blocked":
        artifact["honest_verdict"] = "complete_disqualified_required_terminal_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
    _refresh_artifact(
        artifact,
        receipts=receipts,
        spans=spans,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(artifact, require_validation=True)
    if errors:
        raise NativePilotError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publish", "before_atomic_write", path=RESULT_PATH.as_posix())
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", "after_atomic_write", bytes=(root / RESULT_PATH).stat().st_size)
    return 0


def _no_run_disqualified(
    failed_gate: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:  # pragma: no cover
    artifact = build_blocked_artifact(
        failed_gate=failed_gate,
        preconditions=checks,
        source_hashes=hashes,
        phase_spans=spans,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"complete_disqualified_{failed_gate.get('check')}"
    artifact["verdict_class"] = "disqualified"
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - live orchestration.
    """Validate, acquire one GPU, run 72 forwards, and publish exact evidence."""

    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(started, "preconditions", "before_named_inputs")
    checks, hashes, context = collect_preconditions(root, started, run_date)
    progress(started, "preconditions", "after_named_inputs", completed_units=len(checks))
    spans.append(_phase_span("preconditions", phase_started, started, len(checks), "authenticated"))
    failed = next((row for row in checks if row.get("passed") is not True), None)

    phase_started = time.monotonic()
    affected, affected_ok = _run_affected_validation(root, started)
    spans.append(
        _phase_span(
            "affected_validation",
            phase_started,
            started,
            len(affected),
            "scoped_commands_complete",
        )
    )
    if not affected_ok:
        validation_gate = gate(
            "required_affected_validation",
            "validity",
            True,
            False,
            "is",
            False,
            "Required scoped checks must pass before model work.",
            upstream="affected_validation_receipts",
            field="all_required_checks_passed",
        )
        artifact = _no_run_disqualified(
            validation_gate,
            checks=checks,
            hashes=hashes,
            spans=spans,
            duration_s=time.monotonic() - started,
        )
        return _finish_candidate(root, artifact, affected, spans, started)
    if failed is not None:
        artifact = build_blocked_artifact(
            failed_gate=failed,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        return _finish_candidate(root, artifact, affected, spans, started)

    phase_started = time.monotonic()
    progress(started, "capture_schedule", "before_sealed_role_wrapper")
    captures, capture_manifest = _sealed_capture_groups(context, root=root, started=started)
    progress(
        started,
        "capture_schedule",
        "after_sealed_role_wrapper",
        completed_units=CAPTURE_GROUPS * 2,
    )
    spans.append(
        _phase_span(
            "capture_schedule",
            phase_started,
            started,
            CAPTURE_GROUPS * 2,
            "fit_and_evaluation_schedules_frozen",
        )
    )

    progress(started, "gpu_admission", "before_wait", wait_limit_s=native.ADMISSION_WAIT_S)
    phase_started = time.monotonic()
    selected_gpu, admission = native.wait_for_owned_gpu(started)
    progress(
        started,
        "gpu_admission",
        "after_wait",
        selected_uuid=selected_gpu.get("uuid") if selected_gpu else None,
        waited_s=admission["waited_s"],
    )
    spans.append(
        _phase_span(
            "gpu_admission",
            phase_started,
            started,
            int(selected_gpu is not None),
            "one_gpu_selected" if selected_gpu else "wait_closed",
        )
    )
    admission_check = _contract_gate(
        "owned_gpu_available",
        True,
        selected_gpu is not None,
        upstream="nvidia-smi",
        field="admissible_single_gpu",
        path="nvidia-smi_compute_process_inventory",
    )
    checks.append(admission_check)
    if selected_gpu is None:
        artifact = build_blocked_artifact(
            failed_gate=admission_check,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["gpu_admission_receipt"] = admission
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)

    model_path = Path(context["model_path"])
    lease_dir = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
    progress(started, "gpu_lease", "before_acquire", device_uuid=selected_gpu["uuid"])
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=lease_dir,
            task_id=EXPERIMENT_ID,
            device_uuid=str(selected_gpu["uuid"]),
            expected_model=str(model_path),
            vram_before_mb=int(selected_gpu["memory_used_mb"]),
            ttl_s=4800.0,
        )
    except lease_api.LeaseError as exc:
        progress(started, "gpu_lease", "after_acquire", acquired=False, error=str(exc))
        lease_gate = _contract_gate(
            "fresh_exclusive_gpu_lease",
            True,
            False,
            upstream=str(lease_dir),
            field="exclusive_device_lease_acquired",
            path=str(lease_dir),
        )
        checks.append(lease_gate)
        artifact = build_blocked_artifact(
            failed_gate=lease_gate,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["lease_error"] = f"{type(exc).__name__}:{exc}"
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)
    progress(started, "gpu_lease", "after_acquire", acquired=True, lease_id=lease.lease_id)
    owner = lease.owner_receipt()

    inventory = native._gpu_inventory()
    rechecked = native._admissible_gpu(inventory)
    capacity_stable = bool(rechecked and rechecked.get("uuid") == selected_gpu.get("uuid"))
    capacity_gate = _contract_gate(
        "capacity_recheck_before_load",
        str(selected_gpu["uuid"]),
        rechecked.get("uuid") if rechecked else None,
        upstream="nvidia-smi",
        field="admissible_gpu_uuid",
        path="nvidia-smi_compute_process_inventory",
    )
    checks.append(capacity_gate)
    if not capacity_stable:
        lease.transition("terminal_blocked")
        release = lease.release()
        artifact = build_blocked_artifact(
            failed_gate=capacity_gate,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["gpu_admission_receipt"] = {
            **admission,
            "capacity_recheck_inventory": inventory,
            "lease_release": release,
        }
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)

    runner = NativeOptionRunner(model_path, int(selected_gpu["index"]), str(selected_gpu["uuid"]))
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    schedule: list[JsonDict] = []
    panel: JsonDict = {}
    model_identity: JsonDict = {}
    ownership: JsonDict = {}
    schedule_receipt: JsonDict = {}
    rows_receipt: JsonDict = {}
    reduction: JsonDict = {}
    checkpoint_seconds: list[float] = []
    load_seconds = 0.0
    live_error: str | None = None
    inference_ok = False
    release: JsonDict = {}

    try:
        lease.transition("admitted")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
        phase_started = time.monotonic()
        progress(started, "panel_freeze", "before_vocab_selection")
        panel, schedule, tokenizer_identity = native._call_with_heartbeats(
            lambda: native._load_panel_and_schedule(
                excluded_role_hashes=set(context["sealed"]["excluded_role_hashes"]),
                started=started,
            ),
            started=started,
            phase="panel_freeze",
            pending="vocab_tokenization_and_public_component_freeze",
        )
        schedule_receipt = native._write_jsonl(root / RAW_DIR / "forward_schedule.jsonl", schedule)
        progress(
            started,
            "panel_freeze",
            "after_vocab_selection",
            completed_units=len(panel["groups"]),
            forwards=len(schedule),
        )
        spans.append(
            _phase_span(
                "panel_freeze",
                phase_started,
                started,
                len(panel["groups"]),
                schedule_receipt["path"],
            )
        )

        lease.transition("loading")
        phase_started = time.monotonic()
        progress(started, "model_load", "before", gpu_uuid=selected_gpu["uuid"])
        events.append(_event("model_load", "attempted", "qwen-load"))
        load_started = time.monotonic()
        try:
            model_identity = native._call_with_heartbeats(
                lambda: runner.load(model_hash=str(context["model_hash"]), owner=owner),
                started=started,
                phase="model_load",
                pending="qwen_gguf_weight_load",
            )
        except BaseException:
            events.append(_event("model_load", "failed", "qwen-load"))
            raise
        load_seconds = time.monotonic() - load_started
        events.append(_event("model_load", "completed", "qwen-load"))
        metadata = dict(getattr(runner.llm, "metadata", {}) or {})
        chat_template = metadata.get("tokenizer.chat_template")
        cuda_build = bool(runner.backend.llama_supports_gpu_offload())
        if not chat_template:
            raise NativePilotError("native_chat_template_missing")
        if not cuda_build:
            raise NativePilotError("cuda_build_missing")
        owned_vram = native._owned_vram_mb(os.getpid(), str(selected_gpu["uuid"]))
        block_count = model_identity.get("model_block_count")
        offload_real = isinstance(owned_vram, int) and owned_vram >= 15_000
        observed_layers = int(block_count) if offload_real and isinstance(block_count, int) else 0
        model_identity["native_chat_template"] = {
            "present": True,
            "sha256": sha256_text(str(chat_template)),
        }
        model_identity["cuda_build"] = cuda_build
        model_identity["observed_offload"] = {
            "gpu_uuid": selected_gpu["uuid"],
            "owned_pid": os.getpid(),
            "owned_pid_vram_mb": owned_vram,
            "offloaded_layers": observed_layers,
            "passed": offload_real and observed_layers > 0,
        }
        ownership = {
            **deepcopy(owner),
            "selected_gpu_index": selected_gpu["index"],
            "selected_gpu_uuid": selected_gpu["uuid"],
            "selected_gpu_name": selected_gpu["name"],
            "physical_gpu_uuid": selected_gpu["uuid"],
            "baseline_memory_used_mb": selected_gpu["memory_used_mb"],
            "owned_pid_vram_mb": owned_vram,
            "measured_owned_offloaded_layers": observed_layers,
            "offload_real": offload_real,
            "cuda_build": cuda_build,
            "admission_receipt": admission,
            "capacity_recheck_inventory": inventory,
            "canonical_lease_directory": str(lease_dir),
            "tokenizer_identity": tokenizer_identity,
            "signals_sent": [],
        }
        lease.transition("resident", vram_mb=int(owned_vram or 0))
        lease.transition("inferencing")
        progress(
            started,
            "model_load",
            "after",
            load_seconds=load_seconds,
            owned_vram_mb=owned_vram,
            offloaded_layers=observed_layers,
        )
        spans.append(_phase_span("model_load", phase_started, started, 1, "qwen_resident"))
        if not offload_real or observed_layers <= 0:
            raise NativePilotError("owned_cuda_offload_not_observed")

        phase_started = time.monotonic()
        progress(started, "native_forwards", "before_loop", planned=PILOT_FORWARDS)
        for index, request in enumerate(schedule):
            call_id = f"native-{index:03d}"
            progress(
                started,
                "native_forwards",
                "before_forward",
                completed_units=index,
                planned=PILOT_FORWARDS,
                call_id=call_id,
            )
            events.append(_event("forward", "attempted", call_id))
            try:
                row = runner.score(request, owner)
            except BaseException as exc:  # noqa: BLE001 - exact failed call evidence.
                row = {
                    **deepcopy(dict(request)),
                    "call_id": call_id,
                    "disposition": "failed",
                    "reply": "",
                    "reply_utf8_bytes": 0,
                    "generated_tokens": 0,
                    "error": f"{type(exc).__name__}:{exc}",
                    "server_receipt": {
                        "transport": "native_in_process",
                        "pid": owner.get("pid"),
                        "process_start_ticks": owner.get("pid_start_ticks"),
                        "gpu_uuid": selected_gpu["uuid"],
                        "n_ctx": N_CTX,
                    },
                }
                events.append(_event("forward", "failed", call_id))
            else:
                row["reply"] = ""
                row["reply_utf8_bytes"] = 0
                row["prompt_encoding"] = "utf-8"
                row["reply_encoding"] = "utf-8"
                row["complete_source_window"] = row["prompt"]
                row["complete_response_window"] = ""
                events.append(_event("forward", "completed", call_id))
            rows.append(row)
            checkpoint_started = time.monotonic()
            atomic_json(
                root / CHECKPOINT_PATH,
                {
                    "schema": "carnot.exp7563.v661.checkpoint.v1",
                    "panel_sha256": panel["panel_sha256"],
                    "model_sha256": context["model_hash"],
                    "completed_units": len(rows),
                    "rows_sha256": canonical_hash(rows),
                },
            )
            checkpoint_seconds.append(time.monotonic() - checkpoint_started)
            lease.heartbeat()
            progress(
                started,
                "native_forwards",
                "after_forward",
                completed_units=len(rows),
                planned=PILOT_FORWARDS,
                disposition=row["disposition"],
            )
        rows_receipt = native._write_jsonl(root / ROWS_PATH, rows)
        reduction = native.reduce_native_rows(rows, schedule)
        inference_ok = reduction.get("passed") is True
        progress(
            started,
            "native_forwards",
            "after_loop",
            completed_units=len(rows),
            custody_passed=inference_ok,
        )
        spans.append(
            _phase_span(
                "native_forwards",
                phase_started,
                started,
                len(rows),
                rows_receipt["path"],
            )
        )
    except BaseException as exc:  # noqa: BLE001 - owned failure becomes terminal evidence.
        live_error = f"{type(exc).__name__}:{exc}"
        progress(started, "live_measurement", "error", error=live_error)
    finally:
        progress(started, "model_unload", "before")
        runner.close()
        gc.collect()
        progress(started, "model_unload", "after")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_vram = _wait_for_owned_unload(
                os.getpid(), str(selected_gpu["uuid"]), started=started
            )
            unload_observed = after_vram is None or after_vram < native.GPU_IDLE_MAX_USED_MB
            ownership["post_close_owned_vram_mb"] = after_vram
            ownership["model_unload_threshold_mb"] = native.GPU_IDLE_MAX_USED_MB
            ownership["model_unload_observed"] = unload_observed
            lease.transition(
                "validating",
                vram_mb=int(after_vram or 0),
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition("terminal_complete" if inference_ok else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
        ownership["release"] = release

    if not events:
        no_start_gate = _contract_gate(
            "model_load_never_started",
            True,
            False,
            upstream="owned_live_measurement",
            field="model_load.attempted",
        )
        checks.append(no_start_gate)
        artifact = build_blocked_artifact(
            failed_gate=no_start_gate,
            preconditions=checks,
            source_hashes=hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["live_error"] = live_error
        artifact["ownership_receipt"] = {**owner, "release": release}
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)

    if schedule and not schedule_receipt:
        schedule_receipt = native._write_jsonl(root / RAW_DIR / "forward_schedule.jsonl", schedule)
    if not rows_receipt:
        rows_receipt = native._write_jsonl(root / ROWS_PATH, rows)
    if schedule and not reduction:
        reduction = native.reduce_native_rows(rows, schedule)
    if not reduction:
        reduction = {
            "passed": False,
            "planned_forward_count": PILOT_FORWARDS,
            "attempted_forward_count": len(rows),
            "complete_forward_count": 0,
            "error_forward_count": len(rows),
            "generated_token_count": 0,
            "failed_checks": ["live_measurement_before_schedule"],
            "comparative_rows": [],
            "benefit_gate_applied": False,
        }
    checkpoint_projection = (
        CAPTURE_FORWARDS * native._p95(checkpoint_seconds)
        if checkpoint_seconds
        else ACQUISITION_LIMIT_S
    )
    try:
        forecasts = forecast_captures(
            load_seconds=load_seconds,
            pilot_rows=rows,
            fit_token_lengths=captures["fit"]["token_lengths"],
            evaluation_token_lengths=captures["evaluation"]["token_lengths"],
            fit_checkpoint_seconds=checkpoint_projection,
            evaluation_checkpoint_seconds=checkpoint_projection,
        )
    except NativePilotError as exc:
        forecasts = {
            "formula": "load + sum(p95_forward * tokens / p95_pilot_tokens) + checkpoint",
            "fit": {"planned_groups": CAPTURE_GROUPS, "planned_forwards": CAPTURE_FORWARDS},
            "evaluation": {
                "planned_groups": CAPTURE_GROUPS,
                "planned_forwards": CAPTURE_FORWARDS,
            },
            "fit_capture_feasible_score": 0,
            "eval_capture_feasible_score": 0,
            "validation_reserve_seconds": VALIDATION_RESERVE_S,
            "closed_reason": str(exc),
        }
    if not ownership:
        ownership = {
            **owner,
            "selected_gpu_uuid": selected_gpu["uuid"],
            "offload_real": False,
            "canonical_lease_directory": str(lease_dir),
            "signals_sent": [],
            "release": release,
        }
    if not model_identity:
        model_identity = {
            "hf_id": MODEL_ID,
            "model_path": str(model_path),
            "model_sha256": context["model_hash"],
            "quantization": "Q4_K_M",
            "runtime": context["runtime"],
            "load_error": live_error,
        }
    raw_receipts = {
        "forward_schedule": schedule_receipt,
        "native_rows": rows_receipt,
        "all_below_20_mib": all(
            int(receipt.get("bytes") or 0) < 20 * 1024 * 1024
            for receipt in (schedule_receipt, rows_receipt)
        ),
    }
    if not raw_receipts["all_below_20_mib"]:
        live_error = live_error or "raw_sidecar_size_limit"
    pilot_cost_rows: list[JsonDict] = [
        {"kind": "model_load", "full_prompt_length": 0, "measured_seconds": load_seconds}
    ]
    pilot_cost_rows.extend(
        {
            "kind": "native_forward",
            "call_id": row.get("call_id"),
            "full_prompt_length": row.get("prompt_token_count"),
            "measured_seconds": row.get("forward_seconds"),
            "disposition": row.get("disposition"),
        }
        for row in rows
    )
    artifact = build_complete_artifact(
        preconditions=checks,
        source_hashes=hashes,
        model_specs=[model_identity],
        events=events,
        panel=panel,
        raw_receipts=raw_receipts,
        reduction=reduction,
        forecasts=forecasts,
        ownership=ownership,
        pilot_cost_rows=pilot_cost_rows,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_validation=False,
    )
    artifact["role_manifest"]["sealed_capture"] = capture_manifest
    artifact["source_label_access"] = captures["source_label_access"]
    if live_error is not None:
        artifact["live_error"] = live_error
        artifact["honest_verdict"] = "complete_disqualified_owned_live_measurement"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _finish_candidate(root, artifact, affected, spans, started)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the live command and the two read-only replay modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the owned pilot or inspect one exact candidate without mutation."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        print(
            json.dumps(
                {
                    "mode": "argument_check",
                    "passed": False,
                    "expected": RUN_DATE,
                    "observed": args.date,
                }
            ),
            flush=True,
        )
        return 1
    if args.cold_replay is not None:
        artifact = _load_object(_argument_path(args.cold_replay))
        errors = validate_artifact(artifact, require_validation=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = _load_object(_argument_path(args.independent_reduce))
        reduction = independent_reduce(artifact)
        print(json.dumps({"mode": "independent_reduce", **reduction}, sort_keys=True), flush=True)
        return int(reduction.get("passed") is not True)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
