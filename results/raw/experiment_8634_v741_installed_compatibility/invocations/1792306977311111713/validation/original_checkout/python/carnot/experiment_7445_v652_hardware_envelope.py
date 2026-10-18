"""Bound future hardware placement with archived complete-service evidence.

This experiment is a host-only aggregation. It reads authenticated Exp7432 and
Exp7440 artifacts, recomputes the persistence bound, and preserves three board
dispositions. It never contacts a board or reruns the sparse/int16 benchmark.

Spec refs: REQ-REPORT-7445 and SCENARIO-REPORT-7445-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7367_v646_board_disposition as board_history
from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
PHASE = 4
EXPERIMENT_ID = "exp7445-v652-hardware-envelope"
SCHEMA = "carnot.exp7445.v652.hardware_envelope.v1"
RESULT_PATH = Path("results/experiment_7445_v652_hardware_envelope.json")
RAW_DIR = Path("results/raw/experiment_7445_v652_hardware_envelope")
MODULE_PATH = Path("python/carnot/experiment_7445_v652_hardware_envelope.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7445_v652_hardware_envelope.py")
TEST_PATH = Path("tests/python/test_experiment_7445_v652_hardware_envelope.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP7432_PATH = Path("results/experiment_7432_v651_update_placement.json")
EXP7432_RAW_PATH = Path("results/raw/experiment_7432_v651_update_placement/raw_evidence.json")
EXP7433_PATH = Path("results/experiment_7433_v651_capstone.json")
EXP7440_PATH = Path("results/experiment_7440_v652_mixture_learning.json")
EXP6559_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")

PERSISTENCE_FRACTION = 0.37696053267914603
TARGET_SPEEDUP = 100.0
TARGET_RESIDUAL_FRACTION = 1.0 / TARGET_SPEEDUP
REQUIRED_STAGES = (
    "mixture_prediction",
    "numeric_update",
    "feedback",
    "hash_checkpoint",
    "durable_persistence",
)
GATEMATE_MISSING_RECEIPT = board_history.MISSING_RECEIPT
ZERO_INVOCATION_COUNTS = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
EXPECTED_UPSTREAM_HASHES = {
    EXP7432_PATH: "sha256:85b3f05a10301bdb9320108fbed74220a8a2f742cde2c2852fe11e7f2bea6977",
    EXP7432_RAW_PATH: "sha256:2ee71910db2fcef5929c04490cbb9c46cd9a40193b4af9b06d57fde4c5bf384e",
    EXP7433_PATH: "sha256:5226140221dbc04fe300ce706278ccee1abc914b017945ca46d7f423636548a8",
    EXP7440_PATH: "sha256:1c57c990b4ae890693ced35f1db0de4d5ce77f932a941e8b104d7b5e5ae630eb",
    EXP6559_PATH: "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7432_v651_update_placement.py"),
    EXP7432_PATH,
    EXP7432_RAW_PATH,
    EXP7433_PATH,
    Path("python/carnot/experiment_7440_v652_mixture_learning.py"),
    EXP7440_PATH,
    EXP6559_PATH,
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_cold_reduce",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint_e2e",
)


def utc_now() -> str:
    """Return one aware UTC boundary for a measured phase."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a phase or slow-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7445] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json(path: Path) -> JsonDict:
    """Read one JSON object; malformed, missing, or list-shaped bytes stay absent."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def gate_row(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    path: str | None,
    field: str,
) -> JsonDict:
    """Keep every gate operand plain and machine-readable."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _identity_gate(
    source: Mapping[str, Any],
    *,
    label: str,
    path: Path,
    field: str,
    expected: Any,
) -> JsonDict:
    """Compare one upstream identity field without merging producer meanings."""

    observed = source.get(field)
    if expected == "complete_*":
        passed = isinstance(observed, str) and observed.startswith("complete_")
        operator = "starts_with"
    else:
        passed = observed == expected
        operator = "=="
    return gate_row(
        f"{label.lower()}_{field}",
        "source_authentication",
        operator,
        expected,
        observed,
        passed,
        "Only the exact terminal producer identity and original flags may enter the audit.",
        upstream=label,
        path=path.as_posix(),
        field=field,
    )


def authenticate_upstreams(
    exp7432: Mapping[str, Any], exp7440: Mapping[str, Any]
) -> list[JsonDict]:
    """Authenticate both producers independently, including original flags."""

    contracts = (
        (
            "Exp7432",
            EXP7432_PATH,
            exp7432,
            {
                "experiment_id": "exp7432-v651-update-placement",
                "milestone": "2026.09.651",
                "status": "complete_*",
                "verdict_class": "null",
                "flagged_adversarial": False,
            },
        ),
        (
            "Exp7440",
            EXP7440_PATH,
            exp7440,
            {
                "experiment_id": "exp7440-v652-mixture-learning",
                "milestone": MILESTONE,
                "status": "complete_*",
                "verdict_class": "null",
                "flagged_adversarial": False,
            },
        ),
    )
    rows: list[JsonDict] = []
    for label, path, source, expected_fields in contracts:
        rows.extend(
            _identity_gate(
                source,
                label=label,
                path=path,
                field=field,
                expected=expected,
            )
            for field, expected in expected_fields.items()
        )
    return rows


def compute_amdahl_rows(fraction: float) -> list[JsonDict]:
    """Compute the infinite-acceleration ceiling and strict 100x condition."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("persistence_fraction_out_of_range")
    ceiling = 1.0 / fraction
    base = {
        "row_type": "amdahl_bound",
        "source_group": "Exp7432_complete_service_persistence",
        "condition": "idealized_infinite_acceleration_of_all_other_work",
        "seed": None,
        "disposition": "complete",
        "censored": False,
        "failed": False,
        "is_new_speed_measurement": False,
    }
    return [
        {
            **base,
            "unit_id": "amdahl:infinite_acceleration_ceiling",
            "bound": "infinite_acceleration_ceiling",
            "formula": "1 / f",
            "observed_unaccelerated_fraction": fraction,
            "service_speed_limit_x": ceiling,
            "interpretation": "upper_bound_not_new_speed_measurement",
        },
        {
            **base,
            "unit_id": "amdahl:one_hundred_x_condition",
            "bound": "one_hundred_x_condition",
            "formula": "f < 1 / target_speedup",
            "target_service_speedup_x": TARGET_SPEEDUP,
            "operator": "<",
            "required_unaccelerated_fraction": TARGET_RESIDUAL_FRACTION,
            "observed_unaccelerated_fraction": fraction,
            "condition_met": fraction < TARGET_RESIDUAL_FRACTION,
            "interpretation": "necessary_idealized_condition_not_sufficient_measurement",
        },
    ]


def _valid_complete_stages(rows: Any) -> list[JsonDict]:
    """Return complete stage rows only when the five-stage set is exact."""

    if not isinstance(rows, list) or len(rows) != len(REQUIRED_STAGES):
        return []
    copied = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(copied) != len(REQUIRED_STAGES):
        return []
    stages = [row.get("stage") for row in copied]
    durations = [row.get("duration_ns") for row in copied]
    valid_durations = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        for value in durations
    )
    if (
        set(stages) != set(REQUIRED_STAGES)
        or len(stages) != len(set(stages))
        or not all(row.get("complete_service") is True for row in copied)
        or not valid_durations
    ):
        return []
    return copied


def reduce_stage_envelope(exp7432: Mapping[str, Any], exp7440: Mapping[str, Any]) -> JsonDict:
    """Use Exp7440 only for a real full decomposition; otherwise retain V651."""

    complete = _valid_complete_stages(exp7440.get("complete_service_stage_rows"))
    hardware = exp7440.get("hardware_path")
    hardware_path = dict(hardware) if isinstance(hardware, Mapping) else {}
    expert_count = hardware_path.get("expert_count")
    prediction_ns = hardware_path.get("prediction_duration_ns")
    normalization_ns = hardware_path.get("weight_normalization_duration_ns")
    summary = exp7432.get("timing_summary")
    timing_rows = [dict(row) for row in summary or [] if isinstance(row, Mapping)]
    return {
        "authority": (
            "Exp7440_complete_service_stage_rows"
            if complete
            else "Exp7432_V651_complete_service_envelope"
        ),
        "complete_stage_decomposition_available": bool(complete),
        "stage_rows": deepcopy(complete),
        "unavailable_stages": [] if complete else list(REQUIRED_STAGES),
        "unavailable_stage_reason": (
            None
            if complete
            else "Exp7440 exposes aggregate durations, not five complete-service stage rows"
        ),
        "complete_service_timing_rows": timing_rows,
        "unchanged_sparse_or_int16_benchmark_rerun": False,
        "exp7440_aggregate_durations_ns": {
            key: hardware_path.get(key)
            for key in (
                "prediction_duration_ns",
                "feedback_update_duration_ns",
                "persistence_duration_ns",
                "read_duration_ns",
            )
        },
        "expert_accounting": {
            "expert_count": expert_count,
            "four_expert_prediction_cost_ns": prediction_ns if expert_count == 4 else None,
            "four_expert_prediction_cost_scope": (
                "Exp7440 aggregate prediction duration across its measured event ledger"
            ),
            "weight_normalization_cost_ns": normalization_ns,
            "weight_normalization_status": (
                "measured_complete_stage"
                if normalization_ns is not None
                else "included_in_aggregate_prediction_or_update_path_not_isolated"
            ),
            "isolated_benchmark_claimed": False,
        },
    }


def build_board_rows(
    historical_rows: Sequence[Mapping[str, Any]], changed_state: Mapping[str, Any]
) -> list[JsonDict]:
    """Preserve three historical dispositions and apply only the read-only gate."""

    by_name = {str(row.get("board")): deepcopy(dict(row)) for row in historical_rows}
    if set(by_name) != {"KV260", "GateMate", "PolarFire"}:
        raise ValueError("board_rows_invalid")
    rows = [by_name[name] for name in ("KV260", "GateMate", "PolarFire")]
    for row in rows:
        row["source_row_sha256"] = row.pop("row_sha256", None)
        row.update(
            {
                "current_execution_venue": "host",
                "read_only_audit": True,
                "new_hardware_execution_claimed": False,
                "present_reachability_asserted": False,
                "hardware_operations_issued": [],
                "hardware_ready_score": 0,
                "hardware_value_score": 0,
                "censored": False,
                "failed": False,
                "disposition": "complete",
            }
        )

    kv260, gate, polarfire = rows
    kv260.update(
        {
            "terminal_state": "graduated_preserved",
            "future_access": "ssh_only",
            "access_mechanism": "ssh kria only",
        }
    )
    changed = changed_state.get("exists") is True
    accepted = changed_state.get("accepted_receipt_count", 0)
    gate.update(
        {
            "terminal_state": (
                "changed_state_future_task_eligible"
                if changed
                else "blocked_changed_physical_state"
            ),
            "honest_verdict": (
                "complete_changed_state_future_task_eligible_no_hardware_execution"
                if changed
                else "blocked_missing_post_exp6559_operator_physical_change"
            ),
            "error": None if changed else GATEMATE_MISSING_RECEIPT,
            "metric": changed,
            "disposition": "complete" if changed else "blocked",
            "changed_state_receipt_path": changed_state.get("search_receipt_path"),
            "changed_state_receipt_hash": changed_state.get("search_receipt_hash"),
            "eligibility_contract": deepcopy(changed_state.get("eligibility_contract")),
            "gate_check_summary": {
                "upstream": "Exp6559 operator changed-state boundary",
                "path": changed_state.get("search_receipt_path"),
                "check": "dated_operator_cable_port_power_board_or_dirtyjtag_change",
                "field": "accepted_receipt_count",
                "operator": ">",
                "expected": 0,
                "observed": accepted,
                "passed": changed,
            },
        }
    )
    polarfire.update(
        {
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "graduated_scope": "hash_matched_cpu_dispatch",
            "fpga_sampling_claimed": False,
            "fpga_sampling_status": "separate_unmeasured_future_task",
        }
    )
    for row in rows:
        row["row_sha256"] = current_work_receipt.canonical_hash(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
    return rows


def external_future_rows() -> list[JsonDict]:
    """Name the exact evidence needed before each external option is justified."""

    requirements = (
        (
            "Extropic Z1T/TSU",
            "authorized device or vendor evaluation access",
            "end-to-end acknowledged service timing showing the numeric path dominates host persistence",
        ),
        (
            "photonic",
            "programmable photonic Ising hardware and a reproducible mapping toolchain",
            "mapped complete-service latency and accuracy against the measured host residual",
        ),
        (
            "D-Wave",
            "authorized Leap QPU access with queue and embedding receipts",
            "QPU plus embedding/readout service cost below the acknowledged host path",
        ),
        (
            "NPU",
            "a supported runtime such as the AMD VitisAI-enabled ONNX Runtime build",
            "an isolated numeric expert/update bottleneck large enough to exceed transfer overhead",
        ),
        (
            "larger FPGA",
            "board and synthesis access beyond the preserved KV260 k_max<=5 limit",
            "a workload that exceeds KV260 capacity with complete acknowledged service timing",
        ),
    )
    return [
        {
            "unit_id": f"future:{option.lower().replace(' ', '_').replace('/', '_')}",
            "row_type": "external_future_hardware",
            "source_group": "future_hardware_option",
            "condition": "not_executed",
            "seed": None,
            "option": option,
            "status": "external_future_work",
            "access_required": access,
            "measured_bottleneck_required": bottleneck,
            "justification_required": f"{access}; {bottleneck}",
            "disposition": "unstarted",
            "censored": False,
            "failed": False,
            "hardware_execution_claimed": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        }
        for option, access, bottleneck in requirements
    ]


def learning_hardware_route(persistence_fraction: float) -> JsonDict:
    """Separate bounded accelerator state from acknowledged durable CPU work."""

    return {
        "accelerator_candidates": ["GPU/NPU", "FPGA"],
        "accelerator_scope": [
            "bounded_numeric_expert_state",
            "four_expert_predictions",
            "log_weight_updates",
            "weight_normalization_when_measured",
        ],
        "cpu_scope": [
            "orchestrated_feedback",
            "acknowledgement",
            "hash_checkpoint",
            "durable_state",
            "crash_recovery",
        ],
        "measured_residual_service_bottleneck_fraction": persistence_fraction,
        "idealized_numeric_acceleration_ceiling_x": 1.0 / persistence_fraction,
        "target_service_speedup_x": TARGET_SPEEDUP,
        "requires_changed_persistence_orchestration_design_for_100x": True,
        "equivalent_acknowledgement_and_crash_semantics_required": True,
        "durability_trade_allowed": False,
        "new_hardware_measurement": False,
    }


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failed operand without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else {}
    return {
        "all_passed": not failed,
        "failed_checks": [row.get("check") for row in failed],
        "first_failed_check": first.get("check"),
        "first_failed_upstream": first.get("upstream"),
        "first_failed_path": first.get("path"),
        "first_failed_field": first.get("field"),
        "first_failed_expected": deepcopy(first.get("expected")),
        "first_failed_observed": deepcopy(first.get("observed")),
    }


def _required_receipts(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every exact command."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _acceptance_gates(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    amdahl_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_required: bool,
    require_terminal: bool,
) -> list[JsonDict]:
    """Keep validity, accounting, and the failed 100x benefit gate separate."""

    source_rows = [row for row in preconditions if row.get("category") != "external_prerequisite"]
    target = next((row for row in amdahl_rows if row.get("bound") == "one_hundred_x_condition"), {})
    required_names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if require_terminal:
        required_names.extend(TERMINAL_CHECK_NAMES)
    validation_ok = (
        _required_receipts(validation_receipts, required_names) if validation_required else True
    )
    return [
        gate_row(
            "authenticated_required_sources",
            "validity",
            "all",
            True,
            all(row.get("passed") is True for row in source_rows),
            all(row.get("passed") is True for row in source_rows),
            "Every dependent reduction uses authenticated exact bytes and producer flags.",
            upstream="current_preconditions",
            path=None,
            field="preconditions_checked",
        ),
        gate_row(
            "amdahl_bound_recomputed",
            "validity",
            "==",
            1.0 / PERSISTENCE_FRACTION,
            next(
                (
                    row.get("service_speed_limit_x")
                    for row in amdahl_rows
                    if row.get("bound") == "infinite_acceleration_ceiling"
                ),
                None,
            ),
            any(
                row.get("bound") == "infinite_acceleration_ceiling"
                and row.get("service_speed_limit_x") == 1.0 / PERSISTENCE_FRACTION
                for row in amdahl_rows
            ),
            "The residual fraction supplies a mathematical ceiling, not a new timing result.",
            upstream="Exp7432",
            path=EXP7432_PATH.as_posix(),
            field="amdahl_rows.service_speed_limit_x",
        ),
        gate_row(
            "three_board_dispositions",
            "accounting",
            "==",
            ["GateMate", "KV260", "PolarFire"],
            sorted(str(row.get("board")) for row in board_rows),
            sorted(str(row.get("board")) for row in board_rows)
            == ["GateMate", "KV260", "PolarFire"],
            "Graduated boards and physical blockers remain independently visible.",
            upstream="Exp7432 board receipts",
            path=EXP7432_PATH.as_posix(),
            field="board_rows.board",
        ),
        gate_row(
            "required_validation_passed",
            "validity",
            "==",
            True,
            validation_ok,
            validation_ok,
            "Only the frozen affected and fresh terminal readers can validate publication.",
            upstream="current_scoped_validation",
            path=None,
            field="validation_receipts",
        ),
        gate_row(
            "one_hundred_x_architectural_condition",
            "benefit",
            "==",
            True,
            target.get("condition_met"),
            target.get("condition_met") is True,
            "A 100x service route requires an unaccelerated fraction strictly below 0.01.",
            upstream="Exp7432 persistence fraction",
            path=EXP7432_PATH.as_posix(),
            field="amdahl_rows.one_hundred_x_condition.condition_met",
        ),
    ]


def _field_principles() -> JsonDict:
    """Explain required top-level fields without wrapping their scalar values."""

    return {
        "schema": "Use one versioned plain top-level contract.",
        "run_date": "Bind the requested date and real UTC/monotonic boundaries.",
        "preconditions_checked": "Name every exact source, identity, flag, and hash check.",
        "MODEL_SPECS": "Current LLM declarations are empty because none is invoked.",
        "model_invoked": "Current attempted model use is separate from archived evidence.",
        "invocation_counts": "Reconcile every current load and generation state.",
        "inference_substrate": "Describe the current host aggregation plainly.",
        "inference_substrate_class": "Declare aggregation so no model duration floor applies.",
        "execution_venue": "Keep host, CPU, CUDA, and external-device identities distinct.",
        "duration_s": "Measure current wall work and separate validation from model time.",
        "phase_spans": "Bind flushed phases to monotonic checkpoints.",
        "random_seed": "Null is valid because no fitting, sampling, or resampling occurs.",
        "reproducibility_checksum": "Bind code, protocol, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Preserve byte identity and original source flags.",
        "rows": "Keep each bound, board, timing envelope, and future option inspectable.",
        "sample_size_budget": "Separate planned, attempted, complete, blocked, and unstarted units.",
        "acceptance_gate_results": "Keep validity and failed benefit gates distinct.",
        "gate_check_summary": "Name exact operands for the first failure and all failed checks.",
        "verifier_is_oracle": "No deployed verifier scores hardware benefit in this audit.",
        "honest_verdict": "A completed valid no-benefit audit is null, not partial.",
        "verdict_class": "Use the closed terminal classification vocabulary.",
        "flagged_adversarial": "Critical validation findings cannot supply readiness.",
        "validation_receipts": "Retain exact argv, environment, exit, duration, and log hashes.",
        "promotion_score": "This milestone authorizes no automatic rollout or publication.",
        "amdahl_rows": "Show measured fractions, formulas, assumptions, and mathematical limits.",
        "board_rows": "Keep KV260, GateMate, and PolarFire dispositions independent.",
        "hardware_ready_score": "Zero because no new device workload is executed.",
        "hardware_value_score": "Zero because host bounds are not hardware performance.",
        "learning_hardware_route": "Separate bounded numeric acceleration from durable CPU service.",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable code/input identities, raw rows, gates, and exact command scope."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "source_artifact_hashes",
        "random_seed",
        "rows",
        "row_shards",
        "amdahl_rows",
        "stage_envelope",
        "board_rows",
        "external_future_rows",
        "learning_hardware_route",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "validation_manifest",
        "validation_receipts",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "hardware_ready_score",
        "hardware_value_score",
        "promotion_score",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
    )
    return current_work_receipt.canonical_hash({key: deepcopy(artifact.get(key)) for key in keys})


def _current_receipt(
    *,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    sidecar_references: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build truthful zero-call provenance for the current host aggregation."""

    return current_work_receipt.build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_hardware_envelope_aggregation",
        inference_substrate_details={
            "cpu": {
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "cuda": {"used": False, "identity": None, "probed": False},
            "external_device": {"used": False, "identity": None, "probed": False},
            "model_load": False,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": False,
            "current_training": False,
            "receipt_class": "small_ebm_training",
            "source": "archived Exp7440/Exp7439 hash-bound receipt only",
            "current_llm_calls": 0,
        },
    )


def _timing_envelope_rows(stage_envelope: Mapping[str, Any]) -> list[JsonDict]:
    """Label copied V651 summary rows as archived complete-service evidence."""

    return [
        {
            "unit_id": f"service:{row.get('batch_size')}:{row.get('arm')}",
            "row_type": "complete_service_envelope",
            "source_group": "Exp7432_V651",
            "condition": row.get("arm"),
            "seed": None,
            "disposition": "complete",
            "censored": False,
            "failed": False,
            "is_new_speed_measurement": False,
            **deepcopy(dict(row)),
        }
        for row in stage_envelope.get("complete_service_timing_rows", [])
        if isinstance(row, Mapping)
    ]


def assemble_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    amdahl_rows: Sequence[Mapping[str, Any]],
    stage_envelope: Mapping[str, Any],
    board_rows: Sequence[Mapping[str, Any]],
    future_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    sidecar_references: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    validation_required: bool,
    require_terminal: bool,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Assemble one terminal envelope with benefit independent from validity."""

    current = _current_receipt(
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
        sidecar_references=sidecar_references,
    )
    gates = _acceptance_gates(
        preconditions=preconditions,
        amdahl_rows=amdahl_rows,
        board_rows=board_rows,
        validation_receipts=validation_receipts,
        validation_required=validation_required,
        require_terminal=require_terminal,
    )
    valid = all(row["passed"] for row in gates if row["category"] != "benefit")
    if flagged_adversarial or not valid:
        verdict_class = "disqualified"
        honest = "complete_disqualified_hardware_envelope_validation_failed"
    else:
        verdict_class = "null"
        honest = "complete_null_hardware_envelope_requires_persistence_orchestration_redesign"
    timing_rows = _timing_envelope_rows(stage_envelope)
    rows = [
        *deepcopy(list(amdahl_rows)),
        *timing_rows,
        *deepcopy(list(board_rows)),
        *deepcopy(list(future_rows)),
    ]
    blocked = sum(row.get("disposition") == "blocked" for row in rows)
    unstarted = sum(row.get("disposition") == "unstarted" for row in rows)
    complete = len(rows) - blocked - unstarted
    validation_duration = sum(float(row.get("duration_s") or 0.0) for row in validation_receipts)
    artifact: JsonDict = {
        **current,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(preconditions)),
        "random_seed": None,
        "random_seed_reason": "deterministic_pure_reduction_no_fitting_sampling_or_resampling",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "row_shards": [],
        "rows": rows,
        "amdahl_rows": deepcopy(list(amdahl_rows)),
        "stage_envelope": deepcopy(dict(stage_envelope)),
        "board_rows": deepcopy(list(board_rows)),
        "external_future_rows": deepcopy(list(future_rows)),
        "learning_hardware_route": learning_hardware_route(PERSISTENCE_FRACTION),
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows) - unstarted,
            "completed": complete,
            "failed": 0,
            "censored": 0,
            "blocked": blocked,
            "unstarted": unstarted,
            "independent_units": len(rows),
            "stopping_rule": (
                "one deterministic reduction per archived bound, timing arm/batch, board, "
                "and named future option; no benchmark rerun"
            ),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_manifest": {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
            "frozen_before_subprocesses": True,
        },
        "validation_required": validation_required,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "capability_e2e": {
            "declared_entrypoint": "required",
            "fresh_process_cold_replay": "required",
            "numbered_e2e": "not_applicable_isolated_reporting_policy_study",
        },
        "duration_breakdown_s": {
            "model": 0.0,
            "computation": max(0.0, float(current["duration_s"]) - validation_duration),
            "cold_start": 0.0,
            "validation": validation_duration,
        },
        "hardware_operations": {
            "physical_detect": 0,
            "flash": 0,
            "ssh_probe": 0,
            "purchase": 0,
            "external_device_workload": 0,
        },
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "promotion_score": 0,
        "field_principles": _field_principles(),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _fixture_sources() -> tuple[JsonDict, JsonDict]:
    """Build compact producer-shaped values for mutation-focused tests."""

    boards = [
        {
            "unit_id": "board:KV260",
            "row_type": "board_disposition",
            "board": "KV260",
            "terminal_state": "graduated_preserved",
            "future_access": "ssh_only",
            "fpga_sampling_claimed": True,
            "hash_matched_cpu_dispatch": False,
            "metric": True,
            "error": None,
        },
        {
            "unit_id": "board:GateMate",
            "row_type": "board_disposition",
            "board": "GateMate",
            "terminal_state": "blocked_changed_physical_state",
            "future_access": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": False,
            "metric": False,
            "error": GATEMATE_MISSING_RECEIPT,
        },
        {
            "unit_id": "board:PolarFire",
            "row_type": "board_disposition",
            "board": "PolarFire",
            "terminal_state": "graduated_cpu_dispatch_preserved",
            "future_access": None,
            "fpga_sampling_claimed": False,
            "hash_matched_cpu_dispatch": True,
            "metric": True,
            "error": None,
        },
    ]
    exp7432 = {
        "experiment_id": "exp7432-v651-update-placement",
        "milestone": "2026.09.651",
        "status": "complete_null_sparse_update_no_registered_complete_service_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "amdahl_analysis": {"observed_unaccelerated_persistence_fraction": PERSISTENCE_FRACTION},
        "timing_summary": [
            {
                "arm": "float32_sparse",
                "baseline_arm": "float32_dense",
                "batch_size": 1,
                "paired_blocks": 30,
                "whole_service_time_ratio": 0.9937845708690535,
                "ci95_lower": 0.9435061459970505,
                "ci95_upper": 1.0470479326842994,
                "speed_gate_passed": False,
            }
        ],
        "board_rows": boards,
    }
    exp7440 = {
        "experiment_id": "exp7440-v652-mixture-learning",
        "milestone": MILESTONE,
        "status": "complete_null_insufficient_online_benefit",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "hardware_path": {
            "expert_count": 4,
            "prediction_duration_ns": 61_465_757_696,
            "feedback_update_duration_ns": 8_945_762_241,
            "persistence_duration_ns": 748_946_870,
            "read_duration_ns": 14_064_423,
        },
    }
    return exp7432, exp7440


def build_fixture_artifact() -> JsonDict:
    """Build one valid compact artifact without touching repository inputs."""

    exp7432, exp7440 = _fixture_sources()
    preconditions = authenticate_upstreams(exp7432, exp7440)
    amdahl = compute_amdahl_rows(PERSISTENCE_FRACTION)
    stages = reduce_stage_envelope(exp7432, exp7440)
    boards = build_board_rows(
        exp7432["board_rows"],
        {
            "exists": False,
            "accepted_receipt_count": 0,
            "search_receipt_path": "fixture:gatemate-search",
            "search_receipt_hash": "sha256:" + "0" * 64,
            "eligibility_contract": {"receipt_date": ">20260823"},
        },
    )
    return assemble_artifact(
        preconditions=preconditions,
        source_hashes={},
        amdahl_rows=amdahl,
        stage_envelope=stages,
        board_rows=boards,
        future_rows=external_future_rows(),
        validation_receipts=[],
        phase_spans=[],
        sidecar_references=[],
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        validation_required=False,
        require_terminal=False,
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute the bound, board identities, row bundle, and zero scores."""

    amdahl = [row for row in artifact.get("amdahl_rows", []) if isinstance(row, Mapping)]
    ceiling_row = next(
        (row for row in amdahl if row.get("bound") == "infinite_acceleration_ceiling"), {}
    )
    condition_row = next(
        (row for row in amdahl if row.get("bound") == "one_hundred_x_condition"), {}
    )
    fraction = ceiling_row.get("observed_unaccelerated_fraction")
    expected_ceiling = (
        1.0 / float(fraction)
        if isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and float(fraction) > 0
        else None
    )
    boards = [row for row in artifact.get("board_rows", []) if isinstance(row, Mapping)]
    board_names = sorted(str(row.get("board")) for row in boards)
    stage = artifact.get("stage_envelope")
    stage_map = dict(stage) if isinstance(stage, Mapping) else {}
    timing = _timing_envelope_rows(stage_map)
    future = [
        dict(row) for row in artifact.get("external_future_rows", []) if isinstance(row, Mapping)
    ]
    expected_rows = [*deepcopy(amdahl), *timing, *deepcopy(boards), *future]
    scores_zero = all(
        artifact.get(field) == 0
        for field in ("hardware_ready_score", "hardware_value_score", "promotion_score")
    )
    matches = (
        expected_ceiling is not None
        and ceiling_row.get("service_speed_limit_x") == expected_ceiling
        and condition_row.get("operator") == "<"
        and condition_row.get("required_unaccelerated_fraction") == TARGET_RESIDUAL_FRACTION
        and condition_row.get("condition_met") is (float(fraction) < TARGET_RESIDUAL_FRACTION)
        and board_names == ["GateMate", "KV260", "PolarFire"]
        and artifact.get("rows") == expected_rows
        and scores_zero
    )
    return {
        "amdahl_ceiling_x": expected_ceiling,
        "one_hundred_x_condition_met": (
            None if expected_ceiling is None else float(fraction) < TARGET_RESIDUAL_FRACTION
        ),
        "board_names": board_names,
        "row_count": len(expected_rows),
        "scores_zero": scores_zero,
        "matches_declared": matches,
    }


def _source_replay_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Rehash only source records explicitly marked for cold replay."""

    errors: list[str] = []
    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return ["source_artifact_hashes_invalid"]
    for label, value in sources.items():
        if not isinstance(value, Mapping) or value.get("verify_on_replay") is not True:
            continue
        path = Path(str(value.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        observed = current_work_receipt.sha256_file(resolved) if resolved.is_file() else None
        if observed != value.get("sha256"):
            errors.append(f"source_hash_mismatch:{label}")
    return errors


def validate_artifact(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> list[str]:
    """Cold-check identity, provenance, bounds, board limits, scores, and checksum."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_boundary_invalid")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    errors.extend(
        f"current_receipt:{error}"
        for error in current_work_receipt.validate_current_work_receipt(artifact, root=root)
    )
    errors.extend(_source_replay_errors(artifact, root))
    reduction = independent_reduce(artifact)
    if reduction["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    if any(
        artifact.get(field) != 0
        for field in ("hardware_ready_score", "hardware_value_score", "promotion_score")
    ):
        errors.append("hardware_score_nonzero")
    boards = {
        str(row.get("board")): row
        for row in artifact.get("board_rows", [])
        if isinstance(row, Mapping)
    }
    if (
        boards.get("KV260", {}).get("future_access") != "ssh_only"
        or boards.get("PolarFire", {}).get("fpga_sampling_claimed") is not False
        or any(row.get("new_hardware_execution_claimed") is not False for row in boards.values())
    ):
        errors.append("board_boundary_invalid")
    route = artifact.get("learning_hardware_route")
    if not isinstance(route, Mapping) or route.get("durability_trade_allowed") is not False:
        errors.append("durability_boundary_invalid")
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    validation_required = artifact.get("validation_required") is True
    if validation_required and not _required_receipts(
        receipts, validation_scope.REQUIRED_CHECK_NAMES
    ):
        errors.append("affected_receipts_invalid")
    if require_terminal and not _required_receipts(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_receipts_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358/Exp7303 affected command plan."""

    return validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, or command drift."""

    return validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)


def _span(
    name: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:  # pragma: no cover - measured entrypoint boundary.
    """Close one real monotonic phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": round(phase_started - run_started, 6),
        "end_s": round(ended - run_started, 6),
        "duration_s": round(ended - phase_started, 6),
        "completed_units": completed_units,
        "checkpoint_reference": checkpoint,
    }


def _source_record(
    path: Path,
    *,
    root: Path,
    role: str,
    source: Mapping[str, Any] | None = None,
    verify_on_replay: bool = True,
) -> JsonDict:  # pragma: no cover - repository-byte integration.
    """Hash one exact input and preserve any original classification flags."""

    resolved = path if path.is_absolute() else root / path
    relative = str(resolved) if path.is_absolute() else path.as_posix()
    value: JsonDict = {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": current_work_receipt.sha256_file(resolved),
        "role": role,
        "verify_on_replay": verify_on_replay,
    }
    if source is not None:
        value.update(
            {
                "original_status": source.get("status"),
                "original_honest_verdict": source.get("honest_verdict"),
                "original_verdict_class": source.get("verdict_class"),
                "original_flagged_adversarial": source.get("flagged_adversarial"),
            }
        )
    return value


def _read_only_gatemate_audit(root: Path, path: Path) -> JsonDict:  # pragma: no cover
    """Run the shipped local-receipt parser and bind this milestone's no-op audit."""

    state = board_history.search_changed_state_receipt(root, path)
    raw = load_json(path)
    raw.update(
        {
            "schema": "carnot.exp7445.gatemate_changed_state_audit.v1",
            "run_date": RUN_DATE,
            "reader": "carnot.experiment_7367_v646_board_disposition.search_changed_state_receipt",
            "audit_mode": "read_only_changed_state",
            "hardware_operations_issued": [],
            "physical_detect_commands_issued": [],
            "flash_commands_issued": [],
            "ssh_commands_issued": [],
            "purchase_operations_issued": [],
        }
    )
    current_work_receipt.atomic_json(path, raw)
    state.update(
        {
            "search_receipt_path": path.relative_to(root).as_posix(),
            "search_receipt_hash": current_work_receipt.sha256_file(path),
            "accepted_receipt_count": raw.get("accepted_receipt_count", 0),
            "eligibility_contract": deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT),
        }
    )
    return state


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - entrypoint integration.
    """Authenticate source bytes, producer flags, raw timings, and board hash chains."""

    checks: list[JsonDict] = []
    source_hashes: JsonDict = {}
    for relative in REQUIRED_SOURCE_PATHS:
        resolved = root / relative
        present = resolved.is_file() and resolved.stat().st_size > 0
        expected_hash = EXPECTED_UPSTREAM_HASHES.get(relative)
        observed_hash = current_work_receipt.sha256_file(resolved) if present else None
        passed = present and (expected_hash is None or observed_hash == expected_hash)
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "source_authentication",
                "==",
                expected_hash or "readable_nonempty_bytes",
                observed_hash
                if expected_hash
                else ("readable_nonempty_bytes" if present else None),
                passed,
                "Dependent work begins only after exact source bytes are observed.",
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="sha256" if expected_hash else "bytes",
            )
        )
        if present:
            source_hashes[relative.as_posix()] = _source_record(
                relative, root=root, role="required_source"
            )

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_row(
            "driving_requirement",
            "source_authentication",
            "contains",
            "REQ-REPORT-7445",
            "REQ-REPORT-7445" if "REQ-REPORT-7445" in spec_text else None,
            "REQ-REPORT-7445" in spec_text,
            "The capability requirement exists before implementation behavior runs.",
            upstream="OpenSpec",
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    exp7432 = load_json(root / EXP7432_PATH)
    exp7440 = load_json(root / EXP7440_PATH)
    exp7433 = load_json(root / EXP7433_PATH)
    checks.extend(authenticate_upstreams(exp7432, exp7440))
    for relative, source in (
        (EXP7432_PATH, exp7432),
        (EXP7440_PATH, exp7440),
        (EXP7433_PATH, exp7433),
    ):
        if (root / relative).is_file():
            source_hashes[relative.as_posix()] = _source_record(
                relative, root=root, role="terminal_artifact", source=source
            )

    fraction = (exp7432.get("amdahl_analysis") or {}).get(
        "observed_unaccelerated_persistence_fraction"
    )
    checks.append(
        gate_row(
            "exp7432_persistence_fraction",
            "source_authentication",
            "==",
            PERSISTENCE_FRACTION,
            fraction,
            fraction == PERSISTENCE_FRACTION,
            "The Amdahl bound uses the exact recorded persistence fraction.",
            upstream="Exp7432",
            path=EXP7432_PATH.as_posix(),
            field="amdahl_analysis.observed_unaccelerated_persistence_fraction",
        )
    )
    capstone_fraction = (
        (exp7433.get("diagnostic_summary") or {})
        .get("amdahl_analysis", {})
        .get("observed_unaccelerated_persistence_fraction")
    )
    capstone_source_hash = (
        (exp7433.get("source_artifact_hashes") or {})
        .get("exp7432-update-placement", {})
        .get("sha256")
    )
    checks.extend(
        [
            gate_row(
                "exp7433_crosscheck_persistence_fraction",
                "source_authentication",
                "==",
                PERSISTENCE_FRACTION,
                capstone_fraction,
                capstone_fraction == PERSISTENCE_FRACTION,
                "The V651 capstone independently preserved the same archived fraction.",
                upstream="Exp7433",
                path=EXP7433_PATH.as_posix(),
                field="diagnostic_summary.amdahl_analysis.observed_unaccelerated_persistence_fraction",
            ),
            gate_row(
                "exp7433_crosscheck_exp7432_hash",
                "source_authentication",
                "==",
                EXPECTED_UPSTREAM_HASHES[EXP7432_PATH],
                capstone_source_hash,
                capstone_source_hash == EXPECTED_UPSTREAM_HASHES[EXP7432_PATH],
                "The capstone's typed source reference must name the same Exp7432 bytes.",
                upstream="Exp7433",
                path=EXP7433_PATH.as_posix(),
                field="source_artifact_hashes.exp7432-update-placement.sha256",
            ),
        ]
    )

    raw7432 = load_json(root / EXP7432_RAW_PATH)
    timing_equal = raw7432.get("timing_rows") == exp7432.get("timing_rows")
    boards_equal = raw7432.get("board_rows") == exp7432.get("board_rows")
    for field, observed in (("timing_rows", timing_equal), ("board_rows", boards_equal)):
        checks.append(
            gate_row(
                f"exp7432_raw_{field}_match",
                "source_authentication",
                "==",
                True,
                observed,
                observed,
                "The terminal reduction must match its exact raw evidence bytes.",
                upstream="Exp7432 raw evidence",
                path=EXP7432_RAW_PATH.as_posix(),
                field=field,
            )
        )

    historical_boards = [
        dict(row) for row in exp7432.get("board_rows", []) if isinstance(row, Mapping)
    ]
    for row in historical_boards:
        source_path = Path(str(row.get("last_authenticated_path") or ""))
        resolved = source_path if source_path.is_absolute() else root / source_path
        observed = current_work_receipt.sha256_file(resolved) if resolved.is_file() else None
        expected = row.get("last_authenticated_hash")
        board = str(row.get("board"))
        checks.append(
            gate_row(
                f"board_receipt_hash:{board}",
                "source_authentication",
                "==",
                expected,
                observed,
                observed is not None and observed == expected,
                "Each board disposition follows its recorded terminal receipt hash.",
                upstream=f"Exp7432 board:{board}",
                path=str(source_path),
                field="last_authenticated_hash",
            )
        )
        if resolved.is_file():
            source_hashes[f"board_receipt:{board}"] = _source_record(
                source_path,
                root=root,
                role=f"historical_board_receipt:{board}",
            )

    audit_path = root / RAW_DIR / "preconditions/gatemate_changed_state_audit.json"
    changed_state = _read_only_gatemate_audit(root, audit_path)
    source_hashes[audit_path.relative_to(root).as_posix()] = _source_record(
        audit_path.relative_to(root), root=root, role="read_only_changed_state_audit"
    )
    checks.append(
        gate_row(
            "gatemate_changed_state_branch",
            "external_prerequisite",
            ">",
            0,
            changed_state.get("accepted_receipt_count", 0),
            changed_state.get("exists") is True,
            "Only a dated operator cable, port, power, board, or DirtyJTAG change after Exp6559 reopens a future task.",
            upstream="Exp6559 operator changed-state boundary",
            path=audit_path.relative_to(root).as_posix(),
            field="accepted_receipt_count",
        )
    )
    return (
        checks,
        source_hashes,
        {
            "exp7432": exp7432,
            "exp7440": exp7440,
            "historical_board_rows": historical_boards,
            "changed_state": changed_state,
        },
    )


def _terminal_commands(
    root: Path, candidate: Path
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build fresh readers and unchanged strict checks for the exact candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reduce",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
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
    )
    categories = ("completion", "completion", "safety", "completion")
    return [
        validation_contract.PlannedCommand(spec, category, True)
        for spec, category in zip(specs, categories, strict=True)
    ]


def _entrypoint_receipt(
    root: Path, started_at_utc: str, duration_s: float
) -> JsonDict:  # pragma: no cover
    """Hash the declared command/environment as this capability's E2E receipt."""

    path = root / RAW_DIR / "validation/entrypoint/declared_entrypoint_e2e.json"
    value = {
        "argv": [".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE],
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "duration_s": duration_s,
        "environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }
    current_work_receipt.atomic_json(path, value)
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(value["argv"]),
        "command_argv": value["argv"],
        "command_environment": value["environment"],
        "scope": "declared_capability_entrypoint",
        "exit_code": 0,
        "duration_s": duration_s,
        "log_path": path.relative_to(root).as_posix(),
        "log_sha256": current_work_receipt.sha256_file(path),
        "passed": True,
        "timed_out": False,
        "required": True,
        "command_category": "completion",
        "started_at_utc": started_at_utc,
        "ended_at_utc": value["completed_at_utc"],
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared capability E2E.
    """Authenticate, reduce, validate, and atomically publish the envelope."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(run_started, "startup", "flushed", completed_units=0)

    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before_authentication", completed_units=0)
    preconditions, source_hashes, context = collect_preconditions(root)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started,
            len(preconditions),
            (raw_dir / "preconditions/gatemate_changed_state_audit.json").as_posix(),
        )
    )
    progress(
        run_started,
        "preconditions",
        "after_authentication",
        completed_units=len(preconditions),
    )
    required_failed = next(
        (
            row
            for row in preconditions
            if row.get("category") != "external_prerequisite" and row.get("passed") is not True
        ),
        None,
    )
    if required_failed is not None:
        raise RuntimeError(
            "blocked_required_source:"
            + json.dumps(gate_check_summary([required_failed]), sort_keys=True)
        )

    phase_started = time.monotonic()
    progress(run_started, "reduction", "before_aggregation", completed_units=0)
    amdahl = compute_amdahl_rows(PERSISTENCE_FRACTION)
    stages = reduce_stage_envelope(context["exp7432"], context["exp7440"])
    boards = build_board_rows(context["historical_board_rows"], context["changed_state"])
    futures = external_future_rows()
    raw_value = {
        "schema": "carnot.exp7445.raw_evidence.v1",
        "exp7432_sha256": source_hashes[EXP7432_PATH.as_posix()]["sha256"],
        "exp7440_sha256": source_hashes[EXP7440_PATH.as_posix()]["sha256"],
        "amdahl_rows": amdahl,
        "stage_envelope": stages,
        "board_rows": boards,
        "external_future_rows": futures,
    }
    raw_path = raw_dir / "raw_evidence.json"
    current_work_receipt.atomic_json(raw_path, raw_value)
    source_hashes[raw_path.relative_to(root).as_posix()] = _source_record(
        raw_path.relative_to(root), root=root, role="current_raw_reduction"
    )
    archive_path = raw_dir / "archived_training_and_model_receipts.json"
    archive_ref = current_work_receipt.write_immutable_sidecar(
        archive_path,
        scope="historical_model_receipts",
        payload={
            "source_path": EXP7440_PATH.as_posix(),
            "source_sha256": source_hashes[EXP7440_PATH.as_posix()]["sha256"],
            "small_ebm_training": deepcopy(context["exp7440"].get("small_ebm_training")),
            "current_model_invoked": False,
            "current_invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        },
        root=root,
    )
    source_hashes[archive_path.relative_to(root).as_posix()] = _source_record(
        archive_path.relative_to(root), root=root, role="typed_archived_receipts"
    )
    reduced_units = (
        len(amdahl) + len(stages["complete_service_timing_rows"]) + len(boards) + len(futures)
    )
    spans.append(_span("reduction", phase_started, run_started, reduced_units, raw_path.as_posix()))
    progress(
        run_started,
        "reduction",
        "after_aggregation",
        completed_units=reduced_units,
    )

    private = Path(tempfile.mkdtemp(prefix="carnot-exp7445-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    phase_started = time.monotonic()
    progress(
        run_started,
        "affected_validation",
        "before_subprocesses",
        completed_units=0,
    )
    affected = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ],
        log_dir=raw_dir / "validation/affected",
    )
    affected_reduction = validation_contract.reduce_affected_receipts(
        root, AFFECTED_MANIFEST, affected
    )
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            run_started,
            len(affected),
            (raw_dir / "validation/affected").as_posix(),
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError(f"affected_validation_failed:{affected_reduction}")

    candidate = assemble_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        amdahl_rows=amdahl,
        stage_envelope=stages,
        board_rows=boards,
        future_rows=futures,
        validation_receipts=affected,
        phase_spans=spans,
        sidecar_references=[archive_ref],
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        validation_required=True,
        require_terminal=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    current_work_receipt.atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(
        run_started,
        "terminal_validation",
        "before_subprocesses",
        completed_units=0,
    )
    terminal = validation_contract.run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            run_started,
            len(terminal),
            (raw_dir / "validation/terminal").as_posix(),
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    entry_receipt = _entrypoint_receipt(root, started_at, time.monotonic() - run_started)
    all_receipts = [*affected, *terminal, entry_receipt]
    final = assemble_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        amdahl_rows=amdahl,
        stage_envelope=stages,
        board_rows=boards,
        future_rows=futures,
        validation_receipts=all_receipts,
        phase_spans=spans,
        sidecar_references=[archive_ref],
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        validation_required=True,
        require_terminal=True,
        flagged_adversarial=critical or not terminal_passed,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    destination = output_path if output_path.is_absolute() else root / output_path
    progress(run_started, "publish", "before_atomic_terminal", path=destination)
    current_work_receipt.atomic_json(candidate_path, final)
    current_work_receipt.atomic_json(destination, final)
    progress(
        run_started,
        "publish",
        "after_atomic_terminal",
        status=final["status"],
        completed_units=len(final["rows"]),
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse run and read-only fresh-process modes for the thin entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or one read-only fresh-process candidate reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        artifact = load_json(args.cold_replay)
        reduction = independent_reduce(artifact)
        errors = validate_artifact(artifact, root=REPO_ROOT, require_terminal=False)
        print(
            json.dumps(
                {"independent_reduction": reduction, "validation_errors": errors},
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = load_json(args.independent_reduce)
        reduction = independent_reduce(artifact)
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction["matches_declared"] is not True)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
