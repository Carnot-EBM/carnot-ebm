"""Preserve three board dispositions without contacting any board.

The reducer authenticates the latest hardware envelope, retains the narrower
historical graduation scopes, and uses the shipped GateMate receipt reader.
Host durability can guide future evidence needs, but it cannot create a board
performance claim.

Spec refs: REQ-HW-7459 and SCENARIO-HW-7459-*.
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

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7367_v646_board_disposition as disposition_reader
from carnot import experiment_7445_v652_hardware_envelope as hardware_history
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
PHASE = 4
EXPERIMENT_ID = "exp7459-v653-board-continuity"
SCHEMA = "carnot.exp7459.v653.board_continuity.v1"

RESULT_PATH = Path("results/experiment_7459_v653_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7459_v653_board_continuity")
MODULE_PATH = Path("python/carnot/experiment_7459_v653_board_continuity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7459_v653_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7459_v653_board_continuity.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
HARDWARE_SOURCE_PATH = Path("results/experiment_7445_v652_hardware_envelope.json")
GRADUATION_PATH = Path("results/experiment_7314_v642_board_continuity.json")
EXP6559_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
DURABLE_SOURCE_PATH = Path("results/experiment_7458_v653_durable_updates.json")

EXPECTED_SOURCE_HASHES = {
    HARDWARE_SOURCE_PATH: "sha256:bbf535fc1df3fb9b579abb0d2ecc6b5bd67c525f9abef1aea2cb92de9cdcf328",
    GRADUATION_PATH: "sha256:c83cc85d16c082898992b3d34197d5d6bb07dfae5cd5bb9beca10bbbc20d51e6",
    EXP6559_PATH: "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66",
    DURABLE_SOURCE_PATH: "sha256:e21e8116f43e95c560dd8ea6f584c9b7ebd52c4d27a9779e9d71ebf33422b3ad",
}

KV260_TERMINAL_CRITERION = (
    "board-level programmable-logic latency transcript and successful KV260 synthesis"
)
POLARFIRE_TERMINAL_CRITERION = (
    "end-to-end hash-matched CPU dispatch with retained raw transcript evidence"
)
GATEMATE_TERMINAL_CRITERION = "n=16 Ising tile flashed and smoke-tested on programmable logic"
GATEMATE_MISSING_REASON = (
    "no operator-authored dated GateMate cable, port, power, board, or "
    "DirtyJTAG change after Exp6559"
)
GATEMATE_NEXT_PREREQUISITE = (
    "operator records a dated GateMate cable, port, power, board, or DirtyJTAG change after Exp6559"
)

ZERO_INVOCATION_COUNTS = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
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
    Path("python/carnot/experiment_7367_v646_board_disposition.py"),
    Path("python/carnot/experiment_7445_v652_hardware_envelope.py"),
    SPEC_PATH,
    HARDWARE_SOURCE_PATH,
    GRADUATION_PATH,
    EXP6559_PATH,
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("docs/jtag-wiring-gatemate-dirtyjtag.md"),
    Path("ops/known-issues.md"),
    Path("ops/operator-followup.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(
        TEST_PATH.as_posix(),
        "tests/python/test_experiment_7445_v652_hardware_envelope.py",
        "tests/python/test_experiment_7367_v646_board_disposition.py",
    ),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint_e2e",
)

REQUIRED_FIELDS = {
    "schema",
    "experiment_id",
    "milestone",
    "phase",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "current_invocation_events",
    "current_run_id",
    "current_owner_pid",
    "event_count",
    "event_sha256",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "started_monotonic_ns",
    "ended_monotonic_ns",
    "phase_spans",
    "receipt_sidecars",
    "small_ebm_training",
    "random_seed",
    "random_seed_reason",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "validation_required",
    "validation_manifest",
    "field_principles",
    "promotion_score",
    "board_rows",
    "hardware_ready_score",
    "hardware_value_score",
    "changed_state_evidence",
    "hardware_wishlist_disposition",
    "hardware_operations_issued",
    "raw_evidence_reference",
    "independent_reduction",
    "capability_e2e",
    "execution_host",
}


def utc_now() -> str:
    """Return an aware UTC timestamp for a real task boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase and subprocess boundaries with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7459] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and return an empty object for any other shape."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    path: str | None,
    field: str,
) -> JsonDict:
    """Keep each expected and observed gate value directly machine-readable."""

    return hardware_history.gate_row(
        check,
        category,
        "==",
        expected,
        observed,
        passed,
        principle,
        upstream=upstream,
        path=path,
        field=field,
    )


def _identity_gates(
    source: Mapping[str, Any],
    *,
    label: str,
    path: Path,
    expected: Mapping[str, Any],
) -> list[JsonDict]:
    """Authenticate one producer without merging its fields with another."""

    return [
        _gate(
            f"{label.lower()}_{field}",
            "source_authentication",
            expected_value,
            source.get(field),
            source.get(field) == expected_value,
            "Only exact terminal producer identity and original flags enter this audit.",
            upstream=label,
            path=path.as_posix(),
            field=field,
        )
        for field, expected_value in expected.items()
    ]


def authenticate_sources(
    hardware: Mapping[str, Any],
    cutoff: Mapping[str, Any],
    durable: Mapping[str, Any] | None,
) -> list[JsonDict]:
    """Authenticate board, cutoff, and optional host-durability producers."""

    gates = _identity_gates(
        hardware,
        label="hardware",
        path=HARDWARE_SOURCE_PATH,
        expected={
            "schema": "carnot.exp7445.v652.hardware_envelope.v1",
            "experiment_id": "exp7445-v652-hardware-envelope",
            "milestone": "2026.09.652",
            "run_date": "20260920",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
    )
    gates.extend(
        _identity_gates(
            cutoff,
            label="exp6559",
            path=EXP6559_PATH,
            expected={
                "schema": "carnot.experiment_6559.gatemate_changed_state_continuity.v567",
                "experiment_id": "exp6559-gatemate-changed-state-continuity",
                "milestone": "2026.08.567",
                "run_date": "20260823",
                "verdict_class": "blocked",
            },
        )
    )
    if durable is None:
        gates.append(
            _gate(
                "durable_update_optional",
                "optional_context",
                "available_or_absent",
                "absent",
                True,
                "An independent board audit continues when same-milestone host timing is absent.",
                upstream="Exp7458",
                path=DURABLE_SOURCE_PATH.as_posix(),
                field="presence",
            )
        )
    else:
        gates.extend(
            _identity_gates(
                durable,
                label="durable",
                path=DURABLE_SOURCE_PATH,
                expected={
                    "schema": "carnot.exp7458.v653.durable_updates.v1",
                    "experiment_id": "exp7458-v653-durable-updates",
                    "milestone": MILESTONE,
                    "run_date": RUN_DATE,
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                    "durable_update_complete_score": 1,
                    "durable_update_value_score": 0,
                },
            )
        )
    boards = hardware.get("board_rows")
    names = sorted(str(row.get("board")) for row in boards or [] if isinstance(row, Mapping))
    gates.append(
        _gate(
            "hardware_board_names",
            "source_authentication",
            ["GateMate", "KV260", "PolarFire"],
            names,
            names == ["GateMate", "KV260", "PolarFire"],
            "All three boards need independent source rows before reduction.",
            upstream="Exp7445",
            path=HARDWARE_SOURCE_PATH.as_posix(),
            field="board_rows[].board",
        )
    )
    return gates


def normalize_changed_state(changed_state: Mapping[str, Any]) -> JsonDict:
    """Keep the selected physical evidence separate from the read-only search."""

    exists = changed_state.get("exists") is True
    conditions = changed_state.get("changed_conditions")
    changed = dict(conditions) if isinstance(conditions, Mapping) else {}
    changed_fields = sorted(name for name, value in changed.items() if value is True)
    return {
        "exists": exists,
        "disposition": (
            "changed_physical_prerequisite_recorded_future_task_only"
            if exists
            else "blocked_unchanged_physical_prerequisite"
        ),
        "cutoff_experiment": "Exp6559",
        "cutoff_date": "20260823",
        "accepted_receipt_count": changed_state.get("accepted_receipt_count", 0),
        "latest_receipt_date": changed_state.get("latest_receipt_date"),
        "receipt_timestamp": changed_state.get("receipt_timestamp"),
        "changed_fields": changed_fields,
        "evidence_path": changed_state.get("evidence_path"),
        "evidence_hash": changed_state.get("evidence_hash"),
        "search_receipt_path": changed_state.get("search_receipt_path"),
        "search_receipt_hash": changed_state.get("search_receipt_hash"),
        "eligibility_contract": deepcopy(changed_state.get("eligibility_contract")),
        "hardware_operations_issued": [],
        "detect_count": 0,
        "flash_count": 0,
        "ssh_probe_count": 0,
        "purchase_count": 0,
    }


def search_changed_state_evidence(
    root: Path, raw_path: Path
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint.
    """Reuse the approved reader and bind its raw output to this task."""

    result = disposition_reader.search_changed_state_receipt(root, raw_path)
    raw = _load_object(raw_path)
    candidates = [dict(row) for row in raw.get("candidate_rows") or [] if isinstance(row, Mapping)]
    selected = raw.get("selected_receipt")
    selected_row = dict(selected) if isinstance(selected, Mapping) else {}
    dates = [str(row["receipt_date"]) for row in candidates if row.get("receipt_date")]
    fields = selected_row.get("material_physical_fields")
    changed_fields = list(fields) if isinstance(fields, list) else []
    raw.update(
        {
            "schema": "carnot.exp7459.gatemate_changed_state_evidence.v1",
            "run_date": RUN_DATE,
            "reader": (
                "carnot.experiment_7367_v646_board_disposition.search_changed_state_receipt"
            ),
            "cutoff_experiment": "Exp6559",
            "hardware_operations_issued": [],
        }
    )
    current_work_receipt.atomic_json(raw_path, raw)
    result.update(
        {
            "latest_receipt_date": max(dates) if dates else None,
            "receipt_timestamp": selected_row.get("receipt_timestamp"),
            "changed_conditions": {name: True for name in changed_fields},
            "evidence_path": selected_row.get("path"),
            "evidence_hash": selected_row.get("evidence_hash"),
            "search_receipt_path": raw_path.relative_to(root).as_posix(),
            "search_receipt_hash": current_work_receipt.sha256_file(raw_path),
            "accepted_receipt_count": raw.get("accepted_receipt_count", 0),
            "eligibility_contract": deepcopy(disposition_reader.PHYSICAL_RECEIPT_CONTRACT),
            "hardware_operations_issued": [],
        }
    )
    return result


def reduce_board_rows(
    hardware: Mapping[str, Any], changed_state: Mapping[str, Any]
) -> list[JsonDict]:
    """Narrow the shipped dispositions without broadening historical claims."""

    source_rows = hardware.get("board_rows")
    if not isinstance(source_rows, list):
        raise ValueError("hardware_board_rows_missing")
    shared_rows = hardware_history.build_board_rows(source_rows, changed_state)
    by_board = {str(row.get("board")): dict(row) for row in shared_rows}
    source_by_board = {
        str(row.get("board")): row for row in source_rows if isinstance(row, Mapping)
    }
    if set(by_board) != {"KV260", "PolarFire", "GateMate"}:
        raise ValueError("board_rows_invalid")
    normalized_changed = normalize_changed_state(changed_state)
    output: list[JsonDict] = []
    metadata = {
        "KV260": {
            "terminal_criterion": KV260_TERMINAL_CRITERION,
            "exact_claim_scope": "historical_kv260_fpga_fabric_sampling_only",
            "future_access": "ssh_only",
            "access_mechanism": "ssh kria only",
        },
        "PolarFire": {
            "terminal_criterion": POLARFIRE_TERMINAL_CRITERION,
            "exact_claim_scope": ("historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"),
            "fpga_sampling_claimed": False,
            "processor_class": "polarfire_linux_cpu",
        },
        "GateMate": {
            "terminal_criterion": GATEMATE_TERMINAL_CRITERION,
            "exact_claim_scope": "no_current_execution_changed_state_gate_only",
        },
    }
    for board in ("KV260", "PolarFire", "GateMate"):
        row = by_board[board]
        row.pop("row_sha256", None)
        row.update(deepcopy(metadata[board]))
        row.update(
            {
                "source_artifact_path": HARDWARE_SOURCE_PATH.as_posix(),
                "source_artifact_sha256": EXPECTED_SOURCE_HASHES[HARDWARE_SOURCE_PATH],
                "source_row_sha256": source_by_board[board].get("row_sha256"),
                "evidence_path": row.get("last_authenticated_path"),
                "evidence_sha256": row.get("last_authenticated_hash"),
                "evidence_date": row.get("last_authenticated_date"),
                "historical_graduation_transcript": {
                    "path": row.get("last_authenticated_path"),
                    "sha256": row.get("last_authenticated_hash"),
                    "date": row.get("last_authenticated_date"),
                },
                "current_disposition_date": RUN_DATE,
                "historical_evidence_only": True,
                "new_hardware_execution_claimed": False,
                "present_reachability_asserted": False,
                "hardware_operations_issued": [],
                "hardware_ready_score": 0,
                "hardware_value_score": 0,
                "read_only_audit": True,
                "failed": False,
                "censored": False,
            }
        )
        if board == "GateMate":
            changed = normalized_changed["exists"] is True
            row.update(
                {
                    "disposition": "complete" if changed else "blocked",
                    "availability_class": (
                        "future_bounded_task_eligible" if changed else "blocked"
                    ),
                    "terminal_state": (
                        "changed_state_future_task_eligible"
                        if changed
                        else "blocked_changed_physical_state"
                    ),
                    "honest_verdict": (
                        "complete_changed_physical_prerequisite_future_bounded_task_only"
                        if changed
                        else "blocked_unchanged_physical_prerequisite"
                    ),
                    "error": None if changed else GATEMATE_MISSING_REASON,
                    "metric": changed,
                    "exact_next_prerequisite": (
                        "one separately authorized bounded GateMate bring-up task"
                        if changed
                        else GATEMATE_NEXT_PREREQUISITE
                    ),
                    "changed_state_evidence": deepcopy(normalized_changed),
                    "bounded_next_bringup_plan": {
                        "current_task_authorized": False,
                        "maximum_hardware_actions": 1,
                        "future_action": "one bounded read-only detect, then stop",
                        "requires_separate_authorization": True,
                        "flash_authorized_by_current_task": False,
                    },
                    "evidence_path": normalized_changed.get("search_receipt_path"),
                    "evidence_sha256": normalized_changed.get("search_receipt_hash"),
                    "evidence_date": RUN_DATE,
                    "latest_operator_receipt_date": normalized_changed.get("latest_receipt_date"),
                }
            )
        row["row_sha256"] = current_work_receipt.canonical_hash(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        output.append(row)
    return output


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently recompute board counts, claim boundaries, and zero scores."""

    by_board = {str(row.get("board")): row for row in rows}
    names_exact = set(by_board) == {"KV260", "PolarFire", "GateMate"}
    operations = sum(len(row.get("hardware_operations_issued") or []) for row in rows)
    graduated = sum(row.get("availability_class") == "graduated_historical" for row in rows)
    blocked = sum(row.get("disposition") == "blocked" for row in rows)
    gate = by_board.get("GateMate", {})
    return {
        "board_count": len(rows),
        "board_names_exact": names_exact,
        "graduated_count": graduated,
        "blocked_count": blocked,
        "hardware_operation_count": operations,
        "kv260_ssh_only": by_board.get("KV260", {}).get("future_access") == "ssh_only",
        "polarfire_cpu_not_fpga": (
            by_board.get("PolarFire", {}).get("hash_matched_cpu_dispatch") is True
            and by_board.get("PolarFire", {}).get("fpga_sampling_claimed") is False
        ),
        "gatemate_changed_state": str(gate.get("honest_verdict") or "").startswith(
            "complete_changed_physical_prerequisite"
        ),
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
    }


def build_hardware_wishlist_disposition(
    durable: Mapping[str, Any] | None,
) -> JsonDict:
    """Translate host residuals into future evidence needs, never board value."""

    timing = durable.get("timing_summary") if isinstance(durable, Mapping) else None
    values = dict(timing) if isinstance(timing, Mapping) else {}
    available = bool(values)
    residual = values.get("residual_host_fraction") if available else None
    ratio_upper = values.get("ratio_ci95_upper") if available else None
    shared = {
        "currently_justified": False,
        "vendor_report_is_local_timing": False,
        "hardware_value_score": 0,
    }
    routes = [
        {
            **shared,
            "option": "NPU",
            "measured_residual_bottleneck": residual,
            "evidence_required": (
                "an isolated numeric expert or update kernel that dominates transfer and "
                "the measured durable host residual"
            ),
            "local_access_authenticated": False,
        },
        {
            **shared,
            "option": "larger FPGA",
            "measured_residual_bottleneck": residual,
            "evidence_required": (
                "an authenticated workload above KV260 k_max<=5 whose device-mappable "
                "numeric cost dominates complete acknowledged service"
            ),
            "local_access_authenticated": False,
        },
        {
            **shared,
            "option": "authenticated Extropic run",
            "measured_residual_bottleneck": residual,
            "evidence_required": (
                "authorized local or vendor-evaluation access plus end-to-end transfer, "
                "readout, accuracy, and acknowledged service timing"
            ),
            "local_access_authenticated": False,
        },
    ]
    return {
        "measurement_available": available,
        "source_path": DURABLE_SOURCE_PATH.as_posix() if available else None,
        "source_sha256": EXPECTED_SOURCE_HASHES[DURABLE_SOURCE_PATH] if available else None,
        "source_honest_verdict": durable.get("honest_verdict") if available else None,
        "residual_host_fraction": residual,
        "service_ratio_ci95_upper": ratio_upper,
        "durable_update_value_score": (
            durable.get("durable_update_value_score") if available else None
        ),
        "measured_conclusion": (
            "durable host persistence and orchestration remain the dominant residual"
            if available
            else "same-milestone host durability measurement unavailable"
        ),
        "independent_board_audit_continues": True,
        "web_page_establishes_access": False,
        "hardware_value_score": 0,
        "routes": routes,
    }


def _required_receipts(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one passing, non-timeout receipt for every named check."""

    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is False
        for name in names
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose external blocks without confusing them with invalid evidence."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    validity = [
        row for row in gates if row.get("category") not in {"external_prerequisite", "benefit"}
    ]
    return {
        "all_passed": not failures,
        "all_validity_checks_passed": all(row.get("passed") is True for row in validity),
        "failed_checks": [row.get("check") for row in failures],
        "first_failure": failures[0] if failures else None,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain every ordinary field without wrapping its scalar value."""

    specific = {
        "schema": "Use a versioned schema with exact experiment and milestone identity.",
        "run_date": "Use 20260920 with actual UTC and monotonic boundaries.",
        "preconditions_checked": "Name each source path, identity, flag, and observed gate value.",
        "MODEL_SPECS": "Use an empty list because this aggregation performs no LLM work.",
        "model_invoked": "Count current attempted model work, not cited historical work.",
        "invocation_counts": "Balance every current attempted and terminal LLM call state.",
        "inference_substrate": "Name read-only board evidence aggregation on the host.",
        "inference_substrate_class": "Declare aggregation without simulated execution or duration padding.",
        "execution_venue": "Use host and keep historical device venues in board rows.",
        "duration_s": "Measure actual current work with a monotonic clock.",
        "phase_spans": "Bind measured phases, progress, and current clock segment.",
        "random_seed": "Use null because this deterministic audit has no randomness.",
        "reproducibility_checksum": "Bind code, protocol, immutable inputs, raw rows, and validation scope.",
        "source_artifact_hashes": "Preserve exact bytes and original source flags.",
        "rows": "Retain one complete row for each board, including the blocked board.",
        "sample_size_budget": "Account for all three planned board units with a fixed stop rule.",
        "acceptance_gate_results": "Keep validity and external prerequisite gates distinct.",
        "gate_check_summary": "Name the exact failed GateMate field without hiding valid branches.",
        "verifier_is_oracle": "Use false because receipts supply evidence, not scoring authority.",
        "honest_verdict": "Complete the audit while naming the unchanged external board block.",
        "verdict_class": "Use the closed terminal enum; this complete no-value audit is null.",
        "flagged_adversarial": "Any critical current finding disqualifies board readiness.",
        "validation_receipts": "Record actual scoped commands, environments, exits, durations, and hashes.",
        "field_principles": "Explain fields here while gate values stay bare scalars.",
        "promotion_score": "Always zero because this milestone authorizes no rollout.",
        "board_rows": "Three separately authenticated dispositions prevent reachability from becoming performance.",
        "hardware_ready_score": "Always zero because no new device workload is executed.",
        "hardware_value_score": "Always zero because host timing and vendor context are not board evidence.",
        "changed_state_evidence": "Require a dated operator change before a future GateMate retry.",
        "hardware_wishlist_disposition": "Use host residuals only to state future evidence needs.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable code, inputs, raw evidence, reduction, and validation scope."""

    return current_work_receipt.canonical_hash(
        {
            key: artifact.get(key)
            for key in (
                "schema",
                "experiment_id",
                "milestone",
                "run_date",
                "preconditions_checked",
                "source_artifact_hashes",
                "board_rows",
                "changed_state_evidence",
                "hardware_wishlist_disposition",
                "raw_evidence_reference",
                "validation_manifest",
                "validation_receipts",
                "acceptance_gate_results",
                "independent_reduction",
                "verdict_class",
                "flagged_adversarial",
            )
        }
    )


def _source_record(path: Path, *, root: Path, role: str) -> JsonDict:
    """Hash exact bytes and retain terminal flags when the source has them."""

    resolved = root / path
    source = _load_object(resolved) if resolved.suffix == ".json" else {}
    return {
        "path": path.as_posix(),
        "sha256": current_work_receipt.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "role": role,
        "original_status": source.get("status"),
        "original_honest_verdict": source.get("honest_verdict"),
        "original_verdict_class": source.get("verdict_class"),
        "original_flagged_adversarial": source.get("flagged_adversarial"),
        "verify_on_replay": True,
    }


def collect_preconditions(
    root: Path,
) -> tuple[
    list[JsonDict], dict[str, JsonDict], JsonDict, JsonDict, JsonDict | None
]:  # pragma: no cover
    """Authenticate required bytes, exact upstream hashes, and original flags."""

    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "source_authentication",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
                "Every declared source must exist before dependent reduction.",
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
        if available:
            sources[relative.as_posix()] = _source_record(
                relative,
                root=root,
                role=(
                    "upstream_terminal_artifact"
                    if relative in EXPECTED_SOURCE_HASHES
                    else "required_source"
                ),
            )

    durable_path = root / DURABLE_SOURCE_PATH
    durable = _load_object(durable_path) if durable_path.is_file() else None
    if durable_path.is_file():
        sources[DURABLE_SOURCE_PATH.as_posix()] = _source_record(
            DURABLE_SOURCE_PATH, root=root, role="optional_same_milestone_context"
        )
    checks.append(
        _gate(
            "durable_source_presence",
            "optional_context",
            "available_or_absent",
            "available" if durable is not None else "absent",
            True,
            "Same-milestone timing may be absent without blocking board evidence.",
            upstream="Exp7458",
            path=DURABLE_SOURCE_PATH.as_posix(),
            field="presence",
        )
    )

    for path, expected_hash in EXPECTED_SOURCE_HASHES.items():
        if path == DURABLE_SOURCE_PATH and durable is None:
            continue
        observed = (
            sources.get(path.as_posix(), {}).get("sha256") if path.as_posix() in sources else None
        )
        checks.append(
            _gate(
                f"source_hash:{path.as_posix()}",
                "source_authentication",
                expected_hash,
                observed,
                observed == expected_hash,
                "An immutable producer cannot inherit authority after byte drift.",
                upstream=path.as_posix(),
                path=path.as_posix(),
                field="sha256",
            )
        )

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _gate(
            "driving_requirement",
            "source_authentication",
            "REQ-HW-7459",
            "REQ-HW-7459" if "REQ-HW-7459" in spec_text else None,
            "REQ-HW-7459" in spec_text,
            "Behavior starts only after its requirement exists.",
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    hardware = _load_object(root / HARDWARE_SOURCE_PATH)
    cutoff = _load_object(root / EXP6559_PATH)
    checks.extend(authenticate_sources(hardware, cutoff, durable))
    return checks, sources, hardware, cutoff, durable


def _source_replay_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Rehash declared immutable sources during a cold replay."""

    errors: list[str] = []
    for label, value in (artifact.get("source_artifact_hashes") or {}).items():
        if not isinstance(value, Mapping) or value.get("verify_on_replay") is not True:
            continue
        path = Path(str(value.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        observed = current_work_receipt.sha256_file(resolved) if resolved.is_file() else None
        if observed != value.get("sha256"):
            errors.append(f"source_hash_mismatch:{label}")
    return errors


def _raw_evidence_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Reload raw rows so a terminal summary cannot replace measured evidence."""

    reference = artifact.get("raw_evidence_reference")
    if not isinstance(reference, Mapping):
        return ["raw_evidence_reference_invalid"]
    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or current_work_receipt.sha256_file(resolved) != reference.get(
        "sha256"
    ):
        return ["raw_evidence_hash_mismatch"]
    raw = _load_object(resolved)
    if (
        raw.get("board_rows") != artifact.get("board_rows")
        or raw.get("changed_state_evidence") != artifact.get("changed_state_evidence")
        or raw.get("hardware_wishlist_disposition") != artifact.get("hardware_wishlist_disposition")
    ):
        return ["raw_rows_mismatch"]
    return []


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute terminal board claims directly from per-board rows."""

    rows = [dict(row) for row in artifact.get("board_rows") or [] if isinstance(row, Mapping)]
    reduced = reduce_rows(rows)
    declared = {
        "hardware_ready_score": artifact.get("hardware_ready_score"),
        "hardware_value_score": artifact.get("hardware_value_score"),
    }
    expected = {
        "hardware_ready_score": reduced["hardware_ready_score"],
        "hardware_value_score": reduced["hardware_value_score"],
    }
    valid = (
        reduced["board_count"] == 3
        and reduced["board_names_exact"] is True
        and reduced["graduated_count"] == 2
        and reduced["hardware_operation_count"] == 0
        and reduced["kv260_ssh_only"] is True
        and reduced["polarfire_cpu_not_fpga"] is True
        and declared == expected
    )
    return {**reduced, "declared_scores": declared, "matches_declared": valid}


def _validation_manifest() -> JsonDict:
    """Freeze exact files passed to the affected-only validation runner."""

    return {
        "experiment_id": AFFECTED_MANIFEST.experiment_id,
        "test_paths": list(AFFECTED_MANIFEST.test_paths),
        "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
        "static_paths": list(AFFECTED_MANIFEST.static_paths),
        "full_python_suite": False,
    }


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    changed_state_evidence: Mapping[str, Any],
    hardware_wishlist_disposition: Mapping[str, Any],
    raw_evidence_reference: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_required: bool,
    sidecar_references: Sequence[Mapping[str, Any]],
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete record from raw rows and current receipts."""

    rows = [deepcopy(dict(row)) for row in board_rows]
    reduction_shell: JsonDict = {
        "board_rows": rows,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
    }
    reduction = independent_reduce(reduction_shell)
    changed = reduction["gatemate_changed_state"] is True
    affected_passed = (
        _required_receipts(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
        if validation_required
        else True
    )
    source_passed = bool(preconditions) and all(row.get("passed") is True for row in preconditions)
    gates = [
        _gate(
            "source_authentication",
            "validity",
            True,
            source_passed,
            source_passed,
            "All required sources and original flags must authenticate.",
            upstream="declared source manifest",
            path=None,
            field="preconditions_checked[].passed",
        ),
        _gate(
            "three_independent_board_rows",
            "validity",
            True,
            reduction["board_names_exact"] and reduction["board_count"] == 3,
            reduction["board_names_exact"] and reduction["board_count"] == 3,
            "A missing board cannot borrow another board's claim.",
            upstream="board_rows",
            path=str(raw_evidence_reference.get("path")),
            field="board_rows[].board",
        ),
        _gate(
            "zero_current_hardware_operations",
            "validity",
            0,
            reduction["hardware_operation_count"],
            reduction["hardware_operation_count"] == 0,
            "This task is read-only even when changed evidence exists.",
            upstream="board_rows",
            path=str(raw_evidence_reference.get("path")),
            field="hardware_operations_issued",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            affected_passed,
            affected_passed,
            "Only the frozen affected manifest can qualify this artifact.",
            upstream="validation_receipts",
            path=None,
            field="required affected command exits",
        ),
        _gate(
            "gatemate_changed_physical_prerequisite",
            "external_prerequisite",
            True,
            changed,
            changed,
            "Only a dated operator physical change reopens later GateMate work.",
            upstream="Exp6559 changed-state boundary",
            path=str(changed_state_evidence.get("search_receipt_path")),
            field="accepted_receipt_count",
        ),
        _gate(
            "new_board_value",
            "benefit",
            1,
            0,
            False,
            "Host timing and historical board evidence provide no new board value.",
            upstream="current read-only audit",
            path=None,
            field="hardware_value_score",
        ),
    ]
    current = current_work_receipt.build_current_work_receipt(
        run_id="exp7459-v653-board-continuity",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="read_only_board_evidence_aggregation_no_llm",
        inference_substrate_details={
            "board_commands": 0,
            "model_commands": 0,
            "historical_device_evidence_only": True,
            "host_durability_context_only": True,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False},
    )
    blocked = next((row for row in rows if row.get("board") == "GateMate"), {})
    status = (
        "complete_null_board_continuity_changed_evidence_future_task_only"
        if changed
        else "complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "execution_host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "random_seed": None,
        "random_seed_reason": "No stochastic sampling, fitting, projection, or resampling occurs.",
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            key: deepcopy(dict(value)) for key, value in source_hashes.items()
        },
        "rows": rows,
        "board_rows": rows,
        "sample_size_budget": {
            "planned_independent_units": 3,
            "attempted_independent_units": len(rows),
            "completed_independent_units": sum(
                row.get("disposition") == "complete" for row in rows
            ),
            "failed_independent_units": 0,
            "blocked_independent_units": sum(row.get("disposition") == "blocked" for row in rows),
            "censored_independent_units": 0,
            "unstarted_independent_units": max(0, 3 - len(rows)),
            "stopping_rule": (
                "Reduce KV260, PolarFire, and GateMate once; issue no current board action."
            ),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": status,
        "verdict_class": "null",
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_required": validation_required,
        "validation_manifest": _validation_manifest(),
        "field_principles": {},
        "promotion_score": 0,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "changed_state_evidence": deepcopy(dict(changed_state_evidence)),
        "hardware_wishlist_disposition": deepcopy(dict(hardware_wishlist_disposition)),
        "hardware_operations_issued": [],
        "raw_evidence_reference": deepcopy(dict(raw_evidence_reference)),
        "independent_reduction": reduction,
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay_required": validation_required,
            "numbered_e2e_applicable": [],
            "reason": "Isolated reporting study; shared training, sampling, ARC, and bindings are unchanged.",
        },
        "blocked_board_summary": {
            "board": blocked.get("board"),
            "honest_verdict": blocked.get("honest_verdict"),
            "exact_next_prerequisite": blocked.get("exact_next_prerequisite"),
        },
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> list[str]:
    """Cold-check identity, raw reduction, score limits, and exact receipts."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in artifact]
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity_mismatch")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("current_model_boundary_invalid")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles_mismatch")
    errors.extend(
        f"current_receipt:{error}"
        for error in current_work_receipt.validate_current_work_receipt(artifact, root=root)
    )
    errors.extend(_source_replay_errors(artifact, root))
    errors.extend(_raw_evidence_errors(artifact, root))
    reduced = independent_reduce(artifact)
    if reduced["matches_declared"] is not True or artifact.get("independent_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    if any(
        artifact.get(field) != 0
        for field in ("hardware_ready_score", "hardware_value_score", "promotion_score")
    ):
        errors.append("hardware_scores_nonzero")
    boards = {
        str(row.get("board")): row
        for row in artifact.get("board_rows") or []
        if isinstance(row, Mapping)
    }
    if (
        boards.get("KV260", {}).get("future_access") != "ssh_only"
        or boards.get("PolarFire", {}).get("fpga_sampling_claimed") is not False
        or any(row.get("new_hardware_execution_claimed") is not False for row in boards.values())
        or any(row.get("hardware_operations_issued") != [] for row in boards.values())
    ):
        errors.append("board_claim_boundary_invalid")
    wishlist = artifact.get("hardware_wishlist_disposition")
    if (
        not isinstance(wishlist, Mapping)
        or wishlist.get("hardware_value_score") != 0
        or wishlist.get("web_page_establishes_access") is not False
    ):
        errors.append("hardware_wishlist_boundary_invalid")
    receipts = [
        dict(row) for row in artifact.get("validation_receipts") or [] if isinstance(row, Mapping)
    ]
    if artifact.get("validation_required") is True and not _required_receipts(
        receipts, validation_scope.REQUIRED_CHECK_NAMES
    ):
        errors.append("affected_receipts_invalid")
    if require_terminal and not _required_receipts(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_receipts_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_fixture_artifact(root: Path, private_root: Path) -> JsonDict:
    """Build a compact no-command artifact for reducer and mutation tests."""

    hardware = _load_object(root / HARDWARE_SOURCE_PATH)
    cutoff = _load_object(root / EXP6559_PATH)
    durable = _load_object(root / DURABLE_SOURCE_PATH)
    changed = {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260911",
        "search_receipt_path": str(private_root / "gatemate_search.json"),
        "search_receipt_hash": "sha256:" + "1" * 64,
        "eligibility_contract": deepcopy(disposition_reader.PHYSICAL_RECEIPT_CONTRACT),
        "hardware_operations_issued": [],
    }
    boards = reduce_board_rows(hardware, changed)
    normalized = normalize_changed_state(changed)
    wishlist = build_hardware_wishlist_disposition(durable)
    raw_path = private_root / "raw_evidence.json"
    current_work_receipt.atomic_json(
        raw_path,
        {
            "schema": "carnot.exp7459.raw_evidence.v1",
            "board_rows": boards,
            "changed_state_evidence": normalized,
            "hardware_wishlist_disposition": wishlist,
        },
    )
    critical = (HARDWARE_SOURCE_PATH, GRADUATION_PATH, EXP6559_PATH, DURABLE_SOURCE_PATH)
    source_hashes = {
        path.as_posix(): _source_record(path, root=root, role="fixture_source") for path in critical
    }
    now = time.monotonic_ns()
    return build_artifact(
        preconditions=authenticate_sources(hardware, cutoff, durable),
        source_hashes=source_hashes,
        board_rows=boards,
        changed_state_evidence=normalized,
        hardware_wishlist_disposition=wishlist,
        raw_evidence_reference={
            "path": str(raw_path),
            "sha256": current_work_receipt.sha256_file(raw_path),
        },
        validation_receipts=[],
        validation_required=False,
        sidecar_references=[],
        started_monotonic_ns=now,
        ended_monotonic_ns=now,
        phase_spans=[],
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:00+00:00",
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 and Exp7303 affected-file command plan."""

    return validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)


def _phase_span(
    phase: str, phase_started: float, run_started: float
) -> JsonDict:  # pragma: no cover
    """Record one disjoint monotonic phase span."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build cold replay, raw reduction, and unchanged strict readers."""

    python = ".venv/bin/python"
    common = (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (*common, "--cold-replay", str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (*common, "--independent-reduce", str(candidate)),
            "measured_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
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
            "measured_candidate",
        ),
    )
    categories = ("completion", "completion", "safety", "completion")
    return [
        validation_contract.PlannedCommand(spec, category, True)
        for spec, category in zip(specs, categories, strict=True)
    ]


def _entrypoint_receipt(started_at_utc: str, duration_s: float) -> JsonDict:  # pragma: no cover
    """Record the exact declared command as the capability E2E receipt."""

    argv = [".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE]
    return {
        "name": "declared_entrypoint_e2e",
        "command": " ".join(argv),
        "command_argv": argv,
        "scope": "capability_e2e",
        "command_category": "completion",
        "required": True,
        "started_at_utc": started_at_utc,
        "ended_at_utc": utc_now(),
        "duration_s": duration_s,
        "exit_code": 0,
        "timed_out": False,
        "passed": True,
        "log_path": None,
        "log_sha256": None,
        "environment": {
            key: os.environ[key]
            for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }


def run_experiment(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint.
    """Run read-only reduction, scoped validation, terminal readers, and publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    repo = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, source_hashes, hardware, _cutoff, durable = collect_preconditions(repo)
    if not all(row.get("passed") is True for row in preconditions):
        failure = next(row for row in preconditions if row.get("passed") is not True)
        raise RuntimeError(f"blocked_precondition:{failure}")
    spans.append(_phase_span("preconditions", phase_started, started))
    progress(started, "preconditions", "end", checks=len(preconditions))

    phase_started = time.monotonic()
    progress(started, "evidence", "start")
    physical = search_changed_state_evidence(repo, raw_dir / "gatemate_changed_state_evidence.json")
    normalized = normalize_changed_state(physical)
    board_rows = reduce_board_rows(hardware, physical)
    wishlist = build_hardware_wishlist_disposition(durable)
    sidecar = current_work_receipt.write_immutable_sidecar(
        raw_dir / "historical_model_and_scripted_receipts.json",
        scope="historical_model_receipts",
        payload={
            "current_MODEL_SPECS": [],
            "current_model_invoked": False,
            "current_invocation_counts": ZERO_INVOCATION_COUNTS,
            "sources": [
                {
                    "path": HARDWARE_SOURCE_PATH.as_posix(),
                    "sha256": source_hashes[HARDWARE_SOURCE_PATH.as_posix()]["sha256"],
                    "historical_receipt_sidecars": hardware.get("receipt_sidecars") or [],
                },
                {
                    "path": DURABLE_SOURCE_PATH.as_posix(),
                    "sha256": source_hashes.get(DURABLE_SOURCE_PATH.as_posix(), {}).get("sha256"),
                    "historical_MODEL_SPECS": durable.get("MODEL_SPECS") if durable else None,
                    "historical_model_invoked": (durable.get("model_invoked") if durable else None),
                },
            ],
        },
        root=repo,
    )
    raw_path = raw_dir / "raw_evidence.json"
    current_work_receipt.atomic_json(
        raw_path,
        {
            "schema": "carnot.exp7459.raw_evidence.v1",
            "board_rows": board_rows,
            "changed_state_evidence": normalized,
            "hardware_wishlist_disposition": wishlist,
        },
    )
    raw_reference = {
        "path": raw_path.relative_to(repo).as_posix(),
        "sha256": current_work_receipt.sha256_file(raw_path),
    }
    spans.append(_phase_span("evidence", phase_started, started))
    progress(
        started,
        "evidence",
        "end",
        boards=len(board_rows),
        gatemate_changed=normalized["exists"],
    )

    phase_started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="exp7459-validation-", dir="/tmp"))
    commands = build_validation_plan(repo, private)
    plan_errors = validate_validation_plan(repo, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "validation", "before_affected_subprocesses", commands=len(commands))
    affected = validation_contract.run_categorized_commands(
        repo,
        [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ],
        log_dir=raw_dir / "validation/affected",
    )
    affected_reduction = validation_contract.reduce_affected_receipts(
        repo, AFFECTED_MANIFEST, affected
    )
    spans.append(_phase_span("affected_validation", phase_started, started))
    progress(
        started,
        "validation",
        "after_affected_subprocesses",
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError("required_affected_validation_failed")

    candidate_path = raw_dir / "measured_terminal_candidate.json"
    ended_ns = time.monotonic_ns()
    candidate = build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        board_rows=board_rows,
        changed_state_evidence=normalized,
        hardware_wishlist_disposition=wishlist,
        raw_evidence_reference=raw_reference,
        validation_receipts=affected,
        validation_required=True,
        sidecar_references=[sidecar],
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
    )
    errors = validate_artifact(candidate, root=repo, require_terminal=False)
    if errors:
        raise RuntimeError(f"measured_candidate_invalid:{errors}")
    current_work_receipt.atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_terminal_subprocesses")
    terminal = validation_contract.run_categorized_commands(
        repo,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_phase_span("terminal_validation", phase_started, started))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_terminal_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("required_terminal_validation_failed")

    entrypoint = _entrypoint_receipt(started_at, time.monotonic() - started)
    final = build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        board_rows=board_rows,
        changed_state_evidence=normalized,
        hardware_wishlist_disposition=wishlist,
        raw_evidence_reference=raw_reference,
        validation_receipts=[*affected, *terminal, entrypoint],
        validation_required=True,
        sidecar_references=[sidecar],
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
    )
    errors = validate_artifact(final, root=repo, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    destination = output_path if output_path.is_absolute() else repo / output_path
    progress(started, "publish", "before_atomic_write", path=destination)
    current_work_receipt.atomic_json(candidate_path, final)
    current_work_receipt.atomic_json(destination, final)
    progress(started, "publish", "after_atomic_write", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin public entrypoint and cold-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or one fresh-process read-only terminal check."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay:
        value = _load_object(args.cold_replay)
        errors = validate_artifact(value, root=REPO_ROOT, require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        value = _load_object(args.independent_reduce)
        reduced = independent_reduce(value)
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return int(reduced.get("matches_declared") is not True)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
