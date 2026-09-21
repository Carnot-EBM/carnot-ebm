"""Audit three board dispositions without contacting hardware.

The reducer authenticates prior bytes, narrows each historical claim, and
checks the approved GateMate receipt sources. Optional selector and prefix
results can guide future evidence needs. They cannot prove hardware access.

Spec refs: REQ-HW-7473 and SCENARIO-HW-7473-*.
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
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.654"
PHASE = 4
EXPERIMENT_ID = "exp7473-v654-board-continuity"
SCHEMA = "carnot.exp7473.v654.board_continuity.v1"

RESULT_PATH = Path("results/experiment_7473_v654_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7473_v654_board_continuity")
MODULE_PATH = Path("python/carnot/experiment_7473_v654_board_continuity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7473_v654_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7473_v654_board_continuity.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
PRIOR_PATH = Path("results/experiment_7459_v653_board_continuity.json")
GRADUATION_PATH = Path("results/experiment_7314_v642_board_continuity.json")
CUTOFF_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
SELECTOR_PATH = Path("results/experiment_7466_v654_typed_energy_calibration.json")
PREFIX_PATH = Path("results/experiment_7472_v654_prefix_service.json")

EXPECTED_SOURCE_HASHES = {
    PRIOR_PATH: "sha256:9d497771c5537896774c804c648c9aa3edcca2c0d8687de27adb1b1a46bfc769",
    GRADUATION_PATH: "sha256:c83cc85d16c082898992b3d34197d5d6bb07dfae5cd5bb9beca10bbbc20d51e6",
    CUTOFF_PATH: "sha256:59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66",
}
EXPECTED_PRIOR_ROW_HASHES = {
    "KV260": "sha256:6967ecc85882912cc1b4232cd5a67040d9ba6daaa195ac630bb3584e1c0debe5",
    "PolarFire": "sha256:fe1b9b6e13445757f0e3733c99f5fc75e6cc3fdb1a3e4b139305c2ea0bb9596e",
    "GateMate": "sha256:902dcf6385398aa6ed014b636b1794505b1256ee7475ecf4311ecc85f3d55b28",
}

PHYSICAL_RECEIPT_CONTRACT = deepcopy(disposition_reader.PHYSICAL_RECEIPT_CONTRACT)
ZERO_INVOCATION_COUNTS = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
GATEMATE_MISSING_RECEIPT = (
    "a dated operator-authored GateMate cable, port, power, board, JTAG, or "
    "DirtyJTAG physical-change receipt newer than Exp6559"
)

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7367_v646_board_disposition.py"),
    SPEC_PATH,
    PRIOR_PATH,
    GRADUATION_PATH,
    CUTOFF_PATH,
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    Path("ops/operator-followup.md"),
)

AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
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
    "clock_identity",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
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
    "duration_breakdown_s",
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
    "board_rows",
    "gatemate_changed_state_score",
    "hardware_operations_issued",
    "hardware_operation_count",
    "hardware_wishlist_disposition",
    "hardware_ready_score",
    "hardware_value_score",
    "promotion_score",
    "changed_state_evidence",
    "raw_evidence_reference",
    "independent_reduction",
    "capability_e2e",
    "execution_host",
}


def utc_now() -> str:
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and subprocess boundary with elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7473] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Load one JSON mapping and fail closed for malformed external bytes."""

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
    operator: str = "==",
) -> JsonDict:
    """Keep each expected and observed value in a stable gate shape."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "operator": operator,
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _identity_gates(
    value: Mapping[str, Any],
    *,
    label: str,
    path: Path,
    expected: Mapping[str, Any],
) -> list[JsonDict]:
    """Authenticate producer fields without changing an original flag."""

    return [
        _gate(
            f"{label}_{field}",
            "source_authentication",
            target,
            value.get(field),
            value.get(field) == target,
            "Only exact producer identity and original terminal flags enter this audit.",
            upstream=label,
            path=path.as_posix(),
            field=field,
        )
        for field, target in expected.items()
    ]


def _optional_producer_gate(
    value: Mapping[str, Any] | None, *, label: str, path: Path, experiment_id: str
) -> list[JsonDict]:
    """Accept absence, but authenticate a same-milestone producer when present."""

    if value is None:
        return [
            _gate(
                f"{label}_optional",
                "optional_context",
                "produced_or_absent",
                "absent",
                True,
                "An unexecuted earlier task stays absent and cannot block board accounting.",
                upstream=label,
                path=path.as_posix(),
                field="presence",
            )
        ]
    expected = {
        "experiment_id": experiment_id,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "flagged_adversarial": False,
    }
    gates = _identity_gates(value, label=label, path=path, expected=expected)
    verdict = value.get("verdict_class")
    gates.append(
        _gate(
            f"{label}_verdict_class",
            "source_authentication",
            ["null", "positive", "circular_positive"],
            verdict,
            verdict in {"null", "positive", "circular_positive"},
            "Only a terminal valid producer can inform a future hardware route.",
            upstream=label,
            path=path.as_posix(),
            field="verdict_class",
            operator="in",
        )
    )
    return gates


def authenticate_sources(
    prior: Mapping[str, Any],
    graduation: Mapping[str, Any],
    selector: Mapping[str, Any] | None,
    prefix: Mapping[str, Any] | None,
) -> list[JsonDict]:
    """Authenticate historical board rows, flags, and optional V654 context."""

    gates = _identity_gates(
        prior,
        label="prior",
        path=PRIOR_PATH,
        expected={
            "schema": "carnot.exp7459.v653.board_continuity.v1",
            "experiment_id": "exp7459-v653-board-continuity",
            "milestone": "2026.09.653",
            "run_date": "20260920",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "hardware_ready_score": 0,
            "hardware_value_score": 0,
        },
    )
    gates.extend(
        _identity_gates(
            graduation,
            label="graduation",
            path=GRADUATION_PATH,
            expected={
                "schema": "carnot.exp7314.v642.board_continuity.v1",
                "experiment_id": 7314,
                "milestone": "2026.09.642",
                "run_date": "20260915",
                "verdict_class": "positive",
            },
        )
    )
    rows = prior.get("board_rows")
    by_board = {str(row.get("board")): row for row in rows or [] if isinstance(row, Mapping)}
    gates.append(
        _gate(
            "prior_board_names",
            "source_authentication",
            ["GateMate", "KV260", "PolarFire"],
            sorted(by_board),
            sorted(by_board) == ["GateMate", "KV260", "PolarFire"],
            "Each board needs its own source row.",
            upstream="prior",
            path=PRIOR_PATH.as_posix(),
            field="board_rows[].board",
        )
    )
    for board, expected_hash in EXPECTED_PRIOR_ROW_HASHES.items():
        observed = by_board.get(board, {}).get("row_sha256")
        gates.append(
            _gate(
                f"prior_{board.lower()}_row_hash",
                "source_authentication",
                expected_hash,
                observed,
                observed == expected_hash,
                "A changed source row cannot inherit its historical graduation.",
                upstream="prior",
                path=PRIOR_PATH.as_posix(),
                field=f"board_rows[{board}].row_sha256",
            )
        )
    gates.extend(
        _optional_producer_gate(
            selector,
            label="selector",
            path=SELECTOR_PATH,
            experiment_id="exp7466-v654-typed-energy-calibration",
        )
    )
    gates.extend(
        _optional_producer_gate(
            prefix,
            label="prefix_service",
            path=PREFIX_PATH,
            experiment_id="exp7472-v654-prefix-service",
        )
    )
    return gates


def normalize_changed_state(changed_state: Mapping[str, Any]) -> JsonDict:
    """Reduce one approved receipt search without creating board authority."""

    exists = changed_state.get("exists") is True
    conditions = changed_state.get("changed_conditions")
    changed = dict(conditions) if isinstance(conditions, Mapping) else {}
    return {
        "exists": exists,
        "disposition": (
            "changed_physical_prerequisite_recorded_future_probe_only"
            if exists
            else "blocked_unchanged_physical_prerequisite"
        ),
        "cutoff_experiment": "Exp6559",
        "cutoff_date": "20260823",
        "accepted_receipt_count": changed_state.get("accepted_receipt_count", 0),
        "latest_receipt_date": changed_state.get("latest_receipt_date"),
        "receipt_timestamp": changed_state.get("receipt_timestamp"),
        "changed_fields": sorted(name for name, value in changed.items() if value is True),
        "evidence_path": changed_state.get("evidence_path"),
        "evidence_hash": changed_state.get("evidence_hash"),
        "search_receipt_path": changed_state.get("search_receipt_path"),
        "search_receipt_hash": changed_state.get("search_receipt_hash"),
        "eligibility_contract": deepcopy(
            changed_state.get("eligibility_contract") or PHYSICAL_RECEIPT_CONTRACT
        ),
        "missing_receipt": None if exists else GATEMATE_MISSING_RECEIPT,
        "hardware_operations_issued": [],
        "detect_count": 0,
        "flash_count": 0,
        "power_command_count": 0,
        "ssh_command_count": 0,
        "remote_command_count": 0,
    }


def search_changed_state_evidence(root: Path, raw_path: Path) -> JsonDict:  # pragma: no cover
    """Run only the approved local text parser and bind its raw search bytes."""

    result = disposition_reader.search_changed_state_receipt(root, raw_path)
    raw = _load_object(raw_path)
    candidates = [dict(row) for row in raw.get("candidate_rows") or [] if isinstance(row, Mapping)]
    accepted = [row for row in candidates if row.get("valid") is True]
    selected = max(accepted, key=lambda row: str(row.get("receipt_date"))) if accepted else {}
    dates = [str(row["receipt_date"]) for row in candidates if row.get("receipt_date")]
    raw.update(
        {
            "schema": "carnot.exp7473.gatemate_changed_state_evidence.v1",
            "run_date": RUN_DATE,
            "reader": (
                "carnot.experiment_7367_v646_board_disposition.search_changed_state_receipt"
            ),
            "cutoff_experiment": "Exp6559",
            "hardware_operations_issued": [],
        }
    )
    current_work_receipt.atomic_json(raw_path, raw)
    fields = selected.get("material_physical_fields")
    result.update(
        {
            "latest_receipt_date": max(dates) if dates else "20260823",
            "receipt_timestamp": (selected.get("raw_receipt") or {}).get("receipt_timestamp")
            if isinstance(selected.get("raw_receipt"), Mapping)
            else None,
            "changed_conditions": {str(name): True for name in fields or []},
            "evidence_path": result.get("source_path"),
            "evidence_hash": result.get("evidence_hash"),
            "search_receipt_path": raw_path.relative_to(root).as_posix(),
            "search_receipt_hash": current_work_receipt.sha256_file(raw_path),
            "accepted_receipt_count": len(accepted),
            "eligibility_contract": deepcopy(PHYSICAL_RECEIPT_CONTRACT),
            "hardware_operations_issued": [],
        }
    )
    return result


def reduce_board_rows(prior: Mapping[str, Any], changed_state: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve three narrow claims and update only current dispositions."""

    source_rows = prior.get("board_rows")
    if not isinstance(source_rows, list):
        raise ValueError("prior_board_rows_missing")
    by_board = {str(row.get("board")): row for row in source_rows if isinstance(row, Mapping)}
    if set(by_board) != {"KV260", "PolarFire", "GateMate"}:
        raise ValueError("prior_board_rows_invalid")
    physical = normalize_changed_state(changed_state)
    changed = physical["exists"] is True
    rows: list[JsonDict] = []
    for board in ("KV260", "PolarFire", "GateMate"):
        source = by_board[board]
        row = deepcopy(dict(source))
        source_row_hash = row.pop("row_sha256", None)
        row.update(
            {
                "source_artifact_path": PRIOR_PATH.as_posix(),
                "source_artifact_sha256": EXPECTED_SOURCE_HASHES[PRIOR_PATH],
                "source_row_sha256": source_row_hash,
                "source_original_status": prior.get("status"),
                "source_original_verdict_class": prior.get("verdict_class"),
                "source_original_flagged_adversarial": prior.get("flagged_adversarial"),
                "current_disposition_date": RUN_DATE,
                "current_execution_venue": "host",
                "historical_evidence_only": True,
                "present_reachability_asserted": False,
                "new_hardware_execution_claimed": False,
                "hardware_operations_issued": [],
                "hardware_operation_count": 0,
                "hardware_ready_score": 0,
                "hardware_value_score": 0,
                "read_only_audit": True,
                "failed": False,
                "censored": False,
                "row_type": "board_disposition",
                "arm": "aggregation_from_upstream_artifacts",
                "seed": None,
                "unit_id": f"board:{board}",
                "gate_check_summary": None,
            }
        )
        if board == "KV260":
            row.update(
                {
                    "disposition": "complete",
                    "current_disposition": "graduated_preserved",
                    "exact_claim_scope": "historical_kv260_fpga_fabric_sampling_only",
                    "architecture_limit": "k_max<=5",
                    "future_access": "ssh kria only",
                    "access_mechanism": "ssh kria only",
                    "fpga_sampling_claimed": True,
                    "hash_matched_cpu_dispatch": False,
                    "metric": True,
                    "abstention": False,
                    "error": None,
                }
            )
        elif board == "PolarFire":
            row.update(
                {
                    "disposition": "complete",
                    "current_disposition": "graduated_cpu_dispatch_preserved",
                    "exact_claim_scope": (
                        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
                    ),
                    "processor_class": "polarfire_linux_cpu",
                    "hash_matched_cpu_dispatch": True,
                    "fpga_sampling_claimed": False,
                    "fpga_sampling_status": "separate_unmeasured_future_task",
                    "metric": True,
                    "abstention": False,
                    "error": None,
                }
            )
        else:
            search_path = str(physical.get("search_receipt_path"))
            summary = None
            if not changed:
                summary = {
                    "check": "dated_operator_cable_port_power_board_or_dirtyjtag_change",
                    "upstream": "Exp6559 physical boundary",
                    "path": search_path,
                    "field": "accepted_receipt_count",
                    "expected": ">0",
                    "observed": physical.get("accepted_receipt_count", 0),
                    "operator": ">",
                    "passed": False,
                }
            row.update(
                {
                    "disposition": "complete" if changed else "blocked",
                    "current_disposition": (
                        "changed_physical_prerequisite_future_probe_only"
                        if changed
                        else "blocked_unchanged_physical_prerequisite"
                    ),
                    "availability_class": "future_probe_eligible" if changed else "blocked",
                    "exact_claim_scope": "no_current_execution_changed_state_gate_only",
                    "fpga_sampling_claimed": False,
                    "hash_matched_cpu_dispatch": False,
                    "honest_verdict": (
                        "complete_changed_physical_prerequisite_future_probe_only"
                        if changed
                        else "blocked_unchanged_physical_prerequisite"
                    ),
                    "metric": changed,
                    "abstention": not changed,
                    "error": None if changed else GATEMATE_MISSING_RECEIPT,
                    "exact_next_prerequisite": (
                        "a separately reviewed bounded future probe"
                        if changed
                        else GATEMATE_MISSING_RECEIPT
                    ),
                    "changed_state_evidence": deepcopy(physical),
                    "gatemate_changed_state_score": int(changed),
                    "gate_check_summary": summary,
                    "future_probe": {
                        "authorized_in_current_task": False,
                        "requires_separate_review": True,
                        "maximum_detect_actions": 1,
                        "flash_authorized": False,
                    },
                    "latest_operator_receipt_date": physical.get("latest_receipt_date"),
                    "evidence_path": search_path,
                    "evidence_sha256": physical.get("search_receipt_hash"),
                    "missing_receipt": physical.get("missing_receipt"),
                }
            )
        row["row_sha256"] = current_work_receipt.canonical_hash(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        rows.append(row)
    return rows


def _producer_evidence(
    value: Mapping[str, Any] | None,
    *,
    path: Path,
    score_fields: Sequence[str],
) -> JsonDict:
    """Retain optional producer flags and scores without inventing availability."""

    if value is None:
        return {
            "availability": "not_produced",
            "path": path.as_posix(),
            "sha256": None,
            **{field: None for field in score_fields},
        }
    return {
        "availability": "produced",
        "path": path.as_posix(),
        "experiment_id": value.get("experiment_id"),
        "status": value.get("status"),
        "honest_verdict": value.get("honest_verdict"),
        "verdict_class": value.get("verdict_class"),
        "flagged_adversarial": value.get("flagged_adversarial"),
        **{field: value.get(field) for field in score_fields},
    }


def build_hardware_wishlist_disposition(
    selector: Mapping[str, Any] | None, prefix: Mapping[str, Any] | None
) -> JsonDict:
    """Map software evidence to future routes without claiming device access."""

    return {
        "selector_evidence": _producer_evidence(
            selector,
            path=SELECTOR_PATH,
            score_fields=(
                "static_capture_complete_score",
                "static_probability_value_score",
                "typed_decision_value_score",
            ),
        ),
        "prefix_service_evidence": _producer_evidence(
            prefix,
            path=PREFIX_PATH,
            score_fields=("prefix_parity_score", "prefix_service_value_score"),
        ),
        "npu_route": {
            "device": "AMD XDNA NPU",
            "local_hardware_reported_in_wishlist": True,
            "software_prerequisite_satisfied": False,
            "missing_prerequisite": (
                "AMD custom onnxruntime with the VitisAI Execution Provider, or an "
                "authenticated supported XDNA toolchain"
            ),
            "sdk_or_paper_proves_availability": False,
            "local_workload_timing_available": False,
        },
        "larger_fpga_route": {
            "kv260_architecture_limit": "k_max<=5",
            "evidence_required": (
                "an authenticated workload above k_max<=5 with complete transfer, "
                "verification, and acknowledged service timing"
            ),
            "local_access_authenticated": False,
        },
        "tsu_route": {
            "device": "Extropic TSU",
            "authenticated_access": False,
            "missing_prerequisite": (
                "authorized local or vendor-evaluation access with end-to-end transfer, "
                "readout, accuracy, and acknowledged service receipts"
            ),
            "vendor_material_proves_availability": False,
        },
        "paper_or_sdk_proves_device_availability": False,
        "hardware_speed_claimed": False,
        "hardware_power_claimed": False,
        "hardware_energy_claimed": False,
        "production_readiness_claimed": False,
        "purchase_count": 0,
        "vendor_contact_count": 0,
        "hardware_value_score": 0,
    }


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute board scope and operation counts from the raw rows."""

    rows = [row for row in artifact.get("board_rows") or [] if isinstance(row, Mapping)]
    by_board = {str(row.get("board")): row for row in rows}
    exact = set(by_board) == {"KV260", "PolarFire", "GateMate"}
    operations = sum(len(row.get("hardware_operations_issued") or []) for row in rows)
    changed = by_board.get("GateMate", {}).get("changed_state_evidence", {}).get("exists") is True
    reduced = {
        "board_count": len(rows),
        "board_names_exact": exact,
        "graduated_count": sum(row.get("disposition") == "complete" for row in rows),
        "blocked_count": sum(row.get("disposition") == "blocked" for row in rows),
        "kv260_k_max_preserved": by_board.get("KV260", {}).get("architecture_limit") == "k_max<=5",
        "kv260_ssh_kria_only": by_board.get("KV260", {}).get("future_access") == "ssh kria only",
        "polarfire_cpu_not_fpga": (
            by_board.get("PolarFire", {}).get("hash_matched_cpu_dispatch") is True
            and by_board.get("PolarFire", {}).get("fpga_sampling_claimed") is False
        ),
        "gatemate_changed_state_score": int(changed),
        "hardware_operation_count": operations,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
    }
    declared_changed = artifact.get("gatemate_changed_state_score", int(changed))
    declared_operations = artifact.get("hardware_operation_count", operations)
    reduced["matches_declared"] = (
        exact
        and reduced["kv260_k_max_preserved"]
        and reduced["kv260_ssh_kria_only"]
        and reduced["polarfire_cpu_not_fpga"]
        and declared_changed == int(changed)
        and declared_operations == operations
        and artifact.get("hardware_operations_issued", []) == []
        and artifact.get("hardware_ready_score", 0) == 0
        and artifact.get("hardware_value_score", 0) == 0
    )
    return reduced


def _required_receipts(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one passing receipt for every exact command name."""

    for name in names:
        matching = [row for row in receipts if row.get("name") == name]
        if len(matching) != 1 or matching[0].get("passed") is not True:
            return False
    return True


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep external and benefit failures separate from validity failures."""

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
    """Explain why every artifact field exists."""

    specific = {
        "schema": "Use a versioned schema with exact experiment, milestone, and terminal status.",
        "run_date": "Use 20260921 with measured UTC and monotonic boundaries.",
        "preconditions_checked": "Record exact paths, ownership, device identity, and observed prerequisites.",
        "MODEL_SPECS": "Use an empty list because this task makes no current LLM call.",
        "model_specs": "Mirror the empty current model list for lowercase readers.",
        "model_invoked": "Distinguish current attempted calls from archived model-shaped evidence.",
        "invocation_counts": "Balance attempted and terminal current model call states.",
        "inference_substrate": "Use aggregation_from_upstream_artifacts for this pure reducer.",
        "inference_substrate_class": "Declare aggregation because no model or board work runs.",
        "execution_venue": "Use host and keep historical device venues inside board rows.",
        "duration_s": "Measure current work with a monotonic clock and never pad it.",
        "phase_spans": "Bind progress events, monotonic timings, and completed-unit checkpoints.",
        "random_seed": "Use null because this evidence audit is deterministic.",
        "reproducibility_checksum": "Bind code, protocol, sources, raw rows, and validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes, flags, and terminal classes.",
        "rows": "Keep one per-board unit, including failures, blocks, and abstentions.",
        "sample_size_budget": "Account for planned, attempted, complete, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Keep validity, external prerequisite, and benefit gates distinct.",
        "gate_check_summary": "Name the first exact failed check and retain all failed checks.",
        "honest_verdict": "Use a complete terminal null without hiding the GateMate block.",
        "verdict_class": "Use the closed null class for a valid audit with no new hardware value.",
        "verifier_is_oracle": "Use false because source receipts, not this verifier, provide evidence.",
        "flagged_adversarial": "Retain any structural reader failure without clearing it for a gate.",
        "validation_receipts": "Capture exact affected commands, exits, timing, environments, and log hashes.",
        "field_principles": "Explain each field without wrapping its ordinary JSON value.",
        "board_rows": "Separate historical graduation, current receipt availability, and future prerequisites.",
        "gatemate_changed_state_score": "Use a bare 0 or 1 from a dated physical-change receipt, not detection.",
        "hardware_operations_issued": "Keep empty because this is a read-only evidence audit.",
        "hardware_wishlist_disposition": "Map selector and service evidence to NPU and TSU prerequisites only.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable sources, rows, gates, and validation scope."""

    fields = (
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
    return current_work_receipt.canonical_hash({key: artifact.get(key) for key in fields})


def _source_record(path: Path, *, root: Path, role: str) -> JsonDict:
    """Hash exact source bytes and retain original terminal flags."""

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
    list[JsonDict], dict[str, JsonDict], JsonDict, JsonDict, JsonDict | None, JsonDict | None
]:  # pragma: no cover
    """Authenticate required bytes, exact producers, flags, and optional inputs."""

    gates: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        gates.append(
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
    for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
        observed = sources.get(relative.as_posix(), {}).get("sha256")
        gates.append(
            _gate(
                f"exact_bytes:{relative.as_posix()}",
                "source_authentication",
                expected_hash,
                observed,
                observed == expected_hash,
                "Historical authority is byte-bound.",
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="sha256",
            )
        )
    prior = _load_object(root / PRIOR_PATH)
    graduation = _load_object(root / GRADUATION_PATH)
    selector = _load_object(root / SELECTOR_PATH) if (root / SELECTOR_PATH).is_file() else None
    prefix = _load_object(root / PREFIX_PATH) if (root / PREFIX_PATH).is_file() else None
    for relative, value in ((SELECTOR_PATH, selector), (PREFIX_PATH, prefix)):
        if value is not None:
            sources[relative.as_posix()] = _source_record(
                relative, root=root, role="optional_same_milestone_context"
            )
    gates.extend(authenticate_sources(prior, graduation, selector, prefix))
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    refs_present = all(
        ref in spec_text
        for ref in (
            "REQ-HW-7473",
            "SCENARIO-HW-7473-UNCHANGED",
            "SCENARIO-HW-7473-CHANGED",
            "SCENARIO-HW-7473-WISHLIST",
        )
    )
    gates.append(
        _gate(
            "driving_spec",
            "source_authentication",
            True,
            refs_present,
            refs_present,
            "The capability requirement must exist before implementation executes.",
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-HW-7473 and scenarios",
        )
    )
    return gates, sources, prior, graduation, selector, prefix


def _validation_manifest() -> JsonDict:
    """Freeze the exact affected-only files before subprocess work."""

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
    """Build one complete null artifact from raw rows and measured receipts."""

    rows = [deepcopy(dict(row)) for row in board_rows]
    reduction = independent_reduce({"board_rows": rows})
    changed = reduction["gatemate_changed_state_score"] == 1
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
            "All required bytes, identities, rows, and original flags must authenticate.",
            upstream="declared sources",
            path=None,
            field="preconditions_checked[].passed",
        ),
        _gate(
            "three_board_dispositions",
            "validity",
            True,
            reduction["board_names_exact"],
            reduction["board_names_exact"],
            "A board cannot borrow another board's evidence.",
            upstream="raw board rows",
            path=str(raw_evidence_reference.get("path")),
            field="board_rows[].board",
        ),
        _gate(
            "zero_current_hardware_operations",
            "validity",
            0,
            reduction["hardware_operation_count"],
            reduction["hardware_operation_count"] == 0,
            "This task is read-only even when a changed receipt exists.",
            upstream="raw board rows",
            path=str(raw_evidence_reference.get("path")),
            field="hardware_operations_issued",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            affected_passed,
            affected_passed,
            "Only the frozen affected-file command set can qualify the artifact.",
            upstream="validation receipts",
            path=None,
            field="required affected command exits",
        ),
        _gate(
            "gatemate_changed_physical_prerequisite",
            "external_prerequisite",
            1,
            reduction["gatemate_changed_state_score"],
            changed,
            "Only a dated operator physical change permits a future reviewed probe.",
            upstream="Exp6559 physical boundary",
            path=str(changed_state_evidence.get("search_receipt_path")),
            field="accepted_receipt_count",
            operator=">=",
        ),
        _gate(
            "new_hardware_value",
            "benefit",
            1,
            0,
            False,
            "Historical evidence and host aggregation provide no new hardware value.",
            upstream="current audit",
            path=None,
            field="hardware_value_score",
        ),
    ]
    current = current_work_receipt.build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={
            "board_commands": 0,
            "model_commands": 0,
            "historical_device_evidence_only": True,
            "optional_selector_present": (
                hardware_wishlist_disposition.get("selector_evidence", {}).get("availability")
                == "produced"
            ),
            "optional_prefix_service_present": (
                hardware_wishlist_disposition.get("prefix_service_evidence", {}).get("availability")
                == "produced"
            ),
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={
            "performed": False,
            "reason": "Numeric head fitting belongs to its separate producer receipt.",
        },
    )
    status = (
        "complete_null_board_continuity_changed_receipt_future_probe_only"
        if changed
        else "complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite"
    )
    phase_totals = {
        "aggregation": sum(
            float(row.get("duration_s", 0.0))
            for row in phase_spans
            if row.get("phase") in {"preconditions", "evidence"}
        ),
        "validation": sum(
            float(row.get("duration_s", 0.0))
            for row in phase_spans
            if row.get("phase") in {"affected_validation", "terminal_validation"}
        ),
        "model_load": 0.0,
        "forward": 0.0,
        "generation": 0.0,
        "numeric_fitting": 0.0,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "clock_identity": {
            "utc": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **current,
        "model_specs": [],
        "duration_breakdown_s": phase_totals,
        "execution_host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_execution": True,
            "cuda_execution": False,
        },
        "random_seed": None,
        "random_seed_reason": "The audit uses deterministic ordering and no sampling or fit.",
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
            "stopping_rule": "Reduce each named board once and issue no hardware action.",
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
        "gatemate_changed_state_score": reduction["gatemate_changed_state_score"],
        "hardware_operations_issued": [],
        "hardware_operation_count": reduction["hardware_operation_count"],
        "hardware_wishlist_disposition": deepcopy(dict(hardware_wishlist_disposition)),
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "promotion_score": 0,
        "changed_state_evidence": deepcopy(dict(changed_state_evidence)),
        "raw_evidence_reference": deepcopy(dict(raw_evidence_reference)),
        "independent_reduction": {},
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay_required": validation_required,
            "numbered_e2e_applicable": [],
            "reason": (
                "This isolated reporting change does not alter shared training, sampling, "
                "ARC, bindings, or Rust code."
            ),
        },
    }
    artifact["independent_reduction"] = independent_reduce(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _source_replay_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Rehash each declared replay source."""

    errors: list[str] = []
    for label, record in (artifact.get("source_artifact_hashes") or {}).items():
        if not isinstance(record, Mapping) or record.get("verify_on_replay") is not True:
            continue
        path = root / str(record.get("path") or label)
        observed = current_work_receipt.sha256_file(path) if path.is_file() else None
        if observed != record.get("sha256"):
            errors.append(f"source_hash:{label}")
    return errors


def _raw_evidence_errors(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Verify the raw row shard and its embedded rows."""

    reference = artifact.get("raw_evidence_reference")
    if not isinstance(reference, Mapping):
        return ["raw_reference"]
    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        return ["raw_missing"]
    if current_work_receipt.sha256_file(resolved) != reference.get("sha256"):
        return ["raw_hash"]
    raw = _load_object(resolved)
    if raw.get("board_rows") != artifact.get("board_rows"):
        return ["raw_board_rows"]
    return []


def validate_artifact(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> list[str]:
    """Cold-check identity, boundaries, rows, hashes, and exact receipts."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in artifact]
    identity = (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    )
    if identity != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity")
    if artifact.get("MODEL_SPECS") != []:
        errors.append("uppercase_model_specs")
    if artifact.get("model_specs") != []:
        errors.append("lowercase_model_specs")
    if (
        artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("current_inference_boundary")
    errors.extend(
        f"current_receipt:{error}"
        for error in current_work_receipt.validate_current_work_receipt(artifact, root=root)
    )
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field_principles")
    errors.extend(_source_replay_errors(artifact, root))
    errors.extend(_raw_evidence_errors(artifact, root))
    reduced = independent_reduce(artifact)
    if reduced.get("matches_declared") is not True:
        errors.append("independent_reduction")
    if artifact.get("independent_reduction") != reduced:
        errors.append("independent_reduction_receipt")
    if artifact.get("gatemate_changed_state_score") != reduced["gatemate_changed_state_score"]:
        errors.append("gatemate_score")
    if artifact.get("hardware_operations_issued") != []:
        errors.append("hardware_operations")
    if artifact.get("hardware_operation_count") != 0:
        errors.append("hardware_operation_count")
    if any(
        artifact.get(field) != 0
        for field in ("hardware_ready_score", "hardware_value_score", "promotion_score")
    ):
        errors.append("hardware_scores")
    boards = {
        str(row.get("board")): row
        for row in artifact.get("board_rows") or []
        if isinstance(row, Mapping)
    }
    if (
        boards.get("KV260", {}).get("architecture_limit") != "k_max<=5"
        or boards.get("KV260", {}).get("future_access") != "ssh kria only"
        or boards.get("PolarFire", {}).get("hash_matched_cpu_dispatch") is not True
        or boards.get("PolarFire", {}).get("fpga_sampling_claimed") is not False
    ):
        errors.append("board_claim_boundary")
    for row in boards.values():
        expected_hash = current_work_receipt.canonical_hash(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        if row.get("row_sha256") != expected_hash:
            errors.append(f"row_hash:{row.get('board')}")
        if row.get("disposition") == "blocked" and not isinstance(
            row.get("gate_check_summary"), Mapping
        ):
            errors.append("blocked_board_gate_summary")
    wishlist = artifact.get("hardware_wishlist_disposition")
    if (
        not isinstance(wishlist, Mapping)
        or wishlist.get("paper_or_sdk_proves_device_availability") is not False
        or wishlist.get("hardware_speed_claimed") is not False
        or wishlist.get("purchase_count") != 0
        or wishlist.get("vendor_contact_count") != 0
    ):
        errors.append("wishlist_boundary")
    receipts = [
        row for row in artifact.get("validation_receipts") or [] if isinstance(row, Mapping)
    ]
    if artifact.get("validation_required") is True and not _required_receipts(
        receipts, validation_scope.REQUIRED_CHECK_NAMES
    ):
        errors.append("affected_receipts")
    if require_terminal and not _required_receipts(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_receipts")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def build_fixture_artifact(root: Path, private_root: Path) -> JsonDict:
    """Build a compact terminal-shaped record for mutation tests."""

    prior = _load_object(root / PRIOR_PATH)
    graduation = _load_object(root / GRADUATION_PATH)
    changed = {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260823",
        "search_receipt_path": str(private_root / "gatemate_search.json"),
        "search_receipt_hash": "sha256:" + "1" * 64,
        "eligibility_contract": deepcopy(PHYSICAL_RECEIPT_CONTRACT),
        "hardware_operations_issued": [],
    }
    rows = reduce_board_rows(prior, changed)
    normalized = normalize_changed_state(changed)
    wishlist = build_hardware_wishlist_disposition(None, None)
    raw_path = private_root / "raw_evidence.json"
    current_work_receipt.atomic_json(
        raw_path,
        {
            "schema": "carnot.exp7473.raw_evidence.v1",
            "board_rows": rows,
            "changed_state_evidence": normalized,
            "hardware_wishlist_disposition": wishlist,
        },
    )
    paths = (PRIOR_PATH, GRADUATION_PATH, CUTOFF_PATH)
    sources = {
        path.as_posix(): _source_record(path, root=root, role="fixture_source") for path in paths
    }
    now = time.monotonic_ns()
    return build_artifact(
        preconditions=authenticate_sources(prior, graduation, None, None),
        source_hashes=sources,
        board_rows=rows,
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
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:00+00:00",
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed affected-file commands through the shared helpers."""

    return validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing parents, and command drift."""

    return validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)


def _phase_span(
    phase: str, phase_started: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover
    """Record one disjoint monotonic phase and its completed unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
    }


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build fresh replay, independent reduction, and strict reader commands."""

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
    """Record the exact outer command as the capability E2E receipt."""

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
            for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            if key in os.environ
        },
    }


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Run read-only aggregation, scoped checks, terminal readers, and publish."""

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
    preconditions, source_hashes, prior, _graduation, selector, prefix = collect_preconditions(repo)
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        raise RuntimeError(f"blocked_precondition:{failed}")
    spans.append(_phase_span("preconditions", phase_started, started, len(preconditions)))
    progress(started, "preconditions", "end", completed_units=len(preconditions))

    phase_started = time.monotonic()
    progress(started, "evidence", "start")
    physical = search_changed_state_evidence(repo, raw_dir / "gatemate_changed_state_evidence.json")
    normalized = normalize_changed_state(physical)
    board_rows = reduce_board_rows(prior, physical)
    wishlist = build_hardware_wishlist_disposition(selector, prefix)
    sidecar = current_work_receipt.write_immutable_sidecar(
        raw_dir / "historical_model_and_scripted_receipts.json",
        scope="historical_model_receipts",
        payload={
            "current_MODEL_SPECS": [],
            "current_model_specs": [],
            "current_model_invoked": False,
            "current_invocation_counts": ZERO_INVOCATION_COUNTS,
            "sources": [
                {
                    "path": PRIOR_PATH.as_posix(),
                    "sha256": source_hashes[PRIOR_PATH.as_posix()]["sha256"],
                    "historical_MODEL_SPECS": prior.get("MODEL_SPECS"),
                    "historical_model_invoked": prior.get("model_invoked"),
                    "historical_receipt_sidecars": prior.get("receipt_sidecars") or [],
                },
                {
                    "path": SELECTOR_PATH.as_posix(),
                    "sha256": source_hashes.get(SELECTOR_PATH.as_posix(), {}).get("sha256"),
                    "historical_MODEL_SPECS": selector.get("MODEL_SPECS") if selector else None,
                    "historical_model_invoked": selector.get("model_invoked") if selector else None,
                },
                {
                    "path": PREFIX_PATH.as_posix(),
                    "sha256": source_hashes.get(PREFIX_PATH.as_posix(), {}).get("sha256"),
                    "historical_MODEL_SPECS": prefix.get("MODEL_SPECS") if prefix else None,
                    "historical_model_invoked": prefix.get("model_invoked") if prefix else None,
                },
            ],
        },
        root=repo,
    )
    raw_path = raw_dir / "raw_evidence.json"
    current_work_receipt.atomic_json(
        raw_path,
        {
            "schema": "carnot.exp7473.raw_evidence.v1",
            "board_rows": board_rows,
            "changed_state_evidence": normalized,
            "hardware_wishlist_disposition": wishlist,
        },
    )
    raw_reference = {
        "path": raw_path.relative_to(repo).as_posix(),
        "sha256": current_work_receipt.sha256_file(raw_path),
    }
    spans.append(_phase_span("evidence", phase_started, started, len(board_rows)))
    progress(
        started,
        "evidence",
        "end",
        completed_units=len(board_rows),
        gatemate_changed=normalized["exists"],
    )

    phase_started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="exp7473-validation-", dir="/tmp"))
    commands = build_validation_plan(repo, private)
    plan_errors = validate_validation_plan(repo, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "validation", "before_affected_subprocesses", completed_units=0)
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
    spans.append(_phase_span("affected_validation", phase_started, started, len(affected)))
    progress(
        started,
        "validation",
        "after_affected_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise RuntimeError("required_affected_validation_failed")

    candidate_path = raw_dir / "measured_terminal_candidate.json"
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
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
    )
    errors = validate_artifact(candidate, root=repo, require_terminal=False)
    if errors:
        raise RuntimeError(f"measured_candidate_invalid:{errors}")
    current_work_receipt.atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_terminal_subprocesses", completed_units=0)
    terminal = validation_contract.run_categorized_commands(
        repo,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_phase_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_terminal_subprocesses",
        completed_units=len(terminal),
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
    """Parse the public entrypoint and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the audit or one read-only fresh-process terminal check."""

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
