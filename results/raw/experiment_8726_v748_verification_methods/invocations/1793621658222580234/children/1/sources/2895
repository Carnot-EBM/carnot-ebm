"""Build the V647 hardware envelope from authenticated historical evidence.

This module performs JSON parsing, hashing, and deterministic arithmetic on the
host. It never contacts a board or model. A missing placement producer remains
missing, and a disqualified producer never becomes performance evidence.

Spec refs: REQ-REPORT-7379 and SCENARIO-REPORT-7379-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot import experiment_7367_v646_board_disposition as board_history
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7379
TASK_ID = "exp7379-hardware-envelope"
MILESTONE = "2026.09.647"
RUN_DATE = "20260918"
SCHEMA = "carnot.experiment_7379.v647_hardware_envelope.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7379_v647_hardware_envelope.json")
RAW_DIR = Path("results/raw/experiment_7379_v647_hardware_envelope")
BOARD_SOURCE_PATH = Path("results/experiment_7367_v646_board_disposition.json")
COST_SOURCE_PATH = Path("results/experiment_7340_v644_native_cost.json")
MEMORY_SOURCE_PATH = Path("results/experiment_7374_v647_prospective_memory.json")
ISING_SOURCE_PATH = Path("results/experiment_7378_v647_ising_audit.json")
VALIDATION_BOUNDARY_PATH = Path("results/experiment_7358_v646_validation_contract.json")
MODULE_PATH = Path("python/carnot/experiment_7379_v647_hardware_envelope.py")
TEST_PATH = Path("tests/python/test_experiment_7379_v647_hardware_envelope.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7379_v647_hardware_envelope.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

HYPOTHETICAL_KERNEL_RATE = 100.0
HUNDRED_X_UNACCELERATED_FRACTION_MAX = 0.01
INVALID_SCIENCE_CLASSES = {"blocked", "disqualified", "partial"}
SERVICE_FIELDS = (
    "solve_time_ns",
    "certificate_discovery_time_ns",
    "certificate_check_time_ns",
    "certificate_update_time_ns",
    "serialization_time_ns",
    "host_overhead_time_ns",
)
ZERO_CURRENT_COUNTS = {
    "loads": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
    "generations": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
}
PHYSICAL_RECEIPT_CONTRACT = deepcopy(board_history.PHYSICAL_RECEIPT_CONTRACT)
MISSING_RECEIPT = board_history.MISSING_RECEIPT

REQUIRED_FIELDS = {
    "schema",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
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
    "repository_health",
    "field_principles",
    "promotion_score",
    "board_disposition_complete_score",
    "board_rows",
    "changed_state_receipt",
    "placement_envelope_rows",
    "hardware_ready_score",
    "hardware_value_score",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("research-references.md"),
    SPEC_PATH,
    BOARD_SOURCE_PATH,
    COST_SOURCE_PATH,
    ISING_SOURCE_PATH,
    VALIDATION_BOUNDARY_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7367_v646_board_disposition.py"),
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

V647_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=TASK_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep retained evidence, validation logs, and terminal output separate."""

    artifact: Path
    raw_evidence: Path
    changed_state_search: Path
    historical_models: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:  # pragma: no cover
        """Resolve task outputs below the selected repository."""

        return cls.under(root)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Resolve isolated outputs below a repository or test directory."""

        raw = root / RAW_DIR
        return cls(
            artifact=root / RESULT_PATH,
            raw_evidence=raw / "hardware_envelope_evidence.json",
            changed_state_search=raw / "gatemate_changed_state_search.json",
            historical_models=raw / "historical_model_receipts.json",
            terminal_candidate=raw / "measured_terminal_candidate.json",
            validation_dir=raw / "validation",
        )


def utc_now() -> str:
    """Return an aware UTC timestamp for a real task boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase and subprocess boundaries so long work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7379] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a replacement cannot inherit evidence authority."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for rows and independently reduced claims."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete fsynced JSON bytes with one local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_object(path: Path) -> JsonDict | None:
    """Return one JSON object, or ``None`` for absent or malformed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _check(
    check: str,
    category: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Retain the exact expectation and observation for one gate."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _resolved_source_path(root: Path, value: Any) -> Path:
    """Resolve a recorded absolute or repository-relative evidence path."""

    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _receipt_authentication(root: Path, board: Mapping[str, Any]) -> dict[str, bool]:
    """Check every named board row and its latest retained receipt bytes."""

    result: dict[str, bool] = {}
    for row in board.get("board_rows") or []:
        if not isinstance(row, Mapping) or not isinstance(row.get("board"), str):
            continue
        bare = {key: item for key, item in row.items() if key != "row_sha256"}
        row_hash_ok = row.get("row_sha256") == canonical_hash(bare)
        receipt_path = _resolved_source_path(root, row.get("latest_receipt_path"))
        expected_hash = row.get("latest_receipt_hash")
        receipt_ok = receipt_path.is_file() and sha256_file(receipt_path) == expected_hash
        result[str(row["board"])] = row_hash_ok and receipt_ok
    return result


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate required bytes, producer classes, and referenced receipts."""

    for destination in paths.__dict__.values():
        destination.parent.mkdir(parents=True, exist_ok=True)

    checks: list[JsonDict] = []
    hashes: dict[str, str | None] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                f"source_bytes:{relative.as_posix()}",
                "required_source",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    memory_path = root / MEMORY_SOURCE_PATH
    hashes[MEMORY_SOURCE_PATH.as_posix()] = (
        sha256_file(memory_path) if memory_path.is_file() else None
    )

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    spec_ok = "REQ-REPORT-7379" in spec_text
    checks.append(
        _check(
            "driving_capability",
            "required_source",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7379",
            "REQ-REPORT-7379" if spec_ok else None,
            spec_ok,
        )
    )

    board = _load_object(root / BOARD_SOURCE_PATH) or {}
    cost = _load_object(root / COST_SOURCE_PATH) or {}
    boundary = _load_object(root / VALIDATION_BOUNDARY_PATH) or {}
    receipts = _receipt_authentication(root, board)
    board_valid = (
        board.get("experiment_id") == 7367
        and board.get("milestone") == "2026.09.646"
        and board.get("status") == "blocked"
        and board.get("verdict_class") == "blocked"
        and board.get("board_disposition_complete_score") == 1
        and board.get("required_checks_passed") is True
        and board.get("flagged_adversarial") is False
        and board_history.validate_artifact(board) == []
        and set(receipts) == {"KV260", "GateMate", "PolarFire"}
        and all(receipts.values())
    )
    checks.append(
        _check(
            "authenticated_board_history",
            "required_source",
            BOARD_SOURCE_PATH.as_posix(),
            "identity/class/row_hash/latest_receipt_hash",
            {
                "experiment_id": 7367,
                "verdict_class": "blocked_diagnostic",
                "boards": ["KV260", "GateMate", "PolarFire"],
                "all_receipts_authenticated": True,
            },
            {
                "experiment_id": board.get("experiment_id"),
                "verdict_class": board.get("verdict_class"),
                "boards": sorted(receipts),
                "all_receipts_authenticated": bool(receipts) and all(receipts.values()),
            },
            board_valid,
        )
    )

    cost_gate = (cost.get("acceptance_gate_results") or {}).get("native_ten_x") or {}
    cost_valid = (
        cost.get("experiment_id") == 7340
        and cost.get("milestone") == "2026.09.644"
        and cost.get("status") == "complete"
        and cost.get("verdict_class") == "null"
        and cost.get("native_cost_complete_score") == 1
        and cost.get("native_ten_x_score") == 0
        and cost_gate.get("passed") is False
        and cost.get("required_checks_passed") is True
    )
    checks.append(
        _check(
            "complete_cost_null",
            "required_source",
            COST_SOURCE_PATH.as_posix(),
            "identity/class/native_ten_x_score/acceptance_gate_results.native_ten_x.passed",
            [7340, "null", 0, False],
            [
                cost.get("experiment_id"),
                cost.get("verdict_class"),
                cost.get("native_ten_x_score"),
                cost_gate.get("passed"),
            ],
            cost_valid,
        )
    )

    boundary_ok = (
        boundary.get("experiment_id") == "exp7358-validation-contract"
        and boundary.get("validation_contract_ready_score") == 1
        and boundary.get("flagged_adversarial") is False
    )
    checks.append(
        _check(
            "exp7358_command_boundary",
            "required_source",
            VALIDATION_BOUNDARY_PATH.as_posix(),
            "validation_contract_ready_score",
            1,
            boundary.get("validation_contract_ready_score"),
            boundary_ok,
        )
    )

    context: JsonDict = {
        "board_source": {
            "experiment_id": board.get("experiment_id"),
            "status": board.get("status"),
            "verdict_class": board.get("verdict_class"),
            "board_disposition_complete_score": board.get("board_disposition_complete_score"),
            "diagnostic_only": True,
            "authorizes_readiness": False,
        },
        "cost_source": {
            "experiment_id": cost.get("experiment_id"),
            "status": cost.get("status"),
            "verdict_class": cost.get("verdict_class"),
            "native_cost_complete_score": cost.get("native_cost_complete_score"),
            "native_ten_x_score": cost.get("native_ten_x_score"),
            "complete_boundary_gate_passed": cost_gate.get("passed"),
        },
        "receipt_authentication": receipts,
        "validation_boundary": {
            "experiment_id": boundary.get("experiment_id"),
            "validation_contract_ready_score": boundary.get("validation_contract_ready_score"),
        },
    }
    return checks, hashes, context


def amdahl_bounds(replaceable_fraction: float, kernel_rate: float) -> JsonDict:
    """Compute finite and infinite-kernel bounds from one measured fraction."""

    if not 0.0 <= replaceable_fraction < 1.0:
        raise ValueError("replaceable_fraction must be in [0, 1)")
    if not math.isfinite(kernel_rate) or kernel_rate <= 0.0:
        raise ValueError("kernel_rate must be a positive finite number")
    unaccelerated = 1.0 - replaceable_fraction
    return {
        "hypothetical_kernel_rate": kernel_rate,
        "bounded_full_service_speedup": 1.0 / (unaccelerated + replaceable_fraction / kernel_rate),
        "infinite_kernel_upper_bound": 1.0 / unaccelerated,
        "hundred_x_necessary_condition": {
            "target_total_speedup": 100.0,
            "unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "observed_unaccelerated_fraction": unaccelerated,
            "passed": unaccelerated <= HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "condition_scope": "necessary_even_with_infinitely_fast_kernel_not_sufficient",
        },
    }


def _unavailable_placement_row(
    source_name: str,
    path: Path,
    source_class: str,
    failed_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Name one unusable producer without inventing performance measurements."""

    row: JsonDict = {
        "unit_id": f"placement:{source_name}",
        "row_type": "placement_envelope",
        "source_experiment": source_name,
        "source_path": path.as_posix(),
        "source_class": source_class,
        "outcome": "placement_input_unavailable",
        "failed_field": failed_field,
        "expected_value": expected,
        "observed_value": observed,
        "performance_evidence_eligible": False,
        "service_time_decomposition_ns": None,
        "measured_replaceable_fraction": None,
        "unaccelerated_fraction": None,
        "hypothetical_kernel_rate": HYPOTHETICAL_KERNEL_RATE,
        "bounded_full_service_speedup": None,
        "infinite_kernel_upper_bound": None,
        "hundred_x_necessary_condition": {
            "target_total_speedup": 100.0,
            "unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "observed_unaccelerated_fraction": None,
            "passed": False,
            "condition_scope": "necessary_even_with_infinitely_fast_kernel_not_evaluable",
        },
        "full_path_version_upload_count": None,
        "update_cadence_queries": None,
        "recommendation": "retain_cpu",
        "metric": None,
        "censored": True,
        "error": f"{failed_field}: expected {expected!r}, observed {observed!r}",
    }
    row["row_sha256"] = canonical_hash(row)
    return row


def assess_placement_source(
    source_name: str,
    path: Path,
    expected_identity: Any,
    artifact: Mapping[str, Any] | None,
    *,
    kernel_rate: float = HYPOTHETICAL_KERNEL_RATE,
) -> list[JsonDict]:
    """Reduce only an eligible producer with complete measured service rows."""

    if artifact is None:
        return [
            _unavailable_placement_row(
                source_name, path, "missing", "path", "readable JSON object", "missing"
            )
        ]
    source_class = str(artifact.get("verdict_class") or "unknown")
    eligibility = (
        ("experiment_id", expected_identity, artifact.get("experiment_id")),
        ("status", "terminal complete status", artifact.get("status")),
        (
            "verdict_class",
            "positive|circular_positive|null",
            artifact.get("verdict_class"),
        ),
        ("flagged_adversarial", False, artifact.get("flagged_adversarial")),
        ("required_checks_passed", True, artifact.get("required_checks_passed")),
    )
    for field, expected, observed in eligibility:
        if field == "status":
            passed = isinstance(observed, str) and observed.startswith("complete")
        elif field == "verdict_class":
            passed = observed not in INVALID_SCIENCE_CLASSES and observed in {
                "positive",
                "circular_positive",
                "null",
            }
        else:
            passed = observed == expected
        if not passed:
            return [
                _unavailable_placement_row(
                    source_name, path, source_class, field, expected, observed
                )
            ]

    service_rows = artifact.get("placement_service_rows")
    if not isinstance(service_rows, list) or not service_rows:
        return [
            _unavailable_placement_row(
                source_name,
                path,
                source_class,
                "placement_service_rows",
                "nonempty measured complete-query rows",
                service_rows,
            )
        ]

    reduced: list[JsonDict] = []
    for index, value in enumerate(service_rows):
        measured = value if isinstance(value, Mapping) else {}
        for field in SERVICE_FIELDS:
            observed = measured.get(field)
            if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                return [
                    _unavailable_placement_row(
                        source_name,
                        path,
                        source_class,
                        f"placement_service_rows[{index}].{field}",
                        "nonnegative measured number",
                        observed,
                    )
                ]
            if observed < 0:
                return [
                    _unavailable_placement_row(
                        source_name,
                        path,
                        source_class,
                        f"placement_service_rows[{index}].{field}",
                        "nonnegative measured number",
                        observed,
                    )
                ]
        component_total = sum(float(measured[field]) for field in SERVICE_FIELDS)
        total = measured.get("total_service_time_ns")
        if (
            not isinstance(total, (int, float))
            or isinstance(total, bool)
            or not math.isclose(float(total), component_total, rel_tol=0.0, abs_tol=1e-9)
        ):
            expected_total: int | float = (
                int(component_total) if component_total.is_integer() else component_total
            )
            return [
                _unavailable_placement_row(
                    source_name,
                    path,
                    source_class,
                    f"placement_service_rows[{index}].total_service_time_ns",
                    expected_total,
                    total,
                )
            ]
        components = measured.get("replaceable_components")
        if (
            not isinstance(components, list)
            or not components
            or any(component not in SERVICE_FIELDS for component in components)
        ):
            return [
                _unavailable_placement_row(
                    source_name,
                    path,
                    source_class,
                    f"placement_service_rows[{index}].replaceable_components",
                    f"nonempty subset of {list(SERVICE_FIELDS)}",
                    components,
                )
            ]
        uploads = measured.get("full_path_version_upload_count")
        cadence = measured.get("update_cadence_queries")
        if not isinstance(uploads, int) or isinstance(uploads, bool) or uploads < 0:
            return [
                _unavailable_placement_row(
                    source_name,
                    path,
                    source_class,
                    f"placement_service_rows[{index}].full_path_version_upload_count",
                    "nonnegative integer",
                    uploads,
                )
            ]
        if not isinstance(cadence, (int, float)) or isinstance(cadence, bool) or cadence <= 0:
            return [
                _unavailable_placement_row(
                    source_name,
                    path,
                    source_class,
                    f"placement_service_rows[{index}].update_cadence_queries",
                    "positive measured number",
                    cadence,
                )
            ]

        replaceable = sum(float(measured[field]) for field in components)
        fraction = replaceable / float(total)
        bounds = amdahl_bounds(fraction, kernel_rate)
        decomposition = {
            "solve": measured["solve_time_ns"],
            "certificate_discovery": measured["certificate_discovery_time_ns"],
            "certificate_check": measured["certificate_check_time_ns"],
            "certificate_update": measured["certificate_update_time_ns"],
            "serialization": measured["serialization_time_ns"],
            "host_overhead": measured["host_overhead_time_ns"],
            "total": total,
        }
        row = {
            "unit_id": f"placement:{source_name}:{measured.get('unit_id', index)}",
            "row_type": "placement_envelope",
            "source_experiment": source_name,
            "source_path": path.as_posix(),
            "source_class": source_class,
            "outcome": "measured_placement_envelope",
            "failed_field": None,
            "expected_value": None,
            "observed_value": None,
            "performance_evidence_eligible": True,
            "service_time_decomposition_ns": decomposition,
            "replaceable_components": list(components),
            "measured_replaceable_fraction": fraction,
            "unaccelerated_fraction": 1.0 - fraction,
            **bounds,
            "full_path_version_upload_count": uploads,
            "update_cadence_queries": cadence,
            "recommendation": (
                "evaluate_alternate_substrate"
                if bounds["bounded_full_service_speedup"] >= 100.0
                else "retain_cpu"
            ),
            "metric": bounds["bounded_full_service_speedup"],
            "censored": False,
            "error": None,
        }
        row["row_sha256"] = canonical_hash(row)
        reduced.append(row)
    return reduced


def load_placement_envelope(root: Path) -> list[JsonDict]:
    """Inspect both declared producers and preserve every ineligibility reason."""

    sources = (
        (
            "Exp7374",
            MEMORY_SOURCE_PATH,
            "exp7374-prospective-memory",
            _load_object(root / MEMORY_SOURCE_PATH),
        ),
        (
            "Exp7378",
            ISING_SOURCE_PATH,
            "exp7378-v647-ising-audit",
            _load_object(root / ISING_SOURCE_PATH),
        ),
    )
    rows: list[JsonDict] = []
    for name, path, identity, artifact in sources:
        rows.extend(assess_placement_source(name, path, identity, artifact))
    return rows


def build_board_rows(board: Mapping[str, Any], physical: Mapping[str, Any]) -> list[JsonDict]:
    """Retain three authenticated board claims with their exact venue boundary."""

    rows: list[JsonDict] = []
    for source in board.get("board_rows") or []:
        if not isinstance(source, Mapping) or source.get("board") not in {
            "KV260",
            "GateMate",
            "PolarFire",
        }:
            continue
        name = str(source["board"])
        terminal_state = str(source.get("disposition"))
        next_prerequisite = source.get("exact_next_condition")
        if name == "GateMate":
            changed = physical.get("exists") is True
            terminal_state = (
                "future_bounded_task_eligible" if changed else "blocked_changed_physical_state"
            )
            next_prerequisite = (
                board_history.board_source.GATEMATE_FUTURE_ACTION
                if changed
                else board_history.board_source.GATEMATE_OPERATOR_ACTION
            )
        row: JsonDict = {
            "unit_id": f"board:{name}",
            "row_type": "board_disposition",
            "board": name,
            "last_authenticated_venue": source.get("execution_venue"),
            "last_authenticated_date": source.get("evidence_date"),
            "last_authenticated_hash": source.get("latest_receipt_hash"),
            "last_authenticated_path": source.get("latest_receipt_path"),
            "terminal_state": terminal_state,
            "exact_next_prerequisite": next_prerequisite,
            "availability_class": source.get("availability_class"),
            "historical_evidence_only": True,
            "present_reachability_asserted": False,
            "new_hardware_execution_claimed": False,
            "fpga_sampling_claimed": source.get("fpga_sampling_claimed") is True,
            "future_access": "ssh_only" if name == "KV260" else None,
            "architecture_limit": "k_max<=5" if name == "KV260" else None,
            "hash_matched_cpu_dispatch": name == "PolarFire",
            "metric": terminal_state != "blocked_changed_physical_state",
            "censored": False,
            "error": None if name != "GateMate" or physical.get("exists") else MISSING_RECEIPT,
        }
        if name == "GateMate":
            row["fpga_sampling_claimed"] = False
            row["availability_class"] = (
                "future_bounded_task_eligible" if physical.get("exists") is True else "blocked"
            )
            row["metric"] = physical.get("exists") is True
        row["row_sha256"] = canonical_hash(row)
        rows.append(row)
    return rows


def device_reconciliation() -> list[JsonDict]:
    """Separate owned history, blocked access, and dated vendor context."""

    return [
        {
            "device": "KV260",
            "class": "owned_graduated_historical",
            "access_assumed": False,
            "future_access": "ssh_only",
            "architecture_limit": "k_max<=5",
            "measured_carnot_gain": None,
        },
        {
            "device": "PolarFire",
            "class": "owned_cpu_dispatch_historical",
            "access_assumed": False,
            "fpga_sampling_observed": False,
            "measured_carnot_gain": None,
        },
        {
            "device": "GateMate",
            "class": "owned_externally_blocked",
            "access_assumed": False,
            "measured_carnot_gain": None,
        },
        {
            "device": "Extropic Z1T",
            "class": "vendor_announcement",
            "vendor_source_date": "20260904",
            "v647_evidence_date": RUN_DATE,
            "source_path": "research-references.md",
            "access_assumed": False,
            "owned_by_carnot": False,
            "measured_carnot_gain": None,
        },
        {
            "device": "TSU",
            "class": "unavailable_not_assumed",
            "access_assumed": False,
            "measured_carnot_gain": None,
        },
        {
            "device": "NPU",
            "class": "unavailable_not_assumed",
            "access_assumed": False,
            "measured_carnot_gain": None,
        },
    ]


def _write_historical_sidecar(path: Path, board_hash: str, board: Mapping[str, Any]) -> JsonDict:
    """Keep old receipt references separate from this task's zero model calls."""

    value: JsonDict = {
        "schema": "carnot.experiment_7379.historical_model_receipts.v1",
        "label": "historical_hash_bound_no_current_model_work",
        "authorizes_current_inference": False,
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(ZERO_CURRENT_COUNTS),
        "source": {
            "path": BOARD_SOURCE_PATH.as_posix(),
            "sha256": board_hash,
            "historical_receipts": deepcopy(
                (board.get("invocation_counts") or {}).get("historical") or {}
            ),
        },
    }
    atomic_json(path, value)
    return value


def _span(name: str, start: float, end: float, origin: float, units: int) -> JsonDict:
    """Record one measured monotonic phase interval without duration padding."""

    return {
        "phase": name,
        "started_monotonic_offset_s": max(0.0, start - origin),
        "ended_monotonic_offset_s": max(0.0, end - origin),
        "duration_s": max(0.0, end - start),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every failed gate and expose the first exact terminal blocker."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = deepcopy(failures[0]) if failures else None
    if first is not None:
        first.pop("passed", None)
        first.pop("category", None)
    return {
        "passed": not failures,
        "check_count": len(checks),
        "failed_count": len(failures),
        "checks": [dict(row) for row in checks],
        "failures": failures,
        "first_failure": first,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary artifact fields without wrapping their values."""

    specific = {
        "schema": "Version the schema while keeping experiment_id and milestone ordinary top-level fields.",
        "status": "Use a terminal state only after real work and required validation.",
        "run_date": "Use 20260918 together with actual UTC start and completion timestamps.",
        "preconditions_checked": "Record exact paths, identities, hashes, classes, and resources before reduction.",
        "MODEL_SPECS": "List intended current models; host-only aggregation intends none.",
        "model_invoked": "Set true for any attempted current load or generation; this task attempted none.",
        "invocation_counts": "Separate zero current calls from labeled hash-bound historical receipts.",
        "inference_substrate": "Name actual host aggregation and retain historical provenance in a sidecar.",
        "inference_substrate_class": "Use the closed aggregation class that matches actual work.",
        "execution_venue": "Record host CPU work and no current board execution.",
        "duration_s": "Use measured monotonic elapsed time without sleep or invented duration.",
        "phase_spans": "Retain measured read, build, load, generate, evaluate, validate, and write boundaries.",
        "random_seed": "Use null because deterministic evidence accounting does not sample.",
        "reproducibility_checksum": "Bind exact code, sources, formula, protocol, raw rows, gates, and validation logs.",
        "source_artifact_hashes": "Bind exact producer paths and bytes while preserving original classes.",
        "rows": "Retain every board and placement unit, failure, metric, cost, and censoring disposition.",
        "sample_size_budget": "Declare planned, attempted, completed, censored units and the fixed stopping rule.",
        "acceptance_gate_results": "Separate expected, observed, and passed values for each gate class.",
        "gate_check_summary": "Name every blocked upstream, field, expectation, and observation.",
        "verifier_is_oracle": "Disclose that the artifact reducer defines record truth, not hardware efficacy.",
        "honest_verdict": "Name external absence precisely and never disguise it as success.",
        "verdict_class": "Use the closed class; unchanged external absence is blocked, not partial.",
        "flagged_adversarial": "Flag a current critical independent finding and exclude it from readiness.",
        "validation_receipts": "Retain argv, environment, scope, exit, elapsed time, and exact log hash.",
        "repository_health": "Keep dated unrelated failures distinct from affected checks.",
        "field_principles": "Explain fields directly without wrapping dictionaries or numeric values.",
        "promotion_score": "Remain zero because this milestone authorizes no rollout or publication.",
        "board_disposition_complete_score": "Equal one for three authenticated dispositions, including an external block.",
        "board_rows": "Keep board, authenticated venue, date, hash, terminal state, and exact prerequisite.",
        "changed_state_receipt": "Store a qualifying operator change or null with an exact failed gate.",
        "placement_envelope_rows": "Use measured fractions and bounds only, or name an unavailable input exactly.",
        "hardware_ready_score": "Remain zero because no current board qualification or physical action occurs.",
        "hardware_value_score": "Remain zero because projections and vendor claims are not measured gains.",
    }
    return {
        key: specific.get(key, f"Retain the ordinary {key} value with its evidence scope.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable protocol, source, row, gate, and validation evidence."""

    receipts = [
        {
            key: row.get(key)
            for key in (
                "name",
                "command_argv",
                "command_environment",
                "scope",
                "exit_code",
                "log_sha256",
                "passed",
            )
        }
        for row in artifact.get("validation_receipts") or []
        if isinstance(row, Mapping)
    ]
    stable = {
        "schema": artifact.get("schema"),
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "milestone": artifact.get("milestone"),
        "run_date": artifact.get("run_date"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "rows": artifact.get("rows"),
        "native_cost_boundary": artifact.get("native_cost_boundary"),
        "amdahl_protocol": artifact.get("amdahl_protocol"),
        "gate_check_summary": artifact.get("gate_check_summary"),
        "honest_verdict": artifact.get("honest_verdict"),
        "verdict_class": artifact.get("verdict_class"),
        "scores": {
            key: artifact.get(key)
            for key in (
                "board_disposition_complete_score",
                "hardware_ready_score",
                "hardware_value_score",
                "promotion_score",
            )
        },
        "validation_receipts": receipts,
    }
    return canonical_hash(stable)


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    validation: Mapping[str, Any],
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Build one terminal record from authenticated board and placement inputs."""

    origin = time.monotonic()
    started_at = utc_now()
    read_started = time.monotonic()
    checks, hashes, context = collect_preconditions(root, paths)
    board = _load_object(root / BOARD_SOURCE_PATH) or {}
    cost = _load_object(root / COST_SOURCE_PATH) or {}
    placement_rows = load_placement_envelope(root)
    physical = board_history.search_changed_state_receipt(
        root,
        paths.changed_state_search,
        candidate_paths=candidate_paths,
    )
    read_ended = time.monotonic()

    build_started = time.monotonic()
    historical = _write_historical_sidecar(
        paths.historical_models,
        str(hashes[BOARD_SOURCE_PATH.as_posix()]),
        board,
    )
    board_rows = build_board_rows(board, physical)
    devices = device_reconciliation()
    build_ended = time.monotonic()

    load_started = time.monotonic()
    load_ended = time.monotonic()
    generate_started = time.monotonic()
    generate_ended = time.monotonic()

    evaluate_started = time.monotonic()
    validation_ok = validation.get("required_checks_passed") is True
    required_source_ok = all(
        row.get("passed") is True for row in checks if row.get("category") == "required_source"
    )
    board_complete = len(board_rows) == 3 and {row["board"] for row in board_rows} == {
        "KV260",
        "GateMate",
        "PolarFire",
    }
    placement_available = bool(placement_rows) and all(
        row.get("performance_evidence_eligible") is True for row in placement_rows
    )
    changed_receipt = deepcopy(physical) if physical.get("exists") is True else None
    if changed_receipt is not None:
        timestamp = changed_receipt.get("receipt_timestamp")
        changed_receipt["receipt_date"] = (
            str(timestamp)[:10].replace("-", "")
            if timestamp
            else changed_receipt.get("date_evidence")
        )
    physical_gate = _check(
        "gatemate_changed_physical_state_receipt",
        "external_prerequisite",
        "physical_state_receipt",
        "receipt_date/operator_authored/provenance/changed_field",
        PHYSICAL_RECEIPT_CONTRACT,
        (
            changed_receipt
            if changed_receipt is not None
            else {
                "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                "selected_source_path": physical.get("source_path"),
                "absence": MISSING_RECEIPT,
            }
        ),
        changed_receipt is not None,
    )
    placement_checks = [
        _check(
            f"placement_source:{row['source_experiment']}",
            "placement_input",
            str(row["source_path"]),
            str(row["failed_field"] or "performance_evidence_eligible"),
            row["expected_value"] if row["failed_field"] else True,
            row["observed_value"] if row["failed_field"] else True,
            row.get("performance_evidence_eligible") is True,
        )
        for row in placement_rows
    ]
    validation_gate = _check(
        "affected_required_validation",
        "required_validation",
        "Exp7303 scoped runner through Exp7358 command plan",
        "required_checks_passed",
        True,
        validation.get("required_checks_passed"),
        validation_ok,
    )
    gate_checks = [*checks, physical_gate, *placement_checks, validation_gate]

    if not validation_ok:
        status = "complete"
        verdict_class = "disqualified"
        honest = (
            "complete_disqualified: affected validation failed; retained diagnostic rows "
            "authorize no readiness, hardware value, or promotion"
        )
    elif not required_source_ok or not board_complete:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_source_precondition: required board, cost, receipt, or capability "
            "evidence failed authentication"
        )
    elif changed_receipt is None:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_changed_physical_state: no operator-authored dated GateMate cable, "
            "port, board, power, JTAG, or DirtyJTAG change exists after Exp6559; three "
            "board dispositions are complete and unavailable placement inputs create no speed claim"
        )
    elif not placement_available:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_placement_input_unavailable: no eligible measured complete-query "
            "placement rows exist; board accounting is complete and CPU retention remains"
        )
    else:
        status = "complete"
        verdict_class = "null"
        honest = (
            "complete_null: measured placement rows do not authorize current hardware "
            "readiness, value, promotion, or automatic deployment"
        )

    native_gate = (cost.get("acceptance_gate_results") or {}).get("native_ten_x") or {}
    native_boundary = {
        "source_path": COST_SOURCE_PATH.as_posix(),
        "source_verdict_class": cost.get("verdict_class"),
        "measurement_scope": "complete Python-to-result request boundary",
        "native_ten_x_score": cost.get("native_ten_x_score"),
        "complete_boundary_gate_expected": native_gate.get("expected"),
        "complete_boundary_gate_observed": native_gate.get("observed"),
        "complete_boundary_gate_passed": native_gate.get("passed"),
        "erased_by_inner_loop_or_vendor_claim": False,
        "new_native_benchmark_run": False,
        "rust_port_authorized": False,
    }
    evaluate_ended = time.monotonic()

    validate_started = time.monotonic()
    score = int(required_source_ok and board_complete and validation_ok)
    validation_receipts = deepcopy(validation.get("validation_receipts") or [])
    validate_ended = time.monotonic()

    write_started = time.monotonic()
    raw: JsonDict = {
        "schema": "carnot.experiment_7379.raw_hardware_envelope.v1",
        "source_hashes": deepcopy(hashes),
        "board_rows": board_rows,
        "placement_envelope_rows": placement_rows,
        "native_cost_boundary": native_boundary,
        "physical_state": deepcopy(physical),
    }
    atomic_json(paths.raw_evidence, raw)
    hashes[str(paths.raw_evidence)] = sha256_file(paths.raw_evidence)
    hashes[str(paths.changed_state_search)] = sha256_file(paths.changed_state_search)
    hashes[str(paths.historical_models)] = sha256_file(paths.historical_models)
    write_ended = time.monotonic()

    all_rows = [*board_rows, *placement_rows]
    acceptance = {
        "source_authentication": {
            "expected": "exact Exp7367 rows/receipts and complete Exp7340 null",
            "observed": required_source_ok,
            "passed": required_source_ok,
        },
        "affected_validation": {
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
        },
        "board_accounting": {
            "expected": ["KV260", "GateMate", "PolarFire"],
            "observed": [row["board"] for row in board_rows],
            "passed": board_complete,
        },
        "changed_physical_state": {
            "expected": PHYSICAL_RECEIPT_CONTRACT,
            "observed": changed_receipt,
            "passed": changed_receipt is not None,
        },
        "placement_measurement": {
            "expected": "eligible complete-query measured rows",
            "observed": [row["outcome"] for row in placement_rows],
            "passed": placement_available,
        },
        "native_cost_null_preserved": {
            "expected": {"native_ten_x_score": 0, "gate_passed": False},
            "observed": {
                "native_ten_x_score": native_boundary["native_ten_x_score"],
                "gate_passed": native_boundary["complete_boundary_gate_passed"],
            },
            "passed": native_boundary["native_ten_x_score"] == 0
            and native_boundary["complete_boundary_gate_passed"] is False,
        },
        "hardware_safety": {
            "expected": "zero model, board, download, install, purchase, and vendor operations",
            "observed": 0,
            "passed": True,
        },
        "hardware_value": {
            "expected": "current authenticated measured hardware gain",
            "observed": None,
            "passed": False,
        },
        "promotion": {"expected": 0, "observed": 0, "passed": True},
    }
    phase_spans = [
        _span("read", read_started, read_ended, origin, len(checks)),
        _span("build", build_started, build_ended, origin, len(board_rows)),
        _span("load", load_started, load_ended, origin, 0),
        _span("generate", generate_started, generate_ended, origin, 0),
        _span("evaluate", evaluate_started, evaluate_ended, origin, len(all_rows)),
        _span("validate", validate_started, validate_ended, origin, len(validation_receipts)),
        _span("write", write_started, write_ended, origin, 3),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_COUNTS),
            "historical": {
                "sidecar_count": 1,
                "sidecars": [
                    {
                        "path": str(paths.historical_models),
                        "sha256": hashes[str(paths.historical_models)],
                        "label": historical["label"],
                    }
                ],
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "operation": "CPython JSON parsing, SHA-256 hashing, and Amdahl arithmetic",
            "hostname": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "host_cpu",
        },
        "duration_s": max(write_ended - origin, 0.000001),
        "phase_spans": phase_spans,
        "random_seed": None,
        "source_artifact_hashes": hashes,
        "source_artifact_states": context,
        "rows": all_rows,
        "board_rows": board_rows,
        "placement_envelope_rows": placement_rows,
        "sample_size_budget": {
            "board_rows_planned": 3,
            "board_rows_attempted": len(board_rows),
            "board_rows_completed": len(board_rows),
            "board_rows_censored": 0,
            "placement_sources_planned": 2,
            "placement_sources_attempted": 2,
            "placement_sources_eligible": sum(
                row["performance_evidence_eligible"] is True for row in placement_rows
            ),
            "placement_sources_unavailable": sum(
                row["performance_evidence_eligible"] is False for row in placement_rows
            ),
            "placement_rows_completed": sum(
                row["performance_evidence_eligible"] is True for row in placement_rows
            ),
            "placement_rows_censored": sum(
                row["performance_evidence_eligible"] is False for row in placement_rows
            ),
            "new_hardware_runs_planned": 0,
            "new_hardware_runs_attempted": 0,
            "new_hardware_runs_completed": 0,
            "new_hardware_runs_censored": 0,
            "remaining_work": 0,
            "stopping_rule": (
                "inspect Exp7374 and Exp7378 once; reduce eligible measured rows only; "
                "account for exactly KV260, GateMate, and PolarFire; stop before hardware work"
            ),
        },
        "acceptance_gate_results": acceptance,
        "gate_check_summary": _gate_summary(gate_checks),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": validation_receipts,
        "required_checks_passed": validation_ok,
        "missing_required_commands": deepcopy(validation.get("missing_required_commands") or []),
        "failed_required_commands": deepcopy(validation.get("failed_required_commands") or []),
        "duplicate_required_commands": deepcopy(
            validation.get("duplicate_required_commands") or []
        ),
        "repository_health": deepcopy(
            board.get("repository_health") or validation.get("repository_health") or {}
        ),
        "board_disposition_complete_score": score,
        "changed_state_receipt": changed_receipt,
        "native_cost_boundary": native_boundary,
        "amdahl_protocol": {
            "formula": "1 / ((1 - replaceable_fraction) + replaceable_fraction / kernel_rate)",
            "hypothetical_kernel_rate": HYPOTHETICAL_KERNEL_RATE,
            "infinite_kernel_limit": "1 / unaccelerated_fraction",
            "hundred_x_unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "necessary_not_sufficient": True,
            "requires_measured_complete_service_fraction": True,
        },
        "placement_recommendation": "retain_cpu",
        "device_reconciliation": devices,
        "prior_failure_recurrence": {
            "source_experiment": "exp7367-board-disposition",
            "prior_honest_verdict": board.get("honest_verdict"),
            "same_external_block": changed_receipt is None,
            "retire_if_same_verdict": True,
            "action": "preserve block and do not repeat physical diagnostics",
        },
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "promotion_score": 0,
        "hardware_operations": {
            "usb": 0,
            "ssh": 0,
            "jtag": 0,
            "flash": 0,
            "reset": 0,
            "fpga": 0,
            "device_access": 0,
            "downloads": 0,
            "driver_installations": 0,
            "purchases": 0,
            "vendor_contacts": 0,
            "native_benchmarks": 0,
            "rust_ports": 0,
        },
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _rows_valid(rows: Sequence[Any]) -> bool:
    """Check that every retained row is hash-bound to its complete content."""

    return all(
        isinstance(row, Mapping)
        and row.get("row_sha256")
        == canonical_hash({key: item for key, item in row.items() if key != "row_sha256"})
        for row in rows
    )


def validate_artifact(value: object) -> list[str]:
    """Reject forged conclusions, missing evidence, and success-shaped blocks."""

    artifact = value if isinstance(value, Mapping) else {}
    board_rows = artifact.get("board_rows") if isinstance(artifact.get("board_rows"), list) else []
    placement = (
        artifact.get("placement_envelope_rows")
        if isinstance(artifact.get("placement_envelope_rows"), list)
        else []
    )
    current = (artifact.get("invocation_counts") or {}).get("current")
    validation_ok = artifact.get("required_checks_passed") is True
    failures = {
        "required_fields": not REQUIRED_FIELDS.issubset(artifact),
        "identity": artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "model_declaration": artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or current != ZERO_CURRENT_COUNTS,
        "substrate": artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "board_rows": len(board_rows) != 3
        or {row.get("board") for row in board_rows if isinstance(row, Mapping)}
        != {"KV260", "GateMate", "PolarFire"}
        or not _rows_valid(board_rows),
        "placement_rows": len(placement) < 2 or not _rows_valid(placement),
        "rows": artifact.get("rows") != [*board_rows, *placement],
        "scores": artifact.get("hardware_ready_score") != 0
        or artifact.get("hardware_value_score") != 0
        or artifact.get("promotion_score") != 0
        or artifact.get("board_disposition_complete_score") not in (0, 1),
        "operations": any(
            count != 0 for count in (artifact.get("hardware_operations") or {}).values()
        ),
        "native_null": (artifact.get("native_cost_boundary") or {}).get("native_ten_x_score") != 0
        or (artifact.get("native_cost_boundary") or {}).get("complete_boundary_gate_passed")
        is not False,
        "verdict": artifact.get("verdict_class") not in command_boundary.CLOSED_VERDICTS
        or (not validation_ok and artifact.get("verdict_class") != "disqualified")
        or (
            validation_ok
            and artifact.get("changed_state_receipt") is None
            and artifact.get("verdict_class") != "blocked"
        ),
        "field_principles": not REQUIRED_FIELDS.issubset(artifact.get("field_principles") or {}),
        "checksum": artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
    }
    return [name for name, failed in failures.items() if failed]


def independent_reduce(artifact: Mapping[str, Any], raw: Mapping[str, Any]) -> JsonDict:
    """Recompute row and source conclusions without trusting headline scores."""

    raw_boards = raw.get("board_rows") if isinstance(raw.get("board_rows"), list) else []
    raw_placement = (
        raw.get("placement_envelope_rows")
        if isinstance(raw.get("placement_envelope_rows"), list)
        else []
    )
    source_paths = (
        BOARD_SOURCE_PATH.as_posix(),
        COST_SOURCE_PATH.as_posix(),
        MEMORY_SOURCE_PATH.as_posix(),
        ISING_SOURCE_PATH.as_posix(),
    )
    return {
        "candidate_board_rows_match": artifact.get("board_rows") == raw_boards,
        "candidate_placement_rows_match": artifact.get("placement_envelope_rows") == raw_placement,
        "candidate_rows_match": artifact.get("rows") == [*raw_boards, *raw_placement],
        "candidate_score_matches": artifact.get("board_disposition_complete_score")
        == int(
            len(raw_boards) == 3
            and {row.get("board") for row in raw_boards}
            == {
                "KV260",
                "GateMate",
                "PolarFire",
            }
            and artifact.get("required_checks_passed") is True
        ),
        "source_hashes_match": all(
            (artifact.get("source_artifact_hashes") or {}).get(path)
            == (raw.get("source_hashes") or {}).get(path)
            for path in source_paths
        ),
    }


def cold_validate_candidate(paths: ExperimentPaths) -> list[str]:
    """Reload the candidate and independently reduce its retained raw rows."""

    candidate = _load_object(paths.terminal_candidate) or {}
    raw = _load_object(paths.raw_evidence) or {}
    replay = independent_reduce(candidate, raw)
    errors = validate_artifact(candidate)
    for field, error in (
        ("candidate_board_rows_match", "candidate_board_rows_mismatch"),
        ("candidate_placement_rows_match", "candidate_placement_rows_mismatch"),
        ("candidate_rows_match", "candidate_rows_mismatch"),
        ("candidate_score_matches", "candidate_score_mismatch"),
        ("source_hashes_match", "candidate_source_hashes_mismatch"),
    ):
        if replay[field] is not True:
            errors.append(error)
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a record that passes local validation."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the actual Exp7358 plan for only this task's affected files."""

    return command_boundary.build_command_plan(root, V647_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, absent private parents, and any command drift."""

    return command_boundary.validate_command_plan(root, V647_MANIFEST, commands)


def terminal_commands(
    root: Path, paths: ExperimentPaths
) -> list[command_boundary.PlannedCommand]:  # pragma: no cover
    """Build cold replay and the two mandatory strict terminal readers."""

    python = str(root / ".venv/bin/python")
    replay = (
        "from pathlib import Path; "
        "from carnot.experiment_7379_v647_hardware_envelope import ExperimentPaths,cold_validate_candidate; "
        f"p=ExperimentPaths(Path({str(paths.artifact)!r}),Path({str(paths.raw_evidence)!r}),"
        f"Path({str(paths.changed_state_search)!r}),Path({str(paths.historical_models)!r}),"
        f"Path({str(paths.terminal_candidate)!r}),Path({str(paths.validation_dir)!r})); "
        "e=cold_validate_candidate(p); print({'errors':e}, flush=True); raise SystemExit(bool(e))"
    )
    specs = [
        ("independent_reducer", (python, "-u", "-c", replay), "independent_replay"),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(paths.terminal_candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(paths.terminal_candidate),
            ),
            "safety",
        ),
    ]
    return [
        command_boundary.PlannedCommand(
            validation_scope.CommandSpec(name, argv, scope), scope, True
        )
        for name, argv, scope in specs
    ]


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:  # pragma: no cover
    """Run scoped checks, cold replay, strict readers, and atomic publication."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7379-"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    progress(started, "preconditions", "after", commands=len(commands))

    planned = [
        command_boundary.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    progress(started, "validation", "before_subprocess_group", units=len(planned))
    receipts = command_boundary.run_categorized_commands(
        root,
        planned,
        log_dir=paths.validation_dir / "affected",
        heartbeat_s=60.0,
    )
    reduced = command_boundary.reduce_affected_receipts(root, V647_MANIFEST, receipts)
    validation = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "validation_receipts": receipts,
    }
    progress(started, "validation", "after_subprocess_group", passed=reduced["passed"])

    progress(started, "evaluation", "before")
    artifact = build_artifact(root, paths, validation)
    atomic_json(paths.terminal_candidate, artifact)
    progress(
        started,
        "evaluation",
        "after",
        boards=len(artifact["board_rows"]),
        placement_rows=len(artifact["placement_envelope_rows"]),
    )

    terminal = terminal_commands(root, paths)
    progress(started, "terminal_validation", "before_subprocess_group", units=len(terminal))
    terminal_receipts = command_boundary.run_categorized_commands(
        root,
        terminal,
        log_dir=paths.validation_dir / "terminal",
        heartbeat_s=60.0,
    )
    progress(started, "terminal_validation", "after_subprocess_group")
    terminal_failed = [row["name"] for row in terminal_receipts if row.get("passed") is not True]
    artifact["validation_receipts"].extend(terminal_receipts)
    artifact["acceptance_gate_results"]["terminal_validators"] = {
        "expected": [row.spec.name for row in terminal],
        "observed": {
            "passed": [row["name"] for row in terminal_receipts if row.get("passed") is True],
            "failed": terminal_failed,
        },
        "passed": not terminal_failed,
    }
    terminal_gate = _check(
        "terminal_validation",
        "required_validation",
        "cold replay and strict terminal readers",
        "all terminal commands pass",
        True,
        not terminal_failed,
        not terminal_failed,
    )
    artifact["gate_check_summary"] = _gate_summary(
        [*artifact["gate_check_summary"]["checks"], terminal_gate]
    )
    artifact["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and row.get("passed") is not True
        for row in terminal_receipts
    )
    if terminal_failed:
        artifact["status"] = "complete"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified: terminal replay or strict validation failed; all "
            "readiness, value, and promotion scores remain zero"
        )
        artifact["board_disposition_complete_score"] = 0
        artifact["required_checks_passed"] = False
        artifact["failed_required_commands"] = sorted(
            set([*artifact["failed_required_commands"], *terminal_failed])
        )
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)

    progress(started, "write", "before_atomic_terminal", path=paths.artifact)
    write_artifact(paths.artifact, artifact)
    progress(started, "write", "after_atomic_terminal", path=paths.artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover
    """Parse the frozen execution date without adding launcher behavior."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Execute the frozen V647 task from its declared thin entrypoint."""

    print("[exp7379] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run date must be {RUN_DATE}")
    artifact = run_experiment(REPO_ROOT, ExperimentPaths.defaults(REPO_ROOT))
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
