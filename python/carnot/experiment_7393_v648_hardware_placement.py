"""Build the V648 decision-head placement boundary from retained evidence.

This module reads existing JSON and numeric checkpoints on the host. It does
not contact a model or a board. Missing stage costs stay missing, so a small
kernel measurement cannot stand in for the complete service boundary.

Spec refs: REQ-REPORT-7393 and SCENARIO-REPORT-7393-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot import experiment_7379_v647_hardware_envelope as prior
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7393
TASK_ID = "exp7393-hardware-placement"
MILESTONE = "2026.09.648"
RUN_DATE = "20260918"
SCHEMA = "carnot.experiment_7393.v648_hardware_placement.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7393_v648_hardware_placement.json")
RAW_DIR = Path("results/raw/experiment_7393_v648_hardware_placement")
HARDWARE_SOURCE_PATH = Path("results/experiment_7379_v647_hardware_envelope.json")
NATIVE_COST_SOURCE_PATH = Path("results/experiment_7340_v644_native_cost.json")
TRAINING_SOURCE_PATH = Path("results/experiment_7385_v648_decision_training.json")
ONLINE_SOURCE_PATH = Path("results/experiment_7386_v648_online_decisions.json")
PROOF_SOURCE_PATH = Path("results/experiment_7389_v648_proof_learning.json")
VALIDATION_BOUNDARY_PATH = Path("results/experiment_7358_v646_validation_contract.json")
MODULE_PATH = Path("python/carnot/experiment_7393_v648_hardware_placement.py")
TEST_PATH = Path("tests/python/test_experiment_7393_v648_hardware_placement.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7393_v648_hardware_placement.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

ASSUMED_DEVICE_RATE = 100.0
HUNDRED_X_UNACCELERATED_FRACTION_MAX = 0.01
STAGE_NAMES = (
    "prediction",
    "update",
    "certificate",
    "serialization",
    "orchestration",
)
REPLACEABLE_STAGES = ("prediction", "update", "certificate", "serialization")
EMPTY_STAGE_COSTS = {name: None for name in STAGE_NAMES}
VALID_SCIENCE_CLASSES = {"positive", "circular_positive", "null"}
ZERO_CURRENT_COUNTS = deepcopy(prior.ZERO_CURRENT_COUNTS)
PHYSICAL_RECEIPT_CONTRACT = deepcopy(prior.PHYSICAL_RECEIPT_CONTRACT)
MISSING_RECEIPT = prior.MISSING_RECEIPT

sha256_file = prior.sha256_file
canonical_hash = prior.canonical_hash
atomic_json = prior.atomic_json
utc_now = prior.utc_now

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
    Path("ops/known-issues.md"),
    SPEC_PATH,
    HARDWARE_SOURCE_PATH,
    NATIVE_COST_SOURCE_PATH,
    TRAINING_SOURCE_PATH,
    ONLINE_SOURCE_PATH,
    VALIDATION_BOUNDARY_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7379_v647_hardware_envelope.py"),
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

V648_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=TASK_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, validation logs, and the terminal result separate."""

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
        """Resolve isolated output paths below a repository or test directory."""

        raw = root / RAW_DIR
        return cls(
            artifact=root / RESULT_PATH,
            raw_evidence=raw / "hardware_placement_evidence.json",
            changed_state_search=raw / "gatemate_changed_state_search.json",
            historical_models=raw / "historical_model_inputs.json",
            terminal_candidate=raw / "terminal_candidate.json",
            validation_dir=raw / "validation",
        )


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each boundary so a bounded host task stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7393] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict | None:
    """Return one JSON object, or ``None`` for missing or malformed bytes."""

    return prior._load_object(path)


def _check(
    check: str,
    category: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record one exact expectation without hiding a failed prerequisite."""

    return prior._check(check, category, upstream, field, expected, observed, passed)


def _required_validation_passed(artifact: Mapping[str, Any]) -> bool:
    """Require at least one passing required-validation gate and no failure."""

    gates = [
        row
        for row in artifact.get("acceptance_gate_results") or []
        if isinstance(row, Mapping) and row.get("category") == "required_validation"
    ]
    return bool(gates) and all(row.get("passed") is True for row in gates)


def _authenticate_hardware_receipts(root: Path, hardware: Mapping[str, Any]) -> dict[str, bool]:
    """Verify Exp7379 row hashes and the exact receipt bytes each row cites."""

    authenticated: dict[str, bool] = {}
    for source in hardware.get("board_rows") or []:
        if not isinstance(source, Mapping) or not isinstance(source.get("board"), str):
            continue
        bare = {key: value for key, value in source.items() if key != "row_sha256"}
        row_ok = source.get("row_sha256") == canonical_hash(bare)
        receipt = prior._resolved_source_path(root, source.get("last_authenticated_path"))
        receipt_ok = receipt.is_file() and sha256_file(receipt) == source.get(
            "last_authenticated_hash"
        )
        authenticated[str(source["board"])] = row_ok and receipt_ok
    return authenticated


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate exact sources before reducing placement or board claims."""

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

    proof_path = root / PROOF_SOURCE_PATH
    hashes[PROOF_SOURCE_PATH.as_posix()] = sha256_file(proof_path) if proof_path.is_file() else None
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    spec_ok = "REQ-REPORT-7393" in spec_text
    checks.append(
        _check(
            "driving_capability",
            "required_source",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7393",
            "REQ-REPORT-7393" if spec_ok else None,
            spec_ok,
        )
    )

    hardware = _load_object(root / HARDWARE_SOURCE_PATH) or {}
    receipt_auth = _authenticate_hardware_receipts(root, hardware)
    hardware_ok = (
        hardware.get("experiment_id") == 7379
        and hardware.get("milestone") == "2026.09.647"
        and hardware.get("verdict_class") == "blocked"
        and hardware.get("flagged_adversarial") is False
        and hardware.get("board_disposition_complete_score") == 1
        and hardware.get("schema") == "carnot.experiment_7379.v647_hardware_envelope.v1"
        and hardware.get("run_date") == "20260917"
        and hardware.get("reproducibility_checksum") == prior.reproducibility_checksum(hardware)
        and set(receipt_auth) == {"KV260", "GateMate", "PolarFire"}
        and all(receipt_auth.values())
    )
    checks.append(
        _check(
            "authenticated_exp7379_hardware_envelope",
            "required_source",
            HARDWARE_SOURCE_PATH.as_posix(),
            "identity/class/board_rows/latest_receipts",
            [7379, "blocked", 1, True],
            [
                hardware.get("experiment_id"),
                hardware.get("verdict_class"),
                hardware.get("board_disposition_complete_score"),
                bool(receipt_auth) and all(receipt_auth.values()),
            ],
            hardware_ok,
        )
    )

    boundary = _load_object(root / VALIDATION_BOUNDARY_PATH) or {}
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

    stage_sources: dict[str, JsonDict] = {}
    for name, path in (
        ("Exp7385", TRAINING_SOURCE_PATH),
        ("Exp7386", ONLINE_SOURCE_PATH),
        ("Exp7389", PROOF_SOURCE_PATH),
    ):
        artifact = _load_object(root / path)
        stage_sources[name] = {
            "path": path.as_posix(),
            "available": artifact is not None,
            "experiment_id": artifact.get("experiment_id") if artifact else None,
            "status": artifact.get("status") if artifact else None,
            "verdict_class": artifact.get("verdict_class") if artifact else None,
            "flagged_adversarial": artifact.get("flagged_adversarial") if artifact else None,
        }

    context: JsonDict = {
        "hardware_source": {
            "experiment_id": hardware.get("experiment_id"),
            "verdict_class": hardware.get("verdict_class"),
            "flagged_adversarial": hardware.get("flagged_adversarial"),
            "board_disposition_complete_score": hardware.get("board_disposition_complete_score"),
            "authenticated_board_receipts": dict(sorted(receipt_auth.items())),
        },
        "stage_sources": stage_sources,
        "native_cost_source": {
            "path": NATIVE_COST_SOURCE_PATH.as_posix(),
            "historical_diagnostic_only": True,
            "used_as_current_complete_boundary": False,
        },
    }
    return checks, hashes, context


def reduce_complete_service(
    stage_costs_s: Mapping[str, Any], *, assumed_device_rate: float = ASSUMED_DEVICE_RATE
) -> JsonDict:
    """Compute fractions only from all five measured service components."""

    if set(stage_costs_s) != set(STAGE_NAMES):
        raise ValueError("stage costs must contain exactly all five stage names")
    if not math.isfinite(assumed_device_rate) or assumed_device_rate <= 0:
        raise ValueError("assumed_device_rate must be positive finite")
    costs: dict[str, float] = {}
    for name in STAGE_NAMES:
        value = stage_costs_s[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError("stage costs must be nonnegative finite numbers")
        costs[name] = float(value)
    total = sum(costs.values())
    if total <= 0:
        raise ValueError("complete service time must be positive")
    fractions = {name: costs[name] / total for name in STAGE_NAMES}
    replaceable = sum(fractions[name] for name in REPLACEABLE_STAGES)
    bounds = prior.amdahl_bounds(replaceable, assumed_device_rate)
    unaccelerated = 1.0 - replaceable
    return {
        "complete_service_time_s": total,
        "stage_fractions": fractions,
        "replaceable_stages": list(REPLACEABLE_STAGES),
        "measured_replaceable_fraction": replaceable,
        "unaccelerated_fraction": unaccelerated,
        "assumed_device_rate": assumed_device_rate,
        "assumed_device_rate_is_measured": False,
        "bounded_full_service_speedup": bounds["bounded_full_service_speedup"],
        "infinite_device_upper_bound": bounds["infinite_kernel_upper_bound"],
        "hundred_x_necessary_condition": bounds["hundred_x_necessary_condition"],
        "orchestration_dominates": fractions["orchestration"] >= 0.5,
        "recommendation": "retain_cpu_or_batch_before_port",
    }


def _unavailable_stage_row(
    source_name: str,
    path: Path,
    source_class: str,
    failed_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Name one unusable producer without manufacturing stage measurements."""

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
        "source_evidence_eligible": False,
        "complete_boundary_eligible": False,
        "diagnostic_measurements_used_for_readiness": False,
        "stage_costs_s": deepcopy(EMPTY_STAGE_COSTS),
        "complete_service_time_s": None,
        "stage_fractions": deepcopy(EMPTY_STAGE_COSTS),
        "measured_replaceable_fraction": None,
        "unaccelerated_fraction": None,
        "assumed_device_rate": ASSUMED_DEVICE_RATE,
        "assumed_device_rate_is_measured": False,
        "bounded_full_service_speedup": None,
        "infinite_device_upper_bound": None,
        "hundred_x_necessary_condition": {
            "target_total_speedup": 100.0,
            "unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "observed_unaccelerated_fraction": None,
            "passed": False,
            "condition_scope": "not_evaluable_without_complete_measured_service",
        },
        "recommendation": "retain_cpu_or_batch_before_port",
        "metric": None,
        "censored": True,
        "error": f"{failed_field}: expected {expected!r}, observed {observed!r}",
    }
    row["row_sha256"] = canonical_hash(row)
    return row


def _measured_prediction_cost(artifact: Mapping[str, Any]) -> float | None:
    """Return the median authentic Exp7385 per-row prediction time."""

    values: list[float] = []
    for source in artifact.get("rows") or []:
        if not isinstance(source, Mapping):
            continue
        measured = source.get("measured_cost")
        value = measured.get("cpu_scoring_duration_s") if isinstance(measured, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            if value >= 0:
                values.append(float(value))
    return statistics.median(values) if values else None


def assess_stage_source(
    source_name: str,
    path: Path,
    expected_identity: Any,
    artifact: Mapping[str, Any] | None,
    *,
    capture_field: str,
) -> JsonDict:
    """Reduce one stage producer only after its class and gates are eligible."""

    if artifact is None:
        return _unavailable_stage_row(
            source_name, path, "missing", "path", "readable JSON object", "missing"
        )
    source_class = str(artifact.get("verdict_class") or "unknown")
    eligibility = (
        ("experiment_id", expected_identity, artifact.get("experiment_id")),
        ("status", "terminal complete status", artifact.get("status")),
        ("verdict_class", "positive|circular_positive|null", artifact.get("verdict_class")),
        ("flagged_adversarial", False, artifact.get("flagged_adversarial")),
        (capture_field, 1, artifact.get(capture_field)),
        ("required_validation", True, _required_validation_passed(artifact)),
    )
    for field, expected, observed in eligibility:
        if field == "status":
            passed = isinstance(observed, str) and observed.startswith("complete")
        elif field == "verdict_class":
            passed = observed in VALID_SCIENCE_CLASSES
        else:
            passed = observed == expected
        if not passed:
            return _unavailable_stage_row(
                source_name, path, source_class, field, expected, observed
            )

    direct = artifact.get("placement_stage_costs_s")
    if isinstance(direct, Mapping):  # pragma: no cover - future eligible producer shape.
        try:
            complete = reduce_complete_service(direct)
        except ValueError as error:
            return _unavailable_stage_row(
                source_name,
                path,
                source_class,
                "placement_stage_costs_s",
                list(STAGE_NAMES),
                str(error),
            )
        row = {
            "unit_id": f"placement:{source_name}",
            "row_type": "placement_envelope",
            "source_experiment": source_name,
            "source_path": path.as_posix(),
            "source_class": source_class,
            "outcome": "complete_measured_service_boundary",
            "failed_field": None,
            "expected_value": None,
            "observed_value": None,
            "source_evidence_eligible": True,
            "complete_boundary_eligible": True,
            "diagnostic_measurements_used_for_readiness": False,
            "stage_costs_s": {name: float(direct[name]) for name in STAGE_NAMES},
            **complete,
            "metric": complete["bounded_full_service_speedup"],
            "censored": False,
            "error": None,
        }
        row["row_sha256"] = canonical_hash(row)
        return row

    prediction = _measured_prediction_cost(artifact)
    if prediction is None:
        return _unavailable_stage_row(
            source_name,
            path,
            source_class,
            "rows.measured_cost.cpu_scoring_duration_s",
            "nonempty nonnegative measured values",
            None,
        )
    costs = {**EMPTY_STAGE_COSTS, "prediction": prediction}
    row = {
        "unit_id": f"placement:{source_name}",
        "row_type": "placement_envelope",
        "source_experiment": source_name,
        "source_path": path.as_posix(),
        "source_class": source_class,
        "outcome": "measured_partial_stage_costs",
        "failed_field": "complete_service_time_s",
        "expected_value": "all prediction/update/certificate/serialization/orchestration costs",
        "observed_value": costs,
        "source_evidence_eligible": True,
        "complete_boundary_eligible": False,
        "diagnostic_measurements_used_for_readiness": False,
        "stage_costs_s": costs,
        "complete_service_time_s": None,
        "stage_fractions": deepcopy(EMPTY_STAGE_COSTS),
        "measured_replaceable_fraction": None,
        "unaccelerated_fraction": None,
        "assumed_device_rate": ASSUMED_DEVICE_RATE,
        "assumed_device_rate_is_measured": False,
        "bounded_full_service_speedup": None,
        "infinite_device_upper_bound": None,
        "hundred_x_necessary_condition": {
            "target_total_speedup": 100.0,
            "unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "observed_unaccelerated_fraction": None,
            "passed": False,
            "condition_scope": "not_evaluable_without_complete_measured_service",
        },
        "recommendation": "retain_cpu_or_batch_before_port",
        "metric": prediction,
        "censored": True,
        "error": "complete measured service denominator is unavailable",
    }
    row["row_sha256"] = canonical_hash(row)
    return row


def load_placement_envelope(root: Path) -> list[JsonDict]:
    """Inspect all three planned producers once and preserve their exact class."""

    sources = (
        (
            "Exp7385",
            TRAINING_SOURCE_PATH,
            "exp7385-decision-training",
            "decision_capture_complete_score",
        ),
        (
            "Exp7386",
            ONLINE_SOURCE_PATH,
            "exp7386-online-decisions",
            "online_capture_complete_score",
        ),
        (
            "Exp7389",
            PROOF_SOURCE_PATH,
            "exp7389-proof-learning",
            "proof_memory_capture_complete_score",
        ),
    )
    return [
        assess_stage_source(
            name,
            path,
            identity,
            _load_object(root / path),
            capture_field=capture,
        )
        for name, path, identity, capture in sources
    ]


def _numeric_scalars(value: Any) -> list[float]:
    """Flatten a checkpoint tree while rejecting nonnumeric payload values."""

    if isinstance(value, bool):
        raise ValueError("numeric checkpoint contains a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("numeric checkpoint contains a non-finite value")
        return [number]
    if isinstance(value, Mapping):
        result: list[float] = []
        for item in value.values():
            result.extend(_numeric_scalars(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_numeric_scalars(item))
        return result
    raise ValueError("numeric checkpoint contains a nonnumeric value")


def measure_numeric_checkpoint(path: Path, *, expected_hash: str, arm: str, seed: int) -> JsonDict:
    """Count actual numeric weights and bind the exact checkpoint bytes."""

    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError("numeric checkpoint hash mismatch")
    checkpoint = _load_object(path)
    if checkpoint is None or not isinstance(checkpoint.get("weights"), Mapping):
        raise ValueError("numeric checkpoint is missing weights")
    values = _numeric_scalars(checkpoint["weights"])
    return {
        "source_experiment": "Exp7385",
        "checkpoint_path": path.as_posix(),
        "checkpoint_sha256": expected_hash,
        "checkpoint_file_bytes": path.stat().st_size,
        "arm": arm,
        "seed": seed,
        "architecture": deepcopy(checkpoint.get("architecture")),
        "numeric_parameter_count": len(values),
        "assumed_device_numeric_format": "float32",
        "bytes_per_parameter": 4,
        "estimated_minimum_transfer_bytes": len(values) * 4,
        "transfer_is_measured": False,
        "estimate_scope": "minimum numeric weights only; excludes protocol and framing",
    }


def measure_selector_checkpoint(root: Path) -> JsonDict:
    """Select the sealed primary Exp7385 head and inspect its real checkpoint."""

    artifact = _load_object(root / TRAINING_SOURCE_PATH) or {}
    protocol = artifact.get("protocol_identity") or {}
    arm = str(protocol.get("primary_value_arm"))
    seeds = protocol.get("seeds") or []
    seed = int(seeds[0])
    matches = [
        row
        for row in artifact.get("checkpoint_manifest") or []
        if isinstance(row, Mapping) and row.get("arm") == arm and row.get("seed") == seed
    ]
    if len(matches) != 1:  # pragma: no cover - authenticated current producer has one row.
        raise ValueError("primary selector checkpoint is not unique")
    row = matches[0]
    relative = Path(str(row["path"]))
    measured = measure_numeric_checkpoint(
        root / relative,
        expected_hash=str(row["sha256"]),
        arm=arm,
        seed=seed,
    )
    measured["checkpoint_path"] = relative.as_posix()
    expected_architecture = {"input_dim": 2, "hidden_dims": [4], "output_dim": 1}
    if measured["numeric_parameter_count"] != 17:  # pragma: no cover - fail-closed mutation.
        raise ValueError("selector checkpoint does not contain 17 numeric parameters")
    if measured["architecture"] != expected_architecture:  # pragma: no cover
        raise ValueError("selector checkpoint architecture mismatch")
    return measured


def build_board_rows(hardware: Mapping[str, Any], physical: Mapping[str, Any]) -> list[JsonDict]:
    """Carry forward three authenticated rows without claiming present access."""

    rows: list[JsonDict] = []
    for source in hardware.get("board_rows") or []:
        if not isinstance(source, Mapping) or source.get("board") not in {
            "KV260",
            "GateMate",
            "PolarFire",
        }:
            continue
        board = str(source["board"])
        terminal = source.get("terminal_state")
        prerequisite = source.get("exact_next_prerequisite")
        if board == "GateMate":
            changed = physical.get("exists") is True
            terminal = (
                "future_bounded_task_eligible" if changed else "blocked_changed_physical_state"
            )
            prerequisite = (
                prior.board_history.board_source.GATEMATE_FUTURE_ACTION
                if changed
                else prior.board_history.board_source.GATEMATE_OPERATOR_ACTION
            )
        row: JsonDict = {
            "unit_id": f"board:{board}",
            "row_type": "board_disposition",
            "board": board,
            "last_authenticated_date": source.get("last_authenticated_date"),
            "last_authenticated_hash": source.get("last_authenticated_hash"),
            "last_authenticated_path": source.get("last_authenticated_path"),
            "last_authenticated_venue": source.get("last_authenticated_venue"),
            "terminal_state": terminal,
            "exact_next_prerequisite": prerequisite,
            "availability_class": (
                "future_bounded_task_eligible"
                if board == "GateMate" and physical.get("exists") is True
                else source.get("availability_class")
            ),
            "historical_evidence_only": True,
            "present_reachability_asserted": False,
            "new_hardware_execution_claimed": False,
            "fpga_sampling_claimed": source.get("fpga_sampling_claimed") is True,
            "hash_matched_cpu_dispatch": board == "PolarFire",
            "future_access": "ssh_only" if board == "KV260" else None,
            "architecture_limit": "k_max<=5" if board == "KV260" else None,
            "metric": board != "GateMate" or physical.get("exists") is True,
            "censored": False,
            "error": None if board != "GateMate" or physical.get("exists") else MISSING_RECEIPT,
        }
        if board == "GateMate":
            row["fpga_sampling_claimed"] = False
        row["row_sha256"] = canonical_hash(row)
        rows.append(row)
    return rows


def device_paths() -> list[JsonDict]:
    """Record bounded future paths without operating any current device."""

    return [
        {
            "device": "CPU",
            "role": "retain current 17-parameter selector and batch orchestration",
            "available_for_current_task": True,
            "operation_performed": False,
        },
        {
            "device": "GPU",
            "role": "future batched dense-head control after a complete cost boundary",
            "available_for_current_task": False,
            "operation_performed": False,
        },
        {
            "device": "NPU",
            "role": "future low-rate selector control after driver and complete-boundary evidence",
            "available_for_current_task": False,
            "driver_installed": False,
            "operation_performed": False,
        },
        {
            "device": "sparse proof-check FPGA",
            "role": "future sparse proof checking; not the PolarFire CPU dispatch",
            "available_for_current_task": False,
            "operation_performed": False,
        },
        {
            "device": "KV260",
            "role": "graduated historical fabric path",
            "available_for_current_task": False,
            "future_access": "ssh_only",
            "architecture_limit": "k_max<=5",
            "operation_performed": False,
        },
        {
            "device": "Extropic Z1T",
            "role": "vendor context only",
            "context_date": "20260904",
            "available_for_current_task": False,
            "owned_by_carnot": False,
            "vendor_contacted": False,
            "operation_performed": False,
        },
    ]


def _write_historical_sidecar(
    path: Path, hashes: Mapping[str, str | None], sources: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Keep producer model metadata labeled and separate from current zero calls."""

    rows = []
    for name, source_path in (
        ("Exp7379", HARDWARE_SOURCE_PATH),
        ("Exp7385", TRAINING_SOURCE_PATH),
        ("Exp7386", ONLINE_SOURCE_PATH),
        ("Exp7389", PROOF_SOURCE_PATH),
    ):
        source = sources.get(name) or {}
        rows.append(
            {
                "source_experiment": name,
                "path": source_path.as_posix(),
                "sha256": hashes.get(source_path.as_posix()),
                "available": bool(source),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
                "inference_substrate_class": source.get("inference_substrate_class"),
                "MODEL_SPECS": deepcopy(source.get("MODEL_SPECS")),
                "model_invoked": source.get("model_invoked"),
                "invocation_counts": deepcopy(source.get("invocation_counts")),
                "authorizes_current_inference": False,
                "authorizes_current_readiness": False,
            }
        )
    value = {
        "schema": "carnot.experiment_7393.historical_model_inputs.v1",
        "label": "historical_hash_bound_no_current_model_work",
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(ZERO_CURRENT_COUNTS),
        "sources": rows,
    }
    atomic_json(path, value)
    return value


def _span(name: str, start: float, end: float, origin: float, units: int) -> JsonDict:
    """Record one real monotonic phase without padding its duration."""

    return {
        "phase": name,
        "started_monotonic_offset_s": max(0.0, start - origin),
        "ended_monotonic_offset_s": max(0.0, end - origin),
        "duration_s": max(0.0, end - start),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed check and the first exact terminal blocker."""

    return prior._gate_summary(checks)


def _rows_valid(rows: Sequence[Any]) -> bool:
    """Require every retained row hash to bind its complete content."""

    return prior._rows_valid(rows)


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain output fields without wrapping ordinary values."""

    specific = {
        "schema": "Version the schema while keeping experiment_id and milestone at the top level.",
        "status": "Publish a terminal state only after real work and required validation.",
        "run_date": "Use 20260918 with actual UTC start and completion timestamps.",
        "preconditions_checked": "Record exact source paths, hashes, classes, gates, and receipt checks.",
        "MODEL_SPECS": "List current LLM models; this host aggregation uses none.",
        "model_invoked": "Set true for any attempted current LLM load or generation; none occurred.",
        "invocation_counts": "Keep zero current calls separate from labeled historical source metadata.",
        "inference_substrate": "Describe actual host JSON, checkpoint, hash, and arithmetic work.",
        "inference_substrate_class": "Use the closed aggregation class for this host reducer.",
        "execution_venue": "Use the closed host value and keep device detail in other fields.",
        "duration_s": "Use measured monotonic time without sleep or invented duration.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Use null because this deterministic reduction performs no resampling.",
        "reproducibility_checksum": "Bind code, sources, protocol, rows, gates, and validation receipts.",
        "source_artifact_hashes": "Bind exact producer bytes and keep missing inputs explicit.",
        "rows": "Retain every board and placement row with costs, failures, and censoring.",
        "sample_size_budget": "Declare planned, attempted, eligible, censored, and unstarted units.",
        "acceptance_gate_results": "Separate expected, observed, operator, and pass state for each gate.",
        "gate_check_summary": "Name each failed upstream, field, expected value, and observed value.",
        "verifier_is_oracle": "Disclose that the reducer defines record truth, not hardware efficacy.",
        "honest_verdict": "Name the unchanged external GateMate absence precisely.",
        "verdict_class": "Use blocked for unchanged external absence and disqualified for validation failure.",
        "flagged_adversarial": "Flag a critical current validator finding and exclude readiness.",
        "validation_receipts": "Retain executed argv, environment, scope, exit, duration, and log hash.",
        "repository_health": "Keep unrelated dated health separate from affected checks.",
        "field_principles": "Explain each output field without wrapping its ordinary value.",
        "promotion_score": "Remain zero because no rollout, weight change, or publication is authorized.",
        "board_disposition_complete_score": "Equal one for three authenticated board rows, including the block.",
        "board_rows": "Keep each board's date, hash, venue, state, and exact next prerequisite.",
        "changed_state_receipt": "Store a qualifying operator receipt or null with an exact failed check.",
        "placement_envelope_rows": "Use complete measured fractions or name each missing stage exactly.",
        "hardware_ready_score": "Remain zero because no current board operation or qualification occurs.",
        "hardware_value_score": "Remain zero because a projection is not measured hardware acceleration.",
    }
    return {
        key: specific.get(key, f"Retain the ordinary {key} value with its evidence scope.")
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable source, row, gate, score, and validation evidence."""

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
        "selector_checkpoint": artifact.get("selector_checkpoint"),
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
    """Build one terminal result from board history and current stage costs."""

    origin = time.monotonic()
    started_at = utc_now()
    read_started = time.monotonic()
    checks, hashes, context = collect_preconditions(root, paths)
    sources = {
        "Exp7379": _load_object(root / HARDWARE_SOURCE_PATH) or {},
        "Exp7385": _load_object(root / TRAINING_SOURCE_PATH) or {},
        "Exp7386": _load_object(root / ONLINE_SOURCE_PATH) or {},
        "Exp7389": _load_object(root / PROOF_SOURCE_PATH) or {},
    }
    hardware = sources["Exp7379"]
    placement_rows = load_placement_envelope(root)
    selector = measure_selector_checkpoint(root)
    physical = prior.board_history.search_changed_state_receipt(
        root,
        paths.changed_state_search,
        candidate_paths=candidate_paths,
    )
    read_ended = time.monotonic()

    build_started = time.monotonic()
    historical = _write_historical_sidecar(paths.historical_models, hashes, sources)
    board_rows = build_board_rows(hardware, physical)
    planned_devices = device_paths()
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
    complete_placement = any(
        row.get("complete_boundary_eligible") is True for row in placement_rows
    )
    changed_receipt = deepcopy(physical) if physical.get("exists") is True else None
    if changed_receipt is not None:  # pragma: no cover - no qualifying repository receipt exists.
        timestamp = changed_receipt.get("receipt_timestamp")
        changed_receipt["receipt_date"] = (
            str(timestamp)[:10].replace("-", "-").replace("-", "")
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
            str(row["failed_field"] or "complete_boundary_eligible"),
            row["expected_value"] if row["failed_field"] else True,
            row["observed_value"] if row["failed_field"] else True,
            row.get("complete_boundary_eligible") is True,
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
            "complete_disqualified: affected validation failed; diagnostic rows grant no "
            "hardware readiness, value, or promotion"
        )
    elif not required_source_ok or not board_complete:  # pragma: no cover - defensive mutation.
        status = "blocked"
        verdict_class = "blocked"
        honest = "blocked_source_precondition: required hardware evidence failed authentication"
    elif changed_receipt is None:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_changed_physical_state: no operator-authored dated GateMate cable, "
            "port, board, power, JTAG, or DirtyJTAG change exists after Exp6559; three "
            "board dispositions are complete and incomplete stage costs create no speed claim"
        )
    elif not complete_placement:  # pragma: no cover - physical receipt is currently absent.
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_placement_input_unavailable: no complete measured service boundary "
            "exists; retain CPU execution or batch before a port"
        )
    else:  # pragma: no cover - requires future complete stage and operator evidence.
        status = "complete"
        verdict_class = "null"
        honest = "complete_null: placement projection authorizes no hardware readiness or value"
    evaluate_ended = time.monotonic()

    validate_started = time.monotonic()
    board_score = int(required_source_ok and board_complete and validation_ok)
    validation_receipts = deepcopy(validation.get("validation_receipts") or [])
    validate_ended = time.monotonic()

    write_started = time.monotonic()
    raw: JsonDict = {
        "schema": "carnot.experiment_7393.raw_hardware_placement.v1",
        "source_hashes": deepcopy(hashes),
        "board_rows": board_rows,
        "placement_envelope_rows": placement_rows,
        "selector_checkpoint": selector,
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
            "expected": True,
            "observed": required_source_ok,
            "operator": "==",
            "passed": required_source_ok,
        },
        "affected_validation": {
            "expected": True,
            "observed": validation_ok,
            "operator": "==",
            "passed": validation_ok,
        },
        "board_accounting": {
            "expected": ["KV260", "GateMate", "PolarFire"],
            "observed": [row["board"] for row in board_rows],
            "operator": "set_equal",
            "passed": board_complete,
        },
        "changed_physical_state": {
            "expected": PHYSICAL_RECEIPT_CONTRACT,
            "observed": changed_receipt,
            "operator": "contract_match",
            "passed": changed_receipt is not None,
        },
        "complete_service_measurement": {
            "expected": "all five measured stage costs",
            "observed": [row["outcome"] for row in placement_rows],
            "operator": "any_complete_boundary",
            "passed": complete_placement,
        },
        "hundred_x_necessary_condition": {
            "expected": {"unaccelerated_fraction": "<=0.01"},
            "observed": None,
            "operator": "<=",
            "passed": False,
        },
        "hardware_safety": {
            "expected": "zero model, board, download, install, purchase, and vendor operations",
            "observed": 0,
            "operator": "==",
            "passed": True,
        },
        "hardware_value": {
            "expected": "measured current hardware acceleration",
            "observed": None,
            "operator": "present",
            "passed": False,
        },
        "promotion": {"expected": 0, "observed": 0, "operator": "==", "passed": True},
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
        "inference_substrate": "host_cpu_json_checkpoint_hash_and_amdahl_reduction",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "operation": "CPython JSON parsing, SHA-256 hashing, checkpoint counting, and Amdahl arithmetic",
            "hostname": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "host_cpu",
            "small_ebm_training": {
                "performed_in_current_task": False,
                "historical_sources_only": ["Exp7385", "Exp7386"],
            },
        },
        "duration_s": max(write_ended - origin, 0.000001),
        "phase_spans": phase_spans,
        "random_seed": None,
        "source_artifact_hashes": hashes,
        "source_artifact_states": context,
        "rows": all_rows,
        "board_rows": board_rows,
        "placement_envelope_rows": placement_rows,
        "selector_checkpoint": selector,
        "sample_size_budget": {
            "board_rows_planned": 3,
            "board_rows_attempted": len(board_rows),
            "board_rows_completed": len(board_rows),
            "board_rows_censored": 0,
            "placement_sources_planned": 3,
            "placement_sources_attempted": 3,
            "placement_sources_eligible": sum(
                row["source_evidence_eligible"] is True for row in placement_rows
            ),
            "placement_sources_unavailable": sum(
                row["source_evidence_eligible"] is False for row in placement_rows
            ),
            "complete_service_rows": sum(
                row["complete_boundary_eligible"] is True for row in placement_rows
            ),
            "placement_rows_censored": sum(row["censored"] is True for row in placement_rows),
            "new_hardware_runs_planned": 0,
            "new_hardware_runs_attempted": 0,
            "new_hardware_runs_completed": 0,
            "new_hardware_runs_censored": 0,
            "unstarted_units": 0,
            "remaining_work": 0,
            "stopping_rule": (
                "inspect Exp7385, Exp7386, and Exp7389 once; authenticate three board rows; "
                "stop before any model, driver, network, board, purchase, or vendor operation"
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
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "board_disposition_complete_score": board_score,
        "changed_state_receipt": changed_receipt,
        "amdahl_protocol": {
            "formula": "1 / (unaccelerated_fraction + replaceable_fraction / assumed_device_rate)",
            "assumed_device_rate": ASSUMED_DEVICE_RATE,
            "assumed_device_rate_is_measured": False,
            "infinite_device_limit": "1 / unaccelerated_fraction",
            "hundred_x_unaccelerated_fraction_max": HUNDRED_X_UNACCELERATED_FRACTION_MAX,
            "necessary_not_sufficient": True,
            "requires_complete_measured_service": True,
        },
        "placement_recommendation": "retain_cpu_or_batch_before_port",
        "device_paths": planned_devices,
        "prior_failure_recurrence": {
            "source_experiment": "exp7379-hardware-envelope",
            "prior_honest_verdict": hardware.get("honest_verdict"),
            "same_external_block": changed_receipt is None,
            "retire_if_same_verdict": True,
            "addressed_by": (
                "preserve board evidence and inspect new decision-head costs without a physical retry"
            ),
        },
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "promotion_score": 0,
        "hardware_operations": {
            "model_loads": 0,
            "model_generations": 0,
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
        },
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Reject altered rows, promoted projections, and success-shaped blocks."""

    artifact = value if isinstance(value, Mapping) else {}
    boards = artifact.get("board_rows") if isinstance(artifact.get("board_rows"), list) else []
    placement = (
        artifact.get("placement_envelope_rows")
        if isinstance(artifact.get("placement_envelope_rows"), list)
        else []
    )
    current = (artifact.get("invocation_counts") or {}).get("current")
    validation_ok = artifact.get("required_checks_passed") is True
    selector = artifact.get("selector_checkpoint") or {}
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
        "substrate": artifact.get("inference_substrate")
        != "host_cpu_json_checkpoint_hash_and_amdahl_reduction"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "board_rows": len(boards) != 3
        or {row.get("board") for row in boards if isinstance(row, Mapping)}
        != {"KV260", "GateMate", "PolarFire"}
        or not _rows_valid(boards),
        "placement_rows": len(placement) != 3 or not _rows_valid(placement),
        "rows": artifact.get("rows") != [*boards, *placement],
        "selector_checkpoint": selector.get("numeric_parameter_count") != 17
        or selector.get("estimated_minimum_transfer_bytes") != 68
        or selector.get("transfer_is_measured") is not False,
        "scores": artifact.get("hardware_ready_score") != 0
        or artifact.get("hardware_value_score") != 0
        or artifact.get("promotion_score") != 0
        or artifact.get("board_disposition_complete_score") not in (0, 1),
        "operations": any(
            count != 0 for count in (artifact.get("hardware_operations") or {}).values()
        ),
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
    """Recompute candidate equality and scores from retained raw evidence."""

    raw_boards = raw.get("board_rows") if isinstance(raw.get("board_rows"), list) else []
    raw_placement = (
        raw.get("placement_envelope_rows")
        if isinstance(raw.get("placement_envelope_rows"), list)
        else []
    )
    source_paths = (
        HARDWARE_SOURCE_PATH.as_posix(),
        TRAINING_SOURCE_PATH.as_posix(),
        ONLINE_SOURCE_PATH.as_posix(),
        PROOF_SOURCE_PATH.as_posix(),
    )
    expected_score = int(
        len(raw_boards) == 3
        and {row.get("board") for row in raw_boards} == {"KV260", "GateMate", "PolarFire"}
        and artifact.get("required_checks_passed") is True
    )
    return {
        "candidate_board_rows_match": artifact.get("board_rows") == raw_boards,
        "candidate_placement_rows_match": artifact.get("placement_envelope_rows") == raw_placement,
        "candidate_rows_match": artifact.get("rows") == [*raw_boards, *raw_placement],
        "candidate_selector_matches": artifact.get("selector_checkpoint")
        == raw.get("selector_checkpoint"),
        "candidate_score_matches": artifact.get("board_disposition_complete_score")
        == expected_score,
        "source_hashes_match": all(
            (artifact.get("source_artifact_hashes") or {}).get(path)
            == (raw.get("source_hashes") or {}).get(path)
            for path in source_paths
        ),
    }


def cold_validate_candidate(paths: ExperimentPaths) -> list[str]:
    """Reload the candidate and independently compare all retained raw rows."""

    candidate = _load_object(paths.terminal_candidate) or {}
    raw = _load_object(paths.raw_evidence) or {}
    replay = independent_reduce(candidate, raw)
    errors = validate_artifact(candidate)
    names = {
        "candidate_board_rows_match": "candidate_board_rows_mismatch",
        "candidate_placement_rows_match": "candidate_placement_rows_mismatch",
        "candidate_rows_match": "candidate_rows_mismatch",
        "candidate_selector_matches": "candidate_selector_mismatch",
        "candidate_score_matches": "candidate_score_mismatch",
        "source_hashes_match": "candidate_source_hashes_mismatch",
    }
    errors.extend(error for field, error in names.items() if replay[field] is not True)
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a locally valid terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the actual Exp7358 command plan for this task's files."""

    return command_boundary.build_command_plan(root, V648_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad validation targets, missing parents, and command drift."""

    return command_boundary.validate_command_plan(root, V648_MANIFEST, commands)


def terminal_commands(
    root: Path, paths: ExperimentPaths
) -> list[command_boundary.PlannedCommand]:  # pragma: no cover
    """Build cold replay and the two required strict terminal readers."""

    python = str(root / ".venv/bin/python")
    replay = (
        "from pathlib import Path; "
        "from carnot.experiment_7393_v648_hardware_placement import ExperimentPaths,cold_validate_candidate; "
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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7393-"))
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
    reduced = command_boundary.reduce_affected_receipts(root, V648_MANIFEST, receipts)
    validation = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "validation_receipts": receipts,
        "repository_health": validation_scope.build_repository_health([]),
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
        "operator": "all_pass",
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
    """Parse the frozen date without adding launcher behavior."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Execute the frozen V648 host placement task."""

    print("[exp7393] phase=startup event=flushed", flush=True)
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
