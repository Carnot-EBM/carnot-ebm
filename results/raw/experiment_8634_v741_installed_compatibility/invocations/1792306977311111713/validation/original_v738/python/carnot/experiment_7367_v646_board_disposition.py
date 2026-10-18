"""Produce the V646 board disposition from authenticated historical evidence.

The module performs host-only accounting. It does not contact hardware, load a
model, or turn an old timing receipt into a current availability claim.

Spec refs: REQ-REPORT-7367 and SCENARIO-REPORT-7367-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7355_v645_board_state as board_source
from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7367
TASK_ID = "exp7367-board-disposition"
MILESTONE = "2026.09.646"
RUN_DATE = "20260917"
SCHEMA = "carnot.experiment_7367.v646_board_disposition.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7367_v646_board_disposition.json")
RAW_DIR = Path("results/raw/experiment_7367_v646_board_disposition")
BOARD_SOURCE_PATH = Path("results/experiment_7355_v645_board_state.json")
COST_SOURCE_PATH = Path("results/experiment_7340_v644_native_cost.json")
VALIDATION_BOUNDARY_PATH = Path("results/experiment_7358_v646_validation_contract.json")
HISTORICAL_MODEL_SIDECAR_PATH = Path(
    "results/raw/experiment_7355_v645_board_state/historical_model_receipts.json"
)
MODULE_PATH = Path("python/carnot/experiment_7367_v646_board_disposition.py")
TEST_PATH = Path("tests/python/test_experiment_7367_v646_board_disposition.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7367_v646_board_disposition.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PHYSICAL_RECEIPT_CONTRACT = deepcopy(board_source.PHYSICAL_RECEIPT_CONTRACT)
MISSING_RECEIPT = board_source.MISSING_RECEIPT
ZERO_INVOCATION_COUNTS = deepcopy(command_boundary.ZERO_INVOCATION_COUNTS)
REQUIRED_ARTIFACT_FIELDS = {
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
    "board_disposition_complete_score",
    "board_rows",
    "changed_state_receipt",
    "hardware_ready_score",
    "hardware_value_score",
    "promotion_score",
}
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("research-hardware-wishlist.md"),
    Path("research-references.md"),
    SPEC_PATH,
    BOARD_SOURCE_PATH,
    COST_SOURCE_PATH,
    VALIDATION_BOUNDARY_PATH,
    HISTORICAL_MODEL_SIDECAR_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7355_v645_board_state.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

V646_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=TASK_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, validation logs, and terminal output separate."""

    artifact: Path
    raw_rows: Path
    physical_state_search: Path
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
            raw_rows=raw / "board_rows.json",
            physical_state_search=raw / "gatemate_physical_state_receipt_search.json",
            historical_models=raw / "historical_model_receipts.json",
            terminal_candidate=raw / "terminal_candidate.json",
            validation_dir=raw / "validation",
        )


def utc_now() -> str:
    """Return an aware timestamp for a real task boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase and subprocess boundaries so long work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7367] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a later replacement cannot inherit authority."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for rows and reduced claims."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file only after its complete bytes reach a temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_object(path: Path) -> JsonDict:
    """Load one required JSON object and reject every other shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is not a mapping: {path}")
    return value


def _check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep the exact expected and observed values for each prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def source_errors(board: Mapping[str, Any], cost: Mapping[str, Any]) -> list[str]:
    """Reject source drift before any diagnostic value is reduced."""

    failures = {
        "board_source_identity": board.get("experiment_id") != 7355
        or board.get("milestone") != "2026.09.645",
        "board_source_terminal_scope": board.get("status") != "blocked"
        or board.get("verdict_class") != "blocked",
        "board_source_validation": board.get("required_checks_passed") is not True
        or board.get("flagged_adversarial") is not False,
        "board_source_rows": board.get("board_disposition_complete_score") != 1
        or len(board.get("board_rows") or []) != 3,
        "cost_source_identity": cost.get("experiment_id") != 7340
        or cost.get("milestone") != "2026.09.644",
        "cost_source_terminal_scope": cost.get("status") != "complete"
        or cost.get("verdict_class") != "null",
        "cost_source_validation": cost.get("required_checks_passed") is not True
        or cost.get("flagged_adversarial") not in (None, False),
        "native_tenfold_boundary": cost.get("native_cost_complete_score") != 1
        or cost.get("native_ten_x_score") != 0
        or (cost.get("acceptance_gate_results") or {}).get("native_ten_x", {}).get("passed")
        is not False,
    }
    return [name for name, failed in failures.items() if failed]


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Authenticate all exact inputs before board or cost reduction."""

    for destination in paths.__dict__.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        sizes[relative.as_posix()] = path.stat().st_size if path.is_file() else 0
        if path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)

    checks = [
        _check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size > 0 for size in sizes.values()),
        )
    ]
    board = _load_object(root / BOARD_SOURCE_PATH)
    cost = _load_object(root / COST_SOURCE_PATH)
    boundary = _load_object(root / VALIDATION_BOUNDARY_PATH)
    sidecar = _load_object(root / HISTORICAL_MODEL_SIDECAR_PATH)
    errors = source_errors(board, cost)
    checks.extend(
        [
            _check(
                "driving_capability_spec",
                SPEC_PATH.as_posix(),
                "REQ-REPORT-7367 and scenarios",
                True,
                "REQ-REPORT-7367" in (root / SPEC_PATH).read_text(encoding="utf-8"),
                "REQ-REPORT-7367" in (root / SPEC_PATH).read_text(encoding="utf-8"),
            ),
            _check(
                "authenticated_historical_sources",
                "Exp7355 and Exp7340",
                "identity/status/validation/value-boundary",
                [],
                errors,
                not errors,
            ),
            _check(
                "exp7358_command_boundary",
                VALIDATION_BOUNDARY_PATH.as_posix(),
                "validation_contract_ready_score",
                1,
                boundary.get("validation_contract_ready_score"),
                boundary.get("validation_contract_ready_score") == 1
                and boundary.get("flagged_adversarial") is False,
            ),
            _check(
                "historical_model_sidecar",
                HISTORICAL_MODEL_SIDECAR_PATH.as_posix(),
                "current_MODEL_SPECS/current_model_invoked",
                {"current_MODEL_SPECS": [], "current_model_invoked": False},
                {
                    "current_MODEL_SPECS": sidecar.get("current_MODEL_SPECS"),
                    "current_model_invoked": sidecar.get("current_model_invoked"),
                },
                sidecar.get("current_MODEL_SPECS") == []
                and sidecar.get("current_model_invoked") is False,
            ),
            _check(
                "task_owned_destinations",
                "host",
                "parent paths",
                True,
                all(
                    path.parent.exists() or path.parent.parent.exists()
                    for path in paths.__dict__.values()
                ),
                all(
                    path.parent.exists() or path.parent.parent.exists()
                    for path in paths.__dict__.values()
                ),
            ),
        ]
    )
    context = {
        "board_source": {
            "experiment_id": board.get("experiment_id"),
            "milestone": board.get("milestone"),
            "status": board.get("status"),
            "verdict_class": board.get("verdict_class"),
            "board_disposition_complete_score": board.get("board_disposition_complete_score"),
            "diagnostic_only": True,
            "authorizes_promotion": False,
        },
        "cost_source": {
            "experiment_id": cost.get("experiment_id"),
            "milestone": cost.get("milestone"),
            "status": cost.get("status"),
            "verdict_class": cost.get("verdict_class"),
            "native_cost_complete_score": cost.get("native_cost_complete_score"),
            "native_ten_x_score": cost.get("native_ten_x_score"),
        },
        "validation_boundary": {
            "experiment_id": boundary.get("experiment_id"),
            "validation_contract_ready_score": boundary.get("validation_contract_ready_score"),
        },
        "historical_model_source": deepcopy(sidecar),
    }
    return checks, hashes, context


def search_changed_state_receipt(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Reuse the approved parser and label this task's read-only search."""

    result = board_source.search_physical_state_receipts(
        root, raw_path, candidate_paths=candidate_paths
    )
    raw = _load_object(raw_path)
    raw.update(
        {
            "schema": "carnot.experiment_7367.gatemate_physical_state_search.v1",
            "run_date": RUN_DATE,
            "reader": "carnot.experiment_7355_v645_board_state.search_physical_state_receipts",
            "authorized_scope": "approved local operator sources only",
            "hardware_operations_issued": [],
        }
    )
    atomic_json(raw_path, raw)
    result.update(
        {
            "search_receipt_path": str(raw_path),
            "search_receipt_hash": sha256_file(raw_path),
            "accepted_receipt_count": raw.get("accepted_receipt_count", 0),
            "eligibility_contract": deepcopy(PHYSICAL_RECEIPT_CONTRACT),
        }
    )
    return result


def write_historical_model_sidecar(
    path: Path, source_hash: str, source: Mapping[str, Any]
) -> JsonDict:
    """Keep old model-shaped fields separate from this task's zero calls."""

    value = {
        "schema": "carnot.experiment_7367.historical_model_receipts.v1",
        "label": "historical_diagnostic_only_no_current_generation",
        "authorizes_current_inference": False,
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "source": {
            "path": HISTORICAL_MODEL_SIDECAR_PATH.as_posix(),
            "sha256": source_hash,
            "historical_MODEL_SPECS": deepcopy(
                source.get("source", {}).get("historical_MODEL_SPECS")
            ),
            "historical_model_invoked": source.get("source", {}).get("historical_model_invoked"),
            "historical_invocation_counts": deepcopy(
                source.get("source", {}).get("historical_invocation_counts") or {}
            ),
        },
    }
    atomic_json(path, value)
    return value


def build_board_rows(
    board: Mapping[str, Any], cost: Mapping[str, Any], physical: Mapping[str, Any]
) -> list[JsonDict]:
    """Keep three board claims separate and retain their original denominators."""

    metadata = {
        "KV260": {
            "execution_venue": "kv260_fpga_fabric",
            "denominator": {"unit": "fabric_samples", "count": 32},
            "availability_class": "graduated_historical",
            "fpga_sampling_claimed": True,
        },
        "PolarFire": {
            "execution_venue": "polarfire_linux_cpu",
            "denominator": {"unit": "cpu_dispatches", "count": 1},
            "availability_class": "graduated_historical",
            "fpga_sampling_claimed": False,
        },
        "GateMate": {
            "execution_venue": "none_read_only",
            "denominator": {"unit": "new_hardware_runs", "count": 0},
            "availability_class": (
                "future_bounded_task_eligible" if physical.get("exists") is True else "blocked"
            ),
            "fpga_sampling_claimed": False,
        },
    }
    output: list[JsonDict] = []
    for value in board.get("board_rows") or []:
        if not isinstance(value, Mapping) or value.get("board") not in metadata:
            continue
        row = deepcopy(dict(value))
        row.pop("row_sha256", None)
        name = str(row["board"])
        row.update(deepcopy(metadata[name]))
        row.update(
            {
                "evidence_date": row.get("latest_receipt_date"),
                "prerequisite": row.get("prerequisite_check"),
                "next_action": row.get("exact_next_condition"),
                "native_tenfold_speed_gate": None,
                "complete_boundary_native_ten_x_score": cost.get("native_ten_x_score"),
                "new_hardware_execution_claimed": False,
                "present_availability_asserted": False,
                "source_used_as_readiness_gate": False,
                "hardware_operations_issued": [],
            }
        )
        if name == "GateMate":
            changed = physical.get("exists") is True
            row.update(
                {
                    "disposition": (
                        "changed_physical_state_future_experiment_eligible"
                        if changed
                        else "blocked_changed_physical_state"
                    ),
                    "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                    "operator_source_path": physical.get("source_path"),
                    "next_action": (
                        board_source.GATEMATE_FUTURE_ACTION
                        if changed
                        else board_source.GATEMATE_OPERATOR_ACTION
                    ),
                    "error": None if changed else MISSING_RECEIPT,
                    "failed_value": None if changed else MISSING_RECEIPT,
                    "abstention": not changed,
                    "metric": changed,
                }
            )
        row["row_sha256"] = canonical_hash(row)
        output.append(row)
    return output


def reconcile_devices() -> list[JsonDict]:
    """Separate owned history, an owned block, and vendor-announced access."""

    return [
        {
            "device": "KV260",
            "class": "owned_graduated_historical",
            "owned_by_carnot": True,
            "availability": "historical execution preserved; current reachability not asserted",
            "source_date": "20260915",
            "runtime_claimed": False,
            "speed_claimed": False,
        },
        {
            "device": "PolarFire",
            "class": "owned_graduated_historical",
            "owned_by_carnot": True,
            "availability": "historical CPU dispatch preserved; FPGA sampling not observed",
            "source_date": "20260915",
            "runtime_claimed": False,
            "speed_claimed": False,
        },
        {
            "device": "GateMate",
            "class": "owned_blocked",
            "owned_by_carnot": True,
            "availability": "blocked pending dated operator physical-change receipt",
            "source_date": RUN_DATE,
            "runtime_claimed": False,
            "speed_claimed": False,
        },
        {
            "device": "Extropic Z1",
            "class": "vendor_announced",
            "owned_by_carnot": False,
            "availability": "vendor early access targeted for 2027",
            "source_date": RUN_DATE,
            "source_path": "research-references.md",
            "source_statement": "2026-09-17 source status; Z1 Stick and Card early access in 2027",
            "authenticated_carnot_route": False,
            "runtime_claimed": False,
            "speed_claimed": False,
        },
    ]


def reduce_board_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce accounting only after all three named rows remain distinct."""

    boards = [row.get("board") for row in rows]
    complete = len(rows) == 3 and set(boards) == {"KV260", "PolarFire", "GateMate"}
    return {
        "row_count": len(rows),
        "boards": boards,
        "complete": complete,
        "blocked_boards": [
            row.get("board") for row in rows if row.get("availability_class") == "blocked"
        ],
        "graduated_historical_boards": [
            row.get("board")
            for row in rows
            if row.get("availability_class") == "graduated_historical"
        ],
    }


def native_cost_context(cost: Mapping[str, Any]) -> JsonDict:
    """Preserve the complete-boundary null without creating a board speed gate."""

    gate = deepcopy((cost.get("acceptance_gate_results") or {}).get("native_ten_x") or {})
    return {
        "source_path": COST_SOURCE_PATH.as_posix(),
        "measurement_scope": "complete Python-to-result request boundary",
        "sample_size_budget": deepcopy(cost.get("sample_size_budget")),
        "native_ten_x_score": cost.get("native_ten_x_score"),
        "complete_boundary_gate_expected": gate.get("expected"),
        "complete_boundary_ci95_lower_by_batch_size": gate.get("observed"),
        "complete_boundary_gate_passed": gate.get("passed"),
        "inner_kernel_combined_with_service": False,
        "board_speed_gate_created": False,
    }


def _span(name: str, start: float, end: float, origin: float, units: int) -> JsonDict:
    """Record one measured, disjoint monotonic phase interval."""

    return {
        "phase": name,
        "start_s": max(0.0, start - origin),
        "end_s": max(0.0, end - origin),
        "duration_s": max(0.0, end - start),
        "completed_units": units,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failure and expose the first exact blocking check."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = deepcopy(failures[0]) if failures else None
    if first is not None:
        first.pop("passed", None)
    return {
        "passed": not failures,
        "checks": [dict(row) for row in checks],
        "failures": failures,
        "first_failure": first,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields without wrapping their JSON values."""

    specific = {
        "schema": "Version this record and retain ordinary top-level experiment identity.",
        "status": "Use a terminal status only after actual work and affected validation.",
        "run_date": "Use 20260917 and retain actual UTC timestamps.",
        "preconditions_checked": "Check exact inputs, resources, and required fields before reduction.",
        "MODEL_SPECS": "List actual intended current models; this aggregation intends none.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Separate current call outcomes from labeled historical receipts.",
        "inference_substrate": "Name actual host aggregation and keep historical inference in sidecars.",
        "inference_substrate_class": "Use the actual closed duration class without padding.",
        "execution_venue": "Record host CPU work and make no V646 board-execution claim.",
        "duration_s": "Use measured monotonic elapsed time without sleep or synthetic time.",
        "phase_spans": "Retain disjoint measured load, generation, evaluation, validation, and write spans.",
        "random_seed": "Use null because deterministic evidence reconciliation does not sample.",
        "reproducibility_checksum": "Bind code, settings, evaluator, inputs, and raw evidence.",
        "source_artifact_hashes": "Hash exact producer paths and preserve their original classes.",
        "rows": "Retain every board unit, cost boundary, failure, and abstention disposition.",
        "sample_size_budget": "Freeze planned, attempted, completed, censored units and stopping rules.",
        "acceptance_gate_results": "Separate validation, safety, value, and promotion outcomes.",
        "gate_check_summary": "Name the exact upstream, field, expected value, and observed value.",
        "verifier_is_oracle": "Disclose that the evaluator defines artifact-format truth.",
        "honest_verdict": "Distinguish complete accounting from blocked hardware science.",
        "verdict_class": "Use the closed terminal class; external unchanged absence is blocked.",
        "flagged_adversarial": "Set true for current critical findings and prevent promotion.",
        "validation_receipts": "Retain exact command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Keep dated unrelated failures separate from affected validation.",
        "field_principles": "Explain fields without wrapping scalar gates or dictionaries.",
        "board_disposition_complete_score": "Count three authenticated rows only; do not authorize science.",
        "board_rows": "Keep three provenance, venue, prerequisite, availability, and action rows.",
        "changed_state_receipt": "Store dated operator evidence or null after the exact failed check.",
        "hardware_ready_score": "Remain zero because this task performs no hardware qualification.",
        "hardware_value_score": "Remain zero because this task measures no new board performance.",
        "promotion_score": "Remain zero because history and vendor announcements authorize no claim.",
    }
    return {
        key: specific.get(key, "Retain this evidence in its ordinary JSON type.") for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence and conclusions while excluding runtime receipts."""

    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "source_artifact_hashes",
            "preconditions_checked",
            "board_rows",
            "native_cost_context",
            "device_reconciliation",
            "changed_state_receipt",
            "sample_size_budget",
            "acceptance_gate_results",
            "gate_check_summary",
            "board_disposition_complete_score",
            "hardware_ready_score",
            "hardware_value_score",
            "promotion_score",
            "verdict_class",
        )
    }
    return canonical_hash(bound)


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    validation: Mapping[str, Any],
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Build a terminal record from current bytes and approved receipt sources."""

    origin = time.monotonic()
    started_at = utc_now()
    load_started = time.monotonic()
    checks, source_hashes, context = collect_preconditions(root, paths)
    board = _load_object(root / BOARD_SOURCE_PATH)
    cost = _load_object(root / COST_SOURCE_PATH)
    historical = _load_object(root / HISTORICAL_MODEL_SIDECAR_PATH)
    load_ended = time.monotonic()

    generation_started = load_ended
    generation_ended = generation_started
    evaluation_started = generation_ended
    physical = search_changed_state_receipt(
        root, paths.physical_state_search, candidate_paths=candidate_paths
    )
    rows = build_board_rows(board, cost, physical)
    for row in rows:
        row["input_artifact_path"] = BOARD_SOURCE_PATH.as_posix()
        row["input_artifact_sha256"] = source_hashes[BOARD_SOURCE_PATH.as_posix()]
        row.pop("row_sha256", None)
        row["row_sha256"] = canonical_hash(row)
    reduction = reduce_board_rows(rows)
    devices = reconcile_devices()
    native_context = native_cost_context(cost)
    evaluation_ended = time.monotonic()

    validation_started = evaluation_ended
    source_ok = all(row.get("passed") is True for row in checks)
    validation_ok = validation.get("required_checks_passed") is True
    receipt_exists = physical.get("exists") is True
    validation_ended = time.monotonic()

    write_started = validation_ended
    historical_receipt = write_historical_model_sidecar(
        paths.historical_models,
        source_hashes[HISTORICAL_MODEL_SIDECAR_PATH.as_posix()],
        historical,
    )
    atomic_json(
        paths.raw_rows,
        {
            "schema": "carnot.experiment_7367.board_rows.v1",
            "source_hashes": {
                BOARD_SOURCE_PATH.as_posix(): source_hashes[BOARD_SOURCE_PATH.as_posix()],
                COST_SOURCE_PATH.as_posix(): source_hashes[COST_SOURCE_PATH.as_posix()],
            },
            "rows": rows,
            "reduction": reduction,
        },
    )
    source_hashes[str(paths.physical_state_search)] = sha256_file(paths.physical_state_search)
    source_hashes[str(paths.historical_models)] = sha256_file(paths.historical_models)
    source_hashes[str(paths.raw_rows)] = sha256_file(paths.raw_rows)
    write_ended = time.monotonic()

    physical_gate = _check(
        "gatemate_changed_physical_state_receipt",
        "physical_state_receipt",
        "receipt_date/operator_authored/provenance/changed_field",
        deepcopy(PHYSICAL_RECEIPT_CONTRACT),
        (
            {
                "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                "selected_source_path": physical.get("source_path"),
                "absence": physical.get("observed_missing_receipt"),
            }
            if not receipt_exists
            else {
                "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                "selected_source_path": physical.get("source_path"),
                "receipt_date": physical.get("date_evidence"),
            }
        ),
        receipt_exists,
    )
    validation_gate = _check(
        "affected_required_validation",
        "Exp7303 scoped runner through Exp7358 command plan",
        "required_checks_passed",
        True,
        validation.get("required_checks_passed"),
        validation_ok,
    )
    gate_checks = [*checks, validation_gate, physical_gate]
    if not source_ok:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_source_precondition: one or more exact historical inputs failed authentication"
        )
    elif not validation_ok:
        status = "complete"
        verdict_class = "disqualified"
        honest = (
            "complete_disqualified: affected required validation failed; no board claim is promoted"
        )
    elif not receipt_exists:
        status = "blocked"
        verdict_class = "blocked"
        honest = (
            "blocked_changed_physical_state: no operator-authored dated GateMate cable, port, "
            "board, power, JTAG, or DirtyJTAG change exists after Exp6559; three diagnostic "
            "board dispositions are complete, while readiness, value, and promotion remain zero"
        )
    else:
        status = "complete"
        verdict_class = "null"
        honest = (
            "complete_null_changed_state_receipt_future_task_only: the dated receipt only makes "
            "a separate bounded GateMate task eligible; V646 ran no hardware and promotes nothing"
        )

    score = int(source_ok and validation_ok and reduction["complete"])
    changed_receipt = (
        {
            "receipt_date": physical.get("date_evidence"),
            "operator_authored": True,
            "source_path": physical.get("source_path"),
            "source_sha256": physical.get("evidence_hash"),
            "changed_conditions": deepcopy(physical.get("changed_conditions") or {}),
            "authorizes_current_operation": False,
            "next_action": "future separately bounded integration task",
        }
        if receipt_exists
        else None
    )
    acceptance = {
        "source_authentication": {
            "expected": "exact authenticated Exp7355, Exp7340, and Exp7358 boundary bytes",
            "observed": source_ok,
            "passed": source_ok,
        },
        "affected_validation": {
            "expected": "all eight exact scoped checks pass",
            "observed": validation.get("required_checks_passed"),
            "passed": validation_ok,
        },
        "board_accounting": {
            "expected": {"boards": ["KV260", "PolarFire", "GateMate"], "count": 3},
            "observed": reduction,
            "passed": reduction["complete"],
        },
        "changed_physical_state": {
            "expected": deepcopy(PHYSICAL_RECEIPT_CONTRACT),
            "observed": changed_receipt,
            "passed": receipt_exists,
        },
        "hardware_safety": {
            "expected": "zero current hardware and external operations",
            "observed": 0,
            "passed": True,
        },
        "hardware_value": {
            "expected": "new authenticated board performance measurement",
            "observed": None,
            "passed": False,
        },
        "promotion": {
            "expected": "qualified current hardware value",
            "observed": None,
            "passed": False,
        },
    }
    phase_spans = [
        _span("load", load_started, load_ended, origin, 3),
        _span("generation", generation_started, generation_ended, origin, 0),
        _span("evaluation", evaluation_started, evaluation_ended, origin, len(rows)),
        _span(
            "validation",
            validation_started,
            validation_ended,
            origin,
            len(validation.get("validation_receipts") or []),
        ),
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
            "current": deepcopy(ZERO_INVOCATION_COUNTS),
            "historical": {
                "sidecar_count": 1,
                "sidecars": [
                    {
                        "path": str(paths.historical_models),
                        "sha256": source_hashes[str(paths.historical_models)],
                        "label": historical_receipt["label"],
                    }
                ],
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "operation": "CPython JSON parsing, SHA-256 hashing, and deterministic reduction",
            "hostname": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "host_cpu",
        },
        "duration_s": max(write_ended - origin, 0.0001),
        "phase_spans": phase_spans,
        "random_seed": None,
        "source_artifact_hashes": source_hashes,
        "source_artifact_states": context,
        "rows": rows,
        "board_rows": rows,
        "raw_row_reduction": reduction,
        "sample_size_budget": {
            "board_rows_planned": 3,
            "board_rows_attempted": len(rows),
            "board_rows_completed": len(rows),
            "board_rows_censored": 0,
            "new_hardware_runs_planned": 0,
            "new_hardware_runs_attempted": 0,
            "new_hardware_runs_completed": 0,
            "new_hardware_runs_censored": 0,
            "stopping_rule": "account for exactly KV260, PolarFire, and GateMate; stop before hardware work",
        },
        "acceptance_gate_results": acceptance,
        "gate_check_summary": _gate_summary(gate_checks),
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
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
        "native_tenfold_speed_gate": None,
        "native_cost_context": native_context,
        "device_reconciliation": devices,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "promotion_score": 0,
        "hardware_operations": {
            "usb": 0,
            "ssh": 0,
            "jtag": 0,
            "fpga": 0,
            "rocm": 0,
            "thermodynamic_devices": 0,
            "purchases": 0,
            "vendor_contacts": 0,
        },
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Reject changed conclusions, missing evidence, and success-shaped blocks."""

    artifact = value if isinstance(value, Mapping) else {}
    rows = artifact.get("board_rows") if isinstance(artifact.get("board_rows"), list) else []
    row_hashes_valid = all(
        isinstance(row, Mapping)
        and row.get("row_sha256")
        == canonical_hash({key: item for key, item in row.items() if key != "row_sha256"})
        for row in rows
    )
    current_counts = (artifact.get("invocation_counts") or {}).get("current", {})
    failures = {
        "required_fields": not REQUIRED_ARTIFACT_FIELDS.issubset(artifact),
        "identity": artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "model_declaration": artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or current_counts != ZERO_INVOCATION_COUNTS,
        "substrate": artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "rows": reduce_board_rows(rows).get("complete") is not True
        or artifact.get("rows") != rows
        or not row_hashes_valid,
        "native_boundary": artifact.get("native_tenfold_speed_gate") is not None
        or (artifact.get("native_cost_context") or {}).get("inner_kernel_combined_with_service")
        is not False,
        "scores": artifact.get("hardware_ready_score") != 0
        or artifact.get("hardware_value_score") != 0
        or artifact.get("promotion_score") != 0
        or artifact.get("board_disposition_complete_score") not in (0, 1),
        "operations": any(
            value != 0 for value in (artifact.get("hardware_operations") or {}).values()
        ),
        "verdict": artifact.get("verdict_class") not in command_boundary.CLOSED_VERDICTS
        or (
            artifact.get("changed_state_receipt") is None
            and artifact.get("required_checks_passed") is True
            and artifact.get("verdict_class") != "blocked"
        )
        or (
            artifact.get("required_checks_passed") is False
            and artifact.get("verdict_class") != "disqualified"
        ),
        "field_principles": not REQUIRED_ARTIFACT_FIELDS.issubset(
            artifact.get("field_principles") or {}
        ),
        "checksum": artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
    }
    return [name for name, failed in failures.items() if failed]


def independent_reduce(artifact: Mapping[str, Any], raw: Mapping[str, Any]) -> JsonDict:
    """Recompute row and source conclusions without trusting headline scores."""

    rows = raw.get("rows") if isinstance(raw.get("rows"), list) else []
    reduction = reduce_board_rows(rows)
    return {
        "candidate_rows_match": artifact.get("board_rows") == rows,
        "candidate_score_matches": artifact.get("board_disposition_complete_score")
        == int(reduction["complete"] and artifact.get("required_checks_passed") is True),
        "source_hashes_match": all(
            artifact.get("source_artifact_hashes", {}).get(path)
            == raw.get("source_hashes", {}).get(path)
            for path in (BOARD_SOURCE_PATH.as_posix(), COST_SOURCE_PATH.as_posix())
        ),
        "reduction": reduction,
    }


def cold_validate_candidate(paths: ExperimentPaths) -> list[str]:
    """Reload the candidate and independently reduce the retained raw rows."""

    candidate = _load_object(paths.terminal_candidate)
    raw = _load_object(paths.raw_rows)
    replay = independent_reduce(candidate, raw)
    errors = validate_artifact(candidate)
    if replay["candidate_rows_match"] is not True:
        errors.append("candidate_rows_mismatch")
    if replay["candidate_score_matches"] is not True:
        errors.append("candidate_score_mismatch")
    if replay["source_hashes_match"] is not True:
        errors.append("candidate_source_hashes_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically write only an artifact that passes local validation."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Use the shipped V646 boundary with this task's exact affected files."""

    return command_boundary.build_command_plan(root, V646_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any broad test target or command outside the affected manifest."""

    return command_boundary.validate_command_plan(root, V646_MANIFEST, commands)


def _terminal_commands(
    root: Path, paths: ExperimentPaths
) -> list[command_boundary.PlannedCommand]:  # pragma: no cover
    """Build exact candidate replay and terminal-validator commands."""

    python = str(root / ".venv/bin/python")
    replay = (
        "from pathlib import Path; "
        "from carnot.experiment_7367_v646_board_disposition import ExperimentPaths,cold_validate_candidate; "
        f"p=ExperimentPaths(Path({str(paths.artifact)!r}),Path({str(paths.raw_rows)!r}),"
        f"Path({str(paths.physical_state_search)!r}),Path({str(paths.historical_models)!r}),"
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
            validation_scope.CommandSpec(name, argv, category), category, True
        )
        for name, argv, category in specs
    ]


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:  # pragma: no cover
    """Run scoped validation, cold replay, terminal checks, and atomic output."""

    started = time.monotonic()
    progress(started, "preconditions", "before")
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7367-"))
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
    reduced = command_boundary.reduce_affected_receipts(root, V646_MANIFEST, receipts)
    validation = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "validation_receipts": receipts,
    }
    progress(
        started,
        "validation",
        "after_subprocess_group",
        passed=reduced["passed"],
    )

    progress(started, "evaluation", "before")
    artifact = build_artifact(root, paths, validation)
    atomic_json(paths.terminal_candidate, artifact)
    progress(started, "evaluation", "after", rows=len(artifact["board_rows"]))

    terminal = _terminal_commands(root, paths)
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
    artifact["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and row.get("passed") is not True
        for row in terminal_receipts
    )
    if terminal_failed:
        artifact["status"] = "complete"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified: terminal validation failed; diagnostic rows remain, while all "
            "readiness, value, and promotion scores stay zero"
        )
        artifact["board_disposition_complete_score"] = 0
        artifact["required_checks_passed"] = False
        artifact["failed_required_commands"] = sorted(
            set([*artifact["failed_required_commands"], *terminal_failed])
        )
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(paths.artifact, artifact)
    progress(started, "write", "after_atomic_terminal", path=paths.artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover
    """Parse the frozen run date without adding launcher behavior."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Execute the frozen V646 task from the declared thin entrypoint."""

    print("[exp7367] phase=startup event=flushed", flush=True)
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
