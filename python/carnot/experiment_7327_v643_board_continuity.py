"""Produce the V643 read-only board-continuity receipt.

This adapter authenticates Exp7314 and reuses its shipped receipt and board
readers. It does not contact a board or infer current board availability from
historical execution.

Spec refs: REQ-ISING-7327 and SCENARIO-ISING-7327-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_7314_v642_board_continuity as previous
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7327
TASK_ID = "exp7327-board-continuity"
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
SCHEMA = "carnot.experiment_7327.v643_board_continuity.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7327_v643_board_continuity.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7327_v643_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7327_v643_board_continuity")
UPSTREAM_PATH = Path("results/experiment_7314_v642_board_continuity.json")
CUTOFF_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
MODULE_PATH = Path("python/carnot/experiment_7327_v643_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7327_v643_board_continuity.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7327_v643_board_continuity.py")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REFERENCE_PATH = Path("research-references.md")
WISHLIST_PATH = Path("research-hardware-wishlist.md")

PHYSICAL_RECEIPT_CONTRACT = previous.PHYSICAL_RECEIPT_CONTRACT
MISSING_RECEIPT = previous.MISSING_RECEIPT
GATEMATE_FUTURE_ACTION = previous.GATEMATE_FUTURE_ACTION
GATEMATE_OPERATOR_ACTION = previous.GATEMATE_OPERATOR_ACTION

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Preserve board evidence and GateMate changed-state conditions",
    "phase": 4,
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 10,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "prior_failure_ids": [
        "exp5166-hardware-continuity-board-timing-v473",
        "exp5179-hardware-continuity-board-timing-v474",
    ],
    "prompt_sha256": "sha256:73f6bfa633562c6a8b7e3c1a43826f25513edf13c2176a86feb43ed415c621b2",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7314_v642_board_continuity.py"),
    UPSTREAM_PATH,
    CUTOFF_PATH,
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    WISHLIST_PATH,
    SPEC_PATH,
    ROADMAP_PATH,
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

INVOCATION_COUNTS = {
    "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
    "generations": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
}

sha256_file = previous.sha256_file
artifact_checksum = previous.artifact_checksum
reduce_board_rows = previous.reduce_board_rows
_read_json = previous._read_json
_atomic_json = previous._atomic_json
_writable_destination = previous._writable_destination
_check = previous._check


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, raw, and terminal files in separate locations."""

    artifact: Path
    checkpoint: Path
    physical_state_search: Path
    historical_models: Path
    raw_rows: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve every task-owned output below one repository root."""

        raw = root / RAW_DIR
        return cls(
            root / RESULT_PATH,
            root / CHECKPOINT_PATH,
            raw / "gatemate_physical_state_receipt_search.json",
            raw / "historical_model_receipts.json",
            raw / "board_rows.json",
            raw / "terminal_candidate.json",
            raw / "validation",
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Give tests private outputs that cannot change the research record."""

        return cls.defaults(root)


def read_task_contract(root: Path) -> JsonDict | None:
    """Read only the roadmap fields that fix the V643 task identity."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary.
        return None
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):
        return None
    task = next(
        (row for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    result = {
        key: deepcopy(task.get(key))
        for key in EXPECTED_TASK_CONTRACT
        if key not in {"prior_failure_ids", "prompt_sha256"}
    }
    failures = task.get("prior_failures")
    result["prior_failure_ids"] = (
        [row.get("experiment_id") for row in failures if isinstance(row, Mapping)]
        if isinstance(failures, list)
        else None
    )
    prompt = task.get("prompt")
    result["prompt_sha256"] = previous.sha256_text(prompt) if isinstance(prompt, str) else None
    return result


def original_reference_observation(root: Path, receipt: Mapping[str, Any]) -> JsonDict:
    """Verify every original KV260 and PolarFire evidence byte named by Exp7314."""

    paths: dict[str, str] = {}
    boards_found: set[str] = set()
    board_rows = receipt.get("board_rows")
    rows = board_rows if isinstance(board_rows, list) else []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("board") not in {"KV260", "PolarFire"}:
            continue
        boards_found.add(str(row["board"]))
        evidence = [
            {"path": row.get("latest_receipt_path"), "sha256": row.get("latest_receipt_hash")},
            *(row.get("referenced_evidence") or []),
        ]
        for item in evidence:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                paths[str(item["path"])] = str(item.get("sha256"))
    observations: list[JsonDict] = []
    for path_text, expected in paths.items():
        path = Path(path_text)
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        observations.append(
            {
                "path": path_text,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed is not None and observed == expected,
            }
        )
    return {
        "rows": observations,
        "boards_found": sorted(boards_found),
        "all_match": boards_found == {"KV260", "PolarFire"}
        and bool(observations)
        and all(row["passed"] for row in observations),
    }


def _scope_observation(receipt: Mapping[str, Any]) -> JsonDict:
    """Read the two graduated claim classes without broadening either claim."""

    kv260 = previous._board_row(receipt, "KV260") or {}
    polarfire = previous._board_row(receipt, "PolarFire") or {}
    return {
        "kv260_fabric_execution": kv260.get("fabric_execution_completed"),
        "kv260_processor_class": kv260.get("processor_class"),
        "polarfire_cpu_dispatch": polarfire.get("terminal_criterion_met"),
        "polarfire_processor_class": polarfire.get("processor_class"),
        "polarfire_fpga_sampling": polarfire.get("programmable_logic_sampling_observed"),
    }


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Authenticate task identity, Exp7314, and every original evidence byte."""

    print("[exp7327] phase=preconditions event=start", flush=True)
    checks: list[JsonDict] = []
    sizes = {
        path.as_posix(): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    checks.append(
        _check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            "all nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )
    hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[path.as_posix()] not in (None, 0)
    }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    spec_state = {
        "requirement": "REQ-ISING-7327" in spec_text,
        "scenarios": "SCENARIO-ISING-7327-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7327 and scenarios",
            {"requirement": True, "scenarios": True},
            spec_state,
            all(spec_state.values()),
        )
    )
    task = read_task_contract(root)
    checks.append(
        _check(
            "roadmap_task_contract",
            ROADMAP_PATH.as_posix(),
            TASK_ID,
            EXPECTED_TASK_CONTRACT,
            task if task is not None else "missing_task_contract",
            task == EXPECTED_TASK_CONTRACT,
        )
    )
    outputs = {
        "artifact": _writable_destination(paths.artifact),
        "checkpoint": _writable_destination(paths.checkpoint),
        "physical_state_search": _writable_destination(paths.physical_state_search),
        "historical_models": _writable_destination(paths.historical_models),
        "raw_rows": _writable_destination(paths.raw_rows),
        "terminal_candidate": _writable_destination(paths.terminal_candidate),
    }
    checks.append(
        _check(
            "task_owned_outputs",
            "host",
            "writable destinations",
            {key: True for key in outputs},
            outputs,
            all(outputs.values()),
        )
    )
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary.
        manifest = None
    checks.append(
        _check(
            "exclusion_manifest_loaded",
            EXCLUSION_PATH.as_posix(),
            "YAML mapping",
            True,
            isinstance(manifest, Mapping),
            isinstance(manifest, Mapping),
        )
    )
    receipt = _read_json(root / UPSTREAM_PATH)
    quarantine = previous.current.quarantine_authority._quarantine(receipt, manifest, "7314")
    checks.append(
        _check(
            "exp7314_not_quarantined_or_disqualified",
            UPSTREAM_PATH.as_posix(),
            "quarantined_or_disqualified_or_invalidated_or_retired",
            False,
            quarantine,
            bool(receipt) and quarantine.get("quarantined") is False,
        )
    )
    expected_identity = {
        "experiment_id": 7314,
        "milestone": "2026.09.642",
        "status": "complete",
        "verdict_class": "positive",
        "board_continuity_complete_score": 1,
    }
    observed_identity = {key: receipt.get(key) for key in expected_identity}
    checks.append(
        _check(
            "exp7314_terminal_identity",
            UPSTREAM_PATH.as_posix(),
            "producer identity and terminal class",
            expected_identity,
            observed_identity,
            observed_identity == expected_identity,
        )
    )
    try:
        checksum_matches = receipt.get("reproducibility_checksum") == artifact_checksum(receipt)
    except (TypeError, ValueError):  # pragma: no cover - malformed JSON value boundary.
        checksum_matches = False
    checks.append(
        _check(
            "exp7314_reproducibility_checksum",
            UPSTREAM_PATH.as_posix(),
            "reproducibility_checksum",
            "hash of immutable artifact fields",
            receipt.get("reproducibility_checksum"),
            checksum_matches,
        )
    )
    references = original_reference_observation(root, receipt)
    for row in references["rows"]:
        if row["passed"] is True:
            hashes[str(row["path"])] = str(row["observed_sha256"])
    checks.append(
        _check(
            "exp7314_original_terminal_reference_hashes",
            UPSTREAM_PATH.as_posix(),
            "KV260 and PolarFire referenced evidence",
            "all current bytes hash-match",
            references,
            references["all_match"] is True,
        )
    )
    expected_scope = {
        "kv260_fabric_execution": True,
        "kv260_processor_class": "fpga_fabric",
        "polarfire_cpu_dispatch": True,
        "polarfire_processor_class": "cpu",
        "polarfire_fpga_sampling": False,
    }
    scope = _scope_observation(receipt)
    checks.append(
        _check(
            "graduated_board_execution_scopes",
            UPSTREAM_PATH.as_posix(),
            "historical processor claim classes",
            expected_scope,
            scope,
            scope == expected_scope,
        )
    )
    physical = receipt.get("physical_state_receipt")
    expected_cutoff_hash = (
        physical.get("cutoff_source_hash") if isinstance(physical, Mapping) else None
    )
    observed_cutoff_hash = (
        sha256_file(root / CUTOFF_PATH) if (root / CUTOFF_PATH).is_file() else None
    )
    cutoff = {
        "path": CUTOFF_PATH.as_posix(),
        "expected_sha256": expected_cutoff_hash,
        "observed_sha256": observed_cutoff_hash,
    }
    checks.append(
        _check(
            "exp6559_physical_state_boundary",
            CUTOFF_PATH.as_posix(),
            "cutoff_source_hash",
            expected_cutoff_hash,
            observed_cutoff_hash,
            observed_cutoff_hash is not None and observed_cutoff_hash == expected_cutoff_hash,
        )
    )
    reference_text = (
        (root / REFERENCE_PATH).read_text(encoding="utf-8")
        if (root / REFERENCE_PATH).is_file()
        else ""
    )
    literature = {
        "v643_refresh": "## V643 planning research — 2026-09-15" in reference_text,
        "extropic_sparse": "sparse probabilistic computation with" in reference_text,
        "sparsekan": "SparseKAN, 2608.00859" in reference_text,
        "no_local_tsu": "No local TSU" in reference_text,
    }
    checks.append(
        _check(
            "v643_literature_boundaries",
            REFERENCE_PATH.as_posix(),
            "Extropic sparse hardware and SparseKAN context",
            {key: True for key in literature},
            literature,
            all(literature.values()),
        )
    )
    print(f"[exp7327] phase=preconditions event=end checks={len(checks)}", flush=True)
    return (
        checks,
        hashes,
        {
            "receipt": receipt,
            "manifest": manifest,
            "quarantine": quarantine,
            "upstream_checksum_matches": checksum_matches,
            "reference_observation": references,
            "scope_observation": scope,
            "cutoff_observation": cutoff,
        },
    )


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep historical model declarations outside the current task identity."""

    source = _read_json(root / UPSTREAM_PATH)
    receipt = {
        "schema": "carnot.experiment_7327.historical_model_receipts.v1",
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(INVOCATION_COUNTS),
        "source": {
            "path": UPSTREAM_PATH.as_posix(),
            "sha256": sha256_file(root / UPSTREAM_PATH),
            "experiment_id": source.get("experiment_id"),
            "status": source.get("status"),
            "verdict_class": source.get("verdict_class"),
            "historical_MODEL_SPECS": source.get("MODEL_SPECS"),
            "historical_model_invoked": source.get("model_invoked"),
            "historical_invocation_counts": source.get("invocation_counts"),
        },
    }
    _atomic_json(path, receipt)
    return receipt


def search_physical_state_receipts(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Use the shipped GateMate parser and bind its result to V643."""

    result = previous.search_physical_state_receipts(
        root, raw_path, candidate_paths=candidate_paths
    )
    raw = _read_json(raw_path)
    raw.update(
        {
            "schema": "carnot.experiment_7327.gatemate_physical_state_search.v1",
            "run_date": RUN_DATE,
            "reader": "carnot.experiment_7314_v642_board_continuity.search_physical_state_receipts",
        }
    )
    _atomic_json(raw_path, raw)
    result.update(
        {
            "search_receipt_path": str(raw_path),
            "search_receipt_hash": sha256_file(raw_path),
            "eligibility_contract": PHYSICAL_RECEIPT_CONTRACT,
            "accepted_receipt_count": raw.get("accepted_receipt_count", 0),
            "approved_local_sources_only": True,
        }
    )
    return result


def build_board_rows(
    root: Path, receipt: Mapping[str, Any], physical: Mapping[str, Any]
) -> list[JsonDict]:
    """Adapt the shipped three-row builder to the authenticated Exp7314 source."""

    upstreams = {
        "receipt": receipt,
        "kv260_row": previous._board_row(receipt, "KV260"),
        "polarfire_row": previous._board_row(receipt, "PolarFire"),
    }
    rows = previous.build_board_rows(root, upstreams, physical)
    upstream_hash = sha256_file(root / UPSTREAM_PATH)
    output: list[JsonDict] = []
    for source in rows:
        row = deepcopy(source)
        row.pop("row_sha256", None)
        row.update(
            {
                "source_artifact_path": UPSTREAM_PATH.as_posix(),
                "source_artifact_hash": upstream_hash,
                "historical_graduation_preserved": row["board"] in {"KV260", "PolarFire"},
                "present_availability_asserted": False,
                "current_evidence": "read_only_artifact_aggregation",
                "hardware_operations_issued": [],
                "hardware_command_count": 0,
            }
        )
        if row["board"] != "GateMate":
            row.update(
                {
                    "latest_receipt_path": UPSTREAM_PATH.as_posix(),
                    "latest_receipt_hash": upstream_hash,
                    "latest_receipt_date": receipt.get("run_date"),
                    "latest_receipt_authenticated": True,
                }
            )
        output.append(previous.current._finish_row(row))
    return output


def deployment_relevance() -> JsonDict:
    """Record literature context and the exact missing local authorities."""

    return {
        "source_path": REFERENCE_PATH.as_posix(),
        "extropic": {
            "context": "sparse probabilistic computation split between Z1 and a companion FPGA",
            "local_tsu_authority": False,
            "local_device_access": False,
            "scientific_result": False,
            "next_condition": "authenticated local device or credentials with transfer, latency, power, and sample evidence",
            "satisfied_by": "operator or authorized hardware provider",
            "contact_issued": False,
        },
        "sparsekan": {
            "context": "compression of basis functions, neurons, and precision for a future predictor",
            "useful_target_predictor": False,
            "scientific_result": False,
            "next_condition": "a predictor with measured value and complete transfer-cost accounting",
            "satisfied_by": "future research task after predictor value is established",
            "contact_issued": False,
        },
        "availability_is_scientific_result": False,
        "external_actions": [],
    }


def next_hardware_conditions() -> JsonDict:
    """Name the next authority and action for each still-bounded hardware lane."""

    return {
        "gatemate": {
            "condition": GATEMATE_OPERATOR_ACTION,
            "satisfied_by": "operator",
            "future_action_if_satisfied": GATEMATE_FUTURE_ACTION,
            "current_task_authority": "read_only",
        },
        "kv260": {
            "condition": "any future access uses ssh kria only",
            "satisfied_by": "future authorized hardware task",
            "present_availability_claimed": False,
        },
        "polarfire": {
            "condition": "FPGA sampling needs a separate authorized fabric task",
            "satisfied_by": "future authorized hardware task",
            "cpu_dispatch_already_preserved": True,
        },
        "extropic_tsu": {
            "condition": "authenticated local TSU device or credentials",
            "satisfied_by": "operator or authorized hardware provider",
            "local_authority_present": False,
        },
    }


def _field_principles() -> JsonDict:
    """Explain each required field without wrapping executable values."""

    return {
        "schema": "Version this artifact; keep ordinary top-level experiment_id and milestone.",
        "status": "Write terminal output only after current work and required validation.",
        "run_date": "Use 20260915; preserve actual UTC timestamps and monotonic phase spans.",
        "preconditions_checked": "Record input identities, availability, and the exact failed check.",
        "MODEL_SPECS": "Current executable identities only; this task invokes no LLM.",
        "model_invoked": "True for any attempted model load or generation, including failure.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
        "inference_substrate": "Describe actual computation using the recognized substrate literal.",
        "inference_substrate_class": "Use the actual closed substrate class.",
        "execution_venue": "Use host; historical board work is not current board execution.",
        "duration_s": "Measure real elapsed time without sleeping or padding.",
        "phase_spans": "Record disjoint monotonic spans, units, boundaries, and pending operations.",
        "random_seed": "Seal independent development and evaluation seeds before results.",
        "reproducibility_checksum": "Bind code, inputs, settings, validation, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and original evidence bytes.",
        "rows": "Emit every board with metrics, costs, failures, abstentions, and censoring.",
        "sample_size_budget": "Record planned, attempted, complete, censored, and stopping rule.",
        "acceptance_gate_results": "Each check keeps expected, observed, passed, and principle.",
        "gate_check_summary": "Every block names upstream, check, field, expected, and observed.",
        "verifier_is_oracle": "Shared reader authority forbids a new positive scientific claim.",
        "honest_verdict": "External absence starts blocked_; completed validation failure starts complete_disqualified.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "validation_receipts": "Keep exact command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Preserve dated unrelated failures without passing current checks.",
        "field_principles": "Explain why fields exist without wrapping executable values.",
        "board_continuity_complete_score": "One requires three authenticated dispositions and exact next conditions.",
        "board_rows": "Separate historical graduation, current evidence, blocked conditions, and processors.",
        "hardware_operations_issued_count": "Exactly zero for this read-only continuity task.",
        "next_hardware_conditions": "Name GateMate changes, SSH-only KV260 access, and absent TSU authority.",
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give every acceptance gate one explicit comparison shape."""

    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def _phase_span(name: str, start: float, origin: float, units: int) -> JsonDict:
    """Close one disjoint monotonic phase and report its completed units."""

    end = time.monotonic()
    print(f"[exp7327] phase={name} event=end units={units} elapsed_s={end - start:.3f}", flush=True)
    return {
        "phase": name,
        "start_s": start - origin,
        "end_s": end - origin,
        "units": units,
        "checkpoint_boundaries": 1,
        "pending_operations": [],
    }


def _normalized_validation(validation: Mapping[str, Any]) -> list[JsonDict]:
    """Retain the runner fields and add the artifact's required receipt aliases."""

    output: list[JsonDict] = []
    for source in validation.get("validation_receipts", []):
        row = deepcopy(dict(source))
        row["elapsed_s"] = row.get("duration_s")
        row["log_hash"] = row.get("log_sha256")
        output.append(row)
    return output


def _summary(
    checks: Sequence[Mapping[str, Any]],
    physical: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Keep every failed dependency in order with its exact observed value."""

    failures = [
        {
            "upstream": row.get("upstream"),
            "check": row.get("check"),
            "field": row.get("field"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]
    physical_failure = {
        "upstream": "physical_state_receipt",
        "check": "gatemate_changed_physical_state_receipt",
        "field": "receipt_date/operator_authored/provenance/changed_field",
        "expected_value": PHYSICAL_RECEIPT_CONTRACT,
        "observed_value": {
            "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
            "selected_source_path": physical.get("source_path"),
            "absence": MISSING_RECEIPT,
        },
    }
    if physical and physical.get("exists") is not True:
        failures.append(physical_failure)
    if validation.get("required_checks_passed") is not True:
        failures.insert(
            0,
            {
                "upstream": "exp7303_scoped_validation",
                "check": "required_scoped_checks",
                "field": "required_checks_passed",
                "expected_value": True,
                "observed_value": {
                    "required_checks_passed": validation.get("required_checks_passed"),
                    "missing": validation.get("missing_required_commands", []),
                    "failed": validation.get("failed_required_commands", []),
                    "duplicate": validation.get("duplicate_required_commands", []),
                },
            },
        )
    return {
        "passed": not failures,
        "first_failure": failures[0] if failures else None,
        "failures": failures,
        "checks": [deepcopy(dict(row)) for row in checks],
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create every required field before assigning a terminal outcome."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "status": "candidate",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "model_load_count": 0,
        "generation_count": 0,
        "model_invocation_count": 0,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {
            "development": 7327001,
            "evaluation": 7327002,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial: board continuity has not completed",
        "verdict_class": "partial",
        "validation_receipts": [],
        "required_checks_passed": False,
        "missing_required_commands": list(REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {},
        "field_principles": _field_principles(),
        "board_continuity_complete_score": 0,
        "board_rows": [],
        "hardware_operations_issued_count": 0,
        "hardware_operations_issued": [],
        "external_actions": [],
        "physical_state_receipt": {},
        "deployment_relevance": deployment_relevance(),
        "next_hardware_conditions": next_hardware_conditions(),
        "source_artifact_states": {},
    }


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    validation: Mapping[str, Any],
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate immutable evidence and return one terminal in-memory receipt."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    phase_start = time.monotonic()
    checks, hashes, context = collect_preconditions(root, paths)
    spans.append(_phase_span("preconditions", phase_start, origin, len(checks)))
    preconditions_passed = all(row["passed"] is True for row in checks)

    physical: JsonDict = {}
    rows: list[JsonDict] = []
    if preconditions_passed:
        print("[exp7327] phase=evidence event=start", flush=True)
        phase_start = time.monotonic()
        write_historical_model_receipt(root, paths.historical_models)
        physical = search_physical_state_receipts(
            root, paths.physical_state_search, candidate_paths=candidate_paths
        )
        rows = build_board_rows(root, context["receipt"], physical)
        _atomic_json(paths.raw_rows, {"rows": rows})
        hashes.update(
            {
                paths.historical_models.relative_to(
                    paths.artifact.parent.parent
                ).as_posix(): sha256_file(paths.historical_models),
                paths.physical_state_search.relative_to(
                    paths.artifact.parent.parent
                ).as_posix(): sha256_file(paths.physical_state_search),
                paths.raw_rows.relative_to(paths.artifact.parent.parent).as_posix(): sha256_file(
                    paths.raw_rows
                ),
            }
        )
        spans.append(_phase_span("evidence", phase_start, origin, len(rows)))

    artifact = _base_artifact(checks, hashes, spans)
    reduced = reduce_board_rows(rows)
    validation_rows = _normalized_validation(validation)
    validation_passed = validation.get("required_checks_passed") is True
    physical_exists = physical.get("exists") is True
    summary = _summary(checks, physical, validation)
    artifact.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": time.monotonic() - origin,
            "rows": rows,
            "board_rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": len(rows),
                "complete": len(rows),
                "censored": 0,
                "stopping_rule": "one authenticated read-only disposition for each named board",
            },
            "acceptance_gate_results": {
                "exp7314_and_original_evidence_authenticated": _gate(
                    True,
                    preconditions_passed,
                    preconditions_passed,
                    "A producer or original evidence mismatch blocks this task.",
                ),
                "three_board_scopes_preserved": _gate(
                    {"rows": 3, "reduced_score": 1},
                    {
                        "rows": reduced["board_count"],
                        "reduced_score": reduced["board_disposition_complete_score"],
                    },
                    reduced["board_disposition_complete_score"] == 1,
                    "Historical fabric and CPU scopes remain separate from current availability.",
                ),
                "gatemate_changed_physical_state_receipt": _gate(
                    PHYSICAL_RECEIPT_CONTRACT,
                    {
                        "exists": physical_exists,
                        "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                        "observed_missing_receipt": physical.get("observed_missing_receipt"),
                    },
                    physical_exists,
                    "Only a later operator-authored physical change enables future work.",
                ),
                "required_scoped_validation": _gate(
                    True,
                    validation.get("required_checks_passed"),
                    validation_passed,
                    "Every affected test, coverage, format, lint, type, and spec check must pass.",
                ),
                "read_only_operation_boundary": _gate(
                    {"hardware_operations": 0, "external_actions": 0},
                    {"hardware_operations": 0, "external_actions": 0},
                    True,
                    "This task cannot issue a board or external operation.",
                ),
                "literature_context_bounded": _gate(
                    {"availability_is_scientific_result": False, "external_actions": []},
                    {
                        "availability_is_scientific_result": deployment_relevance()[
                            "availability_is_scientific_result"
                        ],
                        "external_actions": deployment_relevance()["external_actions"],
                    },
                    True,
                    "Extropic and SparseKAN remain context without a local result.",
                ),
            },
            "gate_check_summary": summary,
            "validation_receipts": validation_rows,
            "required_checks_passed": validation_passed,
            "missing_required_commands": deepcopy(validation.get("missing_required_commands", [])),
            "failed_required_commands": deepcopy(validation.get("failed_required_commands", [])),
            "duplicate_required_commands": deepcopy(
                validation.get("duplicate_required_commands", [])
            ),
            "repository_health": deepcopy(validation.get("repository_health", {})),
            "physical_state_receipt": physical,
            "source_artifact_states": {
                UPSTREAM_PATH.as_posix(): {
                    "producer_experiment_id": context.get("receipt", {}).get("experiment_id"),
                    "terminal_status": context.get("receipt", {}).get("status"),
                    "terminal_class": context.get("receipt", {}).get("verdict_class"),
                    "quarantine": deepcopy(context.get("quarantine", {})),
                    "checksum_matches": context.get("upstream_checksum_matches"),
                }
            },
        }
    )

    if not validation_passed:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: affected scoped validation failed",
                "board_continuity_complete_score": 0,
            }
        )
    elif not preconditions_passed:
        failure = summary["first_failure"]
        artifact.update(
            {
                "status": "blocked",
                "verdict_class": "blocked",
                "honest_verdict": (
                    f"blocked_external_precondition: {failure['upstream']} check "
                    f"{failure['check']} field {failure['field']} expected "
                    f"{failure['expected_value']!r}; observed {failure['observed_value']!r}"
                ),
                "board_continuity_complete_score": 0,
            }
        )
    elif not physical_exists:
        artifact.update(
            {
                "status": "blocked",
                "verdict_class": "blocked",
                "honest_verdict": (
                    "blocked_changed_physical_state: no operator-authored dated GateMate cable, "
                    "port, board, power, JTAG, or DirtyJTAG change exists after Exp6559; KV260 "
                    "fabric graduation and PolarFire CPU dispatch remain preserved without a "
                    "present-availability claim; zero hardware or external operations were issued"
                ),
                "board_continuity_complete_score": 0,
            }
        )
    else:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "circular_positive",
                "honest_verdict": (
                    "complete: three authenticated board dispositions are preserved and a later "
                    "GateMate receipt enables one future action; this shared-reader result is not "
                    "a new scientific or availability claim"
                ),
                "board_continuity_complete_score": int(
                    reduced["board_disposition_complete_score"] == 1
                ),
            }
        )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "milestone",
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
        "validation_receipts",
        "repository_health",
        "field_principles",
        "board_continuity_complete_score",
        "board_rows",
        "hardware_operations_issued_count",
        "next_hardware_conditions",
    }
)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject identity, scope, operation, validation, and scoring drift."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, error: str) -> None:
        if condition:
            errors.append(error)

    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(artifact.get("field_principles") != _field_principles(), "field_principles")
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != INVOCATION_COUNTS
        or any(
            artifact.get(key) != 0
            for key in (
                "model_load_count",
                "generation_count",
                "model_invocation_count",
                "current_model_load_count",
                "current_generation_count",
            )
        ),
        "model_declaration",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    add(
        artifact.get("hardware_operations_issued_count") != 0
        or artifact.get("hardware_operations_issued") != []
        or artifact.get("external_actions") != [],
        "hardware_operations",
    )
    rows = artifact.get("board_rows")
    reduced = reduce_board_rows(rows)
    by_board = (
        {
            row.get("board"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        }
        if isinstance(rows, list)
        else {}
    )
    if rows:
        add(
            reduced["board_disposition_complete_score"] != 1
            or by_board.get("KV260", {}).get("processor_class") != "fpga_fabric"
            or by_board.get("KV260", {}).get("fabric_execution_completed") is not True
            or by_board.get("PolarFire", {}).get("processor_class") != "cpu"
            or by_board.get("PolarFire", {}).get("programmable_logic_sampling_observed")
            is not False
            or any(
                row.get("present_availability_asserted") is not False for row in by_board.values()
            ),
            "board_scope",
        )
    physical = artifact.get("physical_state_receipt")
    physical_exists = isinstance(physical, Mapping) and physical.get("exists") is True
    score = artifact.get("board_continuity_complete_score")
    verdict = artifact.get("verdict_class")
    add(
        verdict
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict",
    )
    add(verdict in {"blocked", "disqualified"} and score != 0, "blocked_score")
    add(
        score == 1
        and (
            reduced["board_disposition_complete_score"] != 1
            or not physical_exists
            or artifact.get("required_checks_passed") is not True
        ),
        "positive_score",
    )
    add(
        artifact.get("required_checks_passed") is not True and verdict != "disqualified",
        "validation_class",
    )
    if rows and not physical_exists and verdict != "disqualified":
        add(
            verdict != "blocked"
            or artifact.get("status") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith(
                "blocked_changed_physical_state:"
            ),
            "physical_state_class",
        )
    add(artifact.get("deployment_relevance") != deployment_relevance(), "deployment_relevance")
    add(artifact.get("next_hardware_conditions") != next_hardware_conditions(), "next_conditions")
    add(not isinstance(artifact.get("phase_spans"), list), "phase_spans")
    try:
        checksum_matches = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_matches = False
    add(not checksum_matches, "checksum")
    return sorted(set(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a receipt accepted by the V643 validator."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _historical_failures(receipt: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    """Carry prior repository-wide failures as dated health observations."""

    health = receipt.get("repository_health")
    rows = health.get("current_repository_wide_receipts", []) if isinstance(health, Mapping) else []
    return [
        {
            "date": receipt.get("run_date"),
            "source_experiment_id": 7314,
            "classification": "unrelated_repository_wide_failure_observation",
            "command": row.get("command"),
            "exit_code": row.get("exit_code"),
            "duration_s": row.get("elapsed_s"),
            "log_sha256": row.get("log_hash"),
            "resolved": False,
        }
        for row in rows
        if isinstance(row, Mapping) and row.get("exit_code") != 0
    ]


def _terminal_validators(
    root: Path, candidate: Path, log_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run the two required artifact validators on the measured candidate."""

    return run_commands(
        root,
        [
            CommandSpec(
                "adversarial_verify",
                (
                    str(root / ".venv/bin/python"),
                    "-u",
                    "scripts/adversarial_verify.py",
                    str(candidate),
                ),
                "measured terminal candidate",
            ),
            CommandSpec(
                "verdict_row_consistency_strict",
                (
                    str(root / ".venv/bin/python"),
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured terminal candidate",
            ),
        ],
        log_dir=log_dir,
    )


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:  # pragma: no cover
    """Run scoped checks, aggregate evidence, and validate the terminal candidate."""

    origin = time.monotonic()
    paths.validation_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        paths.checkpoint,
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "in_progress",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "terminal_artifact_path": str(paths.artifact),
        },
    )
    receipt = _read_json(root / UPSTREAM_PATH)
    print("[exp7327] phase=scoped_validation event=start", flush=True)
    validation_started = time.monotonic()
    scoped_basetemp = Path("/tmp/carnot-exp7327-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    validation = run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[ENTRYPOINT_PATH.as_posix()],
        basetemp=scoped_basetemp,
        coverage_file=paths.validation_dir / ".coverage",
        log_dir=paths.validation_dir / "scoped",
        historical_failures=_historical_failures(receipt),
    )
    validation_elapsed = time.monotonic() - validation_started
    print(
        f"[exp7327] phase=scoped_validation event=end checks={len(REQUIRED_CHECK_NAMES)} "
        f"elapsed_s={validation_elapsed:.3f}",
        flush=True,
    )
    artifact = build_artifact(root, paths, validation)
    for span in artifact["phase_spans"]:
        span["start_s"] += validation_elapsed
        span["end_s"] += validation_elapsed
    artifact["phase_spans"].insert(
        0,
        {
            "phase": "scoped_validation",
            "start_s": 0.0,
            "end_s": validation_elapsed,
            "units": len(REQUIRED_CHECK_NAMES),
            "checkpoint_boundaries": len(REQUIRED_CHECK_NAMES),
            "pending_operations": [],
        },
    )
    artifact["duration_s"] = time.monotonic() - origin
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(paths.terminal_candidate, artifact)

    print("[exp7327] phase=terminal_validators event=start", flush=True)
    terminal_started = time.monotonic()
    validators = _terminal_validators(
        root, paths.terminal_candidate, paths.validation_dir / "terminal"
    )
    terminal_elapsed = time.monotonic() - terminal_started
    artifact["validation_receipts"].extend(
        _normalized_validation({"validation_receipts": validators})
    )
    artifact["acceptance_gate_results"]["terminal_candidate_validation"] = _gate(
        {"adversarial_verify": True, "verdict_row_consistency_strict": True},
        {row["name"]: row["passed"] for row in validators},
        all(row["passed"] for row in validators),
        "Both independent terminal validators must accept the measured candidate.",
    )
    if not all(row["passed"] for row in validators):
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: terminal artifact validation failed",
                "board_continuity_complete_score": 0,
            }
        )
    artifact["phase_spans"].append(
        {
            "phase": "terminal_validators",
            "start_s": artifact["duration_s"],
            "end_s": artifact["duration_s"] + terminal_elapsed,
            "units": len(validators),
            "checkpoint_boundaries": len(validators),
            "pending_operations": [],
        }
    )
    artifact["duration_s"] = time.monotonic() - origin
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    print(
        f"[exp7327] phase=terminal_validators event=end units={len(validators)} "
        f"elapsed_s={terminal_elapsed:.3f}",
        flush=True,
    )
    write_artifact(paths.artifact, artifact)
    print(
        f"[exp7327] phase=terminal_write event=end path={paths.artifact} "
        f"verdict={artifact['verdict_class']}",
        flush=True,
    )
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover
    """Parse the fixed date and optional read-only validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Print immediately, then validate or produce one terminal receipt."""

    print("[exp7327] phase=startup event=start model_loads=0 hardware_operations=0", flush=True)
    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(_read_json(args.validate))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        raise SystemExit("--date 20260915 is required")
    artifact = run_experiment(REPO_ROOT, ExperimentPaths.defaults(REPO_ROOT))
    return int(artifact["verdict_class"] == "disqualified")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
