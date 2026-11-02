"""Produce the V644 read-only board-disposition artifact.

This reducer preserves old board claim classes and checks the one operator-only
GateMate prerequisite. It does not contact hardware or turn a completed
disposition into a readiness claim.

Spec refs: REQ-REPORT-7341 and SCENARIO-REPORT-7341-*.
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

from carnot import experiment_7327_v643_board_continuity as previous
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7341
TASK_ID = "exp7341-board-continuity"
MILESTONE = "2026.09.644"
RUN_DATE = "20260916"
SCHEMA = "carnot.experiment_7341.v644_board_continuity.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7341_v644_board_continuity.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7341_v644_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7341_v644_board_continuity")
AUTHORING_FAILURE_PATH = RAW_DIR / "validation/failed_basetemp_parent_artifact.json"
DIAGNOSTIC_PATH = Path("results/experiment_7327_v643_board_continuity.json")
GRADUATED_PATH = Path("results/experiment_7314_v642_board_continuity.json")
CUTOFF_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
MODULE_PATH = Path("python/carnot/experiment_7341_v644_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7341_v644_board_continuity.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7341_v644_board_continuity.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REFERENCE_PATH = Path("research-references.md")

PHYSICAL_RECEIPT_CONTRACT = previous.PHYSICAL_RECEIPT_CONTRACT
MISSING_RECEIPT = previous.MISSING_RECEIPT
GATEMATE_FUTURE_ACTION = previous.GATEMATE_FUTURE_ACTION
GATEMATE_OPERATOR_ACTION = previous.GATEMATE_OPERATOR_ACTION

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Record GateMate prerequisites and preserve graduated board evidence",
    "phase": 4,
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 10,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "prior_failure_ids": ["exp7327-board-continuity"],
    "prompt_sha256": "sha256:bcb228530e32ae13b00a482bf454d5c1f5f223d329c804bb2dc6d6aa5a497d76",
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
    DIAGNOSTIC_PATH,
    GRADUATED_PATH,
    CUTOFF_PATH,
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    Path("research-hardware-wishlist.md"),
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
_read_json = previous._read_json
_atomic_json = previous._atomic_json
_writable_destination = previous._writable_destination
_check = previous._check


def reduce_board_rows(rows: Any) -> JsonDict:
    """Normalize one V644 label, then delegate every check to the shipped reducer."""

    normalized = deepcopy(rows)
    if isinstance(normalized, list):
        for row in normalized:
            if (
                isinstance(row, dict)
                and row.get("disposition") == "changed_physical_state_future_experiment_eligible"
            ):
                row["disposition"] = "changed_physical_state_future_action_enabled"
                row.pop("row_sha256", None)
                previous.previous.current._finish_row(row)
    return previous.reduce_board_rows(normalized)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, the candidate, and the terminal file separate."""

    artifact: Path
    checkpoint: Path
    physical_state_search: Path
    historical_models: Path
    raw_rows: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve each task-owned output below the selected repository root."""

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
        """Give tests private destinations that cannot alter research evidence."""

        return cls.defaults(root)


def read_task_contract(root: Path) -> JsonDict | None:
    """Read only fields that fix this task's identity and prior boundary."""

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
    result["prompt_sha256"] = (
        previous.previous.sha256_text(prompt) if isinstance(prompt, str) else None
    )
    return result


def _downstream_science_gate_references(root: Path) -> list[JsonDict]:
    """Find structured later gates that would wrongly consume this task."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary.
        return [{"task_id": "unreadable_roadmap", "gate": None}]
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):
        return [{"task_id": "malformed_roadmap", "gate": None}]
    found_current = False
    references: list[JsonDict] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        if task.get("id") == TASK_ID:
            found_current = True
            continue
        if not found_current:
            continue
        gates = task.get("gated_on")
        if TASK_ID in json.dumps(gates, sort_keys=True):
            references.append({"task_id": task.get("id"), "gate": deepcopy(gates)})
    return references


def _quarantine(receipt: Mapping[str, Any], manifest: Any, experiment_id: str) -> JsonDict:
    """Use the shipped quarantine authority for both historical artifacts."""

    result = previous.previous.current.quarantine_authority._quarantine(
        receipt, manifest, experiment_id
    )
    return dict(result)


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Authenticate task identity, diagnostic history, and original evidence."""

    print("[exp7341] phase=preconditions event=start", flush=True)
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
        "requirement": "REQ-REPORT-7341" in spec_text,
        "scenarios": "SCENARIO-REPORT-7341-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7341 and scenarios",
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

    diagnostic = _read_json(root / DIAGNOSTIC_PATH)
    graduated = _read_json(root / GRADUATED_PATH)
    diagnostic_quarantine = _quarantine(diagnostic, manifest, "7327")
    graduated_quarantine = _quarantine(graduated, manifest, "7314")
    checks.append(
        _check(
            "historical_artifacts_not_quarantined",
            f"{DIAGNOSTIC_PATH.as_posix()} and {GRADUATED_PATH.as_posix()}",
            "quarantined",
            {"diagnostic": False, "graduated": False},
            {
                "diagnostic": diagnostic_quarantine.get("quarantined"),
                "graduated": graduated_quarantine.get("quarantined"),
            },
            diagnostic_quarantine.get("quarantined") is False
            and graduated_quarantine.get("quarantined") is False,
        )
    )
    expected_diagnostic = {
        "experiment_id": 7327,
        "milestone": "2026.09.643",
        "status": "blocked",
        "verdict_class": "blocked",
        "board_continuity_complete_score": 0,
    }
    observed_diagnostic = {key: diagnostic.get(key) for key in expected_diagnostic}
    checks.append(
        _check(
            "diagnostic_identity",
            DIAGNOSTIC_PATH.as_posix(),
            "canonical blocked diagnostic",
            expected_diagnostic,
            observed_diagnostic,
            observed_diagnostic == expected_diagnostic,
        )
    )
    expected_graduated = {
        "experiment_id": 7314,
        "milestone": "2026.09.642",
        "status": "complete",
        "verdict_class": "positive",
        "board_continuity_complete_score": 1,
    }
    observed_graduated = {key: graduated.get(key) for key in expected_graduated}
    checks.append(
        _check(
            "graduated_identity",
            GRADUATED_PATH.as_posix(),
            "graduated board evidence identity",
            expected_graduated,
            observed_graduated,
            observed_graduated == expected_graduated,
        )
    )
    try:
        diagnostic_checksum_matches = diagnostic.get(
            "reproducibility_checksum"
        ) == artifact_checksum(diagnostic)
        graduated_checksum_matches = graduated.get("reproducibility_checksum") == artifact_checksum(
            graduated
        )
    except (TypeError, ValueError):  # pragma: no cover - malformed historical JSON boundary.
        diagnostic_checksum_matches = False
        graduated_checksum_matches = False
    checks.append(
        _check(
            "historical_reproducibility_checksums",
            "historical board artifacts",
            "reproducibility_checksum",
            {"diagnostic": True, "graduated": True},
            {
                "diagnostic": diagnostic_checksum_matches,
                "graduated": graduated_checksum_matches,
            },
            diagnostic_checksum_matches and graduated_checksum_matches,
        )
    )
    references = previous.original_reference_observation(root, diagnostic)
    for row in references["rows"]:
        if row["passed"] is True:
            hashes[str(row["path"])] = str(row["observed_sha256"])
    checks.append(
        _check(
            "original_board_evidence_hashes",
            DIAGNOSTIC_PATH.as_posix(),
            "KV260 fabric and PolarFire CPU-dispatch references",
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
    scope = previous._scope_observation(diagnostic)
    checks.append(
        _check(
            "graduated_board_execution_scopes",
            DIAGNOSTIC_PATH.as_posix(),
            "historical processor claim classes",
            expected_scope,
            scope,
            scope == expected_scope,
        )
    )
    physical = diagnostic.get("physical_state_receipt")
    expected_cutoff_hash = (
        physical.get("cutoff_source_hash") if isinstance(physical, Mapping) else None
    )
    observed_cutoff_hash = (
        sha256_file(root / CUTOFF_PATH) if (root / CUTOFF_PATH).is_file() else None
    )
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
    references_text = (
        (root / REFERENCE_PATH).read_text(encoding="utf-8")
        if (root / REFERENCE_PATH).is_file()
        else ""
    )
    deployment = {
        "v644_scan": "V644 planning scan" in references_text,
        "extropic_z1t": "Z1T report" in references_text,
        "extropic_boundary": "No local TSU availability or measured Carnot gain follows."
        in references_text,
        "kan_deferred": "Keep KAN deployment deferred" in references_text,
    }
    checks.append(
        _check(
            "v644_deployment_context",
            REFERENCE_PATH.as_posix(),
            "Extropic and KAN remain external context",
            {key: True for key in deployment},
            deployment,
            all(deployment.values()),
        )
    )
    downstream = _downstream_science_gate_references(root)
    checks.append(
        _check(
            "no_downstream_science_gate",
            ROADMAP_PATH.as_posix(),
            "later gated_on references to Exp7341",
            [],
            downstream,
            not downstream,
        )
    )
    print(f"[exp7341] phase=preconditions event=end checks={len(checks)}", flush=True)
    return (
        checks,
        hashes,
        {
            "diagnostic_receipt": diagnostic,
            "graduated_receipt": graduated,
            "manifest": manifest,
            "diagnostic_quarantine": diagnostic_quarantine,
            "graduated_quarantine": graduated_quarantine,
            "diagnostic_checksum_matches": diagnostic_checksum_matches,
            "graduated_checksum_matches": graduated_checksum_matches,
            "diagnostic_used_as_readiness_gate": False,
            "reference_observation": references,
            "scope_observation": scope,
            "downstream_science_gate_references": downstream,
        },
    )


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep historical model-shaped fields outside the current task identity."""

    sources: list[JsonDict] = []
    for source_path in (DIAGNOSTIC_PATH, GRADUATED_PATH):
        source = _read_json(root / source_path)
        sources.append(
            {
                "path": source_path.as_posix(),
                "sha256": sha256_file(root / source_path),
                "experiment_id": source.get("experiment_id"),
                "status": source.get("status"),
                "verdict_class": source.get("verdict_class"),
                "historical_MODEL_SPECS": source.get("MODEL_SPECS"),
                "historical_model_invoked": source.get("model_invoked"),
                "historical_invocation_counts": source.get("invocation_counts"),
            }
        )
    receipt = {
        "schema": "carnot.experiment_7341.historical_model_receipts.v1",
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(INVOCATION_COUNTS),
        "sources": sources,
    }
    _atomic_json(path, receipt)
    return receipt


def search_physical_state_receipts(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Use the shipped approved-source parser and bind the V644 search."""

    result = previous.search_physical_state_receipts(
        root, raw_path, candidate_paths=candidate_paths
    )
    raw = _read_json(raw_path)
    raw.update(
        {
            "schema": "carnot.experiment_7341.gatemate_physical_state_search.v1",
            "run_date": RUN_DATE,
            "reader": "carnot.experiment_7327_v643_board_continuity.search_physical_state_receipts",
            "hardware_operations_issued": [],
            "installation_operations_issued": [],
            "procurement_operations_issued": [],
            "external_messages_issued": [],
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
            "installation_operations_issued": [],
            "procurement_operations_issued": [],
            "external_messages_issued": [],
        }
    )
    return result


def build_board_rows(diagnostic: Mapping[str, Any], physical: Mapping[str, Any]) -> list[JsonDict]:
    """Advance three diagnostic rows without broadening any hardware claim."""

    source_hash = sha256_file(REPO_ROOT / DIAGNOSTIC_PATH)
    source_rows = diagnostic.get("board_rows")
    rows = source_rows if isinstance(source_rows, list) else []
    output: list[JsonDict] = []
    changed = physical.get("exists") is True
    for source in rows:
        if not isinstance(source, Mapping):
            continue
        row = deepcopy(dict(source))
        row.pop("row_sha256", None)
        row.update(
            {
                "source_artifact_path": DIAGNOSTIC_PATH.as_posix(),
                "source_artifact_hash": source_hash,
                "source_terminal_class": diagnostic.get("verdict_class"),
                "source_used_as_readiness_gate": False,
                "present_availability_asserted": False,
                "current_evidence": "read_only_artifact_aggregation",
                "hardware_operations_issued": [],
                "hardware_command_count": 0,
                "installation_operations_issued": [],
                "procurement_operations_issued": [],
                "external_messages_issued": [],
            }
        )
        if row.get("board") == "GateMate":
            row.update(
                {
                    "processor_class": "not_executed" if changed else "unavailable",
                    "latest_receipt_path": physical.get("search_receipt_path"),
                    "latest_receipt_date": RUN_DATE,
                    "latest_receipt_hash": physical.get("search_receipt_hash"),
                    "latest_receipt_authenticated": True,
                    "operator_source_path": physical.get("source_path"),
                    "operator_author_evidence": physical.get("author_evidence"),
                    "operator_date_evidence": physical.get("date_evidence"),
                    "operator_evidence_hash": physical.get("evidence_hash"),
                    "operator_changed_conditions": deepcopy(physical.get("changed_conditions", {})),
                    "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                    "physical_receipt_contract": PHYSICAL_RECEIPT_CONTRACT,
                    "observed_state": (
                        "operator_changed_physical_state_recorded"
                        if changed
                        else "operator_changed_physical_state_receipt_missing"
                    ),
                    "observed_missing_receipt": physical.get("observed_missing_receipt"),
                    "disposition": (
                        "changed_physical_state_future_experiment_eligible"
                        if changed
                        else "blocked_changed_physical_state"
                    ),
                    "exact_next_condition": (
                        GATEMATE_FUTURE_ACTION if changed else GATEMATE_OPERATOR_ACTION
                    ),
                    "failed_value": None if changed else MISSING_RECEIPT,
                    "metric": changed,
                    "error": None if changed else MISSING_RECEIPT,
                    "abstention": not changed,
                }
            )
        output.append(previous.previous.current._finish_row(row))
    return output


def deployment_relevance() -> JsonDict:
    """Keep V644 vendor and KAN references outside Carnot measurements."""

    return {
        "scan": "V644",
        "source_path": REFERENCE_PATH.as_posix(),
        "extropic": {
            "context": "Z1T sparse computation split between Z1 and FPGA",
            "projected_vendor_efficiency": "external_projection_only",
            "carnot_speedup_claimed": False,
            "local_tsu_authority": False,
            "local_device_access": False,
            "required_future_evidence": "transfer, readout, latency, power, and samples",
        },
        "kan": {
            "context": "future deployment option for a useful learned predictor",
            "replacement_authorized": False,
            "current_exact_integer_constraints_need_replacement": False,
            "required_future_evidence": "useful predictor and matched benchmark",
        },
        "purchase_required": False,
        "procurement_operations_issued": [],
        "vendor_contacts_issued": [],
    }


def hardware_path(*, physical_exists: bool) -> JsonDict:
    """State the bounded path without implying that this task used a board."""

    return {
        "constraint_state": "acquired_exact_integer_constraints",
        "current_execution": "host_read_only_aggregation",
        "kv260_historical_scope": "authenticated_fpga_fabric_execution",
        "kv260_future_access": "ssh kria only",
        "polarfire_scope": "cpu_dispatch_not_fpga_sampling",
        "gatemate_state": (
            "future_experiment_eligible_from_dated_operator_receipt"
            if physical_exists
            else "blocked_pending_dated_operator_physical_change"
        ),
        "extropic_and_kan": "external_deployment_context_only",
        "purchase_required_this_milestone": False,
    }


def next_hardware_conditions() -> JsonDict:
    """Name the authority needed before any later hardware experiment."""

    return {
        "gatemate": {
            "condition": GATEMATE_OPERATOR_ACTION,
            "satisfied_by": "operator",
            "future_action_if_satisfied": GATEMATE_FUTURE_ACTION,
            "current_task_authority": "read_only_eligibility_record_only",
        },
        "kv260": {
            "condition": "any future access uses ssh kria only",
            "satisfied_by": "future authorized hardware experiment",
            "present_availability_claimed": False,
        },
        "polarfire": {
            "condition": "FPGA sampling requires a separate authorized fabric experiment",
            "satisfied_by": "future authorized hardware experiment",
            "cpu_dispatch_already_preserved": True,
        },
        "purchase": {
            "condition": "none for milestone 2026.09.644",
            "required": False,
        },
    }


def _field_principles() -> JsonDict:
    """Preserve each requested principle beside ordinary executable fields."""

    return {
        "schema": "Version the artifact while preserving ordinary top-level experiment_id and milestone.",
        "status": "Publish a terminal result only after current work and affected validation.",
        "run_date": "Use 20260916 and retain actual UTC timestamps.",
        "preconditions_checked": "Name input identity, availability and each failed check before work.",
        "MODEL_SPECS": "List actual intended/current executable identities; any LLM task includes unsloth/Qwen3.8-27B-GGUF.",
        "model_invoked": "True when any real load or generation is attempted, even if no usable answer arrives.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight loads and generations.",
        "inference_substrate": "Declare actual computation; historical model evidence is not a current invocation.",
        "inference_substrate_class": "Declare the real duration class; small fixed-token runs are model_bounded_generation.",
        "execution_venue": "Use host; old board receipts never imply current board execution.",
        "duration_s": "Measure real monotonic elapsed time; never pad a duration floor.",
        "phase_spans": "Disjoint stage spans, completed units, checkpoint positions and pending operations explain cost.",
        "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, public inputs, evaluator identity and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers rather than similarly named older results.",
        "rows": "Every comparative unit and arm carries metrics, costs, abstentions, failures and censoring.",
        "sample_size_budget": "Record planned, attempted, completed and censored units with a fixed stopping rule.",
        "acceptance_gate_results": "Each gate records expected, observed and passed; separate completion from value.",
        "gate_check_summary": "Every blocked_* must identify upstream, failed check, artifact field, expected and observed value.",
        "verifier_is_oracle": "True whenever the execution authority defines correctness; independent code alone does not remove circularity.",
        "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_ and names the check.",
        "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; external absence is blocked.",
        "validation_receipts": "Record exact command, affected scope, exit code, duration and log hash including failures.",
        "repository_health": "Keep unrelated repository failures as dated observations, separate from affected required checks.",
        "field_principles": "Explain each field without wrapping executable scores or ordinary dictionaries.",
        "board_disposition_complete_score": "One means three complete dispositions, including a known external block; it is not hardware readiness.",
        "board_rows": "Preserve original terminal claim classes and exact reopening conditions.",
        "hardware_operations_issued_count": "Zero is required for this read-only task.",
        "next_hardware_conditions": "Name the dated physical-state receipt and future SSH-only access conditions.",
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Give each acceptance gate one explicit comparison shape."""

    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def _phase_span(name: str, start: float, origin: float, units: int) -> JsonDict:
    """Close one disjoint monotonic phase with completed-unit accounting."""

    end = time.monotonic()
    print(f"[exp7341] phase={name} event=end units={units} elapsed_s={end - start:.3f}", flush=True)
    return {
        "phase": name,
        "start_s": start - origin,
        "end_s": end - origin,
        "completed_units": units,
        "checkpoint_positions": [units],
        "pending_operations": [],
    }


def _normalized_validation(validation: Mapping[str, Any]) -> list[JsonDict]:
    """Retain runner fields and add the required receipt aliases."""

    output: list[JsonDict] = []
    receipts = validation.get("validation_receipts")
    for source in receipts if isinstance(receipts, list) else []:
        if not isinstance(source, Mapping):
            continue
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
    """Retain each exact failed check without hiding the expected block."""

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
    if physical and physical.get("exists") is not True:
        failures.append(
            {
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
        )
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
    """Create all required fields before a terminal outcome is selected."""

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
            "development": 7341001,
            "evaluation": 7341002,
            "resampling": 7341003,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial: board dispositions have not completed",
        "verdict_class": "partial",
        "validation_receipts": [],
        "required_checks_passed": False,
        "missing_required_commands": list(REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {},
        "field_principles": _field_principles(),
        "board_disposition_complete_score": 0,
        "hardware_readiness_score": 0,
        "hardware_promotion_score": 0,
        "board_execution_promotion_score": 0,
        "scientific_value_score": 0,
        "board_rows": [],
        "hardware_operations_issued_count": 0,
        "hardware_operations_issued": [],
        "ssh_operations_issued": [],
        "usb_operations_issued": [],
        "jtag_operations_issued": [],
        "flash_operations_issued": [],
        "installation_operations_issued": [],
        "procurement_operations_issued": [],
        "external_messages_issued": [],
        "physical_state_receipt": {},
        "deployment_relevance": deployment_relevance(),
        "purchase_required": False,
        "hardware_path": hardware_path(physical_exists=False),
        "next_hardware_conditions": next_hardware_conditions(),
        "source_artifact_states": {},
        "downstream_science_gate_references": [],
    }


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    validation: Mapping[str, Any],
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate immutable evidence and return a terminal in-memory receipt."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    phase_start = time.monotonic()
    checks, hashes, context = collect_preconditions(root, paths)
    spans.append(_phase_span("preconditions", phase_start, origin, len(checks)))
    preconditions_passed = all(row["passed"] is True for row in checks)

    physical: JsonDict = {}
    rows: list[JsonDict] = []
    if preconditions_passed:
        print("[exp7341] phase=evidence event=start", flush=True)
        phase_start = time.monotonic()
        write_historical_model_receipt(root, paths.historical_models)
        physical = search_physical_state_receipts(
            root, paths.physical_state_search, candidate_paths=candidate_paths
        )
        rows = build_board_rows(context["diagnostic_receipt"], physical)
        _atomic_json(paths.raw_rows, {"schema": SCHEMA + ".raw_rows", "rows": rows})
        for path in (paths.historical_models, paths.physical_state_search, paths.raw_rows):
            hashes[str(path)] = sha256_file(path)
        spans.append(_phase_span("evidence", phase_start, origin, len(rows)))

    artifact = _base_artifact(checks, hashes, spans)
    reduced = reduce_board_rows(rows)
    validation_passed = validation.get("required_checks_passed") is True
    physical_exists = physical.get("exists") is True
    summary = _summary(checks, physical, validation)
    disposition_complete = reduced["board_disposition_complete_score"] == 1
    artifact.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": time.monotonic() - origin,
            "rows": rows,
            "board_rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": len(rows),
                "completed": len(rows),
                "censored": 0,
                "stopping_rule": "one authenticated read-only disposition for each named board",
            },
            "acceptance_gate_results": {
                "historical_sources_authenticated": _gate(
                    True,
                    preconditions_passed,
                    preconditions_passed,
                    "Historical artifacts and original evidence must hash-match.",
                ),
                "diagnostic_not_consumed_as_readiness": _gate(
                    False,
                    context.get("diagnostic_used_as_readiness_gate"),
                    context.get("diagnostic_used_as_readiness_gate") is False,
                    "The blocked Exp7327 score cannot authorize current work.",
                ),
                "three_board_dispositions_complete": _gate(
                    {"rows": 3, "score": 1},
                    {
                        "rows": reduced["board_count"],
                        "score": reduced["board_disposition_complete_score"],
                    },
                    disposition_complete,
                    "A known external block is a complete disposition, not readiness.",
                ),
                "gatemate_changed_physical_state_receipt": _gate(
                    PHYSICAL_RECEIPT_CONTRACT,
                    {
                        "exists": physical_exists,
                        "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                        "observed_missing_receipt": physical.get("observed_missing_receipt"),
                    },
                    physical_exists,
                    "Only a dated operator physical change enables a future experiment.",
                ),
                "required_scoped_validation": _gate(
                    True,
                    validation.get("required_checks_passed"),
                    validation_passed,
                    "All affected tests, coverage, lint, format, type, and spec checks must pass.",
                ),
                "read_only_operation_boundary": _gate(
                    {"hardware": 0, "installation": 0, "procurement": 0, "messages": 0},
                    {"hardware": 0, "installation": 0, "procurement": 0, "messages": 0},
                    True,
                    "This task records eligibility and cannot issue an operation.",
                ),
                "no_downstream_science_gate": _gate(
                    [],
                    context.get("downstream_science_gate_references", []),
                    context.get("downstream_science_gate_references", []) == [],
                    "Board disposition cannot block unrelated science.",
                ),
            },
            "gate_check_summary": summary,
            "validation_receipts": _normalized_validation(validation),
            "required_checks_passed": validation_passed,
            "missing_required_commands": deepcopy(validation.get("missing_required_commands", [])),
            "failed_required_commands": deepcopy(validation.get("failed_required_commands", [])),
            "duplicate_required_commands": deepcopy(
                validation.get("duplicate_required_commands", [])
            ),
            "repository_health": deepcopy(validation.get("repository_health", {})),
            "physical_state_receipt": physical,
            "hardware_path": hardware_path(physical_exists=physical_exists),
            "downstream_science_gate_references": deepcopy(
                context.get("downstream_science_gate_references", [])
            ),
            "source_artifact_states": {
                DIAGNOSTIC_PATH.as_posix(): {
                    "producer_experiment_id": context.get("diagnostic_receipt", {}).get(
                        "experiment_id"
                    ),
                    "terminal_status": context.get("diagnostic_receipt", {}).get("status"),
                    "terminal_class": context.get("diagnostic_receipt", {}).get("verdict_class"),
                    "quarantine": deepcopy(context.get("diagnostic_quarantine", {})),
                    "checksum_matches": context.get("diagnostic_checksum_matches"),
                    "used_as_readiness_gate": False,
                },
                GRADUATED_PATH.as_posix(): {
                    "producer_experiment_id": context.get("graduated_receipt", {}).get(
                        "experiment_id"
                    ),
                    "terminal_status": context.get("graduated_receipt", {}).get("status"),
                    "terminal_class": context.get("graduated_receipt", {}).get("verdict_class"),
                    "quarantine": deepcopy(context.get("graduated_quarantine", {})),
                    "checksum_matches": context.get("graduated_checksum_matches"),
                    "used_as_readiness_gate": False,
                },
            },
        }
    )

    if not validation_passed:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: affected scoped validation failed",
                "board_disposition_complete_score": 0,
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
                "board_disposition_complete_score": 0,
            }
        )
    elif not physical_exists:
        artifact.update(
            {
                "status": "blocked",
                "verdict_class": "blocked",
                "honest_verdict": (
                    "blocked_changed_physical_state: no operator-authored dated GateMate cable, "
                    "port, board, power, JTAG, or DirtyJTAG change exists after Exp6559; three "
                    "authenticated dispositions are complete, but hardware readiness and "
                    "promotion remain zero; zero hardware or external operations were issued"
                ),
                "board_disposition_complete_score": int(disposition_complete),
            }
        )
    else:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "circular_positive",
                "honest_verdict": (
                    "complete: three authenticated board dispositions are recorded and the "
                    "dated GateMate receipt makes a separate future experiment eligible; no "
                    "hardware, purchase, or external operation was issued"
                ),
                "board_disposition_complete_score": int(disposition_complete),
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
        "board_disposition_complete_score",
        "board_rows",
        "hardware_operations_issued_count",
        "next_hardware_conditions",
    }
)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject identity, scope, operation, score, and checksum drift."""

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
    operation_fields = (
        "hardware_operations_issued",
        "ssh_operations_issued",
        "usb_operations_issued",
        "jtag_operations_issued",
        "flash_operations_issued",
        "installation_operations_issued",
        "procurement_operations_issued",
        "external_messages_issued",
    )
    add(
        artifact.get("hardware_operations_issued_count") != 0
        or any(artifact.get(key) != [] for key in operation_fields),
        "operations",
    )
    rows_value = artifact.get("board_rows")
    rows = rows_value if isinstance(rows_value, list) else []
    reduced = reduce_board_rows(rows)
    by_board = {
        row.get("board"): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("board"), str)
    }
    add(artifact.get("rows") != rows_value, "rows_alias")
    if rows:
        add(
            reduced["board_disposition_complete_score"] != 1
            or set(by_board) != {"KV260", "GateMate", "PolarFire"}
            or by_board.get("KV260", {}).get("processor_class") != "fpga_fabric"
            or by_board.get("KV260", {}).get("fabric_execution_completed") is not True
            or by_board.get("PolarFire", {}).get("processor_class") != "cpu"
            or by_board.get("PolarFire", {}).get("programmable_logic_sampling_observed")
            is not False
            or any(
                row.get("present_availability_asserted") is not False
                or row.get("hardware_operations_issued") != []
                for row in by_board.values()
            ),
            "board_scope",
        )
    score = artifact.get("board_disposition_complete_score")
    verdict = artifact.get("verdict_class")
    expected_score = int(bool(rows) and reduced["board_disposition_complete_score"] == 1)
    if verdict == "disqualified":
        expected_score = 0
    add(score != expected_score, "disposition_score")
    add(
        any(
            artifact.get(key) != 0
            for key in (
                "hardware_readiness_score",
                "hardware_promotion_score",
                "board_execution_promotion_score",
                "scientific_value_score",
            )
        ),
        "readiness_or_promotion",
    )
    add(
        verdict
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict",
    )
    add(
        artifact.get("required_checks_passed") is not True and verdict != "disqualified",
        "validation_class",
    )
    physical = artifact.get("physical_state_receipt")
    physical_exists = isinstance(physical, Mapping) and physical.get("exists") is True
    if rows and not physical_exists and verdict != "disqualified":
        add(
            verdict != "blocked"
            or artifact.get("status") != "blocked"
            or score != 1
            or not str(artifact.get("honest_verdict", "")).startswith(
                "blocked_changed_physical_state:"
            ),
            "physical_state_class",
        )
    states = artifact.get("source_artifact_states")
    diagnostic_state = (
        states.get(DIAGNOSTIC_PATH.as_posix(), {}) if isinstance(states, Mapping) else {}
    )
    if rows:
        add(diagnostic_state.get("used_as_readiness_gate") is not False, "diagnostic_gate")
    add(artifact.get("deployment_relevance") != deployment_relevance(), "deployment_context")
    add(
        artifact.get("hardware_path") != hardware_path(physical_exists=physical_exists),
        "hardware_path",
    )
    add(artifact.get("next_hardware_conditions") != next_hardware_conditions(), "next_conditions")
    add(artifact.get("purchase_required") is not False, "purchase")
    add(artifact.get("downstream_science_gate_references") != [], "downstream_gate")
    add(not isinstance(artifact.get("phase_spans"), list), "phase_spans")
    try:
        checksum_matches = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_matches = False
    add(not checksum_matches, "checksum")
    return sorted(set(errors))


def cold_validate_candidate(paths: ExperimentPaths) -> list[str]:
    """Reload candidate and raw rows so in-memory state cannot hide drift."""

    candidate = _read_json(paths.terminal_candidate)
    raw = _read_json(paths.raw_rows)
    raw_rows = raw.get("rows") if isinstance(raw, Mapping) else None
    errors = validate_artifact(candidate) if candidate else ["candidate_not_json"]
    if candidate.get("board_rows") != raw_rows:
        errors.append("candidate_rows_mismatch")
    reduced = reduce_board_rows(raw_rows)
    if reduced["board_disposition_complete_score"] != 1:
        errors.append("raw_row_reduction")
    return sorted(set(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Atomically publish only a receipt accepted by the V644 validator."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _historical_failures(
    root: Path, receipt: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover
    """Carry prior repository failures as observations, not current checks."""

    health = receipt.get("repository_health")
    rows = health.get("historical_failures", []) if isinstance(health, Mapping) else []
    failures = [deepcopy(dict(row)) for row in rows if isinstance(row, Mapping)]
    authoring_path = root / AUTHORING_FAILURE_PATH
    authoring = _read_json(authoring_path)
    if authoring:
        receipts = authoring.get("validation_receipts")
        failed_receipts = (
            [
                deepcopy(dict(row))
                for row in receipts
                if isinstance(row, Mapping) and row.get("exit_code") != 0
            ]
            if isinstance(receipts, list)
            else []
        )
        failures.append(
            {
                "date": RUN_DATE,
                "classification": "affected_authoring_failure_corrected",
                "cause": "private pytest basetemp parent was absent",
                "source_path": AUTHORING_FAILURE_PATH.as_posix(),
                "source_sha256": sha256_file(authoring_path),
                "validation_receipts": failed_receipts,
                "resolved": True,
            }
        )
    return failures


def _terminal_validators(
    root: Path, candidate: Path, log_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run both required independent validators on the measured candidate."""

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
    """Run affected checks, aggregate real files, and publish after validation."""

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
    diagnostic = _read_json(root / DIAGNOSTIC_PATH)
    print("[exp7341] phase=scoped_validation event=start", flush=True)
    validation_started = time.monotonic()
    scoped_basetemp = Path("/tmp/carnot-exp7341-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    validation = run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[ENTRYPOINT_PATH.as_posix()],
        basetemp=scoped_basetemp,
        coverage_file=paths.validation_dir / ".coverage_final",
        log_dir=paths.validation_dir / "scoped_final",
        historical_failures=_historical_failures(root, diagnostic),
    )
    validation_elapsed = time.monotonic() - validation_started
    print(
        f"[exp7341] phase=scoped_validation event=end checks={len(REQUIRED_CHECK_NAMES)} "
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
            "completed_units": len(REQUIRED_CHECK_NAMES),
            "checkpoint_positions": list(range(1, len(REQUIRED_CHECK_NAMES) + 1)),
            "pending_operations": [],
        },
    )
    artifact["duration_s"] = time.monotonic() - origin
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(paths.terminal_candidate, artifact)

    print("[exp7341] phase=cold_candidate_validation event=start", flush=True)
    cold_started = time.monotonic()
    cold_errors = cold_validate_candidate(paths)
    cold_elapsed = time.monotonic() - cold_started
    print(
        f"[exp7341] phase=cold_candidate_validation event=end errors={len(cold_errors)} "
        f"elapsed_s={cold_elapsed:.3f}",
        flush=True,
    )
    artifact["acceptance_gate_results"]["cold_candidate_reduction"] = _gate(
        [],
        cold_errors,
        not cold_errors,
        "Reloaded candidate bytes and independently reduced raw rows must agree.",
    )

    print("[exp7341] phase=terminal_validators event=start", flush=True)
    terminal_started = time.monotonic()
    validators = _terminal_validators(
        root, paths.terminal_candidate, paths.validation_dir / "terminal"
    )
    terminal_elapsed = time.monotonic() - terminal_started
    artifact["validation_receipts"].extend(
        _normalized_validation({"validation_receipts": validators})
    )
    validators_passed = all(row["passed"] for row in validators)
    artifact["acceptance_gate_results"]["terminal_candidate_validation"] = _gate(
        {"adversarial_verify": True, "verdict_row_consistency_strict": True},
        {row["name"]: row["passed"] for row in validators},
        validators_passed,
        "Both independent terminal validators must accept the measured candidate.",
    )
    if cold_errors or not validators_passed:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: terminal artifact validation failed",
                "board_disposition_complete_score": 0,
            }
        )
    artifact["phase_spans"].extend(
        [
            {
                "phase": "cold_candidate_validation",
                "start_s": artifact["duration_s"],
                "end_s": artifact["duration_s"] + cold_elapsed,
                "completed_units": 1,
                "checkpoint_positions": [1],
                "pending_operations": [],
            },
            {
                "phase": "terminal_validators",
                "start_s": artifact["duration_s"] + cold_elapsed,
                "end_s": artifact["duration_s"] + cold_elapsed + terminal_elapsed,
                "completed_units": len(validators),
                "checkpoint_positions": list(range(1, len(validators) + 1)),
                "pending_operations": [],
            },
        ]
    )
    artifact["duration_s"] = time.monotonic() - origin
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    print(
        f"[exp7341] phase=terminal_validators event=end units={len(validators)} "
        f"elapsed_s={terminal_elapsed:.3f}",
        flush=True,
    )
    write_artifact(paths.artifact, artifact)
    print(
        f"[exp7341] phase=terminal_write event=end path={paths.artifact} "
        f"verdict={artifact['verdict_class']}",
        flush=True,
    )
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover
    """Parse the fixed execution date or a read-only validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or validate an existing artifact without mutation."""

    print("[exp7341] phase=entry event=start", flush=True)
    args = _parser().parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(_read_json(args.validate))
        print(json.dumps({"path": str(args.validate), "errors": errors}, sort_keys=True))
        return int(bool(errors))
    if args.date != RUN_DATE:
        raise SystemExit(f"--date {RUN_DATE} is required")
    artifact = run_experiment(REPO_ROOT, ExperimentPaths.defaults())
    return int(artifact["verdict_class"] == "disqualified")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
