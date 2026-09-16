"""Produce the V645 read-only board-state artifact.

This adapter advances the authenticated V644 rows through the existing receipt
parser. It records current evidence without contacting any board or claiming
that historical availability continues today.

Spec refs: REQ-REPORT-7355 and SCENARIO-REPORT-7355-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_7341_v644_board_continuity as previous
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7355
TASK_ID = "exp7355-board-state"
MILESTONE = "2026.09.645"
RUN_DATE = "20260916"
SCHEMA = "carnot.experiment_7355.v645_board_state.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7355_v645_board_state.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7355_v645_board_state.json")
RAW_DIR = Path("results/raw/experiment_7355_v645_board_state")
PRIOR_PATH = Path("results/experiment_7341_v644_board_continuity.json")
CUTOFF_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
MODULE_PATH = Path("python/carnot/experiment_7355_v645_board_state.py")
TEST_PATH = Path("tests/python/test_experiment_7355_v645_board_state.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7355_v645_board_state.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REFERENCE_PATH = Path("research-references.md")

PHYSICAL_RECEIPT_CONTRACT = previous.PHYSICAL_RECEIPT_CONTRACT
MISSING_RECEIPT = previous.MISSING_RECEIPT
GATEMATE_FUTURE_ACTION = previous.GATEMATE_FUTURE_ACTION
GATEMATE_OPERATOR_ACTION = previous.GATEMATE_OPERATOR_ACTION
INVOCATION_COUNTS = deepcopy(previous.INVOCATION_COUNTS)
ZERO_SCORE_FIELDS = (
    "hardware_readiness_score",
    "hardware_value_score",
    "hardware_promotion_score",
    "board_execution_promotion_score",
    "scientific_value_score",
)

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Record GateMate changed-state prerequisites and graduated board evidence",
    "phase": 4,
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 10,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "prior_failure_ids": ["exp7341-board-continuity"],
    "prompt_sha256": "sha256:1c55b1449c085d8995b96238c3c744c9178d366294636fbbbd07d58463e75cab",
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
    Path("python/carnot/experiment_7341_v644_board_continuity.py"),
    PRIOR_PATH,
    CUTOFF_PATH,
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/operator-followup.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    ROADMAP_PATH,
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

sha256_file = previous.sha256_file
artifact_checksum = previous.artifact_checksum
reduce_board_rows = previous.reduce_board_rows
_read_json = previous._read_json
_atomic_json = previous._atomic_json
_writable_destination = previous._writable_destination
_check = previous._check
_finish_row = previous.previous.previous.current._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw evidence, validation logs, and terminal output separate."""

    artifact: Path
    checkpoint: Path
    physical_state_search: Path
    historical_models: Path
    raw_rows: Path
    terminal_candidate: Path
    validation_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve each task-owned output below one repository root."""

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
    """Read only roadmap fields that fix this task and its prior boundary."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary.
        return None
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):  # pragma: no cover - malformed contract boundary.
        return None
    task = next(
        (row for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:  # pragma: no cover - missing contract boundary.
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
        "sha256:" + hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt, str) else None
    )
    return result


def _downstream_science_gate_references(root: Path) -> list[JsonDict]:
    """Find later structured gates that would wrongly consume this disposition."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary.
        return [{"task_id": "unreadable_roadmap", "gate": None}]
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):  # pragma: no cover - malformed contract boundary.
        return [{"task_id": "malformed_roadmap", "gate": None}]
    found_current = False
    references: list[JsonDict] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        if task.get("id") == TASK_ID:
            found_current = True
            continue
        if found_current and TASK_ID in json.dumps(task.get("gated_on"), sort_keys=True):
            references.append({"task_id": task.get("id"), "gate": deepcopy(task.get("gated_on"))})
    return references


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Authenticate the current contract, Exp7341, its rows, and the cutoff."""

    print("[exp7355] phase=preconditions event=start", flush=True)
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
        "requirement": "REQ-REPORT-7355" in spec_text,
        "scenarios": "SCENARIO-REPORT-7355-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7355 and scenarios",
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

    prior = _read_json(root / PRIOR_PATH)
    quarantine = previous._quarantine(prior, manifest, "7341")
    checks.append(
        _check(
            "prior_artifact_not_quarantined",
            PRIOR_PATH.as_posix(),
            "quarantined",
            False,
            quarantine.get("quarantined"),
            quarantine.get("quarantined") is False,
        )
    )
    expected_prior = {
        "experiment_id": 7341,
        "milestone": "2026.09.644",
        "status": "blocked",
        "verdict_class": "blocked",
        "board_disposition_complete_score": 1,
    }
    observed_prior = {key: prior.get(key) for key in expected_prior}
    prior_validation_errors = previous.validate_artifact(prior) if prior else ["missing"]
    checks.append(
        _check(
            "prior_terminal_identity",
            PRIOR_PATH.as_posix(),
            "diagnostic-only terminal artifact",
            {"identity": expected_prior, "validation_errors": []},
            {"identity": observed_prior, "validation_errors": prior_validation_errors},
            observed_prior == expected_prior and not prior_validation_errors,
        )
    )
    reference_observation = previous.previous.original_reference_observation(root, prior)
    for row in reference_observation["rows"]:
        if row["passed"] is True:
            hashes[str(row["path"])] = str(row["observed_sha256"])
    checks.append(
        _check(
            "original_board_evidence_hashes",
            PRIOR_PATH.as_posix(),
            "KV260 fabric and PolarFire CPU-dispatch references",
            "all current bytes hash-match",
            reference_observation,
            reference_observation["all_match"] is True,
        )
    )
    expected_scope = {
        "kv260_fabric_execution": True,
        "kv260_processor_class": "fpga_fabric",
        "polarfire_cpu_dispatch": True,
        "polarfire_processor_class": "cpu",
        "polarfire_fpga_sampling": False,
    }
    scope = previous.previous._scope_observation(prior)
    checks.append(
        _check(
            "graduated_board_execution_scopes",
            PRIOR_PATH.as_posix(),
            "historical processor claim classes",
            expected_scope,
            scope,
            scope == expected_scope,
        )
    )
    physical = prior.get("physical_state_receipt")
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
    reference_text = (
        (root / REFERENCE_PATH).read_text(encoding="utf-8")
        if (root / REFERENCE_PATH).is_file()
        else ""
    )
    deployment = {
        "v645_scan": "## V645 planning scan" in reference_text,
        "extropic_z1t": "Z1T report" in reference_text,
        "extropic_projection_limit": "estimates as projections" in reference_text,
        "kan_context": "No new KAN branch before a useful learned predictor exists."
        in reference_text,
    }
    checks.append(
        _check(
            "v645_deployment_context",
            REFERENCE_PATH.as_posix(),
            "Extropic Z1T and KAN remain deployment context",
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
            "later gated_on references to Exp7355",
            [],
            downstream,
            not downstream,
        )
    )
    print(f"[exp7355] phase=preconditions event=end checks={len(checks)}", flush=True)
    return (
        checks,
        hashes,
        {
            "prior_receipt": prior,
            "prior_quarantine": quarantine,
            "prior_validation_errors": prior_validation_errors,
            "prior_used_as_readiness_gate": False,
            "reference_observation": reference_observation,
            "scope_observation": scope,
            "downstream_science_gate_references": downstream,
        },
    )


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep historical model-shaped fields outside the current task identity."""

    source = _read_json(root / PRIOR_PATH)
    receipt = {
        "schema": "carnot.experiment_7355.historical_model_receipts.v1",
        "current_MODEL_SPECS": [],
        "current_model_invoked": False,
        "current_invocation_counts": deepcopy(INVOCATION_COUNTS),
        "source": {
            "path": PRIOR_PATH.as_posix(),
            "sha256": sha256_file(root / PRIOR_PATH),
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
    """Run the shipped approved-source parser and bind the V645 search."""

    result = previous.search_physical_state_receipts(
        root, raw_path, candidate_paths=candidate_paths
    )
    raw = _read_json(raw_path)
    raw.update(
        {
            "schema": "carnot.experiment_7355.gatemate_physical_state_search.v1",
            "run_date": RUN_DATE,
            "reader": "carnot.experiment_7341_v644_board_continuity.search_physical_state_receipts",
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


def build_board_rows(
    root: Path, prior: Mapping[str, Any], physical: Mapping[str, Any]
) -> list[JsonDict]:
    """Advance the three authenticated rows without broadening their claims."""

    source_rows = prior.get("board_rows")
    rows = source_rows if isinstance(source_rows, list) else []
    source_hash = sha256_file(root / PRIOR_PATH)
    changed = physical.get("exists") is True
    output: list[JsonDict] = []
    for source in rows:
        if not isinstance(source, Mapping):
            continue
        row = deepcopy(dict(source))
        row.pop("row_sha256", None)
        row.update(
            {
                "source_artifact_path": PRIOR_PATH.as_posix(),
                "source_artifact_hash": source_hash,
                "source_terminal_class": prior.get("verdict_class"),
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
        output.append(_finish_row(row))
    return output


def deployment_relevance() -> JsonDict:
    """Keep V645 Extropic and KAN references outside Carnot measurements."""

    return {
        "scan": "V645",
        "source_path": REFERENCE_PATH.as_posix(),
        "extropic": {
            "device": "Z1T",
            "context": "sparse computation and projected chip energy",
            "projected_vendor_efficiency": "external_projection_only",
            "dense_readout_included": False,
            "vocabulary_logits_included": False,
            "carnot_speedup_claimed": False,
            "local_device_access": False,
        },
        "kan": {
            "context": "future continual classifier after a useful predictor exists",
            "deployment_context_only": True,
            "replacement_authorized": False,
            "scientific_result": False,
        },
        "purchase_required": False,
        "procurement_operations_issued": [],
        "vendor_contacts_issued": [],
    }


def hardware_path(*, physical_exists: bool) -> JsonDict:
    """State the bounded future path without claiming present board access."""

    return {
        "current_execution": "host_read_only_aggregation",
        "kv260_historical_scope": "authenticated_fpga_fabric_execution",
        "kv260_future_access": "ssh kria only",
        "polarfire_scope": "cpu_dispatch_not_fpga_sampling",
        "gatemate_state": (
            "future_experiment_eligible_from_dated_operator_receipt"
            if physical_exists
            else "blocked_pending_dated_operator_physical_change"
        ),
        "extropic_z1t_and_kan": "external_deployment_context_only",
        "purchase_required_this_milestone": False,
    }


def next_hardware_conditions() -> JsonDict:
    """Name the authority required before any later hardware operation."""

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
        "purchase": {"condition": "none for milestone 2026.09.645", "required": False},
    }


def _field_principles() -> JsonDict:
    """Explain required fields while leaving executable values unwrapped."""

    return {
        "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
        "status": "Write a terminal result only after actual work and affected checks.",
        "run_date": "Use 20260916 and record real UTC timestamps as well.",
        "preconditions_checked": "Record each actual input and resource check before dependent work.",
        "MODEL_SPECS": "List actual intended model identities; this task has no current model.",
        "model_invoked": "True for any attempted current model load or generation, including failures.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
        "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
        "inference_substrate_class": "Use the closed duration class that matches the actual run.",
        "execution_venue": "Use host; this milestone makes no new board-execution claim.",
        "duration_s": "Measure monotonic time and never wait to pass a duration floor.",
        "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
        "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
        "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
        "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
        "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
        "gate_check_summary": "Every block names upstream, failed check, exact field, expected and observed.",
        "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
        "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_.",
        "verdict_class": "Use positive, circular_positive, null, blocked, disqualified or partial.",
        "flagged_adversarial": "False only after current verification; a critical finding prevents promotion.",
        "validation_receipts": "Retain command, scope, exit code, elapsed time and log hash, including failures.",
        "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
        "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
        "board_disposition_complete_score": "Accounting completeness does not assert board availability.",
        "board_rows": "Keep fabric, CPU dispatch and absent physical change distinct.",
        "hardware_readiness_score": "Zero because this task performs no current hardware operation.",
        "hardware_value_score": "Zero because historical evidence is not a new speed result.",
    }


def _phase_span(name: str, start: float, origin: float, units: int) -> JsonDict:
    """Close one disjoint monotonic phase and report completed units."""

    end = time.monotonic()
    print(f"[exp7355] phase={name} event=end units={units} elapsed_s={end - start:.3f}", flush=True)
    return {
        "phase": name,
        "start_s": start - origin,
        "end_s": end - origin,
        "completed_units": units,
        "checkpoint_positions": [units],
        "pending_operations": [],
    }


def _summary(
    checks: Sequence[Mapping[str, Any]],
    physical: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Keep every failed check, including the expected external block."""

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


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    validation: Mapping[str, Any],
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate immutable evidence into one terminal in-memory record."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    phase_start = time.monotonic()
    checks, hashes, context = collect_preconditions(root, paths)
    spans.append(_phase_span("preconditions", phase_start, origin, len(checks)))
    preconditions_passed = all(row["passed"] is True for row in checks)

    physical: JsonDict = {}
    rows: list[JsonDict] = []
    if preconditions_passed:
        print("[exp7355] phase=evidence event=start", flush=True)
        phase_start = time.monotonic()
        write_historical_model_receipt(root, paths.historical_models)
        physical = search_physical_state_receipts(
            root, paths.physical_state_search, candidate_paths=candidate_paths
        )
        rows = build_board_rows(root, context["prior_receipt"], physical)
        _atomic_json(paths.raw_rows, {"schema": SCHEMA + ".raw_rows", "rows": rows})
        for path in (paths.historical_models, paths.physical_state_search, paths.raw_rows):
            hashes[str(path)] = sha256_file(path)
        spans.append(_phase_span("evidence", phase_start, origin, len(rows)))

    artifact = previous._base_artifact(checks, hashes, spans)
    reduced = reduce_board_rows(rows)
    validation_passed = validation.get("required_checks_passed") is True
    physical_exists = physical.get("exists") is True
    disposition_complete = reduced["board_disposition_complete_score"] == 1
    summary = _summary(checks, physical, validation)
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "task_id": TASK_ID,
            "milestone": MILESTONE,
            "phase": 4,
            "run_date": RUN_DATE,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": time.monotonic() - origin,
            "random_seed": {
                "development": 7355001,
                "evaluation": 7355002,
                "resampling": 7355003,
                "sealed_before_results": True,
            },
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
                "historical_sources_authenticated": previous._gate(
                    True,
                    preconditions_passed,
                    preconditions_passed,
                    "The prior artifact, original board evidence, and cutoff must authenticate.",
                ),
                "prior_not_consumed_as_readiness": previous._gate(
                    False,
                    context.get("prior_used_as_readiness_gate"),
                    context.get("prior_used_as_readiness_gate") is False,
                    "The blocked Exp7341 disposition cannot authorize current work.",
                ),
                "three_board_dispositions_complete": previous._gate(
                    {"rows": 3, "score": 1},
                    {
                        "rows": reduced["board_count"],
                        "score": reduced["board_disposition_complete_score"],
                    },
                    disposition_complete,
                    "A known external block is a complete disposition, not readiness.",
                ),
                "gatemate_changed_physical_state_receipt": previous._gate(
                    PHYSICAL_RECEIPT_CONTRACT,
                    {
                        "exists": physical_exists,
                        "accepted_receipt_count": physical.get("accepted_receipt_count", 0),
                        "observed_missing_receipt": physical.get("observed_missing_receipt"),
                    },
                    physical_exists,
                    "Only a dated operator physical change enables a future operation.",
                ),
                "required_scoped_validation": previous._gate(
                    True,
                    validation.get("required_checks_passed"),
                    validation_passed,
                    "All affected tests, coverage, lint, format, type, and spec checks must pass.",
                ),
                "read_only_operation_boundary": previous._gate(
                    {"hardware": 0, "installation": 0, "procurement": 0, "messages": 0},
                    {"hardware": 0, "installation": 0, "procurement": 0, "messages": 0},
                    True,
                    "This task records state and cannot issue an operation.",
                ),
                "no_downstream_science_gate": previous._gate(
                    [],
                    context.get("downstream_science_gate_references", []),
                    context.get("downstream_science_gate_references", []) == [],
                    "Board accounting cannot block unrelated science.",
                ),
            },
            "gate_check_summary": summary,
            "validation_receipts": previous._normalized_validation(validation),
            "required_checks_passed": validation_passed,
            "missing_required_commands": deepcopy(validation.get("missing_required_commands", [])),
            "failed_required_commands": deepcopy(validation.get("failed_required_commands", [])),
            "duplicate_required_commands": deepcopy(
                validation.get("duplicate_required_commands", [])
            ),
            "repository_health": deepcopy(validation.get("repository_health", {})),
            "field_principles": _field_principles(),
            "flagged_adversarial": False,
            "hardware_value_score": 0,
            "physical_state_receipt": physical,
            "deployment_relevance": deployment_relevance(),
            "purchase_required": False,
            "hardware_path": hardware_path(physical_exists=physical_exists),
            "next_hardware_conditions": next_hardware_conditions(),
            "downstream_science_gate_references": deepcopy(
                context.get("downstream_science_gate_references", [])
            ),
            "source_artifact_states": {
                PRIOR_PATH.as_posix(): {
                    "producer_experiment_id": context.get("prior_receipt", {}).get("experiment_id"),
                    "terminal_status": context.get("prior_receipt", {}).get("status"),
                    "terminal_class": context.get("prior_receipt", {}).get("verdict_class"),
                    "quarantine": deepcopy(context.get("prior_quarantine", {})),
                    "validation_errors": deepcopy(context.get("prior_validation_errors", [])),
                    "used_as_readiness_gate": False,
                },
                CUTOFF_PATH.as_posix(): {
                    "producer_experiment_id": 6559,
                    "sha256": hashes.get(CUTOFF_PATH.as_posix()),
                    "used_as_readiness_gate": False,
                },
            },
        }
    )
    for field in ZERO_SCORE_FIELDS:
        artifact[field] = 0

    if not validation_passed:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: affected scoped validation failed",
                "board_disposition_complete_score": 0,
            }
        )
    elif not preconditions_passed:  # pragma: no cover - exercised by real fail-closed boundary.
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
                    "authenticated dispositions are complete, but hardware readiness, value, "
                    "and promotion remain zero; zero hardware or external operations were issued"
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
                    "dated GateMate receipt makes one future bounded operation eligible; no "
                    "hardware, purchase, or external operation was issued"
                ),
                "board_disposition_complete_score": int(disposition_complete),
            }
        )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


REQUIRED_ARTIFACT_FIELDS = previous.REQUIRED_ARTIFACT_FIELDS | {
    "flagged_adversarial",
    "hardware_readiness_score",
    "hardware_value_score",
}


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
                or row.get("source_artifact_path") != PRIOR_PATH.as_posix()
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
        any(artifact.get(field) != 0 for field in ZERO_SCORE_FIELDS),
        "readiness_value_or_promotion",
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
    prior_state = states.get(PRIOR_PATH.as_posix(), {}) if isinstance(states, Mapping) else {}
    if rows:
        add(
            prior_state.get("used_as_readiness_gate") is not False
            or prior_state.get("producer_experiment_id") != 7341
            or prior_state.get("terminal_class") != "blocked"
            or prior_state.get("validation_errors") != [],
            "prior_source_state",
        )
    add(artifact.get("deployment_relevance") != deployment_relevance(), "deployment_context")
    add(
        artifact.get("hardware_path") != hardware_path(physical_exists=physical_exists),
        "hardware_path",
    )
    add(artifact.get("next_hardware_conditions") != next_hardware_conditions(), "next_conditions")
    add(artifact.get("purchase_required") is not False, "purchase")
    add(artifact.get("downstream_science_gate_references") != [], "downstream_gate")
    add(artifact.get("flagged_adversarial") is not False, "adversarial_state")
    add(not isinstance(artifact.get("phase_spans"), list), "phase_spans")
    try:
        checksum_matches = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):  # pragma: no cover - malformed serialization boundary.
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
    """Atomically publish only a receipt accepted by the V645 validator."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


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
    prior = _read_json(root / PRIOR_PATH)
    print("[exp7355] phase=scoped_validation event=start", flush=True)
    validation_started = time.monotonic()
    scoped_basetemp = Path("/tmp/carnot-exp7355-scoped")
    scoped_basetemp.mkdir(parents=True, exist_ok=True)
    validation = run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[ENTRYPOINT_PATH.as_posix()],
        basetemp=scoped_basetemp,
        coverage_file=paths.validation_dir / ".coverage",
        log_dir=paths.validation_dir / "scoped",
        historical_failures=previous._historical_failures(root, prior),
    )
    validation_elapsed = time.monotonic() - validation_started
    print(
        f"[exp7355] phase=scoped_validation event=end checks={len(REQUIRED_CHECK_NAMES)} "
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

    print("[exp7355] phase=cold_candidate_validation event=start", flush=True)
    cold_started = time.monotonic()
    cold_errors = cold_validate_candidate(paths)
    cold_elapsed = time.monotonic() - cold_started
    print(
        f"[exp7355] phase=cold_candidate_validation event=end errors={len(cold_errors)} "
        f"elapsed_s={cold_elapsed:.3f}",
        flush=True,
    )
    artifact["acceptance_gate_results"]["cold_candidate_reduction"] = previous._gate(
        [],
        cold_errors,
        not cold_errors,
        "Reloaded candidate bytes and independently reduced raw rows must agree.",
    )

    print("[exp7355] phase=terminal_validators event=start", flush=True)
    terminal_started = time.monotonic()
    validators = previous._terminal_validators(
        root, paths.terminal_candidate, paths.validation_dir / "terminal"
    )
    terminal_elapsed = time.monotonic() - terminal_started
    artifact["validation_receipts"].extend(
        previous._normalized_validation({"validation_receipts": validators})
    )
    validators_passed = all(row["passed"] for row in validators)
    adversarial_passed = next(
        (row["passed"] for row in validators if row["name"] == "adversarial_verify"), False
    )
    artifact["flagged_adversarial"] = not adversarial_passed
    artifact["acceptance_gate_results"]["terminal_candidate_validation"] = previous._gate(
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
        f"[exp7355] phase=terminal_validators event=end units={len(validators)} "
        f"elapsed_s={terminal_elapsed:.3f}",
        flush=True,
    )
    write_artifact(paths.artifact, artifact)
    print(
        f"[exp7355] phase=terminal_write event=end path={paths.artifact} "
        f"verdict={artifact['verdict_class']}",
        flush=True,
    )
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover
    """Parse the fixed execution date or one read-only validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the experiment or validate an existing artifact without mutation."""

    print("[exp7355] phase=entry event=start", flush=True)
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
