"""Produce the V642 read-only board-continuity receipt.

This task checks the prior receipt and its original evidence. It never contacts
a board. The shipped V641 readers remain the authority for old measurements.

Spec: REQ-ISING-7314 and SCENARIO-ISING-7314-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import time
from typing import Any, cast

import yaml

from carnot import experiment_6559_gatemate_changed_state_continuity as receipt_authority
from carnot import experiment_7300_v641_board_continuity as current


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260915"
EXPERIMENT_ID = 7314
TASK_ID = "exp7314-board-continuity"
MILESTONE = "2026.09.642"
SCHEMA = "carnot.exp7314.v642.board_continuity.v1"
RANDOM_SEED = 7314

RESULT_PATH = Path("results/experiment_7314_v642_board_continuity.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7314_v642_board_continuity.json")
RAW_DIR = Path("results/raw/experiment_7314")
PHYSICAL_STATE_SEARCH_PATH = RAW_DIR / "gatemate_physical_state_receipt_search.json"
HISTORICAL_MODELS_PATH = RAW_DIR / "historical_model_receipts.json"
NEGATIVE_FIXTURES_PATH = RAW_DIR / "negative_fixture_receipts.json"
RAW_ROWS_PATH = RAW_DIR / "board_rows.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
VALIDATION_DIR = RAW_DIR / "validation"
REPOSITORY_HEALTH_DIR = RAW_DIR / "repository_health"
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7314_v642_board_continuity.py")
MODULE_PATH = Path("python/carnot/experiment_7314_v642_board_continuity.py")
TEST_PATH = Path("tests/python/test_experiment_7314_v642_board_continuity.py")
SPEC_PATH = current.SPEC_PATH
ROADMAP_PATH = current.ROADMAP_PATH
EXCLUSION_PATH = current.EXCLUSION_PATH
UPSTREAM_PATH = current.RESULT_PATH
GATEMATE_CUTOFF_PATH = current.GATEMATE_CUTOFF_PATH
REFERENCE_PATH = Path("research-references.md")
WISHLIST_PATH = Path("research-hardware-wishlist.md")

KV260_TERMINAL_CRITERION = current.KV260_TERMINAL_CRITERION
POLARFIRE_TERMINAL_CRITERION = current.POLARFIRE_TERMINAL_CRITERION
GATEMATE_CUTOFF_DATE = current.GATEMATE_CUTOFF_DATE
MISSING_RECEIPT = current.MISSING_RECEIPT
GATEMATE_OPERATOR_ACTION = current.GATEMATE_OPERATOR_ACTION
GATEMATE_FUTURE_ACTION = current.GATEMATE_FUTURE_ACTION

PHYSICAL_RECEIPT_CONTRACT: JsonDict = {
    "receipt_date": ">20260823",
    "operator_authored": True,
    "provenance": [path.as_posix() for path in receipt_authority.APPROVED_RECEIPT_LOCATIONS],
    "changed_field_any_of": ["cable", "port", "power", "board", "jtag", "dirtyjtag"],
}

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "Record GateMate prerequisites and graduated-board continuity",
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
    "prompt_sha256": "sha256:d6f8537175d2edb7db77090ef66c3c8c8034a48db02c980baf83602e3ef98099",
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    MODULE_PATH,
    ENTRYPOINT_PATH,
    TEST_PATH,
    UPSTREAM_PATH,
    GATEMATE_CUTOFF_PATH,
    REFERENCE_PATH,
    WISHLIST_PATH,
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    ROADMAP_PATH,
)

FIELD_PRINCIPLES = {
    **current.FIELD_PRINCIPLES,
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind this receipt to Exp7314.",
    "task_id": "Bind this receipt to the exact V642 roadmap task.",
    "milestone": "Bind this receipt to milestone 2026.09.642.",
    "spec_refs": "Connect the artifact and tests to REQ-ISING-7314 scenarios.",
    "status": "Write terminal results only after validation; keep checkpoints separate.",
    "run_date": "Use 20260915 with actual UTC start and end times.",
    "preconditions_checked": "Hash real inputs and record actual availability and failures.",
    "MODEL_SPECS": "List current executable models; historical identities stay in sidecars.",
    "model_invoked": "True for any attempted load or generation, including unusable output.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
    "inference_substrate": "Describe actual computation with a recognized literal.",
    "inference_substrate_class": "Use the actual class and never pad elapsed time.",
    "execution_venue": "Host aggregation is not GPU, FPGA, or board CPU execution.",
    "duration_s": "Measure total monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Seal development and evaluation seeds before outcomes are observed.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every board with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Retain planned, attempted, complete, and censored counts.",
    "acceptance_gate_results": "Each check states expected, observed, passed, and purpose.",
    "gate_check_summary": "Every blocked row names its exact check and observed failure.",
    "verifier_is_oracle": "Shared authority can support circular mechanics, not new science.",
    "honest_verdict": "Use a terminal prefix and state the bounded finding.",
    "verdict_class": "Use the closed verdict class; external absence is blocked at row scope.",
    "validation_receipts": "Retain exact commands, scope, exit codes, timing, and log hashes.",
    "board_continuity_complete_score": "One means three dispositions and prerequisites are recorded.",
    "board_rows": "Keep KV260 fabric, PolarFire CPU, and GateMate evidence separate.",
    "hardware_operations_issued": "Exactly zero for this read-only task.",
    "physical_state_receipt": "Only a local operator record can enable changed-state work.",
    "deployment_relevance": "External hardware context does not create a local result.",
    "repository_health": "Preserve repository-wide failures without waiving affected tests.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    current.REQUIRED_ARTIFACT_FIELDS
    | {"physical_state_receipt", "deployment_relevance", "repository_health"}
)

sha256_file = current.sha256_file
sha256_text = current.sha256_text
artifact_checksum = current.artifact_checksum
_read_json = current._read_json
_atomic_json = current._atomic_json
_writable_destination = current._writable_destination
_check = current._check
_first_failed = current._first_failed
_gate_summary = current._gate_summary
_gate_result = current._gate_result
_display_path = current._display_path
progress = current.progress
reduce_board_rows = current.reduce_board_rows


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, raw, and terminal outputs in separate locations."""

    artifact: Path
    checkpoint: Path
    physical_state_search: Path
    historical_models: Path
    negative_fixtures: Path
    raw_rows: Path
    terminal_candidate: Path
    validation_dir: Path
    repository_health_dir: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve task-owned outputs below one repository root."""

        return cls(
            root / RESULT_PATH,
            root / CHECKPOINT_PATH,
            root / PHYSICAL_STATE_SEARCH_PATH,
            root / HISTORICAL_MODELS_PATH,
            root / NEGATIVE_FIXTURES_PATH,
            root / RAW_ROWS_PATH,
            root / TERMINAL_CANDIDATE_PATH,
            root / VALIDATION_DIR,
            root / REPOSITORY_HEALTH_DIR,
        )

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Give tests private paths that cannot replace research evidence."""

        return cls.defaults(root)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the fields that fix only the V642 board task identity."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
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
    result["prompt_sha256"] = sha256_text(prompt) if isinstance(prompt, str) else None
    return result


def _board_row(receipt: Mapping[str, Any], board: str) -> Mapping[str, Any] | None:
    """Select one board through the shipped V641 reader."""

    return current._board_row(receipt, board)


def _reference_observation(
    root: Path,
    kv260: Mapping[str, Any] | None,
    polarfire: Mapping[str, Any] | None,
) -> JsonDict:
    """Hash the original terminal evidence named by the V641 rows."""

    return current.reference_observation(root, kv260, polarfire)


def _historical_health_observation(root: Path, receipt: Mapping[str, Any]) -> JsonDict:
    """Verify old repository-health logs before carrying their failures forward."""

    source_rows = receipt.get("baseline_validation_failures")
    rows = source_rows if isinstance(source_rows, list) else []
    observations: list[JsonDict] = []
    for row in rows:
        if not isinstance(row, Mapping):
            observations.append({"path": None, "passed": False})
            continue
        path = Path(str(row.get("log_path", "")))
        observed = sha256_file(path) if path.is_file() else None
        observations.append(
            {
                "path": str(path),
                "expected_sha256": row.get("log_hash"),
                "observed_sha256": observed,
                "passed": observed is not None and observed == row.get("log_hash"),
            }
        )
    return {
        "rows": observations,
        "all_match": bool(observations) and all(row["passed"] for row in observations),
    }


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], dict[str, Any]]:
    """Authenticate V641, original evidence, local references, and output paths."""

    print("[phase 0 check start] V642 sources, spec, and task contract", flush=True)
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
        "requirement": "REQ-ISING-7314" in spec_text,
        "scenarios": "SCENARIO-ISING-7314-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7314 and scenarios",
            {"requirement": True, "scenarios": True},
            spec_state,
            all(spec_state.values()),
        )
    )
    contract = _task_contract(root)
    checks.append(
        _check(
            "roadmap_task_contract",
            ROADMAP_PATH.as_posix(),
            TASK_ID,
            EXPECTED_TASK_CONTRACT,
            contract if contract is not None else "missing_task_contract",
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    print("[phase 0 check start] shipped authorities and writable outputs", flush=True)
    resources = {
        "v641_board_reader": callable(current._board_row),
        "v641_artifact_validator": callable(current.validate_artifact),
        "operator_receipt_parser": callable(current.search_gatemate_operator_receipts),
        "row_reducer": callable(reduce_board_rows),
        "artifact_writable": _writable_destination(paths.artifact),
        "checkpoint_writable": _writable_destination(paths.checkpoint),
        "physical_state_search_writable": _writable_destination(paths.physical_state_search),
        "historical_models_writable": _writable_destination(paths.historical_models),
        "negative_fixtures_writable": _writable_destination(paths.negative_fixtures),
        "raw_rows_writable": _writable_destination(paths.raw_rows),
        "terminal_candidate_writable": _writable_destination(paths.terminal_candidate),
    }
    checks.append(
        _check(
            "authority_and_output_ownership",
            "host",
            "shipped authorities and task-owned outputs",
            "all available and writable",
            resources,
            all(value is True for value in resources.values()),
        )
    )

    print("[phase 0 check start] exclusion state and Exp7300 receipt", flush=True)
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
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
    quarantine = current.quarantine_authority._quarantine(receipt, manifest, "7300")
    checks.append(
        _check(
            "exp7300_not_quarantined_disqualified_or_retired",
            UPSTREAM_PATH.as_posix(),
            "quarantined_or_disqualified_or_retired",
            False,
            quarantine,
            bool(receipt) and quarantine.get("quarantined") is False,
        )
    )
    upstream_errors = current.validate_artifact(receipt) if receipt else ["not_json"]
    checks.append(
        _check(
            "exp7300_latest_validator",
            UPSTREAM_PATH.as_posix(),
            "validator_errors",
            [],
            upstream_errors,
            upstream_errors == [],
        )
    )
    expected_identity = {
        "experiment_id": 7300,
        "status": "complete",
        "verdict_class": "positive",
        "board_continuity_complete_score": 1,
    }
    observed_identity = {key: receipt.get(key) for key in expected_identity}
    checks.append(
        _check(
            "exp7300_terminal_identity",
            UPSTREAM_PATH.as_posix(),
            "producer identity and terminal class",
            expected_identity,
            observed_identity,
            observed_identity == expected_identity,
        )
    )
    kv260 = _board_row(receipt, "KV260")
    polarfire = _board_row(receipt, "PolarFire")
    references = _reference_observation(root, kv260, polarfire)
    for row in references["rows"]:
        if row.get("passed") is True and isinstance(row.get("path"), str):
            hashes[str(row["path"])] = str(row["observed_sha256"])
    checks.append(
        _check(
            "exp7300_original_terminal_reference_hashes",
            UPSTREAM_PATH.as_posix(),
            "KV260 and PolarFire original evidence hashes",
            "all current bytes hash-match",
            references,
            references["all_match"] is True,
        )
    )
    expected_scope = {
        "kv260_met": True,
        "kv260_processor": "fpga_fabric",
        "polarfire_met": True,
        "polarfire_processor": "cpu",
        "polarfire_fpga_sampling": False,
    }
    observed_scope = {
        "kv260_met": kv260.get("terminal_criterion_met") if kv260 else None,
        "kv260_processor": kv260.get("processor_class") if kv260 else None,
        "polarfire_met": polarfire.get("terminal_criterion_met") if polarfire else None,
        "polarfire_processor": polarfire.get("processor_class") if polarfire else None,
        "polarfire_fpga_sampling": (
            polarfire.get("programmable_logic_sampling_observed") if polarfire else None
        ),
    }
    checks.append(
        _check(
            "graduated_board_execution_scopes",
            UPSTREAM_PATH.as_posix(),
            "FPGA fabric and CPU dispatch boundary",
            expected_scope,
            observed_scope,
            observed_scope == expected_scope,
        )
    )
    physical = receipt.get("physical_state_receipt") or receipt.get("changed_state_receipt")
    cutoff_hash = hashes.get(GATEMATE_CUTOFF_PATH.as_posix())
    cutoff_observed = {
        "receipt_present": (root / GATEMATE_CUTOFF_PATH).is_file(),
        "expected_by_exp7300": (
            physical.get("cutoff_source_hash") if isinstance(physical, Mapping) else None
        ),
        "observed_sha256": cutoff_hash,
    }
    checks.append(
        _check(
            "exp6559_boundary_authenticated",
            GATEMATE_CUTOFF_PATH.as_posix(),
            "presence and hash",
            {"present": True, "hash_match": True},
            cutoff_observed,
            cutoff_observed["receipt_present"] is True
            and cutoff_observed["expected_by_exp7300"] == cutoff_hash,
        )
    )

    reference_text = (
        (root / REFERENCE_PATH).read_text(encoding="utf-8")
        if (root / REFERENCE_PATH).is_file()
        else ""
    )
    wishlist_text = (
        (root / WISHLIST_PATH).read_text(encoding="utf-8")
        if (root / WISHLIST_PATH).is_file()
        else ""
    )
    deployment_sources = {
        "v642_refresh": "## V642 planning refresh — 2026-09-14" in reference_text,
        "extropic_access_limit": "No local device access was established." in reference_text,
        "kan_deferred": "defer a KAN hardware experiment until its predictor has value"
        in reference_text,
        "wishlist_extropic_limit": "no local hardware access" in wishlist_text.lower(),
    }
    checks.append(
        _check(
            "deployment_reference_boundaries",
            f"{REFERENCE_PATH.as_posix()} and {WISHLIST_PATH.as_posix()}",
            "Extropic and KAN relevance with access limits",
            {key: True for key in deployment_sources},
            deployment_sources,
            all(deployment_sources.values()),
        )
    )

    health = _historical_health_observation(root, receipt)
    for row in health["rows"]:
        if row.get("passed") is True:
            hashes[_display_path(root, Path(str(row["path"])))] = str(row["observed_sha256"])
    checks.append(
        _check(
            "exp7300_repository_health_log_hashes",
            UPSTREAM_PATH.as_posix(),
            "baseline_validation_failures log hashes",
            "all current bytes hash-match",
            health,
            health["all_match"] is True,
        )
    )
    return (
        checks,
        hashes,
        {
            "receipt": receipt,
            "kv260_row": kv260,
            "polarfire_row": polarfire,
            "manifest": manifest,
            "upstream_quarantine": quarantine,
            "reference_observation": references,
            "historical_health": health,
        },
    )


def write_historical_model_receipt(root: Path, path: Path) -> JsonDict:
    """Keep all prior model declarations outside the current task identity."""

    source = _read_json(root / UPSTREAM_PATH)
    receipt = {
        "schema": "carnot.exp7314.historical_model_receipts.v1",
        "current_invocation_model_specs": [],
        "current_invocation_model_count": 0,
        "injected_invocation_payload_count": 0,
        "source_receipts": [
            {
                "source_path": UPSTREAM_PATH.as_posix(),
                "source_sha256": sha256_file(root / UPSTREAM_PATH),
                "producer_experiment_id": source.get("experiment_id"),
                "terminal_status": source.get("status"),
                "terminal_class": source.get("verdict_class"),
                "historical_MODEL_SPECS": source.get("MODEL_SPECS"),
                "historical_model_invoked": source.get("model_invoked"),
                "historical_invocation_counts": source.get("invocation_counts"),
            }
        ],
    }
    _atomic_json(path, receipt)
    return receipt


def search_physical_state_receipts(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Use the shipped parser and bind its result to the V642 receipt contract."""

    result = current.search_gatemate_operator_receipts(
        root, raw_path, candidate_paths=candidate_paths
    )
    raw = _read_json(raw_path)
    raw.update(
        {
            "schema": "carnot.exp7314.gatemate_physical_state_search.v1",
            "run_date": RUN_DATE,
            "eligibility_contract": PHYSICAL_RECEIPT_CONTRACT,
            "approved_local_sources_only": True,
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
    root: Path, upstreams: Mapping[str, Any], physical: Mapping[str, Any]
) -> list[JsonDict]:
    """Advance three receipt rows without changing their evidence authority."""

    receipt = cast(Mapping[str, Any], upstreams["receipt"])
    upstream_hash = sha256_file(root / UPSTREAM_PATH)
    changed = physical.get("exists") is True
    sources = {
        "KV260": cast(Mapping[str, Any], upstreams["kv260_row"]),
        "GateMate": cast(Mapping[str, Any], _board_row(receipt, "GateMate")),
        "PolarFire": cast(Mapping[str, Any], upstreams["polarfire_row"]),
    }
    rows: list[JsonDict] = []
    for board, source in sources.items():
        row = deepcopy(dict(source))
        row.pop("row_sha256", None)
        if board == "GateMate":
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
                    "physical_receipt_contract": PHYSICAL_RECEIPT_CONTRACT,
                    "observed_state": (
                        "operator_changed_physical_state_recorded"
                        if changed
                        else "operator_changed_physical_state_receipt_missing"
                    ),
                    "observed_missing_receipt": physical.get("observed_missing_receipt"),
                    "disposition": (
                        "changed_physical_state_future_action_enabled"
                        if changed
                        else "blocked_changed_physical_state"
                    ),
                    "exact_next_condition": (
                        str(physical.get("newly_enabled_next_action"))
                        if changed
                        else GATEMATE_OPERATOR_ACTION
                    ),
                    "failed_value": None if changed else MISSING_RECEIPT,
                    "metric": changed,
                    "error": None if changed else MISSING_RECEIPT,
                    "abstention": not changed,
                }
            )
        else:
            evidence = [
                {
                    "path": source.get("latest_receipt_path"),
                    "sha256": source.get("latest_receipt_hash"),
                    "run_date": source.get("latest_receipt_date"),
                },
                *deepcopy(source.get("referenced_evidence", [])),
            ]
            row.update(
                {
                    "latest_receipt_path": UPSTREAM_PATH.as_posix(),
                    "latest_receipt_date": receipt.get("run_date"),
                    "latest_receipt_hash": upstream_hash,
                    "latest_receipt_authenticated": True,
                    "referenced_evidence": evidence,
                }
            )
        row["hardware_operations_issued"] = []
        row["hardware_command_count"] = 0
        rows.append(current._finish_row(row))
    return rows


def _invocation_counts() -> JsonDict:
    """Return zero for every current model event state."""

    return {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "model_loads_failed": 0,
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "generation_calls_failed": 0,
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": 0,
    }


def deployment_relevance() -> JsonDict:
    """Record external deployment context without claiming local execution."""

    return {
        "source_paths": [REFERENCE_PATH.as_posix(), WISHLIST_PATH.as_posix()],
        "extropic": {
            "relevance": "sparse probabilistic hardware with companion processing",
            "local_device_access": False,
            "deployment_status": "deferred_until_authenticated_device_access",
            "required_future_evidence": "graph placement, transfer cost, device latency, and power",
            "procurement_or_vendor_contact": False,
            "local_performance_claim": False,
        },
        "kan": {
            "relevance": "bounded spline tables remain a possible FPGA deployment form",
            "deployment_status": "deferred",
            "required_future_evidence": "a useful target predictor and complete transfer-cost accounting",
            "procurement_or_vendor_contact": False,
            "local_performance_claim": False,
        },
        "availability_is_scientific_result": False,
        "external_actions": [],
    }


def read_validation_receipts(directory: Path) -> list[JsonDict]:
    """Read exact command, exit, elapsed, and hash fields from log sidecars."""

    rows = current.read_validation_receipts(directory)
    for row in rows:
        text = Path(str(row["log_path"])).read_text(encoding="utf-8")
        match = re.search(r"(?m)^\[elapsed_s\] ([0-9]+(?:\.[0-9]+)?)$", text)
        row["elapsed_s"] = float(match.group(1)) if match else None
    return rows


def _base_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Mapping[str, float],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Create every required field before selecting a terminal finding."""

    artifact = current._base_artifact(
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        phase_spans=phase_spans,
        checks=checks,
        hashes=hashes,
    )
    artifact.update(
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "task_id": TASK_ID,
            "milestone": MILESTONE,
            "run_date": RUN_DATE,
            "spec_refs": [
                "REQ-ISING-7314",
                "SCENARIO-ISING-7314-PREFLIGHT",
                "SCENARIO-ISING-7314-BOARDS",
                "SCENARIO-ISING-7314-GATEMATE",
                "SCENARIO-ISING-7314-DEPLOYMENT",
                "SCENARIO-ISING-7314-ARTIFACT",
            ],
            "field_principles": FIELD_PRINCIPLES,
            "random_seed": RANDOM_SEED,
            "invocation_counts": _invocation_counts(),
            "board_continuity_complete_score": 0,
            "board_disposition_complete_score": 0,
            "physical_state_receipt": {},
            "changed_state_receipt": {},
            "deployment_relevance": deployment_relevance(),
            "repository_health": {},
            "external_actions": [],
        }
    )
    return artifact


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Aggregate immutable board evidence in memory without a board command."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: dict[str, float] = {}

    phase_started = time.monotonic()
    progress(0, "start", "authenticate V642 sources before aggregation")
    checks, hashes, upstreams = collect_preconditions(root, paths)
    spans["phase_0_preconditions"] = time.monotonic() - phase_started
    failed = _first_failed(checks)
    progress(0, "end", f"preconditions failed={int(failed is not None)}")
    if failed is not None:
        artifact = _base_artifact(
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            checks=checks,
            hashes=hashes,
        )
        artifact["honest_verdict"] = (
            f"blocked_{failed['check']}: expected {failed['expected_value']!r}; "
            f"observed {failed['observed_value']!r}"
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    _atomic_json(
        paths.checkpoint,
        {
            "schema": SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "in_progress",
            "started_at_utc": started_at,
            "terminal_artifact_path": str(paths.artifact),
        },
    )

    phase_started = time.monotonic()
    progress(1, "start", "write historical and negative-control sidecars")
    write_historical_model_receipt(root, paths.historical_models)
    negative = current.write_negative_fixture_receipt(paths.negative_fixtures)
    for path in (paths.historical_models, paths.negative_fixtures):
        hashes[_display_path(root, path)] = sha256_file(path)
    spans["phase_1_hashed_sidecars"] = time.monotonic() - phase_started
    progress(1, "end", f"sidecars=2 negative-fixtures={negative['fixture_count']}")

    phase_started = time.monotonic()
    progress(2, "start", "search approved local GateMate operator records")
    physical = search_physical_state_receipts(
        root, paths.physical_state_search, candidate_paths=candidate_paths
    )
    hashes[_display_path(root, paths.physical_state_search)] = physical["search_receipt_hash"]
    spans["phase_2_physical_state_receipt"] = time.monotonic() - phase_started
    progress(2, "end", f"accepted-receipts={int(physical['exists'])} commands=0")

    phase_started = time.monotonic()
    progress(3, "start", "write and independently reduce three board rows")
    rows = build_board_rows(root, upstreams, physical)
    _atomic_json(paths.raw_rows, {"rows": rows})
    hashes[_display_path(root, paths.raw_rows)] = sha256_file(paths.raw_rows)
    reduced = reduce_board_rows(_read_json(paths.raw_rows).get("rows", []))
    spans["phase_3_board_reducer"] = time.monotonic() - phase_started
    progress(
        3,
        "end",
        f"completed-units={reduced['board_count']} elapsed={time.monotonic() - started:.6f}s",
    )

    phase_started = time.monotonic()
    progress(4, "start", "assemble terminal host aggregation in memory")
    validation_rows = read_validation_receipts(paths.validation_dir)
    health_rows = read_validation_receipts(paths.repository_health_dir)
    historical_failures = cast(Mapping[str, Any], upstreams["receipt"]).get(
        "baseline_validation_failures", []
    )
    for row in [*validation_rows, *health_rows]:
        hashes[_display_path(root, Path(str(row["log_path"])))] = str(row["log_hash"])
    validation_rows.append(
        {
            "command": "internal reduce_board_rows(raw board rows)",
            "exit_code": 0,
            "elapsed_s": spans["phase_3_board_reducer"],
            "log_hash": sha256_text(json.dumps(reduced, sort_keys=True)),
        }
    )
    spans["phase_4_artifact_assembly"] = time.monotonic() - phase_started
    artifact = _base_artifact(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        checks=checks,
        hashes=hashes,
    )
    summary = _gate_summary(checks)
    summary["board_blocks"] = (
        []
        if physical["exists"]
        else [
            {
                "verdict": "blocked_changed_physical_state",
                "upstream": "physical_state_receipt",
                "check": "gatemate_changed_physical_state_receipt",
                "field": "receipt_date/operator_authored/provenance/changed_field",
                "expected_value": PHYSICAL_RECEIPT_CONTRACT,
                "observed_value": {
                    "accepted_receipt_count": physical["accepted_receipt_count"],
                    "selected_source_path": physical.get("source_path"),
                    "absence": MISSING_RECEIPT,
                },
            }
        ]
    )
    relevance = deployment_relevance()
    repository_health = {
        "scope": "repository-wide history is separate from affected V642 validation",
        "historical_exp7300_failures": deepcopy(historical_failures),
        "historical_log_hashes_authenticated": upstreams["historical_health"]["all_match"],
        "current_repository_wide_receipts": health_rows,
        "does_not_waive_affected_test_failure": True,
    }
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "aggregation",
            "rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": 3,
                "complete": 3,
                "completed": 3,
                "censored": 0,
                "independent_units_planned": 3,
                "independent_units_completed": 3,
                "stopping_rule": "one authenticated read-only disposition per board",
            },
            "acceptance_gate_results": {
                "exp7300_and_original_references_authenticated": _gate_result(
                    "Reject a producer or original terminal byte mismatch.",
                    True,
                    upstreams["reference_observation"]["all_match"],
                    upstreams["reference_observation"]["all_match"] is True,
                ),
                "kv260_fabric_scope_preserved": _gate_result(
                    "Fabric graduation is distinct from present availability.",
                    {"processor_class": "fpga_fabric", "fabric_execution_completed": True},
                    {
                        "processor_class": rows[0]["processor_class"],
                        "fabric_execution_completed": rows[0]["fabric_execution_completed"],
                    },
                    rows[0]["processor_class"] == "fpga_fabric"
                    and rows[0]["fabric_execution_completed"] is True,
                ),
                "gatemate_changed_state_disposition": _gate_result(
                    "Record a valid later receipt or the exact failed receipt contract.",
                    "changed receipt or blocked_changed_physical_state",
                    rows[1]["disposition"],
                    rows[1]["disposition"]
                    in {
                        "blocked_changed_physical_state",
                        "changed_physical_state_future_action_enabled",
                    },
                ),
                "polarfire_cpu_scope_preserved": _gate_result(
                    "CPU dispatch is not FPGA sampling or host emulation.",
                    {"processor_class": "cpu", "fpga_sampling": False},
                    {
                        "processor_class": rows[2]["processor_class"],
                        "fpga_sampling": rows[2]["programmable_logic_sampling_observed"],
                    },
                    rows[2]["processor_class"] == "cpu"
                    and rows[2]["programmable_logic_sampling_observed"] is False,
                ),
                "deployment_context_bounded": _gate_result(
                    "External relevance cannot become a local device result.",
                    {"external_actions": [], "availability_is_scientific_result": False},
                    {
                        "external_actions": relevance["external_actions"],
                        "availability_is_scientific_result": relevance[
                            "availability_is_scientific_result"
                        ],
                    },
                    relevance["external_actions"] == []
                    and relevance["availability_is_scientific_result"] is False,
                ),
                "negative_receipt_fixtures_fail_closed": _gate_result(
                    "Missing and malformed receipts authorize no command.",
                    {"count": 2, "all_failed_closed": True},
                    {
                        "count": negative["fixture_count"],
                        "all_failed_closed": negative["all_failed_closed"],
                    },
                    negative["fixture_count"] == 2 and negative["all_failed_closed"] is True,
                ),
                "hardware_operations_issued_count": _gate_result(
                    "This task issues no hardware operation.", 0, 0, True
                ),
                "three_continuity_rows_reduce": _gate_result(
                    "Three authenticated rows retain exact next conditions.",
                    1,
                    reduced["board_disposition_complete_score"],
                    reduced["board_disposition_complete_score"] == 1,
                ),
            },
            "gate_check_summary": summary,
            "verdict_class": "positive",
            "honest_verdict": (
                "complete: three authenticated board dispositions and exact next conditions "
                "are recorded. KV260 fabric graduation remains preserved with SSH-only future "
                "access. PolarFire hash-matched CPU dispatch remains preserved and is not FPGA "
                "sampling. "
                + (
                    "A later GateMate receipt enables one bounded future integration step. "
                    if physical["exists"]
                    else "GateMate remains blocked_changed_physical_state because no qualifying "
                    "operator receipt exists. "
                )
                + "Extropic and KAN are deployment context only. This task issued zero hardware "
                "or external operations."
            ),
            "validation_receipts": validation_rows,
            "baseline_validation_failures": deepcopy(historical_failures),
            "repository_health": repository_health,
            "board_disposition_complete_score": reduced["board_disposition_complete_score"],
            "board_continuity_complete_score": reduced["board_disposition_complete_score"],
            "board_rows": rows,
            "operator_state_receipt": physical,
            "changed_state_receipt": physical,
            "physical_state_receipt": physical,
            "deployment_relevance": relevance,
            "source_artifact_states": {
                UPSTREAM_PATH.as_posix(): {
                    "producer_experiment_id": 7300,
                    "terminal_status": cast(Mapping[str, Any], upstreams["receipt"]).get("status"),
                    "terminal_class": cast(Mapping[str, Any], upstreams["receipt"]).get(
                        "verdict_class"
                    ),
                    "quarantine": upstreams["upstream_quarantine"],
                }
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(4, "end", "terminal receipt assembled; hardware-operations=0")
    return artifact


def _v641_compatible(artifact: Mapping[str, Any]) -> JsonDict:
    """Translate only V642 identity fields for the shipped V641 validator."""

    compatible = deepcopy(dict(artifact))
    compatible.update(
        {
            "schema": current.SCHEMA,
            "experiment_id": current.EXPERIMENT_ID,
            "task_id": current.TASK_ID,
            "milestone": current.MILESTONE,
            "run_date": "20260913",
            "field_principles": current.FIELD_PRINCIPLES,
            "changed_state_receipt": artifact.get("physical_state_receipt"),
        }
    )
    old_count_keys = {
        "model_loads_attempted",
        "model_loads_completed",
        "model_loads_failed",
        "generation_calls_attempted",
        "generation_calls_completed",
        "generation_calls_failed",
        "usable_answers",
    }
    compatible["invocation_counts"] = {
        key: value
        for key, value in cast(Mapping[str, Any], artifact.get("invocation_counts", {})).items()
        if key in old_count_keys
    }
    gates = cast(Mapping[str, Any], artifact.get("acceptance_gate_results", {}))
    compatible["acceptance_gate_results"] = {
        "source_and_references_authenticated": gates.get(
            "exp7300_and_original_references_authenticated"
        ),
        "kv260_fabric_scope_preserved": gates.get("kv260_fabric_scope_preserved"),
        "polarfire_cpu_scope_preserved": gates.get("polarfire_cpu_scope_preserved"),
        "gatemate_changed_state_disposition": gates.get("gatemate_changed_state_disposition"),
        "negative_receipt_fixtures_fail_closed": gates.get("negative_receipt_fixtures_fail_closed"),
        "hardware_operations_issued_count": gates.get("hardware_operations_issued_count"),
        "three_dispositions_reduce": gates.get("three_continuity_rows_reduce"),
    }
    compatible["reproducibility_checksum"] = artifact_checksum(compatible)
    return compatible


def validate_artifact(artifact: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Check V642 boundaries, then delegate old evidence to the V641 validator."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("invocation_counts") != _invocation_counts():
        errors.append("invocation_counts")
    try:
        checksum_matches = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_matches = False
    if not checksum_matches:
        errors.append("reproducibility_checksum")
    rows = artifact.get("board_rows")
    reduced = reduce_board_rows(rows)
    if (
        artifact.get("board_continuity_complete_score")
        != reduced["board_disposition_complete_score"]
    ):
        errors.append("board_continuity_score")
    by_board = (
        {
            row.get("board"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        }
        if isinstance(rows, list)
        else {}
    )
    if artifact.get("status") == "complete" and (
        len(by_board) != 3
        or by_board.get("KV260", {}).get("processor_class") != "fpga_fabric"
        or by_board.get("KV260", {}).get("fabric_execution_completed") is not True
        or by_board.get("PolarFire", {}).get("processor_class") != "cpu"
        or by_board.get("PolarFire", {}).get("programmable_logic_sampling_observed") is not False
        or by_board.get("PolarFire", {}).get("host_emulation") is not False
    ):
        errors.append("board_row_scope")
    if artifact.get("hardware_operations_issued") != [] or artifact.get("external_actions") != []:
        errors.append("hardware_operations")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("model_load_count") != 0
        or artifact.get("generation_count") != 0
        or artifact.get("model_invocation_count") != 0
    ):
        errors.append("model_declaration")
    if artifact.get("status") == "complete" and (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate")
    if artifact.get("deployment_relevance") != deployment_relevance():
        errors.append("deployment_relevance")
    physical = artifact.get("physical_state_receipt")
    if artifact.get("status") == "complete" and (
        not isinstance(physical, Mapping)
        or physical.get("eligibility_contract") != PHYSICAL_RECEIPT_CONTRACT
        or physical.get("hardware_operations_issued") != []
    ):
        errors.append("physical_state_receipt")
    try:
        compatible = _v641_compatible(artifact)
    except (AttributeError, TypeError, ValueError):
        return errors
    for error in current.validate_artifact(compatible, root=root):
        if error not in errors:
            errors.append(error)
    return errors


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish only bytes accepted by the validator chain."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7314 artifact: {errors}")
    _atomic_json(path, artifact)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Build, validate, and atomically publish the read-only receipt."""

    artifact = build_artifact(root, paths)
    progress(5, "before", "write and validate the measured terminal candidate")
    _atomic_json(paths.terminal_candidate, artifact)
    errors = validate_artifact(_read_json(paths.terminal_candidate), root=root)
    progress(5, "after", f"candidate validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7314 artifact: {errors}")
    progress(6, "before", "atomic terminal write")
    receipt = atomic_write(paths.artifact, artifact)
    progress(6, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Keep production and read-only validation on one thin CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or validate existing bytes without mutation."""

    progress(0, "entry", "Exp7314 host aggregation; model-loads=0 hardware-operations=0")
    args = _parser().parse_args(argv)
    try:
        if args.validate is not None:
            progress(5, "before", f"read-only validation path={args.validate}")
            artifact = _read_json(args.validate)
            errors = ["artifact_not_json_object"] if not artifact else validate_artifact(artifact)
            progress(5, "after", f"read-only validation errors={len(errors)}")
            if errors:
                print(f"validation_failed errors={errors}", flush=True)
                return 2
            print("validation_passed", flush=True)
            return 0
        if args.date != RUN_DATE:
            raise ValueError(f"run date must be {RUN_DATE}")
        root = args.root.resolve()
        paths = ExperimentPaths.defaults(root)
        artifact = run_experiment(root, paths)
        print(
            f"experiment_complete status={artifact['status']} "
            f"score={artifact['board_continuity_complete_score']} output={paths.artifact}",
            flush=True,
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment_error type={type(exc).__name__} message={exc}", flush=True)
        return 2
