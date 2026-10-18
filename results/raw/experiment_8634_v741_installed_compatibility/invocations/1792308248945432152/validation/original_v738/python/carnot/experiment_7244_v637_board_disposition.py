"""Produce a read-only disposition for KV260, GateMate, and PolarFire.

This module authenticates existing receipts. It does not contact a board. The
GateMate scan reuses the approved operator-receipt parser, while the other rows
retain completed evidence instead of repeating successful smoke tests.

Spec: REQ-ISING-7244 and SCENARIO-ISING-7244-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_6559_gatemate_changed_state_continuity as exp6559
from carnot import experiment_7217_v635_abi_board_readiness as exp7217
from carnot import experiment_7231_v636_board_continuity as exp7231
from carnot import experiment_7243_v637_native_memory as exp7243


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260912"
EXPERIMENT_ID = 7244
TASK_ID = "exp7244-board-disposition"
MILESTONE = "2026.09.637"
SCHEMA = "carnot.exp7244.v637.board_disposition.v1"

RESULT_PATH = Path("results/experiment_7244_v637_board_disposition.json")
RAW_SEARCH_PATH = Path("results/raw/experiment_7244/gatemate_operator_receipt_search.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7244_v637_board_disposition.json")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7244_v637_board_disposition.py")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
CONTINUITY_PATH = Path("results/experiment_7231_v636_board_continuity.json")
POLARFIRE_RAW_PATH = Path("results/raw/experiment_7231/polarfire_dispatch.json")
GATEMATE_CUTOFF_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
GATEMATE_AUDIT_PATH = Path("results/experiment_7146_v627_gatemate_changed_state.json")
KV260_PATH = Path("results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json")
KV260_CONFIRM_PATH = Path(
    "results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"
)
MEMORY_PATH = Path("results/experiment_7243_v637_native_memory.json")

KV260_TERMINAL_CRITERION = (
    "board-level programmable-logic latency transcript and successful KV260 synthesis"
)
POLARFIRE_TERMINAL_CRITERION = (
    "end-to-end hash-matched CPU dispatch with retained raw transcript evidence"
)
GATEMATE_TERMINAL_CRITERION = "n=16 Ising tile flashed and smoke-tested on programmable logic"
GATEMATE_CUTOFF_DATE = "20260823"

EXPECTED_TASK_CONTRACT: JsonDict = {
    "id": TASK_ID,
    "title": "GateMate changed-state review and graduated-board disposition",
    "track": "hardware",
    "priority": "high",
    "requires_gpu": False,
    "max_turns": 20,
    "estimated_wall_time_min": 15,
    "per_unit_rows": True,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    ROADMAP_PATH,
    Path("python/carnot/experiment_7231_v636_board_continuity.py"),
    Path("python/carnot/experiment_7244_v637_board_disposition.py"),
    ENTRYPOINT_PATH,
    Path("tests/python/test_experiment_7244_v637_board_disposition.py"),
    CONTINUITY_PATH,
    POLARFIRE_RAW_PATH,
    GATEMATE_CUTOFF_PATH,
    GATEMATE_AUDIT_PATH,
    KV260_PATH,
    KV260_CONFIRM_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "Bind the receipt to Exp7244.",
    "task_id": "Bind the receipt to the exact roadmap task.",
    "milestone": "Bind the receipt to V637.",
    "spec_refs": "Connect tests and output to the driving requirement and scenarios.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Record the actual UTC start time.",
    "completed_at_utc": "Record the actual UTC end time.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "phase_spans_s": "Measured monotonic duration for each numbered aggregation phase.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources are separate.",
    "model_invocation_count": "Current model calls remain zero for this aggregation.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared units, attempted, completed and censored counts, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked verdict names check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True only when the verification authority also defines correctness.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "honest_verdict": "Completed findings start complete; external absence starts blocked.",
    "acceptance_gate_results": "Preserve each frozen criterion and its actual result independently.",
    "board_disposition_complete_score": "All three dispositions include exact next conditions, including GateMate absence.",
    "board_rows": "Record board, latest source, terminal criterion, observed state, and exact next prerequisite.",
    "hardware_operations_issued": "The list is empty because this task is read-only.",
    "operator_state_receipt": "Record dated GateMate physical evidence or explicit absence.",
    "operation_map": "Separate real memory work from prospective device operations and retain unknowns.",
    "memory_footprint": "Use authenticated V637 memory data when present; absence does not block dispositions.",
    "operator_prerequisites": "Name required operator inputs without purchases, vendor contact, or package installation.",
    "external_actions": "No purchase, vendor contact, package install, upload, or publication occurs.",
    "validation_receipts": "Record the producer replay and independent reducer result used for the terminal write.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep terminal, raw-search, and checkpoint outputs in separate locations."""

    artifact: Path
    gatemate_search: Path
    checkpoint: Path

    @classmethod
    def defaults(cls, root: Path = REPO_ROOT) -> ExperimentPaths:
        """Resolve production paths under one explicit repository root."""

        return cls(root / RESULT_PATH, root / RAW_SEARCH_PATH, root / CHECKPOINT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Give tests private outputs so replay cannot change research results."""

        return cls(root / RESULT_PATH, root / RAW_SEARCH_PATH, root / CHECKPOINT_PATH)


def progress(phase: int, boundary: str, operation: str) -> None:
    """Flush each true phase boundary so a run never appears silent."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def sha256_file(path: Path) -> str:
    """Hash exact file bytes with the repository's tagged SHA-256 format."""

    return exp7231.sha256_file(path)


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum field itself."""

    return exp7231.artifact_checksum(payload)


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the exact two-key principle representation."""

    return exp7231.unwrap_principled_value(value)


def _read_json(path: Path) -> JsonDict:
    """Return an empty object for missing, malformed, or non-object JSON."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Use same-directory replacement so readers never see partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _writable_destination(path: Path) -> bool:
    """Probe a destination directory without leaving test data behind."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7244-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
    except OSError:
        return False
    return True


def _check(
    name: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Give each precondition the same fail-closed diagnostic shape."""

    return {
        "check": name,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _task_contract(root: Path) -> JsonDict | None:
    """Read only the roadmap fields that define the Exp7244 task."""

    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else None
    if not isinstance(tasks, list):
        return None
    for task in tasks:
        if isinstance(task, Mapping) and task.get("id") == TASK_ID:
            return {key: deepcopy(task.get(key)) for key in EXPECTED_TASK_CONTRACT}
    return None


def _manifest_match(value: Any, experiment_number: str) -> bool:
    """Match only a real experiment identifier in the exclusion manifest."""

    return exp7217._manifest_mentions_experiment(value, experiment_number)


def _quarantine(value: Mapping[str, Any], manifest: Any, experiment_number: str) -> JsonDict:
    """Reject explicit or manifest quarantine before any gate is consumed."""

    return exp7217.upstream_quarantine_observation(
        value, manifest_match=_manifest_match(manifest, experiment_number)
    )


def _row_for_board(rows: Any, board: str) -> Mapping[str, Any] | None:
    """Select exactly one named board row from an upstream receipt."""

    matches = (
        [row for row in rows if isinstance(row, Mapping) and row.get("board") == board]
        if isinstance(rows, list)
        else []
    )
    return matches[0] if len(matches) == 1 else None


def collect_preconditions(
    root: Path, paths: ExperimentPaths
) -> tuple[list[JsonDict], dict[str, str], dict[str, JsonDict]]:
    """Authenticate required bytes, imports, cross-hashes, and writable paths."""

    print("[phase 0 check start] source bytes and driving contract", flush=True)
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
        "requirement": "REQ-ISING-7244" in spec_text,
        "scenarios": "SCENARIO-ISING-7244-" in spec_text,
    }
    checks.append(
        _check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ISING-7244 and scenarios",
            {"requirement": True, "scenarios": True},
            spec_state,
            all(spec_state.values()),
        )
    )
    contract = _task_contract(root)
    contract_observed: Any = contract if contract is not None else "missing_task_contract"
    checks.append(
        _check(
            "roadmap_task_contract",
            ROADMAP_PATH.as_posix(),
            TASK_ID,
            EXPECTED_TASK_CONTRACT,
            contract_observed,
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    print("[phase 0 check start] imports and output destinations", flush=True)
    resources = {
        "python": str(Path(os.sys.executable).absolute()),
        "yaml": yaml.__version__,
        "exp6559_receipt_parser": callable(exp6559.search_dated_receipts),
        "exp7231_validator": callable(exp7231.validate_artifact),
        "exp7243_validator": callable(exp7243.validate_artifact),
        "artifact_writable": _writable_destination(paths.artifact),
        "raw_writable": _writable_destination(paths.gatemate_search),
        "checkpoint_writable": _writable_destination(paths.checkpoint),
    }
    checks.append(
        _check(
            "imports_resources_and_outputs",
            "host",
            "python,yaml,producer validators,raw/checkpoint/result",
            "all available and writable",
            resources,
            all(
                value is True
                for key, value in resources.items()
                if key.endswith(("parser", "validator", "writable"))
            ),
        )
    )

    print("[phase 0 check start] mandatory upstream quarantine and cross-hashes", flush=True)
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
    continuity = _read_json(root / CONTINUITY_PATH)
    raw = _read_json(root / POLARFIRE_RAW_PATH)
    cutoff = _read_json(root / GATEMATE_CUTOFF_PATH)
    kv260 = _read_json(root / KV260_PATH)
    kv260_confirm = _read_json(root / KV260_CONFIRM_PATH)
    mandatory = {
        "continuity": (continuity, CONTINUITY_PATH, "7231"),
        "polarfire_raw": (raw, POLARFIRE_RAW_PATH, "7231"),
        "gatemate_cutoff": (cutoff, GATEMATE_CUTOFF_PATH, "6559"),
        "kv260": (kv260, KV260_PATH, "3709"),
        "kv260_confirm": (kv260_confirm, KV260_CONFIRM_PATH, "3721"),
    }
    quarantines: dict[str, JsonDict] = {}
    for name, (value, path, number) in mandatory.items():
        observation = _quarantine(value, manifest, number)
        quarantines[name] = observation
        checks.append(
            _check(
                f"{name}_not_quarantined",
                path.as_posix(),
                "quarantined",
                False,
                observation,
                bool(value) and observation.get("quarantined") is False,
            )
        )

    continuity_errors = (
        exp7231.validate_artifact(continuity)
        if continuity and quarantines["continuity"].get("quarantined") is False
        else ["not_authenticated"]
    )
    checks.append(
        _check(
            "exp7231_producer_authentication",
            CONTINUITY_PATH.as_posix(),
            "shipped_validator_errors",
            [],
            continuity_errors,
            continuity_errors == [],
        )
    )
    polar_row = _row_for_board(continuity.get("board_rows"), "PolarFire")
    raw_hash = hashes.get(POLARFIRE_RAW_PATH.as_posix())
    raw_state = {
        "dispatch_completed": raw.get("dispatch_completed"),
        "transport_exit_code": raw.get("transport_exit_code"),
        "workload_exit_code": raw.get("workload_exit_code"),
        "input_hash_matches": raw.get("input_hash_matches"),
        "output_hash_matches": raw.get("output_hash_matches"),
        "raw_sha256": raw_hash,
        "row_raw_sha256": polar_row.get("raw_dispatch_transcript_hash") if polar_row else None,
        "processor_class": raw.get("processor_class"),
        "programmable_logic_sampling_observed": raw.get("programmable_logic_sampling_observed"),
    }
    raw_expected = {
        "dispatch_completed": True,
        "transport_exit_code": 0,
        "workload_exit_code": 0,
        "input_hash_matches": True,
        "output_hash_matches": True,
        "raw_sha256": raw_hash,
        "row_raw_sha256": raw_hash,
        "processor_class": "cpu",
        "programmable_logic_sampling_observed": False,
    }
    checks.append(
        _check(
            "polarfire_end_to_end_hash_match",
            POLARFIRE_RAW_PATH.as_posix(),
            "dispatch,input,output,raw hash,processor",
            raw_expected,
            raw_state,
            raw_state == raw_expected and quarantines["polarfire_raw"].get("quarantined") is False,
        )
    )

    kv_row = _row_for_board(continuity.get("board_rows"), "KV260")
    kv_hash = hashes.get(KV260_PATH.as_posix())
    confirm_hash = hashes.get(KV260_CONFIRM_PATH.as_posix())
    kv_state = {
        "criterion": kv_row.get("terminal_criterion") if kv_row else None,
        "criterion_met": kv_row.get("terminal_criterion_met") if kv_row else None,
        "disposition": kv_row.get("disposition") if kv_row else None,
        "latency_hash": kv_hash,
        "row_latency_hash": kv_row.get("raw_transcript_hash") if kv_row else None,
        "sample_count": kv260.get("board_harness_summary", {}).get("sample_count"),
        "confirm_hash": confirm_hash,
        "overlay_loaded": kv260_confirm.get("kv260_overlay_loaded"),
        "terminal_confirmed": kv260_confirm.get("kv260_terminal_condition_confirmed"),
        "confirm_latency_hash": "sha256:"
        + str(kv260_confirm.get("kv260_terminal_transcript_sha256")),
    }
    kv_expected = {
        "criterion": KV260_TERMINAL_CRITERION,
        "criterion_met": True,
        "disposition": "graduated_preserved",
        "latency_hash": kv_hash,
        "row_latency_hash": kv_hash,
        "sample_count": 32,
        "confirm_hash": confirm_hash,
        "overlay_loaded": True,
        "terminal_confirmed": True,
        "confirm_latency_hash": kv_hash,
    }
    checks.append(
        _check(
            "kv260_synthesis_and_latency_terminal",
            f"{KV260_CONFIRM_PATH.as_posix()} + {KV260_PATH.as_posix()}",
            "synthesis/overlay and latency transcript",
            kv_expected,
            kv_state,
            kv_state == kv_expected
            and quarantines["kv260"].get("quarantined") is False
            and quarantines["kv260_confirm"].get("quarantined") is False,
        )
    )

    operator = continuity.get("operator_state_receipt")
    cutoff_hash = hashes.get(GATEMATE_CUTOFF_PATH.as_posix())
    operator_state = {
        "cutoff_experiment": operator.get("cutoff_experiment")
        if isinstance(operator, Mapping)
        else None,
        "cutoff_date": operator.get("cutoff_date") if isinstance(operator, Mapping) else None,
        "cutoff_source_hash": operator.get("cutoff_source_hash")
        if isinstance(operator, Mapping)
        else None,
        "actual_cutoff_hash": cutoff_hash,
    }
    operator_expected = {
        "cutoff_experiment": "Exp6559",
        "cutoff_date": GATEMATE_CUTOFF_DATE,
        "cutoff_source_hash": cutoff_hash,
        "actual_cutoff_hash": cutoff_hash,
    }
    checks.append(
        _check(
            "gatemate_exp6559_boundary",
            GATEMATE_CUTOFF_PATH.as_posix(),
            "cutoff experiment,date,hash",
            operator_expected,
            operator_state,
            operator_state == operator_expected
            and quarantines["gatemate_cutoff"].get("quarantined") is False,
        )
    )

    print("[phase 0 check start] optional V637 memory receipt", flush=True)
    memory_path = root / MEMORY_PATH
    memory = _read_json(memory_path)
    memory_hash = (
        sha256_file(memory_path) if memory_path.is_file() and memory_path.stat().st_size else None
    )
    if memory_hash is not None:
        hashes[MEMORY_PATH.as_posix()] = memory_hash
    memory_quarantine = _quarantine(memory, manifest, "7243") if memory else {"quarantined": False}
    memory_errors = (
        exp7243.validate_artifact(memory)
        if memory and memory_quarantine.get("quarantined") is False
        else ["unavailable_or_rejected"]
    )
    memory_accepted = bool(memory) and memory_errors == []
    checks.append(
        _check(
            "optional_v637_memory_receipt",
            MEMORY_PATH.as_posix(),
            "available_and_authenticated_or_unavailable",
            True,
            {
                "available": bool(memory),
                "accepted": memory_accepted,
                "sha256": memory_hash,
                "quarantine": memory_quarantine,
                "validator_errors": memory_errors,
            },
            True,
        )
    )
    return (
        checks,
        hashes,
        {
            "continuity": continuity,
            "polarfire_raw": raw,
            "gatemate_cutoff": cutoff,
            "kv260": kv260,
            "kv260_confirm": kv260_confirm,
            "memory": memory if memory_accepted else {},
        },
    )


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the first failed gate so a blocked verdict has one cause."""

    return next((row for row in checks if row.get("passed") is not True), None)


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all gates and expose exact fields for the first failure."""

    failed = _first_failed(checks)
    return {
        "passed": failed is None,
        "failed_check": failed.get("check") if failed else None,
        "upstream": failed.get("upstream") if failed else None,
        "artifact_field": failed.get("field") if failed else None,
        "field": failed.get("field") if failed else None,
        "expected_value": failed.get("expected_value") if failed else None,
        "observed_value": failed.get("observed_value") if failed else None,
        "checks": [dict(row) for row in checks],
    }


def _custom_receipt_rows(candidate_paths: Sequence[Path]) -> list[JsonDict]:
    """Convert explicit test or future receipt files to the approved row shape."""

    rows: list[JsonDict] = []
    for index, path in enumerate(candidate_paths, start=1):
        value = _read_json(path)
        receipt = value.get("physical_state_receipt")
        data = receipt if isinstance(receipt, Mapping) else {}
        conditions = data.get("changed_conditions")
        changed = conditions if isinstance(conditions, Mapping) else {}
        timestamp = data.get("receipt_timestamp")
        rows.append(
            {
                "row_id": f"receipt-{index:03d}",
                "path": str(path),
                "receipt_timestamp": timestamp,
                "receipt_date": str(timestamp)[:10].replace("-", "") if timestamp else None,
                "operator_authored": data.get("authored_by") == "operator",
                "material_physical_fields": [
                    name for name in ("cable", "port", "power") if changed.get(name)
                ],
                "target_ok": data.get("exists") is True,
                "valid": bool(
                    data.get("exists") is True
                    and data.get("authored_by") == "operator"
                    and timestamp
                    and str(timestamp)[:10].replace("-", "") > GATEMATE_CUTOFF_DATE
                    and any(changed.get(name) for name in ("cable", "port", "power"))
                ),
                "changed_conditions": {
                    name: changed.get(name) for name in ("cable", "port", "power")
                },
                "evidence_hash": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def search_gatemate_operator_receipts(
    root: Path,
    raw_path: Path,
    *,
    candidate_paths: Sequence[Path] | None = None,
) -> JsonDict:
    """Search approved operator sources after Exp6559 and write a raw receipt."""

    if candidate_paths is None:
        parsed = exp6559.search_dated_receipts(root, GATEMATE_CUTOFF_DATE)
        rows = [
            dict(row)
            for row in parsed
            if row.get("operator_authored") is True
            and row.get("target_ok") is True
            and bool(row.get("material_physical_fields"))
        ]
        for row in rows:
            row["receipt_timestamp"] = None
            row["changed_conditions"] = {name: None for name in ("cable", "port", "power")}
            row["evidence_hash"] = sha256_file(root / str(row["path"]))
    else:
        rows = _custom_receipt_rows(candidate_paths)
    accepted = [row for row in rows if row.get("valid") is True]
    selected = (
        max(accepted, key=lambda row: str(row.get("receipt_timestamp"))) if accepted else None
    )
    source_files = sorted({str(row.get("path")) for row in rows})
    raw = {
        "schema": "carnot.exp7244.gatemate_operator_receipt_search.v1",
        "run_date": RUN_DATE,
        "cutoff_experiment": "Exp6559",
        "cutoff_date": GATEMATE_CUTOFF_DATE,
        "search_scope": "operator-authored GateMate physical-state receipts only",
        "source_files": [
            {
                "path": path,
                "sha256": sha256_file(Path(path) if Path(path).is_absolute() else root / path),
            }
            for path in source_files
        ],
        "candidate_rows": rows,
        "accepted_receipt_count": len(accepted),
        "selected_receipt": selected,
        "explicit_absence": selected is None,
        "hardware_operations_issued": [],
        "forbidden_operations_issued": [],
    }
    _atomic_json(raw_path, raw)
    conditions = selected.get("changed_conditions", {}) if selected else {}
    return {
        "exists": selected is not None,
        "newer_than_exp6559": selected is not None,
        "cutoff_experiment": "Exp6559",
        "cutoff_date": GATEMATE_CUTOFF_DATE,
        "cutoff_source_path": GATEMATE_CUTOFF_PATH.as_posix(),
        "cutoff_source_hash": sha256_file(root / GATEMATE_CUTOFF_PATH),
        "receipt_timestamp": selected.get("receipt_timestamp") if selected else None,
        "changed_conditions": {name: conditions.get(name) for name in ("cable", "port", "power")},
        "evidence_path": selected.get("path") if selected else None,
        "evidence_hash": selected.get("evidence_hash") if selected else None,
        "explicit_absence": selected is None,
        "absence_reason": (
            None
            if selected
            else "no operator-authored GateMate cable, port, or power change after Exp6559"
        ),
        "search_receipt_path": str(raw_path),
        "search_receipt_hash": sha256_file(raw_path),
        "authorized_next_task": (
            "one bounded GateMate detect in a later task" if selected else None
        ),
        "hardware_operations_issued": [],
        "hardware_command_count": 0,
    }


def _finish_row(row: JsonDict) -> JsonDict:
    """Add comparison fields and hash one complete board row."""

    row.setdefault("arm", "aggregation_from_upstream_artifacts")
    row.setdefault("seed", None)
    row.setdefault("metric", False)
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row.pop("row_sha256", None)
    row["row_sha256"] = exp7217.sha256_json(row)
    return row


def build_board_rows(
    root: Path, upstreams: Mapping[str, JsonDict], operator: Mapping[str, Any]
) -> list[JsonDict]:
    """Build one terminal row per board without accessing any board."""

    continuity = upstreams["continuity"]
    raw = upstreams["polarfire_raw"]
    kv260 = upstreams["kv260"]
    kv260_confirm = upstreams["kv260_confirm"]
    continuity_hash = sha256_file(root / CONTINUITY_PATH)
    raw_hash = sha256_file(root / POLARFIRE_RAW_PATH)
    kv_hash = sha256_file(root / KV260_PATH)
    confirm_hash = sha256_file(root / KV260_CONFIRM_PATH)
    changed = operator.get("exists") is True
    gate_next = (
        "in a later task, run one bounded GateMate detect named by the operator receipt"
        if changed
        else "record a dated operator GateMate cable, port, or power change after Exp6559"
    )
    rows = [
        {
            "unit_id": "board:KV260",
            "board": "KV260",
            "execution_venue": "kv260",
            "processor_class": "fpga_fabric",
            "latest_receipt_path": CONTINUITY_PATH.as_posix(),
            "latest_receipt_date": continuity.get("run_date"),
            "latest_receipt_hash": continuity_hash,
            "latest_receipt_authenticated": True,
            "supporting_evidence": [
                {"path": KV260_PATH.as_posix(), "sha256": kv_hash},
                {"path": KV260_CONFIRM_PATH.as_posix(), "sha256": confirm_hash},
            ],
            "terminal_criterion": KV260_TERMINAL_CRITERION,
            "terminal_criterion_met": True,
            "observed_state": {
                "synthesis_and_overlay_confirmed": kv260_confirm.get("kv260_overlay_loaded")
                is True,
                "programmable_logic_latency_transcript": kv260.get("terminal_condition_met")
                is True,
                "sample_count": kv260.get("board_harness_summary", {}).get("sample_count"),
                "median_latency_ms": kv260.get("board_latency_median_ms"),
            },
            "disposition": "graduated_preserved",
            "exact_next_prerequisite": "none; future access uses ssh kria only",
            "host_block_devices_relevant": False,
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": True,
        },
        {
            "unit_id": "board:GateMate",
            "board": "GateMate",
            "execution_venue": "gatemate",
            "processor_class": "unavailable" if not changed else "not_executed",
            "latest_receipt_path": (
                operator.get("evidence_path") if changed else CONTINUITY_PATH.as_posix()
            ),
            "latest_receipt_date": (
                operator.get("receipt_timestamp") if changed else continuity.get("run_date")
            ),
            "latest_receipt_hash": (operator.get("evidence_hash") if changed else continuity_hash),
            "latest_receipt_authenticated": True,
            "search_receipt_path": operator.get("search_receipt_path"),
            "search_receipt_hash": operator.get("search_receipt_hash"),
            "terminal_criterion": GATEMATE_TERMINAL_CRITERION,
            "terminal_criterion_met": False,
            "observed_state": (
                "new_operator_physical_state_receipt_recorded"
                if changed
                else "no_operator_physical_state_receipt_after_exp6559"
            ),
            "disposition": (
                "authorized_later_action" if changed else "blocked_inherited_no_new_physical_state"
            ),
            "exact_next_prerequisite": gate_next,
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": changed,
            "error": None if changed else gate_next,
            "abstention": not changed,
        },
        {
            "unit_id": "board:PolarFire",
            "board": "PolarFire",
            "execution_venue": "polarfire",
            "processor_class": "cpu",
            "latest_receipt_path": POLARFIRE_RAW_PATH.as_posix(),
            "latest_receipt_date": raw.get("run_date"),
            "latest_receipt_hash": raw_hash,
            "latest_receipt_authenticated": True,
            "supporting_evidence": [
                {"path": CONTINUITY_PATH.as_posix(), "sha256": continuity_hash}
            ],
            "terminal_criterion": POLARFIRE_TERMINAL_CRITERION,
            "terminal_criterion_met": True,
            "observed_state": {
                "dispatch_completed": raw.get("dispatch_completed"),
                "binary_sha256": raw.get("binary_sha256"),
                "input_sha256": raw.get("input_sha256"),
                "input_hash_matches": raw.get("input_hash_matches"),
                "output_sha256": raw.get("workload_output_sha256"),
                "output_hash_matches": raw.get("output_hash_matches"),
                "transport_exit_code": raw.get("transport_exit_code"),
                "workload_exit_code": raw.get("workload_exit_code"),
            },
            "disposition": "graduated_cpu_dispatch_preserved",
            "exact_next_prerequisite": "none for CPU dispatch; FPGA sampling remains a separate task",
            "programmable_logic_sampling_observed": False,
            "smoke_repeated": False,
            "hardware_operations_issued": [],
            "hardware_command_count": 0,
            "metric": True,
        },
    ]
    return [_finish_row(row) for row in rows]


def operation_map(memory: Mapping[str, Any]) -> list[JsonDict]:
    """Map measured Rust work and prospective device roles without claiming fit."""

    gaps = memory.get("hardware_target_gaps")
    values = gaps if isinstance(gaps, Mapping) else {}
    available = bool(values)
    common = {
        "memory_footprint_available": available,
        "archive_mask_bytes_at_capacity_four": values.get("archive_mask_bytes_at_capacity_four"),
        "packed_active_masks_bytes": values.get("packed_active_masks_bytes"),
        "controller_state_bytes": values.get("controller_state_bytes"),
        "vendor_connectivity_degree": 16,
        "degree_16_connectivity_establishes_fit": False,
        "topology": "unknown",
        "bandwidth": "unknown",
        "power": "unknown",
        "latency": "unknown",
    }
    return [
        {
            **common,
            "target": "cpu_rust",
            "prospective": False,
            "executed_in_source": available,
            "operations": [
                "archive_mask_updates",
                "validation_counter_updates",
                "lookup_operations",
            ],
            "storage_role": "host memory and serialized controller state",
        },
        {
            **common,
            "target": "fpga_bram",
            "prospective": True,
            "executed_in_source": False,
            "operations": ["archive masks", "validation counters", "lookup tables"],
            "storage_role": "prospective BRAM placement only",
        },
        {
            **common,
            "target": "tsu",
            "prospective": True,
            "executed_in_source": False,
            "operations": ["prospective sparse lookup and counter update"],
            "storage_role": "unmapped prospective TSU role",
        },
    ]


def _memory_footprint(root: Path, memory: Mapping[str, Any]) -> JsonDict:
    """Summarize accepted memory evidence without turning absence into a block."""

    mapping = operation_map(memory)
    cpu = mapping[0]
    available = cpu["memory_footprint_available"] is True
    return {
        "available": available,
        "source_path": MEMORY_PATH.as_posix() if available else None,
        "source_hash": sha256_file(root / MEMORY_PATH) if available else None,
        "source_authenticated": available,
        "unavailable_reason": None
        if available
        else "V637 memory footprint unavailable or rejected",
        "archive_mask_bytes_at_capacity_four": cpu["archive_mask_bytes_at_capacity_four"],
        "packed_active_masks_bytes": cpu["packed_active_masks_bytes"],
        "controller_state_bytes": cpu["controller_state_bytes"],
    }


def _base_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Mapping[str, float],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:
    """Create every required field before selecting terminal status."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7244",
            "SCENARIO-ISING-7244-PREFLIGHT",
            "SCENARIO-ISING-7244-BOARDS",
            "SCENARIO-ISING-7244-GATEMATE",
            "SCENARIO-ISING-7244-PLACEMENT",
            "SCENARIO-ISING-7244-ARTIFACT",
        ],
        "status": "in_progress",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": duration_s,
        "phase_spans_s": dict(phase_spans),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned": 3,
            "attempted": 0,
            "completed": 0,
            "censored": 3,
            "independent_units_planned": 3,
            "independent_units_completed": 0,
            "stopping_rule": "one read-only disposition per board; do not repeat completed smoke tests",
        },
        "random_seed": 7244,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external_precondition",
        "acceptance_gate_results": {},
        "board_disposition_complete_score": 0,
        "board_rows": [],
        "hardware_operations_issued": [],
        "operator_state_receipt": {},
        "operation_map": [],
        "memory_footprint": {
            "available": False,
            "unavailable_reason": "aggregation did not start",
        },
        "operator_prerequisites": [],
        "external_actions": {
            "purchases": [],
            "vendor_contacts": [],
            "package_installations": [],
            "uploads": [],
            "publications": [],
        },
        "validation_receipts": {},
    }


def build_artifact(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Replay public receipts and assemble a terminal disposition in memory."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    phase_spans: dict[str, float] = {}
    phase_started = time.monotonic()
    progress(0, "start", "preconditions before disposition aggregation")
    checks, hashes, upstreams = collect_preconditions(root, paths)
    phase_spans["phase_0_preconditions"] = time.monotonic() - phase_started
    failed = _first_failed(checks)
    progress(0, "end", f"preconditions failed={int(failed is not None)}")
    if failed is not None:
        artifact = _base_artifact(
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
            phase_spans=phase_spans,
            checks=checks,
            hashes=hashes,
        )
        artifact["status"] = "blocked_external_precondition"
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
    progress(1, "start", "authenticated KV260 and PolarFire evidence replay")
    progress(1, "end", "two graduated evidence chains retained; no smoke repeated")
    phase_spans["phase_1_graduated_boards"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    progress(2, "start", "operator-authored GateMate physical-state receipt search")
    operator = search_gatemate_operator_receipts(root, paths.gatemate_search)
    hashes[
        paths.gatemate_search.relative_to(root).as_posix()
        if paths.gatemate_search.is_relative_to(root)
        else str(paths.gatemate_search)
    ] = operator["search_receipt_hash"]
    progress(
        2,
        "end",
        "GateMate receipt found=" + str(operator["exists"]).lower() + "; hardware commands=0",
    )
    phase_spans["phase_2_gatemate_search"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    progress(3, "start", "three independent board disposition rows")
    rows = build_board_rows(root, upstreams, operator)
    progress(3, "end", f"terminal board rows={len(rows)}")
    phase_spans["phase_3_board_rows"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    progress(4, "start", "memory footprint and prospective device operation map")
    mapping = operation_map(upstreams["memory"])
    footprint = _memory_footprint(root, upstreams["memory"])
    progress(
        4, "end", f"memory available={str(footprint['available']).lower()}; device fit unknown"
    )
    phase_spans["phase_4_operation_map"] = time.monotonic() - phase_started

    artifact = _base_artifact(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        checks=checks,
        hashes=hashes,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "aggregation",
            "rows": rows,
            "sample_size_budget": {
                "planned": 3,
                "attempted": 3,
                "completed": 3,
                "censored": 0,
                "independent_units_planned": 3,
                "independent_units_completed": 3,
                "stopping_rule": "one read-only disposition per board; do not repeat completed smoke tests",
            },
            "verdict_class": "positive",
            "honest_verdict": (
                "complete: all three current board dispositions and exact next conditions are "
                "recorded. KV260 synthesis-and-latency graduation and PolarFire hash-matched CPU "
                "dispatch remain independently authenticated. PolarFire CPU dispatch is not FPGA "
                "sampling. GateMate remains blocked at board-row scope because no operator-authored "
                "physical-state receipt after Exp6559 exists. No hardware operation was issued."
            ),
            "acceptance_gate_results": {
                "kv260_synthesis_and_latency_terminal": {
                    "expected": True,
                    "actual": True,
                    "pass": True,
                },
                "polarfire_end_to_end_hash_matched_cpu_dispatch": {
                    "expected": True,
                    "actual": True,
                    "pass": True,
                },
                "polarfire_programmable_logic_sampling_claimed": {
                    "expected": False,
                    "actual": False,
                    "pass": True,
                },
                "gatemate_disposition_recorded": {
                    "expected": True,
                    "actual": True,
                    "pass": True,
                },
                "hardware_operations_issued_count": {
                    "expected": 0,
                    "actual": 0,
                    "pass": True,
                },
                "three_exact_next_conditions_recorded": {
                    "expected": 3,
                    "actual": 3,
                    "pass": True,
                },
            },
            "board_disposition_complete_score": 1,
            "board_rows": rows,
            "hardware_operations_issued": [],
            "operator_state_receipt": operator,
            "operation_map": mapping,
            "memory_footprint": footprint,
            "operator_prerequisites": [
                {
                    "board": "GateMate",
                    "required": not operator["exists"],
                    "condition": "dated operator cable, port, or power change after Exp6559",
                    "next_task": "one bounded GateMate detect only after that receipt",
                },
                {
                    "board": "KV260",
                    "required": False,
                    "condition": "ssh kria is the sole access precondition for any future task",
                    "host_block_devices_relevant": False,
                },
                {
                    "board": "prospective FPGA or TSU placement",
                    "required": True,
                    "condition": "measure topology, bandwidth, power, and latency before a fit claim",
                },
            ],
            "validation_receipts": {
                "public_input_replay": {
                    "producer": "build_artifact",
                    "source_count": len(hashes),
                    "board_rows_produced": len(rows),
                    "passed": len(rows) == 3,
                },
                "independent_reducer": {
                    "validator": "validate_artifact",
                    "scheduled_before_terminal_write": True,
                },
            },
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(4, "end", "terminal artifact assembled in memory")
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Independently recompute identity, row, claim, and hash invariants."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(artifact.get("schema") != SCHEMA, "schema")
    add(
        artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("task_id") != TASK_ID
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(
        artifact.get("execution_venue") != "host" or not artifact.get("execution_host"),
        "execution_identity",
    )
    add(
        not isinstance(artifact.get("duration_s"), (int, float)) or artifact["duration_s"] < 0,
        "duration",
    )
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            datetime.fromisoformat(str(artifact.get(field)))
        except ValueError:
            add(True, field)
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("model_invocation_count") != 0,
        "model_declaration",
    )
    add(artifact.get("verifier_is_oracle") is not False, "verifier_authority")
    add(artifact.get("hardware_operations_issued") != [], "hardware_operations")
    external = artifact.get("external_actions")
    add(
        not isinstance(external, Mapping)
        or any(
            external.get(name) != []
            for name in (
                "purchases",
                "vendor_contacts",
                "package_installations",
                "uploads",
                "publications",
            )
        ),
        "external_actions",
    )
    try:
        checksum_ok = artifact.get("reproducibility_checksum") == artifact_checksum(artifact)
    except (TypeError, ValueError):
        checksum_ok = False
    add(not checksum_ok, "reproducibility_checksum")

    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        summary = artifact.get("gate_check_summary")
        add(artifact.get("status") != "blocked_external_precondition", "blocked_status")
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run",
            "blocked_substrate",
        )
        add(artifact.get("rows") != [] or artifact.get("board_rows") != [], "blocked_rows")
        add(artifact.get("board_disposition_complete_score") != 0, "blocked_completion_score")
        add(
            not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or any(
                summary.get(name) is None
                for name in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            ),
            "blocked_gate_summary",
        )
        return errors

    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation",
        "substrate",
    )
    rows = artifact.get("board_rows")
    by_board = (
        {
            row.get("board"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("board"), str)
        }
        if isinstance(rows, list)
        else {}
    )
    row_schema_ok = all(
        {"unit_id", "arm", "seed", "metric", "error", "abstention", "row_sha256"} <= set(row)
        and row.get("row_sha256")
        == exp7217.sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and row.get("latest_receipt_authenticated") is True
        and bool(row.get("latest_receipt_path"))
        and bool(row.get("latest_receipt_date"))
        and bool(row.get("latest_receipt_hash"))
        and bool(row.get("exact_next_prerequisite"))
        and row.get("hardware_operations_issued") == []
        and row.get("hardware_command_count") == 0
        for row in by_board.values()
    )
    board_ok = (
        set(by_board) == {"KV260", "GateMate", "PolarFire"}
        and row_schema_ok
        and by_board["KV260"].get("terminal_criterion") == KV260_TERMINAL_CRITERION
        and by_board["KV260"].get("terminal_criterion_met") is True
        and by_board["KV260"].get("disposition") == "graduated_preserved"
        and by_board["KV260"].get("host_block_devices_relevant") is False
        and by_board["PolarFire"].get("terminal_criterion") == POLARFIRE_TERMINAL_CRITERION
        and by_board["PolarFire"].get("terminal_criterion_met") is True
        and by_board["PolarFire"].get("processor_class") == "cpu"
        and by_board["PolarFire"].get("programmable_logic_sampling_observed") is False
        and by_board["PolarFire"].get("smoke_repeated") is False
        and by_board["GateMate"].get("disposition")
        in {"blocked_inherited_no_new_physical_state", "authorized_later_action"}
    )
    add(not board_ok, "board_rows")
    add(artifact.get("rows") != rows, "rows")
    add(artifact.get("board_disposition_complete_score") != int(board_ok), "completion_score")

    operator = artifact.get("operator_state_receipt")
    operator_ok = (
        isinstance(operator, Mapping)
        and operator.get("hardware_operations_issued") == []
        and operator.get("hardware_command_count") == 0
        and operator.get("cutoff_experiment") == "Exp6559"
        and operator.get("cutoff_date") == GATEMATE_CUTOFF_DATE
        and bool(operator.get("search_receipt_path"))
        and bool(operator.get("search_receipt_hash"))
    )
    if operator_ok and operator.get("exists") is True:
        conditions = operator.get("changed_conditions")
        operator_ok = bool(
            operator.get("receipt_timestamp")
            and operator.get("evidence_hash")
            and isinstance(conditions, Mapping)
            and any(conditions.get(name) for name in ("cable", "port", "power"))
            and operator.get("authorized_next_task")
        )
    elif operator_ok:
        operator_ok = (
            operator.get("explicit_absence") is True
            and operator.get("receipt_timestamp") is None
            and operator.get("evidence_hash") is None
            and operator.get("authorized_next_task") is None
        )
    add(not operator_ok, "operator_state_receipt")

    mapping = artifact.get("operation_map")
    mapped = (
        {
            row.get("target"): row
            for row in mapping
            if isinstance(row, Mapping) and isinstance(row.get("target"), str)
        }
        if isinstance(mapping, list)
        else {}
    )
    mapping_ok = (
        set(mapped) == {"cpu_rust", "fpga_bram", "tsu"}
        and "validation_counter_updates" in mapped["cpu_rust"].get("operations", [])
        and mapped["fpga_bram"].get("prospective") is True
        and mapped["tsu"].get("prospective") is True
        and all(
            row.get("degree_16_connectivity_establishes_fit") is False for row in mapped.values()
        )
        and all(row.get("topology") == "unknown" for row in mapped.values())
        and all(row.get("bandwidth") == "unknown" for row in mapped.values())
        and all(row.get("power") == "unknown" for row in mapped.values())
        and all(row.get("latency") == "unknown" for row in mapped.values())
    )
    add(not mapping_ok, "operation_map")
    footprint = artifact.get("memory_footprint")
    add(
        not isinstance(footprint, Mapping)
        or footprint.get("available")
        is not mapped.get("cpu_rust", {}).get("memory_footprint_available"),
        "memory_footprint",
    )
    budget = artifact.get("sample_size_budget")
    add(
        not isinstance(budget, Mapping)
        or budget.get("planned") != 3
        or budget.get("attempted") != 3
        or budget.get("completed") != 3
        or budget.get("censored") != 0
        or budget.get("independent_units_planned") != 3
        or budget.get("independent_units_completed") != 3,
        "sample_size_budget",
    )
    gates = artifact.get("acceptance_gate_results")
    add(
        not isinstance(gates, Mapping)
        or set(gates)
        != {
            "kv260_synthesis_and_latency_terminal",
            "polarfire_end_to_end_hash_matched_cpu_dispatch",
            "polarfire_programmable_logic_sampling_claimed",
            "gatemate_disposition_recorded",
            "hardware_operations_issued_count",
            "three_exact_next_conditions_recorded",
        }
        or any(
            not isinstance(value, Mapping) or value.get("pass") is not True
            for value in gates.values()
        ),
        "acceptance_gate_results",
    )
    add(
        artifact.get("status") != "complete"
        or artifact.get("verdict_class") != "positive"
        or not str(artifact.get("honest_verdict", "")).startswith("complete:")
        or artifact.get("gate_check_summary", {}).get("passed") is not True,
        "terminal_state",
    )
    if root is not None:
        hashes = artifact.get("source_artifact_hashes")
        add(
            not isinstance(hashes, Mapping)
            or any(
                not (Path(path) if Path(path).is_absolute() else root / path).is_file()
                or sha256_file(Path(path) if Path(path).is_absolute() else root / path) != digest
                for path, digest in hashes.items()
            ),
            "source_artifact_hashes",
        )
    return errors


def atomic_write(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish only complete or externally blocked evidence."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7244 artifact: {errors}")
    _atomic_json(path, artifact)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, paths: ExperimentPaths) -> JsonDict:
    """Replay public input, reduce it independently, and publish atomically."""

    artifact = build_artifact(root, paths)
    progress(5, "before", "final independent artifact validation")
    errors = validate_artifact(artifact, root=root)
    progress(5, "after", f"final independent artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7244 artifact: {errors}")
    progress(6, "before", "atomic terminal write")
    receipt = atomic_write(paths.artifact, artifact)
    progress(6, "after", f"atomic terminal write bytes={receipt['bytes']}")
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Keep production replay and read-only validation on one entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-search", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only review or validate existing bytes without mutation."""

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
        defaults = ExperimentPaths.defaults(root)
        paths = ExperimentPaths(
            args.output or defaults.artifact,
            args.raw_search or defaults.gatemate_search,
            args.checkpoint or defaults.checkpoint,
        )
        paths = ExperimentPaths(
            paths.artifact if paths.artifact.is_absolute() else root / paths.artifact,
            paths.gatemate_search
            if paths.gatemate_search.is_absolute()
            else root / paths.gatemate_search,
            paths.checkpoint if paths.checkpoint.is_absolute() else root / paths.checkpoint,
        )
        artifact = run_experiment(root, paths)
        print(
            f"experiment_complete status={artifact['status']} "
            f"score={artifact['board_disposition_complete_score']} output={paths.artifact}",
            flush=True,
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment_error type={type(exc).__name__} message={exc}", flush=True)
        return 2
