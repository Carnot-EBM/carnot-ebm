"""Bind the V650 contract and qualify durable experiment evidence.

The experiment uses host-only aggregation. It does not repeat missing V649
science. Its bundle controls test whether future producers can retain raw rows
and resume safely after an interrupted atomic write.

Spec refs: REQ-REPORT-7409 and SCENARIO-REPORT-7409-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import (
    _field_declared,
    parse_markdown_contract,
    parse_yaml_contract,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
EXPERIMENT_ID = "exp7409-evidence-custody"
SCHEMA = "carnot.exp7409.v650.evidence_custody.v1"
PHASE = 1
RANDOM_SEED = {"bundle_units": 7_409_650_01, "fault_controls": 7_409_650_02}
RESULT_SIZE_LIMIT_BYTES = 20 * 1024 * 1024

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
RESULT_PATH = Path("results/experiment_7409_v650_evidence_custody.json")
RAW_DIR = Path("results/raw/experiment_7409_v650_evidence_custody")
MODULE_PATH = Path("python/carnot/experiment_7409_v650_evidence_custody.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7409_v650_evidence_custody.py")
TEST_PATH = Path("tests/python/test_experiment_7409_v650_evidence_custody.py")
BUNDLE_CHECKPOINT_NAME = "checkpoint.json"
BUNDLE_MANIFEST_NAME = "manifest.json"
BUNDLE_PROTOCOL_ID = "carnot.exp7409.bundle.v1"

EXPECTED_TASK_IDS = (
    "exp7409-evidence-custody",
    "exp7410-source-corpus",
    "exp7411-arc-call-budget",
    "exp7412-source-features",
    "exp7413-source-calibration",
    "exp7414-selected-feedback",
    "exp7415-decision-audit",
    "exp7416-anchored-extraction",
    "exp7417-extraction-audit",
    "exp7418-revision-memory",
    "exp7419-precision-placement",
    "exp7420-capstone",
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-studying.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    DESIGN_PATH,
    ROADMAP_PATH,
    Path("scripts/experiment_template.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7395_v649_receipt_protocol.py"),
    Path("python/carnot/experiment_7408_v649_capstone.py"),
    Path("results/experiment_7408_v649_capstone.json"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

PRODUCERS = (
    {
        "experiment_id": "exp7397-delayed-adapter",
        "terminal": Path("results/experiment_7397_v649_delayed_adapter.json"),
        "module": Path("python/carnot/experiment_7397_v649_delayed_adapter.py"),
        "entrypoint": Path("scripts/experiments/experiment_7397_v649_delayed_adapter.py"),
        "raw_dir": Path("results/raw/experiment_7397_v649_delayed_adapter"),
        "checkpoint_dir": Path("results/checkpoints/experiment_7397_v649_delayed_adapter"),
        "record_sidecar": Path(
            "results/raw/experiment_7399_v649_online_trial/upstream_exp7397_context.json"
        ),
    },
    {
        "experiment_id": "exp7399-online-trial",
        "terminal": Path("results/experiment_7399_v649_online_trial.json"),
        "module": Path("python/carnot/experiment_7399_v649_online_trial.py"),
        "entrypoint": Path("scripts/experiments/experiment_7399_v649_online_trial.py"),
        "raw_dir": Path("results/raw/experiment_7399_v649_online_trial"),
        "checkpoint_dir": Path("results/checkpoints/experiment_7399_v649_online_trial"),
        "record_sidecar": None,
    },
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "phase",
    "status",
    "run_date",
    "started_at_utc",
    "ended_at_utc",
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
    "evidence_custody_ready_score",
    "missing_producer_rows",
    "contract_rows",
    "custody_fault_rows",
    "deferred_size_gate",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned ordinary top-level fields include experiment_id, milestone, and terminal status.",
    "run_date": "The frozen date is 20260919, with actual UTC start and end timestamps.",
    "preconditions_checked": "Exact paths, hashes, and resource checks precede dependent work.",
    "MODEL_SPECS": "Current LLM calls require unsloth/Qwen3.8-27B-GGUF; this host aggregation uses none.",
    "model_invoked": "Only an actual current attempted LLM call makes this true.",
    "invocation_counts": "Owned current attempted, completed, failed, cancelled, and in-flight events determine counts.",
    "inference_substrate": "This is a truthful string; device and software details are separate.",
    "inference_substrate_class": "The class describes current aggregation without duration padding.",
    "execution_venue": "The closed venue string is host; device details are separate.",
    "duration_s": "Monotonic current duration is separate from historical and validation time.",
    "phase_spans": "Real phase boundaries retain checkpoints and heartbeat counts.",
    "random_seed": "Frozen fitting, sampling, and resampling seeds are explicit or null when inapplicable.",
    "reproducibility_checksum": "The checksum binds code, protocol, input bytes, and raw rows.",
    "source_artifact_hashes": "Exact source paths and hashes retain original sidecar facts.",
    "rows": "Every contract, producer, and fault-control unit keeps its disposition.",
    "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted counts are separate.",
    "acceptance_gate_results": "Each check records category, operator, expected, observed, pass state, and principle.",
    "gate_check_summary": "Blocked inputs name exact upstream paths, checks, fields, and expected and observed values.",
    "verifier_is_oracle": "Custody verification does not share authority with later scientific correctness.",
    "honest_verdict": "Completed findings use complete_; unchanged external absence stays blocked in producer rows.",
    "verdict_class": "The closed class is positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Critical findings remain visible and cannot supply readiness.",
    "validation_receipts": "Exact argv, scoped environment, exits, durations, required names, and hashed logs are retained.",
    "field_principles": "Fields are explained separately; gate scalars stay ordinary values.",
    "promotion_score": "This remains zero; no rollout, external publication, or generator update follows.",
    "evidence_custody_ready_score": "One certifies only this bundle and the twelve-task comparison.",
    "missing_producer_rows": "Each prior path keeps observed bytes, recorded hashes, and bounded recovery disposition.",
    "contract_rows": "Twelve records are parsed independently from both authorities.",
    "custody_fault_rows": "Each interruption or corruption fixture retains its expected recovery result.",
    "deferred_size_gate": "A reviewable outer-loop change remains needed; conductor protection is not asserted.",
}


def utc_now() -> str:
    """Return an aware UTC timestamp for an observed boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a measured phase boundary so long work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7409] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large checkpoint at once."""

    return current_work_receipt.sha256_file(path)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so removed evidence changes the identity."""

    return current_work_receipt.canonical_hash(value)


def load_json(path: Path) -> JsonDict:
    """Read one JSON mapping and return an empty mapping for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def load_yaml(path: Path) -> JsonDict:
    """Read one YAML mapping and return an empty mapping for malformed bytes."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _public_contract_row(row: Mapping[str, Any]) -> JsonDict:
    """Keep only fields that both contract authorities must declare."""

    return {
        key: deepcopy(row.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def compare_contract_authorities(
    markdown: str,
    roadmap: Mapping[str, Any],
    *,
    quarantined_ids: Sequence[str] = (),
) -> JsonDict:
    """Compare generic parser outputs and verify each consumed producer field."""

    errors: list[str] = []
    try:
        markdown_contract = parse_markdown_contract(markdown)
    except ValueError as exc:
        markdown_contract = {"milestone": None, "tasks": []}
        errors.append(f"markdown_parse_error:{exc}")
    try:
        yaml_contract = parse_yaml_contract(roadmap)
    except ValueError as exc:
        yaml_contract = {"milestone": None, "tasks": []}
        errors.append(f"yaml_parse_error:{exc}")

    markdown_rows = list(markdown_contract.get("tasks") or [])
    yaml_rows = list(yaml_contract.get("tasks") or [])
    yaml_tasks = {
        task.get("id"): task for task in roadmap.get("tasks", []) if isinstance(task, Mapping)
    }
    rows: list[JsonDict] = []
    for index in range(max(len(markdown_rows), len(yaml_rows), len(EXPECTED_TASK_IDS))):
        expected = markdown_rows[index] if index < len(markdown_rows) else None
        observed = yaml_rows[index] if index < len(yaml_rows) else None
        task_id = str((observed or expected or {}).get("id") or "")
        gate_declarations: list[JsonDict] = []
        for gate in (observed or {}).get("gates") or []:
            producer = yaml_tasks.get(gate.get("upstream"), {})
            field = str(gate.get("artifact_field") or "")
            declared = _field_declared(str(producer.get("prompt") or ""), field)
            gate_declarations.append(
                {
                    "upstream": gate.get("upstream"),
                    "artifact_field": field,
                    "declared": declared,
                }
            )
        fields_match = bool(
            expected
            and observed
            and _public_contract_row(expected) == _public_contract_row(observed)
        )
        producer_fields_declared = all(row["declared"] for row in gate_declarations)
        quarantined = task_id in set(quarantined_ids)
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id or None,
                "title": (observed or expected or {}).get("title"),
                "markdown": _public_contract_row(expected or {}),
                "yaml": _public_contract_row(observed or {}),
                "fields_match": fields_match,
                "producer_field_rows": gate_declarations,
                "producer_fields_declared": producer_fields_declared,
                "quarantined_input": quarantined,
                "passed": fields_match and producer_fields_declared and not quarantined,
            }
        )

    markdown_milestone = markdown_contract.get("milestone")
    yaml_milestone = yaml_contract.get("milestone")
    if markdown_milestone != MILESTONE:
        errors.append("markdown_milestone_mismatch")
    if yaml_milestone != MILESTONE:
        errors.append("yaml_milestone_mismatch")
    if len(markdown_rows) != len(EXPECTED_TASK_IDS):
        errors.append("markdown_task_count_mismatch")
    if len(yaml_rows) != len(EXPECTED_TASK_IDS):
        errors.append("yaml_task_count_mismatch")
    if [row.get("id") for row in markdown_rows] != list(EXPECTED_TASK_IDS):
        errors.append("markdown_order_mismatch")
    if [row.get("id") for row in yaml_rows] != list(EXPECTED_TASK_IDS):
        errors.append("yaml_order_mismatch")
    if any(not row["passed"] for row in rows):
        errors.append("contract_row_mismatch")
    for task_id in quarantined_ids:
        errors.append(f"quarantined_input:{task_id}")
    return {
        "milestone": yaml_milestone,
        "markdown_milestone": markdown_milestone,
        "yaml_milestone": yaml_milestone,
        "contract_rows": rows,
        "errors": list(dict.fromkeys(errors)),
        "passed": not errors,
        "advisory_only": True,
    }


def run_contract_mutation_controls(markdown: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise the five required private contract defects."""

    cases: list[tuple[str, str, JsonDict, tuple[str, ...]]] = []
    removed = deepcopy(dict(roadmap))
    removed["tasks"] = list(removed.get("tasks") or [])[:-1]
    cases.append(("removed", markdown, removed, ()))

    reordered = deepcopy(dict(roadmap))
    reordered_tasks = list(reordered.get("tasks") or [])
    reordered_tasks[0], reordered_tasks[1] = reordered_tasks[1], reordered_tasks[0]
    reordered["tasks"] = reordered_tasks
    cases.append(("reordered", markdown, reordered, ()))

    stale = deepcopy(dict(roadmap))
    stale["milestone"] = "2026.09.649"
    cases.append(("stale_milestone", markdown, stale, ()))

    wrong = deepcopy(dict(roadmap))
    wrong_tasks = list(wrong.get("tasks") or [])
    wrong_tasks[0] = {**wrong_tasks[0], "title": "wrong title"}
    wrong["tasks"] = wrong_tasks
    cases.append(("wrong_field", markdown, wrong, ()))
    cases.append(("quarantined_input", markdown, deepcopy(dict(roadmap)), (EXPECTED_TASK_IDS[0],)))

    rows: list[JsonDict] = []
    for name, changed_markdown, changed_roadmap, quarantined in cases:
        result = compare_contract_authorities(
            changed_markdown,
            changed_roadmap,
            quarantined_ids=quarantined,
        )
        rows.append(
            {
                "mutation": name,
                "expected": "rejected",
                "observed_errors": result["errors"],
                "rejected": result["passed"] is False,
                "passed": result["passed"] is False,
            }
        )
    return rows


def _git_output(root: Path, argv: Sequence[str]) -> tuple[bool, str]:
    """Run one read-only Git query and keep an unknown state on tool failure."""

    result = subprocess.run(  # noqa: S603 - fixed executable and argument vector.
        ("git", *argv),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip()


def _tracked_state(root: Path, relative: Path) -> tuple[bool | None, list[str]]:
    """Return current tracking and commits that named the exact path."""

    tracked_ok, tracked_output = _git_output(root, ("ls-files", "--", relative.as_posix()))
    history_ok, history_output = _git_output(
        root,
        ("log", "--all", "--format=%H", "--", relative.as_posix()),
    )
    tracked = bool(tracked_output) if tracked_ok else None
    history = history_output.splitlines() if history_ok and history_output else []
    return tracked, history


@lru_cache(maxsize=8)
def _directory_manifest(root_text: str, relative_text: str) -> tuple[JsonDict, ...]:
    """Hash one immutable support tree once per process."""

    root = Path(root_text)
    relative = Path(relative_text)
    directory = root / relative
    rows: list[JsonDict] = []
    if not directory.is_dir():
        return ()
    tracked_ok, tracked_output = _git_output(root, ("ls-files", "--", relative.as_posix()))
    tracked_paths = set(tracked_output.splitlines()) if tracked_ok else set()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        label = path.relative_to(root)
        rows.append(
            {
                "path": label.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "tracked": label.as_posix() in tracked_paths if tracked_ok else None,
            }
        )
    return tuple(rows)


@lru_cache(maxsize=32)
def _path_state(root: Path, relative: Path) -> tuple[str, bool | None, list[str]]:
    """Distinguish present, missing, stripped, untracked, and unknown paths."""

    tracked, history = _tracked_state(root, relative)
    exists = (root / relative).is_file()
    if exists and tracked:
        state = "present_tracked"
    elif exists:
        state = "present_untracked"
    elif history:
        state = "stripped"
    elif tracked is False:
        state = "missing"
    else:
        state = "unknown"
    return state, tracked, history


def inspect_missing_producers(root: Path) -> list[JsonDict]:
    """Inspect only named V649 producer paths and their bound support bytes."""

    rows: list[JsonDict] = []
    resolved = root.resolve()
    for producer in PRODUCERS:
        terminal = producer["terminal"]
        module = producer["module"]
        entrypoint = producer["entrypoint"]
        terminal_state, terminal_tracked, terminal_history = _path_state(resolved, terminal)
        module_state, _module_tracked, module_history = _path_state(resolved, module)
        entrypoint_state, _entry_tracked, entrypoint_history = _path_state(resolved, entrypoint)
        checkpoint_rows = list(
            _directory_manifest(str(resolved), producer["checkpoint_dir"].as_posix())
        )
        raw_rows = list(_directory_manifest(str(resolved), producer["raw_dir"].as_posix()))
        record_sidecar = producer["record_sidecar"]
        recorded: JsonDict = load_json(resolved / record_sidecar) if record_sidecar else {}
        recorded_hash = recorded.get("source_sha256")
        observed_hash = (
            sha256_file(resolved / terminal) if (resolved / terminal).is_file() else None
        )
        exact_prior_identity_available = bool(recorded_hash and observed_hash == recorded_hash)
        rows.append(
            {
                "experiment_id": producer["experiment_id"],
                "declared_path": terminal.as_posix(),
                "terminal_state": "present" if (resolved / terminal).is_file() else "missing",
                "terminal_path_state": terminal_state,
                "terminal_tracked": terminal_tracked,
                "terminal_history_commits": terminal_history,
                "observed_bytes": (resolved / terminal).stat().st_size
                if (resolved / terminal).is_file()
                else None,
                "observed_terminal_sha256": observed_hash,
                "recorded_terminal_sha256": recorded_hash,
                "recorded_identity_sidecar": record_sidecar.as_posix() if record_sidecar else None,
                "recorded_status": recorded.get("status"),
                "recorded_verdict_class": recorded.get("verdict_class"),
                "recorded_flagged_adversarial": recorded.get("flagged_adversarial"),
                "producer_module_path": module.as_posix(),
                "producer_module_state": module_state,
                "producer_module_history_commits": module_history,
                "producer_entrypoint_path": entrypoint.as_posix(),
                "producer_entrypoint_state": entrypoint_state,
                "producer_entrypoint_history_commits": entrypoint_history,
                "checkpoint_file_count": len(checkpoint_rows),
                "checkpoint_manifest": checkpoint_rows,
                "checkpoint_manifest_sha256": canonical_hash(checkpoint_rows),
                "raw_evidence_file_count": len(raw_rows),
                "raw_evidence_manifest": raw_rows,
                "raw_evidence_manifest_sha256": canonical_hash(raw_rows),
                "exact_prior_identity_available": exact_prior_identity_available,
                "recovery_performed": False,
                "recovery_disposition": (
                    "not_needed_terminal_present"
                    if observed_hash
                    else "not_recovered_no_authenticated_prior_bytes"
                ),
                "history_rewrite_cause_established": False,
                "absence_cause": "unknown" if observed_hash is None else None,
                "science_eligible": False,
                "availability_verdict": "blocked_missing_declared_artifact",
            }
        )
    return rows


class EvidenceBundle:
    """Publish raw units before a numeric checkpoint and final hash manifest."""

    def __init__(
        self,
        directory: Path,
        *,
        protocol_id: str,
        max_payload_bytes: int = RESULT_SIZE_LIMIT_BYTES,
    ) -> None:
        self.directory = directory
        self.protocol_id = protocol_id
        self.max_payload_bytes = max_payload_bytes
        self.rows_dir = directory / "rows"
        self.checkpoint_path = directory / BUNDLE_CHECKPOINT_NAME
        self.manifest_path = directory / BUNDLE_MANIFEST_NAME

    def _checkpoint(self) -> JsonDict:
        """Load a valid checkpoint or create the empty numeric state."""

        if not self.checkpoint_path.exists():
            return {
                "schema": "carnot.evidence_bundle.checkpoint.v1",
                "protocol_id": self.protocol_id,
                "completed_units": [],
                "numeric_state": {"completed_unit_count": 0, "value_sum": 0.0},
            }
        try:
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("checkpoint_mapping_required") from exc
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint_mapping_required")
        if value.get("protocol_id") != self.protocol_id:
            raise ValueError("stale_completion_protocol")
        units = value.get("completed_units")
        if not isinstance(units, list):
            raise ValueError("checkpoint_units_required")
        return dict(value)

    def record_unit(self, row: Mapping[str, Any]) -> JsonDict:
        """Write one raw row, then advance the checkpoint exactly once."""

        unit_id = row.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("unit_id_required")
        encoded = (json.dumps(dict(row), indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > self.max_payload_bytes:
            raise ValueError("oversized_payload")
        checkpoint = self._checkpoint()
        existing = {
            item.get("unit_id"): item
            for item in checkpoint["completed_units"]
            if isinstance(item, Mapping)
        }
        row_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if unit_id in existing:
            if existing[unit_id].get("sha256") != row_hash:
                raise ValueError(f"unit_conflict:{unit_id}")
            return dict(existing[unit_id])

        self.rows_dir.mkdir(parents=True, exist_ok=True)
        path = self.rows_dir / f"{unit_id}.json"
        current_work_receipt.atomic_json(path, dict(row))
        reference = {
            "unit_id": unit_id,
            "path": path.relative_to(self.directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        completed = [*checkpoint["completed_units"], reference]
        values = [
            float(item.get("value", 0.0))
            for item in self._read_references(completed)
            if isinstance(item.get("value", 0.0), (int, float))
            and not isinstance(item.get("value", 0.0), bool)
        ]
        new_checkpoint = {
            "schema": "carnot.evidence_bundle.checkpoint.v1",
            "protocol_id": self.protocol_id,
            "completed_units": completed,
            "numeric_state": {
                "completed_unit_count": len(completed),
                "value_sum": sum(values),
            },
        }
        current_work_receipt.atomic_json(self.checkpoint_path, new_checkpoint)
        return reference

    def _read_references(self, references: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        """Reload exact referenced rows and reject missing or changed bytes."""

        rows: list[JsonDict] = []
        seen: set[str] = set()
        for reference in references:
            unit_id = str(reference.get("unit_id") or "")
            if not unit_id or unit_id in seen:
                raise ValueError("checkpoint_unit_duplicate")
            seen.add(unit_id)
            path = self.directory / str(reference.get("path") or "")
            if not path.is_file():
                raise ValueError(f"missing_sidecar:{unit_id}")
            if sha256_file(path) != reference.get("sha256"):
                raise ValueError(f"corrupted_bytes:{unit_id}")
            value = load_json(path)
            if value.get("unit_id") != unit_id:
                raise ValueError(f"unit_identity_mismatch:{unit_id}")
            rows.append(value)
        return rows

    def recover_rows(self) -> list[JsonDict]:
        """Recover checkpointed units in their original order exactly once."""

        checkpoint = self._checkpoint()
        return self._read_references(checkpoint["completed_units"])

    def finalize(self) -> JsonDict:
        """Bind raw rows and numeric state after every unit is durable."""

        checkpoint = self._checkpoint()
        rows = self.recover_rows()
        manifest = {
            "schema": "carnot.evidence_bundle.manifest.v1",
            "protocol_id": self.protocol_id,
            "completed_unit_count": len(rows),
            "numeric_state": deepcopy(checkpoint["numeric_state"]),
            "checkpoint": {
                "path": BUNDLE_CHECKPOINT_NAME,
                "size_bytes": self.checkpoint_path.stat().st_size,
                "sha256": sha256_file(self.checkpoint_path),
            },
            "raw_files": deepcopy(checkpoint["completed_units"]),
        }
        current_work_receipt.atomic_json(self.manifest_path, manifest)
        return manifest


def verify_bundle(directory: Path, *, expected_protocol_id: str) -> list[str]:
    """Reload a bundle and name missing, corrupt, stale, or oversized evidence."""

    manifest_path = directory / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        return ["missing_manifest"]
    manifest = load_json(manifest_path)
    errors: list[str] = []
    if manifest.get("protocol_id") != expected_protocol_id:
        errors.append("stale_completion")
    bundle = EvidenceBundle(directory, protocol_id=expected_protocol_id)
    try:
        rows = bundle.recover_rows()
    except ValueError as exc:
        rows = []
        errors.append(str(exc).split(":", 1)[0])
    for reference in manifest.get("raw_files") or []:
        path = directory / str(reference.get("path") or "")
        if not path.is_file():
            errors.append("missing_sidecar")
        elif path.stat().st_size > RESULT_SIZE_LIMIT_BYTES:
            errors.append("oversized_payload")
        elif sha256_file(path) != reference.get("sha256"):
            errors.append("corrupted_bytes")
    checkpoint = manifest.get("checkpoint") or {}
    checkpoint_path = directory / str(checkpoint.get("path") or "")
    if not checkpoint_path.is_file() or (
        checkpoint_path.is_file() and sha256_file(checkpoint_path) != checkpoint.get("sha256")
    ):
        errors.append("checkpoint_hash_mismatch")
    if manifest.get("completed_unit_count") != len(rows):
        errors.append("completed_unit_count_mismatch")
    return list(dict.fromkeys(errors))


def _fault_child(path: Path) -> None:  # pragma: no cover - abrupt subprocess fixture.
    """Flush temporary bytes, then exit before the atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-kill")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({"status": "bootstrap_must_not_publish"}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os._exit(91)


def run_bundle_fault_controls(directory: Path) -> list[JsonDict]:
    """Fault-inject interruption, loss, corruption, staleness, and excess size."""

    directory.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []

    kill_terminal = directory / "kill_before_rename" / "terminal.json"
    child = subprocess.run(  # noqa: S603 - owned Python child with fixed module.
        (
            sys.executable,
            "-u",
            "-m",
            "carnot.experiment_7409_v650_evidence_custody",
            "--fault-child",
            str(kill_terminal),
        ),
        check=False,
    )
    restart = EvidenceBundle(directory / "kill_before_rename" / "bundle", protocol_id="kill-v1")
    restart.record_unit({"unit_id": "u1", "value": 1.0})
    restart.record_unit({"unit_id": "u1", "value": 1.0})
    restart.finalize()
    recovered = restart.recover_rows()
    rows.append(
        {
            "fault": "kill_before_rename",
            "expected": "nonzero_exit_no_terminal_and_one_recovered_unit",
            "observed_exit_code": child.returncode,
            "terminal_path_observed": kill_terminal.exists(),
            "recovered_unit_count": len(recovered),
            "passed": child.returncode == 91 and not kill_terminal.exists() and len(recovered) == 1,
        }
    )

    missing = EvidenceBundle(directory / "missing_sidecar", protocol_id="missing-v1")
    missing.record_unit({"unit_id": "u1", "value": 1.0})
    missing.finalize()
    (missing.rows_dir / "u1.json").unlink()
    missing_errors = verify_bundle(missing.directory, expected_protocol_id="missing-v1")
    rows.append(
        {
            "fault": "missing_sidecar",
            "expected": "missing_sidecar",
            "observed_errors": missing_errors,
            "passed": "missing_sidecar" in missing_errors,
        }
    )

    corrupt = EvidenceBundle(directory / "corrupted_bytes", protocol_id="corrupt-v1")
    corrupt.record_unit({"unit_id": "u1", "value": 1.0})
    corrupt.finalize()
    (corrupt.rows_dir / "u1.json").write_text('{"unit_id":"u1","value":2}\n', encoding="utf-8")
    corrupt_errors = verify_bundle(corrupt.directory, expected_protocol_id="corrupt-v1")
    rows.append(
        {
            "fault": "corrupted_bytes",
            "expected": "corrupted_bytes",
            "observed_errors": corrupt_errors,
            "passed": "corrupted_bytes" in corrupt_errors,
        }
    )

    stale = EvidenceBundle(directory / "stale_completion", protocol_id="old-v1")
    stale.record_unit({"unit_id": "u1", "value": 1.0})
    stale.finalize()
    stale_errors = verify_bundle(stale.directory, expected_protocol_id="new-v2")
    rows.append(
        {
            "fault": "stale_completion",
            "expected": "stale_completion",
            "observed_errors": stale_errors,
            "passed": "stale_completion" in stale_errors,
        }
    )

    oversized_bytes = RESULT_SIZE_LIMIT_BYTES + 1
    oversized = EvidenceBundle(directory / "oversized_payload", protocol_id="large-v1")
    oversized_rejected = False
    try:
        oversized.record_unit({"unit_id": "u1", "payload": "x" * oversized_bytes})
    except ValueError as exc:
        oversized_rejected = str(exc) == "oversized_payload"
    rows.append(
        {
            "fault": "oversized_payload",
            "expected": f"<={RESULT_SIZE_LIMIT_BYTES}",
            "observed_bytes": oversized_bytes,
            "passed": oversized_rejected and not oversized.manifest_path.exists(),
        }
    )
    return rows


def file_size_inventory(paths: Sequence[Path]) -> list[JsonDict]:
    """Record byte sizes before output writes and retain missing paths."""

    return [
        {
            "path": str(path),
            "state": "present" if path.is_file() else "missing",
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        for path in paths
    ]


def deferred_size_gate() -> JsonDict:
    """Describe the unimplemented commit-time size guard without overstating it."""

    return {
        "status": "deferred_outer_loop",
        "conductor_protected": False,
        "proposed_limit_bytes": RESULT_SIZE_LIMIT_BYTES,
        "proposal": "Add a reviewed pre-commit check that rejects each new results JSON above 20 MiB unless a small hash manifest points to shards.",
        "implementation_target": "pre-commit or repository validation outside scripts/research_conductor.py",
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Authenticate exact sources, the requirement, and available disk space."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "passed": available,
            }
        )
        if available:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "original_flagged_adversarial": None,
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "field": "REQ-*",
            "expected": "REQ-REPORT-7409",
            "observed": "REQ-REPORT-7409" if "REQ-REPORT-7409" in spec else None,
            "passed": "REQ-REPORT-7409" in spec,
        }
    )
    exclusion = (
        (root / EXCLUSION_PATH).read_text(encoding="utf-8")
        if (root / EXCLUSION_PATH).is_file()
        else ""
    )
    quarantined = "experiment_id: 7409" in exclusion
    checks.append(
        {
            "check": "current_task_not_quarantined",
            "upstream": EXCLUSION_PATH.as_posix(),
            "field": EXPERIMENT_ID,
            "expected": False,
            "observed": quarantined,
            "passed": not quarantined,
        }
    )
    free_bytes = shutil.disk_usage(root).free
    checks.append(
        {
            "check": "repository_write_capacity",
            "upstream": str(root),
            "field": "free_bytes",
            "expected": f">={RESULT_SIZE_LIMIT_BYTES}",
            "observed": free_bytes,
            "passed": free_bytes >= RESULT_SIZE_LIMIT_BYTES,
        }
    )
    for producer in PRODUCERS:
        terminal = producer["terminal"]
        present = (root / terminal).is_file()
        checks.append(
            {
                "check": f"external_producer_state:{producer['experiment_id']}",
                "upstream": terminal.as_posix(),
                "field": "bytes",
                "expected": "missing_external_input_recorded",
                "observed": "present_external_input"
                if present
                else "missing_external_input_recorded",
                "passed": True,
                "blocking_current_custody_work": False,
            }
        )
    return checks, hashes


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the frozen Exp7358 command set for only current affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful receipt and matching log bytes for each name."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and any(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is False
            for row in receipts
        )
        for name in names
    )


def _validation_logs_valid(receipts: Sequence[Mapping[str, Any]], root: Path) -> bool:
    """Rehash every receipt log so command metadata cannot float free."""

    for row in receipts:
        label = row.get("log_path")
        expected = row.get("log_sha256")
        if not isinstance(label, str) or not isinstance(expected, str):
            return False
        path = Path(label)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != expected:
            return False
    return True


def _source_hashes_valid(sources: object, root: Path) -> bool:
    """Rehash each declared source so inputs cannot drift after reduction."""

    if not isinstance(sources, Mapping) or not sources:
        return False
    for row in sources.values():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return False
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            return False
    return True


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    principle: str,
) -> JsonDict:
    """Build one ordinary gate row with explicit operands and meaning."""

    passed = observed == expected if operator == "==" else bool(observed)
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every artifact field except the checksum slot itself."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _row_ledger(
    contract_rows: Sequence[Mapping[str, Any]],
    producers: Sequence[Mapping[str, Any]],
    faults: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Expose every comparative unit with one stable row shape."""

    rows: list[JsonDict] = []
    for row in contract_rows:
        rows.append(
            {
                "unit_id": row.get("task_id"),
                "arm": "markdown_vs_yaml",
                "status": "complete",
                "passed": row.get("passed") is True,
                "censored": False,
                "failures": [] if row.get("passed") is True else ["contract_mismatch"],
            }
        )
    for row in producers:
        rows.append(
            {
                "unit_id": row.get("experiment_id"),
                "arm": "missing_producer_diagnosis",
                "status": "blocked_missing_declared_artifact",
                "passed": row.get("terminal_state") == "missing",
                "censored": False,
                "failures": ["declared_artifact_bytes"],
            }
        )
    for row in faults:
        rows.append(
            {
                "unit_id": row.get("fault"),
                "arm": "custody_fault_control",
                "status": "complete",
                "passed": row.get("passed") is True,
                "censored": False,
                "failures": [] if row.get("passed") is True else [str(row.get("fault"))],
            }
        )
    return rows


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each ordinary top-level field without wrapping its value."""

    return {
        key: FIELD_PRINCIPLES.get(
            key,
            "This ordinary top-level field retains measured identity, evidence, or reduction state.",
        )
        for key in keys
    }


def build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    producers: Sequence[Mapping[str, Any]],
    faults: Sequence[Mapping[str, Any]],
    bundle_dir: Path,
    receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    sizes_before: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal-shaped record from independently reducible evidence."""

    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    manifest = load_json(manifest_path)
    bundle_errors = verify_bundle(bundle_dir, expected_protocol_id=BUNDLE_PROTOCOL_ID)
    contract_passed = contract.get("passed") is True and all(
        row.get("rejected") is True for row in mutations
    )
    fault_passed = len(faults) == 5 and all(row.get("passed") is True for row in faults)
    affected_passed = _receipts_pass(receipts, AFFECTED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    logs_valid = _validation_logs_valid(receipts, root)
    gates = [
        _gate(
            "twelve_task_contract",
            "completion",
            True,
            contract_passed,
            principle="The advisory authorities and all five mutations must agree.",
        ),
        _gate(
            "bundle_hash_replay",
            "safety",
            [],
            bundle_errors,
            principle="Every raw unit and numeric checkpoint must retain exact bytes.",
        ),
        _gate(
            "custody_fault_controls",
            "safety",
            True,
            fault_passed,
            principle="All five interruption and corruption controls must reject their defect.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            affected_passed,
            principle="All eight scoped checks must pass once each.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            True,
            terminal_passed,
            principle="Cold replay, independent reduction, and both strict readers must pass.",
        ),
        _gate(
            "validation_log_hashes",
            "safety",
            True,
            logs_valid,
            principle="Every command receipt must bind the exact log bytes.",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    receipt = current_work_receipt.build_current_work_receipt(
        run_id="exp7409-current-host-aggregation",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={
            "python": sys.version.split()[0],
            "current_llm_operations": 0,
            "small_ebm_training_performed": False,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False, "receipts": []},
    )
    blocked_inputs = [
        {
            "upstream": row["experiment_id"],
            "path": row["declared_path"],
            "check": "declared_artifact_bytes",
            "field": "bytes",
            "expected": "readable_nonempty_bytes",
            "observed": None,
            "producer_verdict": "blocked_missing_declared_artifact",
        }
        for row in producers
        if row.get("terminal_state") == "missing"
    ]
    failed_gates = [row for row in gates if not row["passed"]]
    rows = _row_ledger(contract.get("contract_rows") or [], producers, faults)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "complete_evidence_custody_ready"
        if ready
        else "complete_disqualified_custody_checks",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **receipt,
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned": 19,
            "attempted": len(rows),
            "completed": len(rows),
            "failed": sum(row["passed"] is False for row in rows),
            "censored": 0,
            "unstarted": max(0, 19 - len(rows)),
            "independent_groups": {"contract_tasks": 12, "producer_paths": 2, "fault_controls": 5},
            "stop_rule": "Stop after twelve contract rows, two named producers, and five frozen fault controls.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "all_required_checks_passed": not failed_gates,
            "first_failure": failed_gates[0] if failed_gates else None,
            "failed_checks": [row["check"] for row in failed_gates],
            "blocked_external_inputs": blocked_inputs,
        },
        "verifier_is_oracle": False,
        "honest_verdict": (
            "complete_null_evidence_custody_ready_missing_v649_producers_unavailable"
            if ready
            else "complete_disqualified_evidence_custody_checks_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": {
            "status": "not_used_as_current_gate",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "promotion_score": 0,
        "evidence_custody_ready_score": ready,
        "scientific_value_score": 0,
        "missing_producer_rows": [deepcopy(dict(row)) for row in producers],
        "contract_rows": deepcopy(list(contract.get("contract_rows") or [])),
        "contract_mutation_rows": [deepcopy(dict(row)) for row in mutations],
        "custody_fault_rows": [deepcopy(dict(row)) for row in faults],
        "evidence_bundle": {
            "path": str(bundle_dir),
            "manifest": manifest,
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "verification_errors": bundle_errors,
        },
        "file_sizes_before_writing": [deepcopy(dict(row)) for row in sizes_before],
        "result_size_limit_bytes": RESULT_SIZE_LIMIT_BYTES,
        "deferred_size_gate": deferred_size_gate(),
        "reproducibility_checksum": "",
        "field_principles": {},
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    root: Path, bundle_dir: Path, receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a deterministic fixture from real authorities and temporary bundle bytes."""

    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    producers = inspect_missing_producers(root)
    bundle = EvidenceBundle(bundle_dir, protocol_id=BUNDLE_PROTOCOL_ID)
    bundle.record_unit({"unit_id": "contract", "value": 12, "status": "complete"})
    bundle.record_unit({"unit_id": "producers", "value": 2, "status": "complete"})
    bundle.finalize()
    fault_dir = bundle_dir.parent / f"{bundle_dir.name}-faults"
    faults = run_bundle_fault_controls(fault_dir)
    preconditions, hashes = collect_preconditions(root)
    return build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=hashes,
        contract=contract,
        mutations=mutations,
        producers=producers,
        faults=faults,
        bundle_dir=bundle_dir,
        receipts=receipts,
        require_terminal=True,
        started_at_utc="2026-09-19T12:00:00+00:00",
        ended_at_utc="2026-09-19T12:00:01+00:00",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        phase_spans=[],
        sizes_before=file_size_inventory([root / RESULT_PATH]),
    )


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_terminal: bool = True,
) -> list[str]:
    """Cold-check identity, sources, contract, bundle, logs, score, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    receipt_errors = current_work_receipt.validate_current_work_receipt(artifact, root=root)
    if receipt_errors or artifact.get("inference_substrate_class") != "aggregation":
        errors.append("current_work_receipt_invalid")
    if not _source_hashes_valid(artifact.get("source_artifact_hashes"), root):
        errors.append("source_hash_mismatch")

    roadmap = load_yaml(root / ROADMAP_PATH)
    try:
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
    except OSError:  # pragma: no cover - authenticated design bytes are a precondition.
        contract = {"contract_rows": [], "passed": False}
    if artifact.get("contract_rows") != contract.get("contract_rows"):
        errors.append("contract_rows_mismatch")
    mutations = artifact.get("contract_mutation_rows")
    if (
        not isinstance(mutations, list)
        or len(mutations) != 5
        or not all(isinstance(row, Mapping) and row.get("rejected") is True for row in mutations)
    ):
        errors.append("contract_mutations_invalid")
    producers = inspect_missing_producers(root)
    if artifact.get("missing_producer_rows") != producers:
        errors.append("missing_producer_rows_mismatch")

    bundle_row = artifact.get("evidence_bundle")
    if not isinstance(bundle_row, Mapping):
        errors.append("evidence_bundle_invalid")
    else:
        bundle_path = Path(str(bundle_row.get("path") or ""))
        if not bundle_path.is_absolute():
            bundle_path = root / bundle_path
        bundle_errors = verify_bundle(bundle_path, expected_protocol_id=BUNDLE_PROTOCOL_ID)
        manifest_path = bundle_path / BUNDLE_MANIFEST_NAME
        if (
            bundle_errors
            or not manifest_path.is_file()
            or (
                manifest_path.is_file()
                and sha256_file(manifest_path) != bundle_row.get("manifest_sha256")
            )
        ):
            errors.append("evidence_bundle_invalid")

    receipts = artifact.get("validation_receipts")
    if not isinstance(receipts, list) or not _receipts_pass(receipts, AFFECTED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if require_terminal and (
        not isinstance(receipts, list) or not _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    ):
        errors.append("terminal_validation_invalid")
    if not isinstance(receipts, list) or not _validation_logs_valid(receipts, root):
        errors.append("validation_log_hash_mismatch")

    faults = artifact.get("custody_fault_rows")
    fault_passed = (
        isinstance(faults, list)
        and len(faults) == 5
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in faults)
    )
    expected_ready = int(
        contract.get("passed") is True
        and fault_passed
        and not errors
        and isinstance(receipts, list)
        and _receipts_pass(receipts, AFFECTED_CHECK_NAMES)
        and (not require_terminal or _receipts_pass(receipts, TERMINAL_CHECK_NAMES))
    )
    if artifact.get("evidence_custody_ready_score") != expected_ready:
        errors.append("custody_score_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _phase_span(
    phase: str, phase_started: float, run_started: float, *, checkpoint: str, units: int
) -> JsonDict:  # pragma: no cover - authentic entrypoint timing.
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
    }


def _terminal_commands(
    root: Path, candidate: Path
) -> list[PlannedCommand]:  # pragma: no cover - capability E2E.
    """Build fresh-process replay, independent reduction, and strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7409_v650_evidence_custody import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v,require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);"
        "raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
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
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - public E2E.
    """Run custody controls, scoped checks, cold readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    sizes_before = file_size_inventory(
        [root / RESULT_PATH, raw_dir / "measured_terminal_candidate.json"]
    )

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    preconditions, source_hashes = collect_preconditions(root)
    if not all(row.get("passed") is True for row in preconditions):
        raise RuntimeError("precondition_failure")
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            checkpoint="source_identities_authenticated",
            units=len(preconditions),
        )
    )
    progress(started, "preconditions", "after", units=len(preconditions))

    for phase in ("load", "generate"):
        phase_started = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(
            _phase_span(phase, phase_started, started, checkpoint="no_current_llm_work", units=0)
        )
        progress(started, phase, "after", current_llm_operations=0)

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_contract_and_producer_reads")
    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    producers = inspect_missing_producers(root)
    current_work_receipt.atomic_json(
        raw_dir / "missing_producer_diagnostic.json", {"rows": producers}
    )
    bundle = EvidenceBundle(raw_dir / "bundle", protocol_id=BUNDLE_PROTOCOL_ID)
    bundle.record_unit({"unit_id": "contract", "value": 12, "status": "complete"})
    for row in producers:
        bundle.record_unit(
            {
                "unit_id": str(row["experiment_id"]),
                "value": 0,
                "status": "blocked_missing_declared_artifact",
            }
        )
    bundle.finalize()
    fault_root = Path(tempfile.mkdtemp(prefix="carnot-exp7409-faults-", dir="/tmp"))
    faults = run_bundle_fault_controls(fault_root)
    spans.append(
        _phase_span(
            "evaluate",
            phase_started,
            started,
            checkpoint="raw_units_and_fault_rows_durable",
            units=12 + len(producers) + len(faults),
        )
    )
    progress(started, "evaluate", "after_contract_and_producer_reads", units=19)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7409-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    phase_started = time.monotonic()
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "validate",
            phase_started,
            started,
            checkpoint="affected_checks_complete",
            units=len(affected),
        )
    )
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])

    candidate = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        contract=contract,
        mutations=mutations,
        producers=producers,
        faults=faults,
        bundle_dir=raw_dir / "bundle",
        receipts=affected,
        require_terminal=False,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        sizes_before=sizes_before,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    current_work_receipt.atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_commands))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            checkpoint="cold_replay_and_strict_readers_complete",
            units=len(terminal),
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=_receipts_pass(terminal, TERMINAL_CHECK_NAMES),
    )

    final = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        contract=contract,
        mutations=mutations,
        producers=producers,
        faults=faults,
        bundle_dir=raw_dir / "bundle",
        receipts=[*affected, *terminal],
        require_terminal=True,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
        sizes_before=sizes_before,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    encoded_size = len((json.dumps(final, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if encoded_size > RESULT_SIZE_LIMIT_BYTES:
        raise RuntimeError(f"terminal_artifact_oversized:{encoded_size}")
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix(), size=encoded_size)
    current_work_receipt.atomic_json(candidate_path, final)
    current_work_receipt.atomic_json(root / RESULT_PATH, final)
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix(), size=encoded_size)
    return final


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V650 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse normal, cold-validation, and private fault-child modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--fault-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fault_child is None and args.date is None:
        parser.error("--date is required")
    return args


def _run_default_main(run_date: str) -> int:  # pragma: no cover - declared E2E path.
    """Run and summarize the normal entrypoint after public argument parsing."""

    artifact = run_experiment(REPO_ROOT, run_date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "evidence_custody_ready_score": artifact["evidence_custody_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run V650 custody or cold-validate one measured candidate."""

    print("[exp7409] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    if args.fault_child is not None:  # pragma: no cover - executed by the owned fault child.
        _fault_child(args.fault_child)
    if args.validate is not None:
        value = load_json(args.validate)
        errors = validate_artifact(value, require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    return _run_default_main(args.date)  # pragma: no cover - declared E2E path.


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
