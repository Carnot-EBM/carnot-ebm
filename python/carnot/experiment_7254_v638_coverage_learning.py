"""Measure prospective learning with bounded coverage memory.

The run reuses the sealed Exp7253 controller and stream fixture. It measures a
finite CPU algorithm. It does not invoke an LLM or make a neural learning claim.

Spec refs: REQ-CL-7254 and SCENARIO-CL-7254-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import socket
import time
from typing import Any

from carnot import experiment_7199_v634_bounded_acquisition as exp7199
from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7253_v638_coverage_memory as exp7253
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7254
SCHEMA = "carnot.exp7254.v638_coverage_learning.v1"
MILESTONE = "2026.09.638"
RUN_DATE = "20260912"
RANDOM_SEED = 7_254_000
BOOTSTRAP_SEED = 7_254_951
BOOTSTRAP_DRAWS = 10_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
WRAPPER_PATH = Path("scripts/experiments/experiment_7254_v638_coverage_learning.py")
TEST_PATH = Path("tests/python/test_experiment_7254_v638_coverage_learning.py")
DEFAULT_UPSTREAM_ARTIFACT = Path("results/experiment_7253_v638_coverage_memory.json")
DEFAULT_ARTIFACT = Path("results/experiment_7254_v638_coverage_learning.json")
DEFAULT_PREQUENTIAL_ROWS = Path("results/raw/experiment_7254/prequential_rows.jsonl")
DEFAULT_OPERATION_ROWS = Path("results/raw/experiment_7254/operation_rows.jsonl")
DEFAULT_STATE_MANIFEST = Path("results/checkpoints/experiment_7254_v638_states.json")
DEFAULT_EVIDENCE_SIDECAR = Path("results/checkpoints/experiment_7254_v638_evidence.json")
DEFAULT_PROVISIONAL = Path("results/checkpoints/experiment_7254_v638_in_progress.json")
DEFAULT_LIVE_STATE_ROOT = Path("results/checkpoints/experiment_7254_live_state")
EXPECTED_UPSTREAM_SHA256 = "sha256:1deb4d16e455f6d19e317d040103263ce968c79e71b1cf4fe6d311209f46ec4c"
EXPECTED_CONTROLLER_CONTRACT_SHA256 = (
    "sha256:e0d42323f50215f73901d364e67af894c35034dd49e87d628f3e53713a716d3f"
)
EXPECTED_ARM_CONTRACT_SHA256 = (
    "sha256:a5797b4ddea50013a3e5371a33558fc244737cde96aa9b439941f47f3f2d5862"
)
EXPECTED_STREAM_CONTRACT_SHA256 = (
    "sha256:d7aa2027310a6577b7aed14487fb98191cea63671453690e9c6744099197b3d0"
)
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7240_v637_recurrence_fixture.py"),
    Path("python/carnot/experiment_7241_v637_recurrence_learning.py"),
    Path("python/carnot/experiment_7253_v638_coverage_memory.py"),
    Path("python/carnot/experiment_7254_v638_coverage_learning.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    WRAPPER_PATH,
    TEST_PATH,
    SPEC_PATH,
)
SCENARIO_PATTERN = re.compile(r"SCENARIO-CL-7254-[A-Z-]+")

COMPARISON_SPECS = (
    ("future_error_vs_frozen", "future_error_rate", "frozen_warmup"),
    ("future_error_vs_reset", "future_error_rate", "reset_relearn"),
    ("false_accept_vs_frozen", "false_accept_rate", "frozen_warmup"),
    ("false_accept_vs_reset", "false_accept_rate", "reset_relearn"),
    ("recurrence_error_vs_frozen", "recurrence_error_rate", "frozen_warmup"),
    ("recurrence_error_vs_destructive", "recurrence_error_rate", "destructive_update"),
    (
        "recurrence_error_vs_coverage_shuffled",
        "recurrence_error_rate",
        "coverage_archive_shuffled",
    ),
    ("future_error_vs_fifo_aligned", "future_error_rate", "fifo_archive_aligned"),
)
SCIENTIFIC_GATE_NAMES = (
    "future_error_vs_frozen_upper_ci95_lt_zero",
    "future_error_vs_reset_upper_ci95_lt_zero",
    "false_accept_vs_frozen_upper_ci95_lte_zero",
    "false_accept_vs_reset_upper_ci95_lte_zero",
    "recurrence_error_increase_vs_frozen_lte_0_02",
    "recurrence_error_vs_destructive_upper_ci95_lt_zero",
    "recurrence_error_vs_coverage_shuffled_upper_ci95_lt_zero",
    "valid_reactivation_gt_zero",
    "later_changed_decision_after_reactivation_gt_zero",
    "pre_release_difference_eq_zero",
    "positive_aligned_vs_fifo_value",
    "effective_prospective_shuffle_intervention",
    "oracle_control_headroom_gt_zero",
    "constraint_addition_gt_zero",
    "constraint_deactivation_gt_zero",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
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
    "coverage_run_complete_score",
    "coverage_learning_value_score",
    "continuous_self_learning_task",
    "prequential_rows_path",
    "causal_summary",
    "cost_rows",
)
FIELD_PRINCIPLES = {
    "schema": "Version this artifact and keep task identity as ordinary fields.",
    "experiment_id": "Bind the result to the fixed Exp7254 task.",
    "milestone": "Bind the result to milestone 2026.09.638.",
    "status": "A terminal artifact is complete or blocked; checkpoints remain unfinished.",
    "run_date": "Use 20260912 and keep actual UTC timestamps separately.",
    "started_at_utc": "Record the actual UTC start time.",
    "completed_at_utc": "Record the actual UTC end time.",
    "field_principles": "Keep explanations here and ordinary values at top level.",
    "preconditions_checked": "Record exact inputs, ownership, hashes, imports, and failures.",
    "MODEL_SPECS": "Name only models used now; this exact CPU run uses none.",
    "model_invoked": "Derive model use from current calls, not historical sources.",
    "inference_substrate": "Describe the actual exact CPU work with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its real duration.",
    "execution_venue": "Use host for orchestration; board evidence must be separate.",
    "execution_host": "Record the real hostname outside the venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time without sleeps or invented spans.",
    "random_seed": "Freeze stream, query, shuffle, and bootstrap schedules before outcomes.",
    "reproducibility_checksum": "Bind source, contracts, settings, raw rows, gates, and verdict.",
    "source_artifact_hashes": "Authenticate upstream, source, separated streams, and sidecars.",
    "rows": "Retain each stream-arm metric, error, abstention, cost, and censoring state.",
    "sample_size_budget": "State planned, attempted, completed, censored units, and stopping.",
    "acceptance_gate_results": "Record expected, observed, and pass state for every frozen gate.",
    "gate_check_summary": "Name exact failed external fields for blocked results.",
    "verifier_is_oracle": "Expose exact evaluator use; conformance cannot prove learned verification.",
    "honest_verdict": "Use complete_ for measured findings and blocked_ for external absence.",
    "verdict_class": "Use the closed class; oracle evidence cannot be positive.",
    "validation_receipts": "Record real commands, exit codes, classifications, and log hashes.",
    "coverage_run_complete_score": "One requires every sealed eight-arm event row.",
    "coverage_learning_value_score": "One requires every frozen scientific gate.",
    "continuous_self_learning_task": "Released feedback commits reusable constraint changes.",
    "prequential_rows_path": "A hashed JSONL keeps each chronological prediction and release.",
    "causal_summary": "Count pre-release, reactivation, selection, and later-decision changes.",
    "cost_rows": "Measure update, lookup, serialization, commit, and memory against targets.",
}

gate_check = exp7213.gate_check
gate_summary = exp7213.gate_summary


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw rows, state, evidence, provisional work, and terminal bytes separate."""

    prequential_rows: Path
    operation_rows: Path
    state_manifest: Path
    evidence_sidecar: Path
    provisional: Path
    live_state_root: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Use the fixed result paths below this checkout."""

        return cls.from_results_root(REPO_ROOT / "results")

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put test-owned output below one caller-owned directory."""

        return cls.from_results_root(root)

    @classmethod
    def from_results_root(cls, root: Path) -> ExperimentPaths:
        """Derive all outputs while authenticated inputs remain unchanged."""

        raw = root / "raw" / "experiment_7254"
        checkpoints = root / "checkpoints"
        return cls(
            raw / "prequential_rows.jsonl",
            raw / "operation_rows.jsonl",
            checkpoints / DEFAULT_STATE_MANIFEST.name,
            checkpoints / DEFAULT_EVIDENCE_SIDECAR.name,
            checkpoints / DEFAULT_PROVISIONAL.name,
            checkpoints / DEFAULT_LIVE_STATE_ROOT.name,
            root / DEFAULT_ARTIFACT.name,
        )


@dataclass(frozen=True)
class LearningPanel:
    """Retain raw events, stream summaries, costs, state, and causal counts."""

    prequential_rows: list[JsonDict]
    rows: list[JsonDict]
    operation_rows: list[JsonDict]
    state_entries: list[JsonDict]
    causal_summary: JsonDict
    maximum_memory_bytes: int

    @property
    def cost_rows(self) -> list[JsonDict]:
        """Expose measured operation rows with the task's required field name."""

        return self.operation_rows


def _progress(phase: int, boundary: str, detail: str) -> None:
    """Emit a flushed boundary so a long CPU replay stays observable."""

    print(f"phase {phase} {boundary}: {detail}", flush=True)


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolve repository-relative evidence and preserve absolute test paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes while absence remains an observed failure."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object while malformed evidence stays unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load object rows without treating malformed JSONL as valid evidence."""

    rows: list[JsonDict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    return []
                rows.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output evidence."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _atomic_write(path: Path, payload: bytes) -> JsonDict:
    """Publish complete bytes through one flushed rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": _sha256_path(path), "bytes": len(payload)}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode chronological objects with the shared canonical representation."""

    return b"".join(transactional.canonical_json_bytes(dict(row)) for row in rows)


def _task_identity(text: str) -> JsonDict:
    """Extract only the fixed Exp7254 block from the executable roadmap."""

    match = re.search(r"(?ms)^- id: exp7254-coverage-learning\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(1)
    milestone = re.search(r"(?m)^  milestone: (.+)$", block)
    deliverable = re.search(r"(?m)^  deliverable: (.+)$", block)
    return {
        "id": "exp7254-coverage-learning" if block else None,
        "milestone": None if milestone is None else milestone.group(1).strip(),
        "deliverable": None if deliverable is None else deliverable.group(1).strip(),
    }


def _summary_with_failures(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Add every failed check while keeping the shipped first-failure summary."""

    summary = gate_summary(checks)
    summary["failed_checks"] = [
        {
            key: row.get(key)
            for key in ("check", "upstream", "field", "expected_value", "observed_value")
        }
        for row in checks
        if row.get("passed") is not True
    ]
    return summary


def collect_preconditions(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    upstream_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str | None], JsonDict]:
    """Authenticate the task, Exp7253 contract, streams, imports, and outputs."""

    artifact_path = _resolve(repo_root, upstream_path or DEFAULT_UPSTREAM_ARTIFACT)
    upstream = _load_object(artifact_path)
    hashes = {
        str(_resolve(repo_root, path)): _sha256_path(_resolve(repo_root, path))
        for path in SOURCE_PATHS
    }
    hashes[str(artifact_path)] = _sha256_path(artifact_path)
    spec_path = _resolve(repo_root, SPEC_PATH)
    roadmap_path = _resolve(repo_root, "research-roadmap.yaml")
    exclusion_path = _resolve(repo_root, "ops/exclusion_manifest.yaml")
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    roadmap = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    exclusions = exclusion_path.read_text(encoding="utf-8") if exclusion_path.exists() else ""
    source_state = {
        str(path): ("nonempty" if hashes[str(_resolve(repo_root, path))] is not None else "missing")
        for path in SOURCE_PATHS
    }
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7226_v636_belief_compiler",
            "carnot.experiment_7253_v638_coverage_memory",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    writable = {
        field: _path_writable(getattr(paths, field))
        for field in (
            "prequential_rows",
            "operation_rows",
            "state_manifest",
            "evidence_sidecar",
            "provisional",
            "live_state_root",
            "artifact",
        )
    }
    quarantine = exp7213.quarantine_state(
        upstream,
        exclusions,
        artifact_path.name,
        "exp7253-coverage-memory",
    )
    stream_receipts = upstream.get("stream_manifest", {}).get("receipts", {})
    observed_stream_hashes = {
        name: _sha256_path(_resolve(repo_root, receipt.get("path", "")))
        for name, receipt in stream_receipts.items()
        if name.startswith("prospective_") and isinstance(receipt, Mapping)
    }
    expected_stream_hashes = {
        name: receipt.get("sha256")
        for name, receipt in stream_receipts.items()
        if name.startswith("prospective_") and isinstance(receipt, Mapping)
    }
    checks = [
        gate_check(
            "driving_capability_spec", str(SPEC_PATH), "REQ-CL-7254", True, "REQ-CL-7254" in spec
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7254-*",
            7,
            len(set(SCENARIO_PATTERN.findall(spec))),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            dict.fromkeys(source_state, "nonempty"),
            source_state,
        ),
        gate_check(
            "v638_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            {
                "id": "exp7254-coverage-learning",
                "milestone": MILESTONE,
                "deliverable": str(DEFAULT_ARTIFACT),
            },
            _task_identity(roadmap),
        ),
        gate_check("required_imports", "python", "imports", dict.fromkeys(imports, True), imports),
        gate_check(
            "writable_output_paths",
            "host_filesystem",
            "raw,state,evidence,provisional,terminal",
            dict.fromkeys(writable, True),
            writable,
        ),
        gate_check(
            "exp7253_artifact_hash",
            "exp7253-coverage-memory",
            str(artifact_path),
            EXPECTED_UPSTREAM_SHA256,
            hashes[str(artifact_path)],
        ),
        gate_check(
            "exp7253_status",
            "exp7253-coverage-memory",
            "status",
            "complete",
            upstream.get("status"),
        ),
        gate_check(
            "exp7253_coverage_fixture_ready",
            "exp7253-coverage-memory",
            "coverage_fixture_ready_score",
            1,
            upstream.get("coverage_fixture_ready_score"),
        ),
        gate_check(
            "exp7253_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "exp7253_controller_contract",
            "exp7253-coverage-memory",
            "controller_contract_sha256",
            EXPECTED_CONTROLLER_CONTRACT_SHA256,
            transactional.sha256_json(upstream.get("controller_contract")),
        ),
        gate_check(
            "exp7253_arm_contract",
            "exp7253-coverage-memory",
            "arm_contract_sha256",
            EXPECTED_ARM_CONTRACT_SHA256,
            transactional.sha256_json(upstream.get("arm_contract")),
        ),
        gate_check(
            "exp7253_stream_contract",
            "exp7253-coverage-memory",
            "stream_manifest_sha256",
            EXPECTED_STREAM_CONTRACT_SHA256,
            transactional.sha256_json(upstream.get("stream_manifest")),
        ),
        gate_check(
            "exp7253_stream_sidecars",
            "exp7253-coverage-memory",
            "prospective_stream_receipts",
            expected_stream_hashes,
            observed_stream_hashes,
        ),
    ]
    for name, receipt in stream_receipts.items():
        if name.startswith("prospective_") and isinstance(receipt, Mapping):
            hashes[str(_resolve(repo_root, receipt["path"]))] = observed_stream_hashes[name]
    return checks, hashes, upstream


def load_stream_views(repo_root: Path, upstream: Mapping[str, Any]) -> exp7253.StreamViews:
    """Load only the authenticated prospective views from separated files."""

    receipts = upstream["stream_manifest"]["receipts"]
    public = _read_jsonl(_resolve(repo_root, receipts["prospective_public"]["path"]))
    authority = _read_jsonl(_resolve(repo_root, receipts["prospective_private_authority"]["path"]))
    releases = _read_jsonl(_resolve(repo_root, receipts["prospective_releases"]["path"]))
    views = exp7253.StreamViews(
        public,
        authority,
        releases,
        deepcopy(upstream["stream_manifest"]["prospective"]),
    )
    errors = exp7253.stream_conformance_errors(views, "prospective")
    if errors:
        raise ValueError("exp7253_stream_conformance:" + ",".join(errors))
    return views


def _memory_usage(controller: Any | None, pending: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure live state and shared pending feedback for one arm."""

    if controller is None:
        return {
            "witness_bytes": 0,
            "archive_bytes": 0,
            "pending_bytes": 0,
            "ledger_bytes": 0,
            "controller_bytes": 0,
            "total_bytes": 0,
        }
    if isinstance(controller, exp7253.CoverageArchiveController):
        return controller.memory_usage(pending)
    pending_bytes = len(transactional.canonical_json_bytes(list(pending)))
    controller_bytes = len(controller.state_bytes())
    return {
        "witness_bytes": 0,
        "archive_bytes": 0,
        "pending_bytes": pending_bytes,
        "ledger_bytes": 0,
        "controller_bytes": controller_bytes,
        "total_bytes": controller_bytes + pending_bytes,
    }


def _valid_reactivations(operations: Sequence[Mapping[str, Any]]) -> int:
    """Count selected archives that passed the unchanged released-witness gate."""

    count = 0
    for operation in operations:
        selected = operation.get("reactivated_archive_id")
        nomination = operation.get("nomination_receipt", {})
        candidates = nomination.get("after_evaluations", [])
        count += int(
            selected is not None
            and nomination.get("final_gate")
            == "at_least_8_released_witnesses_and_zero_contradictions"
            and any(
                row.get("archive_id") == selected and row.get("gate_passed") is True
                for row in candidates
            )
        )
    return count


def _percentile(values: Sequence[int | float], probability: float) -> float:
    """Return a deterministic nearest-rank percentile for measured samples."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(probability * (len(ordered) - 1)))))
    return ordered[index]


def _reduce_stream_arm(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one stream arm while keeping abstentions in the error denominator."""

    ordered = sorted(rows, key=lambda row: int(row["chronology_index"]))
    future = [row for row in ordered if int(row["chronology_index"]) >= exp7253.WARMUP_COUNT]
    recurrence = [row for row in future if row["recurrence_eligible"] is True]
    return {
        "unit_id": str(ordered[0]["unit_id"]),
        "stream_id": str(ordered[0]["stream_id"]),
        "seed": int(ordered[0]["seed"]),
        "arm": str(ordered[0]["arm"]),
        "metric": "prospective_full_denominator_error",
        "event_count": len(ordered),
        "future_event_count": len(future),
        "future_error": sum(int(row["full_denominator_error"]) for row in future),
        "future_error_rate": sum(int(row["full_denominator_error"]) for row in future)
        / len(future),
        "false_accept": sum(int(row["false_accept"]) for row in future),
        "false_accept_rate": sum(int(row["false_accept"]) for row in future) / len(future),
        "abstention": sum(int(row["abstention"]) for row in future),
        "abstention_rate": sum(int(row["abstention"]) for row in future) / len(future),
        "recurrence_event_count": len(recurrence),
        "recurrence_error": sum(int(row["full_denominator_error"]) for row in recurrence),
        "recurrence_error_rate": (
            None
            if not recurrence
            else sum(int(row["full_denominator_error"]) for row in recurrence) / len(recurrence)
        ),
        "query_count": sum(int(row["query_selected"]) for row in ordered),
        "release_count": int(ordered[-1]["released_query_count_after"]),
        "constraint_addition_count": sum(int(row["constraint_addition_count"]) for row in ordered),
        "constraint_deactivation_count": sum(
            int(row["constraint_deactivation_count"]) for row in ordered
        ),
        "valid_reactivation_count": sum(int(row["valid_reactivation_count"]) for row in ordered),
        "later_changed_decision_after_reactivation_count": sum(
            int(row["later_changed_decision_after_reactivation"]) for row in ordered
        ),
        "pre_release_difference_count": sum(int(row["pre_release_difference"]) for row in ordered),
        "shuffle_selection_change_count": sum(
            int(row["shuffle_selection_change_count"]) for row in ordered
        ),
        "lookup_p50_ns": _percentile([int(row["lookup_cost_ns"]) for row in ordered], 0.50),
        "lookup_p95_ns": _percentile([int(row["lookup_cost_ns"]) for row in ordered], 0.95),
        "update_p50_ns": _percentile(
            [int(row["update_cost_ns"]) for row in ordered if row["commit_applied"]], 0.50
        ),
        "update_p95_ns": _percentile(
            [int(row["update_cost_ns"]) for row in ordered if row["commit_applied"]], 0.95
        ),
        "durable_commit_p95_ns": _percentile(
            [int(row["durable_commit_cost_ns"]) for row in ordered if row["commit_applied"]],
            0.95,
        ),
        "maximum_memory_bytes": max(int(row["memory_total_bytes"]) for row in ordered),
        "censored": False,
    }


def reduce_prequential_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce raw chronological rows with streams as the independent units."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["stream_id"]), str(row["arm"])), []).append(row)
    arm_order = {arm: index for index, arm in enumerate(exp7253.ARMS)}
    return [
        _reduce_stream_arm(group)
        for _, group in sorted(groups.items(), key=lambda item: (item[0][0], arm_order[item[0][1]]))
    ]


def prequential_row_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check authority isolation, chronology, commit hashes, and state chains."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(row.get("prediction_frozen_before_release") is not True for row in rows), "chronology")
    add(
        any(
            row.get("oracle_control") is not True
            and (
                row.get("held_out_label_visible_to_controller") is not False
                or row.get("controller_input_fields") != ["event_id", "family_id", "numeric_value"]
            )
            for row in rows
        ),
        "future_label_leakage",
    )
    committed = [row for row in rows if row.get("commit_applied") is True]
    add(
        any(
            not str(row.get("commit_parent_hash", "")).startswith("sha256:")
            or not str(row.get("commit_child_hash", "")).startswith("sha256:")
            or row.get("commit_parent_hash") == row.get("commit_child_hash")
            or row.get("state_hash_before_prediction") != row.get("commit_parent_hash")
            for row in committed
        ),
        "commit_hashes",
    )
    add(
        any(
            row.get("commit_parent_hash") is not None or row.get("commit_child_hash") is not None
            for row in rows
            if row.get("commit_applied") is not True
        ),
        "noncommit_hashes",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in committed:
        groups.setdefault((str(row["stream_id"]), str(row["arm"])), []).append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["chronology_index"]))
        add(
            any(
                current["commit_parent_hash"] != previous["commit_child_hash"]
                for previous, current in zip(ordered, ordered[1:])
            ),
            "commit_chain",
        )
    return errors


def independent_reduce(path: Path) -> list[JsonDict]:
    """Reload raw JSONL so producer summaries cannot influence reduction."""

    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("raw_rows_unavailable")
    errors = prequential_row_errors(rows)
    if errors:
        raise ValueError("raw_row_validation_failed:" + ",".join(errors))
    return reduce_prequential_rows(rows)


def causal_summary_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute causal counts from treatment and control event rows."""

    target = [row for row in rows if row.get("arm") == "coverage_archive_aligned"]
    shuffled = [row for row in rows if row.get("arm") == "coverage_archive_shuffled"]
    frozen = [row for row in rows if row.get("arm") == "frozen_warmup"]
    return {
        "constraint_addition_count": sum(int(row["constraint_addition_count"]) for row in target),
        "constraint_deactivation_count": sum(
            int(row["constraint_deactivation_count"]) for row in target
        ),
        "valid_reactivation_count": sum(int(row["valid_reactivation_count"]) for row in target),
        "later_changed_decision_after_reactivation_count": sum(
            int(row["later_changed_decision_after_reactivation"]) for row in target
        ),
        "pre_release_difference_count": sum(int(row["pre_release_difference"]) for row in target),
        "prospective_shuffle_selection_change_count": sum(
            int(row["shuffle_selection_change_count"]) for row in shuffled
        ),
        "oracle_headroom_event_count": sum(
            int(
                int(row["chronology_index"]) >= exp7253.WARMUP_COUNT
                and int(row["full_denominator_error"]) > 0
            )
            for row in frozen
        ),
        "zero_pre_release_differences": all(
            int(row["pre_release_difference"]) == 0 for row in target
        ),
    }


def run_learning_panel(
    views: exp7253.StreamViews,
    *,
    state_root: Path,
    stream_ids: Sequence[str] | None = None,
    progress: bool = False,
) -> LearningPanel:
    """Replay all arms and commit only feedback released after each prediction."""

    selected_streams = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(exp7253.STREAM_COUNT))
    )
    authority = {str(row["event_id"]): row for row in views.authority}
    schedules = {str(row["event_id"]): row for row in views.releases}
    by_stream = {
        stream_id: [row for row in views.public if row["stream_id"] == stream_id]
        for stream_id in selected_streams
    }
    prequential_rows: list[JsonDict] = []
    operation_rows: list[JsonDict] = []
    state_entries: list[JsonDict] = []
    maximum_memory_bytes = 0
    benchmark_start = time.monotonic()
    for stream_offset, stream_id in enumerate(selected_streams):
        events = by_stream[stream_id]
        if len(events) != exp7253.EVENTS_PER_STREAM:
            raise ValueError(f"incomplete_stream:{stream_id}")
        stream_seed = int(authority[str(events[0]["event_id"])]["stream_seed"])
        controllers: dict[str, Any] = {
            "frozen_warmup": exp7226.PackedBeliefController(),
            "reset_relearn": exp7253.CoverageArchiveController(archive_cap=0),
            "destructive_update": exp7226.PackedBeliefController(),
            "fifo_archive_shuffled": exp7253.CoverageArchiveController(
                admission_mode="fifo", nomination_mode="shuffled"
            ),
            "fifo_archive_aligned": exp7253.CoverageArchiveController(
                admission_mode="fifo", nomination_mode="aligned"
            ),
            "coverage_archive_shuffled": exp7253.CoverageArchiveController(
                admission_mode="coverage", nomination_mode="shuffled"
            ),
            "coverage_archive_aligned": exp7253.CoverageArchiveController(
                admission_mode="coverage", nomination_mode="aligned"
            ),
        }
        initial_hashes = {arm: controller.state_hash() for arm, controller in controllers.items()}
        pending: list[JsonDict] = []
        actual_queries = 0
        released_queries = 0
        last_reactivation_index: dict[str, int | None] = dict.fromkeys(controllers)
        for block_index, offset in enumerate(
            range(0, exp7253.EVENTS_PER_STREAM, exp7253.QUERY_BLOCK_SIZE)
        ):
            block = events[offset : offset + exp7253.QUERY_BLOCK_SIZE]
            public_block = [exp7253._controller_input(row) for row in block]
            tie_ranks = exp7199.seeded_tie_ranks(stream_seed, block_index, public_block)
            selected_query = controllers["destructive_update"].select_request(
                public_block, tie_ranks
            )
            selected_id = str(selected_query["event_id"])
            for event in block:
                event_id = str(event["event_id"])
                chronology_index = int(event["chronology_index"])
                controller_event = exp7253._controller_input(event)
                will_query = (
                    event_id == selected_id
                    and actual_queries < exp7253.QUERY_CEILING
                    and len(pending) < exp7253.PENDING_CAPACITY
                )
                predictions: dict[str, tuple[str, float]] = {}
                lookup_costs: dict[str, int] = {}
                state_hashes: dict[str, str | None] = {}
                usages: dict[str, JsonDict] = {}
                for arm, controller in controllers.items():
                    lookup_start = time.perf_counter_ns()
                    predictions[arm] = controller.predict(controller_event)
                    lookup_costs[arm] = time.perf_counter_ns() - lookup_start
                    state_hashes[arm] = controller.state_hash()
                    usages[arm] = _memory_usage(controller, pending)
                truth = authority[event_id]
                predictions["oracle_positive_control"] = (str(truth["exact_label"]), 0.0)
                lookup_costs["oracle_positive_control"] = 0
                state_hashes["oracle_positive_control"] = None
                usages["oracle_positive_control"] = _memory_usage(None, pending)
                positions: dict[str, int] = {}
                frozen_prediction = predictions["frozen_warmup"][0]
                for arm in exp7253.ARMS:
                    prediction, energy = predictions[arm]
                    usage = usages[arm]
                    if arm in exp7253.ARCHIVE_ARMS:
                        maximum_memory_bytes = max(maximum_memory_bytes, int(usage["total_bytes"]))
                    pre_release_difference = int(
                        arm == "coverage_archive_aligned"
                        and released_queries == 0
                        and prediction != frozen_prediction
                    )
                    later_changed = int(
                        arm == "coverage_archive_aligned"
                        and last_reactivation_index[arm] is not None
                        and chronology_index > int(last_reactivation_index[arm])
                        and prediction != frozen_prediction
                    )
                    positions[arm] = len(prequential_rows)
                    row = {
                        "unit_id": f"{stream_id}:{arm}",
                        "stream_id": stream_id,
                        "seed": stream_seed,
                        "arm": arm,
                        "event_id": event_id,
                        "chronology_index": chronology_index,
                        "prediction": prediction,
                        "prediction_energy": energy,
                        "later_released_label": truth["exact_label"],
                        "classification_error": int(
                            prediction != "abstain" and prediction != truth["exact_label"]
                        ),
                        "abstention": int(prediction == "abstain"),
                        "full_denominator_error": int(prediction != truth["exact_label"]),
                        "false_accept": int(
                            prediction == "accept" and truth["exact_label"] == "reject"
                        ),
                        "recurrence_eligible": (
                            truth["drift_pattern"] == "aba_recurrence" and chronology_index >= 768
                        ),
                        "query_selected": will_query,
                        "released_query_count_before": released_queries,
                        "released_query_count_after": released_queries,
                        "state_hash_before_prediction": state_hashes[arm],
                        "commit_applied": False,
                        "commit_parent_hash": None,
                        "commit_child_hash": None,
                        "commit_release_count": 0,
                        "constraint_addition_count": 0,
                        "constraint_deactivation_count": 0,
                        "valid_reactivation_count": 0,
                        "shuffle_selection_change_count": 0,
                        "later_changed_decision_after_reactivation": later_changed,
                        "pre_release_difference": pre_release_difference,
                        "lookup_cost_ns": lookup_costs[arm],
                        "update_cost_ns": 0,
                        "serialization_cost_ns": 0,
                        "durable_commit_cost_ns": 0,
                        "memory_total_bytes": int(usage["total_bytes"]),
                        "memory_witness_bytes": int(usage["witness_bytes"]),
                        "memory_archive_bytes": int(usage["archive_bytes"]),
                        "memory_pending_bytes": int(usage["pending_bytes"]),
                        "memory_ledger_bytes": int(usage["ledger_bytes"]),
                        "controller_input_fields": (
                            []
                            if arm == "oracle_positive_control"
                            else ["event_id", "family_id", "numeric_value"]
                        ),
                        "held_out_label_visible_to_controller": (
                            "not_applicable_evaluator_control"
                            if arm == "oracle_positive_control"
                            else False
                        ),
                        "oracle_control": arm == "oracle_positive_control",
                        "prediction_frozen_before_release": True,
                        "censored": False,
                    }
                    prequential_rows.append(row)
                    operation_rows.append(
                        {
                            "operation": "lookup",
                            "stream_id": stream_id,
                            "arm": arm,
                            "event_id": event_id,
                            "cost_ns": lookup_costs[arm],
                        }
                    )
                if will_query:
                    actual_queries += 1
                    schedule = schedules[event_id]
                    pending.append(
                        {
                            "public": controller_event,
                            "observed_label": schedule["observed_label"],
                            "request_index": chronology_index,
                            "release_index": chronology_index + int(schedule["delay"]),
                        }
                    )
                due = sorted(
                    [row for row in pending if int(row["release_index"]) <= chronology_index],
                    key=lambda row: (int(row["release_index"]), int(row["request_index"])),
                )
                if due:
                    payload = [exp7253._support_release(row) for row in due]
                    update_controllers = dict(controllers)
                    if chronology_index >= exp7253.WARMUP_COUNT:
                        update_controllers.pop("frozen_warmup")
                    for arm, controller in update_controllers.items():
                        parent_hash = controller.state_hash()
                        update_start = time.perf_counter_ns()
                        receipt = controller.commit_batch(
                            payload,
                            current_cycle=chronology_index,
                            expected_parent_hash=parent_hash,
                        )
                        update_cost = time.perf_counter_ns() - update_start
                        operations = receipt.get("operations", [])
                        additions = int(receipt.get("release_count", len(payload)))
                        deactivations = sum(
                            int(
                                operation.get("active_contradiction") is True
                                or operation.get("empty_reset") is True
                            )
                            for operation in operations
                        )
                        valid_reactivations = _valid_reactivations(operations)
                        selection_changes = sum(
                            int(operation.get("selection_changed") is True)
                            for operation in operations
                        )
                        if valid_reactivations:
                            last_reactivation_index[arm] = chronology_index
                        serialization_start = time.perf_counter_ns()
                        state_bytes = controller.state_bytes()
                        serialization_cost = time.perf_counter_ns() - serialization_start
                        durable_start = time.perf_counter_ns()
                        state_path = state_root / stream_id / f"{arm}.json"
                        controller.save(state_path)
                        durable_cost = time.perf_counter_ns() - durable_start
                        row = prequential_rows[positions[arm]]
                        row.update(
                            {
                                "commit_applied": True,
                                "commit_parent_hash": receipt["parent_hash"],
                                "commit_child_hash": receipt["new_state_hash"],
                                "commit_release_count": additions,
                                "constraint_addition_count": additions,
                                "constraint_deactivation_count": deactivations,
                                "valid_reactivation_count": valid_reactivations,
                                "shuffle_selection_change_count": selection_changes,
                                "update_cost_ns": update_cost,
                                "serialization_cost_ns": serialization_cost,
                                "durable_commit_cost_ns": durable_cost,
                            }
                        )
                        for operation, cost in (
                            ("update", update_cost),
                            ("serialization", serialization_cost),
                            ("durable_commit", durable_cost),
                        ):
                            operation_rows.append(
                                {
                                    "operation": operation,
                                    "stream_id": stream_id,
                                    "arm": arm,
                                    "event_id": event_id,
                                    "cost_ns": cost,
                                    "parent_state_sha256": receipt["parent_hash"],
                                    "new_state_sha256": receipt["new_state_hash"],
                                    "release_count": additions,
                                }
                            )
                    released_queries += len(due)
                    pending = [row for row in pending if row not in due]
                for position in positions.values():
                    prequential_rows[position]["released_query_count_after"] = released_queries
        for arm, controller in controllers.items():
            state_entries.append(
                {
                    "stream_id": stream_id,
                    "arm": arm,
                    "initial_state_sha256": initial_hashes[arm],
                    "final_state_sha256": controller.state_hash(),
                    "final_state_bytes": len(controller.state_bytes()),
                    "model_weights_mutated": False,
                }
            )
        if progress:
            print(
                f"phase 4 benchmark unit {stream_offset + 1}/{len(selected_streams)} "
                f"completed_rows={(stream_offset + 1) * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)} "
                f"elapsed_s={time.monotonic() - benchmark_start:.3f}",
                flush=True,
            )
    errors = prequential_row_errors(prequential_rows)
    if errors:
        raise ValueError("prequential_conformance:" + ",".join(errors))
    rows = reduce_prequential_rows(prequential_rows)
    causal = causal_summary_from_rows(prequential_rows)
    return LearningPanel(
        prequential_rows,
        rows,
        operation_rows,
        state_entries,
        causal,
        maximum_memory_bytes,
    )


def _bootstrap_interval(values: Sequence[float], *, draws: int, salt: str) -> JsonDict:
    """Resample paired stream differences with one frozen deterministic seed."""

    if not values:
        return {"estimate": 0.0, "ci95": [0.0, 0.0]}
    seed = int(transactional.sha256_json([BOOTSTRAP_SEED, salt])[-16:], 16)
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return {
        "estimate": sum(values) / len(values),
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
    }


def build_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[JsonDict]:
    """Build frozen paired comparisons without pooling correlated events."""

    by_unit = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    comparisons: list[JsonDict] = []
    for comparison_id, metric, control in COMPARISON_SPECS:
        differences = []
        for seed in seeds:
            target = by_unit[(seed, "coverage_archive_aligned")].get(metric)
            baseline = by_unit[(seed, control)].get(metric)
            if isinstance(target, (int, float)) and isinstance(baseline, (int, float)):
                differences.append({"seed": seed, "difference": float(target) - float(baseline)})
        interval = _bootstrap_interval(
            [row["difference"] for row in differences], draws=draws, salt=comparison_id
        )
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "target_arm": "coverage_archive_aligned",
                "control_arm": control,
                "independent_unit": "stream",
                "independent_unit_count": len(differences),
                "bootstrap_draws": draws,
                "seed_differences": differences,
                **interval,
            }
        )
    return comparisons


def score_acceptance_gates(
    comparisons: Sequence[Mapping[str, Any]],
    causal_summary: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Score every frozen efficacy, safety, retention, FIFO, and shuffle gate."""

    by_id = {str(row["comparison_id"]): row for row in comparisons}

    def upper(name: str) -> float:
        return float(by_id[name]["ci95"][1])

    def estimate(name: str) -> float:
        return float(by_id[name]["estimate"])

    definitions = (
        (
            "future_error_vs_frozen_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_frozen"),
            upper("future_error_vs_frozen") < 0,
        ),
        (
            "future_error_vs_reset_upper_ci95_lt_zero",
            "<0",
            upper("future_error_vs_reset"),
            upper("future_error_vs_reset") < 0,
        ),
        (
            "false_accept_vs_frozen_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_frozen"),
            upper("false_accept_vs_frozen") <= 0,
        ),
        (
            "false_accept_vs_reset_upper_ci95_lte_zero",
            "<=0",
            upper("false_accept_vs_reset"),
            upper("false_accept_vs_reset") <= 0,
        ),
        (
            "recurrence_error_increase_vs_frozen_lte_0_02",
            "<=0.02",
            estimate("recurrence_error_vs_frozen"),
            estimate("recurrence_error_vs_frozen") <= 0.02,
        ),
        (
            "recurrence_error_vs_destructive_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_destructive"),
            upper("recurrence_error_vs_destructive") < 0,
        ),
        (
            "recurrence_error_vs_coverage_shuffled_upper_ci95_lt_zero",
            "<0",
            upper("recurrence_error_vs_coverage_shuffled"),
            upper("recurrence_error_vs_coverage_shuffled") < 0,
        ),
        (
            "valid_reactivation_gt_zero",
            ">0",
            int(causal_summary["valid_reactivation_count"]),
            int(causal_summary["valid_reactivation_count"]) > 0,
        ),
        (
            "later_changed_decision_after_reactivation_gt_zero",
            ">0",
            int(causal_summary["later_changed_decision_after_reactivation_count"]),
            int(causal_summary["later_changed_decision_after_reactivation_count"]) > 0,
        ),
        (
            "pre_release_difference_eq_zero",
            "==0",
            int(causal_summary["pre_release_difference_count"]),
            int(causal_summary["pre_release_difference_count"]) == 0,
        ),
        (
            "positive_aligned_vs_fifo_value",
            ">0 coverage gain",
            -estimate("future_error_vs_fifo_aligned"),
            -estimate("future_error_vs_fifo_aligned") > 0,
        ),
        (
            "effective_prospective_shuffle_intervention",
            ">0",
            int(causal_summary["prospective_shuffle_selection_change_count"]),
            int(causal_summary["prospective_shuffle_selection_change_count"]) > 0,
        ),
        (
            "oracle_control_headroom_gt_zero",
            ">0",
            int(causal_summary["oracle_headroom_event_count"]),
            int(causal_summary["oracle_headroom_event_count"]) > 0,
        ),
        (
            "constraint_addition_gt_zero",
            ">0",
            int(causal_summary["constraint_addition_count"]),
            int(causal_summary["constraint_addition_count"]) > 0,
        ),
        (
            "constraint_deactivation_gt_zero",
            ">0",
            int(causal_summary["constraint_deactivation_count"]),
            int(causal_summary["constraint_deactivation_count"]) > 0,
        ),
    )
    return {
        name: {
            "principle": "Keep this preregistered criterion separate from run completion.",
            "expected": expected,
            "observed": observed,
            "pass": passed,
        }
        for name, expected, observed, passed in definitions
    }


def classify_result(
    gates: Mapping[str, Mapping[str, Any]],
    *,
    run_complete: bool,
) -> JsonDict:
    """Keep complete nulls, circular positives, and unfinished runs distinct."""

    value = int(run_complete and all(row.get("pass") is True for row in gates.values()))
    if not run_complete:
        return {
            "coverage_learning_value_score": 0,
            "verdict_class": "partial",
            "honest_verdict": "partial_unfinished: scheduled event-arm rows remain incomplete",
        }
    if value:
        return {
            "coverage_learning_value_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive: bounded coverage memory passed every frozen gate",
        }
    return {
        "coverage_learning_value_score": 0,
        "verdict_class": "null",
        "honest_verdict": "complete_null: bounded coverage memory did not pass every frozen gate",
    }


def cost_summary(rows: Sequence[Mapping[str, Any]], *, maximum_memory_bytes: int) -> JsonDict:
    """Reduce measured CPU costs and keep hardware targets separate."""

    names = ("lookup", "update", "serialization", "durable_commit")
    operations: dict[str, JsonDict] = {}
    for name in names:
        values = [int(row["cost_ns"]) for row in rows if row.get("operation") == name]
        operations[name] = {
            "count": len(values),
            "p50_ns": _percentile(values, 0.50),
            "p95_ns": _percentile(values, 0.95),
            "total_ns": sum(values),
        }
    return {
        "clock": "time.perf_counter_ns",
        "operations": operations,
        "maximum_memory_bytes": maximum_memory_bytes,
        "tier_1_update_target": {
            "expected": "p95 < 1000 ns",
            "observed_ns": operations["update"]["p95_ns"],
            "passed": operations["update"]["p95_ns"] < 1_000,
        },
        "tier_2_lookup_target": {
            "expected": "p95 < 1000000 ns",
            "observed_ns": operations["lookup"]["p95_ns"],
            "passed": operations["lookup"]["p95_ns"] < 1_000_000,
        },
        "hardware_acceleration_100x": {
            "target_x": 100.0,
            "measured_x": 1.0,
            "remaining_gap_x": 99.0,
            "passed": False,
        },
        "bounded_bitset_call_graph": maximum_memory_bytes <= exp7253.MEMORY_CAPS["total_bytes"],
        "measured_call_graph": [
            "CoverageArchiveController.predict",
            "CoverageArchiveController.commit_batch",
            "CoverageArchiveController.state_bytes",
            "CoverageArchiveController.save",
        ],
        "hardware_suitability_basis": "bounded survivor-mask bitsets and measured calls only",
    }


def _cost_rows(summary: Mapping[str, Any]) -> list[JsonDict]:
    """Expose one compact row for each measured operation category."""

    return [{"operation": name, **deepcopy(row)} for name, row in summary["operations"].items()]


def _control_release(event_id: str, label: str, index: int) -> JsonDict:
    """Build one released witness for isolated transaction controls."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": 0,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def run_e2e_controls(root: Path) -> list[JsonDict]:
    """Exercise E2E-007 durable restart, rejection, and exact rollback."""

    controller = exp7253.CoverageArchiveController()
    state_path = root / "controller.json"
    controller.save(state_path)
    parent_bytes = controller.state_bytes()
    receipt = controller.commit_batch(
        [_control_release("accepted", "accept", 1)],
        current_cycle=1,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    event = {"event_id": "unseen", "family_id": "lower_bound", "numeric_value": 0}
    prediction = controller.predict(event)
    restored = exp7253.CoverageArchiveController.load(state_path)
    restart_row = {
        "control": "durable_restart_decision_parity",
        "passed": restored.state_hash() == controller.state_hash()
        and restored.predict(event) == prediction,
        "parent_sha256": receipt["parent_hash"],
        "child_sha256": receipt["new_state_hash"],
    }
    child_bytes = restored.state_bytes()
    durable_child = state_path.read_bytes()
    rejected = False
    try:
        restored.commit_batch(
            [_control_release("rejected", "reject", 2)],
            current_cycle=2,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    except exp7253.ArchiveCommitRejected:
        rejected = True
    rejection_row = {
        "control": "rejected_update_preserves_bytes",
        "passed": rejected
        and restored.state_bytes() == child_bytes
        and state_path.read_bytes() == durable_child,
        "unchanged_sha256": restored.state_hash(),
    }
    rollback = restored.rollback(receipt, state_path=state_path)
    rollback_row = {
        "control": "rollback_restores_parent",
        "passed": rollback["byte_identical"] is True
        and restored.state_bytes() == parent_bytes
        and state_path.read_bytes() == parent_bytes,
        "restored_sha256": restored.state_hash(),
    }
    return [restart_row, rejection_row, rollback_row]


def _sample_budget(stream_ids: Sequence[str], *, complete: bool) -> JsonDict:
    """Return the fixed unit, row, censoring, and stopping declaration."""

    streams = len(stream_ids)
    event_rows = streams * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)
    return {
        "independent_units_planned": streams,
        "independent_units_attempted": streams if complete else 0,
        "independent_units_completed": streams if complete else 0,
        "independent_units_censored": 0,
        "events_per_unit": exp7253.EVENTS_PER_STREAM,
        "warmup_events_per_unit": exp7253.WARMUP_COUNT,
        "prospective_events_per_unit": exp7253.EVENTS_PER_STREAM - exp7253.WARMUP_COUNT,
        "arms_per_unit": len(exp7253.ARMS),
        "planned_arm_event_rows": event_rows,
        "completed_arm_event_rows": event_rows if complete else 0,
        "stopping_rule": "all predeclared streams once; no outcome-based extension",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable sources, contracts, rows, gates, receipts, and verdict."""

    excluded = {"started_at_utc", "completed_at_utc", "duration_s", "phase_spans_s"}
    return transactional.sha256_json(
        {
            key: deepcopy(value)
            for key, value in artifact.items()
            if key not in excluded and key != "reproducibility_checksum"
        }
    )


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    stream_ids: Sequence[str],
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before blocking or measured classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "current_invocation_counts": {
            "model_loads": 0,
            "generations": 0,
            "model_invocations": 0,
        },
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": socket.gethostname(),
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": {
            "root": RANDOM_SEED,
            "streams": list(exp7253.STREAM_SEEDS),
            "query_policy": "seeded_tie_ranks_from_exp7199",
            "shuffle": exp7253.SHUFFLE_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "frozen_before_first_prospective_label": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _sample_budget(stream_ids, complete=False),
        "acceptance_gate_results": {},
        "gate_check_summary": _summary_with_failures(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "coverage_run_complete_score": 0,
        "coverage_learning_value_score": 0,
        "continuous_self_learning_task": True,
        "prequential_rows_path": {
            "path": str(paths.prequential_rows),
            "sha256": None,
            "row_count": 0,
        },
        "causal_summary": {},
        "cost_rows": [],
        "operation_rows_path": {
            "path": str(paths.operation_rows),
            "sha256": None,
            "row_count": 0,
        },
        "state_manifest_path": {"path": str(paths.state_manifest), "sha256": None},
        "evidence_sidecar_path": {"path": str(paths.evidence_sidecar), "sha256": None},
        "comparison_rows": [],
        "cost_summary": {},
        "e2e_control_summary": {},
        "immutable_contract_receipt": {
            "controller_contract_sha256": EXPECTED_CONTROLLER_CONTRACT_SHA256,
            "arm_contract_sha256": EXPECTED_ARM_CONTRACT_SHA256,
            "stream_contract_sha256": EXPECTED_STREAM_CONTRACT_SHA256,
        },
        "frozen_configuration_receipt": {},
        "model_weight_immutable_receipt": {
            "no_model_weight_mutation": True,
            "MODEL_SPECS": [],
        },
        "no_model_weight_mutation": True,
        "default_pipeline_modified": False,
        "production_promotion": False,
        "publication_performed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str | None],
    paths: ExperimentPaths,
    stream_ids: Sequence[str],
    *,
    started_at: str,
    duration_s: float,
) -> JsonDict:
    """Return a row-free terminal block for an external prerequisite."""

    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        stream_ids,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    failure = artifact["gate_check_summary"].get("failed_check")
    artifact["honest_verdict"] = f"blocked_external_precondition:{failure}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipt_matches(repo_root: Path, receipt: Mapping[str, Any]) -> bool:
    """Compare a declared hash with the current exact file bytes."""

    path = receipt.get("path")
    return isinstance(path, str) and _sha256_path(_resolve(repo_root, path)) == receipt.get(
        "sha256"
    )


def attach_validation_receipts(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach exact external command receipts and refresh the stable checksum."""

    required = {"command", "exit_code", "classification", "log_sha256"}
    if any(
        set(receipt) != required
        or not isinstance(receipt["command"], str)
        or not isinstance(receipt["exit_code"], int)
        or not isinstance(receipt["classification"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt["log_sha256"])) is None
        for receipt in receipts
    ):
        raise ValueError("validation_receipt_schema")
    changed = deepcopy(dict(artifact))
    changed["validation_receipts"] = [dict(receipt) for receipt in receipts]
    changed["reproducibility_checksum"] = reproducibility_checksum(changed)
    return changed


def build_and_seal(
    repo_root: Path,
    paths: ExperimentPaths,
    *,
    stream_ids: Sequence[str] | None = None,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    upstream_path: Path | None = None,
    progress: bool = False,
) -> JsonDict:
    """Authenticate, replay, reduce, run controls, and seal sidecars."""

    selected = tuple(
        stream_ids or (f"prospective-{index + 1:02d}" for index in range(exp7253.STREAM_COUNT))
    )
    monotonic_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: JsonDict = {}
    if progress:
        _progress(0, "start", "authenticate source, quarantine, imports, outputs, and Exp7253")
    phase_start = time.monotonic()
    checks, source_hashes, upstream = collect_preconditions(
        repo_root, paths, upstream_path=upstream_path
    )
    spans["phase_0_preconditions"] = time.monotonic() - phase_start
    _atomic_write(
        paths.provisional,
        transactional.canonical_json_bytes(
            {"schema": SCHEMA, "status": "in_progress", "phase": 0, "checks": checks}
        ),
    )
    if gate_summary(checks)["passed"] is not True:
        artifact = build_blocked_artifact(
            checks,
            source_hashes,
            paths,
            selected,
            started_at=started_at,
            duration_s=time.monotonic() - monotonic_start,
        )
        if progress:
            _progress(0, "end", "external precondition failed; no CPU replay ran")
            for phase in range(1, 8):
                _progress(phase, "start", "skipped after blocking precondition")
                _progress(phase, "end", "skipped after blocking precondition")
        return artifact
    if progress:
        _progress(0, "end", "all external preconditions passed")

    phase_start = time.monotonic()
    if progress:
        _progress(1, "start", "activate monotonic progress and unbuffered phase boundaries")
    spans["phase_1_progress_contract"] = time.monotonic() - phase_start
    if progress:
        _progress(1, "end", "progress contract active")

    phase_start = time.monotonic()
    if progress:
        _progress(2, "start", "confirm zero model loads, generations, and invocations")
        print("phase 2 BEFORE model load: no model load scheduled", flush=True)
        print("phase 2 AFTER model load: model load count remains zero", flush=True)
        print("phase 2 BEFORE generation: no generation scheduled", flush=True)
        print("phase 2 AFTER generation: generation count remains zero", flush=True)
    spans["phase_2_no_llm"] = time.monotonic() - phase_start
    if progress:
        _progress(2, "end", "MODEL_SPECS is empty and every current counter is zero")

    phase_start = time.monotonic()
    if progress:
        _progress(3, "start", "load authenticated sealed streams and freeze all contracts")
    views = load_stream_views(repo_root, upstream)
    frozen_configuration = {
        "stream_ids": list(selected),
        "events_per_stream": exp7253.EVENTS_PER_STREAM,
        "arms": list(exp7253.ARMS),
        "query_ceiling": exp7253.QUERY_CEILING,
        "release_budget": exp7253.QUERY_CEILING,
        "memory_caps": deepcopy(exp7253.MEMORY_CAPS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": bootstrap_draws,
        "frozen_before_first_prospective_label": True,
    }
    frozen_configuration["sha256"] = transactional.sha256_json(frozen_configuration)
    spans["phase_3_stream_load_and_freeze"] = time.monotonic() - phase_start
    if progress:
        _progress(3, "end", f"configuration_sha256={frozen_configuration['sha256']}")

    phase_start = time.monotonic()
    if progress:
        _progress(4, "start", "BEFORE 32-stream eight-arm CPU benchmark")
        print("phase 4 BEFORE benchmark: exact finite-controller replay", flush=True)
    panel = run_learning_panel(
        views,
        state_root=paths.live_state_root,
        stream_ids=selected,
        progress=progress,
    )
    spans["phase_4_prequential_benchmark"] = time.monotonic() - phase_start
    if progress:
        print("phase 4 AFTER benchmark: scheduled event-arm rows completed", flush=True)
        _progress(4, "end", f"completed_rows={len(panel.prequential_rows)}")

    phase_start = time.monotonic()
    if progress:
        _progress(5, "start", "reduce streams and run 10000 paired bootstrap resamples")
    expected_rows = len(selected) * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)
    run_complete = len(panel.prequential_rows) == expected_rows
    comparisons = build_comparison_rows(panel.rows, draws=bootstrap_draws)
    scientific_gates = score_acceptance_gates(comparisons, panel.causal_summary)
    classification = classify_result(scientific_gates, run_complete=run_complete)
    spans["phase_5_bootstrap_and_gates"] = time.monotonic() - phase_start
    if progress:
        _progress(5, "end", f"learning_value={classification['coverage_learning_value_score']}")

    phase_start = time.monotonic()
    if progress:
        _progress(6, "start", "measure costs and run E2E-007 restart and rollback controls")
    costs = cost_summary(panel.operation_rows, maximum_memory_bytes=panel.maximum_memory_bytes)
    e2e_rows = run_e2e_controls(paths.live_state_root / "e2e")
    if any(row["passed"] is not True for row in e2e_rows):
        raise ValueError("e2e_control_failed")
    prequential_write = _atomic_write(paths.prequential_rows, _jsonl_bytes(panel.prequential_rows))
    operation_write = _atomic_write(paths.operation_rows, _jsonl_bytes(panel.operation_rows))
    state_write = _atomic_write(
        paths.state_manifest,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7254.states.v1",
                "entry_count": len(panel.state_entries),
                "entries": panel.state_entries,
                "no_model_weight_mutation": True,
            }
        ),
    )
    evidence_write = _atomic_write(
        paths.evidence_sidecar,
        transactional.canonical_json_bytes(
            {
                "schema": "carnot.exp7254.evidence.v1",
                "historical_model_receipts": {
                    "exp7253_MODEL_SPECS": upstream.get("MODEL_SPECS"),
                    "exp7253_model_invoked": upstream.get("model_invoked"),
                    "source_artifact_sha256": EXPECTED_UPSTREAM_SHA256,
                },
                "injected_negative_controls": [
                    row for row in e2e_rows if row["control"] == "rejected_update_preserves_bytes"
                ],
                "e2e_control_rows": e2e_rows,
            }
        ),
    )
    for receipt in (prequential_write, operation_write, state_write, evidence_write):
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    reduced = independent_reduce(paths.prequential_rows)
    if reduced != panel.rows:
        raise ValueError("independent_reducer_mismatch")
    bounded_memory = panel.maximum_memory_bytes <= exp7253.MEMORY_CAPS["total_bytes"]
    acceptance = {
        **scientific_gates,
        "complete_prequential_rows": {
            "principle": "Every selected sealed stream must retain all eight arm event rows.",
            "expected": expected_rows,
            "observed": len(panel.prequential_rows),
            "pass": run_complete,
        },
        "bounded_memory": {
            "principle": "Coverage archive memory must stay within the sealed total byte cap.",
            "expected": f"<={exp7253.MEMORY_CAPS['total_bytes']}",
            "observed": panel.maximum_memory_bytes,
            "pass": bounded_memory,
        },
        "e2e_controls": {
            "principle": "Restart, rejection, and rollback controls must all pass.",
            "expected": len(e2e_rows),
            "observed": sum(int(row["passed"] is True) for row in e2e_rows),
            "pass": all(row["passed"] is True for row in e2e_rows),
        },
        "independent_raw_reducer": {
            "principle": "Cold JSONL reduction must reproduce every producer summary.",
            "expected": transactional.sha256_json(panel.rows),
            "observed": transactional.sha256_json(reduced),
            "pass": reduced == panel.rows,
        },
    }
    spans["phase_6_costs_e2e_and_sidecars"] = time.monotonic() - phase_start
    if progress:
        _progress(6, "end", "cost, restart, rejection, rollback, and reducer checks passed")

    completed_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        checks,
        source_hashes,
        paths,
        selected,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - monotonic_start,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "phase_spans_s": spans,
            "random_seed": {
                **artifact["random_seed"],
                "streams": [
                    exp7253.STREAM_SEEDS[int(stream_id.split("-")[1]) - 1] for stream_id in selected
                ],
                "bootstrap_draws": bootstrap_draws,
            },
            "rows": panel.rows,
            "sample_size_budget": _sample_budget(selected, complete=run_complete),
            "acceptance_gate_results": acceptance,
            "verdict_class": classification["verdict_class"],
            "honest_verdict": classification["honest_verdict"],
            "coverage_run_complete_score": int(run_complete),
            "coverage_learning_value_score": classification["coverage_learning_value_score"],
            "prequential_rows_path": {
                **prequential_write,
                "row_count": len(panel.prequential_rows),
                "format": "jsonl",
            },
            "operation_rows_path": {
                **operation_write,
                "row_count": len(panel.operation_rows),
                "format": "jsonl",
            },
            "state_manifest_path": {**state_write, "entry_count": len(panel.state_entries)},
            "evidence_sidecar_path": {**evidence_write, "e2e_control_count": len(e2e_rows)},
            "comparison_rows": comparisons,
            "causal_summary": panel.causal_summary,
            "cost_rows": _cost_rows(costs),
            "cost_summary": costs,
            "e2e_control_summary": {
                "row_count": len(e2e_rows),
                "passed_count": sum(int(row["passed"] is True) for row in e2e_rows),
                "failed_count": sum(int(row["passed"] is not True) for row in e2e_rows),
            },
            "frozen_configuration_receipt": frozen_configuration,
            "production_promotion": classification["coverage_learning_value_score"] == 1,
            "validation_receipts": [
                {
                    "command": f"independent_reduce {paths.prequential_rows}",
                    "exit_code": 0,
                    "classification": "passed",
                    "log_sha256": transactional.sha256_json(reduced),
                }
            ],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=selected,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(paths.provisional, transactional.canonical_json_bytes(artifact))
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
    check_files: bool = False,
) -> list[str]:
    """Cold-check identity, rows, gates, receipts, costs, and classification."""

    selected = tuple(
        expected_stream_ids
        or (f"prospective-{index + 1:02d}" for index in range(exp7253.STREAM_COUNT))
    )
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(any(field not in artifact for field in REQUIRED_ARTIFACT_FIELDS), "required_fields")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    principles = artifact.get("field_principles", {})
    add(
        not isinstance(principles, Mapping)
        or any(field not in principles for field in REQUIRED_ARTIFACT_FIELDS),
        "field_principles",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(
            artifact.get(field) != 0
            for field in (
                "current_model_load_count",
                "current_generation_count",
                "current_inference_count",
            )
        ),
        "model_invocation",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "oracle_declaration")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(artifact.get("default_pipeline_modified") is not False, "production_default")
    add(
        artifact.get("verdict_class")
        not in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"},
        "verdict_class",
    )
    add(
        artifact.get("verdict_class") == "positive" and artifact.get("verifier_is_oracle") is True,
        "oracle_positive_forbidden",
    )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    if artifact.get("status") == "blocked":
        add(
            artifact.get("rows") != []
            or artifact.get("coverage_run_complete_score") != 0
            or artifact.get("coverage_learning_value_score") != 0,
            "blocked_rows",
        )
        add(
            artifact.get("inference_substrate") != "blocked_no_run"
            or artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or artifact.get("gate_check_summary", {}).get("passed") is not False,
            "blocked_contract",
        )
        return errors
    add(artifact.get("status") != "complete", "status")
    if artifact.get("status") != "complete":
        return errors
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "substrate",
    )
    add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
    rows = artifact.get("rows", [])
    expected_units = {(stream_id, arm) for stream_id in selected for arm in exp7253.ARMS}
    add(
        not isinstance(rows, list)
        or {(row.get("stream_id"), row.get("arm")) for row in rows} != expected_units,
        "rows",
    )
    expected_events = len(selected) * exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("completed_arm_event_rows") != expected_events
        or artifact.get("prequential_rows_path", {}).get("row_count") != expected_events
        or artifact.get("coverage_run_complete_score") != 1,
        "completion_score",
    )
    draws = int(artifact.get("random_seed", {}).get("bootstrap_draws", BOOTSTRAP_DRAWS))
    comparisons = build_comparison_rows(rows, draws=draws)
    add(artifact.get("comparison_rows") != comparisons, "comparison_rows")
    scientific = score_acceptance_gates(comparisons, artifact.get("causal_summary", {}))
    recorded_gates = artifact.get("acceptance_gate_results", {})
    add(
        any(recorded_gates.get(name) != scientific.get(name) for name in SCIENTIFIC_GATE_NAMES),
        "acceptance_gate_results",
    )
    classification = classify_result(scientific, run_complete=True)
    add(
        artifact.get("coverage_learning_value_score")
        != classification["coverage_learning_value_score"]
        or artifact.get("verdict_class") != classification["verdict_class"]
        or artifact.get("honest_verdict") != classification["honest_verdict"],
        "terminal_classification",
    )
    e2e = artifact.get("e2e_control_summary", {})
    add(
        e2e.get("row_count") != 3 or e2e.get("passed_count") != 3 or e2e.get("failed_count") != 0,
        "e2e_controls",
    )
    contract = artifact.get("immutable_contract_receipt", {})
    add(
        contract.get("controller_contract_sha256") != EXPECTED_CONTROLLER_CONTRACT_SHA256
        or contract.get("arm_contract_sha256") != EXPECTED_ARM_CONTRACT_SHA256
        or contract.get("stream_contract_sha256") != EXPECTED_STREAM_CONTRACT_SHA256,
        "immutable_contract",
    )
    if check_files:
        receipts = (
            artifact.get("prequential_rows_path", {}),
            artifact.get("operation_rows_path", {}),
            artifact.get("state_manifest_path", {}),
            artifact.get("evidence_sidecar_path", {}),
        )
        add(any(not _receipt_matches(repo_root, receipt) for receipt in receipts), "sidecar_hashes")
        raw_path = _resolve(repo_root, artifact["prequential_rows_path"]["path"])
        raw_rows = _read_jsonl(raw_path)
        add(prequential_row_errors(raw_rows) != [], "prequential_rows")
        add(reduce_prequential_rows(raw_rows) != rows, "independent_reducer")
        add(causal_summary_from_rows(raw_rows) != artifact.get("causal_summary"), "causal_summary")
        operation_path = _resolve(repo_root, artifact["operation_rows_path"]["path"])
        operation_rows = _read_jsonl(operation_path)
        expected_costs = cost_summary(
            operation_rows,
            maximum_memory_bytes=int(
                artifact.get("cost_summary", {}).get("maximum_memory_bytes", 0)
            ),
        )
        add(expected_costs != artifact.get("cost_summary"), "cost_summary")
        add(_cost_rows(expected_costs) != artifact.get("cost_rows"), "cost_rows")
        add(
            any(
                expected is None or _sha256_path(_resolve(repo_root, path)) != expected
                for path, expected in artifact.get("source_artifact_hashes", {}).items()
            ),
            "source_artifact_hashes",
        )
    return errors


def write_artifact(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    expected_stream_ids: Sequence[str] | None = None,
) -> None:
    """Cold-validate and publish terminal bytes through one atomic rename."""

    errors = validate_artifact(
        artifact,
        repo_root=repo_root,
        expected_stream_ids=expected_stream_ids,
        check_files=True,
    )
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_write(path, transactional.canonical_json_bytes(dict(artifact)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and optional private result root."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CPU panel and atomically write one validated terminal result."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"run_date_must_be_{RUN_DATE}")
    paths = (
        ExperimentPaths.defaults()
        if args.output_root is None
        else ExperimentPaths.under(args.output_root)
    )
    artifact = build_and_seal(REPO_ROOT, paths, progress=True)
    _progress(7, "start", "BEFORE final cold validation")
    print("phase 7 BEFORE validation: artifact, raw reducer, costs, and hashes", flush=True)
    errors = validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    print("phase 7 AFTER validation: terminal object is internally valid", flush=True)
    _progress(7, "end", "final cold validation passed")
    _progress(8, "start", "BEFORE atomic terminal write")
    print("phase 8 BEFORE atomic terminal write", flush=True)
    write_artifact(paths.artifact, artifact, repo_root=REPO_ROOT)
    print("phase 8 AFTER atomic terminal write", flush=True)
    _progress(8, "end", f"wrote {paths.artifact}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin wrapper owns execution.
    raise SystemExit(main())
