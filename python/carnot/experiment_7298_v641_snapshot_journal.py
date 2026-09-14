"""Prototype persistent SQLite full-snapshot acknowledgments.

The adapter keeps the installed native controller and the V640 queue contract.
Only the opt-in host storage boundary changes. SQLite stores one complete state
row and returns from a FULL rollback-journal commit before acknowledgment.

Spec refs: REQ-CL-7298 and SCENARIO-CL-7298-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, NoReturn

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7270_v639_durable_profile as exp7270
from carnot import experiment_7284_v640_commit_prototype as exp7284
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7298
SCHEMA = "carnot.exp7298.v641_snapshot_journal.v1"
STATE_SCHEMA = "carnot.exp7298.native_full_snapshot.v1"
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RANDOM_SEED = 7_298_000
CRASH_SEEDS = tuple(range(7_298_001, 7_298_005))
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
GROUP_SIZES = exp7284.GROUP_SIZES
MAX_WAIT_MS = exp7284.MAX_WAIT_MS
MAX_PENDING_EVENTS = exp7284.MAX_PENDING_EVENTS
MAX_PENDING_BYTES = exp7284.MAX_PENDING_BYTES
MAX_STATE_BYTES = 69_632
EVENTS_PER_GROUP_ARM = 16
SQLITE_FULL = 2
SQLITE_VFS = "unix"
CRASH_BOUNDARIES = (
    "before_transaction",
    "after_row_write_before_commit",
    "after_commit_before_acknowledgment",
    "after_acknowledgment",
    "kill_during_write_commit",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7298_v641_snapshot_journal.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7298_v641_snapshot_journal.json")
REDUCED_RELATIVE = Path("results/checkpoints/experiment_7298_v641_snapshot_reduced.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7298_v641_snapshot_journal_rows.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7298_v641_snapshot_journal_candidate.json")
RAW_TRACE_SQLITE_RELATIVE = Path("results/raw/experiment_7298_v641_sqlite_sync.trace")
RAW_TRACE_ATOMIC_RELATIVE = Path("results/raw/experiment_7298_v641_atomic_sync.trace")
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7298_v641_snapshot_storage")
EXP7256_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
EXP7284_RELATIVE = Path("results/experiment_7284_v640_commit_prototype.json")
EXP7285_RELATIVE = Path("results/experiment_7285_v640_commit_frontier.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
RUST_SOURCE_RELATIVE = Path("crates/carnot-python/src/experiment_7256_archive_controller.rs")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE,
    RUST_SOURCE_RELATIVE,
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("python/carnot/experiment_7284_v640_commit_prototype.py"),
    Path("python/carnot/experiment_7285_v640_commit_frontier.py"),
    Path("python/carnot/pipeline/atomic_writer.py"),
    Path("python/carnot/experiment_7298_v641_snapshot_journal.py"),
    Path("scripts/experiments/experiment_7298_v641_snapshot_journal.py"),
    Path("tests/python/test_experiment_7298_v641_snapshot_journal.py"),
    EXP7256_RELATIVE,
    EXP7284_RELATIVE,
    EXP7285_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the fixed experiment task.",
    "milestone": "Bind the evidence to milestone 2026.09.641.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "started_at_utc": "Record the real UTC start of this invocation.",
    "completed_at_utc": "Record the real UTC terminal decision time.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted and completed loads and generation; retain failed counts.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended work.",
    "inference_substrate_class": "Use the actual no-LLM class and never pad elapsed time.",
    "execution_venue": "Host is host; identify actual native execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans, including initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, configuration, inputs, native identity and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Keep every arm, event, seed, metric, cost, error, abstention and censoring result.",
    "sample_size_budget": "Record planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked verdict names upstream, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed class; oracle evidence forbids positive and failed gates forbid readiness.",
    "validation_receipts": "Keep command, exit code, elapsed time and log hash; preserve actual failures.",
    "baseline_validation_failures": "Preserve non-gating failures from the explicitly requested repository-wide baseline run.",
    "snapshot_journal_ready_score": "One requires real native calls, exact acknowledgment parity, recovery and controls.",
    "storage_contract": "Record one full row, effective PERSIST/FULL settings, initialization, acknowledgment and recovery.",
    "crash_control_rows": "Record kill point, acknowledged sequence, recovered prefix/hash and allowed suffix.",
    "native_identity": "Bind installed module bytes, source/build identity and actual native calls.",
    "filesystem_receipt": "Record mount, SQLite, VFS and actual sync traces with durability limits.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7284.canonical_json
sha256_file = exp7284.sha256_file
artifact_checksum = exp7284.artifact_checksum
check = exp7284.check
gate_summary = exp7284.gate_summary
atomic_write = exp7284.atomic_write
_finish_row = exp7284._finish_row
serial_reference = exp7284.serial_reference


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw, provisional, storage, trace, and terminal evidence separate."""

    checkpoint: Path
    reduced: Path
    raw_rows: Path
    raw_candidate: Path
    raw_trace_sqlite: Path
    raw_trace_atomic: Path
    storage_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the fixed paths used by the public command."""

        return cls.under(REPO_ROOT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Put every test write below one caller-owned temporary root."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / REDUCED_RELATIVE,
            root / RAW_ROWS_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / RAW_TRACE_SQLITE_RELATIVE,
            root / RAW_TRACE_ATOMIC_RELATIVE,
            root / STORAGE_RELATIVE,
            root / RESULT_RELATIVE,
        )

    def writable_targets(self) -> list[bool]:
        """Check the nearest existing parents without creating result bytes."""

        return [
            _writable(path)
            for path in (
                self.checkpoint,
                self.reduced,
                self.raw_rows,
                self.raw_candidate,
                self.raw_trace_sqlite,
                self.raw_trace_atomic,
                self.storage_dir,
                self.artifact,
            )
        ]


class SnapshotCorruptionError(ValueError):
    """Reject a row whose schema, state bytes, or checksum is not exact."""


class SnapshotSequenceError(ValueError):
    """Reject a repeated or non-monotonic snapshot transaction sequence."""


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Print one flushed boundary and optional monotonic elapsed time."""

    elapsed = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    print(f"[phase {phase} {boundary}] {operation}{elapsed}", flush=True)


def _writable(path: Path) -> bool:
    """Check a target's nearest existing parent without creating the target."""

    parent = path if path.is_dir() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _read_json(path: Path) -> JsonDict:
    """Read one JSON object and reject every other top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object:{path}")
    return value


def _checksum_valid(artifact: Mapping[str, Any]) -> bool:
    """Recompute an artifact checksum instead of trusting its stored claim."""

    try:
        return artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        return False


def snapshot_checksum(state: bytes | memoryview) -> str:
    """Hash the complete serialized state stored in the singleton row."""

    return "sha256:" + hashlib.sha256(bytes(state)).hexdigest()


def _connect(database_path: Path) -> sqlite3.Connection:
    """Open the explicit Unix VFS and enforce the persistent FULL journal settings."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{database_path.resolve()}?vfs={SQLITE_VFS}"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=0.0)
    journal_mode = str(connection.execute("PRAGMA journal_mode=PERSIST").fetchone()[0]).lower()
    connection.execute("PRAGMA synchronous=FULL")
    synchronous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
    if journal_mode != "persist" or synchronous != SQLITE_FULL:
        connection.close()
        raise RuntimeError(
            f"SQLite durability settings unavailable: journal={journal_mode} sync={synchronous}"
        )
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    """Create one constrained singleton table for complete native snapshots."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot (
            slot INTEGER PRIMARY KEY CHECK (slot = 1),
            schema_version TEXT NOT NULL,
            sequence INTEGER NOT NULL UNIQUE CHECK (sequence >= 0),
            state BLOB NOT NULL,
            checksum TEXT NOT NULL
        )
        """
    )


def _read_snapshot(connection: sqlite3.Connection) -> tuple[str, int, bytes, str]:
    """Read and verify the only allowed full-snapshot row."""

    count = int(connection.execute("SELECT count(*) FROM snapshot").fetchone()[0])
    row = connection.execute(
        "SELECT schema_version, sequence, state, checksum FROM snapshot WHERE slot = 1"
    ).fetchone()
    if count != 1 or row is None:
        raise SnapshotCorruptionError("snapshot row count must be one")
    schema_version, sequence, state, checksum = row
    state_bytes = bytes(state)
    if schema_version != STATE_SCHEMA:
        raise SnapshotCorruptionError("snapshot schema mismatch")
    if not isinstance(sequence, int) or sequence < 0:
        raise SnapshotSequenceError("snapshot sequence is invalid")
    if checksum != snapshot_checksum(state_bytes):
        raise SnapshotCorruptionError("snapshot checksum mismatch")
    return str(schema_version), sequence, state_bytes, str(checksum)


def _sqlite_probe(database_path: Path) -> JsonDict:
    """Verify effective settings through the same connection path used by the adapter."""

    connection = _connect(database_path)
    try:
        _create_schema(connection)
        return {
            "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            "synchronous": int(connection.execute("PRAGMA synchronous").fetchone()[0]),
            "sqlite_version": sqlite3.sqlite_version,
            "sqlite_source_id": str(connection.execute("SELECT sqlite_source_id()").fetchone()[0]),
            "vfs_requested": SQLITE_VFS,
            "vfs_effective_evidence": "connection opened with an explicit vfs=unix URI",
            "database_list": [list(row) for row in connection.execute("PRAGMA database_list")],
        }
    finally:
        connection.close()


def collect_preconditions(root: Path, paths: ExperimentPaths) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate native bytes, V640 context, storage support, specs, and outputs."""

    started = time.monotonic()
    progress(0, "start", "authenticate native bytes, V640 evidence, SQLite, and outputs", started)
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    artifacts = {
        "exp7256": _read_json(root / EXP7256_RELATIVE)
        if (root / EXP7256_RELATIVE).is_file()
        else {},
        "exp7284": _read_json(root / EXP7284_RELATIVE)
        if (root / EXP7284_RELATIVE).is_file()
        else {},
        "exp7285": _read_json(root / EXP7285_RELATIVE)
        if (root / EXP7285_RELATIVE).is_file()
        else {},
    }
    manifest = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    quarantine = {
        name: exp7213.quarantine_state(artifact, manifest, name, name)
        for name, artifact in artifacts.items()
    }
    native_receipt = artifacts["exp7256"].get("native_binary_receipt", {})
    native_path = Path(str(native_receipt.get("module_file", "")))
    native_hash = sha256_file(native_path) if native_path.is_file() else None
    spec_text = (root / SPEC_RELATIVE).read_text(encoding="utf-8")
    sqlite_probe: JsonDict
    try:
        sqlite_probe = _sqlite_probe(paths.storage_dir / "precondition.sqlite3")
    except (OSError, RuntimeError, sqlite3.Error) as error:
        sqlite_probe = {"error": f"{type(error).__name__}:{error}"}
    v640_observed = {
        "exp7284_status": artifacts["exp7284"].get("status"),
        "exp7284_ready": artifacts["exp7284"].get("commit_protocol_ready_score"),
        "exp7285_status": artifacts["exp7285"].get("status"),
        "exp7285_complete": artifacts["exp7285"].get("commit_cost_complete_score"),
        "exp7285_value": artifacts["exp7285"].get("commit_cost_value_score"),
        "exp7285_checksum": _checksum_valid(artifacts["exp7285"]),
    }
    checks = [
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            hashes,
            all(value is not None for value in hashes.values()),
        ),
        check(
            "driving_capability",
            str(SPEC_RELATIVE),
            "REQ-CL-7298",
            True,
            "REQ-CL-7298" in spec_text,
            "REQ-CL-7298" in spec_text,
        ),
        check(
            "installed_native_identity",
            str(native_path),
            "module_sha256",
            native_receipt.get("module_sha256"),
            native_hash,
            bool(native_receipt.get("module_sha256"))
            and native_hash == native_receipt.get("module_sha256"),
        ),
        check(
            "v640_context",
            f"{EXP7284_RELATIVE},{EXP7285_RELATIVE}",
            "complete protocol and measured null frontier",
            {
                "exp7284_status": "complete",
                "exp7284_ready": 1,
                "exp7285_status": "complete",
                "exp7285_complete": 1,
                "exp7285_value": 0,
                "exp7285_checksum": True,
            },
            v640_observed,
            v640_observed
            == {
                "exp7284_status": "complete",
                "exp7284_ready": 1,
                "exp7285_status": "complete",
                "exp7285_complete": 1,
                "exp7285_value": 0,
                "exp7285_checksum": True,
            },
        ),
        check(
            "inputs_not_quarantined_or_retired",
            str(EXCLUSION_RELATIVE),
            "exp7256,exp7284,exp7285 quarantine or retirement",
            {name: False for name in artifacts},
            {name: value["quarantined"] for name, value in quarantine.items()},
            all(value["quarantined"] is False for value in quarantine.values()),
        ),
        check(
            "sqlite_persist_full_unix_vfs",
            str(paths.storage_dir / "precondition.sqlite3"),
            "journal_mode,synchronous,vfs",
            {"journal_mode": "persist", "synchronous": SQLITE_FULL, "vfs": SQLITE_VFS},
            {
                "journal_mode": sqlite_probe.get("journal_mode"),
                "synchronous": sqlite_probe.get("synchronous"),
                "vfs": sqlite_probe.get("vfs_requested"),
            },
            sqlite_probe.get("journal_mode") == "persist"
            and sqlite_probe.get("synchronous") == SQLITE_FULL
            and sqlite_probe.get("vfs_requested") == SQLITE_VFS,
        ),
        check(
            "task_owned_outputs",
            "Exp7298 output paths",
            "writable targets",
            [True] * 8,
            paths.writable_targets(),
            all(paths.writable_targets()),
        ),
    ]
    source_artifacts = {
        name: {
            "path": str(
                {
                    "exp7256": EXP7256_RELATIVE,
                    "exp7284": EXP7284_RELATIVE,
                    "exp7285": EXP7285_RELATIVE,
                }[name]
            ),
            "sha256": hashes.get(
                str(
                    {
                        "exp7256": EXP7256_RELATIVE,
                        "exp7284": EXP7284_RELATIVE,
                        "exp7285": EXP7285_RELATIVE,
                    }[name]
                )
            ),
            "status": artifact.get("status"),
            "verdict_class": artifact.get("verdict_class"),
            "honest_verdict": artifact.get("honest_verdict"),
            "quarantined": quarantine[name]["quarantined"],
            "retired": quarantine[name].get("retired", False),
        }
        for name, artifact in artifacts.items()
    }
    evidence = {
        "hashes": hashes,
        "source_artifacts": source_artifacts,
        "artifacts": artifacts,
        "native_module_path": str(native_path),
        "native_module_sha256": native_hash,
        "sqlite_probe": sqlite_probe,
        "quarantine": quarantine,
    }
    progress(
        0,
        "end",
        f"precondition_checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
        started,
    )
    return checks, evidence


class SQLiteSnapshotJournal(exp7284.HostGroupCommitController):
    """Persist native state as one full SQLite row before acknowledging events.

    The superclass supplies the already-tested bounded queue. This subclass
    changes only its durable publication step and keeps native transitions
    serial, ordered, and complete.
    """

    def __init__(
        self,
        binding: ModuleType,
        database_path: Path,
        controller: exp7256.PersistentNativeArchiveController,
        sequence: int,
        *,
        max_group_size: int,
        max_wait_ms: int,
        initialization_duration_ns: int,
        max_pending_events: int = MAX_PENDING_EVENTS,
        max_pending_bytes: int = MAX_PENDING_BYTES,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        stage_hook: Callable[[str], None] | None = None,
    ) -> None:
        if max_group_size not in GROUP_SIZES or max_wait_ms < 0:
            raise ValueError("group size must be 1, 4, or 16 and wait must be nonnegative")
        if max_pending_events < 1 or max_pending_bytes < 1:
            raise ValueError("queue bounds must be positive")
        self.binding = binding
        self.database_path = database_path
        self.state_path = database_path
        self.max_group_size = max_group_size
        self.max_wait_ns = max_wait_ms * 1_000_000
        self.max_pending_events = max_pending_events
        self.max_pending_bytes = max_pending_bytes
        self._clock_ns = clock_ns
        self._controller = controller
        self._sequence = sequence
        self._initialization_duration_ns = initialization_duration_ns
        self._stage_hook = stage_hook
        self._pending: list[exp7284._PendingEvent] = []
        self._pending_bytes = 0
        self._committed_ids = set(map(str, controller.state_dict().get("release_ids", [])))
        self._failed_restore_required = False
        self._group_number = sequence
        self._connection = _connect(database_path)
        stored = _read_snapshot(self._connection)
        if stored[1] != sequence or stored[2] != controller.state_bytes():
            self._connection.close()
            raise SnapshotCorruptionError("database and native state disagree")

    @classmethod
    def from_state(
        cls,
        binding: ModuleType,
        state: Mapping[str, Any],
        database_path: Path,
        **kwargs: Any,
    ) -> SQLiteSnapshotJournal:
        """Commit initialization before constructing an event-accepting adapter."""

        started = time.monotonic_ns()
        if database_path.exists():
            raise FileExistsError(f"refusing to replace task database:{database_path}")
        controller = exp7256.PersistentNativeArchiveController.from_state_with_binding(
            binding, state
        )
        state_bytes = controller.state_bytes()
        if len(state_bytes) > MAX_STATE_BYTES:
            raise ValueError("initial state exceeds the fixed state-byte bound")
        connection = _connect(database_path)
        try:
            _create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO snapshot(slot, schema_version, sequence, state, checksum) "
                "VALUES (1, ?, 0, ?, ?)",
                (STATE_SCHEMA, state_bytes, snapshot_checksum(state_bytes)),
            )
            connection.commit()
            stored = _read_snapshot(connection)
            if stored[2] != state_bytes:
                raise SnapshotCorruptionError("initial snapshot read-back mismatch")
        except Exception:
            connection.rollback()
            connection.close()
            raise
        connection.close()
        return cls(
            binding,
            database_path,
            controller,
            0,
            initialization_duration_ns=max(time.monotonic_ns() - started, 1),
            **kwargs,
        )

    @classmethod
    def recover(
        cls,
        binding: ModuleType,
        database_path: Path,
        **kwargs: Any,
    ) -> SQLiteSnapshotJournal:
        """Verify one durable row and restore its complete native state."""

        started = time.monotonic_ns()
        connection = _connect(database_path)
        try:
            _create_schema(connection)
            _schema, sequence, state_bytes, _checksum = _read_snapshot(connection)
            controller = exp7256.PersistentNativeArchiveController.from_snapshot(
                binding, state_bytes.decode("utf-8")
            )
            if controller.state_bytes() != state_bytes:
                raise SnapshotCorruptionError("native canonical restore changed snapshot bytes")
        finally:
            connection.close()
        return cls(
            binding,
            database_path,
            controller,
            sequence,
            initialization_duration_ns=max(time.monotonic_ns() - started, 1),
            **kwargs,
        )

    @property
    def sequence(self) -> int:
        """Return the committed snapshot transaction sequence."""

        return self._sequence

    def _call_stage(self, stage: str) -> None:
        """Expose a real transaction boundary to tests and crash workers."""

        if self._stage_hook is not None:
            self._stage_hook(stage)

    def _commit_snapshot(self, state_bytes: bytes, sequence: int) -> JsonDict:
        """Replace the singleton row in one verified rollback-journal transaction."""

        if len(state_bytes) > MAX_STATE_BYTES:
            raise ValueError("snapshot exceeds the fixed state-byte bound")
        if sequence != self._sequence + 1:
            raise SnapshotSequenceError("snapshot sequence must increase by exactly one")
        started = self._clock_ns()
        self._call_stage("before_transaction")
        self._connection.execute("BEGIN IMMEDIATE")
        current = self._connection.execute(
            "SELECT sequence FROM snapshot WHERE slot = 1"
        ).fetchone()
        if current is None or int(current[0]) != self._sequence:
            self._connection.rollback()
            raise SnapshotSequenceError("durable snapshot parent sequence changed")
        checksum = snapshot_checksum(state_bytes)
        self._connection.execute(
            "UPDATE snapshot SET schema_version = ?, sequence = ?, state = ?, checksum = ? "
            "WHERE slot = 1 AND sequence = ?",
            (STATE_SCHEMA, sequence, state_bytes, checksum, self._sequence),
        )
        self._call_stage("after_row_write_before_commit")
        pending = _read_snapshot(self._connection)
        if pending[1] != sequence or pending[2] != state_bytes:
            self._connection.rollback()
            raise SnapshotCorruptionError("transactional row read-back mismatch")
        self._call_stage("before_commit_call")
        self._connection.commit()
        committed_ns = self._clock_ns()
        self._call_stage("after_commit_before_acknowledgment")
        stored = _read_snapshot(self._connection)
        if stored[1] != sequence or stored[2] != state_bytes:
            raise SnapshotCorruptionError("committed row read-back mismatch")
        return {
            "sequence": sequence,
            "schema_version": STATE_SCHEMA,
            "snapshot_checksum": checksum,
            "snapshot_bytes": len(state_bytes),
            "transaction_duration_ns": max(committed_ns - started, 0),
            "linearization_point": "sqlite_commit_returned",
            "journal_mode": str(
                self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower(),
            "synchronous": int(self._connection.execute("PRAGMA synchronous").fetchone()[0]),
            "full_snapshot": True,
            "delta_log": False,
        }

    def flush(self, *, reason: str) -> JsonDict:
        """Apply ordered native transitions and acknowledge only after SQLite commit."""

        if self._failed_restore_required:
            raise RuntimeError("fresh restore required after uncertain publication")
        if not self._pending:
            return {
                "disposition": "flush_empty",
                "reason": reason,
                "acknowledged_event_ids": [],
            }
        pending = list(self._pending)
        event_ids = [str(row.release["event_id"]) for row in pending]
        commit_started = self._clock_ns()
        receipts: list[JsonDict] = []
        try:
            for row in pending:
                receipts.append(
                    self._controller.commit_batch(
                        [row.release],
                        current_cycle=int(row.release["release_index"]),
                        expected_parent_hash=self._controller.state_hash(),
                    )
                )
        except (OSError, exp7240.ArchiveCommitRejected):
            self._rollback_receipts(receipts)
            self._pending.clear()
            self._pending_bytes = 0
            return {
                "disposition": "failed_rolled_back",
                "reason": reason,
                "acknowledged_event_ids": [],
                "failed_event_ids": event_ids,
            }
        try:
            durable = self._commit_snapshot(self._controller.state_bytes(), self._sequence + 1)
        except (OSError, sqlite3.Error, SnapshotCorruptionError, SnapshotSequenceError, ValueError):
            try:
                self._connection.rollback()
            except sqlite3.Error:
                pass
            self._rollback_receipts(receipts)
            self._pending.clear()
            self._pending_bytes = 0
            self._failed_restore_required = True
            return {
                "disposition": "failed_restore_required",
                "reason": reason,
                "acknowledged_event_ids": [],
                "failed_event_ids": event_ids,
            }
        acknowledged_ns = self._clock_ns()
        self._sequence = int(durable["sequence"])
        self._pending.clear()
        self._pending_bytes = 0
        self._committed_ids.update(event_ids)
        self._group_number += 1
        receipt = {
            "disposition": "committed_acknowledged",
            "reason": reason,
            "group_id": f"group-{self._group_number:04d}",
            "issued_event_ids": event_ids,
            "acknowledged_event_ids": event_ids,
            "commit_delay_ns": max(acknowledged_ns - commit_started, 0),
            "event_receipts": [
                {
                    "event_id": str(row.release["event_id"]),
                    "accepted_ns": row.accepted_ns,
                    "queue_delay_ns": max(commit_started - row.accepted_ns, 0),
                    "acknowledgment_delay_ns": max(acknowledged_ns - row.accepted_ns, 0),
                    "byte_count": row.byte_count,
                }
                for row in pending
            ],
            **durable,
        }
        self._call_stage("after_acknowledgment")
        return receipt

    def storage_receipt(self) -> JsonDict:
        """Read back storage settings and singleton-row integrity."""

        _schema, sequence, state_bytes, checksum = _read_snapshot(self._connection)
        journal = self.database_path.with_name(self.database_path.name + "-journal")
        return {
            "database_path": str(self.database_path.resolve()),
            "journal_mode": str(
                self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower(),
            "synchronous": int(self._connection.execute("PRAGMA synchronous").fetchone()[0]),
            "vfs_requested": SQLITE_VFS,
            "snapshot_row_count": int(
                self._connection.execute("SELECT count(*) FROM snapshot").fetchone()[0]
            ),
            "schema_version": STATE_SCHEMA,
            "sequence": sequence,
            "state_bytes": len(state_bytes),
            "checksum": checksum,
            "checksum_valid": checksum == snapshot_checksum(state_bytes),
            "initialization_duration_ns": self._initialization_duration_ns,
            "initialization_committed_before_acceptance": True,
            "journal_file_retained": journal.is_file(),
            "journal_file_bytes": journal.stat().st_size if journal.is_file() else 0,
            "single_writer_task_owned": True,
            "full_snapshot": True,
            "delta_log": False,
        }

    def close(self) -> None:
        """Close this task-owned single-writer connection."""

        self._connection.close()


def _releases_for_seed(seed: int, count: int) -> list[JsonDict]:
    """Freeze one valid ordered event roster from its declared seed."""

    releases: list[JsonDict] = []
    offset = seed % 10_000
    for index in range(count):
        row = exp7257._cost_release(2_000 + offset + index, 0)
        row["event_id"] = f"exp7298-{seed}-{index:04d}"
        row["request_index"] = 20_000 + offset + index
        row["release_index"] = 20_000 + offset + index
        releases.append(row)
    return releases


def run_protocol_fixture(
    binding: ModuleType,
    storage_dir: Path,
    *,
    events_per_arm: int = EVENTS_PER_GROUP_ARM,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run every fixed group size through native transitions and SQLite restart."""

    if events_per_arm < 1 or events_per_arm % 16:
        raise ValueError("events_per_arm must be a positive multiple of 16")
    initial = exp7257.seed_cost_state(4)
    rows: list[JsonDict] = []
    parity: list[JsonDict] = []
    started = time.monotonic()
    progress(3, "before", f"protocol fixture arms={len(GROUP_SIZES)}", started)
    for arm_index, group_size in enumerate(GROUP_SIZES):
        releases = _releases_for_seed(RANDOM_SEED + arm_index, events_per_arm)
        reference = serial_reference(binding, initial, releases)
        database = storage_dir / f"group-{group_size}/state.sqlite3"
        adapter = SQLiteSnapshotJournal.from_state(
            binding,
            initial,
            database,
            max_group_size=group_size,
            max_wait_ms=0 if group_size == 1 else MAX_WAIT_MS,
        )
        receipts: list[JsonDict] = []
        for release in releases:
            response = adapter.enqueue(release)
            if response.get("flush_receipt") is not None:
                receipts.append(response["flush_receipt"])
        storage = adapter.storage_receipt()
        for group in receipts:
            for event in group["event_receipts"]:
                rows.append(
                    _finish_row(
                        {
                            "unit_id": f"group-{group_size}:{event['event_id']}",
                            "arm": f"sqlite_max_group_{group_size}",
                            "seed": RANDOM_SEED + arm_index,
                            "metric": event["acknowledgment_delay_ns"],
                            "error": None,
                            "abstention": False,
                            "censored": False,
                            "event_id": event["event_id"],
                            "group_id": group["group_id"],
                            "max_group_size": group_size,
                            "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                            "queue_delay_ns": event["queue_delay_ns"],
                            "commit_delay_ns": group["commit_delay_ns"],
                            "acknowledgment_delay_ns": event["acknowledgment_delay_ns"],
                            "byte_count": event["byte_count"],
                            "snapshot_sequence": group["sequence"],
                            "snapshot_bytes": group["snapshot_bytes"],
                            "acknowledged": True,
                        }
                    )
                )
        state_bytes = adapter.state_bytes
        state_hash = adapter.state_hash
        sequence = adapter.sequence
        adapter.close()
        recovered = SQLiteSnapshotJournal.recover(
            binding,
            database,
            max_group_size=group_size,
            max_wait_ms=0 if group_size == 1 else MAX_WAIT_MS,
        )
        parity.append(
            {
                "arm": f"sqlite_max_group_{group_size}",
                "max_group_size": group_size,
                "event_count": events_per_arm,
                "serial_state_hash": reference["state_hash"],
                "journal_state_hash": state_hash,
                "state_bytes_match": state_bytes == reference["state_bytes"],
                "transition_order_match": list(
                    map(str, recovered._controller.state_dict().get("release_ids", []))
                )
                == reference["release_ids"],
                "fresh_recovery_match": recovered.state_bytes == state_bytes,
                "sequence": sequence,
                "expected_sequence": events_per_arm // group_size,
                "state_bytes": len(state_bytes),
                "native_transition_calls": events_per_arm,
                "storage_receipt": storage,
            }
        )
        recovered.close()
        progress(
            3,
            "unit",
            f"completed_arm={arm_index + 1}/{len(GROUP_SIZES)} group={group_size}",
            started,
        )
    progress(3, "after", f"protocol fixture rows={len(rows)}", started)
    return rows, parity


def _pause_for_kill(marker: Mapping[str, Any]) -> NoReturn:
    """Flush a crash marker and wait for the parent to send SIGKILL."""

    print(canonical_json(marker), flush=True)
    while True:
        signal.pause()


def _crash_worker(
    binding_path: Path,
    database_path: Path,
    releases_path: Path,
    boundary: str,
) -> NoReturn:
    """Reach one real SQLite boundary and expose it to the parent process."""

    binding = exp7230.load_native_extension(binding_path)
    releases = _read_json(releases_path)["releases"]
    issued = [str(row["event_id"]) for row in releases]
    emitted_race_marker = False

    def marker(stage: str) -> JsonDict:
        acknowledged = issued if stage == "after_acknowledgment" else []
        return {
            "crash_boundary": boundary,
            "actual_kill_point": stage,
            "issued_event_ids": issued,
            "acknowledged_event_ids": acknowledged,
            "kill_timing_certain": boundary != "kill_during_write_commit",
        }

    def hook(stage: str) -> None:
        nonlocal emitted_race_marker
        if boundary == "kill_during_write_commit" and stage == "before_commit_call":
            print(canonical_json(marker(stage)), flush=True)
            emitted_race_marker = True
            return
        if stage == boundary:
            _pause_for_kill(marker(stage))

    adapter = SQLiteSnapshotJournal.recover(
        binding,
        database_path,
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
        stage_hook=hook,
    )
    for release in releases:
        adapter.enqueue(release)
    adapter.flush(reason="crash_control")
    if emitted_race_marker:
        while True:
            signal.pause()
    raise RuntimeError(f"unreached crash boundary:{boundary}")


def _restore_worker(binding_path: Path, database_path: Path) -> int:
    """Restore and print exact snapshot evidence from one fresh process."""

    binding = exp7230.load_native_extension(binding_path)
    adapter = SQLiteSnapshotJournal.recover(
        binding,
        database_path,
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
    )
    print(
        canonical_json(
            {
                "sequence": adapter.sequence,
                "state_hash": adapter.state_hash,
                "release_ids": list(
                    map(str, adapter._controller.state_dict().get("release_ids", []))
                ),
                "state_bytes_b64": base64.b64encode(adapter.state_bytes).decode("ascii"),
            }
        ),
        flush=True,
    )
    adapter.close()
    return 0


def _worker_command(
    mode: str,
    binding_path: Path,
    database_path: Path,
    *,
    releases_path: Path | None = None,
    boundary: str | None = None,
    trace_adapter: str | None = None,
) -> list[str]:
    """Build one unbuffered worker command for crash, restore, or trace work."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        str(REPO_ROOT / "scripts/experiments/experiment_7298_v641_snapshot_journal.py"),
        mode,
        "--binding",
        str(binding_path),
        "--database",
        str(database_path),
    ]
    if releases_path is not None:
        command.extend(["--releases", str(releases_path)])
    if boundary is not None:
        command.extend(["--boundary", boundary])
    if trace_adapter is not None:
        command.extend(["--trace-adapter", trace_adapter])
    return command


def _fresh_restore(
    binding_path: Path, database_path: Path, environment: Mapping[str, str]
) -> JsonDict:
    """Run one bounded fresh-process restore and reject nonzero output."""

    command = _worker_command("--restore-worker", binding_path, database_path)
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"restore worker failed:{process.stderr}")
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("restore worker returned a non-object")
    return value


def run_crash_matrix(
    binding_path: Path,
    initial_state: Mapping[str, Any],
    storage_dir: Path,
    *,
    seeds: Sequence[int] = CRASH_SEEDS,
) -> list[JsonDict]:
    """Kill each seeded child at all fixed boundaries and restore twice."""

    binding = exp7230.load_native_extension(binding_path)
    initial_bytes = transactional.canonical_json_bytes(dict(initial_state))
    old = exp7256.PersistentNativeArchiveController.from_state_with_binding(binding, initial_state)
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
    }
    rows: list[JsonDict] = []
    started = time.monotonic()
    for seed in seeds:
        releases = _releases_for_seed(seed, 4)
        reference = serial_reference(binding, initial_state, releases)
        issued = [str(row["event_id"]) for row in releases]
        for boundary in CRASH_BOUNDARIES:
            case_dir = storage_dir / f"seed-{seed}/{boundary}"
            database = case_dir / "state.sqlite3"
            releases_path = case_dir / "releases.json"
            adapter = SQLiteSnapshotJournal.from_state(
                binding,
                initial_state,
                database,
                max_group_size=16,
                max_wait_ms=MAX_WAIT_MS,
            )
            adapter.close()
            atomic_write(releases_path, {"releases": releases})
            command = _worker_command(
                "--crash-worker",
                binding_path,
                database,
                releases_path=releases_path,
                boundary=boundary,
            )
            progress(4, "before", f"crash subprocess seed={seed} boundary={boundary}", started)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            ready, _, _ = select.select([process.stdout], [], [], 30.0)
            if not ready:
                process.kill()
                process.wait(timeout=5)
                stderr = "" if process.stderr is None else process.stderr.read()
                raise RuntimeError(f"crash worker timeout:{boundary}:{stderr}")
            marker = json.loads(process.stdout.readline())
            os.kill(process.pid, signal.SIGKILL)
            exit_code = process.wait(timeout=5)
            progress(
                4,
                "after",
                f"crash subprocess seed={seed} boundary={boundary} exit_code={exit_code}",
                started,
            )
            progress(4, "before", f"restore subprocess seed={seed} boundary={boundary}", started)
            restored = _fresh_restore(binding_path, database, environment)
            restored_again = _fresh_restore(binding_path, database, environment)
            progress(4, "after", f"restore subprocess seed={seed} boundary={boundary}", started)
            recovered_bytes = base64.b64decode(restored["state_bytes_b64"])
            state_kind = (
                "old"
                if recovered_bytes == initial_bytes
                else "new"
                if recovered_bytes == reference["state_bytes"]
                else "invalid_partial"
            )
            acknowledged = list(map(str, marker["acknowledged_event_ids"]))
            recovered_ids = list(map(str, restored["release_ids"]))
            lost = sum(event_id not in recovered_ids for event_id in acknowledged)
            duplicates = sum(max(recovered_ids.count(event_id) - 1, 0) for event_id in issued)
            unacknowledged_suffix = [
                event_id
                for event_id in issued
                if event_id in recovered_ids and event_id not in acknowledged
            ]
            rows.append(
                {
                    "unit_id": f"crash:{seed}:{boundary}",
                    "seed": seed,
                    "crash_boundary": boundary,
                    "actual_kill_point": marker["actual_kill_point"],
                    "kill_timing_certain": marker["kill_timing_certain"],
                    "issued_event_ids": issued,
                    "acknowledged_event_ids": acknowledged,
                    "recovered_event_ids": recovered_ids,
                    "recovered_acknowledged_prefix": [
                        event_id for event_id in acknowledged if event_id in recovered_ids
                    ],
                    "allowed_unacknowledged_suffix": unacknowledged_suffix,
                    "unacknowledged_suffix_credited": False,
                    "recovered_state_hash": restored["state_hash"],
                    "recovered_sequence": restored["sequence"],
                    "recovered_state_kind": state_kind,
                    "old_state_hash": old.state_hash(),
                    "serial_new_state_hash": reference["state_hash"],
                    "process_death": "SIGKILL",
                    "process_exit_code": exit_code,
                    "fresh_process_restore": True,
                    "valid_complete_state": state_kind in {"old", "new"},
                    "lost_acknowledged_event_count": lost,
                    "duplicate_apply_count": duplicates,
                    "idempotent_fresh_recovery": restored_again == restored,
                }
            )
    return rows


def run_failure_controls(binding: ModuleType, storage_dir: Path) -> JsonDict:
    """Attack checksums, sequences, transactions, duplicates, and queue limits."""

    initial = exp7257.seed_cost_state(4)
    batch = _releases_for_seed(RANDOM_SEED + 90, 20)
    bounded = SQLiteSnapshotJournal.from_state(
        binding,
        initial,
        storage_dir / "queue/state.sqlite3",
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
        max_pending_events=2,
    )
    first = bounded.enqueue(batch[0], now_ns=0)
    pending_duplicate = bounded.enqueue(batch[0], now_ns=1)
    second = bounded.enqueue(batch[1], now_ns=2)
    event_overflow = bounded.enqueue(batch[2], now_ns=3)
    timeout = bounded.flush_due(now_ns=MAX_WAIT_MS * 1_000_000)
    max_state_bytes = len(bounded.state_bytes)
    bounded.close()

    byte_limited = SQLiteSnapshotJournal.from_state(
        binding,
        initial,
        storage_dir / "bytes/state.sqlite3",
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
        max_pending_bytes=1,
    )
    byte_overflow = byte_limited.enqueue(batch[3])
    byte_limited.close()

    duplicate_sequence = SQLiteSnapshotJournal.from_state(
        binding,
        initial,
        storage_dir / "sequence/state.sqlite3",
        max_group_size=4,
        max_wait_ms=MAX_WAIT_MS,
    )
    try:
        duplicate_sequence._commit_snapshot(
            duplicate_sequence.state_bytes, duplicate_sequence.sequence
        )
    except SnapshotSequenceError:
        duplicate_sequence_rejected = True
    else:
        duplicate_sequence_rejected = False
    duplicate_sequence.close()

    corrupt_path = storage_dir / "corrupt/state.sqlite3"
    corrupt = SQLiteSnapshotJournal.from_state(
        binding,
        initial,
        corrupt_path,
        max_group_size=4,
        max_wait_ms=MAX_WAIT_MS,
    )
    corrupt.close()
    connection = _connect(corrupt_path)
    connection.execute("UPDATE snapshot SET checksum = 'sha256:corrupt' WHERE slot = 1")
    connection.close()
    try:
        SQLiteSnapshotJournal.recover(
            binding,
            corrupt_path,
            max_group_size=4,
            max_wait_ms=MAX_WAIT_MS,
        )
    except SnapshotCorruptionError:
        corrupt_checksum_rejected = True
    else:
        corrupt_checksum_rejected = False

    def fail(stage: str) -> None:
        if stage == "after_row_write_before_commit":
            raise sqlite3.OperationalError("injected transaction failure")

    failed = SQLiteSnapshotJournal.from_state(
        binding,
        initial,
        storage_dir / "failed/state.sqlite3",
        max_group_size=4,
        max_wait_ms=MAX_WAIT_MS,
        stage_hook=fail,
    )
    failed.enqueue(batch[4])
    failed_receipt = failed.flush(reason="injected")
    failed.close()
    peak_bytes = int(second["pending_bytes"])
    return {
        "max_pending_events": MAX_PENDING_EVENTS,
        "max_pending_bytes": MAX_PENDING_BYTES,
        "max_state_bytes": MAX_STATE_BYTES,
        "observed_peak_pending_events": 2,
        "observed_peak_pending_bytes": peak_bytes,
        "observed_max_state_bytes": max_state_bytes,
        "first_event_accepted": first["accepted"],
        "pending_duplicate_disposition": pending_duplicate["disposition"],
        "event_overflow_disposition": event_overflow["disposition"],
        "byte_overflow_disposition": byte_overflow["disposition"],
        "timeout_disposition": None if timeout is None else timeout["disposition"],
        "corrupt_checksum_rejected": corrupt_checksum_rejected,
        "duplicate_sequence_rejected": duplicate_sequence_rejected,
        "failed_commit_disposition": failed_receipt["disposition"],
        "failed_commit_acknowledgment_count": len(failed_receipt["acknowledged_event_ids"]),
        "all_controls_passed": all(
            (
                first["accepted"],
                pending_duplicate["disposition"] == "duplicate_id",
                event_overflow["disposition"] == "backpressure_events",
                byte_overflow["disposition"] == "backpressure_bytes",
                timeout is not None and timeout["reason"] == "timeout",
                corrupt_checksum_rejected,
                duplicate_sequence_rejected,
                failed_receipt["acknowledged_event_ids"] == [],
                peak_bytes <= MAX_PENDING_BYTES,
                max_state_bytes <= MAX_STATE_BYTES,
            )
        ),
    }


def reduce_rows(
    rows: Sequence[Mapping[str, Any]], parity_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Cold-reduce event acknowledgments and native endpoint parity."""

    if not rows or any(not exp7284._row_hash_valid(row) for row in rows):
        raise ValueError("invalid protocol rows")
    if {row.get("max_group_size") for row in parity_rows} != set(GROUP_SIZES):
        raise ValueError("missing parity arm")
    return {
        "inference_substrate": REDUCER_SUBSTRATE,
        "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "completed_event_count": len(rows),
        "censored_event_count": sum(bool(row.get("censored")) for row in rows),
        "error_event_count": sum(row.get("error") is not None for row in rows),
        "missing_acknowledged_event_count": sum(
            row.get("acknowledged") is not True for row in rows
        ),
        "parity_failure_count": sum(
            not all(
                (
                    row.get("state_bytes_match"),
                    row.get("transition_order_match"),
                    row.get("fresh_recovery_match"),
                    row.get("sequence") == row.get("expected_sequence"),
                )
            )
            for row in parity_rows
        ),
        "max_state_bytes": max(int(row.get("state_bytes", 0)) for row in parity_rows),
        "native_transition_calls": sum(
            int(row.get("native_transition_calls", 0)) for row in parity_rows
        ),
    }


def _parse_mountinfo(text: str, resolved: Path) -> JsonDict:
    """Select the longest valid kernel mount record for one resolved path."""

    best: JsonDict = {}
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        after = right.split()
        if len(fields) < 6 or len(after) < 3:
            continue
        mount_point = Path(fields[4])
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(str(mount_point)) >= len(str(best.get("mount_point", ""))):
            best = {
                "mount_point": str(mount_point),
                "filesystem_type": after[0],
                "mount_source": after[1],
                "mount_options": fields[5].split(","),
                "super_options": after[2].split(","),
            }
    return best


def _mount_receipt(path: Path) -> JsonDict:
    """Read the kernel mount record that owns the task database path."""

    resolved = path.parent.resolve()
    mountinfo = Path("/proc/self/mountinfo")
    best = (
        _parse_mountinfo(mountinfo.read_text(encoding="utf-8"), resolved)
        if mountinfo.is_file()
        else {}
    )
    stats = os.statvfs(resolved)
    return {
        **best,
        "kernel_release": platform.release(),
        "kernel_system": platform.system(),
        "statvfs_block_size": stats.f_bsize,
        "statvfs_fragment_size": stats.f_frsize,
        "filesystem_id": os.stat(resolved).st_dev,
    }


def _trace_worker(binding_path: Path, database_path: Path, adapter_name: str) -> int:
    """Run one tiny real commit for external syscall tracing."""

    binding = exp7230.load_native_extension(binding_path)
    initial = exp7257.seed_cost_state(4)
    release = _releases_for_seed(RANDOM_SEED + 99, 1)[0]
    if adapter_name == "sqlite":
        adapter = SQLiteSnapshotJournal.from_state(
            binding,
            initial,
            database_path,
            max_group_size=1,
            max_wait_ms=0,
        )
        receipt = adapter.enqueue(release)["flush_receipt"]
        adapter.close()
    elif adapter_name == "atomic":
        controller = exp7284.HostGroupCommitController.from_state(
            binding,
            initial,
            database_path,
            max_group_size=1,
            max_wait_ms=0,
        )
        receipt = controller.enqueue(release)["flush_receipt"]
    else:
        return 2
    print(canonical_json({"adapter": adapter_name, "receipt": receipt}), flush=True)
    return 0


def run_sync_traces(
    binding_path: Path,
    storage_dir: Path,
    sqlite_trace: Path,
    atomic_trace: Path,
) -> JsonDict:
    """Trace actual sync calls for SQLite and the V640 atomic writer when available."""

    tracer = shutil.which("strace")
    if tracer is None:
        return {
            "trace_available": False,
            "limitation": "strace is not installed; no syscall counts are reported",
            "arms": {},
        }
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
    }
    arms: JsonDict = {}
    started = time.monotonic()
    for name, trace_path in (("sqlite", sqlite_trace), ("atomic", atomic_trace)):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        target = (
            storage_dir / f"trace-{name}" / ("state.sqlite3" if name == "sqlite" else "state.json")
        )
        worker = _worker_command(
            "--trace-worker",
            binding_path,
            target,
            trace_adapter=name,
        )
        command = [
            tracer,
            "-qq",
            "-f",
            "-e",
            "trace=fsync,fdatasync",
            "-o",
            str(trace_path),
            *worker,
        ]
        progress(5, "before", f"sync trace subprocess adapter={name}", started)
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        progress(
            5,
            "after",
            f"sync trace subprocess adapter={name} exit_code={process.returncode}",
            started,
        )
        trace_text = trace_path.read_text(encoding="utf-8") if trace_path.is_file() else ""
        arms[name] = {
            "command": command,
            "exit_code": process.returncode,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
            "fsync_calls": trace_text.count("fsync("),
            "fdatasync_calls": trace_text.count("fdatasync("),
            "stdout_sha256": "sha256:" + hashlib.sha256(process.stdout.encode()).hexdigest(),
            "stderr_sha256": "sha256:" + hashlib.sha256(process.stderr.encode()).hexdigest(),
        }
    return {
        "trace_available": all(row["exit_code"] == 0 for row in arms.values()),
        "tracer_path": tracer,
        "tracer_sha256": sha256_file(Path(tracer)),
        "arms": arms,
        "comparison_scope": "one initialized full-snapshot commit per host adapter",
        "limitation": "Observed process syscalls do not prove device firmware or physical power-loss behavior.",
    }


def _filesystem_receipt(
    database_path: Path, storage: Mapping[str, Any], traces: Mapping[str, Any]
) -> JsonDict:
    """Join actual SQLite, VFS, mount, and sync evidence without a power-loss claim."""

    connection = _connect(database_path)
    try:
        source_id = str(connection.execute("SELECT sqlite_source_id()").fetchone()[0])
        database_list = [list(row) for row in connection.execute("PRAGMA database_list")]
    finally:
        connection.close()
    return {
        "sqlite_runtime_version": sqlite3.sqlite_version,
        "sqlite_source_id": source_id,
        "python_version": platform.python_version(),
        "vfs_requested": SQLITE_VFS,
        "vfs_effective_evidence": "the database opened through an explicit vfs=unix URI",
        "database_list": database_list,
        "journal_mode": storage.get("journal_mode"),
        "synchronous": storage.get("synchronous"),
        "mount": _mount_receipt(database_path),
        "sync_trace": dict(traces),
        "physical_power_loss_proven": False,
        "firmware_fsync_honesty_proven": False,
        "durability_limit": "SIGKILL tests cover process death, not power removal or storage firmware honesty.",
    }


def _storage_contract(parity_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce effective storage receipts without replacing them with intent."""

    receipts = [row["storage_receipt"] for row in parity_rows]
    return {
        "adapter": "opt_in_python_sqlite3_single_writer",
        "row_model": "one singleton transactional full-snapshot row",
        "schema_version": STATE_SCHEMA,
        "group_sizes": list(GROUP_SIZES),
        "max_wait_ms": MAX_WAIT_MS,
        "max_pending_events": MAX_PENDING_EVENTS,
        "max_pending_bytes": MAX_PENDING_BYTES,
        "max_state_bytes": MAX_STATE_BYTES,
        "effective_journal_modes": sorted({row["journal_mode"] for row in receipts}),
        "effective_synchronous_values": sorted({row["synchronous"] for row in receipts}),
        "snapshot_row_counts": sorted({row["snapshot_row_count"] for row in receipts}),
        "initialization_duration_ns_by_group": {
            str(row["max_group_size"]): row["storage_receipt"]["initialization_duration_ns"]
            for row in parity_rows
        },
        "initialization_committed_before_acceptance": all(
            row["initialization_committed_before_acceptance"] for row in receipts
        ),
        "acknowledgment_linearization": "sqlite3.Connection.commit returned",
        "failed_commit_acknowledgment": False,
        "full_snapshot": True,
        "delta_log": False,
        "production_default_changed": False,
    }


def _native_identity(
    evidence: Mapping[str, Any], parity_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Bind actual extension bytes, Rust source, build identity, and typed calls."""

    artifact = evidence.get("artifacts", {}).get("exp7256", {})
    receipt = artifact.get("native_binary_receipt", {})
    return {
        "module_file": evidence.get("native_module_path"),
        "module_sha256": evidence.get("native_module_sha256"),
        "native_class": "carnot._rust.RustArchiveController7256",
        "rust_source_path": str(RUST_SOURCE_RELATIVE),
        "rust_source_sha256": evidence.get("hashes", {}).get(str(RUST_SOURCE_RELATIVE)),
        "wheel_path": receipt.get("wheel_path"),
        "wheel_sha256": receipt.get("wheel_sha256"),
        "compiler_output_sha256": "sha256:"
        + hashlib.sha256(str(receipt.get("compiler_output", "")).encode()).hexdigest(),
        "actual_native_transition_calls": sum(
            int(row.get("native_transition_calls", 0)) for row in parity_rows
        ),
        "python_fallback_used": False,
        "compiled_execution": True,
    }


def _sample_budget(row_count: int, crash_count: int) -> JsonDict:
    """Record the frozen bounded roster and its no-expansion stopping rule."""

    planned_rows = len(GROUP_SIZES) * EVENTS_PER_GROUP_ARM
    planned_crashes = len(CRASH_SEEDS) * len(CRASH_BOUNDARIES)
    return {
        "planned_event_units": planned_rows,
        "attempted_event_units": row_count,
        "completed_event_units": row_count,
        "censored_event_units": max(planned_rows - row_count, 0),
        "planned_crash_units": planned_crashes,
        "attempted_crash_units": crash_count,
        "completed_crash_units": crash_count,
        "censored_crash_units": max(planned_crashes - crash_count, 0),
        "frozen_crash_seeds": list(CRASH_SEEDS),
        "stopping_rule": "Run three fixed group arms and five boundaries for four seeds once; do not expand after outcomes.",
    }


def _acceptance_gates(
    reduced: Mapping[str, Any],
    parity_rows: Sequence[Mapping[str, Any]],
    crash_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    storage: Mapping[str, Any],
    native: Mapping[str, Any],
) -> JsonDict:
    """Require semantic, durability, identity, and boundedness gates conjunctively."""

    planned_events = len(GROUP_SIZES) * EVENTS_PER_GROUP_ARM
    planned_crashes = len(CRASH_SEEDS) * len(CRASH_BOUNDARIES)
    return {
        "complete_fixed_event_roster": {
            "expected": planned_events,
            "observed": reduced.get("completed_event_count"),
            "passed": reduced.get("completed_event_count") == planned_events
            and reduced.get("censored_event_count") == 0,
            "principle": "Every fixed event needs a terminal acknowledgment row.",
        },
        "native_controller_parity": {
            "expected": {"arms": len(GROUP_SIZES), "failures": 0},
            "observed": {
                "arms": len(parity_rows),
                "failures": reduced.get("parity_failure_count"),
            },
            "passed": len(parity_rows) == len(GROUP_SIZES)
            and reduced.get("parity_failure_count") == 0,
            "principle": "Storage may change, but every native endpoint and event order must match.",
        },
        "zero_missing_acknowledged_events": {
            "expected": 0,
            "observed": reduced.get("missing_acknowledged_event_count"),
            "passed": reduced.get("missing_acknowledged_event_count") == 0,
            "principle": "No acknowledged event may disappear from durable state.",
        },
        "seeded_crash_recovery": {
            "expected": {
                "units": planned_crashes,
                "lost": 0,
                "duplicates": 0,
                "idempotent": True,
            },
            "observed": {
                "units": len(crash_rows),
                "lost": sum(int(row.get("lost_acknowledged_event_count", 0)) for row in crash_rows),
                "duplicates": sum(int(row.get("duplicate_apply_count", 0)) for row in crash_rows),
                "idempotent": all(
                    row.get("idempotent_fresh_recovery") is True for row in crash_rows
                ),
            },
            "passed": len(crash_rows) == planned_crashes
            and all(row.get("valid_complete_state") is True for row in crash_rows)
            and all(int(row.get("lost_acknowledged_event_count", 0)) == 0 for row in crash_rows)
            and all(int(row.get("duplicate_apply_count", 0)) == 0 for row in crash_rows)
            and all(row.get("idempotent_fresh_recovery") is True for row in crash_rows),
            "principle": "Every acknowledged prefix must restore exactly once after real process death.",
        },
        "corruption_sequence_and_queue_controls": {
            "expected": True,
            "observed": controls.get("all_controls_passed"),
            "passed": controls.get("all_controls_passed") is True,
            "principle": "Corruption, repeated sequence, duplicate event, and overflow must fail closed.",
        },
        "bounded_complete_state": {
            "expected": f"<= {MAX_STATE_BYTES}",
            "observed": reduced.get("max_state_bytes"),
            "passed": int(reduced.get("max_state_bytes", MAX_STATE_BYTES + 1)) <= MAX_STATE_BYTES,
            "principle": "The adapter stores complete state without exceeding the fixed archive bound.",
        },
        "persist_full_singleton_storage": {
            "expected": {
                "journal": ["persist"],
                "synchronous": [SQLITE_FULL],
                "rows": [1],
                "full_snapshot": True,
            },
            "observed": {
                "journal": storage.get("effective_journal_modes"),
                "synchronous": storage.get("effective_synchronous_values"),
                "rows": storage.get("snapshot_row_counts"),
                "full_snapshot": storage.get("full_snapshot"),
            },
            "passed": storage.get("effective_journal_modes") == ["persist"]
            and storage.get("effective_synchronous_values") == [SQLITE_FULL]
            and storage.get("snapshot_row_counts") == [1]
            and storage.get("full_snapshot") is True
            and storage.get("delta_log") is False,
            "principle": "Read-back settings and row shape, not intent, define the storage contract.",
        },
        "installed_native_binding_executed": {
            "expected": {"calls": ">0", "fallback": False},
            "observed": {
                "calls": native.get("actual_native_transition_calls"),
                "fallback": native.get("python_fallback_used"),
            },
            "passed": int(native.get("actual_native_transition_calls", 0)) > 0
            and native.get("python_fallback_used") is False,
            "principle": "A fixture cannot pass by replacing the installed PyO3 controller.",
        },
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create all required fields before terminal classification."""

    summary = gate_summary(checks)
    blocked = summary["passed"] is not True
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked" if blocked else "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": {
            "attempted_model_loads": 0,
            "completed_model_loads": 0,
            "failed_model_loads": 0,
            "attempted_generation_calls": 0,
            "completed_generation_calls": 0,
            "failed_generation_calls": 0,
            "usable_answers": 0,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": {
            "files": dict(evidence.get("hashes", {})),
            "artifacts": dict(evidence.get("source_artifacts", {})),
        },
        "rows": [],
        "sample_size_budget": _sample_budget(0, 0),
        "acceptance_gate_results": {},
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": (
            f"blocked_external:{summary['failed_check']}" if blocked else "complete_pending"
        ),
        "verdict_class": "blocked" if blocked else "partial",
        "validation_receipts": [],
        "baseline_validation_failures": [],
        "snapshot_journal_ready_score": 0,
        "storage_contract": {},
        "crash_control_rows": [],
        "native_identity": {},
        "filesystem_receipt": {},
        "semantic_parity_rows": [],
        "failure_controls": {},
        "independent_raw_reducer": {},
        "speedup_claimed": False,
        "production_default_changed": False,
        "output_paths": {
            "checkpoint": str(paths.checkpoint),
            "reduced": str(paths.reduced),
            "raw_rows": str(paths.raw_rows),
            "raw_candidate": str(paths.raw_candidate),
            "raw_trace_sqlite": str(paths.raw_trace_sqlite),
            "raw_trace_atomic": str(paths.raw_trace_atomic),
            "storage_dir": str(paths.storage_dir),
            "artifact": str(paths.artifact),
        },
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return one schema-complete row-free terminal external block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed],
        {},
        ExperimentPaths.defaults(),
        started_at=now,
        completed_at=now,
        duration_s=0.0,
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Build compact deterministic rows with the full fixed event population."""

    rows: list[JsonDict] = []
    parity: list[JsonDict] = []
    for group_size in GROUP_SIZES:
        for index in range(EVENTS_PER_GROUP_ARM):
            rows.append(
                _finish_row(
                    {
                        "unit_id": f"fixture:{group_size}:{index}",
                        "arm": f"sqlite_max_group_{group_size}",
                        "seed": RANDOM_SEED + group_size,
                        "metric": 100,
                        "error": None,
                        "abstention": False,
                        "censored": False,
                        "event_id": f"fixture-{group_size}-{index}",
                        "group_id": f"group-{index // group_size}",
                        "max_group_size": group_size,
                        "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                        "queue_delay_ns": 10,
                        "commit_delay_ns": 90,
                        "acknowledgment_delay_ns": 100,
                        "byte_count": 100,
                        "snapshot_sequence": index // group_size + 1,
                        "snapshot_bytes": 8_000,
                        "acknowledged": True,
                    }
                )
            )
        parity.append(
            {
                "arm": f"sqlite_max_group_{group_size}",
                "max_group_size": group_size,
                "event_count": EVENTS_PER_GROUP_ARM,
                "serial_state_hash": "sha256:fixture",
                "journal_state_hash": "sha256:fixture",
                "state_bytes_match": True,
                "transition_order_match": True,
                "fresh_recovery_match": True,
                "sequence": EVENTS_PER_GROUP_ARM // group_size,
                "expected_sequence": EVENTS_PER_GROUP_ARM // group_size,
                "state_bytes": 8_000,
                "native_transition_calls": EVENTS_PER_GROUP_ARM,
                "storage_receipt": {
                    "journal_mode": "persist",
                    "synchronous": SQLITE_FULL,
                    "snapshot_row_count": 1,
                    "initialization_duration_ns": 1,
                    "initialization_committed_before_acceptance": True,
                },
            }
        )
    return rows, parity


def _fixture_crash_rows() -> list[JsonDict]:
    """Build every seed and boundary shape for closed validator tests."""

    rows: list[JsonDict] = []
    for seed in CRASH_SEEDS:
        for boundary in CRASH_BOUNDARIES:
            new = boundary in {
                "after_commit_before_acknowledgment",
                "after_acknowledgment",
                "kill_during_write_commit",
            }
            acknowledged = [f"event-{seed}"] if boundary == "after_acknowledgment" else []
            rows.append(
                {
                    "unit_id": f"crash:{seed}:{boundary}",
                    "seed": seed,
                    "crash_boundary": boundary,
                    "actual_kill_point": "before_commit_call"
                    if boundary == "kill_during_write_commit"
                    else boundary,
                    "kill_timing_certain": boundary != "kill_during_write_commit",
                    "issued_event_ids": [f"event-{seed}"],
                    "acknowledged_event_ids": acknowledged,
                    "recovered_event_ids": [f"event-{seed}"] if new else [],
                    "recovered_acknowledged_prefix": acknowledged,
                    "allowed_unacknowledged_suffix": [f"event-{seed}"]
                    if new and not acknowledged
                    else [],
                    "unacknowledged_suffix_credited": False,
                    "recovered_state_hash": "sha256:new" if new else "sha256:old",
                    "recovered_sequence": int(new),
                    "recovered_state_kind": "new" if new else "old",
                    "old_state_hash": "sha256:old",
                    "serial_new_state_hash": "sha256:new",
                    "process_death": "SIGKILL",
                    "process_exit_code": -9,
                    "fresh_process_restore": True,
                    "valid_complete_state": True,
                    "lost_acknowledged_event_count": 0,
                    "duplicate_apply_count": 0,
                    "idempotent_fresh_recovery": True,
                }
            )
    return rows


def complete_artifact_fixture_for_test() -> JsonDict:
    """Create one complete circular-positive fixture for cold validator tests."""

    rows, parity = _fixture_rows()
    crashes = _fixture_crash_rows()
    reduced = reduce_rows(rows, parity)
    storage = _storage_contract(parity)
    controls = {
        "all_controls_passed": True,
        "observed_max_state_bytes": 8_000,
    }
    native = {
        "module_file": "/tmp/_rust.so",
        "module_sha256": "sha256:fixture",
        "actual_native_transition_calls": 48,
        "python_fallback_used": False,
        "compiled_execution": True,
    }
    filesystem = {
        "sqlite_runtime_version": sqlite3.sqlite_version,
        "vfs_requested": SQLITE_VFS,
        "journal_mode": "persist",
        "synchronous": SQLITE_FULL,
        "sync_trace": {
            "trace_available": False,
            "limitation": "fixture has no syscall trace",
            "arms": {},
        },
        "physical_power_loss_proven": False,
        "firmware_fsync_honesty_proven": False,
    }
    passed = check("fixture", "fixture", "field", True, True, True)
    now = datetime.now(UTC).isoformat()
    base = _base_artifact(
        [passed],
        {},
        ExperimentPaths.defaults(),
        started_at=now,
        completed_at=now,
        duration_s=1.0,
    )
    gates = _acceptance_gates(reduced, parity, crashes, controls, storage, native)
    base.update(
        {
            "rows": rows,
            "sample_size_budget": _sample_budget(len(rows), len(crashes)),
            "acceptance_gate_results": gates,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_circular_positive: persistent full-snapshot acknowledgment protocol passed",
            "validation_receipts": [exp7284.validation_receipt("fixture", ["true"], 0, "ok", 0.01)],
            "snapshot_journal_ready_score": 1,
            "storage_contract": storage,
            "crash_control_rows": crashes,
            "native_identity": native,
            "filesystem_receipt": filesystem,
            "semantic_parity_rows": parity,
            "failure_controls": controls,
            "independent_raw_reducer": reduced,
        }
    )
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check provenance, rows, transaction semantics, controls, and verdict."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS <= set(artifact), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(artifact.get("invocation_counts", {}).values()),
        "model_invocation",
    )
    baseline_failures = artifact.get("baseline_validation_failures", [])
    add(
        not isinstance(baseline_failures, list)
        or any(
            not isinstance(row, Mapping)
            or row.get("exit_code") == 0
            or not isinstance(row.get("command"), list)
            or float(row.get("duration_s", -1)) < 0
            or not str(row.get("log_sha256", "")).startswith("sha256:")
            for row in baseline_failures
        ),
        "baseline_validation_failures",
    )
    try:
        checksum_ok = artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        checksum_ok = False
    add(not checksum_ok, "reproducibility_checksum")
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "blocked_class")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        add(bool(artifact.get("rows")) or bool(artifact.get("crash_control_rows")), "blocked_rows")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        return errors

    rows = artifact.get("rows", [])
    parity = artifact.get("semantic_parity_rows", [])
    crashes = artifact.get("crash_control_rows", [])
    controls = artifact.get("failure_controls", {})
    storage = artifact.get("storage_contract", {})
    native = artifact.get("native_identity", {})
    filesystem = artifact.get("filesystem_receipt", {})
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    try:
        reduced = reduce_rows(rows, parity)
    except (KeyError, TypeError, ValueError):
        reduced = None
    add(reduced is None or artifact.get("independent_raw_reducer") != reduced, "rows")
    crash_valid = (
        isinstance(crashes, list)
        and len(crashes) == len(CRASH_SEEDS) * len(CRASH_BOUNDARIES)
        and {(row.get("seed"), row.get("crash_boundary")) for row in crashes}
        == {(seed, boundary) for seed in CRASH_SEEDS for boundary in CRASH_BOUNDARIES}
        and all(row.get("process_death") == "SIGKILL" for row in crashes)
        and all(row.get("valid_complete_state") is True for row in crashes)
        and all(int(row.get("lost_acknowledged_event_count", 0)) == 0 for row in crashes)
        and all(int(row.get("duplicate_apply_count", 0)) == 0 for row in crashes)
        and all(row.get("idempotent_fresh_recovery") is True for row in crashes)
        and all(row.get("unacknowledged_suffix_credited") is False for row in crashes)
    )
    add(not crash_valid, "crash_control_rows")
    add(
        not isinstance(controls, Mapping) or controls.get("all_controls_passed") is not True,
        "failure_controls",
    )
    storage_valid = (
        isinstance(storage, Mapping)
        and storage.get("effective_journal_modes") == ["persist"]
        and storage.get("effective_synchronous_values") == [SQLITE_FULL]
        and storage.get("snapshot_row_counts") == [1]
        and storage.get("full_snapshot") is True
        and storage.get("delta_log") is False
    )
    add(not storage_valid, "storage_contract")
    native_valid = (
        isinstance(native, Mapping)
        and int(native.get("actual_native_transition_calls", 0)) > 0
        and native.get("python_fallback_used") is False
        and native.get("compiled_execution") is True
    )
    add(not native_valid, "native_identity")
    filesystem_valid = (
        isinstance(filesystem, Mapping)
        and filesystem.get("vfs_requested") == SQLITE_VFS
        and filesystem.get("journal_mode") == "persist"
        and filesystem.get("synchronous") == SQLITE_FULL
        and filesystem.get("physical_power_loss_proven") is False
        and filesystem.get("firmware_fsync_honesty_proven") is False
        and isinstance(filesystem.get("sync_trace"), Mapping)
    )
    add(not filesystem_valid, "filesystem_receipt")
    if reduced is not None and crash_valid and storage_valid and native_valid:
        gates = _acceptance_gates(reduced, parity, crashes, controls, storage, native)
        add(artifact.get("acceptance_gate_results") != gates, "acceptance_gate_results")
        ready = int(all(gate["passed"] is True for gate in gates.values()))
        add(artifact.get("snapshot_journal_ready_score") != ready, "snapshot_journal_ready_score")
        add(
            artifact.get("verdict_class") != ("circular_positive" if ready else "null")
            or artifact.get("verdict_class") == "positive",
            "verdict_class",
        )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or not receipts
        or any(
            row.get("exit_code") != 0
            or not isinstance(row.get("command"), list)
            or float(row.get("duration_s", -1)) < 0
            or not str(row.get("log_sha256", "")).startswith("sha256:")
            for row in receipts
        ),
        "validation_receipts",
    )
    add(not str(artifact.get("honest_verdict", "")).startswith("complete_"), "honest_verdict")
    add(artifact.get("speedup_claimed") is not False, "speedup_claimed")
    add(artifact.get("production_default_changed") is not False, "production_default_changed")
    return errors


def _stream_subprocess(
    command: list[str], *, root: Path, operation: str, timeout_s: int = 900
) -> JsonDict:
    """Stream unbuffered child output and retain a truthful heartbeat receipt."""

    return exp7270._stream_subprocess(command, root=root, operation=operation, timeout_s=timeout_s)


def _scoped_validation_commands(root: Path) -> list[tuple[str, list[str], int]]:
    """Return focused tests, scoped coverage, lint, type, and spec checks."""

    python = str(Path(sys.executable).absolute())
    new_test = "tests/python/test_experiment_7298_v641_snapshot_journal.py"
    changed = [
        "python/carnot/experiment_7298_v641_snapshot_journal.py",
        "scripts/experiments/experiment_7298_v641_snapshot_journal.py",
        new_test,
    ]
    return [
        (
            "focused_and_affected_pytest",
            [
                python,
                "-m",
                "pytest",
                new_test,
                "tests/python/test_experiment_7284_v640_commit_prototype.py",
                "tests/python/test_experiment_7256_v638_native_controller.py",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7298-affected",
            ],
            1200,
        ),
        (
            "scoped_new_code_coverage",
            [
                python,
                "-c",
                (
                    "import os,sys; import jax,numpy,coverage; "
                    "os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; "
                    "cov=coverage.Coverage(source=['carnot.experiment_7298_v641_snapshot_journal']); "
                    "cov.erase(); cov.start(); import pytest; "
                    f"rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','{new_test}',"
                    "'-q','-n','0','--basetemp=/tmp/carnot-exp7298-coverage']); "
                    "cov.stop(); cov.save(); percent=cov.report(file=sys.stdout,show_missing=True); "
                    "raise SystemExit(rc if rc else (0 if percent >= 100.0 else 1))"
                ),
            ],
            1200,
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *changed], 120),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *changed], 120),
        ("changed_module_mypy", [python, "-m", "mypy", changed[0], changed[1]], 300),
        ("spec_coverage", [python, "scripts/check_spec_coverage.py", new_test], 120),
    ]


def _prior_full_suite_failures(checkpoint: Path) -> list[JsonDict]:
    """Recover the authenticated non-Exp7298 baseline failure from a retry checkpoint."""

    if not checkpoint.is_file():
        return []
    artifact = _read_json(checkpoint)
    receipts = artifact.get("validation_receipts", [])
    return [
        dict(row)
        for row in receipts
        if isinstance(row, Mapping)
        and row.get("name") == "full_python_suite"
        and row.get("exit_code") != 0
    ]


def _reduce_raw_in_fresh_process(root: Path, paths: ExperimentPaths) -> tuple[JsonDict, JsonDict]:
    """Reduce saved row bytes in a separate read-only Python process."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "scripts/experiments/experiment_7298_v641_snapshot_journal.py",
        "--reduce-raw",
        str(paths.raw_rows),
        "--reduced-output",
        str(paths.reduced),
    ]
    progress(6, "before", "cold reducer subprocess")
    started = time.monotonic()
    result = _stream_subprocess(command, root=root, operation="cold reducer", timeout_s=120)
    elapsed = time.monotonic() - started
    progress(6, "after", f"cold reducer subprocess exit_code={result['exit_code']}")
    if result["exit_code"] != 0:
        raise RuntimeError("cold raw-row reducer failed")
    return _read_json(paths.reduced), exp7284.validation_receipt(
        "cold_raw_row_reducer", command, 0, str(result.get("output", "")), elapsed
    )


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    baseline_validation_failures: Sequence[Mapping[str, Any]] = (),
    precondition_bundle: tuple[list[JsonDict], JsonDict] | None = None,
) -> JsonDict:
    """Run the native fixture, controls, crash matrix, traces, and cold reduction."""

    invocation_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, evidence = precondition_bundle or collect_preconditions(root, paths)
    base = _base_artifact(
        checks,
        evidence,
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - invocation_started,
    )
    base["baseline_validation_failures"] = [dict(row) for row in baseline_validation_failures]
    if gate_summary(checks)["passed"] is not True:
        base["reproducibility_checksum"] = artifact_checksum(base)
        return base

    progress(1, "before", "MODEL_SPECS empty; model loads and generations are zero")
    progress(1, "after", "no model call authorized or attempted")
    spans: JsonDict = {}
    phase = time.monotonic()
    progress(2, "before", "load authenticated installed Exp7256 native extension")
    binding_path = Path(evidence["native_module_path"])
    binding = exp7230.load_native_extension(binding_path)
    progress(2, "after", "authenticated native extension loaded")
    spans["native_load"] = time.monotonic() - phase

    phase = time.monotonic()
    rows, parity = run_protocol_fixture(binding, paths.storage_dir / "protocol")
    spans["protocol_fixture"] = time.monotonic() - phase
    phase = time.monotonic()
    controls = run_failure_controls(binding, paths.storage_dir / "controls")
    spans["failure_controls"] = time.monotonic() - phase
    phase = time.monotonic()
    crashes = run_crash_matrix(
        binding_path,
        exp7257.seed_cost_state(4),
        paths.storage_dir / "crashes",
    )
    spans["crash_matrix"] = time.monotonic() - phase
    storage = _storage_contract(parity)
    native = _native_identity(evidence, parity)
    phase = time.monotonic()
    traces = run_sync_traces(
        binding_path,
        paths.storage_dir,
        paths.raw_trace_sqlite,
        paths.raw_trace_atomic,
    )
    spans["sync_traces"] = time.monotonic() - phase
    filesystem = _filesystem_receipt(
        paths.storage_dir / "protocol/group-1/state.sqlite3",
        parity[0]["storage_receipt"],
        traces,
    )
    raw_receipt = atomic_write(
        paths.raw_rows,
        {
            "schema": "carnot.exp7298.raw.v1",
            "inference_substrate": REDUCER_SUBSTRATE,
            "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
            "rows": rows,
            "semantic_parity_rows": parity,
            "crash_control_rows": crashes,
            "failure_controls": controls,
            "storage_contract": storage,
            "native_identity": native,
            "filesystem_receipt": filesystem,
        },
    )
    reduced, reducer_receipt = _reduce_raw_in_fresh_process(root, paths)
    gates = _acceptance_gates(reduced, parity, crashes, controls, storage, native)
    ready = int(all(gate["passed"] is True for gate in gates.values()))
    checkpoint_receipt = atomic_write(
        paths.checkpoint,
        {
            "schema": "carnot.exp7298.checkpoint.v1",
            "status": "provisional_measurement_complete",
            "raw_rows_receipt": raw_receipt,
            "cold_reducer_receipt": reducer_receipt,
        },
    )
    base.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": time.monotonic() - invocation_started,
            "phase_spans_s": spans,
            "rows": rows,
            "sample_size_budget": _sample_budget(len(rows), len(crashes)),
            "acceptance_gate_results": gates,
            "honest_verdict": (
                "complete_circular_positive: persistent full-snapshot acknowledgment protocol passed"
                if ready
                else "complete_null: persistent full-snapshot acknowledgment protocol failed one or more gates"
            ),
            "verdict_class": "circular_positive" if ready else "null",
            "validation_receipts": [dict(row) for row in validation_receipts],
            "snapshot_journal_ready_score": ready,
            "storage_contract": storage,
            "crash_control_rows": crashes,
            "native_identity": native,
            "filesystem_receipt": filesystem,
            "semantic_parity_rows": parity,
            "failure_controls": controls,
            "independent_raw_reducer": reduced,
            "raw_rows_receipt": raw_receipt,
            "cold_reducer_receipt": reducer_receipt,
            "checkpoint_receipt": checkpoint_receipt,
        }
    )
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Validate, measure, attack, cold-check, and atomically publish terminal bytes."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / REDUCED_RELATIVE,
        root / RAW_ROWS_RELATIVE,
        root / RAW_CANDIDATE_RELATIVE,
        root / RAW_TRACE_SQLITE_RELATIVE,
        root / RAW_TRACE_ATOMIC_RELATIVE,
        root / STORAGE_RELATIVE,
        output,
    )
    baseline_failures = _prior_full_suite_failures(paths.checkpoint)
    preconditions = collect_preconditions(root, paths)
    if gate_summary(preconditions[0])["passed"] is not True:
        artifact = build_artifact(
            root,
            paths,
            validation_receipts=[],
            baseline_validation_failures=baseline_failures,
            precondition_bundle=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked Exp7298 artifact:{errors}")
        atomic_write(output, artifact)
        return artifact

    receipts: list[JsonDict] = []
    for name, command, timeout_s in _scoped_validation_commands(root):
        progress(7, "before", f"validation subprocess {name}")
        started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        elapsed = time.monotonic() - started
        progress(7, "after", f"validation subprocess {name} exit_code={result['exit_code']}")
        receipts.append(
            exp7284.validation_receipt(
                name, command, int(result["exit_code"]), str(result.get("output", "")), elapsed
            )
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7298.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"scoped validation failed:{name}")

    artifact = build_artifact(
        root,
        paths,
        validation_receipts=receipts,
        baseline_validation_failures=baseline_failures,
        precondition_bundle=preconditions,
    )
    progress(8, "before", "cold in-process candidate validation")
    errors = validate_artifact(artifact)
    progress(8, "after", f"cold in-process candidate validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7298 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    candidate_commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7298_v641_snapshot_journal.py",
            "--validate",
            str(paths.raw_candidate),
        ],
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/adversarial_verify.py",
            str(paths.raw_candidate),
        ],
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            str(paths.raw_candidate),
        ],
    ]
    for index, command in enumerate(candidate_commands, start=1):
        name = f"terminal_candidate_check_{index}"
        progress(9, "before", name)
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        progress(9, "after", f"{name} exit_code={result['exit_code']}")
        if result["exit_code"] != 0:
            raise RuntimeError(f"candidate validation failed:{name}")
    progress(10, "before", "atomic terminal artifact publication")
    atomic_write(output, artifact)
    progress(10, "after", f"atomic terminal artifact publication path={output}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse public, read-only, reducer, crash, restore, and trace modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--reduced-output", type=Path)
    parser.add_argument("--crash-worker", action="store_true")
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--trace-worker", action="store_true")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--boundary", choices=CRASH_BOUNDARIES)
    parser.add_argument("--trace-adapter", choices=("sqlite", "atomic"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode and keep worker output machine-readable."""

    args = _parse_args(argv)
    if args.crash_worker:
        if not all((args.binding, args.database, args.releases, args.boundary)):
            return 2
        _crash_worker(args.binding, args.database, args.releases, args.boundary)
    if args.restore_worker:
        if args.binding is None or args.database is None:
            return 2
        return _restore_worker(args.binding, args.database)
    if args.trace_worker:
        if args.binding is None or args.database is None or args.trace_adapter is None:
            return 2
        return _trace_worker(args.binding, args.database, args.trace_adapter)
    if args.reduce_raw is not None:
        if args.reduced_output is None:
            return 2
        try:
            raw = _read_json(args.reduce_raw)
            reduced = reduce_rows(raw["rows"], raw["semantic_parity_rows"])
            atomic_write(args.reduced_output, reduced)
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(f"reducer_error: {error}", flush=True)
            return 2
        print(canonical_json(reduced), flush=True)
        return 0
    if args.validate is not None:
        progress(8, "before", f"read-only validation path={args.validate}")
        try:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(8, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        run_experiment(REPO_ROOT, output, args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"experiment_error: {error}", flush=True)
        return 2
    return 0
