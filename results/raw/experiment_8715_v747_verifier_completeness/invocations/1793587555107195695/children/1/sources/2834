"""Measure persistent full-snapshot storage cost against atomic replacement.

Both arms keep the installed native transition and the V640 queue contract.
The benchmark changes only the durable host storage adapter. It reports host
timing and does not imply FPGA performance or Carnot's tenfold NFR target.

Spec refs: REQ-CL-7299 and SCENARIO-CL-7299-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7270_v639_durable_profile as exp7270
from carnot import experiment_7284_v640_commit_prototype as exp7284
from carnot import experiment_7285_v640_commit_frontier as exp7285
from carnot import experiment_7298_v641_snapshot_journal as exp7298
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7299
SCHEMA = "carnot.exp7299.v641_snapshot_cost.v1"
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RANDOM_SEED = 7_299_000
EVALUATION_SEEDS = tuple(RANDOM_SEED + index for index in range(1, 9))
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
ARRIVAL_PROCESSES = ("burst", "steady_paced", "interactive_dependent")
STORAGE_ARMS = ("atomic_replace", "sqlite_persist")
GROUP_SIZES = (1, 4, 16)
MAX_WAIT_MS = 10
MAX_PENDING_EVENTS = exp7284.MAX_PENDING_EVENTS
MAX_PENDING_BYTES = exp7284.MAX_PENDING_BYTES
STEADY_INTERARRIVAL_NS = 1_000_000
EVENTS_PER_TRIAL = 256
MAX_BENCHMARK_S = 1_800.0
BOOTSTRAP_DRAWS = 10_000
DEPLOYMENT_CANDIDATE = {"storage_arm": "sqlite_persist", "max_group_size": 16}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7299_v641_snapshot_cost.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7299_v641_snapshot_cost.json")
REDUCED_RELATIVE = Path("results/checkpoints/experiment_7299_v641_snapshot_cost_reduced.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7299_v641_snapshot_cost_rows.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7299_v641_snapshot_cost_candidate.json")
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7299_v641_snapshot_cost_storage")
VALIDATION_RELATIVE = Path("results/raw/experiment_7299_v641_snapshot_cost/validation")
EXP7298_RELATIVE = Path("results/experiment_7298_v641_snapshot_journal.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
EXPECTED_EXP7298_SHA256 = "sha256:2f4deeb11ffe3f21f1800f3eeb41a952e1f6bc6625e3849eec33bf022a04d7d5"

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE,
    Path("python/carnot/experiment_7257_v638_native_cost.py"),
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("python/carnot/experiment_7284_v640_commit_prototype.py"),
    Path("python/carnot/experiment_7285_v640_commit_frontier.py"),
    Path("python/carnot/experiment_7298_v641_snapshot_journal.py"),
    Path("python/carnot/experiment_7299_v641_snapshot_cost.py"),
    Path("scripts/experiments/experiment_7299_v641_snapshot_cost.py"),
    Path("tests/python/test_experiment_7299_v641_snapshot_cost.py"),
    EXP7298_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the declared experiment.",
    "milestone": "Bind the evidence to milestone 2026.09.641.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "started_at_utc": "Record when the actual invocation started.",
    "completed_at_utc": "Record when the terminal decision completed.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Use the actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual native or device execution separately.",
    "duration_s": "Use measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Keep every comparative event, arm, seed, metric, cost, error, abstention and censoring result.",
    "sample_size_budget": "Record planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness or value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked verdict names upstream, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_; external absence starts blocked_; retain the actual finding.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "validation_receipts": "Keep command, exit code, elapsed time and log hash; preserve actual failures.",
    "baseline_validation_failures": "Preserve authenticated unrelated full-suite failures without calling them passing validation.",
    "snapshot_capture_complete_score": "One requires complete paired workloads, native identity, timing and independent recovery.",
    "snapshot_value_score": "One requires every frozen group-16 throughput, latency, durability and queue gate.",
    "per_run_results": "Keep every seed, arrival process, arm, group, latency, throughput, cost and censoring unit.",
    "phase_cost_rows": "Keep initialization, queue, native, serialization, write, sync, acknowledgment and recovery totals.",
    "independent_recovery_rows": "Compare acknowledged and recovered sequences plus exact state hashes after every restart.",
    "nfr01_assessment": "Evaluate the original full-boundary tenfold target separately from the local threshold.",
    "independent_raw_reducer": "Bind the fresh-process reduction of immutable raw rows.",
    "limitations": "Expose filesystem caching, firmware sync, and physical power-loss limits.",
    "claim_boundary": "Prevent host storage timing from becoming an FPGA or tenfold Carnot claim.",
    "phase_spans_s": "Keep disjoint measured phase durations for the terminal invocation.",
    "output_paths": "Keep raw, checkpoint, storage, validation, and terminal ownership explicit.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7284.canonical_json
sha256_file = exp7284.sha256_file
artifact_checksum = exp7284.artifact_checksum
check = exp7284.check
gate_summary = exp7284.gate_summary
atomic_write = exp7284.atomic_write
_finish_row = exp7284._finish_row
validation_receipt = exp7284.validation_receipt


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw, provisional, storage, validation, and terminal bytes separate."""

    checkpoint: Path
    reduced: Path
    raw_rows: Path
    raw_candidate: Path
    storage_dir: Path
    validation_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the fixed paths used by the public command."""

        return cls.under(REPO_ROOT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place all test writes below a caller-owned temporary root."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / REDUCED_RELATIVE,
            root / RAW_ROWS_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / STORAGE_RELATIVE,
            root / VALIDATION_RELATIVE,
            root / RESULT_RELATIVE,
        )

    def writable_targets(self) -> list[bool]:
        """Authenticate output parents without creating success-shaped bytes."""

        return [
            _writable(path)
            for path in (
                self.checkpoint,
                self.reduced,
                self.raw_rows,
                self.raw_candidate,
                self.storage_dir,
                self.validation_dir,
                self.artifact,
            )
        ]


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Print a flushed boundary with optional monotonic elapsed time."""

    elapsed = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    print(f"[phase {phase} {boundary}] {operation}{elapsed}", flush=True)


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output bytes."""

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
    """Recompute an artifact checksum instead of trusting its declaration."""

    try:
        return artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        return False


def _sha256_bytes(data: bytes) -> str:
    """Return the project SHA-256 spelling for immutable byte identities."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    exp7298_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate Exp7298, native and storage identities, specs, and outputs."""

    started = time.monotonic()
    progress(0, "start", "authenticate Exp7298, native storage identities, and outputs", started)
    upstream_path = exp7298_path or root / EXP7298_RELATIVE
    upstream = _read_json(upstream_path) if upstream_path.is_file() else {}
    upstream_hash = sha256_file(upstream_path) if upstream_path.is_file() else None
    manifest = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    quarantine = exp7213.quarantine_state(
        upstream, manifest, upstream_path.name, "exp7298-snapshot-journal"
    )
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    hashes[str(upstream_path.resolve())] = upstream_hash
    native_identity = upstream.get("native_identity", {})
    native_path = Path(str(native_identity.get("module_file", "")))
    native_hash = sha256_file(native_path) if native_path.is_file() else None
    storage = upstream.get("storage_contract", {})
    spec_text = (root / SPEC_RELATIVE).read_text(encoding="utf-8")
    expected_storage = {
        "journal": ["persist"],
        "synchronous": [exp7298.SQLITE_FULL],
        "groups": list(GROUP_SIZES),
        "max_wait_ms": MAX_WAIT_MS,
        "max_pending_events": MAX_PENDING_EVENTS,
        "max_pending_bytes": MAX_PENDING_BYTES,
        "full_snapshot": True,
        "delta_log": False,
    }
    observed_storage = {
        "journal": storage.get("effective_journal_modes"),
        "synchronous": storage.get("effective_synchronous_values"),
        "groups": storage.get("group_sizes"),
        "max_wait_ms": storage.get("max_wait_ms"),
        "max_pending_events": storage.get("max_pending_events"),
        "max_pending_bytes": storage.get("max_pending_bytes"),
        "full_snapshot": storage.get("full_snapshot"),
        "delta_log": storage.get("delta_log"),
    }
    checks = [
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            hashes,
            all(hashes.get(str(path)) is not None for path in SOURCE_PATHS),
        ),
        check(
            "driving_capability",
            str(SPEC_RELATIVE),
            "REQ-CL-7299",
            True,
            "REQ-CL-7299" in spec_text,
            "REQ-CL-7299" in spec_text,
        ),
        check(
            "exp7298_artifact_hash",
            str(upstream_path),
            "sha256",
            EXPECTED_EXP7298_SHA256,
            upstream_hash,
            upstream_hash == EXPECTED_EXP7298_SHA256,
        ),
        check(
            "exp7298_snapshot_journal_ready",
            str(upstream_path),
            "status,checksum,snapshot_journal_ready_score",
            {"status": "complete", "checksum": True, "ready": 1},
            {
                "status": upstream.get("status"),
                "checksum": _checksum_valid(upstream),
                "ready": upstream.get("snapshot_journal_ready_score"),
            },
            upstream.get("status") == "complete"
            and _checksum_valid(upstream)
            and upstream.get("snapshot_journal_ready_score") == 1,
        ),
        check(
            "exp7298_not_quarantined_or_retired",
            str(EXCLUSION_RELATIVE),
            "quarantined,retired",
            {"quarantined": False, "retired": False},
            {
                "quarantined": quarantine.get("quarantined"),
                "retired": quarantine.get("retired", False),
            },
            quarantine.get("quarantined") is False and quarantine.get("retired", False) is False,
        ),
        check(
            "effective_storage_identity",
            str(upstream_path),
            "PERSIST,FULL,groups,deadlines,queue,full snapshot",
            expected_storage,
            observed_storage,
            observed_storage == expected_storage,
        ),
        check(
            "native_identity_and_task_owned_outputs",
            str(native_path),
            "module_sha256,writable outputs,filesystem device",
            {
                "module_sha256": native_identity.get("module_sha256"),
                "outputs": [True] * 7,
                "filesystem_device": root.stat().st_dev,
            },
            {
                "module_sha256": native_hash,
                "outputs": paths.writable_targets(),
                "filesystem_device": root.stat().st_dev,
            },
            bool(native_identity.get("module_sha256"))
            and native_hash == native_identity.get("module_sha256")
            and all(paths.writable_targets()),
        ),
    ]
    source_artifact_hashes = {
        "files": hashes,
        "artifacts": {
            "exp7298-snapshot-journal": {
                "path": str(upstream_path.resolve()),
                "sha256": upstream_hash,
                "expected_sha256": EXPECTED_EXP7298_SHA256,
                "status": upstream.get("status"),
                "snapshot_journal_ready_score": upstream.get("snapshot_journal_ready_score"),
                "honest_verdict": upstream.get("honest_verdict"),
                "verdict_class": upstream.get("verdict_class"),
                "quarantined": quarantine.get("quarantined"),
                "retired": quarantine.get("retired", False),
            }
        },
        "authority_boundaries": {
            "native_transition": "installed Exp7256 PyO3 extension",
            "storage_arms": list(STORAGE_ARMS),
            "recovery": "direct state-byte reader independent of measured writer object",
            "reducer": "fresh read-only process over immutable raw rows",
        },
        "resource_ownership": {
            "filesystem_device": root.stat().st_dev,
            "output_paths": "Exp7299 task-owned and disjoint by arm and trial",
        },
    }
    progress(
        0,
        "end",
        f"precondition_checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
        started,
    )
    return checks, {
        "hashes": hashes,
        "source_artifact_hashes": source_artifact_hashes,
        "exp7298": upstream,
        "native_module_path": str(native_path),
        "native_module_sha256": native_hash,
        "storage_identity": observed_storage,
        "quarantine": quarantine,
    }


def _initial_state_bytes() -> bytes:
    """Serialize the one fixed native starting state before arm assignment."""

    return transactional.canonical_json_bytes(exp7257.seed_cost_state(4))


def _event_ids(seed: int, arrival: str, count: int) -> list[str]:
    """Freeze an event identity roster that matched writers share."""

    return [f"exp7299-{seed}-{arrival}-{index:03d}" for index in range(count)]


def _arrival_offsets(arrival: str, count: int) -> list[int]:
    """Freeze release offsets without adding sleeps to burst or interactive work."""

    if arrival == "steady_paced":
        return [index * STEADY_INTERARRIVAL_NS for index in range(count)]
    return [0] * count


def freeze_trial_plan(
    *, seeds: Sequence[int] = EVALUATION_SEEDS, events_per_trial: int = EVENTS_PER_TRIAL
) -> list[JsonDict]:
    """Freeze pairing, storage order, event bytes, and schedules before measurement."""

    initial_hash = _sha256_bytes(_initial_state_bytes())
    plan: list[JsonDict] = []
    for seed in seeds:
        for arrival_index, arrival in enumerate(ARRIVAL_PROCESSES):
            event_ids = _event_ids(seed, arrival, events_per_trial)
            offsets = _arrival_offsets(arrival, events_per_trial)
            schedule_hash = _sha256_bytes(canonical_json(offsets).encode())
            for group_size in GROUP_SIZES:
                order = list(STORAGE_ARMS)
                random.Random(seed * 100 + arrival_index * 10 + group_size).shuffle(order)
                for arm_order, storage_arm in enumerate(order):
                    plan.append(
                        {
                            "unit_id": f"{seed}:{arrival}:group_{group_size}:{storage_arm}",
                            "seed": seed,
                            "arrival_process": arrival,
                            "storage_arm": storage_arm,
                            "max_group_size": group_size,
                            "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                            "arm_order": arm_order,
                            "event_count": events_per_trial,
                            "event_ids": event_ids,
                            "arrival_offsets_ns": offsets,
                            "arrival_schedule_sha256": schedule_hash,
                            "initial_state_sha256": initial_hash,
                        }
                    )
    return plan


def _trial_releases(seed: int, arrival: str, count: int) -> list[JsonDict]:
    """Create one canonical event stream shared byte-for-byte by both writers."""

    releases: list[JsonDict] = []
    for index, event_id in enumerate(_event_ids(seed, arrival, count)):
        release = exp7257._cost_release(seed + index, 0)
        release["event_id"] = event_id
        release["request_index"] = 300_000 + index * 2
        release["release_index"] = 300_001 + index * 2
        releases.append(release)
    return releases


class _MeasuredSQLiteJournal(exp7298.SQLiteSnapshotJournal):
    """Add exclusive timing to the shipped Exp7298 SQLite storage contract."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._submission_costs: dict[str, JsonDict] = {}

    def enqueue(self, release: Mapping[str, Any], *, now_ns: int | None = None) -> JsonDict:
        """Measure validation and serialization without changing queue decisions."""

        if self._failed_restore_required:
            raise RuntimeError("fresh restore required after uncertain publication")
        submitted_ns = self._clock_ns()
        event_id = str(release.get("event_id", ""))
        if not event_id:
            return {"accepted": False, "acknowledged": False, "disposition": "invalid_release"}
        if event_id in self._committed_ids or any(
            str(row.release["event_id"]) == event_id for row in self._pending
        ):
            return {"accepted": False, "acknowledged": False, "disposition": "duplicate_id"}
        validation_started = self._clock_ns()
        try:
            normalized = exp7256.exp7226.PackedBeliefController._validate_release(
                release, int(release.get("release_index", -1))
            )
        except (KeyError, TypeError, ValueError, exp7256.exp7226.CommitRejected):
            return {"accepted": False, "acknowledged": False, "disposition": "invalid_release"}
        validation_ns = max(self._clock_ns() - validation_started, 0)
        serialization_started = self._clock_ns()
        encoded = transactional.canonical_json_bytes(normalized)
        event_serialization_ns = max(self._clock_ns() - serialization_started, 0)
        if len(self._pending) >= self.max_pending_events:
            return {"accepted": False, "acknowledged": False, "disposition": "backpressure_events"}
        if self._pending_bytes + len(encoded) > self.max_pending_bytes:
            return {"accepted": False, "acknowledged": False, "disposition": "backpressure_bytes"}
        accepted_ns = self._clock_ns() if now_ns is None else now_ns
        self._pending.append(exp7284._PendingEvent(dict(normalized), accepted_ns, len(encoded)))
        self._pending_bytes += len(encoded)
        self._submission_costs[event_id] = {
            "submitted_ns": submitted_ns,
            "validation_ns": validation_ns,
            "event_serialization_ns": event_serialization_ns,
        }
        response: JsonDict = {
            "accepted": True,
            "acknowledged": False,
            "disposition": "accepted_pending",
            "event_id": event_id,
            "pending_events": len(self._pending),
            "pending_bytes": self._pending_bytes,
        }
        if len(self._pending) >= self.max_group_size:
            receipt = self.flush(reason="group_size")
            response["flush_receipt"] = receipt
            response["acknowledged"] = event_id in receipt["acknowledged_event_ids"]
            response["disposition"] = "accepted_and_acknowledged"
        return response

    def _commit_snapshot(self, state_bytes: bytes, sequence: int) -> JsonDict:
        """Measure SQL staging and the commit call as separate host costs."""

        if len(state_bytes) > exp7298.MAX_STATE_BYTES:
            raise ValueError("snapshot exceeds the fixed state-byte bound")
        if sequence != self._sequence + 1:
            raise exp7298.SnapshotSequenceError("snapshot sequence must increase by exactly one")
        started = self._clock_ns()
        self._call_stage("before_transaction")
        write_started = self._clock_ns()
        self._connection.execute("BEGIN IMMEDIATE")
        current = self._connection.execute(
            "SELECT sequence FROM snapshot WHERE slot = 1"
        ).fetchone()
        if current is None or int(current[0]) != self._sequence:
            self._connection.rollback()
            raise exp7298.SnapshotSequenceError("durable snapshot parent sequence changed")
        checksum = exp7298.snapshot_checksum(state_bytes)
        self._connection.execute(
            "UPDATE snapshot SET schema_version = ?, sequence = ?, state = ?, checksum = ? "
            "WHERE slot = 1 AND sequence = ?",
            (exp7298.STATE_SCHEMA, sequence, state_bytes, checksum, self._sequence),
        )
        self._call_stage("after_row_write_before_commit")
        pending = exp7298._read_snapshot(self._connection)
        if pending[1] != sequence or pending[2] != state_bytes:
            self._connection.rollback()
            raise exp7298.SnapshotCorruptionError("transactional row read-back mismatch")
        database_write_ns = max(self._clock_ns() - write_started, 0)
        self._call_stage("before_commit_call")
        sync_started = self._clock_ns()
        self._connection.commit()
        sync_ns = max(self._clock_ns() - sync_started, 0)
        self._call_stage("after_commit_before_acknowledgment")
        verify_started = self._clock_ns()
        stored = exp7298._read_snapshot(self._connection)
        if stored[1] != sequence or stored[2] != state_bytes:
            raise exp7298.SnapshotCorruptionError("committed row read-back mismatch")
        post_commit_verify_ns = max(self._clock_ns() - verify_started, 0)
        return {
            "sequence": sequence,
            "schema_version": exp7298.STATE_SCHEMA,
            "snapshot_checksum": checksum,
            "snapshot_bytes": len(state_bytes),
            "transaction_duration_ns": max(self._clock_ns() - started, 0),
            "database_write_ns": database_write_ns,
            "sync_ns": sync_ns,
            "post_commit_verify_ns": post_commit_verify_ns,
            "linearization_point": "sqlite_commit_returned",
            "journal_mode": str(
                self._connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower(),
            "synchronous": int(self._connection.execute("PRAGMA synchronous").fetchone()[0]),
            "full_snapshot": True,
            "delta_log": False,
        }

    def flush(self, *, reason: str) -> JsonDict:
        """Measure native, snapshot, transaction, and acknowledgment phases."""

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
        native_started = self._clock_ns()
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
        native_ns = max(self._clock_ns() - native_started, 0)
        serialization_started = self._clock_ns()
        state_bytes = self._controller.state_bytes()
        state_serialization_ns = max(self._clock_ns() - serialization_started, 0)
        try:
            durable = self._commit_snapshot(state_bytes, self._sequence + 1)
        except (
            OSError,
            sqlite3.Error,
            exp7298.SnapshotCorruptionError,
            exp7298.SnapshotSequenceError,
            ValueError,
        ):
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
        self._sequence = int(durable["sequence"])
        self._pending.clear()
        self._pending_bytes = 0
        self._committed_ids.update(event_ids)
        self._group_number += 1
        acknowledged_ns = self._clock_ns()
        event_receipts: list[JsonDict] = []
        for row in pending:
            event_id = str(row.release["event_id"])
            submission = self._submission_costs.pop(event_id)
            total_ns = max(acknowledged_ns - int(submission["submitted_ns"]), 0)
            known_ns = sum(
                (
                    int(submission["validation_ns"]),
                    int(submission["event_serialization_ns"]),
                    max(commit_started - row.accepted_ns, 0),
                    native_ns,
                    state_serialization_ns,
                    int(durable["database_write_ns"]),
                    int(durable["sync_ns"]),
                    int(durable["post_commit_verify_ns"]),
                )
            )
            event_receipts.append(
                {
                    "event_id": event_id,
                    "submitted_ns": submission["submitted_ns"],
                    "acknowledged_ns": acknowledged_ns,
                    "acknowledgment_latency_ns": total_ns,
                    "visibility_latency_ns": total_ns,
                    "queue_wait_ns": max(commit_started - row.accepted_ns, 0),
                    "validation_ns": submission["validation_ns"],
                    "serialization_ns": int(submission["event_serialization_ns"])
                    + state_serialization_ns,
                    "transition_ns": native_ns,
                    "file_write_ns": int(durable["database_write_ns"])
                    + int(durable["post_commit_verify_ns"]),
                    "file_sync_ns": durable["sync_ns"],
                    "rename_ns": 0,
                    "directory_sync_ns": 0,
                    "protocol_overhead_ns": max(total_ns - known_ns, 0),
                    "byte_count": row.byte_count,
                }
            )
        receipt = {
            "disposition": "committed_acknowledged",
            "reason": reason,
            "group_id": f"group-{self._group_number:04d}",
            "issued_event_ids": event_ids,
            "acknowledged_event_ids": event_ids,
            "commit_delay_ns": max(acknowledged_ns - commit_started, 0),
            "event_receipts": event_receipts,
            "native_transition_ns": native_ns,
            "state_serialization_ns": state_serialization_ns,
            **durable,
        }
        self._call_stage("after_acknowledgment")
        return receipt


def _pending(controller: Any, storage_arm: str) -> bool:
    """Read the pending queue count through each shipped controller surface."""

    return bool(
        controller.pending if storage_arm == "atomic_replace" else controller.pending_events
    )


def _extract_flush(response: JsonDict | None, storage_arm: str) -> JsonDict | None:
    """Normalize the two existing enqueue response shapes without changing them."""

    if storage_arm == "atomic_replace":
        return response
    return None if response is None else response.get("flush_receipt")


def _independent_restore(
    binding: ModuleType, storage_arm: str, path: Path
) -> tuple[exp7256.PersistentNativeArchiveController, int]:
    """Read measured durable bytes without calling either measured writer object."""

    started = time.monotonic_ns()
    if storage_arm == "atomic_replace":
        restored = exp7256.PersistentNativeArchiveController.load(binding, path)
    else:
        connection = exp7298._connect(path)
        try:
            _schema, _sequence, state_bytes, _checksum = exp7298._read_snapshot(connection)
        finally:
            connection.close()
        restored = exp7256.PersistentNativeArchiveController.from_snapshot(
            binding, state_bytes.decode("utf-8")
        )
    return restored, max(time.monotonic_ns() - started, 1)


def _common(plan_row: Mapping[str, Any]) -> JsonDict:
    """Return the identity fields shared by all row types for one trial."""

    return {
        "unit_id": plan_row["unit_id"],
        "seed": plan_row["seed"],
        "arrival_process": plan_row["arrival_process"],
        "storage_arm": plan_row["storage_arm"],
        "max_group_size": plan_row["max_group_size"],
        "max_wait_ms": plan_row["max_wait_ms"],
        "arm_order": plan_row["arm_order"],
        "initial_state_sha256": plan_row["initial_state_sha256"],
        "arrival_schedule_sha256": plan_row["arrival_schedule_sha256"],
    }


def _censored_trial(
    plan_row: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Keep each declared event and paired unit after the benchmark deadline."""

    common = _common(plan_row)
    rows = [
        _finish_row(
            {
                **common,
                "event_id": event_id,
                "event_index": index,
                "metric": None,
                "error": None,
                "abstention": False,
                "censored": True,
                "censoring_reason": "benchmark_budget_exhausted",
                "acknowledged": False,
                "missing_after_restart": False,
                "queue_wait_ns": None,
                "native_transition_ns": None,
                "serialization_ns": None,
                "write_ns": None,
                "sync_ns": None,
                "acknowledgment_ns": None,
                "retry_ns": None,
                "acknowledgment_latency_ns": None,
            }
        )
        for index, event_id in enumerate(plan_row["event_ids"])
    ]
    run = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "censoring_reason": "benchmark_budget_exhausted",
            "event_count": len(rows),
            "completed_event_count": 0,
            "acknowledgment_p95_ns": None,
            "throughput_events_per_s": None,
            "cold_throughput_events_per_s": None,
            "elapsed_ns": None,
            "initialization_ns": None,
            "recovery_ns": None,
            "durable_group_count": 0,
            "missing_acknowledged_event_count": 0,
            "exact_native_state_parity": None,
            "event_stream_sha256": None,
            "final_state_sha256": None,
            "original_queue_limits": True,
            "syncs_disabled": False,
            "event_deadlines_changed": False,
            "native_transition_calls": 0,
            "max_pending_events": MAX_PENDING_EVENTS,
            "max_pending_bytes": MAX_PENDING_BYTES,
        }
    )
    cost = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "initialization_ns": None,
            "queue_wait_ns": None,
            "native_transition_ns": None,
            "serialization_ns": None,
            "write_ns": None,
            "sync_ns": None,
            "acknowledgment_ns": None,
            "retry_ns": None,
            "recovery_ns": None,
            "steady_total_ns": None,
            "cold_total_ns": None,
            "total_with_recovery_ns": None,
            "phase_sum_matches": False,
        }
    )
    recovery = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "recovery_ns": None,
            "acknowledged_sequence_match": None,
            "exact_state_bytes_match": None,
            "state_hash_match": None,
            "next_native_decision_match": None,
            "acknowledged_event_count": 0,
            "recovered_event_count": 0,
            "process_restart": False,
        }
    )
    return rows, run, cost, recovery


def _run_trial(
    binding: ModuleType,
    storage_dir: Path,
    plan_row: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Run one real paired unit and independently reopen its measured bytes."""

    common = _common(plan_row)
    storage_arm = str(plan_row["storage_arm"])
    group_size = int(plan_row["max_group_size"])
    unit_dir = storage_dir / str(plan_row["unit_id"]).replace(":", "-")
    path = unit_dir / ("state.sqlite3" if storage_arm == "sqlite_persist" else "state.json")
    state = exp7257.seed_cost_state(4)
    initialization_started = time.monotonic_ns()
    if storage_arm == "atomic_replace":
        controller: Any = exp7285._MeasuredGroupController(binding, state, path, group_size)
    else:
        controller = _MeasuredSQLiteJournal.from_state(
            binding,
            state,
            path,
            max_group_size=group_size,
            max_wait_ms=0 if group_size == 1 else MAX_WAIT_MS,
        )
    initialization_ns = max(time.monotonic_ns() - initialization_started, 1)
    groups: list[JsonDict] = []
    trial_started = time.monotonic_ns()
    offsets = list(map(int, plan_row["arrival_offsets_ns"]))
    query = {"event_id": "dependent-query", "family_id": "lower_bound", "numeric_value": 0}
    for index, release in enumerate(releases):
        if plan_row["arrival_process"] == "steady_paced":
            remaining = trial_started + offsets[index] - time.monotonic_ns()
            if remaining > 0:
                time.sleep(remaining / 1_000_000_000)
            due = controller.flush_due()
            if due is not None:
                groups.append(due)
        response = controller.enqueue(release)
        receipt = _extract_flush(response, storage_arm)
        if receipt is not None:
            groups.append(receipt)
        if plan_row["arrival_process"] == "interactive_dependent" and _pending(
            controller, storage_arm
        ):
            if storage_arm == "atomic_replace":
                groups.append(controller.dependent_query(query))
            else:
                groups.append(controller.query(query, dependent=True)["flush_receipt"])
    if _pending(controller, storage_arm):
        groups.append(
            controller.flush("end_of_stream")
            if storage_arm == "atomic_replace"
            else controller.flush(reason="end_of_stream")
        )
    elapsed_ns = max(time.monotonic_ns() - trial_started, 1)
    state_bytes = (
        controller.controller.state_bytes()
        if storage_arm == "atomic_replace"
        else controller.state_bytes
    )
    state_hash = (
        controller.controller.state_hash()
        if storage_arm == "atomic_replace"
        else controller.state_hash
    )
    if storage_arm == "sqlite_persist":
        controller.close()
    restored, recovery_ns = _independent_restore(binding, storage_arm, path)
    recovered_ids = list(map(str, restored.state_dict().get("release_ids", [])))
    expected = exp7256.PersistentNativeArchiveController.from_snapshot(
        binding, bytes(reference["state_bytes"]).decode("utf-8")
    )
    restored_decision = restored.predict(query)
    expected_decision = expected.predict(query)
    by_event = {
        str(event["event_id"]): (str(group["group_id"]), event)
        for group in groups
        for event in group["event_receipts"]
    }
    rows: list[JsonDict] = []
    for event_index, release in enumerate(releases):
        event_id = str(release["event_id"])
        group_id, measured = by_event[event_id]
        write_ns = int(measured["file_write_ns"]) + int(measured["rename_ns"])
        sync_ns = int(measured["file_sync_ns"]) + int(measured["directory_sync_ns"])
        acknowledgment_ns = int(measured["protocol_overhead_ns"])
        rows.append(
            _finish_row(
                {
                    **common,
                    "event_id": event_id,
                    "event_index": event_index,
                    "group_id": group_id,
                    "metric": measured["acknowledgment_latency_ns"],
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "censoring_reason": None,
                    "acknowledged": True,
                    "missing_after_restart": event_id not in recovered_ids,
                    "queue_wait_ns": measured["queue_wait_ns"],
                    "native_transition_ns": measured["transition_ns"],
                    "serialization_ns": measured["serialization_ns"],
                    "write_ns": write_ns,
                    "sync_ns": sync_ns,
                    "acknowledgment_ns": acknowledgment_ns,
                    "retry_ns": 0,
                    "acknowledgment_latency_ns": measured["acknowledgment_latency_ns"],
                }
            )
        )
    latency_values = [int(row["acknowledgment_latency_ns"]) for row in rows]
    event_stream_hash = _sha256_bytes(transactional.canonical_json_bytes(list(releases)))
    exact_state = state_bytes == reference["state_bytes"]
    run = _finish_row(
        {
            **common,
            "metric": exp7284._quantile(latency_values, 0.95),
            "error": None,
            "abstention": False,
            "censored": False,
            "censoring_reason": None,
            "event_count": len(rows),
            "completed_event_count": len(rows),
            "acknowledgment_p95_ns": exp7284._quantile(latency_values, 0.95),
            "throughput_events_per_s": len(rows) * 1_000_000_000 / elapsed_ns,
            "cold_throughput_events_per_s": len(rows)
            * 1_000_000_000
            / (elapsed_ns + initialization_ns),
            "elapsed_ns": elapsed_ns,
            "initialization_ns": initialization_ns,
            "recovery_ns": recovery_ns,
            "durable_group_count": len(groups),
            "missing_acknowledged_event_count": sum(
                bool(row["missing_after_restart"]) for row in rows
            ),
            "exact_native_state_parity": exact_state,
            "event_stream_sha256": event_stream_hash,
            "final_state_sha256": _sha256_bytes(state_bytes),
            "original_queue_limits": (
                controller.max_pending_events == MAX_PENDING_EVENTS
                and controller.max_pending_bytes == MAX_PENDING_BYTES
                if storage_arm == "sqlite_persist"
                else True
            ),
            "syncs_disabled": False,
            "event_deadlines_changed": False,
            "native_transition_calls": len(rows),
            "max_pending_events": MAX_PENDING_EVENTS,
            "max_pending_bytes": MAX_PENDING_BYTES,
        }
    )
    phase_names = (
        "queue_wait_ns",
        "native_transition_ns",
        "serialization_ns",
        "write_ns",
        "sync_ns",
        "acknowledgment_ns",
        "retry_ns",
    )
    totals = {name: sum(int(row[name]) for row in rows) for name in phase_names}
    steady_total_ns = sum(totals.values())
    cold_total_ns = initialization_ns + steady_total_ns
    total_with_recovery_ns = cold_total_ns + recovery_ns
    cost = _finish_row(
        {
            **common,
            "metric": total_with_recovery_ns,
            "error": None,
            "abstention": False,
            "censored": False,
            "initialization_ns": initialization_ns,
            **totals,
            "recovery_ns": recovery_ns,
            "steady_total_ns": steady_total_ns,
            "cold_total_ns": cold_total_ns,
            "total_with_recovery_ns": total_with_recovery_ns,
            "phase_sum_matches": total_with_recovery_ns
            == initialization_ns + steady_total_ns + recovery_ns,
        }
    )
    recovery = _finish_row(
        {
            **common,
            "metric": int(exact_state),
            "error": None,
            "abstention": False,
            "censored": False,
            "recovery_ns": recovery_ns,
            "acknowledged_sequence_match": recovered_ids == reference["release_ids"],
            "exact_state_bytes_match": restored.state_bytes() == reference["state_bytes"],
            "state_hash_match": restored.state_hash() == state_hash == reference["state_hash"],
            "next_native_decision_match": restored_decision == expected_decision,
            "acknowledged_event_count": len(rows),
            "recovered_event_count": len(recovered_ids),
            "process_restart": True,
            "measured_artifact_path": str(path.resolve()),
            "measured_artifact_sha256": sha256_file(path),
        }
    )
    return rows, run, cost, recovery


def run_snapshot_benchmark(
    binding: ModuleType,
    storage_dir: Path,
    *,
    seeds: Sequence[int] = EVALUATION_SEEDS,
    events_per_trial: int = EVENTS_PER_TRIAL,
    max_duration_s: float = MAX_BENCHMARK_S,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Run the fixed paired storage frontier and retain every censored event."""

    plan = freeze_trial_plan(seeds=seeds, events_per_trial=events_per_trial)
    rows: list[JsonDict] = []
    runs: list[JsonDict] = []
    costs: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    benchmark_started = time.monotonic()
    deadline = benchmark_started + max(max_duration_s, 0.0)
    progress(
        3, "before", f"snapshot benchmark trials={len(plan)} events={len(plan) * events_per_trial}"
    )
    references: dict[tuple[int, str], JsonDict] = {}
    for index, plan_row in enumerate(plan):
        if time.monotonic() >= deadline:
            event_rows, run, cost, recovery = _censored_trial(plan_row)
        else:
            key = (int(plan_row["seed"]), str(plan_row["arrival_process"]))
            releases = _trial_releases(key[0], key[1], events_per_trial)
            if key not in references:
                references[key] = exp7284.serial_reference(
                    binding, exp7257.seed_cost_state(4), releases
                )
            event_rows, run, cost, recovery = _run_trial(
                binding, storage_dir, plan_row, releases, references[key]
            )
        rows.extend(event_rows)
        runs.append(run)
        costs.append(cost)
        recovery_rows.append(recovery)
        progress(
            3,
            "unit",
            f"completed_trials={index + 1}/{len(plan)} event_rows={len(rows)}",
            benchmark_started,
        )
    progress(3, "after", f"snapshot benchmark event_rows={len(rows)}", benchmark_started)
    return rows, runs, costs, recovery_rows


def _rows_valid(groups: Sequence[Sequence[Mapping[str, Any]]]) -> bool:
    """Recompute each saved row hash before independent reduction."""

    return all(group and all(exp7284._row_hash_valid(row) for row in group) for group in groups)


def _bootstrap(values: Sequence[float], seed: int) -> list[float | None]:
    """Return the preregistered 10,000-draw seed-cluster interval."""

    if not values:
        return [None, None]
    return exp7257._bootstrap_interval(values, seed, BOOTSTRAP_DRAWS)


def reduce_saved_rows(
    rows: Sequence[Mapping[str, Any]],
    per_run_results: Sequence[Mapping[str, Any]],
    phase_cost_rows: Sequence[Mapping[str, Any]],
    independent_recovery_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Cold-reduce fixed group-16 deployment gates from saved row evidence."""

    groups = (rows, per_run_results, phase_cost_rows, independent_recovery_rows)
    if not _rows_valid(groups):
        raise ValueError("invalid snapshot-cost rows")
    seeds = sorted({int(row["seed"]) for row in per_run_results})
    run_index = {
        (
            int(row["seed"]),
            str(row["arrival_process"]),
            int(row["max_group_size"]),
            str(row["storage_arm"]),
        ): row
        for row in per_run_results
    }

    def ratios(field: str, arrival: str, group_size: int) -> list[float]:
        result: list[float] = []
        for seed in seeds:
            atomic = run_index.get((seed, arrival, group_size, "atomic_replace"))
            sqlite = run_index.get((seed, arrival, group_size, "sqlite_persist"))
            if (
                atomic is not None
                and sqlite is not None
                and atomic.get("censored") is False
                and sqlite.get("censored") is False
                and float(atomic.get(field, 0)) > 0
            ):
                result.append(float(sqlite[field]) / float(atomic[field]))
        return result

    burst_ratios = ratios("throughput_events_per_s", "burst", 16)
    cold_burst_ratios = ratios("cold_throughput_events_per_s", "burst", 16)
    latency_index: dict[tuple[int, str], list[float]] = {}
    steady_latencies: list[float] = []
    for row in rows:
        if row.get("censored") is not False or int(row["max_group_size"]) != 16:
            continue
        key = (int(row["seed"]), str(row["storage_arm"]))
        if row["arrival_process"] == "interactive_dependent":
            latency_index.setdefault(key, []).append(float(row["acknowledgment_latency_ns"]))
        if row["arrival_process"] == "steady_paced" and row["storage_arm"] == "sqlite_persist":
            steady_latencies.append(float(row["acknowledgment_latency_ns"]))
    interactive_ratios: list[float] = []
    for seed in seeds:
        atomic_values = latency_index.get((seed, "atomic_replace"), [])
        sqlite_values = latency_index.get((seed, "sqlite_persist"), [])
        if atomic_values and sqlite_values:
            atomic_p95 = exp7284._quantile(atomic_values, 0.95)
            sqlite_p95 = exp7284._quantile(sqlite_values, 0.95)
            if atomic_p95 > 0:
                interactive_ratios.append(sqlite_p95 / atomic_p95)
    burst_ci = _bootstrap(burst_ratios, RANDOM_SEED + 16)
    cold_burst_ci = _bootstrap(cold_burst_ratios, RANDOM_SEED + 116)
    interactive_ci = _bootstrap(interactive_ratios, RANDOM_SEED + 216)
    steady_p95 = exp7284._quantile(steady_latencies, 0.95) if steady_latencies else None
    expected_trials = len(EVALUATION_SEEDS) * len(ARRIVAL_PROCESSES) * len(GROUP_SIZES) * 2
    expected_events = sum(int(row.get("event_count", 0)) for row in per_run_results)
    pairing_complete = True
    for seed in seeds:
        for arrival in ARRIVAL_PROCESSES:
            for group_size in GROUP_SIZES:
                pair = [run_index.get((seed, arrival, group_size, arm)) for arm in STORAGE_ARMS]
                pairing_complete = pairing_complete and all(item is not None for item in pair)
                if all(item is not None for item in pair):
                    pairing_complete = (
                        pairing_complete
                        and len(
                            {
                                (
                                    item.get("initial_state_sha256"),
                                    item.get("arrival_schedule_sha256"),
                                    item.get("event_stream_sha256"),
                                )
                                for item in pair
                            }
                        )
                        == 1
                    )
    population_complete = (
        len(seeds) == len(EVALUATION_SEEDS)
        and len(per_run_results) == expected_trials
        and len(rows) == expected_events
        and all(row.get("censored") is False for group in groups for row in group)
    )
    phase_complete = len(phase_cost_rows) == expected_trials and all(
        row.get("phase_sum_matches") is True for row in phase_cost_rows
    )
    recovery_complete = len(independent_recovery_rows) == expected_trials and all(
        row.get("process_restart") is True
        and row.get("acknowledged_sequence_match") is True
        and row.get("exact_state_bytes_match") is True
        and row.get("state_hash_match") is True
        and row.get("next_native_decision_match") is True
        for row in independent_recovery_rows
    )
    zero_missing = all(
        int(row.get("missing_acknowledged_event_count", 0)) == 0 for row in per_run_results
    ) and all(row.get("missing_after_restart") is False for row in rows)
    exact_parity = all(row.get("exact_native_state_parity") is True for row in per_run_results)
    queue_contract = all(
        row.get("original_queue_limits") is True
        and row.get("syncs_disabled") is False
        and row.get("event_deadlines_changed") is False
        for row in per_run_results
    )
    capture_complete = all(
        (
            population_complete,
            pairing_complete,
            phase_complete,
            recovery_complete,
            zero_missing,
            exact_parity,
            queue_contract,
            len(burst_ratios) == len(EVALUATION_SEEDS),
            len(cold_burst_ratios) == len(EVALUATION_SEEDS),
            len(interactive_ratios) == len(EVALUATION_SEEDS),
        )
    )
    value_passed = (
        capture_complete
        and burst_ci[0] is not None
        and float(burst_ci[0]) >= 1.5
        and cold_burst_ci[0] is not None
        and float(cold_burst_ci[0]) >= 1.5
        and steady_p95 is not None
        and steady_p95 <= 50_000_000
        and interactive_ci[1] is not None
        and float(interactive_ci[1]) <= 1.05
    )
    sensitivity: list[JsonDict] = []
    for group_size in (1, 4):
        values = ratios("throughput_events_per_s", "burst", group_size)
        sensitivity.append(
            {
                "max_group_size": group_size,
                "paired_seed_count": len(values),
                "mean_burst_throughput_ratio": statistics.mean(values) if values else None,
                "burst_throughput_ratio_ci95": _bootstrap(values, RANDOM_SEED + group_size),
                "selected": False,
            }
        )
    return {
        "inference_substrate": REDUCER_SUBSTRATE,
        "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "bootstrap_unit": "seed_cluster",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "deployment_candidate": DEPLOYMENT_CANDIDATE,
        "event_row_count": len(rows),
        "trial_row_count": len(per_run_results),
        "phase_cost_row_count": len(phase_cost_rows),
        "recovery_row_count": len(independent_recovery_rows),
        "population_complete": population_complete,
        "pairing_complete": pairing_complete,
        "phase_cost_complete": phase_complete,
        "recovery_complete": recovery_complete,
        "zero_missing_acknowledged_events": zero_missing,
        "exact_native_state_parity": exact_parity,
        "original_queue_contract": queue_contract,
        "capture_complete": capture_complete,
        "burst_paired_seed_count": len(burst_ratios),
        "burst_throughput_ratio_mean": statistics.mean(burst_ratios) if burst_ratios else None,
        "burst_throughput_ratio_ci95": burst_ci,
        "cold_burst_throughput_ratio_mean": statistics.mean(cold_burst_ratios)
        if cold_burst_ratios
        else None,
        "cold_burst_throughput_ratio_ci95": cold_burst_ci,
        "steady_acknowledgment_p95_ns": steady_p95,
        "interactive_latency_ratio_mean": statistics.mean(interactive_ratios)
        if interactive_ratios
        else None,
        "interactive_latency_ratio_ci95": interactive_ci,
        "sensitivity_results": sensitivity,
        "snapshot_value_gate_passed": value_passed,
        "nfr01_full_boundary_passed": cold_burst_ci[0] is not None
        and float(cold_burst_ci[0]) >= 10.0,
    }


def synthetic_snapshot_rows(
    *,
    burst_ratio: float,
    cold_burst_ratio: float,
    steady_p95_ns: int,
    interactive_ratio: float,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Create a compact full-roster fixture for reducer and validator tests."""

    rows: list[JsonDict] = []
    runs: list[JsonDict] = []
    costs: list[JsonDict] = []
    recovery: list[JsonDict] = []
    for plan_row in freeze_trial_plan(events_per_trial=1):
        common = _common(plan_row)
        storage_arm = str(plan_row["storage_arm"])
        arrival = str(plan_row["arrival_process"])
        group_size = int(plan_row["max_group_size"])
        atomic_throughput = 100.0
        throughput = atomic_throughput
        cold_throughput = atomic_throughput
        if storage_arm == "sqlite_persist" and arrival == "burst" and group_size == 16:
            throughput *= burst_ratio
            cold_throughput *= cold_burst_ratio
        acknowledgment = 1_000_000
        if storage_arm == "sqlite_persist" and arrival == "steady_paced" and group_size == 16:
            acknowledgment = steady_p95_ns
        if (
            storage_arm == "sqlite_persist"
            and arrival == "interactive_dependent"
            and group_size == 16
        ):
            acknowledgment = round(1_000_000 * interactive_ratio)
        event_stream_hash = _sha256_bytes(
            transactional.canonical_json_bytes(_trial_releases(int(plan_row["seed"]), arrival, 1))
        )
        rows.append(
            _finish_row(
                {
                    **common,
                    "event_id": plan_row["event_ids"][0],
                    "event_index": 0,
                    "group_id": "group-0001",
                    "metric": acknowledgment,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "censoring_reason": None,
                    "acknowledged": True,
                    "missing_after_restart": False,
                    "queue_wait_ns": 100,
                    "native_transition_ns": 100,
                    "serialization_ns": 100,
                    "write_ns": 100,
                    "sync_ns": 100,
                    "acknowledgment_ns": acknowledgment - 500,
                    "retry_ns": 0,
                    "acknowledgment_latency_ns": acknowledgment,
                }
            )
        )
        runs.append(
            _finish_row(
                {
                    **common,
                    "metric": acknowledgment,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "censoring_reason": None,
                    "event_count": 1,
                    "completed_event_count": 1,
                    "acknowledgment_p95_ns": acknowledgment,
                    "throughput_events_per_s": throughput,
                    "cold_throughput_events_per_s": cold_throughput,
                    "elapsed_ns": 10_000_000,
                    "initialization_ns": 1_000_000,
                    "recovery_ns": 1_000_000,
                    "durable_group_count": 1,
                    "missing_acknowledged_event_count": 0,
                    "exact_native_state_parity": True,
                    "event_stream_sha256": event_stream_hash,
                    "final_state_sha256": "sha256:fixture",
                    "original_queue_limits": True,
                    "syncs_disabled": False,
                    "event_deadlines_changed": False,
                    "native_transition_calls": 1,
                    "max_pending_events": MAX_PENDING_EVENTS,
                    "max_pending_bytes": MAX_PENDING_BYTES,
                }
            )
        )
        steady = acknowledgment
        costs.append(
            _finish_row(
                {
                    **common,
                    "metric": steady + 2_000_000,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "initialization_ns": 1_000_000,
                    "queue_wait_ns": 100,
                    "native_transition_ns": 100,
                    "serialization_ns": 100,
                    "write_ns": 100,
                    "sync_ns": 100,
                    "acknowledgment_ns": acknowledgment - 500,
                    "retry_ns": 0,
                    "recovery_ns": 1_000_000,
                    "steady_total_ns": steady,
                    "cold_total_ns": steady + 1_000_000,
                    "total_with_recovery_ns": steady + 2_000_000,
                    "phase_sum_matches": True,
                }
            )
        )
        recovery.append(
            _finish_row(
                {
                    **common,
                    "metric": 1,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "recovery_ns": 1_000_000,
                    "acknowledged_sequence_match": True,
                    "exact_state_bytes_match": True,
                    "state_hash_match": True,
                    "next_native_decision_match": True,
                    "acknowledged_event_count": 1,
                    "recovered_event_count": 1,
                    "process_restart": True,
                    "measured_artifact_path": "/tmp/fixture",
                    "measured_artifact_sha256": "sha256:fixture",
                }
            )
        )
    return rows, runs, costs, recovery


def _sample_budget(
    rows: Sequence[Mapping[str, Any]], per_run_results: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Record the frozen roster and the rule that forbids outcome-driven expansion."""

    censored_events = sum(bool(row.get("censored")) for row in rows)
    censored_trials = sum(bool(row.get("censored")) for row in per_run_results)
    return {
        "planned_event_units": len(rows),
        "attempted_event_units": len(rows) - censored_events,
        "completed_event_units": len(rows) - censored_events,
        "censored_event_units": censored_events,
        "planned_trial_units": len(per_run_results),
        "attempted_trial_units": len(per_run_results) - censored_trials,
        "completed_trial_units": len(per_run_results) - censored_trials,
        "censored_trial_units": censored_trials,
        "frozen_evaluation_seeds": list(EVALUATION_SEEDS),
        "arrival_processes": list(ARRIVAL_PROCESSES),
        "storage_arms": list(STORAGE_ARMS),
        "group_sizes": list(GROUP_SIZES),
        "events_per_trial": int(per_run_results[0]["event_count"])
        if per_run_results
        else EVENTS_PER_TRIAL,
        "benchmark_budget_s": MAX_BENCHMARK_S,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "stopping_rule": "Run the frozen eight-seed paired roster once; keep incomplete pairs censored and do not select or add arms after outcomes.",
    }


def _acceptance_gates(reduced: Mapping[str, Any]) -> JsonDict:
    """Keep capture, fixed deployment value, and original NFR-01 gates separate."""

    burst_ci = reduced.get("burst_throughput_ratio_ci95", [None, None])
    cold_ci = reduced.get("cold_burst_throughput_ratio_ci95", [None, None])
    interactive_ci = reduced.get("interactive_latency_ratio_ci95", [None, None])
    return {
        "snapshot_capture_complete": {
            "expected": True,
            "observed": reduced.get("capture_complete"),
            "passed": reduced.get("capture_complete") is True,
            "principle": "Complete paired workloads, timing, identities, and restart evidence are required before a value claim.",
        },
        "fixed_group16_burst_throughput": {
            "expected": {"lower_ci95": ">=1.5", "paired_seeds": 8},
            "observed": {
                "lower_ci95": burst_ci[0],
                "paired_seeds": reduced.get("burst_paired_seed_count"),
            },
            "passed": burst_ci[0] is not None
            and float(burst_ci[0]) >= 1.5
            and reduced.get("burst_paired_seed_count") == len(EVALUATION_SEEDS),
            "principle": "The preregistered SQLite group-16 arm must beat matched atomic group 16 without choosing a winner after measurement.",
        },
        "cold_group16_burst_throughput": {
            "expected": {"lower_ci95": ">=1.5", "includes_initialization": True},
            "observed": {
                "lower_ci95": cold_ci[0],
                "includes_initialization": True,
            },
            "passed": cold_ci[0] is not None and float(cold_ci[0]) >= 1.5,
            "principle": "Initialization stays in the cold boundary so a warm-only win cannot pass.",
        },
        "steady_group16_acknowledgment": {
            "expected": "p95 <= 50000000 ns",
            "observed": reduced.get("steady_acknowledgment_p95_ns"),
            "passed": reduced.get("steady_acknowledgment_p95_ns") is not None
            and float(reduced["steady_acknowledgment_p95_ns"]) <= 50_000_000,
            "principle": "Steady acknowledgments must stay inside the V640 50 ms deployment bound.",
        },
        "interactive_group16_no_regression": {
            "expected": "upper ratio CI95 <= 1.05",
            "observed": interactive_ci[1],
            "passed": interactive_ci[1] is not None and float(interactive_ci[1]) <= 1.05,
            "principle": "Dependent queries force durability, so storage cannot hide a latency regression behind grouping.",
        },
        "durability_native_and_queue_contract": {
            "expected": {
                "missing": False,
                "native_parity": True,
                "recovery": True,
                "queue_contract": True,
            },
            "observed": {
                "missing": not bool(reduced.get("zero_missing_acknowledged_events")),
                "native_parity": reduced.get("exact_native_state_parity"),
                "recovery": reduced.get("recovery_complete"),
                "queue_contract": reduced.get("original_queue_contract"),
            },
            "passed": reduced.get("zero_missing_acknowledged_events") is True
            and reduced.get("exact_native_state_parity") is True
            and reduced.get("recovery_complete") is True
            and reduced.get("original_queue_contract") is True,
            "principle": "A timing change has no value if it weakens durability, native work, event order, or queue bounds.",
        },
        "nfr01_full_boundary_10x": {
            "expected": "cold lower CI95 >= 10.0",
            "observed": cold_ci[0],
            "passed": reduced.get("nfr01_full_boundary_passed") is True,
            "principle": "The original tenfold target remains separate from the local 1.5x storage gate.",
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
    """Create every required field before terminal classification."""

    summary = gate_summary(checks)
    blocked = summary["passed"] is not True
    upstream = evidence.get("exp7298", {})
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
        "random_seed": {
            "development": RANDOM_SEED,
            "independent_evaluation": list(EVALUATION_SEEDS),
            "frozen_before_measurement": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": evidence.get(
            "source_artifact_hashes", {"files": evidence.get("hashes", {}), "artifacts": {}}
        ),
        "rows": [],
        "per_run_results": [],
        "phase_cost_rows": [],
        "independent_recovery_rows": [],
        "sample_size_budget": {
            "planned_event_units": len(EVALUATION_SEEDS)
            * len(ARRIVAL_PROCESSES)
            * len(GROUP_SIZES)
            * len(STORAGE_ARMS)
            * EVENTS_PER_TRIAL,
            "attempted_event_units": 0,
            "completed_event_units": 0,
            "censored_event_units": 0,
            "planned_trial_units": len(EVALUATION_SEEDS)
            * len(ARRIVAL_PROCESSES)
            * len(GROUP_SIZES)
            * len(STORAGE_ARMS),
            "attempted_trial_units": 0,
            "completed_trial_units": 0,
            "censored_trial_units": 0,
            "frozen_evaluation_seeds": list(EVALUATION_SEEDS),
            "stopping_rule": "External precondition failure stopped measurement before any task-owned unit.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": (
            "blocked_external_precondition: Exp7298 snapshot journal evidence is unavailable or changed"
            if blocked
            else "complete_pending_measurement"
        ),
        "verdict_class": "blocked" if blocked else "partial",
        "validation_receipts": [],
        "baseline_validation_failures": [],
        "snapshot_capture_complete_score": 0,
        "snapshot_value_score": 0,
        "independent_raw_reducer": {},
        "nfr01_assessment": {
            "target_speedup": 10.0,
            "observed_cold_lower_ci95": None,
            "passed": False,
            "local_gate_does_not_satisfy_nfr01": True,
        },
        "native_identity": {
            **dict(upstream.get("native_identity", {})),
            "authenticated_module_sha256": evidence.get("native_module_sha256"),
            "actual_native_transition_calls": 0,
        },
        "storage_identity": {
            "atomic_writer": "Exp7285 atomic replacement with file and directory fsync",
            "sqlite_writer": evidence.get("storage_identity", {}),
            "same_local_filesystem_device": evidence.get("source_artifact_hashes", {})
            .get("resource_ownership", {})
            .get("filesystem_device"),
        },
        "limitations": {
            "filesystem_cache_exposed": True,
            "physical_power_loss_proven": False,
            "firmware_fsync_honesty_proven": False,
            "host_performance_predicts_fpga": False,
            "sqlite_commit_phase_includes_internal_database_and_journal_writes": True,
        },
        "claim_boundary": {
            "execution_venue": "host",
            "production_default_changed": False,
            "durability_reduced": False,
            "native_work_reduced": False,
            "updates_reduced": False,
            "queue_contract_changed": False,
            "fpga_performance_claimed": False,
            "tenfold_carnot_acceleration_claimed": False,
        },
        "cold_reducer_receipt": {},
        "output_paths": {
            "checkpoint": str(paths.checkpoint.resolve()),
            "raw_rows": str(paths.raw_rows.resolve()),
            "raw_candidate": str(paths.raw_candidate.resolve()),
            "reduced": str(paths.reduced.resolve()),
            "storage": str(paths.storage_dir.resolve()),
            "validation": str(paths.validation_dir.resolve()),
            "terminal": str(paths.artifact.resolve()),
        },
    }


def _assemble_complete_artifact(
    base: JsonDict,
    measured: tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]],
    reduced: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Join measured evidence and classify only the frozen group-16 candidate."""

    rows, runs, costs, recovery = measured
    artifact = dict(base)
    gates = _acceptance_gates(reduced)
    capture_score = int(reduced.get("capture_complete") is True)
    value_score = int(reduced.get("snapshot_value_gate_passed") is True)
    artifact.update(
        {
            "status": "complete",
            "rows": rows,
            "per_run_results": runs,
            "phase_cost_rows": costs,
            "independent_recovery_rows": recovery,
            "sample_size_budget": _sample_budget(rows, runs),
            "acceptance_gate_results": gates,
            "snapshot_capture_complete_score": capture_score,
            "snapshot_value_score": value_score,
            "independent_raw_reducer": dict(reduced),
            "nfr01_assessment": {
                "target_speedup": 10.0,
                "observed_cold_lower_ci95": reduced.get(
                    "cold_burst_throughput_ratio_ci95", [None, None]
                )[0],
                "passed": reduced.get("nfr01_full_boundary_passed") is True,
                "local_gate_does_not_satisfy_nfr01": True,
            },
            "honest_verdict": (
                "complete_circular_positive: persistent SQLite full snapshots meet the frozen V640 storage deployment bounds"
                if value_score
                else "complete_null: persistent SQLite full snapshots do not meet every frozen V640 storage deployment bound"
            ),
            "verdict_class": "circular_positive" if value_score else "null",
            "validation_receipts": [dict(row) for row in validation_receipts],
        }
    )
    artifact["native_identity"] = {
        **dict(artifact["native_identity"]),
        "actual_native_transition_calls": sum(
            int(row.get("native_transition_calls", 0)) for row in runs
        ),
        "python_fallback_used": False,
        "compiled_execution": True,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return a schema-complete row-free terminal external block fixture."""

    paths = ExperimentPaths.under(Path("/tmp/exp7299-blocked-fixture"))
    artifact = _base_artifact(
        [failed],
        {"hashes": {}, "source_artifact_hashes": {"files": {}, "artifacts": {}}},
        paths,
        started_at="2026-09-14T00:00:00+00:00",
        completed_at="2026-09-14T00:00:00.100000+00:00",
        duration_s=0.1,
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def complete_artifact_fixture_for_test(*, value_passed: bool) -> JsonDict:
    """Return compact complete positive or null evidence for closed validation tests."""

    measured = synthetic_snapshot_rows(
        burst_ratio=2.0 if value_passed else 1.4,
        cold_burst_ratio=1.8 if value_passed else 1.2,
        steady_p95_ns=40_000_000 if value_passed else 55_000_000,
        interactive_ratio=1.0 if value_passed else 1.1,
    )
    reduced = reduce_saved_rows(*measured)
    passed = check("fixture", "fixture", "field", True, True, True)
    paths = ExperimentPaths.under(Path("/tmp/exp7299-complete-fixture"))
    upstream = exp7298.complete_artifact_fixture_for_test()
    base = _base_artifact(
        [passed],
        {
            "hashes": {"fixture": "sha256:fixture"},
            "source_artifact_hashes": {"files": {}, "artifacts": {}},
            "exp7298": upstream,
            "native_module_sha256": upstream.get("native_identity", {}).get("module_sha256"),
            "storage_identity": {
                "journal": ["persist"],
                "synchronous": [exp7298.SQLITE_FULL],
            },
        },
        paths,
        started_at="2026-09-14T00:00:00+00:00",
        completed_at="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
    )
    base["phase_spans_s"] = {"fixture": 1.0}
    receipt = validation_receipt("fixture", ["true"], 0, "ok\n", 0.1)
    return _assemble_complete_artifact(base, measured, reduced, [receipt])


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check provenance, rows, reduction, gates, scores, and claim limits."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(not _checksum_valid(artifact), "reproducibility_checksum")
    add(artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False, "model")
    counts = artifact.get("invocation_counts", {})
    add(not isinstance(counts, Mapping) or any(counts.values()), "invocation_counts")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(float(artifact.get("duration_s", -1)) < 0, "duration_s")
    claim = artifact.get("claim_boundary", {})
    add(
        not isinstance(claim, Mapping)
        or claim.get("fpga_performance_claimed") is not False
        or claim.get("tenfold_carnot_acceleration_claimed") is not False
        or claim.get("production_default_changed") is not False,
        "claim_boundary",
    )
    limitations = artifact.get("limitations", {})
    add(
        not isinstance(limitations, Mapping)
        or limitations.get("physical_power_loss_proven") is not False
        or limitations.get("firmware_fsync_honesty_proven") is not False,
        "limitations",
    )
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "verdict_class")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "honest_verdict")
        add(artifact.get("snapshot_capture_complete_score") != 0, "capture_score")
        add(artifact.get("snapshot_value_score") != 0, "value_score")
        add(
            any(
                artifact.get(name)
                for name in (
                    "rows",
                    "per_run_results",
                    "phase_cost_rows",
                    "independent_recovery_rows",
                )
            ),
            "blocked_rows",
        )
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "gate_check_summary")
        return errors
    add(artifact.get("status") != "complete", "status")
    rows = artifact.get("rows", [])
    runs = artifact.get("per_run_results", [])
    costs = artifact.get("phase_cost_rows", [])
    recovery = artifact.get("independent_recovery_rows", [])
    try:
        reduced = reduce_saved_rows(rows, runs, costs, recovery)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        reduced = None
    add(
        reduced is None or artifact.get("independent_raw_reducer") != reduced,
        "independent_reduction",
    )
    if reduced is not None:
        gates = _acceptance_gates(reduced)
        add(artifact.get("acceptance_gate_results") != gates, "acceptance_gate_results")
        capture = int(reduced["capture_complete"] is True)
        value = int(reduced["snapshot_value_gate_passed"] is True)
        add(artifact.get("snapshot_capture_complete_score") != capture, "capture_score")
        add(artifact.get("snapshot_value_score") != value, "value_score")
        expected_class = "circular_positive" if value else "null"
        add(
            artifact.get("verdict_class") != expected_class
            or artifact.get("verdict_class") == "positive",
            "verdict_class",
        )
        add(
            artifact.get("nfr01_assessment", {}).get("passed")
            is not reduced["nfr01_full_boundary_passed"],
            "nfr01_assessment",
        )
    add(not str(artifact.get("honest_verdict", "")).startswith("complete_"), "honest_verdict")
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
    return errors


def _stream_subprocess(
    command: list[str], *, root: Path, operation: str, timeout_s: int = 1_200
) -> JsonDict:
    """Stream unbuffered child output with truthful heartbeats from the shared helper."""

    return exp7270._stream_subprocess(command, root=root, operation=operation, timeout_s=timeout_s)


def _scoped_validation_commands(root: Path) -> list[tuple[str, list[str], int]]:
    """Return required tests, coverage, formatting, typing, and spec checks."""

    del root
    python = str(Path(sys.executable).absolute())
    test = "tests/python/test_experiment_7299_v641_snapshot_cost.py"
    changed = [
        "python/carnot/experiment_7299_v641_snapshot_cost.py",
        "scripts/experiments/experiment_7299_v641_snapshot_cost.py",
        test,
    ]
    return [
        (
            "focused_pytest",
            [
                python,
                "-m",
                "pytest",
                test,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7299-focused",
            ],
            1_200,
        ),
        (
            "affected_suites",
            [
                python,
                "-m",
                "pytest",
                test,
                "tests/python/test_experiment_7298_v641_snapshot_journal.py",
                "tests/python/test_experiment_7285_v640_commit_frontier.py",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7299-affected",
            ],
            1_200,
        ),
        (
            "full_python_suite",
            [
                python,
                "-m",
                "pytest",
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7299-full",
            ],
            1_800,
        ),
        (
            "scoped_100_percent_coverage",
            [
                python,
                "-c",
                (
                    "import os,sys; import jax,numpy,coverage; "
                    "os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; "
                    "cov=coverage.Coverage(source=['carnot.experiment_7299_v641_snapshot_cost']); "
                    "cov.erase(); cov.start(); import pytest; "
                    f"rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','{test}',"
                    "'-q','-n','0','--basetemp=/tmp/carnot-exp7299-coverage']); "
                    "cov.stop(); cov.save(); percent=cov.report(file=sys.stdout,show_missing=True); "
                    "raise SystemExit(rc if rc else (0 if percent >= 100.0 else 1))"
                ),
            ],
            1_200,
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *changed], 180),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *changed], 180),
        (
            "changed_module_mypy",
            [python, "-m", "mypy", "python/carnot/experiment_7299_v641_snapshot_cost.py"],
            300,
        ),
        ("scoped_spec_coverage", [python, "scripts/check_spec_coverage.py", test], 180),
    ]


def _reduce_raw_in_fresh_process(root: Path, paths: ExperimentPaths) -> tuple[JsonDict, JsonDict]:
    """Reduce immutable saved rows through the public CLI in a fresh process."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "scripts/experiments/experiment_7299_v641_snapshot_cost.py",
        "--reduce-raw",
        str(paths.raw_rows),
        "--reduced-output",
        str(paths.reduced),
    ]
    progress(4, "before", "fresh-process raw reducer")
    started = time.monotonic()
    result = _stream_subprocess(
        command, root=root, operation="independent_raw_reducer", timeout_s=300
    )
    elapsed = time.monotonic() - started
    progress(4, "after", f"fresh-process raw reducer exit_code={result['exit_code']}", started)
    receipt = validation_receipt(
        "independent_raw_reducer",
        command,
        int(result["exit_code"]),
        str(result.get("output", "")),
        elapsed,
    )
    if result["exit_code"] != 0 or not paths.reduced.is_file():
        raise RuntimeError("cold row reduction failed")
    return _read_json(paths.reduced), receipt


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    precondition_bundle: tuple[list[JsonDict], JsonDict] | None = None,
    seeds: Sequence[int] = EVALUATION_SEEDS,
    events_per_trial: int = EVENTS_PER_TRIAL,
) -> JsonDict:
    """Measure, save raw rows, cold-reduce, and assemble one terminal candidate."""

    started_monotonic = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, evidence = precondition_bundle or collect_preconditions(root, paths)
    if gate_summary(checks)["passed"] is not True:
        artifact = _base_artifact(
            checks,
            evidence,
            paths,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=max(time.monotonic() - started_monotonic, 0.0),
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact
    load_started = time.monotonic()
    progress(2, "before", "load authenticated installed native extension", load_started)
    binding = exp7230.load_native_extension(Path(str(evidence["native_module_path"])))
    progress(2, "after", "load authenticated installed native extension", load_started)
    benchmark_started = time.monotonic()
    measured = run_snapshot_benchmark(
        binding,
        paths.storage_dir,
        seeds=seeds,
        events_per_trial=events_per_trial,
        max_duration_s=MAX_BENCHMARK_S,
    )
    benchmark_duration = time.monotonic() - benchmark_started
    raw = {
        "schema": "carnot.exp7299.v641_snapshot_cost.rows.v1",
        "rows": measured[0],
        "per_run_results": measured[1],
        "phase_cost_rows": measured[2],
        "independent_recovery_rows": measured[3],
    }
    progress(4, "before", "persist immutable raw benchmark rows")
    atomic_write(paths.raw_rows, raw)
    progress(4, "after", f"persist immutable raw benchmark rows path={paths.raw_rows}")
    direct = reduce_saved_rows(*measured)
    reduced, cold_receipt = _reduce_raw_in_fresh_process(root, paths)
    if reduced != direct:
        raise ValueError("cold reducer mismatch")
    completed_at = datetime.now(UTC).isoformat()
    duration_s = max(time.monotonic() - started_monotonic, 0.0)
    base = _base_artifact(
        checks,
        evidence,
        paths,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    base["phase_spans_s"] = {
        "native_extension_load": max(benchmark_started - load_started, 0.0),
        "benchmark": benchmark_duration,
        "remaining_reduction_and_assembly": max(duration_s - benchmark_duration, 0.0),
    }
    artifact = _assemble_complete_artifact(
        base, measured, reduced, [*validation_receipts, cold_receipt]
    )
    artifact["cold_reducer_receipt"] = cold_receipt
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _result_receipt(
    name: str, command: list[str], result: Mapping[str, Any], elapsed: float
) -> JsonDict:
    """Normalize streamed or test-injected subprocess results into one receipt."""

    if "log_sha256" in result:
        receipt = dict(result)
        receipt["name"] = name
        receipt["command"] = command
        return receipt
    return validation_receipt(
        name, command, int(result["exit_code"]), str(result.get("output", "")), elapsed
    )


def _prior_full_suite_failures(checkpoint: Path) -> list[JsonDict]:
    """Reuse only the hashed full-suite failure from this task's retry checkpoint."""

    if not checkpoint.is_file():
        return []
    saved = _read_json(checkpoint)
    return [
        dict(row)
        for row in saved.get("validation_receipts", [])
        if isinstance(row, Mapping)
        and row.get("name") == "full_python_suite"
        and row.get("exit_code") != 0
        and str(row.get("log_sha256", "")).startswith("sha256:")
    ]


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Validate, measure, attack, and atomically publish the terminal artifact."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    invocation_started = time.monotonic()
    progress(0, "start", f"Exp7299 run_date={run_date} output={output}", invocation_started)
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / REDUCED_RELATIVE,
        root / RAW_ROWS_RELATIVE,
        root / RAW_CANDIDATE_RELATIVE,
        root / STORAGE_RELATIVE,
        root / VALIDATION_RELATIVE,
        output,
    )
    preconditions = collect_preconditions(root, paths)
    if gate_summary(preconditions[0])["passed"] is not True:
        artifact = build_artifact(
            root,
            paths,
            validation_receipts=[],
            precondition_bundle=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked Exp7299 artifact:{errors}")
        progress(10, "before", "atomic blocked artifact publication", invocation_started)
        atomic_write(output, artifact)
        progress(
            10, "after", f"atomic blocked artifact publication path={output}", invocation_started
        )
        return artifact
    receipts: list[JsonDict] = []
    baseline_failures = _prior_full_suite_failures(paths.checkpoint)
    for name, command, timeout_s in _scoped_validation_commands(root):
        if name == "full_python_suite" and baseline_failures:
            progress(1, "reuse", "preserve prior hashed full-suite baseline failure")
            continue
        progress(1, "before", f"validation subprocess {name}", invocation_started)
        started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        elapsed = time.monotonic() - started
        progress(
            1,
            "after",
            f"validation subprocess {name} exit_code={result['exit_code']}",
            invocation_started,
        )
        receipt = _result_receipt(name, command, result, elapsed)
        paths.validation_dir.mkdir(parents=True, exist_ok=True)
        (paths.validation_dir / f"{len(receipts):02d}-{name}.log").write_text(
            str(result.get("output", "")), encoding="utf-8"
        )
        if result["exit_code"] != 0:
            upstream_baseline = (
                preconditions[1].get("exp7298", {}).get("baseline_validation_failures", [])
            )
            if name == "full_python_suite" and upstream_baseline:
                baseline_failures.append(receipt)
                continue
            receipts.append(receipt)
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7299.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"validation failed:{name}")
        receipts.append(receipt)
    artifact = build_artifact(
        root,
        paths,
        validation_receipts=receipts,
        precondition_bundle=preconditions,
    )
    artifact["baseline_validation_failures"] = baseline_failures
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(6, "before", "cold in-process candidate validation", invocation_started)
    errors = validate_artifact(artifact)
    progress(
        6, "after", f"cold in-process candidate validation errors={len(errors)}", invocation_started
    )
    if errors:
        raise ValueError(f"invalid Exp7299 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    paths.validation_dir.mkdir(parents=True, exist_ok=True)
    candidate_commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7299_v641_snapshot_cost.py",
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
        progress(7, "before", name, invocation_started)
        started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        elapsed = time.monotonic() - started
        progress(7, "after", f"{name} exit_code={result['exit_code']}", invocation_started)
        receipt = _result_receipt(name, command, result, elapsed)
        receipts.append(receipt)
        (paths.validation_dir / f"{len(receipts):02d}-{name}.log").write_text(
            str(result.get("output", "")), encoding="utf-8"
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7299.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"candidate validation failed:{name}")
    artifact["validation_receipts"] = receipts
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = max(time.monotonic() - invocation_started, 0.0)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    final_errors = validate_artifact(artifact)
    if final_errors:
        raise ValueError(f"invalid validated Exp7299 artifact:{final_errors}")
    atomic_write(paths.raw_candidate, artifact)
    progress(8, "before", "atomic terminal artifact publication", invocation_started)
    atomic_write(output, artifact)
    progress(8, "after", f"atomic terminal artifact publication path={output}", invocation_started)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the public run plus read-only validation and reducer modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--reduced-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode and keep all machine-readable outputs explicit."""

    args = _parse_args(argv)
    if args.reduce_raw is not None:
        if args.reduced_output is None:
            return 2
        try:
            raw = _read_json(args.reduce_raw)
            reduced = reduce_saved_rows(
                raw["rows"],
                raw["per_run_results"],
                raw["phase_cost_rows"],
                raw["independent_recovery_rows"],
            )
            atomic_write(args.reduced_output, reduced)
        except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError) as error:
            print(f"reducer_error: {error}", flush=True)
            return 2
        print(canonical_json(reduced), flush=True)
        return 0
    if args.validate is not None:
        progress(6, "before", f"read-only validation path={args.validate}")
        try:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(6, "after", f"read-only validation errors={len(errors)}")
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
