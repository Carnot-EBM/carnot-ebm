"""Prototype durable host group commit over the shipped archive controller.

The wrapper changes when callers receive acknowledgment. It does not change
the event transition or the durable state format. This distinction prevents a
lower sync cost per event from being reported as a same-semantics speedup.

Spec refs: REQ-CL-7284 and SCENARIO-CL-7284-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, NoReturn

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7270_v639_durable_profile as exp7270
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7284
SCHEMA = "carnot.exp7284.v640_commit_prototype.v1"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_284_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
GROUP_SIZES = (1, 4, 16)
MAX_WAIT_MS = 10
MAX_PENDING_EVENTS = 16
MAX_PENDING_BYTES = 65_536
BENCHMARK_EVENTS_PER_ARM = 32
CRASH_BOUNDARIES = (
    "before_write",
    "after_write",
    "after_file_sync",
    "after_rename",
    "after_directory_sync",
    "after_acknowledgment",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7284_v640_commit_prototype.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7284_v640_commit_prototype.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7284_v640_commit_prototype_rows.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7284_v640_commit_prototype_candidate.json")
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7284_v640_commit_storage")
EXP7270_RELATIVE = Path("results/experiment_7270_v639_durable_profile.json")
EXP7256_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
EXPECTED_EXP7270_SHA256 = "sha256:9026a34ec0ffbfaa2c1c0d6127c029fee1ca03c1cc33e191343d4d9d76854be0"

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE,
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("python/carnot/experiment_7257_v638_native_cost.py"),
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("python/carnot/experiment_7284_v640_commit_prototype.py"),
    Path("scripts/experiments/experiment_7284_v640_commit_prototype.py"),
    Path("tests/python/test_experiment_7284_v640_commit_prototype.py"),
    EXP7256_RELATIVE,
    EXP7270_RELATIVE,
)

ACKNOWLEDGMENT_CONTRACT = {
    "acceptance": "A valid unique independent event has entered the bounded host queue.",
    "queueing": "Queued events do not mutate or expose controller state.",
    "durable_linearization": "The group linearizes when directory fsync completes after atomic replacement.",
    "visibility": "The complete group becomes visible only after durable linearization.",
    "acknowledgment": "Every ordered event is acknowledged only after the shared durable commit.",
    "changed_latency_semantics": True,
    "same_semantics_speedup_claim_permitted": False,
    "dependent_query_rule": "A dependent query forces a flush and charges the full flush delay.",
}

FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Bind the evidence to the fixed experiment identity.",
    "milestone": "Bind the evidence to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Record when the real invocation starts.",
    "completed_at_utc": "Record when the real invocation ends.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle.",
    "gate_check_summary": "For blocked evidence name the upstream, exact check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed class; oracle evidence forbids positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "commit_protocol_ready_score": "One requires durable acknowledgment, order, exactly-once replay, and bounded visibility.",
    "acknowledgment_contract": "Name changed latency semantics and the dependent-query flush rule.",
    "crash_rows": "Every crash boundary records issued events, acknowledgments, and recovered state.",
    "semantic_parity_rows": "Grouped transitions must match the serial reference exactly.",
    "queue_bounds": "Include pending events, bytes, backpressure, and timeout dispositions.",
    "same_semantics_speed_claim": "The changed acknowledgment unit forbids a same-semantics speed claim.",
    "production_default_changed": "An opt-in prototype cannot silently change the shipped default.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7270.canonical_json
sha256_file = exp7270.sha256_file
artifact_checksum = exp7270.artifact_checksum
check = exp7270.check
gate_summary = exp7270.gate_summary
atomic_write = exp7270.atomic_write
_finish_row = exp7270._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep provisional, raw, storage, and terminal evidence separate."""

    checkpoint: Path
    raw_rows: Path
    raw_candidate: Path
    storage_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the paths owned by the public command."""

        return cls.under(REPO_ROOT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place test writes below one caller-owned temporary root."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / RAW_ROWS_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / STORAGE_RELATIVE,
            root / RESULT_RELATIVE,
        )

    def writable_targets(self) -> list[bool]:
        """Check output ownership without creating success-shaped bytes."""

        return [
            _writable(path)
            for path in (
                self.checkpoint,
                self.raw_rows,
                self.raw_candidate,
                self.storage_dir,
                self.artifact,
            )
        ]


@dataclass(frozen=True)
class _PendingEvent:
    """Retain accepted bytes and arrival time until one durable flush."""

    release: JsonDict
    accepted_ns: int
    byte_count: int


class DurablePublicationError(OSError):
    """Report whether a failed durable write already replaced the parent path."""

    def __init__(self, stage: str, *, replaced: bool) -> None:
        super().__init__(f"durable publication failed at {stage}")
        self.stage = stage
        self.replaced = replaced


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Flush one truthful phase boundary with optional monotonic elapsed time."""

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
    """Recompute an upstream checksum instead of trusting its declaration."""

    try:
        return artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        return False


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    exp7270_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate the sync diagnosis, native bytes, spec, exclusions, and outputs."""

    started = time.monotonic()
    progress(
        0, "start", "authenticate Exp7270, native bytes, spec, exclusions, and outputs", started
    )
    profile_path = exp7270_path or root / EXP7270_RELATIVE
    profile = _read_json(profile_path) if profile_path.is_file() else {}
    controller_path = root / EXP7256_RELATIVE
    controller = _read_json(controller_path) if controller_path.is_file() else {}
    manifest = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    profile_quarantine = exp7213.quarantine_state(
        profile, manifest, profile_path.name, "exp7270-v639-durable-profile"
    )
    controller_quarantine = exp7213.quarantine_state(
        controller, manifest, controller_path.name, "exp7256-native-controller"
    )
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    profile_hash = sha256_file(profile_path) if profile_path.is_file() else None
    hashes[str(profile_path.resolve())] = profile_hash
    identity = controller.get("native_binary_receipt", {})
    module_path = Path(str(identity.get("module_file", "")))
    module_hash = sha256_file(module_path) if module_path.is_file() else None
    spec_text = (root / SPEC_RELATIVE).read_text(encoding="utf-8")
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
            "REQ-CL-7284",
            True,
            "REQ-CL-7284" in spec_text,
            "REQ-CL-7284" in spec_text,
        ),
        check(
            "exp7270_artifact_hash",
            str(profile_path),
            "sha256",
            EXPECTED_EXP7270_SHA256,
            profile_hash,
            profile_hash == EXPECTED_EXP7270_SHA256,
        ),
        check(
            "exp7270_complete_sync_diagnosis",
            str(profile_path),
            "status,checksum,complete,bottleneck,journal_warrant",
            {
                "status": "complete",
                "checksum": True,
                "complete": 1,
                "bottleneck": "durable_sync",
                "journal_warrant": 0,
            },
            {
                "status": profile.get("status"),
                "checksum": _checksum_valid(profile),
                "complete": profile.get("durable_profile_complete_score"),
                "bottleneck": profile.get("declared_bottleneck"),
                "journal_warrant": profile.get("journal_optimization_warranted_score"),
            },
            profile.get("status") == "complete"
            and _checksum_valid(profile)
            and profile.get("durable_profile_complete_score") == 1
            and profile.get("declared_bottleneck") == "durable_sync"
            and profile.get("journal_optimization_warranted_score") == 0,
        ),
        check(
            "upstreams_not_quarantined_or_retired",
            str(EXCLUSION_RELATIVE),
            "exp7270,exp7256 quarantine or retirement match",
            {"exp7270": False, "exp7256": False},
            {
                "exp7270": profile_quarantine["quarantined"],
                "exp7256": controller_quarantine["quarantined"],
            },
            not profile_quarantine["quarantined"] and not controller_quarantine["quarantined"],
        ),
        check(
            "native_binary_and_outputs",
            str(module_path),
            "module_sha256,writable outputs",
            {"binary": identity.get("module_sha256"), "outputs": [True] * 5},
            {"binary": module_hash, "outputs": paths.writable_targets()},
            bool(identity.get("module_sha256"))
            and module_hash == identity.get("module_sha256")
            and all(paths.writable_targets()),
        ),
    ]
    progress(
        0,
        "end",
        f"precondition_checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
        started,
    )
    return checks, {
        "hashes": hashes,
        "exp7270": profile,
        "exp7256": controller,
        "native_module_path": str(module_path),
        "native_module_sha256": module_hash,
        "quarantine": {"exp7270": profile_quarantine, "exp7256": controller_quarantine},
    }


def durable_replace(
    path: Path,
    data: bytes,
    *,
    stage_hook: Callable[[str], None] | None = None,
) -> JsonDict:
    """Replace one state after file and directory sync, with observable boundaries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    replaced = False
    stage = "before_write"
    try:
        if stage_hook is not None:
            stage_hook(stage)
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        stage = "after_write"
        if stage_hook is not None:
            stage_hook(stage)
        os.fsync(descriptor)
        stage = "after_file_sync"
        if stage_hook is not None:
            stage_hook(stage)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        replaced = True
        stage = "after_rename"
        if stage_hook is not None:
            stage_hook(stage)
        transactional._fsync_directory(path.parent)
        stage = "after_directory_sync"
        if stage_hook is not None:
            stage_hook(stage)
    except OSError as error:
        raise DurablePublicationError(stage, replaced=replaced) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return {
        "file_fsync": True,
        "rename": True,
        "directory_fsync": True,
        "linearization_point": "directory_fsync_complete",
        "bytes": len(data),
    }


class HostGroupCommitController:
    """Queue independent events and acknowledge them after one durable snapshot.

    The native controller remains the only state-transition implementation.
    This class only delays visibility and shares the existing snapshot commit.
    """

    def __init__(
        self,
        binding: ModuleType,
        state_path: Path,
        *,
        max_group_size: int,
        max_wait_ms: int,
        max_pending_events: int = MAX_PENDING_EVENTS,
        max_pending_bytes: int = MAX_PENDING_BYTES,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if max_group_size < 1 or max_wait_ms < 0:
            raise ValueError("group size must be positive and wait must be nonnegative")
        if max_pending_events < 1 or max_pending_bytes < 1:
            raise ValueError("queue bounds must be positive")
        self.binding = binding
        self.state_path = state_path
        self.max_group_size = max_group_size
        self.max_wait_ns = max_wait_ms * 1_000_000
        self.max_pending_events = max_pending_events
        self.max_pending_bytes = max_pending_bytes
        self._clock_ns = clock_ns
        self._controller = exp7256.PersistentNativeArchiveController.load(binding, state_path)
        self._pending: list[_PendingEvent] = []
        self._pending_bytes = 0
        self._committed_ids = set(map(str, self._controller.state_dict().get("release_ids", [])))
        self._failed_restore_required = False
        self._group_number = 0

    @classmethod
    def from_state(
        cls,
        binding: ModuleType,
        state: Mapping[str, Any],
        state_path: Path,
        **kwargs: Any,
    ) -> HostGroupCommitController:
        """Durably seed one caller-owned path before queueing starts."""

        durable_replace(state_path, transactional.canonical_json_bytes(dict(state)))
        return cls(binding, state_path, **kwargs)

    @property
    def pending_events(self) -> int:
        """Return the count accepted but not acknowledged."""

        return len(self._pending)

    @property
    def pending_bytes(self) -> int:
        """Return canonical bytes held by accepted events."""

        return self._pending_bytes

    @property
    def state_hash(self) -> str:
        """Return the visible committed controller hash."""

        return self._controller.state_hash()

    @property
    def state_bytes(self) -> bytes:
        """Return the visible committed controller bytes."""

        return self._controller.state_bytes()

    def enqueue(self, release: Mapping[str, Any], *, now_ns: int | None = None) -> JsonDict:
        """Accept one unique event or return a bounded rejection disposition."""

        if self._failed_restore_required:
            raise RuntimeError("fresh restore required after uncertain publication")
        event_id = str(release.get("event_id", ""))
        if not event_id:
            return {"accepted": False, "acknowledged": False, "disposition": "invalid_release"}
        if event_id in self._committed_ids or any(
            str(row.release["event_id"]) == event_id for row in self._pending
        ):
            return {"accepted": False, "acknowledged": False, "disposition": "duplicate_id"}
        try:
            normalized = exp7256.exp7226.PackedBeliefController._validate_release(
                release, int(release.get("release_index", -1))
            )
        except (KeyError, TypeError, ValueError, exp7256.exp7226.CommitRejected):
            return {"accepted": False, "acknowledged": False, "disposition": "invalid_release"}
        encoded = transactional.canonical_json_bytes(normalized)
        if len(self._pending) >= self.max_pending_events:
            return {
                "accepted": False,
                "acknowledged": False,
                "disposition": "backpressure_events",
            }
        if self._pending_bytes + len(encoded) > self.max_pending_bytes:
            return {
                "accepted": False,
                "acknowledged": False,
                "disposition": "backpressure_bytes",
            }
        accepted_ns = self._clock_ns() if now_ns is None else now_ns
        self._pending.append(_PendingEvent(dict(normalized), accepted_ns, len(encoded)))
        self._pending_bytes += len(encoded)
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

    def flush_due(self, *, now_ns: int | None = None) -> JsonDict | None:
        """Flush a nonempty group whose oldest event reached the wait limit."""

        if not self._pending:
            return None
        current = self._clock_ns() if now_ns is None else now_ns
        if current - self._pending[0].accepted_ns < self.max_wait_ns:
            return None
        return self.flush(reason="timeout")

    def _rollback_receipts(self, receipts: Sequence[Mapping[str, Any]]) -> None:
        """Undo applied serial transitions while the old durable parent is authoritative."""

        for receipt in reversed(receipts):
            self._controller.rollback(receipt)

    def flush(self, *, reason: str) -> JsonDict:
        """Apply queued transitions serially and acknowledge after shared durability."""

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
                receipt = self._controller.commit_batch(
                    [row.release],
                    current_cycle=int(row.release["release_index"]),
                    expected_parent_hash=self._controller.state_hash(),
                )
                receipts.append(receipt)
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
            durable = durable_replace(self.state_path, self._controller.state_bytes())
        except DurablePublicationError as error:
            if not error.replaced:
                self._rollback_receipts(receipts)
                disposition = "failed_rolled_back"
            else:
                self._failed_restore_required = True
                disposition = "failed_restore_required"
            self._pending.clear()
            self._pending_bytes = 0
            return {
                "disposition": disposition,
                "reason": reason,
                "failure_stage": error.stage,
                "acknowledged_event_ids": [],
                "failed_event_ids": event_ids,
            }
        acknowledged_ns = self._clock_ns()
        self._pending.clear()
        self._pending_bytes = 0
        self._committed_ids.update(event_ids)
        self._group_number += 1
        return {
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

    def query(self, event: Mapping[str, Any], *, dependent: bool = False) -> JsonDict:
        """Read committed state; dependent reads first make queued state durable."""

        if self._failed_restore_required:
            raise RuntimeError("fresh restore required after uncertain publication")
        started = self._clock_ns()
        receipt = self.flush(reason="dependent_query") if dependent and self._pending else None
        prediction = self._controller.predict(event)
        ended = self._clock_ns()
        return {
            "prediction": prediction,
            "state_hash": self._controller.state_hash(),
            "dependent": dependent,
            "flush_forced": receipt is not None,
            "flush_receipt": receipt,
            "dependent_query_delay_ns": max(ended - started, 0),
        }

    def shutdown(self, *, abort: bool = False) -> JsonDict:
        """Flush accepted work by default or discard only unacknowledged work."""

        if abort:
            discarded = [str(row.release["event_id"]) for row in self._pending]
            self._pending.clear()
            self._pending_bytes = 0
            return {
                "disposition": "aborted_unacknowledged" if discarded else "aborted_empty",
                "discarded_event_ids": discarded,
                "acknowledged_event_ids": [],
            }
        if self._pending:
            return self.flush(reason="shutdown")
        return {"disposition": "shutdown_clean", "acknowledged_event_ids": []}


def serial_reference(
    binding: ModuleType,
    initial_state: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Apply the shipped one-event transition as the independent reference."""

    controller = exp7256.PersistentNativeArchiveController.from_state_with_binding(
        binding, initial_state
    )
    for release in releases:
        controller.commit_batch(
            [release],
            current_cycle=int(release["release_index"]),
            expected_parent_hash=controller.state_hash(),
        )
    return {
        "state_bytes": controller.state_bytes(),
        "state_hash": controller.state_hash(),
        "release_ids": list(map(str, controller.state_dict().get("release_ids", []))),
    }


def _benchmark_releases(count: int) -> list[JsonDict]:
    """Freeze one independent ordered event roster before measurement."""

    releases: list[JsonDict] = []
    for index in range(count):
        release = exp7257._cost_release(1_000 + index, 0)
        release["event_id"] = f"exp7284-event-{index:04d}"
        release["request_index"] = 10_000 + index
        release["release_index"] = 10_000 + index
        releases.append(release)
    return releases


def run_group_benchmark(
    binding: ModuleType,
    storage_dir: Path,
    *,
    events_per_arm: int = BENCHMARK_EVENTS_PER_ARM,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Measure the three fixed acknowledgment arms without making a speed claim."""

    if events_per_arm < 1 or events_per_arm % 16:
        raise ValueError("events_per_arm must be a positive multiple of 16")
    storage_dir.mkdir(parents=True, exist_ok=True)
    initial = exp7257.seed_cost_state(4)
    releases = _benchmark_releases(events_per_arm)
    rows: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    started = time.monotonic()
    progress(
        3,
        "before",
        f"group benchmark arms={len(GROUP_SIZES)} events_per_arm={events_per_arm}",
        started,
    )
    for arm_index, group_size in enumerate(GROUP_SIZES):
        wrapper = HostGroupCommitController.from_state(
            binding,
            initial,
            storage_dir / f"group-{group_size}/state.json",
            max_group_size=group_size,
            max_wait_ms=0 if group_size == 1 else MAX_WAIT_MS,
        )
        groups: list[JsonDict] = []
        for release in releases:
            response = wrapper.enqueue(release)
            if response.get("flush_receipt") is not None:
                groups.append(response["flush_receipt"])
        for group in groups:
            for event in group["event_receipts"]:
                rows.append(
                    _finish_row(
                        {
                            "unit_id": f"group-{group_size}:{event['event_id']}",
                            "arm": f"max_group_{group_size}",
                            "seed": RANDOM_SEED + arm_index * events_per_arm + len(rows),
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
                            "dependent_query_delay_ns": 0,
                            "byte_count": event["byte_count"],
                            "acknowledged": True,
                            "same_semantics_speed_claim": False,
                        }
                    )
                )
        reference = serial_reference(binding, initial, releases)
        query_event = {"event_id": "probe", "family_id": "lower_bound", "numeric_value": 0}
        restored = exp7256.PersistentNativeArchiveController.from_snapshot(
            binding, reference["state_bytes"].decode("utf-8")
        )
        parity_rows.append(
            {
                "arm": f"max_group_{group_size}",
                "max_group_size": group_size,
                "event_count": events_per_arm,
                "serial_state_hash": reference["state_hash"],
                "group_state_hash": wrapper.state_hash,
                "state_bytes_match": reference["state_bytes"] == wrapper.state_bytes,
                "transition_order_match": reference["release_ids"]
                == list(map(str, wrapper._controller.state_dict().get("release_ids", []))),
                "decision_parity": restored.predict(query_event)
                == wrapper._controller.predict(query_event),
            }
        )
        progress(
            3,
            "unit",
            f"completed_arm={arm_index + 1}/{len(GROUP_SIZES)} max_group_size={group_size}",
            started,
        )
    progress(3, "after", f"group benchmark rows={len(rows)}", started)
    return rows, parity_rows


def _pause_for_kill(marker: Mapping[str, Any]) -> NoReturn:
    """Expose one reached boundary, then wait for the parent to inject SIGKILL."""

    print(canonical_json(marker), flush=True)
    while True:
        signal.pause()


def _crash_worker(
    binding_path: Path,
    state_path: Path,
    releases_path: Path,
    boundary: str,
) -> NoReturn:
    """Reach one real durability boundary and stop for external process death."""

    binding = exp7230.load_native_extension(binding_path)
    controller = exp7256.PersistentNativeArchiveController.load(binding, state_path)
    releases_object = _read_json(releases_path)
    releases = releases_object["releases"]
    for release in releases:
        controller.commit_batch(
            [release],
            current_cycle=int(release["release_index"]),
            expected_parent_hash=controller.state_hash(),
        )

    def hook(stage: str) -> None:
        if stage == boundary:
            _pause_for_kill(
                {
                    "crash_boundary": stage,
                    "issued_event_ids": [str(row["event_id"]) for row in releases],
                    "acknowledged_event_ids": [],
                }
            )

    durable_replace(state_path, controller.state_bytes(), stage_hook=hook)
    if boundary == "after_acknowledgment":
        _pause_for_kill(
            {
                "crash_boundary": boundary,
                "issued_event_ids": [str(row["event_id"]) for row in releases],
                "acknowledged_event_ids": [str(row["event_id"]) for row in releases],
            }
        )
    raise RuntimeError(f"unreached crash boundary:{boundary}")


def _restore_worker(binding_path: Path, state_path: Path) -> int:
    """Load durable bytes in a fresh process and print only the restore receipt."""

    binding = exp7230.load_native_extension(binding_path)
    controller = exp7256.PersistentNativeArchiveController.load(binding, state_path)
    print(
        canonical_json(
            {
                "state_hash": controller.state_hash(),
                "release_ids": list(map(str, controller.state_dict().get("release_ids", []))),
                "state_bytes_b64": base64.b64encode(controller.state_bytes()).decode("ascii"),
            }
        ),
        flush=True,
    )
    return 0


def _crash_command(
    binding_path: Path, state_path: Path, releases_path: Path, boundary: str
) -> list[str]:
    """Build the fixed child command used by the real process-death matrix."""

    return [
        str(Path(sys.executable).absolute()),
        "-u",
        str(REPO_ROOT / "scripts/experiments/experiment_7284_v640_commit_prototype.py"),
        "--crash-worker",
        "--binding",
        str(binding_path),
        "--state",
        str(state_path),
        "--releases",
        str(releases_path),
        "--boundary",
        boundary,
    ]


def run_crash_matrix(
    binding_path: Path,
    initial_state: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
    storage_dir: Path,
) -> list[JsonDict]:
    """Kill six child processes and restore every state in another fresh process."""

    binding = exp7230.load_native_extension(binding_path)
    initial_bytes = transactional.canonical_json_bytes(dict(initial_state))
    old = exp7256.PersistentNativeArchiveController.from_state_with_binding(binding, initial_state)
    reference = serial_reference(binding, initial_state, releases)
    issued = [str(row["event_id"]) for row in releases]
    rows: list[JsonDict] = []
    started = time.monotonic()
    for index, boundary in enumerate(CRASH_BOUNDARIES):
        case_dir = storage_dir / boundary
        state_path = case_dir / "state.json"
        releases_path = case_dir / "releases.json"
        durable_replace(state_path, initial_bytes)
        atomic_write(releases_path, {"releases": [dict(row) for row in releases]})
        command = _crash_command(binding_path, state_path, releases_path, boundary)
        progress(4, "before", f"crash subprocess {boundary}", started)
        environment = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
        }
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
        progress(4, "after", f"crash subprocess {boundary} exit_code={exit_code}", started)

        restore_command = [
            str(Path(sys.executable).absolute()),
            "-u",
            str(REPO_ROOT / "scripts/experiments/experiment_7284_v640_commit_prototype.py"),
            "--restore-worker",
            "--binding",
            str(binding_path),
            "--state",
            str(state_path),
        ]
        progress(4, "before", f"fresh restore subprocess {boundary}", started)
        restored_process = subprocess.run(
            restore_command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        progress(
            4,
            "after",
            f"fresh restore subprocess {boundary} exit_code={restored_process.returncode}",
            started,
        )
        if restored_process.returncode != 0:
            raise RuntimeError(f"restore worker failed:{boundary}:{restored_process.stderr}")
        restored = json.loads(restored_process.stdout)
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
        duplicate_count = sum(max(recovered_ids.count(event_id) - 1, 0) for event_id in issued)
        rows.append(
            {
                "unit_id": f"crash:{boundary}",
                "seed": RANDOM_SEED + index,
                "crash_boundary": boundary,
                "issued_event_ids": issued,
                "acknowledged_event_ids": acknowledged,
                "recovered_event_ids": recovered_ids,
                "recovered_state_hash": restored["state_hash"],
                "recovered_state_kind": state_kind,
                "old_state_hash": old.state_hash(),
                "serial_new_state_hash": reference["state_hash"],
                "process_death": "SIGKILL",
                "process_exit_code": exit_code,
                "fresh_process_restore": True,
                "valid_complete_state": state_kind in {"old", "new"},
                "invalid_partial_state": state_kind == "invalid_partial",
                "lost_acknowledged_event_count": lost,
                "duplicate_apply_count": duplicate_count,
                "orphan_temporary_count": len(list(case_dir.glob(".*.tmp"))),
            }
        )
    return rows


def run_queue_controls(binding: ModuleType, storage_dir: Path) -> JsonDict:
    """Exercise queue limits, timeout, duplicates, shutdown, and query flush."""

    state = exp7257.seed_cost_state(4)
    releases = _benchmark_releases(6)
    bounded = HostGroupCommitController.from_state(
        binding,
        state,
        storage_dir / "bounded/state.json",
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
        max_pending_events=2,
        max_pending_bytes=MAX_PENDING_BYTES,
    )
    accepted = bounded.enqueue(releases[0], now_ns=0)
    pending_duplicate = bounded.enqueue(releases[0], now_ns=1)
    second_accepted = bounded.enqueue(releases[1], now_ns=2)
    event_backpressure = bounded.enqueue(releases[2], now_ns=3)
    timeout = bounded.flush_due(now_ns=MAX_WAIT_MS * 1_000_000)
    committed_duplicate = bounded.enqueue(releases[0], now_ns=4)

    byte_limited = HostGroupCommitController.from_state(
        binding,
        state,
        storage_dir / "bytes/state.json",
        max_group_size=16,
        max_wait_ms=MAX_WAIT_MS,
        max_pending_bytes=1,
    )
    byte_backpressure = byte_limited.enqueue(releases[3])
    abort = byte_limited.shutdown(abort=True)

    dependent = HostGroupCommitController.from_state(
        binding,
        state,
        storage_dir / "dependent/state.json",
        max_group_size=4,
        max_wait_ms=MAX_WAIT_MS,
    )
    dependent.enqueue(releases[4])
    query = dependent.query(
        {"event_id": "dependent-query", "family_id": "lower_bound", "numeric_value": 0},
        dependent=True,
    )
    shutdown = dependent.shutdown()
    return {
        "max_pending_events": MAX_PENDING_EVENTS,
        "max_pending_bytes": MAX_PENDING_BYTES,
        "observed_peak_pending_events": 2,
        "observed_peak_pending_bytes": second_accepted["pending_bytes"],
        "producer_backpressure": "reject_before_acceptance",
        "event_backpressure_disposition": event_backpressure["disposition"],
        "byte_backpressure_disposition": byte_backpressure["disposition"],
        "pending_duplicate_disposition": pending_duplicate["disposition"],
        "committed_duplicate_disposition": committed_duplicate["disposition"],
        "timeout_disposition": None if timeout is None else timeout["disposition"],
        "timeout_reason": None if timeout is None else timeout["reason"],
        "abort_disposition": abort["disposition"],
        "shutdown_disposition": shutdown["disposition"],
        "dependent_query_forced_flush": query["flush_forced"],
        "dependent_query_delay_ns": query["dependent_query_delay_ns"],
        "dependent_query_commit_delay_ns": query["flush_receipt"]["commit_delay_ns"],
        "all_controls_passed": all(
            (
                accepted["accepted"],
                event_backpressure["disposition"] == "backpressure_events",
                byte_backpressure["disposition"] == "backpressure_bytes",
                pending_duplicate["disposition"] == "duplicate_id",
                committed_duplicate["disposition"] == "duplicate_id",
                timeout is not None and timeout["reason"] == "timeout",
                query["flush_forced"],
                query["dependent_query_delay_ns"] >= query["flush_receipt"]["commit_delay_ns"],
            )
        ),
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Return one deterministic linear quantile for a nonempty sequence."""

    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _row_hash_valid(row: Mapping[str, Any]) -> bool:
    """Recompute one row hash without trusting its stored value."""

    material = dict(row)
    stored = material.pop("row_sha256", None)
    return (
        stored
        == "sha256:" + __import__("hashlib").sha256(canonical_json(material).encode()).hexdigest()
    )


def reduce_rows(
    rows: Sequence[Mapping[str, Any]], parity_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Cold-reduce every event and arm while preserving changed latency semantics."""

    if not rows or any(not _row_hash_valid(row) for row in rows):
        raise ValueError("invalid benchmark rows")
    arm_summaries: list[JsonDict] = []
    for group_size in GROUP_SIZES:
        selected = [row for row in rows if row.get("max_group_size") == group_size]
        if not selected:
            raise ValueError(f"missing benchmark arm:{group_size}")
        latencies = [float(row["acknowledgment_delay_ns"]) for row in selected]
        arm_summaries.append(
            {
                "arm": f"max_group_{group_size}",
                "max_group_size": group_size,
                "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                "event_count": len(selected),
                "durable_group_count": len({str(row["group_id"]) for row in selected}),
                "p50_acknowledgment_delay_ns": statistics.median(latencies),
                "p95_acknowledgment_delay_ns": _quantile(latencies, 0.95),
                "same_semantics_speed_claim": False,
            }
        )
    return {
        "inference_substrate": REDUCER_SUBSTRATE,
        "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "completed_event_count": len(rows),
        "censored_event_count": sum(bool(row.get("censored")) for row in rows),
        "error_event_count": sum(row.get("error") is not None for row in rows),
        "acknowledgment_failure_count": sum(row.get("acknowledged") is not True for row in rows),
        "parity_failure_count": sum(
            not all(
                (
                    row.get("state_bytes_match"),
                    row.get("transition_order_match"),
                    row.get("decision_parity"),
                )
            )
            for row in parity_rows
        ),
        "arm_summaries": arm_summaries,
    }


def validation_receipt(
    name: str, command: Sequence[str], exit_code: int, output: str, duration_s: float
) -> JsonDict:
    """Bind one actual command to its result, duration, and complete log hash."""

    import hashlib

    return {
        "name": name,
        "command": list(command),
        "exit_code": exit_code,
        "duration_s": duration_s,
        "log_sha256": "sha256:" + hashlib.sha256(output.encode()).hexdigest(),
    }


def _sample_budget(event_count: int, crash_count: int) -> JsonDict:
    """Record the fixed bounded roster and its no-expansion stopping rule."""

    return {
        "planned_event_units": event_count,
        "attempted_event_units": event_count,
        "completed_event_units": event_count,
        "censored_event_units": 0,
        "planned_crash_units": len(CRASH_BOUNDARIES),
        "attempted_crash_units": crash_count,
        "completed_crash_units": crash_count,
        "censored_crash_units": 0,
        "stopping_rule": "Run the fixed three arms and six crash boundaries once; do not expand after the result.",
    }


def _acceptance_gates(
    summary: Mapping[str, Any],
    crash_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    queue_bounds: Mapping[str, Any],
) -> JsonDict:
    """Keep completeness and semantic readiness separate from any speed claim."""

    expected_events = int(summary.get("completed_event_count", 0))
    return {
        "complete_fixed_event_roster": {
            "expected": expected_events,
            "observed": expected_events - int(summary.get("censored_event_count", 0)),
            "passed": int(summary.get("censored_event_count", 0)) == 0,
            "principle": "Every fixed event must have a terminal acknowledgment row.",
        },
        "serial_transition_parity": {
            "expected": 0,
            "observed": int(summary.get("parity_failure_count", 0)),
            "passed": int(summary.get("parity_failure_count", 0)) == 0
            and len(parity_rows) == len(GROUP_SIZES),
            "principle": "Grouping may share durability but cannot change an event transition.",
        },
        "durable_acknowledgment_crash_safety": {
            "expected": {
                "boundaries": len(CRASH_BOUNDARIES),
                "lost_acknowledged": 0,
                "invalid_partial": 0,
                "duplicates": 0,
            },
            "observed": {
                "boundaries": len(crash_rows),
                "lost_acknowledged": sum(
                    int(row.get("lost_acknowledged_event_count", 0)) for row in crash_rows
                ),
                "invalid_partial": sum(
                    bool(row.get("invalid_partial_state")) for row in crash_rows
                ),
                "duplicates": sum(int(row.get("duplicate_apply_count", 0)) for row in crash_rows),
            },
            "passed": len(crash_rows) == len(CRASH_BOUNDARIES)
            and all(row.get("valid_complete_state") is True for row in crash_rows)
            and all(int(row.get("lost_acknowledged_event_count", 0)) == 0 for row in crash_rows)
            and all(int(row.get("duplicate_apply_count", 0)) == 0 for row in crash_rows),
            "principle": "Acknowledged events must survive exactly once after every crash boundary.",
        },
        "bounded_visibility_and_backpressure": {
            "expected": True,
            "observed": queue_bounds.get("all_controls_passed"),
            "passed": queue_bounds.get("all_controls_passed") is True,
            "principle": "No queued state may leak, and producers must receive bounded dispositions.",
        },
        "changed_latency_semantics_disclosed": {
            "expected": {"changed": True, "same_semantics_speed_claim": False},
            "observed": {"changed": True, "same_semantics_speed_claim": False},
            "passed": True,
            "principle": "Shared acknowledgment changes latency semantics and cannot support the old speed claim.",
        },
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Create every required field before terminal classification."""

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
            "attempted_generation_calls": 0,
            "completed_generation_calls": 0,
            "usable_answers": 0,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": dict(hashes),
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
        "commit_protocol_ready_score": 0,
        "acknowledgment_contract": ACKNOWLEDGMENT_CONTRACT,
        "crash_rows": [],
        "semantic_parity_rows": [],
        "queue_bounds": {},
        "same_semantics_speed_claim": False,
        "production_default_changed": False,
        "output_paths": {
            "checkpoint": str(paths.checkpoint),
            "raw_rows": str(paths.raw_rows),
            "raw_candidate": str(paths.raw_candidate),
            "storage_dir": str(paths.storage_dir),
            "artifact": str(paths.artifact),
        },
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return one schema-complete row-free terminal external block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _assemble_complete_artifact(
    base: JsonDict,
    rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    crash_rows: Sequence[Mapping[str, Any]],
    queue_bounds: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    raw_rows_receipt: Mapping[str, Any],
) -> JsonDict:
    """Join raw evidence and classify semantic readiness without a speed claim."""

    summary = reduce_rows(rows, parity_rows)
    gates = _acceptance_gates(summary, crash_rows, parity_rows, queue_bounds)
    ready = int(all(gate["passed"] is True for gate in gates.values()))
    base.update(
        {
            "status": "complete",
            "rows": [dict(row) for row in rows],
            "sample_size_budget": _sample_budget(len(rows), len(crash_rows)),
            "acceptance_gate_results": gates,
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete_circular_positive: host group commit satisfies the changed acknowledgment contract"
                if ready
                else "complete_null: host group commit failed one or more semantic readiness gates"
            ),
            "validation_receipts": [dict(row) for row in validation_receipts],
            "commit_protocol_ready_score": ready,
            "crash_rows": [dict(row) for row in crash_rows],
            "semantic_parity_rows": [dict(row) for row in parity_rows],
            "queue_bounds": dict(queue_bounds),
            "independent_raw_row_reducer": summary,
            "raw_rows_receipt": dict(raw_rows_receipt),
        }
    )
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def _fixture_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Build compact valid rows for closed validator tests."""

    rows: list[JsonDict] = []
    parity: list[JsonDict] = []
    for group_size in GROUP_SIZES:
        rows.append(
            _finish_row(
                {
                    "unit_id": f"fixture:{group_size}",
                    "arm": f"max_group_{group_size}",
                    "seed": RANDOM_SEED + group_size,
                    "metric": group_size * 100,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "event_id": f"fixture-{group_size}",
                    "group_id": "group-0001",
                    "max_group_size": group_size,
                    "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                    "queue_delay_ns": 10,
                    "commit_delay_ns": 90,
                    "acknowledgment_delay_ns": group_size * 100,
                    "dependent_query_delay_ns": 0,
                    "byte_count": 100,
                    "acknowledged": True,
                    "same_semantics_speed_claim": False,
                }
            )
        )
        parity.append(
            {
                "arm": f"max_group_{group_size}",
                "max_group_size": group_size,
                "event_count": 1,
                "serial_state_hash": "sha256:fixture",
                "group_state_hash": "sha256:fixture",
                "state_bytes_match": True,
                "transition_order_match": True,
                "decision_parity": True,
            }
        )
    return rows, parity


def _fixture_crash_rows() -> list[JsonDict]:
    """Build all six valid crash boundary shapes for validator tests."""

    issued = ["fixture-event"]
    return [
        {
            "unit_id": f"crash:{boundary}",
            "seed": RANDOM_SEED + index,
            "crash_boundary": boundary,
            "issued_event_ids": issued,
            "acknowledged_event_ids": issued if boundary == "after_acknowledgment" else [],
            "recovered_event_ids": issued if boundary in CRASH_BOUNDARIES[3:] else [],
            "recovered_state_hash": "sha256:new"
            if boundary in CRASH_BOUNDARIES[3:]
            else "sha256:old",
            "recovered_state_kind": "new" if boundary in CRASH_BOUNDARIES[3:] else "old",
            "old_state_hash": "sha256:old",
            "serial_new_state_hash": "sha256:new",
            "process_death": "SIGKILL",
            "process_exit_code": -9,
            "fresh_process_restore": True,
            "valid_complete_state": True,
            "invalid_partial_state": False,
            "lost_acknowledged_event_count": 0,
            "duplicate_apply_count": 0,
            "orphan_temporary_count": int(boundary in CRASH_BOUNDARIES[:3]),
        }
        for index, boundary in enumerate(CRASH_BOUNDARIES)
    ]


def _fixture_queue_bounds() -> JsonDict:
    """Return one complete queue-control fixture."""

    return {
        "max_pending_events": MAX_PENDING_EVENTS,
        "max_pending_bytes": MAX_PENDING_BYTES,
        "observed_peak_pending_events": 2,
        "observed_peak_pending_bytes": 200,
        "producer_backpressure": "reject_before_acceptance",
        "event_backpressure_disposition": "backpressure_events",
        "byte_backpressure_disposition": "backpressure_bytes",
        "pending_duplicate_disposition": "duplicate_id",
        "committed_duplicate_disposition": "duplicate_id",
        "timeout_disposition": "committed_acknowledged",
        "timeout_reason": "timeout",
        "abort_disposition": "aborted_empty",
        "shutdown_disposition": "shutdown_clean",
        "dependent_query_forced_flush": True,
        "dependent_query_delay_ns": 100,
        "dependent_query_commit_delay_ns": 90,
        "all_controls_passed": True,
    }


def complete_artifact_fixture_for_test() -> JsonDict:
    """Create one complete semantic-readiness artifact for validator tests."""

    rows, parity = _fixture_rows()
    crashes = _fixture_crash_rows()
    queue = _fixture_queue_bounds()
    now = datetime.now(UTC).isoformat()
    passed = check("fixture", "fixture", "field", True, True, True)
    base = _base_artifact(
        [passed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    return _assemble_complete_artifact(
        base,
        rows,
        parity,
        crashes,
        queue,
        [validation_receipt("fixture", ["true"], 0, "ok", 0.01)],
        raw_rows_receipt={"path": "fixture", "sha256": "sha256:fixture"},
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check provenance, reductions, crash semantics, gates, and verdict."""

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
    try:
        checksum_ok = artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        checksum_ok = False
    add(not checksum_ok, "reproducibility_checksum")
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "blocked_class")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_verdict")
        add(
            bool(artifact.get("rows"))
            or bool(artifact.get("crash_rows"))
            or bool(artifact.get("semantic_parity_rows")),
            "blocked_rows",
        )
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        return errors

    rows = artifact.get("rows", [])
    parity = artifact.get("semantic_parity_rows", [])
    crashes = artifact.get("crash_rows", [])
    queue = artifact.get("queue_bounds", {})
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    try:
        reduced = reduce_rows(rows, parity)
    except (KeyError, TypeError, ValueError):
        reduced = None
    add(reduced is None or artifact.get("independent_raw_row_reducer") != reduced, "rows")
    crash_valid = (
        isinstance(crashes, list)
        and [row.get("crash_boundary") for row in crashes] == list(CRASH_BOUNDARIES)
        and all(row.get("process_death") == "SIGKILL" for row in crashes)
        and all(row.get("fresh_process_restore") is True for row in crashes)
        and all(row.get("valid_complete_state") is True for row in crashes)
        and all(int(row.get("lost_acknowledged_event_count", 0)) == 0 for row in crashes)
        and all(int(row.get("duplicate_apply_count", 0)) == 0 for row in crashes)
    )
    add(not crash_valid, "crash_rows")
    parity_valid = (
        isinstance(parity, list)
        and len(parity) == len(GROUP_SIZES)
        and all(row.get("state_bytes_match") is True for row in parity)
        and all(row.get("transition_order_match") is True for row in parity)
        and all(row.get("decision_parity") is True for row in parity)
    )
    add(not parity_valid, "semantic_parity_rows")
    add(
        not isinstance(queue, Mapping) or queue.get("all_controls_passed") is not True,
        "queue_bounds",
    )
    if reduced is not None and crash_valid and parity_valid and isinstance(queue, Mapping):
        gates = _acceptance_gates(reduced, crashes, parity, queue)
        add(artifact.get("acceptance_gate_results") != gates, "acceptance_gate_results")
        ready = int(all(gate["passed"] is True for gate in gates.values()))
        add(artifact.get("commit_protocol_ready_score") != ready, "commit_protocol_ready_score")
        add(
            artifact.get("verdict_class") != ("circular_positive" if ready else "null")
            or artifact.get("verdict_class") == "positive",
            "verdict_class",
        )
    add(
        artifact.get("acknowledgment_contract") != ACKNOWLEDGMENT_CONTRACT,
        "acknowledgment_contract",
    )
    add(artifact.get("same_semantics_speed_claim") is not False, "same_semantics_speed_claim")
    add(artifact.get("production_default_changed") is not False, "production_default_changed")
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
    return errors


def _stream_subprocess(
    command: list[str], *, root: Path, operation: str, timeout_s: int = 900
) -> JsonDict:
    """Stream child output without buffering and preserve failed receipts."""

    return exp7270._stream_subprocess(command, root=root, operation=operation, timeout_s=timeout_s)


def _scoped_validation_commands(root: Path) -> list[tuple[str, list[str], int]]:
    """Return focused tests, scoped coverage, lint, type, and spec checks."""

    python = str(Path(sys.executable).absolute())
    new_test = "tests/python/test_experiment_7284_v640_commit_prototype.py"
    changed = [
        "python/carnot/experiment_7284_v640_commit_prototype.py",
        "scripts/experiments/experiment_7284_v640_commit_prototype.py",
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
                "tests/python/test_experiment_7270_v639_durable_profile.py",
                "tests/python/test_experiment_7256_v638_native_controller.py",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7284-affected",
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
                    "cov=coverage.Coverage(source=['carnot.experiment_7284_v640_commit_prototype']); "
                    "cov.erase(); cov.start(); import pytest; "
                    f"rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','{new_test}',"
                    "'-q','-n','0','--basetemp=/tmp/carnot-exp7284-coverage']); "
                    "cov.stop(); cov.save(); percent=cov.report(file=sys.stdout,show_missing=True); "
                    "raise SystemExit(rc if rc else (0 if percent >= 100.0 else 1))"
                ),
            ],
            900,
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *changed], 120),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *changed], 120),
        ("changed_module_mypy", [python, "-m", "mypy", changed[0], changed[1]], 300),
        ("spec_coverage", [python, "scripts/check_spec_coverage.py", new_test], 120),
    ]


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    precondition_bundle: tuple[list[JsonDict], JsonDict] | None = None,
    events_per_arm: int = BENCHMARK_EVENTS_PER_ARM,
) -> JsonDict:
    """Run the bounded benchmark, controls, crash matrix, and cold reduction."""

    invocation_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, evidence = precondition_bundle or collect_preconditions(root, paths)
    base = _base_artifact(
        checks,
        evidence.get("hashes", {}),
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - invocation_started,
    )
    if gate_summary(checks)["passed"] is not True:
        base["reproducibility_checksum"] = artifact_checksum(base)
        return base

    progress(
        1,
        "before",
        "MODEL_SPECS empty; current model loads and generations are zero",
        invocation_started,
    )
    progress(1, "after", "no model call authorized or attempted", invocation_started)
    spans: JsonDict = {}
    phase = time.monotonic()
    progress(2, "before", "load authenticated shipped Exp7256 native extension", invocation_started)
    binding_path = Path(evidence["native_module_path"])
    binding = exp7230.load_native_extension(binding_path)
    progress(2, "after", "authenticated native extension loaded", invocation_started)
    spans["native_load"] = time.monotonic() - phase

    phase = time.monotonic()
    rows, parity = run_group_benchmark(
        binding, paths.storage_dir / "benchmark", events_per_arm=events_per_arm
    )
    spans["group_benchmark"] = time.monotonic() - phase
    phase = time.monotonic()
    queue_bounds = run_queue_controls(binding, paths.storage_dir / "queue-controls")
    spans["queue_controls"] = time.monotonic() - phase
    phase = time.monotonic()
    crash_rows = run_crash_matrix(
        binding_path,
        exp7257.seed_cost_state(4),
        _benchmark_releases(4),
        paths.storage_dir / "crash-matrix",
    )
    spans["crash_matrix"] = time.monotonic() - phase
    reduced = reduce_rows(rows, parity)
    raw_receipt = atomic_write(
        paths.raw_rows,
        {
            "schema": "carnot.exp7284.raw.v1",
            "inference_substrate": REDUCER_SUBSTRATE,
            "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
            "rows": rows,
            "semantic_parity_rows": parity,
            "crash_rows": crash_rows,
            "queue_bounds": queue_bounds,
            "independent_raw_row_reducer": reduced,
        },
    )
    atomic_write(
        paths.checkpoint,
        {
            "schema": "carnot.exp7284.checkpoint.v1",
            "status": "provisional_measurement_complete",
            "raw_rows_receipt": raw_receipt,
        },
    )
    base["completed_at_utc"] = datetime.now(UTC).isoformat()
    base["duration_s"] = time.monotonic() - invocation_started
    base["phase_spans_s"] = spans
    return _assemble_complete_artifact(
        base,
        rows,
        parity,
        crash_rows,
        queue_bounds,
        validation_receipts,
        raw_rows_receipt=raw_receipt,
    )


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Validate, measure, cold-check, and atomically publish terminal bytes."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / RAW_ROWS_RELATIVE,
        root / RAW_CANDIDATE_RELATIVE,
        root / STORAGE_RELATIVE,
        output,
    )
    preconditions = collect_preconditions(root, paths)
    if gate_summary(preconditions[0])["passed"] is not True:
        artifact = build_artifact(
            root, paths, validation_receipts=[], precondition_bundle=preconditions
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"invalid blocked Exp7284 artifact:{errors}")
        atomic_write(output, artifact)
        return artifact

    receipts: list[JsonDict] = []
    for name, command, timeout_s in _scoped_validation_commands(root):
        progress(5, "before", f"validation subprocess {name}")
        started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        duration = time.monotonic() - started
        progress(5, "after", f"validation subprocess {name} exit_code={result['exit_code']}")
        receipts.append(
            validation_receipt(
                name,
                command,
                int(result["exit_code"]),
                str(result.get("output", "")),
                duration,
            )
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7284.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"scoped validation failed:{name}")

    artifact = build_artifact(
        root, paths, validation_receipts=receipts, precondition_bundle=preconditions
    )
    progress(6, "before", "cold in-process candidate validation")
    errors = validate_artifact(artifact)
    progress(6, "after", f"cold in-process candidate validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7284 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    candidate_commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7284_v640_commit_prototype.py",
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
        progress(7, "before", name)
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        progress(7, "after", f"{name} exit_code={result['exit_code']}")
        if result["exit_code"] != 0:
            raise RuntimeError(f"candidate validation failed:{name}")
    progress(8, "before", "atomic terminal artifact publication")
    atomic_write(output, artifact)
    progress(8, "after", f"atomic terminal artifact publication path={output}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse experiment, read-only validation, crash, and restore modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--crash-worker", action="store_true")
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--boundary", choices=CRASH_BOUNDARIES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode while keeping worker output machine-readable."""

    args = _parse_args(argv)
    if args.crash_worker:
        if (
            args.binding is None
            or args.state is None
            or args.releases is None
            or args.boundary is None
        ):
            return 2
        _crash_worker(args.binding, args.state, args.releases, args.boundary)
    if args.restore_worker:
        if args.binding is None or args.state is None:
            return 2
        return _restore_worker(args.binding, args.state)
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
