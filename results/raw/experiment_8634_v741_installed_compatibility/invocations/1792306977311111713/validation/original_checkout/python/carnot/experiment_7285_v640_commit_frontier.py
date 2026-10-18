"""Measure the qualified acknowledgment latency and throughput frontier.

The benchmark uses the shipped host group-commit protocol. It keeps the
changed acknowledgment contract explicit. It does not claim faster
interactive service or device acceleration.

Spec refs: REQ-CL-7285 and SCENARIO-CL-7285-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from types import ModuleType
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7256_v638_native_controller as exp7256
from carnot import experiment_7257_v638_native_cost as exp7257
from carnot import experiment_7270_v639_durable_profile as exp7270
from carnot import experiment_7284_v640_commit_prototype as exp7284
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7285
SCHEMA = "carnot.exp7285.v640_commit_frontier.v1"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_285_000
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
REDUCER_SUBSTRATE = "aggregation_from_upstream_artifacts"
REDUCER_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
ARRIVAL_CONDITIONS = ("interactive_dependent", "steady_independent", "burst_16")
GROUP_SIZES = (1, 4, 16)
MAX_WAIT_MS = 10
STEADY_INTERARRIVAL_NS = 1_000_000
PAIRED_SEEDS = tuple(RANDOM_SEED + index for index in range(20))
EVENTS_PER_TRIAL = 128
MAX_MEASUREMENT_S = 900.0
BOOTSTRAP_DRAWS = 10_000

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7285_v640_commit_frontier.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7285_v640_commit_frontier.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7285_v640_commit_frontier_rows.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7285_v640_commit_frontier_candidate.json")
REDUCED_RELATIVE = Path("results/checkpoints/experiment_7285_v640_commit_frontier_reduced.json")
STORAGE_RELATIVE = Path("results/checkpoints/experiment_7285_v640_commit_storage")
EXP7284_RELATIVE = Path("results/experiment_7284_v640_commit_prototype.json")
EXP7256_RELATIVE = Path("results/experiment_7256_v638_native_controller.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
EXPECTED_EXP7284_SHA256 = "sha256:a965fd3d97373aee7be0e44ae426c61450290b314d052eab81749a1a1f16c1a2"

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE,
    Path("python/carnot/experiment_7256_v638_native_controller.py"),
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("python/carnot/experiment_7284_v640_commit_prototype.py"),
    Path("python/carnot/experiment_7285_v640_commit_frontier.py"),
    Path("scripts/experiments/experiment_7285_v640_commit_frontier.py"),
    Path("tests/python/test_experiment_7285_v640_commit_frontier.py"),
    EXP7256_RELATIVE,
    EXP7284_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use full generation 60s, bounded generation 10s, load-only 2s, or the correct no-LLM class; never pad duration.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked_* name upstream, exact field or check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_; retain the measured finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive. Failed efficacy gates forbid positive. Only incomplete own work is partial; unchanged external blocks are terminal blocked.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "commit_cost_complete_score": "One means every declared cost cell has a measured or censored disposition.",
    "commit_cost_value_score": "One requires the fixed burst and steady gates with exact durability.",
    "latency_throughput_rows": "Each seed, arrival condition, and arm has acknowledgment and visibility latency.",
    "component_rows": "Queue, serialization, validation and both sync boundaries sum to the reported work.",
    "acceleration_envelope": "Separate measured speedup, NFR-01 10x target, 100x target and infeasibility.",
    "claim_boundary": "Host batch acknowledgment result; no same-semantics interactive or device acceleration claim.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7284.canonical_json
sha256_file = exp7284.sha256_file
artifact_checksum = exp7284.artifact_checksum
check = exp7284.check
gate_summary = exp7284.gate_summary
atomic_write = exp7284.atomic_write
_finish_row = exp7284._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep raw, provisional, storage, reduced, and terminal bytes separate."""

    checkpoint: Path
    raw_rows: Path
    raw_candidate: Path
    reduced: Path
    storage_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return paths owned by the public experiment command."""

        return cls.under(REPO_ROOT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place test writes below one caller-owned temporary root."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / RAW_ROWS_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / REDUCED_RELATIVE,
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
                self.reduced,
                self.storage_dir,
                self.artifact,
            )
        ]


@dataclass(frozen=True)
class _PendingEvent:
    """Retain measured submission work until its shared durable flush."""

    release: JsonDict
    submitted_ns: int
    accepted_ns: int
    validation_ns: int
    event_serialization_ns: int
    byte_count: int


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Print one flushed progress boundary with real monotonic elapsed time."""

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
    exp7284_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate the prototype, native bytes, spec, exclusions, and outputs."""

    started = time.monotonic()
    progress(0, "start", "authenticate Exp7284, native bytes, spec, and outputs", started)
    prototype_path = exp7284_path or root / EXP7284_RELATIVE
    prototype = _read_json(prototype_path) if prototype_path.is_file() else {}
    native_path = root / EXP7256_RELATIVE
    native_artifact = _read_json(native_path) if native_path.is_file() else {}
    manifest = (root / EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    prototype_quarantine = exp7213.quarantine_state(
        prototype, manifest, prototype_path.name, "exp7284-v640-commit-prototype"
    )
    native_quarantine = exp7213.quarantine_state(
        native_artifact, manifest, native_path.name, "exp7256-native-controller"
    )
    hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    prototype_hash = sha256_file(prototype_path) if prototype_path.is_file() else None
    hashes[str(prototype_path.resolve())] = prototype_hash
    binary_receipt = native_artifact.get("native_binary_receipt", {})
    module_path = Path(str(binary_receipt.get("module_file", "")))
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
            "REQ-CL-7285",
            True,
            "REQ-CL-7285" in spec_text,
            "REQ-CL-7285" in spec_text,
        ),
        check(
            "exp7284_artifact_hash",
            str(prototype_path),
            "sha256",
            EXPECTED_EXP7284_SHA256,
            prototype_hash,
            prototype_hash == EXPECTED_EXP7284_SHA256,
        ),
        check(
            "exp7284_complete_protocol_ready",
            str(prototype_path),
            "status,checksum,commit_protocol_ready_score,parity,crash_safety",
            {
                "status": "complete",
                "checksum": True,
                "ready": 1,
                "parity": True,
                "crash_safety": True,
            },
            {
                "status": prototype.get("status"),
                "checksum": _checksum_valid(prototype),
                "ready": prototype.get("commit_protocol_ready_score"),
                "parity": prototype.get("acceptance_gate_results", {})
                .get("serial_transition_parity", {})
                .get("passed"),
                "crash_safety": prototype.get("acceptance_gate_results", {})
                .get("durable_acknowledgment_crash_safety", {})
                .get("passed"),
            },
            prototype.get("status") == "complete"
            and _checksum_valid(prototype)
            and prototype.get("commit_protocol_ready_score") == 1
            and prototype.get("acceptance_gate_results", {})
            .get("serial_transition_parity", {})
            .get("passed")
            is True
            and prototype.get("acceptance_gate_results", {})
            .get("durable_acknowledgment_crash_safety", {})
            .get("passed")
            is True,
        ),
        check(
            "upstreams_not_quarantined_or_retired",
            str(EXCLUSION_RELATIVE),
            "exp7284,exp7256 quarantine or retirement match",
            {"exp7284": False, "exp7256": False},
            {
                "exp7284": prototype_quarantine["quarantined"],
                "exp7256": native_quarantine["quarantined"],
            },
            not prototype_quarantine["quarantined"] and not native_quarantine["quarantined"],
        ),
        check(
            "native_binary_and_outputs",
            str(module_path),
            "module_sha256,writable outputs",
            {"binary": binary_receipt.get("module_sha256"), "outputs": [True] * 6},
            {"binary": module_hash, "outputs": paths.writable_targets()},
            bool(binary_receipt.get("module_sha256"))
            and module_hash == binary_receipt.get("module_sha256")
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
        "source_artifact_hashes": {
            "files": hashes,
            "artifacts": {
                "exp7284": {
                    "path": str(prototype_path.resolve()),
                    "sha256": prototype_hash,
                    "expected_sha256": EXPECTED_EXP7284_SHA256,
                    "status": prototype.get("status"),
                    "quarantined_or_retired": prototype_quarantine["quarantined"],
                },
                "exp7256": {
                    "path": str(native_path.resolve()),
                    "sha256": hashes.get(str(EXP7256_RELATIVE)),
                    "status": native_artifact.get("status"),
                    "quarantined_or_retired": native_quarantine["quarantined"],
                },
            },
            "authority_separation": {
                "serial_reference": "shared exact shipped transition oracle",
                "cold_reducer": "fresh process over saved raw rows",
                "learned_correctness_claim": False,
            },
            "resource_ownership": {
                "output_paths": "experiment_7285 only",
                "filesystem_device": root.stat().st_dev,
            },
        },
        "exp7284": prototype,
        "exp7256": native_artifact,
        "native_module_path": str(module_path),
        "native_module_sha256": module_hash,
        "quarantine": {"exp7284": prototype_quarantine, "exp7256": native_quarantine},
    }


def _trial_event_ids(seed: int, condition: str, count: int) -> list[str]:
    """Freeze one event identity roster that every arm reuses."""

    return [f"exp7285-{seed}-{condition}-{index:03d}" for index in range(count)]


def freeze_trial_plan(
    *, seeds: Sequence[int] = PAIRED_SEEDS, events_per_trial: int = EVENTS_PER_TRIAL
) -> list[JsonDict]:
    """Freeze all paired units and seeded arm order before measurement."""

    plan: list[JsonDict] = []
    for seed in seeds:
        for condition_index, condition in enumerate(ARRIVAL_CONDITIONS):
            order = list(GROUP_SIZES)
            random.Random(seed * 10 + condition_index).shuffle(order)
            event_ids = _trial_event_ids(seed, condition, events_per_trial)
            for arm_order, group_size in enumerate(order):
                plan.append(
                    {
                        "unit_id": f"{seed}:{condition}:max_group_{group_size}",
                        "seed": seed,
                        "arrival_condition": condition,
                        "max_group_size": group_size,
                        "max_wait_ms": 0 if group_size == 1 else MAX_WAIT_MS,
                        "arm_order": arm_order,
                        "event_count": events_per_trial,
                        "event_ids": event_ids,
                    }
                )
    return plan


def _trial_releases(seed: int, condition: str, count: int) -> list[JsonDict]:
    """Create one validated event stream shared byte-for-byte by all arms."""

    releases: list[JsonDict] = []
    for index, event_id in enumerate(_trial_event_ids(seed, condition, count)):
        release = exp7257._cost_release(seed + index, 0)
        release["event_id"] = event_id
        release["request_index"] = 100_000 + index * 2
        release["release_index"] = 100_001 + index * 2
        releases.append(release)
    return releases


def _timed_durable_replace(path: Path, data: bytes) -> JsonDict:
    """Use the prototype sync calls and expose exclusive persistence spans."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    write_started = time.perf_counter_ns()
    written = 0
    while written < len(data):
        written += os.write(descriptor, data[written:])
    write_ns = time.perf_counter_ns() - write_started
    sync_started = time.perf_counter_ns()
    os.fsync(descriptor)
    file_sync_ns = time.perf_counter_ns() - sync_started
    os.close(descriptor)
    rename_started = time.perf_counter_ns()
    os.replace(temporary, path)
    rename_ns = time.perf_counter_ns() - rename_started
    directory_started = time.perf_counter_ns()
    transactional._fsync_directory(path.parent)
    directory_sync_ns = time.perf_counter_ns() - directory_started
    return {
        "file_write_ns": write_ns,
        "file_sync_ns": file_sync_ns,
        "rename_ns": rename_ns,
        "directory_sync_ns": directory_sync_ns,
        "linearization_point": "directory_fsync_complete",
    }


class _MeasuredGroupController:
    """Measure the same serial transition and durable group used by Exp7284."""

    def __init__(self, binding: ModuleType, state: Mapping[str, Any], path: Path, group: int):
        exp7284.durable_replace(path, transactional.canonical_json_bytes(dict(state)))
        self.controller = exp7256.PersistentNativeArchiveController.load(binding, path)
        self.path = path
        self.group = group
        self.wait_ns = 0 if group == 1 else MAX_WAIT_MS * 1_000_000
        self.pending: list[_PendingEvent] = []
        self.pending_bytes = 0
        self.group_index = 0

    def enqueue(self, release: Mapping[str, Any]) -> JsonDict | None:
        """Validate and serialize one event before bounded queue acceptance."""

        submitted = time.perf_counter_ns()
        validation_started = time.perf_counter_ns()
        normalized = exp7256.exp7226.PackedBeliefController._validate_release(
            release, int(release["release_index"])
        )
        validation_ns = time.perf_counter_ns() - validation_started
        serialization_started = time.perf_counter_ns()
        encoded = transactional.canonical_json_bytes(normalized)
        event_serialization_ns = time.perf_counter_ns() - serialization_started
        accepted = time.perf_counter_ns()
        assert len(self.pending) < exp7284.MAX_PENDING_EVENTS
        assert self.pending_bytes + len(encoded) <= exp7284.MAX_PENDING_BYTES
        assert all(row.release["event_id"] != normalized["event_id"] for row in self.pending)
        self.pending.append(
            _PendingEvent(
                dict(normalized),
                submitted,
                accepted,
                validation_ns,
                event_serialization_ns,
                len(encoded),
            )
        )
        self.pending_bytes += len(encoded)
        return self.flush("group_size") if len(self.pending) >= self.group else None

    def flush_due(self) -> JsonDict | None:
        """Flush when the oldest accepted event reaches the fixed wait bound."""

        if self.pending and time.perf_counter_ns() - self.pending[0].accepted_ns >= self.wait_ns:
            return self.flush("timeout")
        return None

    def flush(self, reason: str) -> JsonDict:
        """Apply serial event semantics and publish one measured durable snapshot."""

        assert self.pending
        pending = list(self.pending)
        commit_started = time.perf_counter_ns()
        transition_started = commit_started
        for row in pending:
            self.controller.commit_batch(
                [row.release],
                current_cycle=int(row.release["release_index"]),
                expected_parent_hash=self.controller.state_hash(),
            )
        transition_ns = time.perf_counter_ns() - transition_started
        serialization_started = time.perf_counter_ns()
        state_bytes = self.controller.state_bytes()
        state_serialization_ns = time.perf_counter_ns() - serialization_started
        durable = _timed_durable_replace(self.path, state_bytes)
        acknowledged = time.perf_counter_ns()
        self.group_index += 1
        event_receipts: list[JsonDict] = []
        for row in pending:
            total = acknowledged - row.submitted_ns
            known = sum(
                (
                    row.validation_ns,
                    row.event_serialization_ns,
                    commit_started - row.accepted_ns,
                    transition_ns,
                    state_serialization_ns,
                    int(durable["file_write_ns"]),
                    int(durable["file_sync_ns"]),
                    int(durable["rename_ns"]),
                    int(durable["directory_sync_ns"]),
                )
            )
            event_receipts.append(
                {
                    "event_id": row.release["event_id"],
                    "submitted_ns": row.submitted_ns,
                    "acknowledged_ns": acknowledged,
                    "acknowledgment_latency_ns": total,
                    "visibility_latency_ns": total,
                    "queue_wait_ns": commit_started - row.accepted_ns,
                    "validation_ns": row.validation_ns,
                    "serialization_ns": row.event_serialization_ns + state_serialization_ns,
                    "transition_ns": transition_ns,
                    "file_write_ns": durable["file_write_ns"],
                    "file_sync_ns": durable["file_sync_ns"],
                    "rename_ns": durable["rename_ns"],
                    "directory_sync_ns": durable["directory_sync_ns"],
                    "protocol_overhead_ns": max(total - known, 0),
                    "byte_count": row.byte_count,
                }
            )
        self.pending.clear()
        self.pending_bytes = 0
        return {
            "group_id": f"group-{self.group_index:04d}",
            "reason": reason,
            "event_receipts": event_receipts,
            "acknowledged_event_ids": [row.release["event_id"] for row in pending],
            **durable,
        }

    def dependent_query(self, event: Mapping[str, Any]) -> JsonDict:
        """Force pending work durable before the dependent state read."""

        receipt = self.flush("dependent_query")
        self.controller.predict(event)
        visible = time.perf_counter_ns()
        for row in receipt["event_receipts"]:
            row["visibility_latency_ns"] = visible - int(row["submitted_ns"])
        return receipt


def _censored_trial(
    plan_row: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Keep every declared event when the bounded timing budget is exhausted."""

    common = {
        "unit_id": plan_row["unit_id"],
        "seed": plan_row["seed"],
        "arrival_condition": plan_row["arrival_condition"],
        "arm": f"max_group_{plan_row['max_group_size']}",
        "max_group_size": plan_row["max_group_size"],
        "arm_order": plan_row["arm_order"],
    }
    events = [
        _finish_row(
            {
                **common,
                "event_id": event_id,
                "event_index": index,
                "metric": None,
                "error": None,
                "abstention": False,
                "censored": True,
                "censoring_reason": "measurement_budget_exhausted",
                "acknowledged": False,
                "lost": False,
                "queue_wait_ns": None,
                "validation_ns": None,
                "serialization_ns": None,
                "transition_ns": None,
                "file_write_ns": None,
                "file_sync_ns": None,
                "rename_ns": None,
                "directory_sync_ns": None,
                "protocol_overhead_ns": None,
                "acknowledgment_latency_ns": None,
                "visibility_latency_ns": None,
            }
        )
        for index, event_id in enumerate(plan_row["event_ids"])
    ]
    latency = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "interval_complete": False,
            "event_count": len(events),
            "completed_event_count": 0,
            "acknowledgment_p50_ns": None,
            "acknowledgment_p95_ns": None,
            "visibility_p95_ns": None,
            "throughput_events_per_s": None,
            "elapsed_ns": None,
            "peak_memory_bytes": None,
            "durable_group_count": 0,
            "lost_acknowledged_event_count": 0,
            "unacknowledged_event_count": len(events),
            "no_amortization_control": plan_row["arrival_condition"] == "interactive_dependent",
            "improvement_claimed": False,
        }
    )
    component = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "queue_wait_ns": None,
            "validation_ns": None,
            "serialization_ns": None,
            "transition_ns": None,
            "file_write_ns": None,
            "file_sync_ns": None,
            "rename_ns": None,
            "directory_sync_ns": None,
            "protocol_overhead_ns": None,
            "total_acknowledgment_latency_ns": None,
            "component_sum_matches": False,
        }
    )
    parity = _finish_row(
        {
            **common,
            "metric": None,
            "error": None,
            "abstention": False,
            "censored": True,
            "exact_final_state_parity": None,
            "serial_state_hash": None,
            "group_state_hash": None,
        }
    )
    return events, latency, component, parity


def _run_trial(
    binding: ModuleType,
    storage_dir: Path,
    plan_row: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Run one complete real-arrival trial and retain exclusive event costs."""

    state = exp7257.seed_cost_state(4)
    group_size = int(plan_row["max_group_size"])
    state_path = storage_dir / str(plan_row["unit_id"]).replace(":", "-") / "state.json"
    controller = _MeasuredGroupController(binding, state, state_path, group_size)
    groups: list[JsonDict] = []
    trial_started = time.perf_counter_ns()
    tracemalloc.start()
    next_arrival = trial_started
    query = {"event_id": "dependent-query", "family_id": "lower_bound", "numeric_value": 0}
    for index, release in enumerate(releases):
        if plan_row["arrival_condition"] == "steady_independent":
            next_arrival = trial_started + index * STEADY_INTERARRIVAL_NS
            remaining = next_arrival - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1_000_000_000)
            due = controller.flush_due()
            if due is not None:
                groups.append(due)
        receipt = controller.enqueue(release)
        if receipt is not None:
            groups.append(receipt)
        if plan_row["arrival_condition"] == "interactive_dependent" and controller.pending:
            groups.append(controller.dependent_query(query))
    if controller.pending:
        groups.append(controller.flush("end_of_stream"))
    elapsed_ns = time.perf_counter_ns() - trial_started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    by_event = {
        str(event["event_id"]): (str(group["group_id"]), event)
        for group in groups
        for event in group["event_receipts"]
    }
    restored = exp7256.PersistentNativeArchiveController.load(binding, state_path)
    recovered_ids = list(map(str, restored.state_dict().get("release_ids", [])))
    common = {
        "unit_id": plan_row["unit_id"],
        "seed": plan_row["seed"],
        "arrival_condition": plan_row["arrival_condition"],
        "arm": f"max_group_{group_size}",
        "max_group_size": group_size,
        "arm_order": plan_row["arm_order"],
    }
    event_rows: list[JsonDict] = []
    for index, release in enumerate(releases):
        group_id, measured = by_event[str(release["event_id"])]
        component_sum = sum(
            int(measured[name])
            for name in (
                "queue_wait_ns",
                "validation_ns",
                "serialization_ns",
                "transition_ns",
                "file_write_ns",
                "file_sync_ns",
                "rename_ns",
                "directory_sync_ns",
                "protocol_overhead_ns",
            )
        )
        acknowledgment = int(measured["acknowledgment_latency_ns"])
        event_rows.append(
            _finish_row(
                {
                    **common,
                    "event_id": release["event_id"],
                    "event_index": index,
                    "group_id": group_id,
                    "metric": acknowledgment,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "censoring_reason": None,
                    "acknowledged": True,
                    "lost": str(release["event_id"]) not in recovered_ids,
                    "queue_wait_ns": measured["queue_wait_ns"],
                    "validation_ns": measured["validation_ns"],
                    "serialization_ns": measured["serialization_ns"],
                    "transition_ns": measured["transition_ns"],
                    "file_write_ns": measured["file_write_ns"],
                    "file_sync_ns": measured["file_sync_ns"],
                    "rename_ns": measured["rename_ns"],
                    "directory_sync_ns": measured["directory_sync_ns"],
                    "protocol_overhead_ns": measured["protocol_overhead_ns"],
                    "acknowledgment_latency_ns": acknowledgment,
                    "visibility_latency_ns": measured["visibility_latency_ns"],
                    "component_sum_matches": component_sum == acknowledgment,
                }
            )
        )
    acknowledgment_values = [int(row["acknowledgment_latency_ns"]) for row in event_rows]
    visibility_values = [int(row["visibility_latency_ns"]) for row in event_rows]
    latency = _finish_row(
        {
            **common,
            "metric": exp7284._quantile(acknowledgment_values, 0.95),
            "error": None,
            "abstention": False,
            "censored": False,
            "interval_complete": True,
            "event_count": len(event_rows),
            "completed_event_count": len(event_rows),
            "acknowledgment_p50_ns": statistics.median(acknowledgment_values),
            "acknowledgment_p95_ns": exp7284._quantile(acknowledgment_values, 0.95),
            "visibility_p95_ns": exp7284._quantile(visibility_values, 0.95),
            "throughput_events_per_s": len(event_rows) * 1_000_000_000 / elapsed_ns,
            "elapsed_ns": elapsed_ns,
            "peak_memory_bytes": peak_memory,
            "durable_group_count": len(groups),
            "lost_acknowledged_event_count": sum(bool(row["lost"]) for row in event_rows),
            "unacknowledged_event_count": sum(not bool(row["acknowledged"]) for row in event_rows),
            "no_amortization_control": plan_row["arrival_condition"] == "interactive_dependent",
            "improvement_claimed": False,
        }
    )
    component_names = (
        "queue_wait_ns",
        "validation_ns",
        "serialization_ns",
        "transition_ns",
        "file_write_ns",
        "file_sync_ns",
        "rename_ns",
        "directory_sync_ns",
        "protocol_overhead_ns",
    )
    totals = {name: sum(int(row[name]) for row in event_rows) for name in component_names}
    total_latency = sum(acknowledgment_values)
    component = _finish_row(
        {
            **common,
            "metric": total_latency,
            "error": None,
            "abstention": False,
            "censored": False,
            **totals,
            "total_acknowledgment_latency_ns": total_latency,
            "component_sum_matches": sum(totals.values()) == total_latency,
        }
    )
    parity = _finish_row(
        {
            **common,
            "metric": int(restored.state_bytes() == reference["state_bytes"]),
            "error": None,
            "abstention": False,
            "censored": False,
            "exact_final_state_parity": restored.state_bytes() == reference["state_bytes"],
            "serial_state_hash": reference["state_hash"],
            "group_state_hash": restored.state_hash(),
        }
    )
    return event_rows, latency, component, parity


def run_frontier_benchmark(
    binding: ModuleType,
    storage_dir: Path,
    *,
    seeds: Sequence[int] = PAIRED_SEEDS,
    events_per_trial: int = EVENTS_PER_TRIAL,
    max_duration_s: float = MAX_MEASUREMENT_S,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Run the fixed paired frontier and preserve censored populations."""

    plan = freeze_trial_plan(seeds=seeds, events_per_trial=events_per_trial)
    event_rows: list[JsonDict] = []
    latency_rows: list[JsonDict] = []
    component_rows: list[JsonDict] = []
    parity_rows: list[JsonDict] = []
    benchmark_started = time.monotonic()
    deadline = benchmark_started + max(max_duration_s, 0.0)
    progress(3, "before", f"frontier trials={len(plan)} events={len(plan) * events_per_trial}")
    references: dict[tuple[int, str], JsonDict] = {}
    for index, plan_row in enumerate(plan):
        if time.monotonic() >= deadline:
            events, latency, component, parity = _censored_trial(plan_row)
        else:
            key = (int(plan_row["seed"]), str(plan_row["arrival_condition"]))
            releases = _trial_releases(key[0], key[1], events_per_trial)
            if key not in references:
                references[key] = exp7284.serial_reference(
                    binding, exp7257.seed_cost_state(4), releases
                )
            events, latency, component, parity = _run_trial(
                binding, storage_dir, plan_row, releases, references[key]
            )
        event_rows.extend(events)
        latency_rows.append(latency)
        component_rows.append(component)
        parity_rows.append(parity)
        progress(
            3,
            "unit",
            f"completed_trials={index + 1}/{len(plan)} event_rows={len(event_rows)}",
            benchmark_started,
        )
    progress(3, "after", f"frontier event_rows={len(event_rows)}", benchmark_started)
    return event_rows, latency_rows, component_rows, parity_rows


def _hash_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Recompute every row hash before independent reduction."""

    return all(exp7284._row_hash_valid(row) for row in rows)


def reduce_saved_rows(
    rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Cold-reduce per-seed latency, throughput, components, loss, and parity."""

    if not all(
        (
            rows,
            latency_rows,
            component_rows,
            parity_rows,
            _hash_rows_valid(rows),
            _hash_rows_valid(latency_rows),
            _hash_rows_valid(component_rows),
            _hash_rows_valid(parity_rows),
        )
    ):
        raise ValueError("invalid frontier rows")
    latency_index = {
        (int(row["seed"]), str(row["arrival_condition"]), int(row["max_group_size"])): row
        for row in latency_rows
    }
    complete_latency = [row for row in latency_rows if row.get("interval_complete") is True]
    grouped_results: list[JsonDict] = []
    for group_size in (4, 16):
        ratios: list[float] = []
        for seed in sorted({int(row["seed"]) for row in latency_rows}):
            baseline = latency_index.get((seed, "burst_16", 1))
            grouped = latency_index.get((seed, "burst_16", group_size))
            if (
                baseline is not None
                and grouped is not None
                and baseline.get("throughput_events_per_s")
                and grouped.get("throughput_events_per_s")
            ):
                ratios.append(
                    float(grouped["throughput_events_per_s"])
                    / float(baseline["throughput_events_per_s"])
                )
        interval = (
            exp7257._bootstrap_interval(ratios, RANDOM_SEED + group_size, BOOTSTRAP_DRAWS)
            if ratios
            else [None, None]
        )
        grouped_results.append(
            {
                "max_group_size": group_size,
                "paired_seed_count": len(ratios),
                "mean_burst_throughput_speedup": statistics.mean(ratios) if ratios else None,
                "burst_throughput_speedup_ci95": interval,
                "interval_complete": len(ratios) == len({int(row["seed"]) for row in latency_rows}),
            }
        )
    selected = max(
        grouped_results,
        key=lambda row: (
            float("-inf")
            if row["burst_throughput_speedup_ci95"][0] is None
            else float(row["burst_throughput_speedup_ci95"][0])
        ),
    )
    selected_group = int(selected["max_group_size"])
    steady_values = [
        float(row["acknowledgment_p95_ns"])
        for row in latency_rows
        if row.get("arrival_condition") == "steady_independent"
        and row.get("max_group_size") == selected_group
        and row.get("acknowledgment_p95_ns") is not None
    ]
    interactive_values = [
        float(row["acknowledgment_p95_ns"])
        for row in latency_rows
        if row.get("arrival_condition") == "interactive_dependent"
        and row.get("acknowledgment_p95_ns") is not None
    ]
    planned_events = sum(int(row.get("event_count", 0)) for row in latency_rows)
    population_complete = len(rows) == planned_events and all(
        row.get("censored") in {True, False} for row in rows
    )
    component_complete = len(component_rows) == len(latency_rows) and all(
        row.get("censored") is True or row.get("component_sum_matches") is True
        for row in component_rows
    )
    no_loss = all(int(row.get("lost_acknowledged_event_count", 0)) == 0 for row in latency_rows)
    measured_parity = [row for row in parity_rows if row.get("censored") is not True]
    exact_parity = bool(measured_parity) and all(
        row.get("censored") is True or row.get("exact_final_state_parity") is True
        for row in parity_rows
    )
    lower = selected["burst_throughput_speedup_ci95"][0]
    steady_p95 = max(steady_values) if steady_values else None
    value_passed = (
        no_loss
        and exact_parity
        and lower is not None
        and float(lower) > 1.5
        and steady_p95 is not None
        and steady_p95 <= 50_000_000
        and selected["interval_complete"] is True
    )
    return {
        "inference_substrate": REDUCER_SUBSTRATE,
        "inference_substrate_class": REDUCER_SUBSTRATE_CLASS,
        "event_row_count": len(rows),
        "trial_row_count": len(latency_rows),
        "complete_trial_count": len(complete_latency),
        "censored_event_count": sum(bool(row.get("censored")) for row in rows),
        "population_complete": population_complete,
        "component_cost_complete": component_complete,
        "zero_lost_acknowledged_events": no_loss,
        "exact_final_state_parity": exact_parity,
        "grouped_burst_frontier": grouped_results,
        "selected_group_size": selected_group,
        "burst_throughput_lower_ci95": lower,
        "steady_acknowledgment_p95_ns": steady_p95,
        "interactive_no_amortization_control": {
            "measured_trial_count": len(interactive_values),
            "acknowledgment_p95_ns": max(interactive_values) if interactive_values else None,
            "improvement_claimed": False,
        },
        "value_gate_passed": value_passed,
    }


def run_protocol_failure_controls(
    binding: ModuleType, binding_path: Path, storage_dir: Path
) -> JsonDict:
    """Rerun tiny queue, crash, acknowledgment, and cold-restore controls."""

    started = time.monotonic()
    progress(4, "before", "tiny protocol failure controls and cold restore", started)
    queue = exp7284.run_queue_controls(binding, storage_dir / "queue")
    releases = exp7284._benchmark_releases(4)
    initial = exp7257.seed_cost_state(4)
    crashes = exp7284.run_crash_matrix(binding_path, initial, releases, storage_dir / "crash")
    e2e_state = storage_dir / "e2e/state.json"
    wrapper = exp7284.HostGroupCommitController.from_state(
        binding, initial, e2e_state, max_group_size=4, max_wait_ms=MAX_WAIT_MS
    )
    submitted = [wrapper.enqueue(row) for row in releases]
    acknowledged = submitted[-1]["flush_receipt"]["acknowledged_event_ids"]
    reference = exp7284.serial_reference(binding, initial, releases)
    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        str(REPO_ROOT / "scripts/experiments/experiment_7284_v640_commit_prototype.py"),
        "--restore-worker",
        "--binding",
        str(binding_path),
        "--state",
        str(e2e_state),
    ]
    progress(4, "before", "E2E cold restore subprocess", started)
    restored_process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    progress(
        4,
        "after",
        f"E2E cold restore subprocess exit_code={restored_process.returncode}",
        started,
    )
    restored = json.loads(restored_process.stdout) if restored_process.returncode == 0 else {}
    recovered = list(map(str, restored.get("release_ids", [])))
    lost = sum(event_id not in recovered for event_id in acknowledged)
    e2e = {
        "arrival_stream": "real_burst_4",
        "bounded_queue": True,
        "validated_commit": True,
        "durable_acknowledgment": len(acknowledged) == len(releases),
        "cold_restore": restored_process.returncode == 0,
        "lost_acknowledged_event_count": lost,
        "exact_final_state_parity": restored.get("state_hash") == reference["state_hash"],
        "restore_command": command,
        "restore_exit_code": restored_process.returncode,
    }
    crash_failures = sum(
        not row["valid_complete_state"]
        or row["lost_acknowledged_event_count"] != 0
        or row["duplicate_apply_count"] != 0
        for row in crashes
    )
    passed = (
        queue["all_controls_passed"] is True
        and crash_failures == 0
        and all(
            (
                e2e["durable_acknowledgment"],
                e2e["cold_restore"],
                e2e["lost_acknowledged_event_count"] == 0,
                e2e["exact_final_state_parity"],
            )
        )
    )
    progress(4, "after", f"protocol failure controls passed={passed}", started)
    return {
        "passed": passed,
        "queue_controls": queue,
        "crash_rows": crashes,
        "crash_failure_count": crash_failures,
        "e2e": e2e,
    }


def synthetic_frontier_rows(
    *, grouped_speedup: float, steady_p95_ns: int
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Build a compact 20-seed frontier for reducer and validator tests."""

    events: list[JsonDict] = []
    latency: list[JsonDict] = []
    components: list[JsonDict] = []
    parity: list[JsonDict] = []
    for plan_row in freeze_trial_plan(events_per_trial=1):
        condition = str(plan_row["arrival_condition"])
        group_size = int(plan_row["max_group_size"])
        throughput = 100.0
        if condition == "burst_16" and group_size > 1:
            throughput *= grouped_speedup
        ack = steady_p95_ns if condition == "steady_independent" and group_size > 1 else 1_000_000
        common = {
            "unit_id": plan_row["unit_id"],
            "seed": plan_row["seed"],
            "arrival_condition": condition,
            "arm": f"max_group_{group_size}",
            "max_group_size": group_size,
            "arm_order": plan_row["arm_order"],
        }
        events.append(
            _finish_row(
                {
                    **common,
                    "event_id": plan_row["event_ids"][0],
                    "event_index": 0,
                    "metric": ack,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "censoring_reason": None,
                    "acknowledged": True,
                    "lost": False,
                    "queue_wait_ns": 0,
                    "validation_ns": 100,
                    "serialization_ns": 100,
                    "transition_ns": 100,
                    "file_write_ns": 100,
                    "file_sync_ns": 100,
                    "rename_ns": 100,
                    "directory_sync_ns": 100,
                    "protocol_overhead_ns": ack - 700,
                    "acknowledgment_latency_ns": ack,
                    "visibility_latency_ns": ack,
                    "component_sum_matches": True,
                }
            )
        )
        latency.append(
            _finish_row(
                {
                    **common,
                    "metric": ack,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "interval_complete": True,
                    "event_count": 1,
                    "completed_event_count": 1,
                    "acknowledgment_p50_ns": ack,
                    "acknowledgment_p95_ns": ack,
                    "visibility_p95_ns": ack,
                    "throughput_events_per_s": throughput,
                    "elapsed_ns": 10_000_000,
                    "peak_memory_bytes": 1024,
                    "durable_group_count": 1,
                    "lost_acknowledged_event_count": 0,
                    "unacknowledged_event_count": 0,
                    "no_amortization_control": condition == "interactive_dependent",
                    "improvement_claimed": False,
                }
            )
        )
        components.append(
            _finish_row(
                {
                    **common,
                    "metric": ack,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "queue_wait_ns": 0,
                    "validation_ns": 100,
                    "serialization_ns": 100,
                    "transition_ns": 100,
                    "file_write_ns": 100,
                    "file_sync_ns": 100,
                    "rename_ns": 100,
                    "directory_sync_ns": 100,
                    "protocol_overhead_ns": ack - 700,
                    "total_acknowledgment_latency_ns": ack,
                    "component_sum_matches": True,
                }
            )
        )
        parity.append(
            _finish_row(
                {
                    **common,
                    "metric": 1,
                    "error": None,
                    "abstention": False,
                    "censored": False,
                    "exact_final_state_parity": True,
                    "serial_state_hash": "sha256:fixture",
                    "group_state_hash": "sha256:fixture",
                }
            )
        )
    return events, latency, components, parity


def _sample_budget(rows: Sequence[Mapping[str, Any]], trials: int) -> JsonDict:
    """Record the complete declared roster and its fixed stopping rule."""

    return {
        "planned_event_units": len(rows),
        "attempted_event_units": len(rows) - sum(bool(row.get("censored")) for row in rows),
        "completed_event_units": sum(not bool(row.get("censored")) for row in rows),
        "censored_event_units": sum(bool(row.get("censored")) for row in rows),
        "planned_trial_units": trials,
        "attempted_trial_units": sum(
            1
            for row in {str(item["unit_id"]): item for item in rows}.values()
            if not row["censored"]
        ),
        "completed_trial_units": sum(
            1
            for row in {str(item["unit_id"]): item for item in rows}.values()
            if not row["censored"]
        ),
        "censored_trial_units": sum(
            1 for row in {str(item["unit_id"]): item for item in rows}.values() if row["censored"]
        ),
        "stopping_rule": "Run 20 paired seeds, three arrivals, three arms, and 128 events. Stop timing at 900 seconds and retain censored units.",
    }


def _acceptance_gates(reduced: Mapping[str, Any], controls: Mapping[str, Any]) -> JsonDict:
    """Keep cost completeness, durability, and deployment value separate."""

    return {
        "complete_declared_cost_population": {
            "expected": {"population": True, "components": True},
            "observed": {
                "population": reduced.get("population_complete"),
                "components": reduced.get("component_cost_complete"),
            },
            "passed": reduced.get("population_complete") is True
            and reduced.get("component_cost_complete") is True,
            "principle": "Every fixed event and cost cell needs a measured or censored disposition.",
        },
        "zero_loss_of_acknowledged_events": {
            "expected": True,
            "observed": reduced.get("zero_lost_acknowledged_events"),
            "passed": reduced.get("zero_lost_acknowledged_events") is True,
            "principle": "No throughput result can weaken durable acknowledgment.",
        },
        "exact_final_state_parity": {
            "expected": True,
            "observed": reduced.get("exact_final_state_parity"),
            "passed": reduced.get("exact_final_state_parity") is True,
            "principle": "Every measured endpoint must equal the serial transition reference.",
        },
        "burst_throughput_lower_ci95": {
            "expected": ">1.5",
            "observed": reduced.get("burst_throughput_lower_ci95"),
            "passed": reduced.get("burst_throughput_lower_ci95") is not None
            and float(reduced["burst_throughput_lower_ci95"]) > 1.5,
            "principle": "Deployment value needs a paired burst speedup above 1.5x at the lower bound.",
        },
        "steady_acknowledgment_p95": {
            "expected": "<=50000000 ns",
            "observed": reduced.get("steady_acknowledgment_p95_ns"),
            "passed": reduced.get("steady_acknowledgment_p95_ns") is not None
            and float(reduced["steady_acknowledgment_p95_ns"]) <= 50_000_000,
            "principle": "The selected grouped arm must keep steady p95 acknowledgment within 50 ms.",
        },
        "protocol_failure_controls": {
            "expected": True,
            "observed": controls.get("passed"),
            "passed": controls.get("passed") is True,
            "principle": "Queue, crash, loss, acknowledgment, and cold restore controls must pass.",
        },
        "interactive_no_amortization_control": {
            "expected": False,
            "observed": reduced.get("interactive_no_amortization_control", {}).get(
                "improvement_claimed"
            ),
            "passed": reduced.get("interactive_no_amortization_control", {}).get(
                "improvement_claimed"
            )
            is False,
            "principle": "Forced flush reports cost and cannot claim an interactive gain.",
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
            "attempted_inference_calls": 0,
            "completed_inference_calls": 0,
            "usable_answers": 0,
        },
        "current_model_count": 0,
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "current_inference_count": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "phase_spans_s": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "latency_throughput_rows": [],
        "component_rows": [],
        "parity_rows": [],
        "sample_size_budget": _sample_budget([], 0),
        "acceptance_gate_results": {},
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": (
            f"blocked_external:{summary['failed_check']}" if blocked else "complete_pending"
        ),
        "verdict_class": "blocked" if blocked else "partial",
        "validation_receipts": [],
        "commit_cost_complete_score": 0,
        "commit_cost_value_score": 0,
        "acceleration_envelope": {},
        "claim_boundary": {
            "scope": "host qualified batch acknowledgment",
            "same_semantics_interactive_speed_claim": False,
            "interactive_improvement_claim": False,
            "board_or_tsu_speed_claim": False,
            "production_default_changed": False,
        },
        "protocol_failure_controls": {},
        "independent_raw_row_reducer": {},
        "raw_rows_receipt": {},
        "cold_reducer_receipt": {},
        "benchmark_configuration": {
            "arrival_conditions": list(ARRIVAL_CONDITIONS),
            "group_sizes": list(GROUP_SIZES),
            "max_wait_ms": MAX_WAIT_MS,
            "paired_seeds": list(PAIRED_SEEDS),
            "events_per_condition_arm_seed": EVENTS_PER_TRIAL,
            "steady_interarrival_ns": STEADY_INTERARRIVAL_NS,
            "measurement_budget_s": MAX_MEASUREMENT_S,
            "state_schema": "shipped Exp7256 canonical JSON snapshot",
            "validation_call": "PackedBeliefController._validate_release",
            "file_sync_call": "os.fsync",
            "directory_sync_call": "transactional_constraint_memory._fsync_directory",
            "input_pairing": "identical event bytes within each seed and arrival condition",
            "arm_order": "seeded random permutation before measurement",
        },
        "output_paths": {
            "checkpoint": str(paths.checkpoint),
            "raw_rows": str(paths.raw_rows),
            "raw_candidate": str(paths.raw_candidate),
            "reduced": str(paths.reduced),
            "storage_dir": str(paths.storage_dir),
            "artifact": str(paths.artifact),
        },
    }


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return one schema-complete row-free external block fixture."""

    now = datetime.now(UTC).isoformat()
    artifact = _base_artifact(
        [failed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=0.0
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _acceleration_envelope(reduced: Mapping[str, Any]) -> JsonDict:
    """Separate measured host speedup from 10x, 100x, and device claims."""

    rows = reduced.get("grouped_burst_frontier", [])
    measured = max(
        (
            float(row["mean_burst_throughput_speedup"])
            for row in rows
            if row.get("mean_burst_throughput_speedup") is not None
        ),
        default=0.0,
    )
    return {
        "measured_best_burst_throughput_speedup": measured,
        "nfr_01_10x_target": 10.0,
        "nfr_01_10x_feasible": measured >= 10.0,
        "target_100x": 100.0,
        "target_100x_feasible": measured >= 100.0,
        "infeasibility": {
            "10x": measured < 10.0,
            "100x": measured < 100.0,
            "basis": "Measured host qualified acknowledgment frontier only.",
        },
        "board_or_tsu_speed_claim": False,
    }


def _assemble_complete_artifact(
    base: JsonDict,
    rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    reduced: Mapping[str, Any],
    raw_rows_receipt: Mapping[str, Any],
    cold_reducer_receipt: Mapping[str, Any],
) -> JsonDict:
    """Join measured evidence and classify the fixed bounded deployment gate."""

    gates = _acceptance_gates(reduced, controls)
    complete = int(
        gates["complete_declared_cost_population"]["passed"] is True
        and gates["protocol_failure_controls"]["passed"] is True
    )
    value_gate_names = (
        "zero_loss_of_acknowledged_events",
        "exact_final_state_parity",
        "burst_throughput_lower_ci95",
        "steady_acknowledgment_p95",
        "protocol_failure_controls",
        "interactive_no_amortization_control",
    )
    value = int(complete == 1 and all(gates[name]["passed"] is True for name in value_gate_names))
    base.update(
        {
            "status": "complete",
            "rows": [dict(row) for row in rows],
            "latency_throughput_rows": [dict(row) for row in latency_rows],
            "component_rows": [dict(row) for row in component_rows],
            "parity_rows": [dict(row) for row in parity_rows],
            "sample_size_budget": _sample_budget(rows, len(latency_rows)),
            "acceptance_gate_results": gates,
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: host qualified acknowledgment meets the bounded deployment gate"
                if value
                else "complete_null: measured host commit frontier does not meet every bounded deployment gate"
            ),
            "validation_receipts": [dict(row) for row in validation_receipts],
            "commit_cost_complete_score": complete,
            "commit_cost_value_score": value,
            "acceleration_envelope": _acceleration_envelope(reduced),
            "protocol_failure_controls": dict(controls),
            "independent_raw_row_reducer": dict(reduced),
            "raw_rows_receipt": dict(raw_rows_receipt),
            "cold_reducer_receipt": dict(cold_reducer_receipt),
        }
    )
    base["reproducibility_checksum"] = artifact_checksum(base)
    return base


def _fixture_controls() -> JsonDict:
    """Return compact passing protocol controls for artifact tests."""

    return {
        "passed": True,
        "queue_controls": {"all_controls_passed": True},
        "crash_rows": [],
        "crash_failure_count": 0,
        "e2e": {
            "cold_restore": True,
            "lost_acknowledged_event_count": 0,
            "exact_final_state_parity": True,
        },
    }


def complete_artifact_fixture_for_test(*, value_passed: bool) -> JsonDict:
    """Create one complete positive or null validator fixture."""

    rows = synthetic_frontier_rows(
        grouped_speedup=2.0 if value_passed else 1.0,
        steady_p95_ns=40_000_000 if value_passed else 60_000_000,
    )
    reduced = reduce_saved_rows(*rows)
    now = datetime.now(UTC).isoformat()
    passed = check("fixture", "fixture", "field", True, True, True)
    base = _base_artifact(
        [passed], {}, ExperimentPaths.defaults(), started_at=now, completed_at=now, duration_s=1.0
    )
    return _assemble_complete_artifact(
        base,
        *rows,
        _fixture_controls(),
        [exp7284.validation_receipt("fixture", ["true"], 0, "ok", 0.01)],
        reduced=reduced,
        raw_rows_receipt={"path": "fixture", "sha256": "sha256:fixture"},
        cold_reducer_receipt={"path": "fixture", "sha256": "sha256:fixture"},
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, rows, reduction, gates, receipts, and claim limits."""

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
            or bool(artifact.get("latency_throughput_rows"))
            or bool(artifact.get("component_rows")),
            "blocked_rows",
        )
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        return errors
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS
        or artifact.get("execution_venue") != EXECUTION_VENUE,
        "substrate",
    )
    rows = artifact.get("rows", [])
    latency = artifact.get("latency_throughput_rows", [])
    components = artifact.get("component_rows", [])
    parity = artifact.get("parity_rows", [])
    try:
        reduced = reduce_saved_rows(rows, latency, components, parity)
    except (KeyError, TypeError, ValueError):
        reduced = None
    add(reduced is None or artifact.get("independent_raw_row_reducer") != reduced, "cold_reduction")
    controls = artifact.get("protocol_failure_controls", {})
    if reduced is not None and isinstance(controls, Mapping):
        gates = _acceptance_gates(reduced, controls)
        add(artifact.get("acceptance_gate_results") != gates, "acceptance_gate_results")
        complete = int(
            gates["complete_declared_cost_population"]["passed"] is True
            and gates["protocol_failure_controls"]["passed"] is True
        )
        value_names = (
            "zero_loss_of_acknowledged_events",
            "exact_final_state_parity",
            "burst_throughput_lower_ci95",
            "steady_acknowledgment_p95",
            "protocol_failure_controls",
            "interactive_no_amortization_control",
        )
        value = int(complete == 1 and all(gates[name]["passed"] is True for name in value_names))
        add(artifact.get("commit_cost_complete_score") != complete, "complete_score")
        add(artifact.get("commit_cost_value_score") != value, "value_score")
        add(
            artifact.get("verdict_class") != ("circular_positive" if value else "null")
            or artifact.get("verdict_class") == "positive",
            "verdict_class",
        )
        add(artifact.get("acceleration_envelope") != _acceleration_envelope(reduced), "envelope")
    budget = artifact.get("sample_size_budget", {})
    add(
        budget.get("planned_event_units") != len(rows)
        or budget.get("planned_trial_units") != len(latency),
        "sample_size_budget",
    )
    claim = artifact.get("claim_boundary", {})
    add(
        claim.get("interactive_improvement_claim") is not False
        or claim.get("board_or_tsu_speed_claim") is not False
        or claim.get("production_default_changed") is not False,
        "claim_boundary",
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
    return errors


def _stream_subprocess(
    command: list[str], *, root: Path, operation: str, timeout_s: int = 900
) -> JsonDict:
    """Stream unbuffered child output and keep truthful heartbeat receipts."""

    return exp7270._stream_subprocess(command, root=root, operation=operation, timeout_s=timeout_s)


def _scoped_validation_commands(root: Path) -> list[tuple[str, list[str], int]]:
    """Return the exact focused test, coverage, lint, type, and spec checks."""

    python = str(Path(sys.executable).absolute())
    new_test = "tests/python/test_experiment_7285_v640_commit_frontier.py"
    changed = [
        "python/carnot/experiment_7285_v640_commit_frontier.py",
        "scripts/experiments/experiment_7285_v640_commit_frontier.py",
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
                "tests/python/test_experiment_7270_v639_durable_profile.py",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7285-affected",
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
                    "cov=coverage.Coverage(source=['carnot.experiment_7285_v640_commit_frontier']); "
                    "cov.erase(); cov.start(); import pytest; "
                    f"rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','{new_test}',"
                    "'-q','-n','0','--basetemp=/tmp/carnot-exp7285-coverage']); "
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


def _reduce_raw_in_fresh_process(root: Path, paths: ExperimentPaths) -> tuple[JsonDict, JsonDict]:
    """Run the saved-row reducer in a fresh process before artifact assembly."""

    command = [
        str(Path(sys.executable).absolute()),
        "-u",
        "scripts/experiments/experiment_7285_v640_commit_frontier.py",
        "--reduce-raw",
        str(paths.raw_rows),
        "--reduced-output",
        str(paths.reduced),
    ]
    started = time.monotonic()
    progress(5, "before", "fresh-process cold row reduction", started)
    result = _stream_subprocess(command, root=root, operation="cold_row_reduction", timeout_s=300)
    progress(
        5, "after", f"fresh-process cold row reduction exit_code={result['exit_code']}", started
    )
    if result["exit_code"] != 0:
        raise RuntimeError("cold row reduction failed")
    reduced = _read_json(paths.reduced)
    return reduced, exp7284.validation_receipt(
        "cold_row_reduction", command, 0, str(result.get("output", "")), time.monotonic() - started
    )


def build_artifact(
    root: Path,
    paths: ExperimentPaths,
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    precondition_bundle: tuple[list[JsonDict], JsonDict] | None = None,
    seeds: Sequence[int] = PAIRED_SEEDS,
    events_per_trial: int = EVENTS_PER_TRIAL,
) -> JsonDict:
    """Measure, save raw rows, cold-reduce, rerun controls, and classify."""

    invocation_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, evidence = precondition_bundle or collect_preconditions(root, paths)
    base = _base_artifact(
        checks,
        evidence.get("source_artifact_hashes", evidence.get("hashes", {})),
        paths,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - invocation_started,
    )
    if gate_summary(checks)["passed"] is not True:
        base["reproducibility_checksum"] = artifact_checksum(base)
        return base
    progress(1, "before", "MODEL_SPECS empty; current loads and generations are zero")
    progress(1, "after", "no model call authorized or attempted")
    spans: JsonDict = {}
    phase = time.monotonic()
    progress(2, "before", "load authenticated shipped native extension", invocation_started)
    binding_path = Path(evidence["native_module_path"])
    binding = exp7230.load_native_extension(binding_path)
    progress(2, "after", "authenticated native extension loaded", invocation_started)
    spans["native_load"] = time.monotonic() - phase
    phase = time.monotonic()
    measured = run_frontier_benchmark(
        binding,
        paths.storage_dir / "benchmark",
        seeds=seeds,
        events_per_trial=events_per_trial,
        max_duration_s=MAX_MEASUREMENT_S,
    )
    spans["frontier_benchmark"] = time.monotonic() - phase
    raw_payload = {
        "schema": "carnot.exp7285.raw.v1",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "rows": measured[0],
        "latency_throughput_rows": measured[1],
        "component_rows": measured[2],
        "parity_rows": measured[3],
    }
    phase = time.monotonic()
    controls = run_protocol_failure_controls(
        binding, binding_path, paths.storage_dir / "failure-controls"
    )
    spans["failure_controls"] = time.monotonic() - phase
    raw_receipt = atomic_write(paths.raw_rows, raw_payload)
    reduced, cold_receipt = _reduce_raw_in_fresh_process(root, paths)
    if reduced != reduce_saved_rows(*measured):
        raise ValueError("cold reducer mismatch")
    atomic_write(
        paths.checkpoint,
        {
            "schema": "carnot.exp7285.checkpoint.v1",
            "status": "provisional_measurement_complete",
            "raw_rows_receipt": raw_receipt,
            "cold_reducer_receipt": cold_receipt,
        },
    )
    base["completed_at_utc"] = datetime.now(UTC).isoformat()
    base["duration_s"] = time.monotonic() - invocation_started
    base["phase_spans_s"] = spans
    return _assemble_complete_artifact(
        base,
        *measured,
        controls,
        validation_receipts,
        reduced=reduced,
        raw_rows_receipt=raw_receipt,
        cold_reducer_receipt=cold_receipt,
    )


def run_experiment(root: Path, output: Path, run_date: str) -> JsonDict:
    """Validate, measure, cold-check, and atomically publish terminal bytes."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    paths = ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / RAW_ROWS_RELATIVE,
        root / RAW_CANDIDATE_RELATIVE,
        root / REDUCED_RELATIVE,
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
            raise ValueError(f"invalid blocked Exp7285 artifact:{errors}")
        atomic_write(output, artifact)
        return artifact
    receipts: list[JsonDict] = []
    for name, command, timeout_s in _scoped_validation_commands(root):
        progress(6, "before", f"validation subprocess {name}")
        started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        duration = time.monotonic() - started
        progress(6, "after", f"validation subprocess {name} exit_code={result['exit_code']}")
        receipts.append(
            exp7284.validation_receipt(
                name, command, int(result["exit_code"]), str(result.get("output", "")), duration
            )
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7285.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"scoped validation failed:{name}")
    artifact = build_artifact(
        root, paths, validation_receipts=receipts, precondition_bundle=preconditions
    )
    progress(7, "before", "cold in-process candidate validation")
    errors = validate_artifact(artifact)
    progress(7, "after", f"cold in-process candidate validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7285 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    candidate_commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7285_v640_commit_frontier.py",
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
        progress(8, "before", name)
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        progress(8, "after", f"{name} exit_code={result['exit_code']}")
        if result["exit_code"] != 0:
            raise RuntimeError(f"candidate validation failed:{name}")
    progress(9, "before", "atomic terminal artifact publication")
    atomic_write(output, artifact)
    progress(9, "after", f"atomic terminal artifact publication path={output}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse experiment, read-only validation, and cold-reducer modes."""

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
    if args.validate is not None:
        progress(7, "before", f"read-only validation path={args.validate}")
        try:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(7, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.reduce_raw is not None:
        if args.reduced_output is None:
            return 2
        progress(5, "before", f"cold reduction path={args.reduce_raw}")
        raw = _read_json(args.reduce_raw)
        reduced = reduce_saved_rows(
            raw["rows"],
            raw["latency_throughput_rows"],
            raw["component_rows"],
            raw["parity_rows"],
        )
        atomic_write(args.reduced_output, reduced)
        print(canonical_json(reduced), flush=True)
        progress(5, "after", f"cold reduction output={args.reduced_output}")
        return 0
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
